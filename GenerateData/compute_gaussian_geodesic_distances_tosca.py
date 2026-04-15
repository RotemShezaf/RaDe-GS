#!/usr/bin/env python3
"""
Compute geodesic distances on Gaussian splats from TOSCA meshes.

This script calculates ground truth geodesic distances on a Gaussian splatting
reconstruction of TOSCA shapes.  It supports two mesh modes:

  1. **Ground truth mesh** (``--mesh_type gt``): uses the high-resolution TOSCA
     mesh from ``TrainData/TOSCA/processed/{shape}/mesh_high_res_*.ply``.
  2. **Reconstructed mesh** (``--mesh_type reconstructed``): uses a user-supplied
     mesh (e.g., extracted from the Gaussian reconstruction).

Sources are selected as *mesh vertices that are close to Gaussians* and then
sub-sampled uniformly (farthest-point sampling on the mesh surface) to cover
the shape evenly.

WORKFLOW:
---------
1. Load Gaussian splat from output folder (point_cloud/iteration_X/point_cloud.ply)
2. Load mesh (ground truth or reconstructed)
3. Select candidate source vertices: mesh vertices within a proximity threshold
   of at least one Gaussian
4. Sub-sample candidates uniformly via farthest-point sampling (FPS)
5. Compute exact geodesic distances on the mesh using MMP algorithm (VTP/gdist)
6. Transfer geodesic distances from mesh to Gaussians via barycentric interpolation
7. Save results in a structured format

TOSCA DATA LAYOUT:
------------------
TrainData/TOSCA/processed/
    ├── cat0/
    │   ├── mesh_high_res_arc*.ply
    │   ├── mesh_low_res_arc*.ply
    │   └── ...
    └── ...

TrainData/TOSCA/SyntheticColmapData/
    └── <texture>_texture/<shape>/<resolution>/light_<id>/output/
        └── point_cloud/iteration_<N>/point_cloud.ply

OUTPUT STRUCTURE:
-----------------
{output_folder}/
    geodesic_distance/
        gt_partial/
            sources_range_{start}_{end}.npz    # Partial results for source range
        gt_geodesic.npz                         # Complete merged ground truth

Each .npz file contains:
    - gaussian_positions: (N_gaussians, 3) Gaussian center positions
    - source_indices: (N_sources,) mesh vertex indices of sources
    - source_positions: (N_sources, 3) positions of source vertices
    - geodesic_distances: (N_sources, N_gaussians) geodesic distances
    - closest_mesh_indices: (N_gaussians,) nearest mesh vertex for each Gaussian
    - closest_mesh_distances: (N_gaussians,) Euclidean distance to nearest vertex
    - source_gaussian_indices: (N_sources,) Gaussian index closest to each source

USAGE:
------
# Compute geodesic distances using the ground truth TOSCA mesh:
python compute_gaussian_geodesic_distances_tosca.py \\
    --gaussian_output TrainData/TOSCA/SyntheticColmapData/colors_texture/cat0/high_res/light_0/output \\
    --shape cat0 \\
    --mesh_type gt \\
    --num_sources 200

# Use a reconstructed mesh:
python compute_gaussian_geodesic_distances_tosca.py \\
    --gaussian_output <output_path> \\
    --mesh_type reconstructed \\
    --mesh_path path/to/reconstructed_mesh.ply \\
    --num_sources 200

# Compute partial results (for parallelization):
python compute_gaussian_geodesic_distances_tosca.py \\
    --gaussian_output <output_path> \\
    --shape cat0 --mesh_type gt \\
    --num_sources 200 \\
    --source_start 0 --source_end 100

# Merge partial results:
python compute_gaussian_geodesic_distances_tosca.py \\
    --gaussian_output <output_path> \\
    --merge_only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from scipy.spatial import KDTree
import json
from datetime import datetime
from tqdm import tqdm

# Ensure project root is in sys.path for module imports
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.load_utils import (
    find_available_iterations,
    load_gaussian_data_cpu,
    load_ply,
)
from GenerateData.utils.compute_gaussian_geodesic_distances_helper import (
    compute_geodesic_distances_for_sources,
    compute_and_save_geodesic_pipeline,
    find_closest_mesh_vertices,
    find_closest_mesh_faces_barycentric,
    transfer_geodesic_to_gaussians,
    save_partial_results,
    merge_partial_results,
)
from utils.misc import fps_gs


# ---------------------------------------------------------------------------
#  Mesh preprocessing (adapted from eval_tnt/cull_mesh.py)
# ---------------------------------------------------------------------------

def preprocess_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    keep_largest_component: bool = True,
    decimate_target: Optional[int] = None,
    merge_vertices: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clean and optionally decimate a mesh before geodesic computation.

    Follows the strategy used in ``eval_tnt/cull_mesh.py``:

    1. ``process=True`` + ``merge_vertices()`` to weld duplicates.
    2. Keep only the largest connected component (by area) to remove
       tiny floating fragments that cause ``inf`` in geodesic distances.
    3. Optionally decimate with Open3D quadric decimation to reduce
       vertex count for faster geodesic computation.

    Args:
        vertices:               (V, 3) vertex positions.
        faces:                  (F, 3) face indices.
        keep_largest_component: If *True*, discard all but the largest
                                connected component (by surface area).
        decimate_target:        If given, decimate to this many faces.
                                ``None`` = keep original resolution.
        merge_vertices:         Merge duplicate / near-duplicate vertices.
        verbose:                Print statistics.

    Returns:
        (vertices, faces) after preprocessing.
    """
    import trimesh
    import open3d as o3d

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
    if merge_vertices:
        mesh.merge_vertices()

    if verbose:
        print(f"\n  Mesh preprocessing:")
        print(f"    After merge: {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces")

    # ── Keep largest connected component ──────────────────────────────
    if keep_largest_component:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            areas = np.array([c.area for c in components])
            largest_idx = areas.argmax()
            removed_verts = sum(len(c.vertices) for i, c in enumerate(components) if i != largest_idx)
            mesh = components[largest_idx]
            if verbose:
                print(f"    Connected components: {len(components)} "
                      f"(kept largest with area={areas[largest_idx]:.2f}, "
                      f"removed {removed_verts} vertices in {len(components)-1} fragments)")
        else:
            if verbose:
                print(f"    Connected components: 1 (no fragments to remove)")

    # ── Decimate ──────────────────────────────────────────────────────
    if decimate_target is not None and len(mesh.faces) > decimate_target:
        mesh_o3d = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)),
        )
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(
            target_number_of_triangles=decimate_target
        )
        mesh_o3d.remove_unreferenced_vertices()
        mesh_o3d.remove_degenerate_triangles()
        vertices_out = np.asarray(mesh_o3d.vertices, dtype=np.float64)
        faces_out = np.asarray(mesh_o3d.triangles, dtype=np.int32)
        if verbose:
            print(f"    Decimated: {len(mesh.faces):,} → {len(faces_out):,} faces "
                  f"({len(mesh.vertices):,} → {len(vertices_out):,} vertices)")
        return vertices_out, faces_out

    vertices_out = np.asarray(mesh.vertices, dtype=np.float64)
    faces_out = np.asarray(mesh.faces, dtype=np.int32)
    if verbose:
        print(f"    Final: {len(vertices_out):,} vertices, {len(faces_out):,} faces")
    return vertices_out, faces_out


# ---------------------------------------------------------------------------
#  Curvature-aware projection via PN Triangles
# ---------------------------------------------------------------------------

def project_pn_triangle(
    face_vertex_positions: np.ndarray,  # (G, 3, 3)
    bary_weights: np.ndarray,           # (G, 3)
    vertex_normals_at_face: np.ndarray, # (G, 3, 3)
) -> np.ndarray:
    """Evaluate a PN-triangle (curved patch) at the given barycentric coords.

    Uses the *Curved PN Triangles* method (Vlachos et al., 2001) to compute
    curvature-corrected surface points.  Each flat mesh triangle is replaced
    by a cubic Bézier patch whose 10 control points are derived purely from
    the three vertex positions and their normals — no explicit curvature
    estimation is needed.  The patch interpolates vertex positions exactly and
    is tangent to the vertex normals at the corners, yielding a C⁰ surface
    approximation that is second-order accurate in curvature.

    Advantages over normal-ray projection:
    * **Always succeeds** — no ray–triangle misses, no fallback needed.
    * Fully vectorised (no per-Gaussian loops).
    * On flat mesh regions the result is identical to linear interpolation.

    Parameters
    ----------
    face_vertex_positions : (G, 3, 3)
        Triangle vertices per Gaussian: ``[:, 0]`` = P₁, etc.
    bary_weights : (G, 3)
        Barycentric coordinates (u, v, w) from orthogonal projection.
    vertex_normals_at_face : (G, 3, 3)
        Vertex normals at the three triangle corners.

    Returns
    -------
    proj_points : (G, 3)
        Points on the cubic Bézier patch (curved triangle surface).

    References
    ----------
    Vlachos, A., Peters, J., Boyd, C., & Mitchell, J. L. (2001).
    *Curved PN Triangles.* Proc. ACM SIGGRAPH Symposium on Interactive 3D
    Graphics (I3D).
    """
    P1 = face_vertex_positions[:, 0]  # (G, 3)
    P2 = face_vertex_positions[:, 1]
    P3 = face_vertex_positions[:, 2]

    N1 = vertex_normals_at_face[:, 0]
    N2 = vertex_normals_at_face[:, 1]
    N3 = vertex_normals_at_face[:, 2]

    # Normalise vertex normals
    N1 = N1 / np.maximum(np.linalg.norm(N1, axis=1, keepdims=True), 1e-12)
    N2 = N2 / np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-12)
    N3 = N3 / np.maximum(np.linalg.norm(N3, axis=1, keepdims=True), 1e-12)

    # Edge projection coefficients:
    #   w_ij = dot(Pj − Pi, Ni) — component of edge along vertex normal
    w12 = np.einsum('ij,ij->i', P2 - P1, N1)
    w21 = np.einsum('ij,ij->i', P1 - P2, N2)
    w23 = np.einsum('ij,ij->i', P3 - P2, N2)
    w32 = np.einsum('ij,ij->i', P2 - P3, N3)
    w31 = np.einsum('ij,ij->i', P1 - P3, N3)
    w13 = np.einsum('ij,ij->i', P3 - P1, N1)

    # 10 control points of the cubic Bézier triangle
    # Corner control points (interpolate vertices exactly)
    b300 = P1
    b030 = P2
    b003 = P3

    # Edge control points (tangent-plane constrained)
    b210 = (2 * P1 + P2 - w12[:, None] * N1) / 3
    b120 = (P1 + 2 * P2 - w21[:, None] * N2) / 3
    b021 = (2 * P2 + P3 - w23[:, None] * N2) / 3
    b012 = (P2 + 2 * P3 - w32[:, None] * N3) / 3
    b102 = (2 * P3 + P1 - w31[:, None] * N3) / 3
    b201 = (P3 + 2 * P1 - w13[:, None] * N1) / 3

    # Centre control point
    E = (b210 + b120 + b021 + b012 + b102 + b201) / 6
    V = (P1 + P2 + P3) / 3
    b111 = E + (E - V) / 2

    # Evaluate the cubic Bézier triangle at (u, v, w)
    u = bary_weights[:, 0:1]  # (G, 1)
    v = bary_weights[:, 1:2]
    w = bary_weights[:, 2:3]

    proj = (b300 * u**3 + b030 * v**3 + b003 * w**3
            + 3 * b210 * u**2 * v + 3 * b120 * u * v**2
            + 3 * b021 * v**2 * w + 3 * b012 * v * w**2
            + 3 * b102 * w**2 * u + 3 * b201 * w * u**2
            + 6 * b111 * u * v * w)

    return proj


# ---------------------------------------------------------------------------
#  Normal-guided surface projection (legacy — kept for comparison)
# ---------------------------------------------------------------------------

def project_along_surface_normal(
    gaussian_positions: np.ndarray,    # (G, 3)
    face_vertex_positions: np.ndarray, # (G, 3, 3) — v0, v1, v2 per Gaussian
    bary_weights: np.ndarray,          # (G, 3) — from orthogonal projection
    vertex_normals_at_face: np.ndarray,# (G, 3, 3) — normals at v0, v1, v2
    parallel_threshold: float = 1e-6,
    bary_tolerance: float = -1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-project Gaussians onto mesh faces along the interpolated surface normal.

    Instead of orthogonal closest-point projection, this casts a ray from each
    Gaussian centre along the interpolated mesh normal at the initial projection
    point.  The intersection of the ray with the face plane gives a projection
    that better respects the local surface geometry — especially on curved
    regions where the face normal and the true surface normal diverge.

    Falls back to the original orthogonal projection when the ray is nearly
    parallel to the face or the intersection falls outside the triangle.

    Parameters
    ----------
    gaussian_positions : (G, 3)
    face_vertex_positions : (G, 3, 3)
        Triangle vertices per Gaussian: ``[:, 0]`` = v0, etc.
    bary_weights : (G, 3)
        Barycentric coordinates from the initial orthogonal projection.
    vertex_normals_at_face : (G, 3, 3)
        Vertex normals at the three triangle corners (same ordering as
        ``face_vertex_positions``).
    parallel_threshold : float
        Minimum ``|dot(n_surf, n_face)|`` to accept the ray–plane
        intersection.  Below this the ray is considered parallel.
    bary_tolerance : float
        Negative tolerance for accepting a barycentric coordinate as
        "inside" the triangle (accounts for floating-point noise).

    Returns
    -------
    proj_points : (G, 3)
        Re-projected surface points.
    new_bary : (G, 3)
        Updated barycentric coordinates.
    valid_mask : (G,) bool
        ``True`` where the normal-guided projection was used, ``False``
        where the function fell back to the orthogonal projection.
    """
    v0 = face_vertex_positions[:, 0]  # (G, 3)
    v1 = face_vertex_positions[:, 1]
    v2 = face_vertex_positions[:, 2]

    # Interpolated surface normal at the orthogonal projection point
    n_surf = (bary_weights[:, 0:1] * vertex_normals_at_face[:, 0] +
              bary_weights[:, 1:2] * vertex_normals_at_face[:, 1] +
              bary_weights[:, 2:3] * vertex_normals_at_face[:, 2])
    n_surf_len = np.linalg.norm(n_surf, axis=1, keepdims=True)
    n_surf = n_surf / np.maximum(n_surf_len, 1e-12)

    # Face normal (defines the plane equation)
    edge1 = v1 - v0
    edge2 = v2 - v0
    face_normal = np.cross(edge1, edge2)
    face_normal /= np.maximum(
        np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-12
    )

    # Ray: P(t) = gauss_pos + t * n_surf
    # Plane: dot(X - v0, face_normal) = 0
    # => t = dot(v0 - gauss_pos, face_normal) / dot(n_surf, face_normal)
    denom = np.einsum('ij,ij->i', n_surf, face_normal)       # (G,)
    numer = np.einsum('ij,ij->i', v0 - gaussian_positions, face_normal)

    safe_denom = np.where(np.abs(denom) > parallel_threshold, denom, 1.0)
    t = numer / safe_denom

    new_proj = gaussian_positions + t[:, np.newaxis] * n_surf

    # Barycentric coordinates of *new_proj* on the triangle via Cramer's rule.
    dp  = new_proj - v0
    d00 = np.einsum('ij,ij->i', edge1, edge1)
    d01 = np.einsum('ij,ij->i', edge1, edge2)
    d11 = np.einsum('ij,ij->i', edge2, edge2)
    d20 = np.einsum('ij,ij->i', dp, edge1)
    d21 = np.einsum('ij,ij->i', dp, edge2)

    bary_denom = d00 * d11 - d01 * d01
    safe_bary_denom = np.where(np.abs(bary_denom) > 1e-12, bary_denom, 1.0)
    new_v = (d11 * d20 - d01 * d21) / safe_bary_denom   # weight for v1
    new_w = (d00 * d21 - d01 * d20) / safe_bary_denom   # weight for v2
    new_u = 1.0 - new_v - new_w                          # weight for v0

    new_bary = np.column_stack([new_u, new_v, new_w])

    # Valid if ray not parallel, triangle non-degenerate, and point inside
    valid = (
        (np.abs(denom) > parallel_threshold) &
        (np.abs(bary_denom) > 1e-12) &
        (new_u >= bary_tolerance) &
        (new_v >= bary_tolerance) &
        (new_w >= bary_tolerance)
    )

    # Clamp & re-normalise bary for valid projections
    new_bary_clamped = np.maximum(new_bary, 0.0)
    bary_sum = np.maximum(new_bary_clamped.sum(axis=1, keepdims=True), 1e-12)
    new_bary_clamped /= bary_sum

    new_proj_clamped = (new_bary_clamped[:, 0:1] * v0 +
                        new_bary_clamped[:, 1:2] * v1 +
                        new_bary_clamped[:, 2:3] * v2)

    # Orthogonal fallback
    ortho_proj = (bary_weights[:, 0:1] * v0 +
                  bary_weights[:, 1:2] * v1 +
                  bary_weights[:, 2:3] * v2)

    out_bary = np.where(valid[:, np.newaxis], new_bary_clamped, bary_weights)
    out_proj = np.where(valid[:, np.newaxis], new_proj_clamped, ortho_proj)

    return out_proj, out_bary, valid


# ---------------------------------------------------------------------------
#  Gaussian embedding into mesh
# ---------------------------------------------------------------------------

def embed_gaussians_in_mesh(
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    gaussian_positions: np.ndarray,
    snap_threshold: float = 0.95,
    use_curvature_projection: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project Gaussians onto the mesh surface and insert as new vertices.

    For each Gaussian centre the nearest mesh face is found and its
    barycentric projection computed.  The projected point is then
    *inserted directly into the mesh* as a new vertex, splitting the
    containing face into sub-triangles.  This guarantees that every
    Gaussian maps to an *exact* mesh vertex and eliminates barycentric
    interpolation error when reading off geodesic distances.

    This mirrors the strategy used by
    ``compute_geodesic_mesh_for_gaussians.py`` for polynomial surfaces
    (where Gaussians are placed as Delaunay vertices on the analytic
    surface).

    Special cases
    -------------
    * If the projection lies very close to an existing vertex
      (``max(bary) >= snap_threshold``), the Gaussian is *snapped* to
      that vertex — no new vertex or face is created.
    * When multiple Gaussians project onto the *same* face a local 2-D
      Delaunay triangulation is used to split the face correctly.

    Parameters
    ----------
    mesh_vertices : (V, 3)
        Mesh vertex positions.
    mesh_faces : (F, 3)
        Face (triangle) indices.
    gaussian_positions : (G, 3)
        Gaussian centre positions (world space).
    snap_threshold : float
        If the largest barycentric coordinate of the projection exceeds
        this value the Gaussian is mapped to the corresponding vertex
        without mesh modification.
    use_curvature_projection : bool
        If ``True`` (default), use PN-triangle curvature-aware projection
        (Vlachos et al. 2001) instead of flat barycentric interpolation.
        The PN triangle replaces each flat triangle with a cubic Bézier
        patch derived from vertex positions and normals, giving a
        second-order surface approximation that always succeeds (no
        ray-miss fallback needed).
    verbose : bool
        Print progress information.

    Returns
    -------
    new_vertices : (V', 3)
        Augmented vertex array.
    new_faces : (F', 3)
        Augmented face array.
    gaussian_vertex_indices : (G,)
        For each Gaussian, the index into *new_vertices* of its
        on-surface vertex.
    """
    from collections import defaultdict
    from scipy.spatial import Delaunay

    n_gauss = len(gaussian_positions)
    n_orig_verts = len(mesh_vertices)
    n_orig_faces = len(mesh_faces)

    if verbose:
        print(f"\n  Embedding {n_gauss:,} Gaussians into mesh ...")
        print(f"    Original mesh: {n_orig_verts:,} vertices, "
              f"{n_orig_faces:,} faces")

    # 1. Closest mesh vertex for each Gaussian (needed by the bary helper)
    tree = KDTree(mesh_vertices)
    _, closest_vi = tree.query(gaussian_positions)

    # 2. Closest face + barycentric coordinates
    face_verts_per_gauss, bary_per_gauss = find_closest_mesh_faces_barycentric(
        gaussian_centers=gaussian_positions,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        closest_vertex_indices=closest_vi,
    )

    # 3. Projection points on the surface
    proj_points = (
        bary_per_gauss[:, 0:1] * mesh_vertices[face_verts_per_gauss[:, 0]] +
        bary_per_gauss[:, 1:2] * mesh_vertices[face_verts_per_gauss[:, 1]] +
        bary_per_gauss[:, 2:3] * mesh_vertices[face_verts_per_gauss[:, 2]]
    )

    # 3b. (Optional) Curvature-aware projection via PN Triangles
    if use_curvature_projection:
        import trimesh
        tm = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces,
                             process=False)
        all_vertex_normals = np.asarray(tm.vertex_normals, dtype=np.float64)

        face_vertex_pos = np.stack([
            mesh_vertices[face_verts_per_gauss[:, 0]],
            mesh_vertices[face_verts_per_gauss[:, 1]],
            mesh_vertices[face_verts_per_gauss[:, 2]],
        ], axis=1)  # (G, 3, 3)

        vtx_normals_at_face = np.stack([
            all_vertex_normals[face_verts_per_gauss[:, 0]],
            all_vertex_normals[face_verts_per_gauss[:, 1]],
            all_vertex_normals[face_verts_per_gauss[:, 2]],
        ], axis=1)  # (G, 3, 3)

        proj_points = project_pn_triangle(
            face_vertex_positions=face_vertex_pos,
            bary_weights=bary_per_gauss,
            vertex_normals_at_face=vtx_normals_at_face,
        )

        if verbose:
            # Report displacement from flat projection
            flat_proj = (
                bary_per_gauss[:, 0:1] * mesh_vertices[face_verts_per_gauss[:, 0]] +
                bary_per_gauss[:, 1:2] * mesh_vertices[face_verts_per_gauss[:, 1]] +
                bary_per_gauss[:, 2:3] * mesh_vertices[face_verts_per_gauss[:, 2]]
            )
            displacement = np.linalg.norm(proj_points - flat_proj, axis=1)
            print(f"    PN-triangle curvature correction: "
                  f"mean={displacement.mean():.6f}  "
                  f"max={displacement.max():.6f}")

    # 4. Classify: snap to existing vertex vs. insert new vertex -----------
    gaussian_vertex_indices = np.full(n_gauss, -1, dtype=np.intp)
    max_bary = bary_per_gauss.max(axis=1)
    snap_vertex_col = np.argmax(bary_per_gauss, axis=1)

    # vertex_index -> projected point for snapped Gaussians (used later to
    # overwrite the mesh vertex position with the Gaussian's on-surface projection)
    snapped_updates: dict[int, np.ndarray] = {}

    for g in range(n_gauss):
        if max_bary[g] >= snap_threshold:
            vi = int(face_verts_per_gauss[g, snap_vertex_col[g]])
            gaussian_vertex_indices[g] = vi
            snapped_updates[vi] = proj_points[g]

    n_snapped = int((gaussian_vertex_indices >= 0).sum())
    n_to_insert = n_gauss - n_snapped
    if verbose:
        print(f"    Snapped to existing vertex: {n_snapped:,} "
              f"({100 * n_snapped / n_gauss:.1f}%)")
        print(f"    New vertices to insert: {n_to_insert:,}")

    # 5. Build face lookup: frozenset(vertex_ids) → face_index -----------
    face_lookup: dict[frozenset, int] = {}
    for fi in range(n_orig_faces):
        key = frozenset(mesh_faces[fi].tolist())
        face_lookup[key] = fi

    # 6. Group non-snapped Gaussians by their target face ----------------
    face_insertions: dict[int, list] = defaultdict(list)
    #   face_idx → [(gauss_idx, bary (3,), proj_point (3,)), ...]

    for g in range(n_gauss):
        if gaussian_vertex_indices[g] >= 0:
            continue  # snapped
        face_key = frozenset(face_verts_per_gauss[g].tolist())
        fi = face_lookup.get(face_key)
        if fi is None:
            # Fallback: snap to nearest vertex
            gaussian_vertex_indices[g] = closest_vi[g]
            continue
        face_insertions[fi].append((g, bary_per_gauss[g], proj_points[g]))

    faces_to_split = set(face_insertions.keys())

    if verbose:
        print(f"    Faces to split: {len(faces_to_split):,}")
        if face_insertions:
            counts = [len(v) for v in face_insertions.values()]
            print(f"    Insertions per face: "
                  f"max={max(counts)}, mean={np.mean(counts):.2f}")

    # 7. Build augmented mesh -------------------------------------------
    new_vertices_list = list(mesh_vertices)
    # Replace each snapped-to vertex position with the Gaussian's projected
    # point so that the Gaussian's location is embedded in the mesh rather
    # than the Gaussian being forced to the original vertex position.
    for vi, proj in snapped_updates.items():
        new_vertices_list[vi] = proj
    new_faces_list: list[list[int]] = []

    # Keep un-split original faces
    for fi in range(n_orig_faces):
        if fi not in faces_to_split:
            new_faces_list.append(mesh_faces[fi].tolist())

    # Split faces that have insertions
    for fi, insertions in face_insertions.items():
        orig_face = mesh_faces[fi]
        v0, v1, v2 = int(orig_face[0]), int(orig_face[1]), int(orig_face[2])

        # Assign new vertex indices and record them
        local_vert_indices = [v0, v1, v2]  # local idx 0,1,2
        # 2-D coordinates in the barycentric frame of the face:
        #   v0 → (0,0), v1 → (1,0), v2 → (0,1)
        points_2d = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]

        for g, bary, proj in insertions:
            new_vi = len(new_vertices_list)
            new_vertices_list.append(proj)
            gaussian_vertex_indices[g] = new_vi
            local_vert_indices.append(new_vi)
            points_2d.append([float(bary[1]), float(bary[2])])

        if len(insertions) == 1:
            # Simple case: split triangle into 3 sub-triangles
            p_vi = local_vert_indices[3]
            new_faces_list.append([v0, v1, p_vi])
            new_faces_list.append([v1, v2, p_vi])
            new_faces_list.append([v2, v0, p_vi])
        else:
            # Multiple points: Delaunay in 2-D barycentric coords
            pts_2d = np.array(points_2d)
            tri = Delaunay(pts_2d)
            for simplex in tri.simplices:
                new_faces_list.append(
                    [local_vert_indices[s] for s in simplex]
                )

    new_vertices = np.array(new_vertices_list, dtype=np.float64)
    new_faces = np.array(new_faces_list, dtype=np.int32)

    # Safety: handle any unmapped Gaussians
    unmapped = int((gaussian_vertex_indices < 0).sum())
    if unmapped > 0:
        if verbose:
            print(f"    WARNING: {unmapped} Gaussians could not be mapped — "
                  f"falling back to nearest vertex")
        for g in range(n_gauss):
            if gaussian_vertex_indices[g] < 0:
                gaussian_vertex_indices[g] = closest_vi[g]

    if verbose:
        print(f"    Augmented mesh: {len(new_vertices):,} vertices, "
              f"{len(new_faces):,} faces")
        print(f"    Added {len(new_vertices) - n_orig_verts:,} vertices, "
              f"{len(new_faces) - n_orig_faces + len(faces_to_split):,} "
              f"net new faces")

    return new_vertices, new_faces, gaussian_vertex_indices


# ---------------------------------------------------------------------------
#  TOSCA-specific helpers
# ---------------------------------------------------------------------------

def load_tosca_gt_mesh(
    data_root: Path,
    shape: str,
    resolution: str = "high_res",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a ground-truth TOSCA mesh.

    Args:
        data_root: Root of the *processed* TOSCA directory
                   (e.g., ``TrainData/TOSCA/processed``).
        shape:     Shape name (e.g. ``cat0``).
        resolution: ``'high_res'`` or ``'low_res'``.

    Returns:
        vertices (V, 3), faces (F, 3)
    """
    shape_dir = data_root / shape
    candidates = sorted(shape_dir.glob(f"mesh_{resolution}_*.ply"))
    if not candidates:
        raise FileNotFoundError(
            f"No mesh for shape={shape} resolution={resolution} under {shape_dir}"
        )
    mesh_path = candidates[0]
    print(f"  Loading GT mesh: {mesh_path}")
    vertices, faces = load_ply(str(mesh_path))
    print(f"    Vertices: {len(vertices)},  Faces: {len(faces)}")
    return vertices, faces


def select_sources_near_gaussians(
    mesh_vertices: np.ndarray,
    gaussian_positions: np.ndarray,
    num_sources: int,
    proximity_factor: float = 3.0,
    seed: int = 42,
    gaussian_vertex_indices: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Select source vertices for geodesic computation, via FPS.

    If *gaussian_vertex_indices* is provided (from ``embed_gaussians_in_mesh``),
    the FPS candidates are exactly those embedded mesh vertices.  This
    guarantees every selected source IS a Gaussian footprint on the mesh,
    giving zero transfer error for the source distance.

    Otherwise falls back to the legacy approach: proximity-filtered mesh
    vertices sub-sampled with FPS.

    Args:
        mesh_vertices:            (V, 3)
        gaussian_positions:       (G, 3)
        num_sources:              Desired number of source vertices.
        proximity_factor:         Multiplier on median nearest-Gaussian distance
                                  (legacy mode only).
        seed:                     Random seed for the initial FPS point.
        gaussian_vertex_indices:  (G,) mesh vertex index for each Gaussian,
                                  as returned by ``embed_gaussians_in_mesh``.

    Returns:
        source_indices:   (S,)  mesh vertex indices of selected sources
        source_positions: (S, 3) positions of selected sources
    """
    if gaussian_vertex_indices is not None:
        # ── Embedded-Gaussian mode: FPS over the Gaussian mesh vertices ──
        candidate_indices = gaussian_vertex_indices.astype(np.intp)
        # Remove duplicates (multiple Gaussians may snap to the same vertex)
        candidate_indices = np.unique(candidate_indices)
        n_candidates = len(candidate_indices)
        print(f"\n  Selecting {num_sources} sources from {n_candidates} "
              f"embedded-Gaussian mesh vertices (FPS) ...")

        if n_candidates <= num_sources:
            selected = candidate_indices
        else:
            candidate_positions = mesh_vertices[candidate_indices]
            fps_local_indices = fps_gs(
                positions=candidate_positions,
                n=num_sources,
                attributes=["xyz"],
                device="cpu",
            )
            selected = candidate_indices[fps_local_indices]

        source_positions = mesh_vertices[selected]
        print(f"    Selected {len(selected)} sources via FPS on embedded vertices")
        return selected, source_positions

    # ── Legacy mode: proximity-filtered mesh vertices ─────────────────
    print(f"\n  Selecting {num_sources} source vertices close to Gaussians ...")
    gauss_tree = KDTree(gaussian_positions)
    dists_to_gauss, _ = gauss_tree.query(mesh_vertices)

    median_dist = np.median(dists_to_gauss)
    threshold = proximity_factor * median_dist

    # Gradually relax if we don't get enough candidates
    for factor in [proximity_factor, proximity_factor * 2, proximity_factor * 5, np.inf]:
        threshold = factor * median_dist
        candidate_mask = dists_to_gauss <= threshold
        n_candidates = candidate_mask.sum()
        if n_candidates >= num_sources:
            break
    candidate_indices = np.where(candidate_mask)[0]
    print(f"    Proximity threshold: {threshold:.6f}  ({n_candidates} candidates "
          f"from {len(mesh_vertices)} mesh vertices)")

    if n_candidates <= num_sources:
        # Use all candidates
        selected = candidate_indices
    else:
        # Farthest-point sampling among candidates using shared FPS utility
        candidate_positions = mesh_vertices[candidate_indices]
        fps_local_indices = fps_gs(
            positions=candidate_positions,
            n=num_sources,
            attributes=["xyz"],
            device="cpu",
        )
        selected = candidate_indices[fps_local_indices]

    source_positions = mesh_vertices[selected]
    print(f"    Selected {len(selected)} sources via FPS")
    return selected, source_positions


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute geodesic distances on Gaussian splats from TOSCA meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- required (unless merge_only) ---
    parser.add_argument(
        "--gaussian_output", type=str, required=True,
        help="Path to Gaussian splatting output folder",
    )
    parser.add_argument(
        "--mesh_type", type=str, choices=["gt", "reconstructed"], default="gt",
        help="Which mesh to compute geodesics on: 'gt' = ground-truth TOSCA mesh, "
             "'reconstructed' = user-supplied mesh.",
    )

    # GT mesh options
    parser.add_argument(
        "--shape", type=str, default=None,
        help="TOSCA shape name (e.g. cat0).  Required when --mesh_type gt.",
    )
    parser.add_argument(
        "--data_root", type=str, default="TrainData/TOSCA/processed",
        help="Root of the preprocessed TOSCA directory.",
    )
    parser.add_argument(
        "--mesh_resolution", type=str, choices=["high_res", "low_res"],
        default="high_res",
        help="Resolution of the GT mesh (default: high_res).",
    )

    # Reconstructed mesh options
    parser.add_argument(
        "--mesh_path", type=str, default=None,
        help="Explicit path to a PLY mesh (overrides --mesh_type / --shape).",
    )

    # Source selection
    parser.add_argument(
        "--num_sources", type=int, default=200,
        help="Number of source vertices to sample.",
    )
    parser.add_argument(
        "--proximity_factor", type=float, default=3.0,
        help="Multiplier on median nearest-Gaussian distance for candidate "
             "vertex filtering (default: 3.0).",
    )
    parser.add_argument(
        "--source_start", type=int, default=None,
        help="Start index of source range (partial computation).",
    )
    parser.add_argument(
        "--source_end", type=int, default=None,
        help="End index of source range (exclusive).",
    )

    # Iteration
    parser.add_argument(
        "--iteration", type=int, default=None,
        help="Gaussian training iteration to use (default: highest).",
    )

    # Merge mode
    parser.add_argument(
        "--merge_only", action="store_true",
        help="Only merge existing partial results.",
    )

    # Mesh preprocessing
    parser.add_argument(
        "--no_preprocess", action="store_true",
        help="Skip mesh preprocessing (keep all components, no decimation).",
    )
    parser.add_argument(
        "--decimate_target", type=int, default=None,
        help="Decimate mesh to this many faces before geodesic computation. "
             "Reduces runtime for large meshes. None = keep original.",
    )
    parser.add_argument(
        "--embed_gaussians", action="store_true",
        help="Project each Gaussian onto the mesh surface and insert the "
             "projection as a new mesh vertex.  This eliminates barycentric "
             "interpolation error when transferring geodesic distances and "
             "ensures each Gaussian maps to an exact on-surface vertex.",
    )
    parser.add_argument(
        "--snap_threshold", type=float, default=0.97,
        help="When --embed_gaussians is used, if the largest barycentric "
             "coordinate of the projection exceeds this value the Gaussian "
             "is snapped to the existing vertex.  Default: 0.97.",
    )

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--geodesic_method", type=str, choices=["vtp", "mmp", "fmm"], default="mmp",
        help="Geodesic computation method: 'vtp' (exact, requires manifold mesh), "
             "'mmp' (MMP via pygeodesic, works on non-manifold meshes), "
             "or 'fmm' (fast marching). Default: mmp.",
    )
    parser.add_argument(
        "--n_jobs", type=int, default=None,
        help="Number of parallel jobs for geodesic computation.",
    )

    args = parser.parse_args()

    # Validation
    if not args.merge_only:
        if args.mesh_type == "gt" and args.shape is None:
            parser.error("--shape is required when --mesh_type is 'gt'")

    if args.source_start is not None and args.source_end is None:
        parser.error("--source_end must be specified with --source_start")
    if args.source_end is not None and args.source_start is None:
        parser.error("--source_start must be specified with --source_end")
    if (args.source_start is not None and args.source_end is not None
            and args.source_start >= args.source_end):
        parser.error("--source_start must be less than --source_end")

    return args


# ---------------------------------------------------------------------------
#  Metadata
# ---------------------------------------------------------------------------

def _save_computation_metadata(
    output_folder: Path,
    args: argparse.Namespace,
    num_gaussians: int,
    num_sources: int,
    mesh_path: str,
    num_mesh_vertices: int,
    num_mesh_faces: int,
) -> None:
    metadata = {
        "computation_info": {
            "timestamp": datetime.now().isoformat(),
            "script": "compute_gaussian_geodesic_distances_tosca.py",
            "description": "Ground truth geodesic distances on TOSCA Gaussian splats",
        },
        "mesh": {
            "type": args.mesh_type,
            "shape": args.shape,
            "path": mesh_path,
            "resolution": getattr(args, "mesh_resolution", None),
            "vertices": num_mesh_vertices,
            "faces": num_mesh_faces,
        },
        "gaussian_data": {
            "source_folder": str(args.gaussian_output),
            "iteration": args.iteration if args.iteration else "auto (highest)",
            "num_gaussians": num_gaussians,
        },
        "source_selection": {
            "method": "farthest_point_sampling_near_gaussians",
            "num_sources": num_sources,
            "proximity_factor": args.proximity_factor,
        },
        "parameters": {
            "seed": args.seed,
            "source_start": args.source_start,
            "source_end": args.source_end,
            "n_jobs": args.n_jobs,
            "embed_gaussians": args.embed_gaussians,
            "snap_threshold": args.snap_threshold if args.embed_gaussians else None,
            "geodesic_method": args.geodesic_method,
        },
    }
    out = output_folder / "geodesic_distance" / "computation_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to: {out}")


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\n{'#'*80}")
    print(f"# TOSCA — Gaussian Splat Geodesic Distance Computation")
    print(f"{'#'*80}")

    output_folder = Path(args.gaussian_output)

    # ── Merge-only mode ───────────────────────────────────────────────
    if args.merge_only:
        merge_partial_results(output_folder, verbose=args.verbose)
        return

    # ── 1. Load Gaussians ─────────────────────────────────────────────
    gaussian_data = load_gaussian_data_cpu(output_folder, args.iteration)
    gaussian_positions = gaussian_data.get_xyz()
    gaussian_scales = gaussian_data.get_scaling()
    gaussian_rotations = gaussian_data.get_rotation()
    n_gauss = len(gaussian_positions)
    print(f"  {n_gauss} Gaussians loaded")

    # ── 2. Load mesh ──────────────────────────────────────────────────
    if args.mesh_path is not None:
        # Explicit mesh path provided — use its parent as output folder
        mesh_path_resolved = Path(args.mesh_path)
        mesh_path_str = str(mesh_path_resolved)
        output_folder = mesh_path_resolved.parent
        print(f"\n  Loading mesh from explicit path: {mesh_path_str}")
        print(f"  Output folder overridden to: {output_folder}")
        mesh_vertices, mesh_faces = load_ply(mesh_path_str)
        print(f"    Vertices: {len(mesh_vertices)},  Faces: {len(mesh_faces)}")
        if not args.no_preprocess:
            mesh_vertices, mesh_faces = preprocess_mesh(
                mesh_vertices, mesh_faces,
                keep_largest_component=True,
                decimate_target=args.decimate_target,
                verbose=True,
            )
    elif args.mesh_type == "gt":
        data_root = Path(args.data_root)
        if not data_root.is_absolute():
            data_root = project_root / data_root
        mesh_vertices, mesh_faces = load_tosca_gt_mesh(
            data_root, args.shape, args.mesh_resolution,
        )
        mesh_path_str = str(data_root / args.shape)
    else:
        # Reconstructed mesh — default to gaussian_output/recon.ply
        mesh_path_resolved = output_folder / "recon.ply"
        mesh_path_str = str(mesh_path_resolved)
        print(f"\n  No --mesh_path provided; defaulting to: {mesh_path_str}")
        if not mesh_path_resolved.exists():
            raise FileNotFoundError(
                f"Reconstructed mesh not found at {mesh_path_str}.\n"
                f"Provide --mesh_path explicitly or place the mesh at "
                f"{output_folder}/recon.ply"
            )
        mesh_vertices, mesh_faces = load_ply(mesh_path_str)
        print(f"    Vertices: {len(mesh_vertices)},  Faces: {len(mesh_faces)}")
        if not args.no_preprocess:
            mesh_vertices, mesh_faces = preprocess_mesh(
                mesh_vertices, mesh_faces,
                keep_largest_component=True,
                decimate_target=args.decimate_target,
                verbose=True,
            )

    # ── 3. (Optional) Embed Gaussians into mesh ─────────────────────
    gaussian_vertex_indices = None
    if args.embed_gaussians:
        mesh_vertices, mesh_faces, gaussian_vertex_indices = embed_gaussians_in_mesh(
            mesh_vertices=mesh_vertices,
            mesh_faces=mesh_faces,
            gaussian_positions=gaussian_positions,
            snap_threshold=args.snap_threshold,
            verbose=True,
        )

    # ── 4. Select source vertices ─────────────────────────────────────
    all_source_indices, all_source_positions = select_sources_near_gaussians(
        mesh_vertices=mesh_vertices,
        gaussian_positions=gaussian_positions,
        num_sources=args.num_sources,
        proximity_factor=args.proximity_factor,
        seed=args.seed,
        gaussian_vertex_indices=gaussian_vertex_indices,
    )
    total_sources = len(all_source_indices)

    # Determine source range for this run
    if args.source_start is not None and args.source_end is not None:
        source_start = args.source_start
        source_end = min(args.source_end, total_sources)
    else:
        source_start = 0
        source_end = total_sources

    source_indices = all_source_indices[source_start:source_end]
    source_positions = all_source_positions[source_start:source_end]
    print(f"  Source range: [{source_start}, {source_end}) "
          f"({len(source_indices)} sources)")

    # ── 5. Map sources → nearest Gaussian, then get consistent mesh indices ──
    # First find which Gaussian is nearest to each source position.  Then
    # override source_indices with the mesh vertex closest to THAT Gaussian.
    # This guarantees source_indices[i] is the mesh footprint of Gaussian
    # source_gaussian_indices[i], so geodesic distance from the source to
    # that Gaussian will be ~0.
    gauss_tree = KDTree(gaussian_positions)
    _, source_gaussian_indices = gauss_tree.query(source_positions)
    source_gaussian_indices = source_gaussian_indices.astype(np.intp)
    # Update source_indices and source_positions to be consistent with the
    # nearest Gaussian's mesh footprint.
    source_indices, _ = find_closest_mesh_vertices(
        gaussian_centers=gaussian_positions[source_gaussian_indices],
        mesh_vertices=mesh_vertices,
    )
    source_positions = mesh_vertices[source_indices]

    # ── 6. Find closest mesh vertex / face for each Gaussian ──────────
    gaussian_to_mesh_indices, gaussian_to_mesh_distances = find_closest_mesh_vertices(
        gaussian_centers=gaussian_positions,
        mesh_vertices=mesh_vertices,
    )

    if gaussian_vertex_indices is not None:
        # Embedded path: skip barycentric face search — Gaussians are
        # already mesh vertices.
        barycentric_face_vertices = None
        barycentric_weights = None
        # Override closest-mesh mapping with the exact embedded indices
        gaussian_to_mesh_indices = gaussian_vertex_indices
        gaussian_to_mesh_distances = np.linalg.norm(
            gaussian_positions - mesh_vertices[gaussian_vertex_indices], axis=1
        )
    else:
        barycentric_face_vertices, barycentric_weights = (
            find_closest_mesh_faces_barycentric(
                gaussian_centers=gaussian_positions,
                mesh_vertices=mesh_vertices,
                mesh_faces=mesh_faces,
                closest_vertex_indices=gaussian_to_mesh_indices,
            )
        )

    # ── 7–9. Compute geodesics, transfer to Gaussians, and save ─────
    #         All done in parallel batches via the unified pipeline.
    compute_and_save_geodesic_pipeline(
        vertices=mesh_vertices,
        faces=mesh_faces,
        source_indices=source_indices,
        source_positions=source_positions,
        source_gaussian_indices=source_gaussian_indices,
        gaussian_positions=gaussian_positions,
        gaussian_to_mesh_indices=gaussian_to_mesh_indices,
        gaussian_to_mesh_distances=gaussian_to_mesh_distances,
        geodesic_method=args.geodesic_method,
        n_jobs=args.n_jobs,
        partial_save_dir=str(output_folder / "geodesic_distance" / "mesh_batch_cache"),
        output_dir=str(output_folder / "geodesic_distance" / "gt_partial"),
        barycentric_face_vertices=barycentric_face_vertices,
        barycentric_weights=barycentric_weights,
        gaussian_vertex_indices=gaussian_vertex_indices,
        verbose=args.verbose,
    )

    _save_computation_metadata(
        output_folder=output_folder,
        args=args,
        num_gaussians=n_gauss,
        num_sources=total_sources,
        mesh_path=mesh_path_str,
        num_mesh_vertices=len(mesh_vertices),
        num_mesh_faces=len(mesh_faces),
    )

    print(f"\n{'#'*80}")
    print(f"# Computation Complete")
    print(f"{'#'*80}")
    print(f"\nPartial results saved to:")
    print(f"  {output_folder / 'geodesic_distance' / 'gt_partial'}")
    print(f"\nTo merge all partial results, run:")
    print(f"  python {Path(__file__).name} --gaussian_output {args.gaussian_output} --merge_only")


if __name__ == "__main__":
    main()
