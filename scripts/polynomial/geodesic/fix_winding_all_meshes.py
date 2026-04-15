#!/usr/bin/env python3
"""Fix face winding consistency for all polynomial geodesic meshes.

The mesh builder produces faces with ~6% inconsistent winding, where adjacent faces
have half-edges in the same direction instead of opposite. This breaks VTP geodesic
distance computation ("linked list error").

This script uses BFS-based winding propagation to make all faces consistently oriented:
1. Build face adjacency graph via shared edges (vectorized numpy)
2. BFS from face 0, assigning parity (flip/keep) to each face
3. Flip faces with parity=1 (reverse vertex order)
4. Save updated npz/ply files
"""

import numpy as np
from collections import deque
from pathlib import Path
import time
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def fix_face_winding(faces, nv):
    """Fix face winding consistency using BFS propagation.
    
    Args:
        faces: (nf, 3) int array of face vertex indices
        nv: number of vertices
    
    Returns:
        faces_fixed: (nf, 3) int array with consistent winding
        n_flipped: number of faces that were flipped
        n_inconsistent_before: number of inconsistent edges before fix
    """
    faces = faces.astype(np.int64)
    nf = faces.shape[0]

    # Build half-edge arrays
    he_a = np.empty(3 * nf, dtype=np.int64)
    he_b = np.empty(3 * nf, dtype=np.int64)
    he_a[0::3] = faces[:, 0]; he_b[0::3] = faces[:, 1]
    he_a[1::3] = faces[:, 1]; he_b[1::3] = faces[:, 2]
    he_a[2::3] = faces[:, 2]; he_b[2::3] = faces[:, 0]

    face_of_he = np.arange(3 * nf, dtype=np.int64) // 3

    emin = np.minimum(he_a, he_b)
    emax = np.maximum(he_a, he_b)
    direction = (he_a < he_b).astype(np.int8)
    edge_key = emin * (nv + 1) + emax

    # Sort by edge key
    sort_idx = np.argsort(edge_key)
    sorted_keys = edge_key[sort_idx]
    sorted_face = face_of_he[sort_idx]
    sorted_dirs = direction[sort_idx]

    # Find interior edge pairs
    unique_keys, starts, counts = np.unique(sorted_keys, return_index=True, return_counts=True)
    interior = counts == 2
    interior_starts = starts[interior]

    fi = sorted_face[interior_starts]
    fj = sorted_face[interior_starts + 1]
    di = sorted_dirs[interior_starts]
    dj = sorted_dirs[interior_starts + 1]

    is_inconsistent = (di == dj).astype(np.int8)
    n_inconsistent_before = int(np.sum(is_inconsistent))

    if n_inconsistent_before == 0:
        return faces.copy(), 0, 0

    # Build adjacency list
    adj = [[] for _ in range(nf)]
    for idx in range(len(fi)):
        f1 = int(fi[idx])
        f2 = int(fj[idx])
        inc = int(is_inconsistent[idx])
        adj[f1].append((f2, inc))
        adj[f2].append((f1, inc))

    # BFS parity propagation
    parity = np.full(nf, -1, dtype=np.int8)
    for seed in range(nf):
        if parity[seed] >= 0:
            continue
        parity[seed] = 0
        queue = deque([seed])
        while queue:
            ci = queue.popleft()
            cp = parity[ci]
            for (ni, inc) in adj[ci]:
                if parity[ni] >= 0:
                    continue
                parity[ni] = cp ^ inc
                queue.append(ni)

    # Flip faces
    faces_fixed = faces.copy()
    flip_mask = parity == 1
    faces_fixed[flip_mask] = faces_fixed[flip_mask][:, [0, 2, 1]]
    n_flipped = int(np.sum(flip_mask))

    return faces_fixed.astype(np.int32), n_flipped, n_inconsistent_before


def count_inconsistent_edges(faces, nv):
    """Count inconsistent edges (vectorized)."""
    faces = faces.astype(np.int64)
    nf = faces.shape[0]
    he_a = np.empty(3 * nf, dtype=np.int64)
    he_b = np.empty(3 * nf, dtype=np.int64)
    he_a[0::3] = faces[:, 0]; he_b[0::3] = faces[:, 1]
    he_a[1::3] = faces[:, 1]; he_b[1::3] = faces[:, 2]
    he_a[2::3] = faces[:, 2]; he_b[2::3] = faces[:, 0]
    emin = np.minimum(he_a, he_b)
    emax = np.maximum(he_a, he_b)
    direction = (he_a < he_b).astype(np.int8)
    edge_key = emin * (nv + 1) + emax
    sort_idx = np.argsort(edge_key)
    sk = edge_key[sort_idx]
    sd = direction[sort_idx]
    uk, st, ct = np.unique(sk, return_index=True, return_counts=True)
    interior = ct == 2
    ist = st[interior]
    return int(np.sum(sd[ist] == sd[ist + 1]))


def save_ply(filepath, vertices, faces):
    """Save mesh to PLY format."""
    nv, nf = len(vertices), len(faces)
    with open(filepath, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {nv}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {nf}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true',
                        help='Process ALL meshes (default: only the 4 that failed VTP)')
    cli = parser.parse_args()

    base = Path('/home/rotem.shezaf/RaDe-GS/TrainData/Polynomial/SyntheticColmapData/blue_texture')

    if cli.all:
        surfaces = ['HyperbolicParaboloid', 'Paraboloid', 'Sinusoidal']
        levels = ['level_00', 'level_01', 'level_02', 'level_03', 'level_04']
        lights = ['light_0', 'light_1', 'light_2', 'light_3', 'light_4']
        targets = [(s, lv, lt) for s in surfaces for lv in levels for lt in lights]
    else:
        # Only the 4 meshes whose VTP geodesic computation failed
        targets = [
            ('HyperbolicParaboloid', 'level_02', 'light_2'),
            ('HyperbolicParaboloid', 'level_04', 'light_1'),
            ('Paraboloid', 'level_02', 'light_3'),
            ('Saddle', 'level_04', 'light_1'),
        ]

    from GenerateData.utils.geodesic_mesh_utils import prepare_mesh_for_vtp

    results = []
    for surface, level, light in targets:
        mesh_dir = base / surface / level / light / 'output' / 'geodesic_mesh'
        npz_path = mesh_dir / 'geodesic_mesh_data.npz'
        if not npz_path.exists():
            print(f"SKIP {surface}/{level}/{light}: no npz")
            continue

        tag = f"{surface}/{level}/{light}"
        t0 = time.time()
        print(f"Processing {tag} ...")

        data = dict(np.load(npz_path, allow_pickle=True))
        verts = data['vertices']
        faces = data['faces']
        gi = data['gaussian_vertex_indices']
        nv = verts.shape[0]
        nf_before = faces.shape[0]

        faces_fixed = prepare_mesh_for_vtp(
            verts, faces, gi, verbose=True,
        )
        nf_after = faces_fixed.shape[0]

        # Compact unreferenced vertices after VTP preparation
        used_v = np.unique(faces_fixed.ravel())
        if len(used_v) < len(verts):
            n_orphan = len(verts) - len(used_v)
            remap = np.full(len(verts), -1, dtype=np.int32)
            remap[used_v] = np.arange(len(used_v), dtype=np.int32)
            verts = verts[used_v]
            faces_fixed = remap[faces_fixed]
            gi = remap[gi]
            assert (gi >= 0).all(), "VTP preparation orphaned a Gaussian vertex"
            nv = verts.shape[0]
            print(f"    → compacted {n_orphan} unreferenced vertices")

        if np.array_equal(faces, faces_fixed) and len(used_v) == len(data['vertices']):
            print(f"  OK  {tag}: no changes needed")
            results.append((tag, nf_before, nf_after, 0))
            continue

        # Verify: count remaining inconsistent edges
        n_incon_after = count_inconsistent_edges(faces_fixed, nv)
        if n_incon_after > 0:
            print(f"  WARNING: {n_incon_after} inconsistent edges remain")

        # Save npz
        data['faces'] = faces_fixed
        data['vertices'] = verts
        data['gaussian_vertex_indices'] = gi
        np.savez(npz_path, **data)

        # Save ply
        ply_path = mesh_dir / 'geodesic_mesh.ply'
        if ply_path.exists():
            save_ply(str(ply_path), verts, faces_fixed)

        # Update statistics json
        json_path = mesh_dir / 'mesh_statistics.json'
        if json_path.exists():
            with open(json_path) as jf:
                stats = json.load(jf)
            stats['vtp_prep'] = {
                'faces_before': int(nf_before),
                'faces_after': int(nf_after),
                'inconsistent_edges_after': int(n_incon_after),
            }
            with open(json_path, 'w') as jf:
                json.dump(stats, jf, indent=2)

        elapsed = time.time() - t0
        print(f" FIXED {tag}: {nf_before} → {nf_after} faces ({elapsed:.1f}s)")
        results.append((tag, nf_before, nf_after, elapsed))

    print(f"\n{'='*60}")
    print(f"Summary: {len(results)} meshes processed")
    for tag, nf_before, nf_after, elapsed in results:
        if nf_before != nf_after:
            print(f"  FIXED {tag}: {nf_before} → {nf_after} faces ({elapsed:.1f}s)")
        else:
            print(f"  OK    {tag}: no changes")


if __name__ == '__main__':
    main()
