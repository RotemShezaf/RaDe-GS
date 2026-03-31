#!/usr/bin/env python3
"""
Utilities for building geodesic-quality meshes from Gaussian splats on polynomial surfaces.

This module provides functions for:
- Projecting Gaussian splat positions onto analytical polynomial surfaces
- Sampling additional points uniformly on the surface (via Poisson-disk / min-radius)
- Building a Delaunay triangulation on the (u, v) parameter space
- Saving / loading the resulting mesh + Gaussian-to-vertex mapping
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.spatial import ConvexHull, Delaunay, KDTree
import trimesh

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.GenerateRawPolynomialMesh import evaluate_polynomial

__all__ = [
    "project_gaussians_to_surface",
    "compute_surface_curvature",
    "sample_surface_uniform",
    "build_surface_delaunay",
    "build_surface_delaunay_3d",
    "build_surface_ball_pivoting",
    "build_surface_grid",
    "insert_gaussians_into_grid_mesh",
    "insert_steiner_points",
    "_bowyer_watson_insert",
    "_uniform_laplacian_smooth",
    "_surface_metric_midpoints",
    "_surface_distances",
    "_metric_weighted_laplacian_smooth",
    "_filter_convex_hull_artifacts",
    "_split_large_edges",
    "fix_orphaned_gaussians",
    "_fix_invalid_gaussians_with_rings",
    "refine_bad_triangles",
    "save_geodesic_mesh",
    "load_geodesic_mesh",
]


# ── Surface projection ──────────────────────────────────────────────────────

def _evaluate_gradient(x, y, surface_type):
    """Return (df/dx, df/dy) for the polynomial surface z = f(x, y)."""
    if surface_type == 'Paraboloid':
        return 2.0 * x, 2.0 * y
    elif surface_type == 'Saddle':
        return 2.0 * x, -2.0 * y
    elif surface_type == 'HyperbolicParaboloid':
        return 2.0 * x + y, -2.0 * y + x
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")


def _evaluate_hessian(x, y, surface_type):
    """Return (fxx, fxy, fyy) for the polynomial surface z = f(x, y)."""
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)
    if surface_type == 'Paraboloid':
        return 2.0 * ones, zeros, 2.0 * ones
    elif surface_type == 'Saddle':
        return 2.0 * ones, zeros, -2.0 * ones
    elif surface_type == 'HyperbolicParaboloid':
        return 2.0 * ones, ones, -2.0 * ones
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")


def _project_paraboloid_analytical(px, py, pz):
    """Closest-point projection onto z = x² + y² via depressed cubic.

    Due to radial symmetry the 3-D problem reduces to finding the
    optimal radius *r* on the parabola z = r² closest to
    (p_r, p_z) where p_r = sqrt(px² + py²).

    The optimality condition is the depressed cubic
        2r³ + (1 - 2·p_z)·r - p_r = 0
    solved with Cardano / trigonometric method.
    """
    pr = np.sqrt(px ** 2 + py ** 2)

    # Depressed cubic:  r³ + p·r + q = 0
    # with  p = (1 - 2·pz)/2,  q = -pr/2
    p = (1.0 - 2.0 * pz) / 2.0
    q = -pr / 2.0

    discriminant = -(4.0 * p ** 3 + 27.0 * q ** 2)

    r = np.empty_like(pr)

    # --- Case 1:  discriminant ≤ 0  → one real root (Cardano) ----------
    mask1 = discriminant <= 0
    if np.any(mask1):
        p1, q1 = p[mask1], q[mask1]
        inner = q1 ** 2 / 4.0 + p1 ** 3 / 27.0
        inner = np.maximum(inner, 0.0)
        sqrt_inner = np.sqrt(inner)
        A = np.cbrt(-q1 / 2.0 + sqrt_inner)
        B = np.cbrt(-q1 / 2.0 - sqrt_inner)
        r[mask1] = A + B

    # --- Case 2:  discriminant > 0  → three real roots (trigonometric) --
    mask3 = ~mask1
    if np.any(mask3):
        p3, q3, pr3 = p[mask3], q[mask3], pr[mask3]
        m = 2.0 * np.sqrt(-p3 / 3.0)
        theta = np.arccos(3.0 * q3 / (p3 * m + 1e-30)) / 3.0
        # Three candidate roots
        r0 = m * np.cos(theta)
        r1 = m * np.cos(theta - 2.0 * np.pi / 3.0)
        r2 = m * np.cos(theta - 4.0 * np.pi / 3.0)
        # Pick the non-negative root closest to pr (minimises distance)
        pz3 = pz[mask3]
        best = np.full_like(pr3, np.inf)
        best_r = np.zeros_like(pr3)
        for rc in (r0, r1, r2):
            # Allow slightly negative roots (numerical noise)
            rc_clamp = np.maximum(rc, 0.0)
            dist2 = (rc_clamp - pr3) ** 2 + (rc_clamp ** 2 - pz3) ** 2
            better = dist2 < best
            best_r = np.where(better, rc_clamp, best_r)
            best = np.where(better, dist2, best)
        r[mask3] = best_r

    # Ensure r ≥ 0 (the origin case: pr==0 → r=0 already natural)
    r = np.maximum(r, 0.0)

    # Map back to (x, y): same direction as (px, py), magnitude = r
    scale = np.where(pr > 1e-15, r / pr, 0.0)
    x = px * scale
    y = py * scale
    z = x ** 2 + y ** 2
    return x, y, z


def _project_saddle_polynomial(px, py, pz):
    """Closest-point projection onto z = x² - y² via polynomial root-finding.

    The optimality conditions reduce to a degree-5 polynomial in
    t = 2·(x² - y² - pz) whose coefficients are known analytically.
    We solve for roots with ``np.roots`` and pick the real root giving
    the minimum distance.
    """
    n = len(px)
    x_out = np.empty(n, dtype=np.float64)
    y_out = np.empty(n, dtype=np.float64)

    pr2 = px ** 2 + py ** 2
    pd2 = px ** 2 - py ** 2  # px²-py²

    for i in range(n):
        # Polynomial:  c5·t⁵ + c4·t⁴ + c3·t³ + c2·t² + c1·t + c0 = 0
        pzi = pz[i]
        c5 = 0.5
        c4 = pzi
        c3 = -1.0
        c2 = -2.0 * pzi - pd2[i]
        c1 = 0.5 + 2.0 * pr2[i]
        c0 = pzi - pd2[i]

        roots = np.roots([c5, c4, c3, c2, c1, c0])

        best_dist2 = np.inf
        best_xy = (px[i], py[i])  # fallback: vertical projection

        for root in roots:
            if np.abs(root.imag) > 1e-8:
                continue
            t = root.real
            denom_x = 1.0 + t
            denom_y = 1.0 - t
            if np.abs(denom_x) < 1e-12 or np.abs(denom_y) < 1e-12:
                continue
            xi = px[i] / denom_x
            yi = py[i] / denom_y
            zi = xi ** 2 - yi ** 2
            dist2 = (xi - px[i]) ** 2 + (yi - py[i]) ** 2 + (zi - pzi) ** 2
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_xy = (xi, yi)

        x_out[i] = best_xy[0]
        y_out[i] = best_xy[1]

    z_out = x_out ** 2 - y_out ** 2
    return x_out, y_out, z_out


def _project_hyperbolic_paraboloid_newton(px, py, pz, max_iter=30, tol=1e-12):
    """Closest-point projection onto z = x² - y² + xy via Newton's method.

    The cross-term makes a single-variable polynomial reduction
    impractical, so we use vectorised Newton with analytical Hessian.
    """
    x = px.copy()
    y = py.copy()

    for _ in range(max_iter):
        f = x ** 2 - y ** 2 + x * y
        fx = 2.0 * x + y
        fy = -2.0 * y + x
        r = f - pz

        gx = (x - px) + r * fx
        gy = (y - py) + r * fy

        Hxx = 1.0 + fx * fx + r * 2.0
        Hxy = fx * fy + r * 1.0
        Hyy = 1.0 + fy * fy + r * (-2.0)

        det = Hxx * Hyy - Hxy * Hxy
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)
        dx = -(Hyy * gx - Hxy * gy) / det
        dy = -(-Hxy * gx + Hxx * gy) / det

        x += dx
        y += dy

        grad_norm = np.sqrt(gx * gx + gy * gy)
        if np.max(grad_norm) < tol:
            break

    z = x ** 2 - y ** 2 + x * y
    return x, y, z


def project_gaussians_to_surface(
    positions: np.ndarray,
    surface_type: str,
) -> np.ndarray:
    """Project Gaussian positions onto the closest point on a polynomial surface.

    Uses the exact analytical solution where possible:

    * **Paraboloid** (z = x² + y²): Radial symmetry reduces to a
      depressed cubic solved via Cardano / trigonometric formula.
    * **Saddle** (z = x² - y²): Optimality conditions yield a
      degree-5 polynomial in one variable; roots found via
      ``np.roots`` and the minimum-distance real root is selected.
    * **HyperbolicParaboloid** (z = x² - y² + xy): Vectorised Newton
      with analytical Hessian (cross-term prevents polynomial reduction).

    Parameters
    ----------
    positions : ndarray, shape ``(N, 3)``
        Gaussian center positions.
    surface_type : str
        One of ``'Paraboloid'``, ``'Saddle'``, ``'HyperbolicParaboloid'``.

    Returns
    -------
    projected : ndarray, shape ``(N, 3)``
        Closest points on the surface.
    """
    px = positions[:, 0].astype(np.float64)
    py = positions[:, 1].astype(np.float64)
    pz = positions[:, 2].astype(np.float64)

    if surface_type == 'Paraboloid':
        x, y, z = _project_paraboloid_analytical(px, py, pz)
    elif surface_type == 'Saddle':
        x, y, z = _project_saddle_polynomial(px, py, pz)
    elif surface_type == 'HyperbolicParaboloid':
        x, y, z = _project_hyperbolic_paraboloid_newton(px, py, pz)
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")

    # Guarantee z is exactly on the surface (guards against numerical
    # drift in iterative solvers like Newton for HyperbolicParaboloid).
    z = evaluate_polynomial(x, y, surface_type)

    return np.column_stack([x, y, z])


# ── Surface curvature ───────────────────────────────────────────────────────

def compute_surface_curvature(
    x: np.ndarray,
    y: np.ndarray,
    surface_type: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gaussian curvature *K* and mean curvature *H* analytically.

    For a surface ``z = f(x, y)``:

    .. math::

        K = \\frac{f_{xx}\\,f_{yy} - f_{xy}^{\\,2}}
                 {(1 + f_x^2 + f_y^2)^2}

        H = \\frac{(1+f_y^2)\\,f_{xx} - 2\\,f_x\\,f_y\\,f_{xy}
                  + (1+f_x^2)\\,f_{yy}}
                 {2\\,(1 + f_x^2 + f_y^2)^{3/2}}

    Parameters
    ----------
    x, y : ndarray
        Parameter-space coordinates.
    surface_type : str
        ``'Paraboloid'``, ``'Saddle'``, or ``'HyperbolicParaboloid'``.

    Returns
    -------
    K : ndarray
        Gaussian curvature at each point.
    H : ndarray
        Mean curvature at each point.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if surface_type == "Paraboloid":
        # z = x² + y²
        fx, fy = 2.0 * x, 2.0 * y
        fxx, fyy, fxy = 2.0, 2.0, 0.0
    elif surface_type == "Saddle":
        # z = x² − y²
        fx, fy = 2.0 * x, -2.0 * y
        fxx, fyy, fxy = 2.0, -2.0, 0.0
    elif surface_type == "HyperbolicParaboloid":
        # z = x² − y² + xy
        fx, fy = 2.0 * x + y, -2.0 * y + x
        fxx, fyy, fxy = 2.0, -2.0, 1.0
    else:
        raise ValueError(f"Unknown surface_type: {surface_type}")

    g = 1.0 + fx**2 + fy**2  # 1 + |∇f|²

    K = (fxx * fyy - fxy**2) / g**2
    H = ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy) / (
        2 * g**1.5
    )

    return K, H


# ── Uniform surface sampling ────────────────────────────────────────────────

def sample_surface_uniform(
    surface_type: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    *,
    n_points: Optional[int] = None,
    min_radius: Optional[float] = None,
    existing_xy: Optional[np.ndarray] = None,
    seed: int = 42,
    curvature_adaptive: bool = False,
    curvature_alpha: float = 2.0,
) -> np.ndarray:
    """Sample points uniformly on a polynomial surface via Poisson-disk sampling.

    Uses a fast *jittered-grid + KDTree self-filter* algorithm (O(N log N),
    fully vectorized) instead of per-point dart-throwing.

    Exactly one of *n_points* or *min_radius* must be given.

    * **n_points**: binary-search on the radius to produce approximately
      *n_points* additional surface samples.
    * **min_radius**: use the given minimum spacing directly.

    In both modes, *existing_xy* (the projected Gaussian ``(x, y)``
    positions) is taken into account so that new samples respect the
    spacing constraint with respect to existing points.

    Parameters
    ----------
    surface_type : str
    x_range, y_range : tuple of float
        Domain bounds.
    n_points : int, optional
        Target number of additional points to sample.
    min_radius : float, optional
        Minimum spacing between any two points (existing or new) in the
        ``(x, y)`` parameter space.
    existing_xy : ndarray of shape ``(M, 2)``, optional
        Already-placed ``(x, y)`` positions (e.g. projected Gaussians).
    seed : int
        RNG seed.
    curvature_adaptive : bool
        If ``True``, use curvature-dependent local radius: denser in
        high-curvature regions, sparser in flat regions.
    curvature_alpha : float
        Strength of curvature adaptation.  Local radius is
        ``r_base / sqrt(1 + alpha * |K| / K_ref)`` where ``K_ref`` is
        the median absolute Gaussian curvature over the domain.
        Higher values → more aggressive refinement near curvature.
        Typical range: 1–10.  Default 2.0.

    Returns
    -------
    new_points : ndarray, shape ``(K, 3)``
        Newly sampled surface points.  ``K ≈ n_points`` when *n_points*
        is given; ``K`` depends on *min_radius* otherwise.
    """
    if (n_points is None) == (min_radius is None):
        raise ValueError("Exactly one of n_points or min_radius must be given")

    # Build convex hull of existing points for domain clipping
    hull_del = None
    if existing_xy is not None and len(existing_xy) >= 3:
        try:
            hull_del = _build_hull_delaunay(existing_xy)
        except Exception:
            pass  # degenerate hull — fall back to bounding box only

    if n_points is not None:
        new_xy = _poisson_disk_n_points(
            x_range, y_range, n_points, existing_xy, seed,
            curvature_adaptive=curvature_adaptive,
            curvature_alpha=curvature_alpha,
            surface_type=surface_type,
            hull_delaunay=hull_del,
        )
    else:
        if curvature_adaptive:
            new_xy = _poisson_disk_curvature_adaptive(
                x_range, y_range, min_radius, curvature_alpha,
                surface_type, existing_xy, seed,
                hull_delaunay=hull_del,
            )
        else:
            new_xy = _poisson_disk_by_radius(
                x_range, y_range, min_radius, existing_xy, seed,
                hull_delaunay=hull_del,
            )

    if len(new_xy) == 0:
        return np.empty((0, 3), dtype=np.float64)

    z = evaluate_polynomial(new_xy[:, 0], new_xy[:, 1], surface_type)
    return np.column_stack([new_xy[:, 0], new_xy[:, 1], z])


# ── Fast Poisson-disk internals ─────────────────────────────────────────────

def _build_hull_delaunay(hull_xy: np.ndarray) -> Delaunay:
    """Build a Delaunay triangulation of the convex hull for inside-testing."""
    hull = ConvexHull(hull_xy)
    hull_pts = hull_xy[hull.vertices]
    return Delaunay(hull_pts)


def _filter_inside_hull(
    candidates: np.ndarray,
    hull_delaunay: Optional[Delaunay],
) -> np.ndarray:
    """Keep only candidates that lie inside (or on) the convex hull."""
    if hull_delaunay is None or len(candidates) == 0:
        return candidates
    inside = hull_delaunay.find_simplex(candidates) >= 0
    return candidates[inside]


def _poisson_disk_by_radius(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    radius: float,
    existing_xy: Optional[np.ndarray],
    seed: int,
    hull_delaunay: Optional[Delaunay] = None,
) -> np.ndarray:
    """Fast Poisson-disk sampling via jittered grid + KDTree self-filter.

    Algorithm
    ---------
    1. Tile the ``(x, y)`` domain with cells of size ``radius / √2``.
       Place one uniformly random point per cell (fully vectorized).
    2. Remove candidates within *radius* of any *existing_xy* point
       (single KDTree query, vectorized).
    3. Build a KDTree of the remaining candidates, find all pairs closer
       than *radius*, and greedily keep the lower-index point in each
       conflict (deterministic, fast).

    The result is a set of points with **guaranteed** minimum pairwise
    distance ≥ *radius* (and ≥ *radius* from *existing_xy*).

    Complexity is O(N log N) where N = number of grid cells.
    """
    rng = np.random.RandomState(seed)
    cell = radius / np.sqrt(2)
    x0, x1 = x_range
    y0, y1 = y_range
    nx = int(np.ceil((x1 - x0) / cell))
    ny = int(np.ceil((y1 - y0) / cell))

    # 1. Jittered grid — one random point per cell
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    gx, gy = gx.ravel(), gy.ravel()
    n = len(gx)
    pts_x = x0 + (gx + rng.uniform(0, 1, n)) * cell
    pts_y = y0 + (gy + rng.uniform(0, 1, n)) * cell

    # Clip to domain
    mask = (pts_x >= x0) & (pts_x <= x1) & (pts_y >= y0) & (pts_y <= y1)
    candidates = np.column_stack([pts_x[mask], pts_y[mask]])

    # Filter to convex hull of Gaussians
    candidates = _filter_inside_hull(candidates, hull_delaunay)

    # Shuffle to break grid-order bias in the greedy conflict resolution
    rng.shuffle(candidates)

    # 2. Reject candidates too close to existing points
    if existing_xy is not None and len(existing_xy) > 0:
        tree_exist = KDTree(existing_xy)
        dists, _ = tree_exist.query(candidates)
        candidates = candidates[dists >= radius]

    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # 3. Self-filter: greedily resolve close pairs
    tree = KDTree(candidates)
    pairs = tree.query_pairs(radius)

    if len(pairs) == 0:
        return candidates

    # For each conflicting pair, remove the higher-index point
    remove: set = set()
    for i, j in pairs:
        if i not in remove and j not in remove:
            remove.add(max(i, j))

    keep = np.array(sorted(set(range(len(candidates))) - remove))
    return candidates[keep]


def _poisson_disk_curvature_adaptive(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    r_base: float,
    alpha: float,
    surface_type: str,
    existing_xy: Optional[np.ndarray],
    seed: int,
    hull_delaunay: Optional[Delaunay] = None,
) -> np.ndarray:
    """Adaptive-radius Poisson-disk sampling biased by surface curvature.

    Strategy
    --------
    1. Generate a dense jittered-grid sample at a small cell size so that
       high-curvature regions are well-resolved.  This skips the expensive
       ``query_pairs`` conflict resolution since we thin the sample next.
    2. Thin the sample by curvature-weighted rejection: points in
       low-curvature regions are removed with probability proportional
       to how much their local spacing exceeds the minimum radius.

    This is very fast: the grid is fully vectorized, existing-point
    rejection is a single KDTree query, and thinning is O(N).
    """
    rng = np.random.RandomState(seed)

    # --- Estimate curvature range over domain --------------------------
    nx_probe, ny_probe = 50, 50
    x_probe = np.linspace(x_range[0], x_range[1], nx_probe)
    y_probe = np.linspace(y_range[0], y_range[1], ny_probe)
    Xp, Yp = np.meshgrid(x_probe, y_probe)
    K_probe, _ = compute_surface_curvature(
        Xp.ravel(), Yp.ravel(), surface_type,
    )
    absK_probe = np.abs(K_probe)
    nz = absK_probe[absK_probe > 1e-12]
    K_ref = float(np.median(nz)) if len(nz) > 0 else 1.0
    K_max = float(np.max(absK_probe)) if len(absK_probe) > 0 else 0.0

    # Minimum local radius (at maximum curvature), clamped to 40% of base
    r_min = r_base / np.sqrt(1 + alpha * K_max / K_ref) if K_max > 0 else r_base
    r_min = max(r_min, r_base * 0.40)

    # 1. Generate dense jittered grid at r_min spacing (no query_pairs)
    x0, x1 = x_range
    y0, y1 = y_range
    cell = r_min / np.sqrt(2)
    nx_cells = int(np.ceil((x1 - x0) / cell))
    ny_cells = int(np.ceil((y1 - y0) / cell))

    gx, gy = np.meshgrid(np.arange(nx_cells), np.arange(ny_cells))
    gx, gy = gx.ravel(), gy.ravel()
    n = len(gx)
    pts_x = x0 + (gx + rng.uniform(0, 1, n)) * cell
    pts_y = y0 + (gy + rng.uniform(0, 1, n)) * cell

    # Clip to domain
    mask = (pts_x >= x0) & (pts_x <= x1) & (pts_y >= y0) & (pts_y <= y1)
    candidates = np.column_stack([pts_x[mask], pts_y[mask]])

    # Filter to convex hull of Gaussians
    candidates = _filter_inside_hull(candidates, hull_delaunay)

    # Remove candidates too close to existing points
    if existing_xy is not None and len(existing_xy) > 0:
        tree_exist = KDTree(existing_xy)
        dists, _ = tree_exist.query(candidates)
        candidates = candidates[dists >= r_min]

    if len(candidates) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # 2. Compute local curvature at each candidate
    K_cand, _ = compute_surface_curvature(candidates[:, 0], candidates[:, 1], surface_type)
    absK_cand = np.abs(K_cand)

    # Desired local radius at each point
    local_r = r_base / np.sqrt(1 + alpha * absK_cand / K_ref)

    # Keep probability: (r_min / local_r)^2
    # At max curvature: local_r ≈ r_min  → keep_prob ≈ 1 (keep all)
    # At zero curvature: local_r ≈ r_base → keep_prob ≈ (r_min/r_base)^2 (thin out)
    keep_prob = np.clip((r_min / local_r) ** 2, 0.0, 1.0)

    # Always keep the highest-curvature quartile
    q75 = np.percentile(absK_cand, 75)
    keep_prob[absK_cand >= q75] = 1.0

    # Stochastic thinning
    accept = rng.uniform(0, 1, len(candidates)) < keep_prob
    result = candidates[accept]

    return result


def _poisson_disk_n_points(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    n_points: int,
    existing_xy: Optional[np.ndarray],
    seed: int,
    n_iters: int = 12,
    *,
    curvature_adaptive: bool = False,
    curvature_alpha: float = 2.0,
    surface_type: Optional[str] = None,
    hull_delaunay: Optional[Delaunay] = None,
) -> np.ndarray:
    """Binary-search on the radius to produce ≈ *n_points* new samples.

    Each iteration calls the appropriate Poisson-disk sampler and adjusts
    the radius up (too many points) or down (too few).  Typically
    converges in 6–10 iterations.

    When *curvature_adaptive* is ``True``, the adaptive sampler is used
    so that the resulting distribution is denser in high-curvature
    regions.
    """
    area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    if area <= 0 or n_points <= 0:
        return np.empty((0, 2), dtype=np.float64)

    # Initial bracket: r² ≈ 2·area / n_cells  (cell = r/√2, cell² = r²/2)
    r_lo = np.sqrt(2 * area / (n_points * 4))   # many cells → upper bound
    r_hi = np.sqrt(2 * area / max(n_points // 4, 1))  # few cells → lower bound

    best_pts: Optional[np.ndarray] = None
    best_delta = float("inf")

    for _ in range(n_iters):
        r_mid = (r_lo + r_hi) / 2
        if curvature_adaptive and surface_type is not None:
            pts = _poisson_disk_curvature_adaptive(
                x_range, y_range, r_mid, curvature_alpha,
                surface_type, existing_xy, seed,
                hull_delaunay=hull_delaunay,
            )
        else:
            pts = _poisson_disk_by_radius(
                x_range, y_range, r_mid, existing_xy, seed,
                hull_delaunay=hull_delaunay,
            )
        delta = abs(len(pts) - n_points)

        if delta < best_delta:
            best_pts = pts
            best_delta = delta

        if len(pts) < n_points:
            r_hi = r_mid  # smaller radius → more cells
        else:
            r_lo = r_mid  # larger radius → fewer cells

    return best_pts if best_pts is not None else np.empty((0, 2), dtype=np.float64)


# ── Delaunay triangulation ──────────────────────────────────────────────────

def build_surface_delaunay(
    points_3d: np.ndarray,
    surface_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Delaunay-triangulate a set of surface points in 2-D parameter space.

    Triangulation is performed on the ``(x, y)`` coordinates; the
    resulting faces index into the full 3-D *points_3d* array.

    When *surface_type* is given, every vertex's z-coordinate is
    reprojected onto the analytical surface **before** returning,
    guaranteeing that all vertices lie exactly on the surface.

    Parameters
    ----------
    points_3d : ndarray, shape ``(N, 3)``
        Surface points (x, y, z).
    surface_type : str, optional
        Polynomial surface name (``'Paraboloid'``, ``'Saddle'``, …).
        When given, z-values are recomputed analytically.

    Returns
    -------
    vertices : ndarray, shape ``(N, 3)``
        Surface points (reprojected if *surface_type* was given).
    faces : ndarray, shape ``(F, 3)``
        Triangle connectivity.
    """
    pts = np.array(points_3d, dtype=np.float64)
    if surface_type is not None:
        pts[:, 2] = evaluate_polynomial(pts[:, 0], pts[:, 1], surface_type)
    xy = pts[:, :2]
    tri = Delaunay(xy)
    return pts, tri.simplices.astype(np.int32)


# ── 3-D Delaunay triangulation (GPU / CPU) ──────────────────────────────────

def _extract_surface_faces(
    points: np.ndarray,
    cells: np.ndarray,
    surface_type: Optional[str] = None,
) -> np.ndarray:
    """Extract boundary (surface) triangles from a tetrahedralization.

    A *boundary face* is a triangular face shared by exactly one
    tetrahedron.  When *surface_type* is given, faces whose normal
    is not aligned with the analytical surface normal are discarded
    (removes convex-hull "cap" artefacts on open surfaces).

    Parameters
    ----------
    points : ndarray ``(V, 3)``
    cells : ndarray ``(T, 4)``  — tetrahedra vertex indices
    surface_type : str, optional

    Returns
    -------
    faces : ndarray ``(F, 3)``, dtype int32
    """
    if len(cells) == 0:
        return np.empty((0, 3), dtype=np.int32)

    cells = np.asarray(cells, dtype=np.int64)

    # Four triangular faces per tetrahedron: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
    face_idx = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    all_faces = cells[:, face_idx].reshape(-1, 3)            # (4T, 3)

    # Sort each face's vertex indices for a canonical key
    sorted_faces = np.sort(all_faces, axis=1)

    # Find faces appearing exactly once (boundary)
    _, inverse, counts = np.unique(
        sorted_faces, axis=0, return_inverse=True, return_counts=True,
    )
    boundary_mask = counts[inverse] == 1
    faces = all_faces[boundary_mask].astype(np.int32)

    if len(faces) == 0:
        return faces

    # ── Filter by surface-normal alignment ────────────────────────────
    if surface_type is not None:
        v0 = points[faces[:, 0]]
        v1 = points[faces[:, 1]]
        v2 = points[faces[:, 2]]

        # Face normals
        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(norms, 1e-15)

        # Analytical surface normals at face centroids
        centroids = (v0 + v1 + v2) / 3.0
        cx, cy = centroids[:, 0], centroids[:, 1]
        eps = 1e-6
        dfdx = (evaluate_polynomial(cx + eps, cy, surface_type)
                - evaluate_polynomial(cx - eps, cy, surface_type)) / (2 * eps)
        dfdy = (evaluate_polynomial(cx, cy + eps, surface_type)
                - evaluate_polynomial(cx, cy - eps, surface_type)) / (2 * eps)
        surf_normals = np.column_stack([-dfdx, -dfdy, np.ones(len(faces))])
        surf_normals /= np.linalg.norm(surf_normals, axis=1, keepdims=True)

        dots = np.sum(face_normals * surf_normals, axis=1)
        aligned = np.abs(dots) > 0.1
        faces = faces[aligned]

        # Ensure consistent winding (normal aligns with surface normal)
        dots_kept = dots[aligned]
        flip = dots_kept < 0
        faces[flip] = faces[flip][:, [0, 2, 1]]

    return faces


def build_surface_delaunay_3d(
    points_3d: np.ndarray,
    surface_type: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """3-D Delaunay tetrahedralization → surface-triangle extraction.

    Unlike :func:`build_surface_delaunay` (which triangulates in the
    2-D ``(x, y)`` parameter plane), this function performs a full 3-D
    Delaunay tetrahedralization and extracts the boundary surface faces.
    This respects the actual 3-D geometry of the surface, producing
    better-shaped triangles in high-curvature regions.

    **GPU path** — Uses ``tetranerf.utils.extension.cpp.triangulate``
    (CGAL-based, CUDA-accelerated) when available.

    **CPU fallback** — Uses ``scipy.spatial.Delaunay`` in 3-D.

    Parameters
    ----------
    points_3d : ndarray ``(N, 3)``
        Surface points.
    surface_type : str, optional
        Polynomial surface name.  When given, z-values are reprojected
        and surface-normal filtering is applied to remove convex-hull
        cap faces.

    Returns
    -------
    vertices : ndarray ``(N, 3)``
    faces : ndarray ``(F, 3)``
    """
    pts = np.array(points_3d, dtype=np.float64)
    if surface_type is not None:
        pts[:, 2] = evaluate_polynomial(pts[:, 0], pts[:, 1], surface_type)

    # ── GPU triangulation (tetranerf / CGAL) ──────────────────────────
    cells = None
    gpu_used = False
    try:
        import torch
        from tetranerf.utils.extension import cpp
        pts_f32 = torch.from_numpy(pts.astype(np.float32)).cuda()
        cells_t = cpp.triangulate(pts_f32)
        cells = cells_t.cpu().numpy()
        gpu_used = True
    except Exception:
        pass

    # ── CPU fallback (scipy 3-D Delaunay) ─────────────────────────────
    if cells is None:
        tri = Delaunay(pts)
        cells = tri.simplices

    # ── Extract surface faces ─────────────────────────────────────────
    faces = _extract_surface_faces(pts, cells, surface_type)

    return pts, faces


def build_surface_ball_pivoting(
    points_3d: np.ndarray,
    surface_type: Optional[str] = None,
    radii: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mesh a point set using Open3D's Ball Pivoting Algorithm (BPA).

    Unlike Delaunay (which operates in 2-D parameter space and always
    produces a convex-hull triangulation), BPA rolls a ball of varying
    radii over the 3-D surface and connects triples of points it
    contacts.  This avoids the long skinny triangles that Delaunay
    creates at the convex-hull boundary, producing fewer holes.

    Parameters
    ----------
    points_3d : ndarray, shape ``(N, 3)``
        Surface points (x, y, z).
    surface_type : str, optional
        Polynomial surface name.  When given, z-values are reprojected
        and analytical normals are computed.
    radii : list of float, optional
        Ball radii for BPA.  Default: auto-detected from the point cloud
        (``[1.5×avg_nn, 3×avg_nn, 6×avg_nn]``).

    Returns
    -------
    vertices : ndarray, shape ``(V, 3)``
        Mesh vertices (may differ from input if BPA adds vertices).
    faces : ndarray, shape ``(F, 3)``
        Triangle connectivity.
    """
    import open3d as o3d

    pts = np.array(points_3d, dtype=np.float64)
    if surface_type is not None:
        pts[:, 2] = evaluate_polynomial(pts[:, 0], pts[:, 1], surface_type)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    # Compute normals — use analytical normals for known surfaces
    if surface_type is not None:
        # Analytical normal for z = f(x, y):  n = (-∂f/∂x, -∂f/∂y, 1) / ||...||
        eps = 1e-6
        x, y = pts[:, 0], pts[:, 1]
        z_px = evaluate_polynomial(x + eps, y, surface_type)
        z_mx = evaluate_polynomial(x - eps, y, surface_type)
        z_py = evaluate_polynomial(x, y + eps, surface_type)
        z_my = evaluate_polynomial(x, y - eps, surface_type)
        dfdx = (z_px - z_mx) / (2 * eps)
        dfdy = (z_py - z_my) / (2 * eps)
        normals = np.column_stack([-dfdx, -dfdy, np.ones(len(pts))])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        pcd.normals = o3d.utility.Vector3dVector(normals)
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Auto-detect radii from nearest-neighbour distances
    if radii is None:
        tree = KDTree(pts)
        dd, _ = tree.query(pts, k=2)
        avg_nn = float(np.mean(dd[:, 1]))
        radii = [1.5 * avg_nn, 3.0 * avg_nn, 6.0 * avg_nn]

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii),
    )

    verts_out = np.asarray(mesh.vertices, dtype=np.float64)
    faces_out = np.asarray(mesh.triangles, dtype=np.int32)

    # ── Post-process: fill holes left by BPA ──────────────────────────
    # BPA often fails to cover sparse regions.  We fill holes in two steps:
    #   1. trimesh.repair.fill_holes (ear-clip small boundary loops)
    #   2. Delaunay fallback for any vertices still not part of a face
    if len(faces_out) > 0:
        tm = trimesh.Trimesh(vertices=verts_out, faces=faces_out, process=False)
        trimesh.repair.fill_holes(tm)
        verts_out = np.asarray(tm.vertices, dtype=np.float64)
        faces_out = np.asarray(tm.faces, dtype=np.int32)

    # Delaunay fallback: if any input points are not in the mesh, run
    # a 2-D Delaunay on ALL points and add faces for uncovered vertices.
    if len(faces_out) > 0 and len(pts) > 0:
        covered = set(faces_out.ravel().tolist())
        uncovered = [i for i in range(len(pts)) if i not in covered]
        if uncovered:
            # Full 2-D Delaunay, but only keep faces that include ≥1 uncovered vertex
            xy_all = verts_out[:, :2] if len(verts_out) > len(pts) else pts[:, :2]
            try:
                from scipy.spatial import Delaunay as _Delaunay
                tri_full = _Delaunay(xy_all)
                uncov_set = set(uncovered)
                new_faces = []
                for simplex in tri_full.simplices:
                    if any(int(v) in uncov_set for v in simplex):
                        new_faces.append(simplex)
                if new_faces:
                    new_faces = np.array(new_faces, dtype=np.int32)
                    # Avoid duplicating faces already present
                    existing = set(map(tuple, np.sort(faces_out, axis=1).tolist()))
                    extra = []
                    for f in new_faces:
                        key = tuple(sorted(f.tolist()))
                        if key not in existing:
                            extra.append(f)
                    if extra:
                        faces_out = np.vstack([faces_out, np.array(extra, dtype=np.int32)])
            except Exception:
                pass  # silently fall back to BPA-only mesh

    # Re-project z onto the analytical surface
    if surface_type is not None and len(verts_out) > 0:
        verts_out[:, 2] = evaluate_polynomial(
            verts_out[:, 0], verts_out[:, 1], surface_type,
        )

    return verts_out, faces_out


# ── Grid-based mesh construction ────────────────────────────────────────────


def build_surface_grid(
    surface_type: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    target_edge_length: float = 0.05,
    curvature_adaptive: bool = False,
    curvature_alpha: float = 2.0,
    gaussian_xy: Optional[np.ndarray] = None,
    gaussian_density_alpha: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a structured grid mesh on a polynomial surface.

    Creates a grid in ``(x, y)`` parameter space whose spacing is
    adapted to the surface *arc-length metric*
    ``√(1 + f_x² + f_y²)`` so that edges have roughly equal 3-D
    length.  When *curvature_adaptive* is ``True``, the grid is
    further refined in high-curvature regions.

    When *gaussian_xy* is provided and *gaussian_density_alpha* > 0,
    the grid is additionally refined in regions with high Gaussian
    density, placing more vertices where Gaussians are clustered and
    fewer where they are sparse.

    Each quad cell is split into two triangles along the shorter 3-D
    diagonal for better triangle quality.

    Parameters
    ----------
    surface_type : str
        Polynomial surface name (``'Paraboloid'``, ``'Saddle'``, …).
    x_range, y_range : (float, float)
        Domain bounds in the parameter plane.
    target_edge_length : float
        Desired 3-D edge length on the surface (default 0.05).
    curvature_adaptive : bool
        If ``True``, place denser vertices near high-curvature regions.
    curvature_alpha : float
        Strength of curvature adaptation (default 2.0).
    gaussian_xy : ndarray ``(N, 2)``, optional
        2-D projected positions of Gaussians. Used with
        *gaussian_density_alpha* to make the grid denser near
        Gaussian clusters.
    gaussian_density_alpha : float
        Strength of Gaussian-density adaptation (default 0 = off).
        Values around 2–5 are typical.

    Returns
    -------
    vertices : ndarray ``(V, 3)``
        Grid vertices on the surface.
    faces : ndarray ``(F, 3)``
        Triangulated faces (2 per quad cell).
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    eps = 1e-6
    n_dense = 1000  # integration resolution

    # ── Arc-length adapted x-coordinates ──────────────────────────────
    y_mid = (y_min + y_max) / 2.0
    x_dense = np.linspace(x_min, x_max, n_dense)
    y_arr = np.full(n_dense, y_mid)

    z_xp = evaluate_polynomial(x_dense + eps, y_arr, surface_type)
    z_xm = evaluate_polynomial(x_dense - eps, y_arr, surface_type)
    dfdx = (z_xp - z_xm) / (2.0 * eps)
    metric_x = np.sqrt(1.0 + dfdx ** 2)

    if curvature_adaptive:
        K_x, _ = compute_surface_curvature(x_dense, y_arr, surface_type)
        absK = np.abs(K_x)
        K_ref = max(
            float(np.median(absK[absK > 0])) if (absK > 0).any() else 1e-6,
            1e-6,
        )
        metric_x = metric_x * np.sqrt(1.0 + curvature_alpha * absK / K_ref)

    # Gaussian-density adaptation along x
    if gaussian_xy is not None and gaussian_density_alpha > 0:
        from scipy.stats import gaussian_kde
        gx = gaussian_xy[:, 0]
        gx_in = gx[(gx >= x_min) & (gx <= x_max)]
        if len(gx_in) > 10:
            kde_x = gaussian_kde(gx_in, bw_method='silverman')
            density_x = kde_x(x_dense)
            density_x = density_x / max(float(np.median(density_x[density_x > 0])), 1e-30)
            metric_x = metric_x * np.sqrt(1.0 + gaussian_density_alpha * density_x)

    dx_param = (x_max - x_min) / (n_dense - 1)
    cumul_x = np.concatenate([[0.0], np.cumsum(metric_x[:-1] * dx_param)])
    total_sx = cumul_x[-1]

    n_x = max(int(np.round(total_sx / target_edge_length)) + 1, 3)
    s_targets_x = np.linspace(0.0, total_sx, n_x)
    x_coords = np.interp(s_targets_x, cumul_x, x_dense)

    # ── Arc-length adapted y-coordinates ──────────────────────────────
    x_mid = (x_min + x_max) / 2.0
    y_dense = np.linspace(y_min, y_max, n_dense)
    x_arr = np.full(n_dense, x_mid)

    z_yp = evaluate_polynomial(x_arr, y_dense + eps, surface_type)
    z_ym = evaluate_polynomial(x_arr, y_dense - eps, surface_type)
    dfdy = (z_yp - z_ym) / (2.0 * eps)
    metric_y = np.sqrt(1.0 + dfdy ** 2)

    if curvature_adaptive:
        K_y, _ = compute_surface_curvature(x_arr, y_dense, surface_type)
        absK = np.abs(K_y)
        K_ref_y = max(
            float(np.median(absK[absK > 0])) if (absK > 0).any() else 1e-6,
            1e-6,
        )
        metric_y = metric_y * np.sqrt(1.0 + curvature_alpha * absK / K_ref_y)

    # Gaussian-density adaptation along y
    if gaussian_xy is not None and gaussian_density_alpha > 0:
        from scipy.stats import gaussian_kde
        gy = gaussian_xy[:, 1]
        gy_in = gy[(gy >= y_min) & (gy <= y_max)]
        if len(gy_in) > 10:
            kde_y = gaussian_kde(gy_in, bw_method='silverman')
            density_y = kde_y(y_dense)
            density_y = density_y / max(float(np.median(density_y[density_y > 0])), 1e-30)
            metric_y = metric_y * np.sqrt(1.0 + gaussian_density_alpha * density_y)

    dy_param = (y_max - y_min) / (n_dense - 1)
    cumul_y = np.concatenate([[0.0], np.cumsum(metric_y[:-1] * dy_param)])
    total_sy = cumul_y[-1]

    n_y = max(int(np.round(total_sy / target_edge_length)) + 1, 3)
    s_targets_y = np.linspace(0.0, total_sy, n_y)
    y_coords = np.interp(s_targets_y, cumul_y, y_dense)

    # ── Build grid vertices ───────────────────────────────────────────
    xx, yy = np.meshgrid(x_coords, y_coords, indexing="xy")
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    z = evaluate_polynomial(grid_xy[:, 0], grid_xy[:, 1], surface_type)
    vertices = np.column_stack([grid_xy, z]).astype(np.float64)

    # ── Structured triangulation ──────────────────────────────────────
    # Split each quad along the shorter 3-D diagonal for better quality.
    faces_list: list = []
    for j in range(n_y - 1):
        for i in range(n_x - 1):
            v00 = j * n_x + i
            v10 = j * n_x + (i + 1)
            v01 = (j + 1) * n_x + i
            v11 = (j + 1) * n_x + (i + 1)
            d1 = float(np.linalg.norm(vertices[v00] - vertices[v11]))
            d2 = float(np.linalg.norm(vertices[v10] - vertices[v01]))
            if d1 <= d2:
                faces_list.append([v00, v10, v11])
                faces_list.append([v00, v11, v01])
            else:
                faces_list.append([v00, v10, v01])
                faces_list.append([v10, v11, v01])

    faces = np.array(faces_list, dtype=np.int32)
    return vertices, faces


def insert_gaussians_into_grid_mesh(
    grid_vertices: np.ndarray,
    grid_faces: np.ndarray,
    gaussian_3d: np.ndarray,
    surface_type: str,
    *,
    max_flip_passes: int = 20,
    local_refinement: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert Gaussian vertices into an existing grid mesh.

    Two modes:

    * **Global Delaunay** (``local_refinement=False``, default):
      Merge grid and Gaussian vertices and run a single 2-D Delaunay
      triangulation on the ``(x, y)`` projection.

    * **Local per-triangle refinement** (``local_refinement=True``):
      Keep the grid topology intact.  For each grid triangle, find
      which Gaussians fall inside it and build a local Delaunay
      sub-triangulation of just those points + the 3 corner vertices.
      Triangles with no interior Gaussians are kept unchanged.  This
      avoids the topological artefacts of global Delaunay (long skinny
      triangles bridging distant regions) and is the recommended mode
      for grid meshes.

    Parameters
    ----------
    grid_vertices : ndarray ``(V, 3)``
    grid_faces : ndarray ``(F, 3)``
    gaussian_3d : ndarray ``(G, 3)``
    surface_type : str
    max_flip_passes : int
        Unused (kept for API compat).
    local_refinement : bool
        If ``True``, use per-triangle local insertion instead of global
        Delaunay.  Default ``False`` for backward compatibility.
    verbose : bool

    Returns
    -------
    vertices : ndarray ``(V + G, 3)``
    faces : ndarray ``(F', 3)``
    gaussian_vertex_indices : ndarray ``(G,)``
    """
    n_grid = len(grid_vertices)
    n_gauss = len(gaussian_3d)

    if n_gauss == 0:
        return (
            grid_vertices.copy(),
            grid_faces.copy(),
            np.array([], dtype=np.int32),
        )

    # Merge vertex arrays:  grid first, then Gaussians
    vertices = np.vstack([grid_vertices, gaussian_3d]).astype(np.float64)

    # Reproject z for safety
    vertices[:, 2] = evaluate_polynomial(
        vertices[:, 0], vertices[:, 1], surface_type,
    )

    # ── Snap Gaussians that coincide with existing grid vertices ─────
    # If a projected Gaussian is within 1e-6 of a grid vertex in (x,y),
    # replace the grid vertex with the Gaussian's projected position and
    # remove the Gaussian vertex (remap its index to that grid vertex).
    # Edge case: if two Gaussians compete for the same grid vertex,
    # only the closest one wins; the other stays as a separate vertex.
    grid_kd_2d = KDTree(vertices[:n_grid, :2])
    snap_dist, snap_idx = grid_kd_2d.query(vertices[n_grid:, :2], k=1)
    snap_tol = 1e-6
    snapped_to = np.where(snap_dist < snap_tol, snap_idx, -1)  # -1 = not snapped

    # Resolve conflicts: when multiple Gaussians snap to the same grid
    # vertex, keep only the closest one; un-snap the rest.
    if (snapped_to >= 0).any():
        grid_best_gi: dict[int, int] = {}   # grid_idx → best gaussian_idx
        grid_best_d: dict[int, float] = {}  # grid_idx → best distance
        for gi in range(n_gauss):
            gv = int(snapped_to[gi])
            if gv < 0:
                continue
            d = float(snap_dist[gi])
            if gv not in grid_best_gi or d < grid_best_d[gv]:
                # Un-snap previous winner (if any)
                if gv in grid_best_gi:
                    snapped_to[grid_best_gi[gv]] = -1
                grid_best_gi[gv] = gi
                grid_best_d[gv] = d
            else:
                # This Gaussian loses — un-snap it
                snapped_to[gi] = -1

    if (snapped_to >= 0).any():
        # Replace grid vertex positions with the winning Gaussian positions
        for gi in range(n_gauss):
            if snapped_to[gi] >= 0:
                vertices[int(snapped_to[gi])] = vertices[n_grid + gi]

        # Build remapped gaussian_vertex_indices: snapped ones point to
        # the (now-updated) grid vertex; unsnapped keep new indices.
        new_verts_list = [vertices[:n_grid]]
        new_gauss_indices = np.full(n_gauss, -1, dtype=np.int32)
        next_idx = n_grid
        for gi in range(n_gauss):
            if snapped_to[gi] >= 0:
                new_gauss_indices[gi] = int(snapped_to[gi])
            else:
                new_gauss_indices[gi] = next_idx
                next_idx += 1
                new_verts_list.append(vertices[n_grid + gi : n_grid + gi + 1])
        vertices = np.vstack(new_verts_list)
        gaussian_vertex_indices = new_gauss_indices
    else:
        gaussian_vertex_indices = np.arange(n_grid, n_grid + n_gauss, dtype=np.int32)

    if local_refinement:
        faces_out = _insert_gaussians_local(
            vertices, n_grid, grid_faces, gaussian_vertex_indices,
            verbose=verbose,
        )
        return vertices, faces_out, gaussian_vertex_indices

    # ── Global 2-D Delaunay on all vertices ──────────────────────────
    import time as _time
    t0 = _time.time()
    xy = vertices[:, :2].copy()
    tri = Delaunay(xy)
    faces_arr = tri.simplices.astype(np.int32)
    dt_delaunay = _time.time() - t0

    if verbose:
        print(f"          Delaunay: {len(faces_arr)} faces from "
              f"{len(vertices)} verts ({dt_delaunay:.1f}s)")

    # ── Measure bad-triangle percentage ──────────────────────────────
    ar, min_angle_deg, longest_edge = _triangle_quality(xy, faces_arr)
    bad_ar = ar > 2.0
    bad_angle = min_angle_deg < 20.0
    bad = bad_ar | bad_angle
    pct_bad = 100.0 * bad.sum() / len(faces_arr)
    if verbose:
        print(f"          bad triangles: {bad.sum()}/{len(faces_arr)} "
              f"({pct_bad:.1f}%)")
        print(f"          AR: median={np.median(ar):.3f}, "
              f"p95={np.percentile(ar, 95):.3f}, "
              f"worst={ar.max():.3f}")
        print(f"          min angle: min={min_angle_deg.min():.1f}°, "
              f"median={np.median(min_angle_deg):.1f}°")

    faces_out = faces_arr.astype(np.int32)
    return vertices, faces_out, gaussian_vertex_indices


def _insert_gaussians_local(
    vertices: np.ndarray,
    n_grid: int,
    grid_faces: np.ndarray,
    gaussian_vertex_indices: np.ndarray,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Per-triangle local Gaussian insertion (internal helper).

    For each grid face, locates which Gaussians fall inside it and
    builds a local Delaunay sub-triangulation.  Grid triangles with
    no interior Gaussians are kept unchanged.

    Uses O(1) structured-grid point location when the mesh comes from
    ``build_surface_grid`` (searchsorted on grid coords + vectorised
    barycentric test), falling back to matplotlib ``TrapezoidMapTriFinder``
    for unstructured grids.

    Parameters
    ----------
    vertices : ndarray ``(V_total, 3)``
        Combined vertex array (grid first, then Gaussians).
    n_grid : int
        Number of grid vertices (first *n_grid* rows are grid verts).
    grid_faces : ndarray ``(F, 3)``
        Original grid triangle connectivity.
    gaussian_vertex_indices : ndarray ``(G,)``
        Global vertex indices of Gaussian vertices in *vertices*.
    verbose : bool

    Returns
    -------
    faces : ndarray ``(F', 3)`` int32
    """
    import time as _time

    t0 = _time.time()
    n_gauss = len(gaussian_vertex_indices)
    gauss_xy = vertices[gaussian_vertex_indices, :2]

    # ── Detect structured grid and use O(1) lookup ───────────────────
    n_outside = 0
    use_structured = False
    grid_y = vertices[:n_grid, 1]
    if n_grid >= 4:
        y0 = grid_y[0]
        tol = 1e-10 * max(abs(float(grid_y.max() - grid_y.min())), 1e-15)
        diffs = np.abs(grid_y[1:min(n_grid, 100_000)] - y0)
        n_x = int(np.argmax(diffs > tol)) + 1
        if n_x >= 2:
            n_y = n_grid // n_x
            expected_faces = 2 * (n_x - 1) * (n_y - 1)
            use_structured = (
                n_x * n_y == n_grid
                and n_y >= 2
                and len(grid_faces) == expected_faces
            )

    if use_structured:
        x_coords = vertices[:n_x, 0]          # first row x values
        y_coords = vertices[::n_x, 1][:n_y]   # every n_x-th y value

        # searchsorted → grid cell (ix, iy) for each Gaussian
        ix = np.searchsorted(x_coords, gauss_xy[:, 0]) - 1
        iy = np.searchsorted(y_coords, gauss_xy[:, 1]) - 1
        ix = np.clip(ix, 0, n_x - 2)
        iy = np.clip(iy, 0, n_y - 2)

        # Cell → face pair: faces at 2*cell and 2*cell+1
        cell_idx = iy * (n_x - 1) + ix
        face_base = 2 * cell_idx  # index of first face in cell

        # Vectorised barycentric test to pick which of the 2 faces
        f0_a = grid_faces[face_base, 0]
        f0_b = grid_faces[face_base, 1]
        f0_c = grid_faces[face_base, 2]

        ax = vertices[f0_a, 0]; ay = vertices[f0_a, 1]
        bx = vertices[f0_b, 0]; by = vertices[f0_b, 1]
        cx = vertices[f0_c, 0]; cy = vertices[f0_c, 1]
        gx = gauss_xy[:, 0]; gy = gauss_xy[:, 1]

        denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        u = ((by - cy) * (gx - cx) + (cx - bx) * (gy - cy)) / denom
        v = ((cy - ay) * (gx - cx) + (ax - cx) * (gy - cy)) / denom

        bary_eps = -1e-8
        in_first = (u >= bary_eps) & (v >= bary_eps) & (u + v <= 1.0 - bary_eps)
        gauss_face_ids = np.where(in_first, face_base, face_base + 1).astype(
            np.int64,
        )

        if verbose:
            t_loc = _time.time() - t0
            print(f"          Structured grid {n_x}×{n_y}: point location "
                  f"for {n_gauss} Gaussians in {t_loc:.3f}s")
    else:
        # ── Fallback: matplotlib trifinder (unstructured grids) ──────
        from matplotlib.tri import Triangulation

        grid_tri = Triangulation(
            vertices[:n_grid, 0], vertices[:n_grid, 1], grid_faces,
        )
        finder = grid_tri.get_trifinder()
        gauss_face_ids = finder(gauss_xy[:, 0], gauss_xy[:, 1]).astype(np.int64)

        outside_mask = gauss_face_ids == -1
        n_outside = int(outside_mask.sum())
        if n_outside > 0:
            grid_kd = KDTree(vertices[:n_grid, :2])
            _, nearest_grid_v = grid_kd.query(gauss_xy[outside_mask])
            vert_to_face = np.full(n_grid, -1, dtype=np.int64)
            for col in range(3):
                mask = vert_to_face[grid_faces[:, col]] == -1
                vert_to_face[grid_faces[:, col][mask]] = np.where(mask)[0]
            gauss_face_ids[outside_mask] = vert_to_face[nearest_grid_v]
            if verbose:
                print(f"          {n_outside} Gaussians outside grid → "
                      f"snapped to nearest face")

    # ── Group Gaussians by face ──────────────────────────────────────
    sort_order = np.argsort(gauss_face_ids)
    sorted_face_ids = gauss_face_ids[sort_order]
    sorted_gauss_global = gaussian_vertex_indices[sort_order]

    change_idx = np.where(np.diff(sorted_face_ids) != 0)[0] + 1
    group_starts = np.concatenate([[0], change_idx])
    group_ends = np.concatenate([change_idx, [n_gauss]])
    group_face_ids = sorted_face_ids[group_starts]

    faces_with_gauss = set(group_face_ids.tolist())
    n_faces = len(grid_faces)

    # ── Faces with 0 Gaussians: keep unchanged ──────────────────────
    no_gauss_mask = np.ones(n_faces, dtype=bool)
    no_gauss_mask[list(faces_with_gauss)] = False
    out_parts: list = [grid_faces[no_gauss_mask]]

    n_kept = int(no_gauss_mask.sum())
    n_fan = 0
    n_delaunay = 0
    n_fan_faces = 0
    n_delaunay_faces = 0

    # ── Separate 1-Gaussian (vectorised fan) vs 2+ (local Delaunay) ─
    group_sizes = group_ends - group_starts
    one_mask = group_sizes == 1
    one_fi = group_face_ids[one_mask].astype(np.int64)
    one_gi = sorted_gauss_global[group_starts[one_mask]].astype(np.int64)

    multi_idx = np.where(~one_mask)[0]

    # ── Vectorised fan for 1-Gaussian faces ──────────────────────────
    if len(one_fi) > 0:
        A = grid_faces[one_fi, 0].astype(np.int64)
        B = grid_faces[one_fi, 1].astype(np.int64)
        C = grid_faces[one_fi, 2].astype(np.int64)
        P = one_gi

        # If Gaussian was snapped to a corner vertex, P==A/B/C and the fan
        # produces degenerate triangles.  For those faces just keep the
        # original triangle (P is already a corner, so the grid mesh is fine).
        snapped_mask = (P == A) | (P == B) | (P == C)
        if snapped_mask.any():
            # Keep original face for snapped cases
            kept_fan = np.stack([A[snapped_mask], B[snapped_mask], C[snapped_mask]], axis=1)
            out_parts.append(kept_fan)
            # Only fan-split for un-snapped cases
            keep = ~snapped_mask
            A, B, C, P, one_fi = A[keep], B[keep], C[keep], P[keep], one_fi[keep]

        if len(one_fi) > 0:
            fan = np.empty((3 * len(one_fi), 3), dtype=np.int64)
            fan[0::3, 0] = P; fan[0::3, 1] = A; fan[0::3, 2] = B
            fan[1::3, 0] = P; fan[1::3, 1] = B; fan[1::3, 2] = C
            fan[2::3, 0] = P; fan[2::3, 1] = C; fan[2::3, 2] = A
            out_parts.append(fan)
        n_fan = len(one_fi)
        n_fan_faces = n_fan * 3

    # ── Local Delaunay for 2+-Gaussian faces ─────────────────────────
    if len(multi_idx) > 0:
        # Pre-allocate list for multi-Gaussian faces
        multi_faces_list: list = []
        for mi in multi_idx:
            fi = int(group_face_ids[mi])
            gi_slice = sorted_gauss_global[group_starts[mi]:group_ends[mi]]
            corners = grid_faces[fi].astype(np.int64)
            local_global = np.concatenate([corners, gi_slice.astype(np.int64)])
            local_xy = vertices[local_global, :2]
            local_tri = Delaunay(local_xy)
            multi_faces_list.append(local_global[local_tri.simplices])
        combined = np.vstack(multi_faces_list)
        out_parts.append(combined)
        n_delaunay = len(multi_idx)
        n_delaunay_faces = len(combined)

    faces_out = np.vstack(out_parts).astype(np.int32)
    dt = _time.time() - t0

    if verbose:
        print(f"          Local insertion: {len(faces_out)} faces from "
              f"{len(vertices)} verts ({dt:.1f}s)")
        print(f"            {n_kept} grid faces unchanged, "
              f"{n_fan} fan (1-Gaussian, {n_fan_faces} faces), "
              f"{n_delaunay} local Delaunay (2+, {n_delaunay_faces} faces)")
        if n_outside > 0:
            print(f"            {n_outside} Gaussians outside grid snapped")
        # Quality report
        ar, min_angle_deg, longest = _triangle_quality(
            vertices[:, :2], faces_out,
        )
        bad = (ar > 2.0) | (min_angle_deg < 20.0)
        print(f"          bad triangles: {bad.sum()}/{len(faces_out)} "
              f"({100.0 * bad.sum() / len(faces_out):.1f}%)")
        print(f"          AR: median={np.median(ar):.3f}, "
              f"p95={np.percentile(ar, 95):.3f}, "
              f"worst={ar.max():.3f}")
        print(f"          min angle: min={min_angle_deg.min():.1f}°, "
              f"median={np.median(min_angle_deg):.1f}°")

    return faces_out


# ── Vectorised face splitting with edge midpoints ──────────────────────────


def _split_faces_with_midpoints(
    xy: np.ndarray,
    faces: np.ndarray,
    va_idx: np.ndarray,
    vb_idx: np.ndarray,
    mid_indices: np.ndarray,
) -> np.ndarray:
    """Replace faces with sub-triangles at edge midpoints — no flips needed.

    For each face, determines which of its 3 edges carry a midpoint and
    produces deterministic sub-triangulations:

    * 0 midpoints → keep unchanged
    * 1 midpoint  → 2 sub-faces
    * 2 midpoints → 3 sub-faces (Delaunay-optimal inner diagonal via
      vectorised in-circle test)
    * 3 midpoints → 4 sub-faces (standard midpoint subdivision)

    Fully vectorised — O(F) with no Python per-face loops or flip passes.

    Parameters
    ----------
    xy : (V, 2) float64 — all vertex positions including new midpoints
    faces : (F, 3) int — current face array
    va_idx, vb_idx : (S,) int — endpoint vertex indices per split edge
    mid_indices : (S,) int — midpoint vertex index per split edge

    Returns
    -------
    new_faces : (F', 3) int32
    """
    n_splits = len(va_idx)
    if n_splits == 0:
        return faces.copy()

    n_f = len(faces)
    n_v = len(xy)

    # ── Edge → midpoint lookup (sorted + searchsorted) ────────────────
    e_min = np.minimum(va_idx, vb_idx).astype(np.int64)
    e_max = np.maximum(va_idx, vb_idx).astype(np.int64)
    split_keys = e_min * n_v + e_max

    sort_ord = np.argsort(split_keys)
    sk_sorted = split_keys[sort_ord]
    mi_sorted = np.asarray(mid_indices, dtype=np.int64)[sort_ord]

    def _lookup(emin_arr, emax_arr):
        keys = emin_arr.astype(np.int64) * n_v + emax_arr.astype(np.int64)
        idx = np.searchsorted(sk_sorted, keys)
        idx = np.clip(idx, 0, max(len(sk_sorted) - 1, 0))
        found = sk_sorted[idx] == keys
        return np.where(found, mi_sorted[idx], np.int64(-1))

    f0 = faces[:, 0].astype(np.int64)
    f1 = faces[:, 1].astype(np.int64)
    f2 = faces[:, 2].astype(np.int64)

    m01 = _lookup(np.minimum(f0, f1), np.maximum(f0, f1))
    m12 = _lookup(np.minimum(f1, f2), np.maximum(f1, f2))
    m20 = _lookup(np.minimum(f2, f0), np.maximum(f2, f0))

    h01 = m01 >= 0
    h12 = m12 >= 0
    h20 = m20 >= 0
    n_mids = h01.astype(np.int32) + h12.astype(np.int32) + h20.astype(np.int32)

    out_parts: list = []

    # ── Group 0: no midpoints — keep unchanged ────────────────────────
    g0_mask = n_mids == 0
    if g0_mask.any():
        out_parts.append(faces[g0_mask])

    # ── Group 1: exactly 1 midpoint → 2 sub-faces ─────────────────────
    # For split edge A→B opposite C:  (A, M, C) + (M, B, C)
    for e_has, vA, vB, vC, mAB in [
        (h01 & ~h12 & ~h20, f0, f1, f2, m01),   # split edge 0→1
        (~h01 & h12 & ~h20, f1, f2, f0, m12),   # split edge 1→2
        (~h01 & ~h12 & h20, f2, f0, f1, m20),   # split edge 2→0
    ]:
        sel = np.where(e_has)[0]
        if len(sel) == 0:
            continue
        A = vA[sel]; B = vB[sel]; C = vC[sel]; M = mAB[sel]
        out_parts.append(np.column_stack([A, M, C]))
        out_parts.append(np.column_stack([M, B, C]))

    # ── Group 3: all 3 midpoints → 4 sub-faces ────────────────────────
    g3 = np.where(n_mids == 3)[0]
    if len(g3) > 0:
        A = f0[g3]; B = f1[g3]; C = f2[g3]
        mAB = m01[g3]; mBC = m12[g3]; mCA = m20[g3]
        out_parts.append(np.column_stack([A, mAB, mCA]))
        out_parts.append(np.column_stack([B, mBC, mAB]))
        out_parts.append(np.column_stack([C, mCA, mBC]))
        out_parts.append(np.column_stack([mAB, mBC, mCA]))

    # ── Group 2: exactly 2 midpoints → 3 sub-faces ────────────────────
    # Rotate so the two split edges share vertex B:
    #   split edges A→B (midpoint mAB) and B→C (midpoint mBC),
    #   unsplit edge C→A.
    # Common ear triangle: (mAB, B, mBC).
    # Remaining quad (A, mAB, mBC, C) — pick Delaunay diagonal via
    # vectorised in-circle test.
    for mask_sel, vA, vB, vC, mAB_arr, mBC_arr in [
        (h01 & h12 & ~h20, f0, f1, f2, m01, m12),   # shared v1
        (~h01 & h12 & h20, f1, f2, f0, m12, m20),   # shared v2
        (h01 & ~h12 & h20, f2, f0, f1, m20, m01),   # shared v0
    ]:
        g2 = np.where(mask_sel)[0]
        if len(g2) == 0:
            continue
        A = vA[g2]; B = vB[g2]; C = vC[g2]
        mAB = mAB_arr[g2]; mBC = mBC_arr[g2]

        # Ear at B — always emitted
        out_parts.append(np.column_stack([mAB, B, mBC]))

        # In-circle test: is A inside circumcircle(mAB, mBC, C)?
        # If yes → diagonal A↔mBC ;  otherwise → diagonal mAB↔C
        ax = xy[mAB, 0] - xy[A, 0]
        ay = xy[mAB, 1] - xy[A, 1]
        bx = xy[mBC, 0] - xy[A, 0]
        by = xy[mBC, 1] - xy[A, 1]
        cx = xy[C, 0] - xy[A, 0]
        cy = xy[C, 1] - xy[A, 1]

        a_sq = ax * ax + ay * ay
        b_sq = bx * bx + by * by
        c_sq = cx * cx + cy * cy
        det = (a_sq * (bx * cy - cx * by)
             - b_sq * (ax * cy - cx * ay)
             + c_sq * (ax * by - bx * ay))

        # Adjust sign for CW-oriented (mAB, mBC, C)
        orient = ((xy[mBC, 0] - xy[mAB, 0]) * (xy[C, 1] - xy[mAB, 1])
                - (xy[mBC, 1] - xy[mAB, 1]) * (xy[C, 0] - xy[mAB, 0]))
        det = np.where(orient < 0, -det, det)

        use_b = det > 0  # A inside → flip to diagonal A↔mBC

        # Diagonal mAB↔C  (option A)
        a_idx = np.where(~use_b)[0]
        if len(a_idx) > 0:
            out_parts.append(np.column_stack([
                A[a_idx], mAB[a_idx], C[a_idx]]))
            out_parts.append(np.column_stack([
                mAB[a_idx], mBC[a_idx], C[a_idx]]))

        # Diagonal A↔mBC  (option B)
        b_idx = np.where(use_b)[0]
        if len(b_idx) > 0:
            out_parts.append(np.column_stack([
                A[b_idx], mAB[b_idx], mBC[b_idx]]))
            out_parts.append(np.column_stack([
                A[b_idx], mBC[b_idx], C[b_idx]]))

    if not out_parts:
        return np.empty((0, 3), dtype=np.int32)
    return np.vstack(out_parts).astype(np.int32)


def _lawson_flip_local(
    xy: np.ndarray,
    faces: np.ndarray,
    suspect_face_indices: np.ndarray,
    *,
    n_grid_verts: int = 0,
    verbose: bool = False,
) -> int:
    """Local Lawson flipping using an edge queue.

    Only processes edges of *suspect_face_indices* (i.e. newly-created
    faces from Gaussian insertion) plus edges that cascade from flips.
    Grid-only edges (both endpoints < *n_grid_verts*) are constrained
    and never flipped, preserving the original grid connectivity.

    Parameters
    ----------
    xy : (V, 2) float64
    faces : (F, 3) int — mutable, modified in-place
    suspect_face_indices : array-like — indices of new/modified faces
    n_grid_verts : int — vertices with index < this are grid verts
    verbose : bool

    Returns
    -------
    n_flips : int
    """
    n_f = len(faces)
    n_v = int(xy.shape[0])
    if n_f < 2:
        return 0

    fa = faces  # modify in-place

    # ── Build edge adjacency (vectorised) ────────────────────────────
    fi_rep = np.repeat(np.arange(n_f, dtype=np.int64), 3)
    va_all = fa[:, [0, 1, 2]].ravel().astype(np.int64)
    vb_all = fa[:, [1, 2, 0]].ravel().astype(np.int64)

    e_min = np.minimum(va_all, vb_all)
    e_max = np.maximum(va_all, vb_all)
    edge_keys_all = e_min * n_v + e_max

    sort_order = np.argsort(edge_keys_all)
    sorted_keys = edge_keys_all[sort_order]

    match = sorted_keys[:-1] == sorted_keys[1:]
    first_of_pair = match.copy()
    if len(first_of_pair) > 1:
        first_of_pair[1:] &= ~match[:-1]
    pair_starts = np.where(first_of_pair)[0]

    if len(pair_starts) == 0:
        return 0

    idx1 = sort_order[pair_starts]
    idx2 = sort_order[pair_starts + 1]

    keys_np = edge_keys_all[idx1]
    f1_np = fi_rep[idx1]
    f2_np = fi_rep[idx2]

    # Dict: edge_key (int) -> [fi1, fi2]
    edge_adj: dict = {
        int(k): [int(a), int(b)]
        for k, a, b in zip(keys_np.tolist(), f1_np.tolist(), f2_np.tolist())
    }

    # ── Seed queue with edges of suspect faces (vectorised) ──────────
    suspect_fi = np.asarray(suspect_face_indices, dtype=np.int64)
    if len(suspect_fi) == 0:
        return 0

    sa = fa[suspect_fi, 0].astype(np.int64)
    sb = fa[suspect_fi, 1].astype(np.int64)
    sc = fa[suspect_fi, 2].astype(np.int64)
    all_keys = np.concatenate([
        np.minimum(sa, sb) * n_v + np.maximum(sa, sb),
        np.minimum(sb, sc) * n_v + np.maximum(sb, sc),
        np.minimum(sc, sa) * n_v + np.maximum(sc, sa),
    ])
    suspect_keys = np.unique(all_keys)

    queue = set(suspect_keys.tolist())

    # ── Process edge queue ───────────────────────────────────────────
    n_flips = 0
    xy_arr = np.asarray(xy, dtype=np.float64)

    while queue:
        ekey = queue.pop()
        if ekey not in edge_adj:
            continue

        fi1, fi2 = edge_adj[ekey]
        ea = ekey // n_v
        eb = ekey % n_v

        # Read face vertices
        f1v = (int(fa[fi1, 0]), int(fa[fi1, 1]), int(fa[fi1, 2]))
        f2v = (int(fa[fi2, 0]), int(fa[fi2, 1]), int(fa[fi2, 2]))

        # Verify edge still present (guard against stale adjacency)
        if ea not in f1v or eb not in f1v or ea not in f2v or eb not in f2v:
            del edge_adj[ekey]
            continue

        opp1 = next(v for v in f1v if v != ea and v != eb)
        opp2 = next(v for v in f2v if v != ea and v != eb)

        # ── Convexity of quad (ea, opp1, eb, opp2) ──────────────────
        pax, pay = xy_arr[ea, 0], xy_arr[ea, 1]
        pbx, pby = xy_arr[eb, 0], xy_arr[eb, 1]
        pcx, pcy = xy_arr[opp1, 0], xy_arr[opp1, 1]
        pdx, pdy = xy_arr[opp2, 0], xy_arr[opp2, 1]

        s0 = (pcx - pax) * (pby - pay) - (pcy - pay) * (pbx - pax)
        s1 = (pbx - pcx) * (pdy - pcy) - (pby - pcy) * (pdx - pcx)
        s2 = (pdx - pbx) * (pay - pby) - (pdy - pby) * (pax - pbx)
        s3 = (pax - pdx) * (pcy - pdy) - (pay - pdy) * (pcx - pdx)

        if not ((s0 > 0 and s1 > 0 and s2 > 0 and s3 > 0) or
                (s0 < 0 and s1 < 0 and s2 < 0 and s3 < 0)):
            continue  # not convex

        # ── In-circle determinant ────────────────────────────────────
        ax_d = pax - pdx;  ay_d = pay - pdy
        bx_d = pbx - pdx;  by_d = pby - pdy
        cx_d = pcx - pdx;  cy_d = pcy - pdy

        det = (
            (ax_d * ax_d + ay_d * ay_d) * (bx_d * cy_d - cx_d * by_d)
          - (bx_d * bx_d + by_d * by_d) * (ax_d * cy_d - cx_d * ay_d)
          + (cx_d * cx_d + cy_d * cy_d) * (ax_d * by_d - bx_d * ay_d)
        )
        orient = (pbx - pax) * (pcy - pay) - (pby - pay) * (pcx - pax)
        if orient < 0:
            det = -det

        scale = max(
            ax_d * ax_d + ay_d * ay_d,
            bx_d * bx_d + by_d * by_d,
            cx_d * cx_d + cy_d * cy_d,
        )
        if det <= 1e-12 * max(scale, 1e-30):
            continue  # already Delaunay

        # ── Flip: (ea, eb) → (opp1, opp2) ───────────────────────────
        fa[fi1, 0] = opp1;  fa[fi1, 1] = ea;    fa[fi1, 2] = opp2
        fa[fi2, 0] = opp1;  fa[fi2, 1] = opp2;  fa[fi2, 2] = eb
        n_flips += 1

        # ── Update adjacency ────────────────────────────────────────
        del edge_adj[ekey]

        # New diagonal
        new_key = min(opp1, opp2) * n_v + max(opp1, opp2)
        edge_adj[new_key] = [fi1, fi2]

        # Edge (eb, opp1): moved from fi1 → fi2
        k_b_o1 = min(eb, opp1) * n_v + max(eb, opp1)
        if k_b_o1 in edge_adj:
            entry = edge_adj[k_b_o1]
            if entry[0] == fi1:
                entry[0] = fi2
            elif entry[1] == fi1:
                entry[1] = fi2

        # Edge (ea, opp2): moved from fi2 → fi1
        k_a_o2 = min(ea, opp2) * n_v + max(ea, opp2)
        if k_a_o2 in edge_adj:
            entry = edge_adj[k_a_o2]
            if entry[0] == fi2:
                entry[0] = fi1
            elif entry[1] == fi2:
                entry[1] = fi1

        # Enqueue 4 boundary edges of the quad
        k_o1_a = min(opp1, ea) * n_v + max(opp1, ea)
        k_o2_b = min(opp2, eb) * n_v + max(opp2, eb)
        for k in (k_b_o1, k_a_o2, k_o1_a, k_o2_b):
            if k in edge_adj:
                queue.add(k)

    if verbose:
        print(f"          local Lawson: {n_flips} flips")

    return n_flips


def _lawson_flip_pass(
    xy: np.ndarray,
    faces: np.ndarray,
) -> int:
    """One pass of Lawson flipping — vectorised in-circle test.

    Modifies *faces* **in-place** (must be a writable ``int64`` or
    ``int32`` ndarray of shape ``(F, 3)``).
    Returns the number of flips performed.
    Within a single pass each face participates in at most one flip.
    """
    n_f = len(faces)
    if n_f < 2:
        return 0

    fa = np.asarray(faces)

    # Build half-edge arrays (3 edges per face, vectorised)
    idx012 = np.array([0, 1, 2])
    idx120 = np.array([1, 2, 0])
    v_a = fa[:, idx012].ravel()          # (3*n_f,)
    v_b = fa[:, idx120].ravel()          # (3*n_f,)
    e_fi = np.repeat(np.arange(n_f, dtype=np.int64), 3)
    e_lk = np.tile(np.arange(3, dtype=np.int64), n_f)

    # Sort each edge pair for keying
    e_min = np.minimum(v_a, v_b)
    e_max = np.maximum(v_a, v_b)

    # Cantor pairing for unique edge keys (faster than structured array)
    n_v = int(xy.shape[0])
    edge_keys = e_min.astype(np.int64) * n_v + e_max.astype(np.int64)

    # Find interior edges (appear exactly twice)
    sort_order = np.argsort(edge_keys)
    sorted_keys = edge_keys[sort_order]

    # Consecutive equal pairs → interior edges
    match = sorted_keys[:-1] == sorted_keys[1:]
    # Exclude triplets (non-manifold): an interior edge has exactly 2
    # so we want match[i] == True AND (i==0 or match[i-1]==False)
    # AND (i+1 >= len or match[i+1]==False)
    first_of_pair = match.copy()
    if len(first_of_pair) > 1:
        first_of_pair[1:] &= ~match[:-1]

    pair_starts = np.where(first_of_pair)[0]
    if len(pair_starts) == 0:
        return 0

    idx1 = sort_order[pair_starts]
    idx2 = sort_order[pair_starts + 1]

    fi1 = e_fi[idx1]
    fi2 = e_fi[idx2]
    lk1 = e_lk[idx1]
    lk2 = e_lk[idx2]

    opp1 = fa[fi1, (lk1 + 2) % 3]
    opp2 = fa[fi2, (lk2 + 2) % 3]
    ea = e_min[idx1]
    eb = e_max[idx1]

    # ── Vectorised convexity + in-circle test ─────────────────────────
    pa = xy[ea]
    pb = xy[eb]
    pc = xy[opp1]
    pd = xy[opp2]

    # Cross products for convexity of quadrilateral (a, c, b, d)
    def _vcross(o, a, b):
        return (a[:, 0] - o[:, 0]) * (b[:, 1] - o[:, 1]) - \
               (a[:, 1] - o[:, 1]) * (b[:, 0] - o[:, 0])

    s0 = _vcross(pa, pc, pb)
    s1 = _vcross(pc, pb, pd)
    s2 = _vcross(pb, pd, pa)
    s3 = _vcross(pd, pa, pc)
    convex = (
        ((s0 > 0) & (s1 > 0) & (s2 > 0) & (s3 > 0)) |
        ((s0 < 0) & (s1 < 0) & (s2 < 0) & (s3 < 0))
    )

    # In-circle determinant
    ax_d = pa[:, 0] - pd[:, 0]
    ay_d = pa[:, 1] - pd[:, 1]
    bx_d = pb[:, 0] - pd[:, 0]
    by_d = pb[:, 1] - pd[:, 1]
    cx_d = pc[:, 0] - pd[:, 0]
    cy_d = pc[:, 1] - pd[:, 1]

    det = (
        (ax_d ** 2 + ay_d ** 2) * (bx_d * cy_d - cx_d * by_d)
        - (bx_d ** 2 + by_d ** 2) * (ax_d * cy_d - cx_d * ay_d)
        + (cx_d ** 2 + cy_d ** 2) * (ax_d * by_d - bx_d * ay_d)
    )
    orient = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - \
             (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
    det[orient < 0] *= -1

    # Use a relative tolerance to avoid flipping near-Delaunay edges
    # back and forth (oscillation due to floating-point noise).
    scale = np.maximum(
        ax_d ** 2 + ay_d ** 2,
        np.maximum(bx_d ** 2 + by_d ** 2, cx_d ** 2 + cy_d ** 2),
    )
    eps = 1e-12 * np.maximum(scale, 1e-30)
    should_flip = convex & (det > eps)

    if not should_flip.any():
        return 0

    # ── Greedy independent set (vectorised) ───────────────────────────
    flip_idx = np.where(should_flip)[0]
    f1 = fi1[flip_idx]
    f2 = fi2[flip_idx]
    o1 = opp1[flip_idx]
    o2 = opp2[flip_idx]
    a_v = ea[flip_idx]
    b_v = eb[flip_idx]

    # Each face can participate in at most one flip.
    # Greedy: scan in order, mark used faces.
    used = np.zeros(n_f, dtype=bool)
    keep = np.empty(len(flip_idx), dtype=bool)
    for j in range(len(flip_idx)):
        f1j, f2j = f1[j], f2[j]
        if used[f1j] or used[f2j]:
            keep[j] = False
        else:
            keep[j] = True
            used[f1j] = True
            used[f2j] = True

    sel = np.where(keep)[0]
    n_flips = len(sel)
    if n_flips == 0:
        return 0

    # Apply all selected flips at once (vectorised write)
    ff1 = f1[sel]
    ff2 = f2[sel]
    faces[ff1, 0] = o1[sel]
    faces[ff1, 1] = a_v[sel]
    faces[ff1, 2] = o2[sel]
    faces[ff2, 0] = o1[sel]
    faces[ff2, 1] = o2[sel]
    faces[ff2, 2] = b_v[sel]

    return n_flips


# ── Bad-triangle refinement ─────────────────────────────────────────────────


def _is_boundary_face(faces: np.ndarray) -> np.ndarray:
    """Return a boolean mask: ``True`` for faces with ≥ 1 boundary edge.

    A *boundary* edge is shared by exactly one face (as opposed to an
    interior edge shared by two).  In a Delaunay triangulation of the
    convex hull, boundary faces are exactly those touching the hull
    perimeter.
    """
    n_f = len(faces)
    # Build sorted edge pairs (3 per face, stacked)
    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])  # (3*n_f, 2)

    # Compact edge id via structured dtype → fast unique/count
    dt = np.dtype([("a", np.int64), ("b", np.int64)])
    structured = np.empty(len(all_edges), dtype=dt)
    structured["a"] = all_edges[:, 0]
    structured["b"] = all_edges[:, 1]
    _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
    occ = counts[inv]  # how many faces share each half-edge

    # Face is boundary if any of its 3 edges appears exactly once
    return (occ[:n_f] == 1) | (occ[n_f : 2 * n_f] == 1) | (occ[2 * n_f :] == 1)


def _filter_convex_hull_artifacts(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float,
    surface_type: Optional[str] = None,
    *,
    is_3d: bool = False,
) -> Tuple[np.ndarray, int]:
    """Remove faces that are convex-hull artifacts (boundary + long edge).

    **2-D mode** (``is_3d=False``):  Remove faces that are on the mesh
    boundary *and* have at least one edge longer than *max_edge_length*.
    Interior faces are never removed, preventing holes.

    **3-D mode** (``is_3d=True``):  Remove boundary faces whose normal
    is poorly aligned with the analytical surface normal (reuses the
    same filter as :func:`_extract_surface_faces`).  If
    *surface_type* is ``None``, falls back to the 2-D long-edge rule.

    After removal, V-shaped notches are patched with small fill
    triangles via :func:`_patch_boundary_notches`.

    Returns ``(filtered_faces, n_removed)``.
    """
    if max_edge_length <= 0 or len(faces) == 0:
        return faces, 0

    # ── 3-D normal-alignment filter ───────────────────────────────────
    if is_3d and surface_type is not None:
        n_before = len(faces)
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        face_normals = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals = face_normals / np.maximum(norms, 1e-15)

        centroids = (v0 + v1 + v2) / 3.0
        cx, cy = centroids[:, 0], centroids[:, 1]
        eps = 1e-6
        dfdx = (evaluate_polynomial(cx + eps, cy, surface_type)
                - evaluate_polynomial(cx - eps, cy, surface_type)) / (2 * eps)
        dfdy = (evaluate_polynomial(cx, cy + eps, surface_type)
                - evaluate_polynomial(cx, cy - eps, surface_type)) / (2 * eps)
        surf_normals = np.column_stack([-dfdx, -dfdy, np.ones(len(faces))])
        surf_normals /= np.linalg.norm(surf_normals, axis=1, keepdims=True)

        dots = np.sum(face_normals * surf_normals, axis=1)
        aligned = np.abs(dots) > 0.1
        faces = faces[aligned]
        n_removed_3d = n_before - len(faces)

        # Also apply the 2-D boundary + long-edge rule for any remaining
        # artifacts that passed the normal check
        if len(faces) > 0:
            faces_2d, n_extra = _filter_convex_hull_artifacts(
                vertices, faces, max_edge_length, surface_type, is_3d=False,
            )
            return faces_2d, n_removed_3d + n_extra

        return faces, n_removed_3d

    # ── 2-D boundary + long-edge filter ───────────────────────────────
    fv0 = vertices[faces[:, 0]]
    fv1 = vertices[faces[:, 1]]
    fv2 = vertices[faces[:, 2]]
    edge_max = np.maximum(
        np.maximum(
            np.linalg.norm(fv1 - fv0, axis=1),
            np.linalg.norm(fv2 - fv1, axis=1),
        ),
        np.linalg.norm(fv0 - fv2, axis=1),
    )
    has_long = edge_max > max_edge_length
    is_bdry = _is_boundary_face(faces)

    remove = has_long & is_bdry
    n_removed = int(remove.sum())
    if n_removed == 0:
        return faces, 0

    faces = faces[~remove]

    # ── Patch notches: fill V-shaped gaps left by removal ─────────────
    faces = _patch_boundary_notches(vertices, faces, max_edge_length)

    return faces, n_removed


def _filter_boundary_long_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float,
    surface_type: Optional[str] = None,
    *,
    is_3d: bool = False,
) -> Tuple[np.ndarray, int]:
    """Remove convex-hull artifact faces.  Delegates to :func:`_filter_convex_hull_artifacts`.

    This wrapper preserves the original API, forwarding to the renamed
    implementation.
    """
    return _filter_convex_hull_artifacts(
        vertices, faces, max_edge_length, surface_type, is_3d=is_3d,
    )


def _patch_boundary_notches(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float = float("inf"),
) -> np.ndarray:
    """Fill V-shaped notches in the mesh boundary.

    After boundary face removal, some previously-interior vertices become
    boundary, creating small V-shaped indentations.  For each such
    *notch vertex* (boundary vertex **not** on the 2-D convex hull) that
    has exactly two boundary neighbours, add a single triangle to close
    the notch — but only if every edge of that triangle is within
    *max_edge_length*.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
    faces : ndarray ``(F, 3)``
    max_edge_length : float
        Fill triangles whose longest edge exceeds this are skipped.

    Returns
    -------
    faces : ndarray ``(F', 3)``
        Faces with notch-filling triangles appended.
    """
    from collections import defaultdict

    if len(faces) == 0:
        return faces

    # 2-D convex hull — vertices on the hull are "real" boundary
    # Filter out outlier vertices (e.g. from smoothing blow-ups) before
    # computing the convex hull to prevent QHull crashes.
    xy_hull = vertices[:, :2]
    x_range = float(np.ptp(xy_hull[:, 0]))
    y_range = float(np.ptp(xy_hull[:, 1]))
    domain_size = max(x_range, y_range, 1e-6)
    x_med = float(np.median(xy_hull[:, 0]))
    y_med = float(np.median(xy_hull[:, 1]))
    inlier = (
        (np.abs(xy_hull[:, 0] - x_med) < 10 * domain_size) &
        (np.abs(xy_hull[:, 1] - y_med) < 10 * domain_size)
    )
    hull_pts = xy_hull[inlier]
    if len(hull_pts) < 3:
        return faces
    try:
        hull = ConvexHull(hull_pts)
    except Exception:
        return faces  # cannot compute hull — skip notch patching
    # Map hull vertex indices back to original indices
    inlier_idx = np.where(inlier)[0]
    hull_set = set(int(inlier_idx[v]) for v in hull.vertices)

    # Compute boundary edges once
    n_f = len(faces)
    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])
    dt = np.dtype([("a", np.int64), ("b", np.int64)])
    structured = np.empty(len(all_edges), dtype=dt)
    structured["a"] = all_edges[:, 0]
    structured["b"] = all_edges[:, 1]
    _, inv, counts = np.unique(
        structured, return_inverse=True, return_counts=True,
    )
    occ = counts[inv]
    bdry_edges = all_edges[occ == 1]

    if len(bdry_edges) == 0:
        return faces

    # Build adjacency among boundary vertices
    adj: dict = defaultdict(set)
    for a, b in bdry_edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    new_faces: list = []
    for v, nbrs in adj.items():
        if v in hull_set:
            continue  # genuine hull boundary vertex — leave it
        if len(nbrs) != 2:
            continue  # complex junction — skip
        n1, n2 = list(nbrs)
        # Check all three edges of the fill triangle
        d_vn1 = float(np.linalg.norm(vertices[v] - vertices[n1]))
        d_vn2 = float(np.linalg.norm(vertices[v] - vertices[n2]))
        d_n12 = float(np.linalg.norm(vertices[n1] - vertices[n2]))
        if max(d_vn1, d_vn2, d_n12) > max_edge_length:
            continue  # would re-introduce a long edge
        new_faces.append([v, n1, n2])

    if new_faces:
        faces = np.vstack([faces, np.array(new_faces, dtype=np.int32)])

    return faces


def _count_mesh_holes(faces: np.ndarray) -> int:
    """Count holes in a triangle mesh.

    Returns the number of interior boundary loops (0 = no holes).
    The outer boundary (convex hull) is not counted.

    Uses Euler's formula for graphs: ``n_cycles = E - V + C`` where
    *E* = boundary edges, *V* = boundary vertices, *C* = connected
    components of the boundary graph.  One cycle is the outer boundary;
    every extra cycle is a hole.
    """
    from collections import defaultdict, deque

    if len(faces) == 0:
        return 0

    e0 = np.sort(faces[:, [0, 1]], axis=1)
    e1 = np.sort(faces[:, [1, 2]], axis=1)
    e2 = np.sort(faces[:, [2, 0]], axis=1)
    all_edges = np.vstack([e0, e1, e2])

    dt = np.dtype([("a", np.int64), ("b", np.int64)])
    structured = np.empty(len(all_edges), dtype=dt)
    structured["a"] = all_edges[:, 0]
    structured["b"] = all_edges[:, 1]
    _, inv, counts = np.unique(structured, return_inverse=True, return_counts=True)
    occ = counts[inv]
    boundary_mask = occ == 1
    boundary_edges = all_edges[boundary_mask]

    if len(boundary_edges) == 0:
        return 0  # closed mesh

    n_edges = len(boundary_edges)
    boundary_verts = set(boundary_edges[:, 0].tolist()) | set(
        boundary_edges[:, 1].tolist()
    )
    n_verts = len(boundary_verts)

    # Count connected components via BFS
    adj: dict = defaultdict(set)
    for a, b in boundary_edges:
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    visited: set = set()
    n_components = 0
    for node in adj:
        if node not in visited:
            n_components += 1
            queue = deque([node])
            while queue:
                v = queue.popleft()
                if v in visited:
                    continue
                visited.add(v)
                for nb in adj[v]:
                    if nb not in visited:
                        queue.append(nb)

    # Euler: n_cycles = E - V + C.  Outer boundary = 1 cycle.
    n_cycles = n_edges - n_verts + n_components
    return max(0, n_cycles - 1)


def _triangle_quality(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-triangle aspect ratio, minimum angle (degrees), and longest edge length."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    d10 = v1 - v0
    d21 = v2 - v1
    d02 = v0 - v2

    # Squared edge lengths (avoid sqrt for intermediate calcs)
    e0_sq = (d10 * d10).sum(axis=1)
    e1_sq = (d21 * d21).sum(axis=1)
    e2_sq = (d02 * d02).sum(axis=1)

    e0 = np.sqrt(e0_sq)
    e1 = np.sqrt(e1_sq)
    e2 = np.sqrt(e2_sq)

    shortest = np.minimum(np.minimum(e0, e1), e2)
    longest = np.maximum(np.maximum(e0, e1), e2)
    aspect_ratio = longest / np.maximum(shortest, 1e-15)

    # Min interior angle via law of cosines
    cos_A = np.clip((e1_sq + e2_sq - e0_sq) / (2 * e1 * e2 + 1e-30), -1, 1)
    cos_B = np.clip((e0_sq + e2_sq - e1_sq) / (2 * e0 * e2 + 1e-30), -1, 1)
    cos_C = np.clip((e0_sq + e1_sq - e2_sq) / (2 * e0 * e1 + 1e-30), -1, 1)
    # arccos of the max cosine = min angle (arccos is monotonically decreasing)
    max_cos = np.maximum(np.maximum(cos_A, cos_B), cos_C)
    min_angle = np.degrees(np.arccos(max_cos))

    return aspect_ratio, min_angle, longest


# ── Orphan-Gaussian repair ──────────────────────────────────────────────────

def fix_orphaned_gaussians(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_gauss: int,
    max_edge_length: float,
    surface_type: str,
    *,
    max_attempts: int = 5,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure every Gaussian vertex participates in at least one face.

    After the long-edge filter some Gaussian vertices may have no adjacent
    triangle.  This function adds a ring of 6 helper vertices around each
    orphaned Gaussian — spaced to produce **3-D edge lengths** below
    *max_edge_length* — then re-runs Delaunay + long-edge filter.

    The ring radius is calibrated to the local surface metric at each
    Gaussian: a small test step in XY is used to estimate the local
    3-D-per-2-D scale factor, then ``ring_r_2d = max_edge_length * 0.65 /
    scale``.  A bridge chain toward the nearest used vertex ensures the
    ring stays connected to the rest of the mesh.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
        Full vertex array **before** compaction.
    faces : ndarray ``(F, 3)``
        Face array after long-edge filtering.
    n_gauss : int
    max_edge_length : float
        3-D long-edge threshold.
    surface_type : str
    max_attempts : int
    verbose : bool

    Returns
    -------
    vertices : ndarray ``(V', 3)``
    faces : ndarray ``(F', 3)``
    """
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)
    target_r_3d = max_edge_length * 0.65  # desired 3-D ring radius
    eps = max_edge_length * 0.01          # finite-diff step for scale estimate

    def _local_scale(gxy: np.ndarray, gz: float) -> float:
        """Estimate sqrt(1 + (∂z/∂x)² + (∂z/∂y)²) at *gxy*."""
        zx = evaluate_polynomial(
            np.array([gxy[0] + eps]), np.array([gxy[1]]), surface_type,
        )[0]
        zy = evaluate_polynomial(
            np.array([gxy[0]]), np.array([gxy[1] + eps]), surface_type,
        )[0]
        dzdx = (zx - gz) / eps
        dzdy = (zy - gz) / eps
        return float(np.sqrt(1.0 + dzdx ** 2 + dzdy ** 2))

    angles_rad = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 evenly spaced

    for attempt in range(max_attempts):
        used_set = set(faces.ravel().tolist()) if len(faces) > 0 else set()
        orphaned = [i for i in range(n_gauss) if i not in used_set]
        if not orphaned:
            break

        if verbose:
            print(f"          [orphan fix attempt {attempt}] "
                  f"{len(orphaned)} orphaned Gaussian(s) — adding support rings")

        # KDTree of currently-used vertices (may be empty on first attempt)
        used_has_points = len(used_set) > 0
        if used_has_points:
            used_arr = np.array(sorted(used_set), dtype=np.int32)
            used_xy = vertices[used_arr, :2]
            tree = KDTree(used_xy)

        new_verts: list = []

        for gi in orphaned:
            gxy = vertices[gi, :2]
            gz = float(vertices[gi, 2])

            # Calibrate ring radius to 3-D metric at G
            scale = _local_scale(gxy, gz)
            ring_r_2d = target_r_3d / max(scale, 1e-6)

            # ── 1. Ring of 6 helper points ────────────────────────────
            for ang in angles_rad:
                bxy = gxy + ring_r_2d * np.array([np.cos(ang), np.sin(ang)])
                bz = evaluate_polynomial(
                    np.array([bxy[0]]), np.array([bxy[1]]), surface_type,
                )[0]
                new_verts.append([bxy[0], bxy[1], bz])

            # ── 2. Bridge chain toward nearest used vertex (3-D spacing) ─
            if used_has_points:
                _dist2d, nn_local = tree.query(gxy, k=1)
                nxy = used_xy[nn_local]
                nz = float(evaluate_polynomial(
                    np.array([nxy[0]]), np.array([nxy[1]]), surface_type,
                )[0])
                dist_3d = np.linalg.norm(
                    np.array([nxy[0], nxy[1], nz])
                    - np.array([gxy[0], gxy[1], gz])
                )
                if dist_3d > max_edge_length:
                    n_steps = int(np.ceil(dist_3d / target_r_3d))
                    for s in range(1, n_steps):
                        t = s / n_steps
                        bxy = gxy + t * (nxy - gxy)
                        bz = evaluate_polynomial(
                            np.array([bxy[0]]), np.array([bxy[1]]), surface_type,
                        )[0]
                        new_verts.append([bxy[0], bxy[1], bz])

        if not new_verts:
            break

        vertices = np.vstack([vertices, np.array(new_verts, dtype=np.float64)])

        # Re-run Delaunay + boundary-only long-edge filter
        tri = Delaunay(vertices[:, :2])
        faces = tri.simplices.astype(np.int32)
        if max_edge_length > 0:
            faces, _ = _filter_boundary_long_edges(
                vertices, faces, max_edge_length,
            )

    # Final check
    final_used = set(faces.ravel().tolist()) if len(faces) > 0 else set()
    still_orphaned = [i for i in range(n_gauss) if i not in final_used]
    if still_orphaned and verbose:
        print(f"          ⚠ {len(still_orphaned)} Gaussian(s) still orphaned "
              f"after {max_attempts} fix attempts (will be dropped)")

    return vertices, faces


# ── Steiner-point insertion ─────────────────────────────────────────────────

def _circumcenter_2d(
    ax: np.ndarray, ay: np.ndarray,
    bx: np.ndarray, by: np.ndarray,
    cx: np.ndarray, cy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised circumcenter of triangles in the 2-D ``(x, y)`` plane.

    Parameters
    ----------
    ax, ay, bx, by, cx, cy : ndarray ``(N,)``

    Returns
    -------
    ux, uy : ndarray ``(N,)``  — circumcentre coordinates.
    """
    D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    D = np.where(np.abs(D) < 1e-30, 1e-30, D)
    a2 = ax ** 2 + ay ** 2
    b2 = bx ** 2 + by ** 2
    c2 = cx ** 2 + cy ** 2
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
    return ux, uy


def insert_steiner_points(
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_type: str,
    *,
    max_aspect_ratio: float = 3.0,
    min_angle_deg: float = 20.0,
    max_edge_length: Optional[float] = None,
    max_iterations: int = 3,
    gaussian_vertex_indices: Optional[np.ndarray] = None,
    use_3d_delaunay: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert *Steiner points* (circumcentres of bad triangles) and re-mesh.

    This is a Ruppert-style pre-processing pass that improves the mesh
    before the iterative refinement of :func:`refine_bad_triangles`.

    For each triangle whose aspect ratio exceeds *max_aspect_ratio*
    **or** whose minimum interior angle is below *min_angle_deg*, the
    circumcentre of the ``(x, y)`` footprint is computed, projected
    onto the polynomial surface, and added.  After all Steiner points
    are collected, the mesh is re-triangulated (via 3-D or 2-D Delaunay)
    and boundary-filtered.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
    faces : ndarray ``(F, 3)``
    surface_type : str
    max_aspect_ratio : float
        Triangles with AR above this receive a Steiner point (default 3).
    min_angle_deg : float
        Triangles with min angle below this receive a Steiner point (default 20).
    max_edge_length : float or None
        Long-edge filter threshold.  ``None`` → auto (10× median).
    max_iterations : int
        Maximum number of insertion + re-triangulation rounds (default 3).
    gaussian_vertex_indices : ndarray, optional
        Indices into *vertices* of Gaussian vertices (protected).
    use_3d_delaunay : bool
        If ``True``, re-triangulate with :func:`build_surface_delaunay_3d`;
        otherwise use 2-D Delaunay in ``(x, y)``.
    verbose : bool

    Returns
    -------
    vertices : ndarray ``(V', 3)``
    faces : ndarray ``(F', 3)``
    gaussian_vertex_indices : ndarray
    """
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    if gaussian_vertex_indices is None:
        gaussian_vertex_indices = np.arange(0, dtype=np.int32)
    else:
        gaussian_vertex_indices = np.array(gaussian_vertex_indices, dtype=np.int32)

    # Auto-detect max_edge_length
    if max_edge_length is None:
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        all_el = np.concatenate([
            np.linalg.norm(v1 - v0, axis=1),
            np.linalg.norm(v2 - v1, axis=1),
            np.linalg.norm(v0 - v2, axis=1),
        ])
        max_edge_length = float(10.0 * np.median(all_el))

    for iteration in range(max_iterations):
        ar, min_angle, _ = _triangle_quality(vertices, faces)

        bad = (ar > max_aspect_ratio) | (min_angle < min_angle_deg)
        n_bad = int(bad.sum())
        if n_bad == 0:
            if verbose:
                print(f"        [steiner iter {iteration}] no bad triangles — done")
            break

        bad_faces = faces[bad]

        # Circumcentres in the (x, y) plane, projected onto surface
        ax = vertices[bad_faces[:, 0], 0]
        ay = vertices[bad_faces[:, 0], 1]
        bx = vertices[bad_faces[:, 1], 0]
        by = vertices[bad_faces[:, 1], 1]
        cx = vertices[bad_faces[:, 2], 0]
        cy = vertices[bad_faces[:, 2], 1]

        sx, sy = _circumcenter_2d(ax, ay, bx, by, cx, cy)
        sz = evaluate_polynomial(sx, sy, surface_type)
        steiner_pts = np.column_stack([sx, sy, sz])

        # Deduplicate against existing vertices (avoid very close insertions)
        if len(steiner_pts) > 0:
            tree = KDTree(vertices[:, :2])
            dd, _ = tree.query(steiner_pts[:, :2])
            min_spacing = max_edge_length * 0.05
            keep = dd > min_spacing
            steiner_pts = steiner_pts[keep]

        if len(steiner_pts) == 0:
            if verbose:
                print(f"        [steiner iter {iteration}] "
                      f"all circumcentres too close to existing vertices — done")
            break

        if verbose:
            print(f"        [steiner iter {iteration}] {n_bad} bad triangles, "
                  f"inserting {len(steiner_pts)} Steiner points")

        vertices = np.vstack([vertices, steiner_pts])

        # Re-triangulate
        if use_3d_delaunay:
            vertices, faces = build_surface_delaunay_3d(vertices, surface_type)
        else:
            tri = Delaunay(vertices[:, :2])
            faces = tri.simplices.astype(np.int32)
            # Reproject z
            vertices[:, 2] = evaluate_polynomial(
                vertices[:, 0], vertices[:, 1], surface_type,
            )

        # Long-edge filter
        if max_edge_length > 0:
            faces, _ = _filter_boundary_long_edges(vertices, faces, max_edge_length)

    return vertices, faces, gaussian_vertex_indices


# ── Bowyer–Watson incremental insertion ─────────────────────────────────────

def _bowyer_watson_insert(
    vertices: np.ndarray,
    faces_list: list,
    new_points: np.ndarray,
    surface_type: str,
    adj_cache: Optional[dict] = None,
) -> Tuple[np.ndarray, list, dict]:
    """Insert points one-by-one using the Bowyer–Watson algorithm.

    Operates on the 2-D ``(x, y)`` projection.  For each new point:

    1. Find all faces whose circumcircle contains the point.
    2. Remove those faces, leaving a star-shaped polygonal hole.
    3. Re-triangulate the hole by connecting the new point to the
       boundary edges of the hole.

    This is O(k) per insertion (k = number of affected faces) rather
    than O(N log N) for a full Delaunay rebuild.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
        Current vertex array.  New points will be appended.
    faces_list : list of [int, int, int]
        Current face list (modified in-place).
    new_points : ndarray ``(P, 3)``
        Points to insert.  Must already lie on the surface.
    surface_type : str
    adj_cache : dict, optional
        If provided, an edge→face adjacency dict ``{(va,vb): [fi, …]}``
        that will be maintained incrementally.

    Returns
    -------
    vertices : ndarray ``(V+P, 3)``
    faces_list : list
    adj_cache : dict
    """
    from collections import defaultdict

    xy = vertices[:, :2]
    V0 = len(vertices)

    # Build adjacency: edge (sorted tuple) → list of face indices
    if adj_cache is None:
        adj_cache = defaultdict(list)
        for fi, f in enumerate(faces_list):
            for k in range(3):
                ek = tuple(sorted((f[k], f[(k + 1) % 3])))
                adj_cache[ek].append(fi)

    all_pts = np.vstack([xy, new_points[:, :2]])

    # Pre-grow vertices array
    vertices = np.vstack([vertices, new_points])
    xy = vertices[:, :2]

    for pi in range(len(new_points)):
        vi = V0 + pi
        px, py = float(xy[vi, 0]), float(xy[vi, 1])

        # Find faces whose circumcircle contains (px, py)
        bad_set: set = set()
        # Start from the nearest face (approximate via adj_cache walk)
        # Fall back to scanning all faces
        for fi in range(len(faces_list)):
            f = faces_list[fi]
            if f is None:
                continue
            ax, ay = float(xy[f[0], 0]), float(xy[f[0], 1])
            bx, by = float(xy[f[1], 0]), float(xy[f[1], 1])
            cx, cy = float(xy[f[2], 0]), float(xy[f[2], 1])
            # Circumcircle test
            D = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            if abs(D) < 1e-30:
                continue
            a2 = ax * ax + ay * ay
            b2 = bx * bx + by * by
            c2 = cx * cx + cy * cy
            ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / D
            uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / D
            r2 = (ax - ux) ** 2 + (ay - uy) ** 2
            d2 = (px - ux) ** 2 + (py - uy) ** 2
            if d2 < r2 * (1.0 + 1e-10):
                bad_set.add(fi)

        if not bad_set:
            # No circumcircle contains the point — insert by splitting
            # the nearest face
            best_fi = None
            best_d = float("inf")
            for fi in range(len(faces_list)):
                f = faces_list[fi]
                if f is None:
                    continue
                cmx = (xy[f[0], 0] + xy[f[1], 0] + xy[f[2], 0]) / 3.0
                cmy = (xy[f[0], 1] + xy[f[1], 1] + xy[f[2], 1]) / 3.0
                d = (px - cmx) ** 2 + (py - cmy) ** 2
                if d < best_d:
                    best_d = d
                    best_fi = fi
            if best_fi is not None:
                bad_set.add(best_fi)

        # Collect boundary edges of the cavity (edges shared by exactly
        # one bad face)
        edge_count: dict = defaultdict(int)
        edge_face: dict = {}
        for fi in bad_set:
            f = faces_list[fi]
            for k in range(3):
                ek = tuple(sorted((f[k], f[(k + 1) % 3])))
                edge_count[ek] += 1
                edge_face[ek] = fi

        boundary_edges = [e for e, c in edge_count.items() if c == 1]

        # Remove bad faces from adjacency
        for fi in bad_set:
            f = faces_list[fi]
            for k in range(3):
                ek = tuple(sorted((f[k], f[(k + 1) % 3])))
                if ek in adj_cache:
                    try:
                        adj_cache[ek].remove(fi)
                    except ValueError:
                        pass
                    if not adj_cache[ek]:
                        del adj_cache[ek]
            faces_list[fi] = None  # Mark as deleted

        # Create new faces connecting the new vertex to boundary edges
        for ea, eb in boundary_edges:
            new_fi = len(faces_list)
            faces_list.append([vi, ea, eb])
            for k in range(3):
                f = faces_list[new_fi]
                ek = tuple(sorted((f[k], f[(k + 1) % 3])))
                if ek not in adj_cache:
                    adj_cache[ek] = []
                adj_cache[ek].append(new_fi)

    # Compact: remove None entries
    old_to_new = {}
    compacted: list = []
    for fi, f in enumerate(faces_list):
        if f is not None:
            old_to_new[fi] = len(compacted)
            compacted.append(f)
    faces_list = compacted

    # Rebuild adj_cache after compaction
    new_adj: dict = defaultdict(list)
    for fi, f in enumerate(faces_list):
        for k in range(3):
            ek = tuple(sorted((f[k], f[(k + 1) % 3])))
            new_adj[ek].append(fi)
    adj_cache = new_adj

    return vertices, faces_list, adj_cache


# ── Uniform Laplacian smoothing ─────────────────────────────────────────────

def _uniform_laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    movable_mask: np.ndarray,
    surface_type: str,
    *,
    damping: float = 0.5,
    _adj_cache=None,
) -> Tuple[np.ndarray, "csr_matrix"]:
    """Tangential uniform-Laplacian smoothing using analytical surface normals.

    Computes the uniform Laplacian displacement in **3-D**, projects it
    onto the tangent plane at each vertex (removing the normal component),
    then applies the damped tangential step. This avoids the distortion
    of parameter-space smoothing on curved regions.

    Gaussian vertices are **never** moved — callers should set
    ``movable_mask[gauss_idx] = False``.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
        Modified **in-place**.
    faces : ndarray ``(F, 3)``
    movable_mask : ndarray ``(V,)`` bool
        ``True`` for vertices that may be moved.
    surface_type : str
    damping : float
        Blend factor in (0,1].  1 = full step, 0.5 = half step.
    _adj_cache : sparse matrix or None
        If provided, reuse this adjacency matrix instead of rebuilding.

    Returns
    -------
    (vertices, adj) : tuple
        ``vertices`` is the same object (modified in place).
        ``adj`` is the sparse adjacency matrix for optional reuse.
    """
    from scipy.sparse import csr_matrix

    move_idx = np.where(movable_mask)[0]
    if len(move_idx) == 0:
        return vertices, _adj_cache

    n_v = len(vertices)
    pos = vertices.copy()  # snapshot full 3-D positions

    if _adj_cache is not None:
        adj = _adj_cache
    else:
        # Build sparse adjacency matrix from faces (vectorised)
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        rows = np.concatenate([f0, f0, f1, f1, f2, f2])
        cols = np.concatenate([f1, f2, f0, f2, f0, f1])
        data = np.ones(len(rows), dtype=np.float64)
        adj = csr_matrix((data, (rows, cols)), shape=(n_v, n_v))

    # Degree (number of neighbours per vertex)
    degree = np.asarray(adj.sum(axis=1)).ravel()
    degree = np.maximum(degree, 1.0)  # avoid div-by-zero

    # Unweighted average of neighbours in 3-D
    avg_x = np.asarray(adj.dot(pos[:, 0:1])).ravel() / degree
    avg_y = np.asarray(adj.dot(pos[:, 1:2])).ravel() / degree
    avg_z = np.asarray(adj.dot(pos[:, 2:3])).ravel() / degree

    # 3-D Laplacian displacement for movable vertices
    dx = avg_x[move_idx] - pos[move_idx, 0]
    dy = avg_y[move_idx] - pos[move_idx, 1]
    dz = avg_z[move_idx] - pos[move_idx, 2]

    # Analytical surface normal: n = (-fx, -fy, 1) / ||...||
    fx, fy = _evaluate_gradient(
        pos[move_idx, 0], pos[move_idx, 1], surface_type,
    )
    nx, ny, nz = -fx, -fy, np.ones(len(move_idx))
    inv_norm = 1.0 / np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx *= inv_norm
    ny *= inv_norm
    nz *= inv_norm

    # Project displacement onto tangent plane: d_t = d - (d·n)n
    dot = dx * nx + dy * ny + dz * nz
    dx -= dot * nx
    dy -= dot * ny
    # dz not needed — we reproject z analytically

    # Apply damped tangential displacement (x, y only)
    vertices[move_idx, 0] = pos[move_idx, 0] + damping * dx
    vertices[move_idx, 1] = pos[move_idx, 1] + damping * dy

    # Reproject z onto surface
    vertices[move_idx, 2] = evaluate_polynomial(
        vertices[move_idx, 0], vertices[move_idx, 1], surface_type,
    )

    return vertices, adj


# ── Surface-metric-aware helpers ────────────────────────────────────────────


def _surface_metric_midpoints(
    xy_a: np.ndarray,
    xy_b: np.ndarray,
    surface_type: str,
) -> np.ndarray:
    """Compute geodesic-approximate midpoints between pairs of (x, y) points.

    Instead of the naive parameter-space average ``(A + B) / 2``, this
    applies the first fundamental form at the parameter midpoint to
    correct for surface curvature.

    For a surface *z = f(x, y)* the metric tensor (first fundamental form) is

    .. math::

        G = \\begin{pmatrix}
            1 + f_x^2 & f_x f_y \\\\
            f_x f_y   & 1 + f_y^2
        \\end{pmatrix}

    The midpoint in surface arc-length between A and B is approximated
    by iterative bisection: at each step the parameter midpoint is
    adjusted so that the *surface lengths* of the two sub-segments
    are equal.  One correction step already captures most of the
    curvature effect; two steps are very close to exact.

    Parameters
    ----------
    xy_a, xy_b : ndarray ``(N, 2)``
        Parameter-plane coordinates of edge endpoints.
    surface_type : str

    Returns
    -------
    midpoints_xy : ndarray ``(N, 2)``
    """
    # Start with parameter-space midpoint
    mid = (xy_a + xy_b) / 2.0

    # Two correction iterations are sufficient for polynomial surfaces.
    # Evaluate the metric separately on each sub-segment (A→mid and
    # mid→B) at their respective midpoints to capture the gradient
    # variation along the edge.
    for _ in range(2):
        # Metric at midpoint of (A → mid)
        sub_a = (xy_a + mid) / 2.0
        fx_a, fy_a = _evaluate_gradient(sub_a[:, 0], sub_a[:, 1], surface_type)
        g11_a = 1.0 + fx_a ** 2
        g12_a = fx_a * fy_a
        g22_a = 1.0 + fy_a ** 2

        # Metric at midpoint of (mid → B)
        sub_b = (mid + xy_b) / 2.0
        fx_b, fy_b = _evaluate_gradient(sub_b[:, 0], sub_b[:, 1], surface_type)
        g11_b = 1.0 + fx_b ** 2
        g12_b = fx_b * fy_b
        g22_b = 1.0 + fy_b ** 2

        # Surface length of each sub-segment under its local metric
        da = mid - xy_a  # (N, 2)
        db = xy_b - mid

        len_a_sq = g11_a * da[:, 0] ** 2 + 2.0 * g12_a * da[:, 0] * da[:, 1] + g22_a * da[:, 1] ** 2
        len_b_sq = g11_b * db[:, 0] ** 2 + 2.0 * g12_b * db[:, 0] * db[:, 1] + g22_b * db[:, 1] ** 2

        len_a = np.sqrt(np.maximum(len_a_sq, 0.0))
        len_b = np.sqrt(np.maximum(len_b_sq, 0.0))

        # Shift midpoint so that both halves have approximately equal
        # surface length.  t ∈ (0,1) along A→B:
        #   t = len_a / (len_a + len_b)   (=0.5 when equal)
        total = len_a + len_b
        t = np.where(total > 1e-15, len_a / total, 0.5)
        mid = xy_a + t[:, None] * (xy_b - xy_a)

    return mid


def _surface_distances(
    xy_from: np.ndarray,
    xy_to: np.ndarray,
    surface_type: str,
) -> np.ndarray:
    """Approximate geodesic distance between pairs of (x, y) points.

    Uses the first fundamental form evaluated at the midpoint of each
    segment to convert parameter-space displacement into 3-D surface
    arc-length.

    Parameters
    ----------
    xy_from, xy_to : ndarray ``(N, 2)``
    surface_type : str

    Returns
    -------
    distances : ndarray ``(N,)``
    """
    mid = (xy_from + xy_to) / 2.0
    fx, fy = _evaluate_gradient(mid[:, 0], mid[:, 1], surface_type)

    g11 = 1.0 + fx ** 2
    g12 = fx * fy
    g22 = 1.0 + fy ** 2

    d = xy_to - xy_from
    ds_sq = g11 * d[:, 0] ** 2 + 2.0 * g12 * d[:, 0] * d[:, 1] + g22 * d[:, 1] ** 2
    return np.sqrt(np.maximum(ds_sq, 0.0))


def _metric_weighted_laplacian_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    movable_mask: np.ndarray,
    surface_type: str,
    *,
    damping: float = 0.5,
    _W_cache=None,
) -> Tuple[np.ndarray, "csr_matrix"]:
    """Tangential Laplacian smoothing using analytical surface normals.

    Computes the weighted Laplacian displacement in **3-D**, then projects
    it onto the **tangent plane** at each vertex (removing the normal
    component) before applying.  Neighbours are weighted by inverse
    surface distance so that closer neighbours on the surface contribute
    more.

    Compared to parameter-space smoothing this produces geometrically
    correct displacements: on steep parts of the surface (where the
    gradient is large) parameter-space displacements are disproportionately
    large, but tangential smoothing rescales them properly.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
        Modified **in-place**.
    faces : ndarray ``(F, 3)``
    movable_mask : ndarray ``(V,)`` bool
    surface_type : str
    damping : float
    _W_cache : sparse matrix or None
        If provided, skip weight-matrix construction and use this
        directly.  Returned as the second element so callers can
        reuse it across passes.

    Returns
    -------
    (vertices, W) : tuple
        ``vertices`` is the same object (modified in place).
        ``W`` is the sparse weight matrix for optional reuse.
    """
    from scipy.sparse import csr_matrix

    move_idx = np.where(movable_mask)[0]
    if len(move_idx) == 0:
        return vertices, _W_cache

    n_v = len(vertices)
    pos = vertices.copy()  # snapshot full 3-D positions

    if _W_cache is not None:
        W = _W_cache
    else:
        xy = pos[:, :2]
        f0, f1, f2 = faces[:, 0], faces[:, 1], faces[:, 2]
        rows = np.concatenate([f0, f0, f1, f1, f2, f2])
        cols = np.concatenate([f1, f2, f0, f2, f0, f1])
        edge_sd = _surface_distances(xy[rows], xy[cols], surface_type)
        edge_w = 1.0 / np.maximum(edge_sd, 1e-15)
        W = csr_matrix((edge_w, (rows, cols)), shape=(n_v, n_v))

    w_sum = np.asarray(W.sum(axis=1)).ravel()
    w_sum = np.maximum(w_sum, 1e-15)

    # Weighted average of neighbours in 3-D
    avg_x = np.asarray(W.dot(pos[:, 0:1])).ravel() / w_sum
    avg_y = np.asarray(W.dot(pos[:, 1:2])).ravel() / w_sum
    avg_z = np.asarray(W.dot(pos[:, 2:3])).ravel() / w_sum

    # 3-D Laplacian displacement for movable vertices
    dx = avg_x[move_idx] - pos[move_idx, 0]
    dy = avg_y[move_idx] - pos[move_idx, 1]
    dz = avg_z[move_idx] - pos[move_idx, 2]

    # Analytical surface normal: n = (-fx, -fy, 1) / ||...||
    fx, fy = _evaluate_gradient(
        pos[move_idx, 0], pos[move_idx, 1], surface_type,
    )
    nx, ny, nz = -fx, -fy, np.ones(len(move_idx))
    inv_norm = 1.0 / np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    nx *= inv_norm
    ny *= inv_norm
    nz *= inv_norm

    # Project displacement onto tangent plane: d_t = d - (d·n)n
    dot = dx * nx + dy * ny + dz * nz
    dx -= dot * nx
    dy -= dot * ny
    # dz not needed — we reproject z analytically

    # Apply damped tangential displacement (x, y only)
    vertices[move_idx, 0] = pos[move_idx, 0] + damping * dx
    vertices[move_idx, 1] = pos[move_idx, 1] + damping * dy

    # Reproject z onto surface
    vertices[move_idx, 2] = evaluate_polynomial(
        vertices[move_idx, 0], vertices[move_idx, 1], surface_type,
    )

    return vertices, W


# ── Large-edge splitting ────────────────────────────────────────────────────

def _split_large_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_edge_length: float,
    surface_type: str,
    *,
    is_3d: bool = False,
    max_splits_per_iter: int = 5000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split faces with edges exceeding *max_edge_length* by inserting midpoints.

    Instead of *removing* interior faces with long edges (which creates
    holes), this function *splits* them: for each face with at least one
    edge > *max_edge_length*, the midpoint of the longest edge is
    inserted (projected onto the surface) and the face is re-triangulated
    using Bowyer–Watson.

    Works for both 2-D and 3-D triangulations.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
    faces : ndarray ``(F, 3)``
    max_edge_length : float
    surface_type : str
    is_3d : bool
        If ``True``, re-triangulate with 3-D Delaunay after splitting;
        otherwise use 2-D Bowyer–Watson.
    max_splits_per_iter : int
        Maximum midpoints to insert per call (prevents runaway).
    verbose : bool

    Returns
    -------
    vertices : ndarray ``(V', 3)``
    faces : ndarray ``(F', 3)``
    """
    if max_edge_length <= 0 or len(faces) == 0:
        return vertices, faces

    fv0 = vertices[faces[:, 0]]
    fv1 = vertices[faces[:, 1]]
    fv2 = vertices[faces[:, 2]]

    e_lens = np.column_stack([
        np.linalg.norm(fv1 - fv0, axis=1),   # edge 0→1
        np.linalg.norm(fv2 - fv1, axis=1),   # edge 1→2
        np.linalg.norm(fv0 - fv2, axis=1),   # edge 2→0
    ])
    edge_max = e_lens.max(axis=1)
    bad_mask = edge_max > max_edge_length
    n_bad = int(bad_mask.sum())

    if n_bad == 0:
        return vertices, faces

    # Limit number of splits
    bad_idx = np.where(bad_mask)[0]
    if len(bad_idx) > max_splits_per_iter:
        bad_idx = bad_idx[:max_splits_per_iter]

    bad_f = faces[bad_idx]
    bad_e = e_lens[bad_idx]
    long_idx = np.argmax(bad_e, axis=1)

    # Midpoint of longest edge per bad face
    edge_v = np.array([[0, 1], [1, 2], [2, 0]])
    n_b = len(bad_f)
    va_idx = bad_f[np.arange(n_b), edge_v[long_idx, 0]]
    vb_idx = bad_f[np.arange(n_b), edge_v[long_idx, 1]]

    midpoints_xy = (vertices[va_idx, :2] + vertices[vb_idx, :2]) / 2.0

    # Deduplicate
    min_spacing = max_edge_length * 0.05
    if len(midpoints_xy) > 1:
        rounded = np.round(midpoints_xy / max(min_spacing, 1e-15)).astype(np.int64)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        midpoints_xy = midpoints_xy[np.sort(unique_idx)]

    # Deduplicate against existing vertices
    if len(midpoints_xy) > 0:
        kd = KDTree(vertices[:, :2])
        dd, _ = kd.query(midpoints_xy)
        midpoints_xy = midpoints_xy[dd > min_spacing]

    if len(midpoints_xy) == 0:
        return vertices, faces

    mid_z = evaluate_polynomial(midpoints_xy[:, 0], midpoints_xy[:, 1], surface_type)
    new_pts = np.column_stack([midpoints_xy, mid_z])

    if verbose:
        print(f"          [split_large_edges] inserting {len(new_pts)} midpoints")

    if is_3d:
        vertices = np.vstack([vertices, new_pts])
        vertices, faces = build_surface_delaunay_3d(vertices, surface_type)
    else:
        # Full 2-D Delaunay retriangulation
        vertices = np.vstack([vertices, new_pts])
        tri = Delaunay(vertices[:, :2])
        faces = tri.simplices.astype(np.int32)
        # Reproject z onto surface
        vertices[:, 2] = evaluate_polynomial(
            vertices[:, 0], vertices[:, 1], surface_type,
        )

    return vertices, faces


# ── Ring-based repair for invalid Gaussians ─────────────────────────────────

def _fix_invalid_gaussians_with_rings(
    vertices: np.ndarray,
    faces: np.ndarray,
    is_gaussian: np.ndarray,
    max_edge_length: float,
    surface_type: str,
    gauss_ar_threshold: float,
    gauss_angle_threshold: float,
    gauss_edge_threshold: Optional[float] = None,
    max_ring_steps: int = 1,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add support rings around Gaussian vertices in bad triangles.

    Works like :func:`fix_orphaned_gaussians` but targets all Gaussian
    vertices participating in at least one *bad* (high aspect-ratio or
    low minimum-angle) triangle, not just orphaned ones.

    When two targeted Gaussians share a mesh edge, their ring points may
    nearly coincide.  All proposed ring points are deduplicated via a
    KDTree so that nearby rings share vertices — keeping the insertion
    count minimal.

    Multiple steps can be run (``max_ring_steps > 1``): each step
    re-evaluates which Gaussians are still in bad triangles and adds
    fresh rings with a **halved radius** so that successive passes
    fill in finer detail.  The loop exits early if no new ring points
    are generated.

    Parameters
    ----------
    vertices : ndarray ``(V, 3)``
    faces : ndarray ``(F, 3)``
    is_gaussian : ndarray ``(V,)`` bool
    max_edge_length : float
    surface_type : str
    gauss_ar_threshold, gauss_angle_threshold : float
    gauss_edge_threshold : float or None
    max_ring_steps : int
        Number of ring insertion passes (default 1).
    verbose : bool

    Returns
    -------
    vertices, faces, is_gaussian : updated arrays
    """
    radius_scale = 1.0  # shrinks each step

    for step in range(max(max_ring_steps, 1)):
        n_before = len(vertices)
        vertices, faces, is_gaussian = _ring_fix_single_pass(
            vertices, faces, is_gaussian, max_edge_length, surface_type,
            gauss_ar_threshold, gauss_angle_threshold, gauss_edge_threshold,
            radius_scale=radius_scale, verbose=verbose, step=step,
        )
        n_inserted = len(vertices) - n_before
        if n_inserted == 0:
            if verbose:
                print(f"          [ring fix] step {step}: no new points — done")
            break
        radius_scale *= 0.5  # halve radius for next pass

    return vertices, faces, is_gaussian


def _ring_fix_single_pass(
    vertices: np.ndarray,
    faces: np.ndarray,
    is_gaussian: np.ndarray,
    max_edge_length: float,
    surface_type: str,
    gauss_ar_threshold: float,
    gauss_angle_threshold: float,
    gauss_edge_threshold: Optional[float] = None,
    radius_scale: float = 1.0,
    verbose: bool = False,
    step: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single pass of ring-based support-point insertion (internal).

    Uses **adaptive per-Gaussian ring radius** based on the distance
    to the nearest Gaussian neighbour.  Gaussians in dense clusters
    get small, tightly-spaced rings; isolated Gaussians get larger
    rings. This produces far more surviving points after deduplication
    than a single global radius.
    """
    # ── 1. Identify Gaussian-touching bad triangles ────────────────────
    ar, min_angle, longest_edge = _triangle_quality(vertices, faces)
    touches_gauss = (
        is_gaussian[faces[:, 0]]
        | is_gaussian[faces[:, 1]]
        | is_gaussian[faces[:, 2]]
    )
    bad_gauss = touches_gauss & (
        (ar > gauss_ar_threshold) | (min_angle < gauss_angle_threshold)
    )
    if gauss_edge_threshold is not None:
        bad_gauss = bad_gauss | (
            touches_gauss & (longest_edge > gauss_edge_threshold)
        )
    if not bad_gauss.any():
        return vertices, faces, is_gaussian

    # ── 2. Collect Gaussian vertices in bad triangles ──────────────────
    bad_face_verts = faces[bad_gauss]
    invalid_gauss_mask = np.zeros(len(vertices), dtype=bool)
    for col in range(3):
        vi = bad_face_verts[:, col]
        invalid_gauss_mask[vi] = invalid_gauss_mask[vi] | is_gaussian[vi]
    invalid_gauss_indices = np.where(invalid_gauss_mask)[0]

    if len(invalid_gauss_indices) == 0:
        return vertices, faces, is_gaussian

    # ── 3. Adaptive per-Gaussian ring radius ───────────────────────────
    eps = max_edge_length * 0.01
    gxy = vertices[invalid_gauss_indices, :2]  # (K, 2)
    gz = vertices[invalid_gauss_indices, 2]    # (K,)

    # Local surface metric scale
    zx = evaluate_polynomial(gxy[:, 0] + eps, gxy[:, 1], surface_type)
    zy = evaluate_polynomial(gxy[:, 0], gxy[:, 1] + eps, surface_type)
    dzdx = (zx - gz) / eps
    dzdy = (zy - gz) / eps
    scale = np.sqrt(1.0 + dzdx ** 2 + dzdy ** 2)

    # Find nearest-Gaussian-neighbour distance for each invalid Gaussian.
    # All Gaussian vertices (not just invalid ones) are reference points.
    all_gauss_xy = vertices[is_gaussian, :2]
    kd_gauss = KDTree(all_gauss_xy)
    # k=2 because the nearest neighbour of a point in the set is itself
    dd_gauss, _ = kd_gauss.query(gxy, k=min(2, len(all_gauss_xy)))
    if dd_gauss.ndim == 1:
        nn_dist = dd_gauss
    else:
        nn_dist = dd_gauss[:, 1]  # second-nearest = true nearest neighbour

    # Adaptive 3D ring radius: proportional to local Gaussian spacing,
    # clamped between a sensible min and max.
    r_max_3d = max_edge_length * 0.65 * radius_scale
    median_edge = float(np.median(longest_edge))
    r_min_3d = median_edge * 0.5

    # Target: 2× nearest-neighbour distance — large enough to create a
    # well-spaced ring that doesn't overlap with neighbouring Gaussians.
    target_r_3d = np.clip(nn_dist * 2.0, r_min_3d, r_max_3d)  # (K,)

    # Convert to 2D parameter-space radius accounting for surface metric
    ring_r_2d = target_r_3d / np.maximum(scale, 1e-6)  # (K,)

    # ── 4. Generate ring points (8 per Gaussian for better coverage) ───
    n_ring = 8
    angles = np.linspace(0, 2 * np.pi, n_ring + 1)[:-1]
    cos_a = np.cos(angles)  # (n_ring,)
    sin_a = np.sin(angles)  # (n_ring,)
    offsets_x = ring_r_2d[:, None] * cos_a[None, :]  # (K, n_ring)
    offsets_y = ring_r_2d[:, None] * sin_a[None, :]  # (K, n_ring)
    ring_xy = np.empty((len(invalid_gauss_indices), n_ring, 2))
    ring_xy[:, :, 0] = gxy[:, 0:1] + offsets_x
    ring_xy[:, :, 1] = gxy[:, 1:2] + offsets_y
    ring_xy_flat = ring_xy.reshape(-1, 2)  # (K*n_ring, 2)
    # Track which ring radius each point came from for adaptive dedup
    ring_r_per_pt = np.repeat(target_r_3d, n_ring)  # (K*n_ring,)

    # ── 5. Deduplicate — adaptive spacing per point ────────────────────
    # Dedup spacing is 25% of the *local* ring radius (not global).
    # Use the minimum radius of nearby candidates so dense clusters
    # keep their fine-grained points.
    local_dedup = ring_r_per_pt * 0.25
    # Use a global minimum dedup spacing to avoid creating near-
    # degenerate edges.
    global_min_dedup = max(r_min_3d * 0.2, 1e-7)
    local_dedup = np.maximum(local_dedup, global_min_dedup)

    # Grid-based self-dedup using the *median* local dedup spacing.
    # After the grid dedup, we do a fine-grained KDTree pass.
    median_dedup = float(np.median(local_dedup))
    if len(ring_xy_flat) > 1 and median_dedup > 0:
        rounded = np.round(ring_xy_flat / median_dedup).astype(np.int64)
        _, unique_idx = np.unique(rounded, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        ring_xy_flat = ring_xy_flat[unique_idx]
        ring_r_per_pt = ring_r_per_pt[unique_idx]
        local_dedup = local_dedup[unique_idx]

    # Dedup against existing vertices — keep points far enough from
    # any existing vertex, using a fraction of local ring radius.
    min_spacing_existing = local_dedup * 0.8
    if len(ring_xy_flat) > 0:
        kd_existing = KDTree(vertices[:, :2])
        dd, _ = kd_existing.query(ring_xy_flat)
        keep = dd > min_spacing_existing
        ring_xy_flat = ring_xy_flat[keep]

    if len(ring_xy_flat) == 0:
        return vertices, faces, is_gaussian

    # ── 6. Project ring points onto surface ────────────────────────────
    ring_z = evaluate_polynomial(ring_xy_flat[:, 0], ring_xy_flat[:, 1], surface_type)
    ring_pts = np.column_stack([ring_xy_flat, ring_z])

    # ── 7. Local face-split insertion ──────────────────────────────────
    # Instead of a global Delaunay rebuild (which creates long-range
    # slivers), insert each ring point into its containing face locally:
    # fan-split for 1-point faces, local Delaunay for multi-point faces.
    # Lawson flips then restore the Delaunay property at boundaries.

    # --- Point-location: find each ring point's containing face ---
    xy_v = vertices[:, :2]
    centroids = (
        xy_v[faces[:, 0]] + xy_v[faces[:, 1]] + xy_v[faces[:, 2]]
    ) / 3.0
    cent_kd = KDTree(centroids)
    k_check = min(12, len(faces))
    _, nearest_fi = cent_kd.query(ring_pts[:, :2], k=k_check)
    if nearest_fi.ndim == 1:
        nearest_fi = nearest_fi[:, None]

    containing = np.full(len(ring_pts), -1, dtype=np.int64)
    _px = ring_pts[:, 0]
    _py = ring_pts[:, 1]
    for j in range(nearest_fi.shape[1]):
        still = containing < 0
        if not still.any():
            break
        fi_j = nearest_fi[still, j]
        a = faces[fi_j, 0]; b = faces[fi_j, 1]; c = faces[fi_j, 2]
        ax, ay = xy_v[a, 0], xy_v[a, 1]
        bx, by = xy_v[b, 0], xy_v[b, 1]
        cx, cy = xy_v[c, 0], xy_v[c, 1]
        v0x = cx - ax; v0y = cy - ay
        v1x = bx - ax; v1y = by - ay
        v2x = _px[still] - ax; v2y = _py[still] - ay
        d00 = v0x * v0x + v0y * v0y
        d01 = v0x * v1x + v0y * v1y
        d02 = v0x * v2x + v0y * v2y
        d11 = v1x * v1x + v1y * v1y
        d12 = v1x * v2x + v1y * v2y
        inv = 1.0 / (d00 * d11 - d01 * d01 + 1e-30)
        u = (d11 * d02 - d01 * d12) * inv
        v = (d00 * d12 - d01 * d02) * inv
        inside = (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)
        idx = np.where(still)[0]
        containing[idx[inside]] = fi_j[inside]

    # Keep only ring points that fall inside an existing face
    valid = containing >= 0
    ring_pts = ring_pts[valid]
    ring_face_ids = containing[valid]

    if len(ring_pts) == 0:
        return vertices, faces, is_gaussian

    n_new = len(ring_pts)
    V0 = len(vertices)
    vertices = np.vstack([vertices, ring_pts])
    is_gaussian = np.concatenate([is_gaussian, np.zeros(n_new, dtype=bool)])
    xy = vertices[:, :2]

    ring_global = np.arange(V0, V0 + n_new, dtype=np.int64)

    if verbose:
        print(
            f"          [ring fix] step {step}: {len(invalid_gauss_indices)} invalid Gaussian(s), "
            f"inserting {n_new} ring points "
            f"(radius_scale={radius_scale:.2f}, "
            f"r_median={float(np.median(target_r_3d)):.5f}, "
            f"r_range=[{float(target_r_3d.min()):.5f}, {float(target_r_3d.max()):.5f}])"
        )

    # Group ring points by their containing face
    sort_order = np.argsort(ring_face_ids)
    sorted_face_ids = ring_face_ids[sort_order]
    sorted_ring_global = ring_global[sort_order]

    change_idx = np.where(np.diff(sorted_face_ids) != 0)[0] + 1
    group_starts = np.concatenate([[0], change_idx])
    group_ends = np.concatenate([change_idx, [len(sorted_face_ids)]])
    group_face_ids = sorted_face_ids[group_starts]

    faces_with_ring = set(group_face_ids.tolist())
    n_faces = len(faces)

    # Faces with no ring points: keep unchanged
    no_ring_mask = np.ones(n_faces, dtype=bool)
    for fi in faces_with_ring:
        no_ring_mask[fi] = False
    out_parts: list = [faces[no_ring_mask]]

    # Separate 1-point faces (vectorised fan) vs 2+ (local Delaunay)
    group_sizes = group_ends - group_starts
    one_mask = group_sizes == 1
    one_fi = group_face_ids[one_mask].astype(np.int64)
    one_ri = sorted_ring_global[group_starts[one_mask]].astype(np.int64)
    multi_idx = np.where(~one_mask)[0]

    # Vectorised fan for 1-point faces: [A,B,C]+P → [P,A,B],[P,B,C],[P,C,A]
    if len(one_fi) > 0:
        A = faces[one_fi, 0].astype(np.int64)
        B = faces[one_fi, 1].astype(np.int64)
        C = faces[one_fi, 2].astype(np.int64)
        P = one_ri
        fan = np.empty((3 * len(one_fi), 3), dtype=np.int64)
        fan[0::3, 0] = P; fan[0::3, 1] = A; fan[0::3, 2] = B
        fan[1::3, 0] = P; fan[1::3, 1] = B; fan[1::3, 2] = C
        fan[2::3, 0] = P; fan[2::3, 1] = C; fan[2::3, 2] = A
        out_parts.append(fan)

    # Local Delaunay for multi-point faces
    if len(multi_idx) > 0:
        for mi in multi_idx:
            fi = int(group_face_ids[mi])
            ri_slice = sorted_ring_global[group_starts[mi]:group_ends[mi]]
            corners = faces[fi].astype(np.int64)
            local_global = np.concatenate([corners, ri_slice])
            local_xy = xy[local_global]
            local_tri = Delaunay(local_xy)
            out_parts.append(local_global[local_tri.simplices])

    faces = np.vstack(out_parts).astype(np.int32)

    # Reproject z for consistency
    vertices[:, 2] = evaluate_polynomial(vertices[:, 0], vertices[:, 1], surface_type)

    # Lawson flips to restore Delaunay property at subdivision boundaries
    faces_i64 = faces.astype(np.int64)
    total_flips = 0
    for _ in range(10):
        nf = _lawson_flip_pass(xy, faces_i64)
        total_flips += nf
        if nf == 0:
            break
    faces = faces_i64.astype(np.int32)
    if verbose and total_flips > 0:
        print(f"          [ring local insertion] {total_flips} Lawson flips")

    # Filter convex-hull artifacts
    if max_edge_length > 0:
        faces, _ = _filter_convex_hull_artifacts(
            vertices, faces, max_edge_length,
            surface_type=surface_type, is_3d=False,
        )

    return vertices, faces, is_gaussian


def _circumcenter_steiner_pass(
    vertices: np.ndarray,
    faces: np.ndarray,
    is_gaussian: np.ndarray,
    max_edge_length: float,
    surface_type: str,
    gauss_ar_threshold: float,
    gauss_angle_threshold: float,
    max_iterations: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert circumcenters of bad Gaussian triangles (Ruppert-style).

    For each bad triangle touching a Gaussian vertex, computes the
    circumcenter in the 2-D (x, y) plane.  For obtuse triangles
    where the circumcenter falls outside, an *off-center* point is
    used instead — placed on the perpendicular bisector of the
    shortest edge at a distance that guarantees good angles.

    Insertion uses the same local face-split + Lawson flip strategy
    as ring insertion, avoiding global Delaunay rebuilds.

    Parameters
    ----------
    vertices, faces, is_gaussian : current mesh state
    max_edge_length : float
    surface_type : str
    gauss_ar_threshold, gauss_angle_threshold : float
        Thresholds for identifying bad Gaussian triangles.
    max_iterations : int
        Maximum circumcenter insertion rounds (default 3).
    verbose : bool
    """
    for iteration in range(max_iterations):
        ar, min_angle, _ = _triangle_quality(vertices, faces)
        touches_gauss = (
            is_gaussian[faces[:, 0]]
            | is_gaussian[faces[:, 1]]
            | is_gaussian[faces[:, 2]]
        )
        bad_gauss = touches_gauss & (
            (ar > gauss_ar_threshold) | (min_angle < gauss_angle_threshold)
        )
        n_bad = int(bad_gauss.sum())
        if n_bad == 0:
            break

        bad_f = faces[bad_gauss]
        xy = vertices[:, :2]

        # Circumcenters
        ax, ay = xy[bad_f[:, 0], 0], xy[bad_f[:, 0], 1]
        bx, by = xy[bad_f[:, 1], 0], xy[bad_f[:, 1], 1]
        cx, cy = xy[bad_f[:, 2], 0], xy[bad_f[:, 2], 1]
        sx, sy = _circumcenter_2d(ax, ay, bx, by, cx, cy)

        # Check if circumcenter is inside its triangle (barycentric test)
        v0x, v0y = cx - ax, cy - ay
        v1x, v1y = bx - ax, by - ay
        v2x, v2y = sx - ax, sy - ay
        d00 = v0x * v0x + v0y * v0y
        d01 = v0x * v1x + v0y * v1y
        d02 = v0x * v2x + v0y * v2y
        d11 = v1x * v1x + v1y * v1y
        d12 = v1x * v2x + v1y * v2y
        inv_denom = 1.0 / (d00 * d11 - d01 * d01 + 1e-30)
        u = (d11 * d02 - d01 * d12) * inv_denom
        v = (d00 * d12 - d01 * d02) * inv_denom
        inside = (u >= -1e-6) & (v >= -1e-6) & ((u + v) <= 1.0 + 1e-6)

        # For obtuse triangles: use off-center on the longest edge
        # perpendicular bisector, at sqrt(2)× half the shortest edge
        # from the midpoint — guarantees min angle ~26.6°.
        outside = ~inside
        if outside.any():
            # Compute edge lengths
            e01 = np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            e12 = np.sqrt((cx - bx) ** 2 + (cy - by) ** 2)
            e20 = np.sqrt((ax - cx) ** 2 + (ay - cy) ** 2)
            edges = np.column_stack([e01, e12, e20])
            longest_idx = np.argmax(edges, axis=1)

            # For obtuse triangles, use centroid as the off-center
            # (always inside, creates reasonable sub-triangles)
            cen_x = (ax + bx + cx) / 3.0
            cen_y = (ay + by + cy) / 3.0
            sx[outside] = cen_x[outside]
            sy[outside] = cen_y[outside]

        # Project onto surface
        sz = evaluate_polynomial(sx, sy, surface_type)
        steiner_pts = np.column_stack([sx, sy, sz])

        # Deduplicate against existing vertices
        if len(steiner_pts) > 0:
            kd = KDTree(vertices[:, :2])
            dd, _ = kd.query(steiner_pts[:, :2])
            el = np.concatenate([
                np.linalg.norm(vertices[faces[:, 1], :2] - vertices[faces[:, 0], :2], axis=1),
                np.linalg.norm(vertices[faces[:, 2], :2] - vertices[faces[:, 1], :2], axis=1),
                np.linalg.norm(vertices[faces[:, 0], :2] - vertices[faces[:, 2], :2], axis=1),
            ])
            min_spacing = float(np.median(el)) * 0.15
            keep = dd > min_spacing
            steiner_pts = steiner_pts[keep]

        # Self-dedup via grid
        if len(steiner_pts) > 1:
            rounded = np.round(steiner_pts[:, :2] / max(min_spacing, 1e-15)).astype(np.int64)
            _, unique_idx = np.unique(rounded, axis=0, return_index=True)
            steiner_pts = steiner_pts[np.sort(unique_idx)]

        if len(steiner_pts) == 0:
            if verbose:
                print(f"          [steiner] iter {iteration}: all circumcenters deduped — done")
            break

        # ── Local face-split insertion (same as ring insertion) ─────
        centroids = (
            xy[faces[:, 0]] + xy[faces[:, 1]] + xy[faces[:, 2]]
        ) / 3.0
        cent_kd = KDTree(centroids)
        k_check = min(12, len(faces))
        _, nearest_fi = cent_kd.query(steiner_pts[:, :2], k=k_check)
        if nearest_fi.ndim == 1:
            nearest_fi = nearest_fi[:, None]

        containing = np.full(len(steiner_pts), -1, dtype=np.int64)
        _px, _py = steiner_pts[:, 0], steiner_pts[:, 1]
        for j in range(nearest_fi.shape[1]):
            still = containing < 0
            if not still.any():
                break
            fi_j = nearest_fi[still, j]
            a = faces[fi_j, 0]; b = faces[fi_j, 1]; c = faces[fi_j, 2]
            _ax, _ay = xy[a, 0], xy[a, 1]
            _bx, _by = xy[b, 0], xy[b, 1]
            _cx, _cy = xy[c, 0], xy[c, 1]
            _v0x = _cx - _ax; _v0y = _cy - _ay
            _v1x = _bx - _ax; _v1y = _by - _ay
            _v2x = _px[still] - _ax; _v2y = _py[still] - _ay
            _d00 = _v0x * _v0x + _v0y * _v0y
            _d01 = _v0x * _v1x + _v0y * _v1y
            _d02 = _v0x * _v2x + _v0y * _v2y
            _d11 = _v1x * _v1x + _v1y * _v1y
            _d12 = _v1x * _v2x + _v1y * _v2y
            _inv = 1.0 / (_d00 * _d11 - _d01 * _d01 + 1e-30)
            _u = (_d11 * _d02 - _d01 * _d12) * _inv
            _v = (_d00 * _d12 - _d01 * _d02) * _inv
            _inside = (_u >= -1e-6) & (_v >= -1e-6) & ((_u + _v) <= 1.0 + 1e-6)
            idx = np.where(still)[0]
            containing[idx[_inside]] = fi_j[_inside]

        valid = containing >= 0
        steiner_pts = steiner_pts[valid]
        steiner_face_ids = containing[valid]

        if len(steiner_pts) == 0:
            if verbose:
                print(f"          [steiner] iter {iteration}: no points located — done")
            break

        n_new = len(steiner_pts)
        V0 = len(vertices)
        vertices = np.vstack([vertices, steiner_pts])
        is_gaussian = np.concatenate([is_gaussian, np.zeros(n_new, dtype=bool)])
        xy = vertices[:, :2]
        new_global = np.arange(V0, V0 + n_new, dtype=np.int64)

        # Group by face and fan-split / local Delaunay
        sort_order = np.argsort(steiner_face_ids)
        sorted_fids = steiner_face_ids[sort_order]
        sorted_globals = new_global[sort_order]

        change_idx = np.where(np.diff(sorted_fids) != 0)[0] + 1
        group_starts = np.concatenate([[0], change_idx])
        group_ends = np.concatenate([change_idx, [len(sorted_fids)]])
        group_fids = sorted_fids[group_starts]

        faces_with_pts = set(group_fids.tolist())
        no_pts_mask = np.ones(len(faces), dtype=bool)
        for fi in faces_with_pts:
            no_pts_mask[fi] = False
        out_parts: list = [faces[no_pts_mask]]

        group_sizes = group_ends - group_starts
        one_mask = group_sizes == 1
        one_fi = group_fids[one_mask].astype(np.int64)
        one_pi = sorted_globals[group_starts[one_mask]].astype(np.int64)
        multi_idx = np.where(~one_mask)[0]

        if len(one_fi) > 0:
            A = faces[one_fi, 0].astype(np.int64)
            B = faces[one_fi, 1].astype(np.int64)
            C = faces[one_fi, 2].astype(np.int64)
            P = one_pi
            fan = np.empty((3 * len(one_fi), 3), dtype=np.int64)
            fan[0::3, 0] = P; fan[0::3, 1] = A; fan[0::3, 2] = B
            fan[1::3, 0] = P; fan[1::3, 1] = B; fan[1::3, 2] = C
            fan[2::3, 0] = P; fan[2::3, 1] = C; fan[2::3, 2] = A
            out_parts.append(fan)

        if len(multi_idx) > 0:
            for mi in multi_idx:
                fi = int(group_fids[mi])
                pi_slice = sorted_globals[group_starts[mi]:group_ends[mi]]
                corners = faces[fi].astype(np.int64)
                local_global = np.concatenate([corners, pi_slice])
                local_xy = xy[local_global]
                local_tri = Delaunay(local_xy)
                out_parts.append(local_global[local_tri.simplices])

        faces = np.vstack(out_parts).astype(np.int32)

        # Reproject z
        vertices[:, 2] = evaluate_polynomial(
            vertices[:, 0], vertices[:, 1], surface_type
        )

        # Lawson flips
        faces_i64 = faces.astype(np.int64)
        total_flips = 0
        for _ in range(10):
            nf = _lawson_flip_pass(xy, faces_i64)
            total_flips += nf
            if nf == 0:
                break
        faces = faces_i64.astype(np.int32)

        if verbose:
            ar2, ma2, _ = _triangle_quality(vertices[:, :2], faces)
            bad2 = (ar2 > gauss_ar_threshold) | (ma2 < gauss_angle_threshold)
            print(
                f"          [steiner] iter {iteration}: {n_bad} bad Gauss tri → "
                f"inserted {n_new} circumcenters → {bad2.sum()} bad total "
                f"({total_flips} flips)"
            )

        # Filter convex-hull artifacts
        if max_edge_length > 0:
            faces, _ = _filter_convex_hull_artifacts(
                vertices, faces, max_edge_length,
                surface_type=surface_type, is_3d=False,
            )

    return vertices, faces, is_gaussian


def refine_bad_triangles(
    vertices: np.ndarray,
    faces: np.ndarray,
    surface_type: str,
    *,
    max_aspect_ratio: float = 5.0,
    min_angle_deg: float = 10.0,
    max_area_factor: float = 3.0,
    max_iterations: int = 5,
    max_edge_length: Optional[float] = None,
    gaussian_vertex_indices: Optional[np.ndarray] = None,
    gauss_max_aspect_ratio: Optional[float] = None,
    gauss_min_angle_deg: Optional[float] = None,
    gauss_max_edge_length: Optional[float] = None,
    gauss_max_area_factor: Optional[float] = None,
    warmup_iterations: int = 0,
    ring_fix_invalid_gaussians: bool = False,
    ring_fix_iterations: int = 1,
    surface_aware: bool = False,
    patience: int = 3,
    max_splits_per_iter: int = 50000,
    smoothing_passes: int = 3,
    use_3d_delaunay: bool = False,
    delaunay_flip_polish: bool = False,
    max_flip_passes: int = 20,
    steiner_fix_gaussians: bool = False,
    steiner_max_iterations: int = 3,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Iteratively improve mesh quality by **smoothing** and **splitting**.

    A triangle is *bad* if its aspect ratio or minimum angle exceed
    the given thresholds.  Triangles that touch Gaussian vertices use
    **harsher** thresholds (``gauss_max_aspect_ratio`` /
    ``gauss_min_angle_deg``) so that the mesh near data points is
    forced to higher quality.

    Each iteration performs two complementary stages:

    **Stage 1 — Uniform-Laplacian smoothing**:
      Move every **non-Gaussian** vertex of bad triangles toward the
      unweighted average of its 1-ring neighbours in ``(x, y)``, then
      reproject ``z`` onto the surface.  Gaussian vertices are **never**
      moved.

    **Stage 2 — Longest-edge splitting**:
      For each remaining bad triangle — whether or not it touches a
      Gaussian — insert the midpoint of its *longest* edge (projected
      onto the surface).  A **full 2-D Delaunay retriangulation** is
      performed after each batch of insertions to guarantee a valid
      mesh with no holes.

    Parameters
    ----------
    vertices, faces : ndarrays
    surface_type : str
    max_aspect_ratio : float
        Default AR threshold for non-Gaussian triangles (default 5).
    min_angle_deg : float
        Default min-angle threshold for non-Gaussian triangles (default 10).
    gauss_max_aspect_ratio : float or None
        Harsher AR threshold for triangles touching a Gaussian vertex.
        ``None`` → ``max_aspect_ratio * 0.6`` (i.e. 3.0 when default).
    gauss_min_angle_deg : float or None
        Harsher min-angle threshold for Gaussian-touching triangles.
        ``None`` → ``min_angle_deg * 1.5`` (i.e. 15.0 when default).
    gauss_max_edge_length : float or None
        Max longest-edge threshold for triangles touching a Gaussian vertex.
        When set, Gaussian-touching triangles whose longest edge exceeds this
        value are also classified as bad.  ``None`` → no edge-length criterion
        for Gaussian triangles.
    gauss_max_area_factor : float or None
        Area-factor threshold (multiple of median area) for Gaussian-touching
        triangles.  ``None`` → same as ``max_area_factor``.
    warmup_iterations : int
        Number of initial iterations in which **Stage 2 (longest-edge splitting)
        is suppressed for non-Gaussian triangles**.  During warmup, only
        Gaussian-touching bad triangles are split; non-Gaussian bad triangles
        receive smoothing only.  Once ``iteration >= warmup_iterations`` the
        full split applies to all bad triangles.  Default: 0 (no warmup).
    ring_fix_invalid_gaussians : bool
        If ``True``, run ring-based support-point insertion around every
        Gaussian in a bad triangle before the main smoothing/splitting
        loop (Stage 0).  Default: ``False``.
    surface_aware : bool
        If ``True``, use the first fundamental form of the polynomial
        surface for metric-weighted Laplacian smoothing and geodesic
        midpoint splitting instead of flat parameter-space operations.
        Default: ``False``.
    patience : int
        Number of consecutive iterations without ≥5 % improvement
        before early stopping.  Default: 3.
    max_splits_per_iter : int
        Maximum number of edge midpoints to insert per iteration.
        Worst triangles (highest aspect ratio) are prioritised.
        Prevents mesh blowup when many triangles are bad.  Default: 50000.
    smoothing_passes : int
        Number of Laplacian smoothing passes per iteration before
        splitting.  Multiple passes let smoothing converge more before
        new points are inserted.  Default: 3.
    max_area_factor : float
    max_iterations : int
    max_edge_length : float or None
        ``None`` → auto (10× initial median edge).
    gaussian_vertex_indices : ndarray, optional
    use_3d_delaunay : bool
        If ``True``, use 3-D Delaunay for re-triangulations.
    verbose : bool

    Returns
    -------
    vertices, faces, gaussian_vertex_indices : ndarrays
    """
    vertices = np.array(vertices, dtype=np.float64)
    faces = np.array(faces, dtype=np.int32)

    # ── Gaussian protection mask ───────────────────────────────────────
    is_gaussian = np.zeros(len(vertices), dtype=bool)
    if gaussian_vertex_indices is not None and len(gaussian_vertex_indices) > 0:
        is_gaussian[gaussian_vertex_indices] = True

    # ── Harsher thresholds for Gaussian-touching triangles ─────────────
    if gauss_max_aspect_ratio is None:
        gauss_max_aspect_ratio = max_aspect_ratio * 0.6
    if gauss_min_angle_deg is None:
        gauss_min_angle_deg = min_angle_deg * 1.5

    # ── Auto-detect max_edge_length ────────────────────────────────────
    if max_edge_length is None:
        _v0 = vertices[faces[:, 0]]
        _v1 = vertices[faces[:, 1]]
        _v2 = vertices[faces[:, 2]]
        _all_edges = np.concatenate([
            np.linalg.norm(_v1 - _v0, axis=1),
            np.linalg.norm(_v2 - _v1, axis=1),
            np.linalg.norm(_v0 - _v2, axis=1),
        ])
        max_edge_length = float(10.0 * np.median(_all_edges))
        if verbose:
            print(f"        Auto max_edge_length = {max_edge_length:.6f}")

    min_spacing = max_edge_length * 0.01 if max_edge_length > 0 else 1e-10

    stall_count = 0
    prev_worst_ar = float("inf")
    initial_n_verts = len(vertices)
    # Hard ceiling: stop splitting when vertex count exceeds 3× initial.
    max_total_verts = initial_n_verts * 3

    # ── Persistent KDTree (rebuilt only when vertices are added) ──────
    kd_tree = KDTree(vertices[:, :2])
    kd_valid_size = len(vertices)

    def _bad_mask_with_gauss(ar, min_angle, faces_arr, longest_edge=None):
        """Compute per-face bad mask using harsher thresholds for
        triangles touching Gaussian vertices."""
        # Does each face touch a Gaussian vertex?
        touches_gauss = (
            is_gaussian[faces_arr[:, 0]]
            | is_gaussian[faces_arr[:, 1]]
            | is_gaussian[faces_arr[:, 2]]
        )
        # Harsher thresholds for Gaussian-touching triangles
        bad_gauss = touches_gauss & (
            (ar > gauss_max_aspect_ratio) | (min_angle < gauss_min_angle_deg)
        )
        # Optional edge-length criterion for Gaussian-touching triangles
        if gauss_max_edge_length is not None and longest_edge is not None:
            bad_gauss = bad_gauss | (
                touches_gauss & (longest_edge > gauss_max_edge_length)
            )
        # Normal thresholds for the rest
        bad_other = ~touches_gauss & (
            (ar > max_aspect_ratio) | (min_angle < min_angle_deg)
        )
        return bad_gauss | bad_other, touches_gauss

    # ── Stage 0: Ring-based fix for all invalid Gaussians ─────────────
    # Before smoothing/splitting, add support rings around every Gaussian
    # in a bad triangle.  Adjacent Gaussians share ring points via
    # spatial deduplication.
    if ring_fix_invalid_gaussians and max_iterations > 0:
        vertices, faces, is_gaussian = _fix_invalid_gaussians_with_rings(
            vertices, faces, is_gaussian, max_edge_length, surface_type,
            gauss_ar_threshold=gauss_max_aspect_ratio,
            gauss_angle_threshold=gauss_min_angle_deg,
            gauss_edge_threshold=gauss_max_edge_length,
            max_ring_steps=ring_fix_iterations,
            verbose=verbose,
        )
        # Rebuild cached structures after ring insertion
        kd_tree = KDTree(vertices[:, :2])
        kd_valid_size = len(vertices)

    import time as _time

    for iteration in range(max_iterations):
        _t_iter = _time.monotonic()
        ar, min_angle, longest_edge = _triangle_quality(vertices, faces)

        # ── Combine ALL bad-triangle criteria ──────────────────────────
        bad_mask, touches_gauss = _bad_mask_with_gauss(
            ar, min_angle, faces, longest_edge,
        )
        # Include large-area triangles
        xy0 = vertices[faces[:, 0], :2]
        xy1 = vertices[faces[:, 1], :2]
        xy2 = vertices[faces[:, 2], :2]
        areas_2d = 0.5 * np.abs(
            (xy1[:, 0] - xy0[:, 0]) * (xy2[:, 1] - xy0[:, 1])
            - (xy2[:, 0] - xy0[:, 0]) * (xy1[:, 1] - xy0[:, 1])
        )
        median_area = float(np.median(areas_2d[areas_2d > 0])) if (areas_2d > 0).any() else 0.0
        if median_area > 0:
            eff_gauss_area_factor = (
                gauss_max_area_factor
                if gauss_max_area_factor is not None
                else max_area_factor
            )
            large_mask = (
                (~touches_gauss & (areas_2d > median_area * max_area_factor))
                | (touches_gauss & (areas_2d > median_area * eff_gauss_area_factor))
            )
            bad_mask = bad_mask | large_mask

        # During warmup: restrict to Gaussian-touching only
        if iteration < warmup_iterations:
            bad_mask = bad_mask & touches_gauss
            if verbose:
                print(
                    f"          [warmup iter {iteration + 1}/{warmup_iterations}] "
                    f"splitting Gaussian-touching triangles only"
                )

        n_bad = int(bad_mask.sum())

        if n_bad == 0:
            if verbose:
                print(f"        Refinement iter {iteration}: 0 bad — done.")
            break

        # ── Patience: stop when worst AR stops decreasing ──────────────
        worst_ar = float(ar.max())
        if worst_ar < prev_worst_ar * 0.95:
            stall_count = 0
        else:
            stall_count += 1
        prev_worst_ar = min(prev_worst_ar, worst_ar)

        if stall_count >= patience:
            if verbose:
                print(
                    f"        Refinement iter {iteration}: {n_bad} bad "
                    f"(worst AR={worst_ar:.2f}, stalled {stall_count}×, "
                    f"patience={patience}) — stopping."
                )
            break

        if verbose:
            print(
                f"        Refinement iter {iteration}: {n_bad} bad "
                f"(worst AR={ar.max():.2f}, min angle={min_angle.min():.2f}°)"
            )

        # ── Stage 1: Smooth ALL non-Gaussian vertices globally ─────────
        # Let the entire mesh relax so grid vertices can redistribute
        # to accommodate inserted Gaussians.
        movable = ~is_gaussian

        n_bad_before_smooth = n_bad
        if int(movable.sum()) > 0:
            _w_cache = None
            for _sp in range(smoothing_passes):
                if surface_aware:
                    _, _w_cache = _metric_weighted_laplacian_smooth(
                        vertices, faces, movable, surface_type,
                        damping=0.3, _W_cache=_w_cache,
                    )
                else:
                    _, _w_cache = _uniform_laplacian_smooth(
                        vertices, faces, movable, surface_type,
                        damping=0.3, _adj_cache=_w_cache,
                    )

            # Re-evaluate quality after smoothing
            ar, min_angle, longest_edge = _triangle_quality(vertices, faces)
            bad_mask, touches_gauss = _bad_mask_with_gauss(
                ar, min_angle, faces, longest_edge,
            )
            # Re-apply area and warmup criteria
            if median_area > 0:
                areas_2d = 0.5 * np.abs(
                    (vertices[faces[:, 0], 0] - vertices[faces[:, 2], 0])
                    * (vertices[faces[:, 1], 1] - vertices[faces[:, 2], 1])
                    - (vertices[faces[:, 1], 0] - vertices[faces[:, 2], 0])
                    * (vertices[faces[:, 0], 1] - vertices[faces[:, 2], 1])
                )
                large_mask = (
                    (~touches_gauss & (areas_2d > median_area * max_area_factor))
                    | (touches_gauss & (areas_2d > median_area * eff_gauss_area_factor))
                )
                bad_mask = bad_mask | large_mask
            if iteration < warmup_iterations:
                bad_mask = bad_mask & touches_gauss

            n_bad = int(bad_mask.sum())
            if verbose:
                print(
                    f"          [smooth] {n_bad_before_smooth} → {n_bad} bad "
                    f"after {smoothing_passes} passes"
                )

        if n_bad == 0:
            if verbose:
                print(f"          → all fixed by smoothing — done.")
            break

        # ── Stage 2: Split only triangles still bad after smoothing ────
        remaining_budget = max_total_verts - len(vertices)
        if remaining_budget <= 0:
            if verbose:
                print(
                    f"          → vertex budget exhausted "
                    f"({len(vertices)}/{max_total_verts}) — stopping."
                )
            break

        bad_idx = np.where(bad_mask)[0]

        # Prioritise shape-bad triangles (AR/angle) over edge-length-only.
        shape_bad = (ar[bad_idx] > gauss_max_aspect_ratio) | (
            min_angle[bad_idx] < gauss_min_angle_deg
        )
        shape_first = np.concatenate([
            bad_idx[shape_bad],
            bad_idx[~shape_bad],
        ])
        bad_idx = shape_first

        # Cap by both max_splits_per_iter and remaining vertex budget.
        effective_cap = min(max_splits_per_iter, remaining_budget)
        if len(bad_idx) > effective_cap:
            bad_idx = bad_idx[:effective_cap]

        bad_f = faces[bad_idx]
        bv0 = vertices[bad_f[:, 0]]
        bv1 = vertices[bad_f[:, 1]]
        bv2 = vertices[bad_f[:, 2]]

        e_lens = np.column_stack([
            np.linalg.norm(bv1 - bv0, axis=1),
            np.linalg.norm(bv2 - bv1, axis=1),
            np.linalg.norm(bv0 - bv2, axis=1),
        ])
        long_idx = np.argmax(e_lens, axis=1)

        edge_v = np.array([[0, 1], [1, 2], [2, 0]])
        n_bad_f = len(bad_f)
        va_idx = bad_f[np.arange(n_bad_f), edge_v[long_idx, 0]]
        vb_idx = bad_f[np.arange(n_bad_f), edge_v[long_idx, 1]]

        if surface_aware:
            midpoints_xy = _surface_metric_midpoints(
                vertices[va_idx, :2], vertices[vb_idx, :2], surface_type,
            )
        else:
            midpoints_xy = (vertices[va_idx, :2] + vertices[vb_idx, :2]) / 2.0

        # Deduplicate by edge key (sorted vertex pair) to preserve
        # edge→midpoint correspondence for local splitting.
        edge_min = np.minimum(va_idx, vb_idx)
        edge_max = np.maximum(va_idx, vb_idx)
        edge_keys = edge_min.astype(np.int64) * (len(vertices) + len(midpoints_xy)) + edge_max.astype(np.int64)
        _, unique_idx = np.unique(edge_keys, return_index=True)
        unique_idx = np.sort(unique_idx)
        va_idx = va_idx[unique_idx]
        vb_idx = vb_idx[unique_idx]
        midpoints_xy = midpoints_xy[unique_idx]

        # Deduplicate against existing vertices
        if len(midpoints_xy) > 0:
            if kd_valid_size != len(vertices):
                kd_tree = KDTree(vertices[:, :2])
                kd_valid_size = len(vertices)
            dd, _ = kd_tree.query(midpoints_xy)
            far_enough = dd > min_spacing
            va_idx = va_idx[far_enough]
            vb_idx = vb_idx[far_enough]
            midpoints_xy = midpoints_xy[far_enough]

        if len(midpoints_xy) == 0:
            if verbose:
                print(f"          → no midpoints to insert")
            continue

        # Project midpoints onto surface
        mid_z = evaluate_polynomial(midpoints_xy[:, 0], midpoints_xy[:, 1], surface_type)
        new_pts = np.column_stack([midpoints_xy, mid_z])

        # Vertex indices for the new midpoints
        base_idx = len(vertices)
        mid_indices = np.arange(base_idx, base_idx + len(new_pts), dtype=np.int64)

        if verbose:
            print(
                f"          → inserting {len(new_pts)} edge midpoints"
            )

        # ── Vectorised face splitting (no flips needed) ──────────────
        vertices = np.vstack([vertices, new_pts])
        xy = vertices[:, :2]
        faces = _split_faces_with_midpoints(
            xy, faces, va_idx, vb_idx, mid_indices,
        )

        # Extend is_gaussian mask for new vertices
        n_new = len(new_pts)
        is_gaussian = np.concatenate([is_gaussian, np.zeros(n_new, dtype=bool)])
        kd_valid_size = -1  # invalidate KDTree
        _dt_iter = _time.monotonic() - _t_iter

        if verbose:
            print(
                f"          → {len(vertices)} verts, {len(faces)} faces  "
                f"[{_dt_iter:.1f}s]"
            )

    # ── Post-loop smoothing polish ─────────────────────────────────────
    # A single smoothing pass on the final split state to polish quality
    # without the expense of per-iteration smoothing.
    if smoothing_passes > 0 and max_iterations > 0:
        ar_pre, ma_pre, le_pre = _triangle_quality(vertices, faces)
        bad_pre, _ = _bad_mask_with_gauss(ar_pre, ma_pre, faces, le_pre)
        n_bad_pre = int(bad_pre.sum())
        if n_bad_pre > 0:
            bad_verts = np.unique(faces[np.where(bad_pre)[0]].ravel())
            movable = np.zeros(len(vertices), dtype=bool)
            movable[bad_verts] = True
            movable &= ~is_gaussian

            if int(movable.sum()) > 0:
                saved_verts = vertices.copy()
                _w_cache = None
                for _sp in range(smoothing_passes):
                    if surface_aware:
                        _, _w_cache = _metric_weighted_laplacian_smooth(
                            vertices, faces, movable, surface_type,
                            damping=0.3, _W_cache=_w_cache,
                        )
                    else:
                        _, _w_cache = _uniform_laplacian_smooth(
                            vertices, faces, movable, surface_type,
                            damping=0.3, _adj_cache=_w_cache,
                        )
                # Rollback if smoothing made things worse
                ar_post, ma_post, le_post = _triangle_quality(vertices, faces)
                bad_post, _ = _bad_mask_with_gauss(ar_post, ma_post, faces, le_post)
                n_bad_post = int(bad_post.sum())
                if n_bad_post > n_bad_pre * 1.1:
                    vertices = saved_verts
                    if verbose:
                        print(
                            f"        [polish] rolled back: {n_bad_pre} → "
                            f"{n_bad_post} bad after smoothing"
                        )
                elif verbose:
                    print(
                        f"        [polish] smoothing: {n_bad_pre} → "
                        f"{n_bad_post} bad"
                    )

    gaussian_vertex_indices = np.where(is_gaussian)[0].astype(np.int32)

    # ── Circumcenter Steiner insertion for stubborn Gaussian triangles ─
    if steiner_fix_gaussians and gauss_max_aspect_ratio is not None:
        # Use the general quality thresholds (not the permissive Gaussian-
        # specific ones) so the Steiner pass targets the same triangles
        # the user considers "bad".
        steiner_ar = min(max_aspect_ratio, 2.0)
        steiner_angle = max(min_angle_deg, 20.0)
        vertices, faces, is_gaussian = _circumcenter_steiner_pass(
            vertices, faces, is_gaussian,
            max_edge_length=max_edge_length,
            surface_type=surface_type,
            gauss_ar_threshold=steiner_ar,
            gauss_angle_threshold=steiner_angle,
            max_iterations=steiner_max_iterations,
            verbose=verbose,
        )
        gaussian_vertex_indices = np.where(is_gaussian)[0].astype(np.int32)

    # ── Delaunay flip polish pass ─────────────────────────────────────
    if delaunay_flip_polish:
        xy = vertices[:, :2].copy()
        faces_i64 = faces.astype(np.int64)
        total_flips = 0
        for _fp in range(max_flip_passes):
            n_flips = _lawson_flip_pass(xy, faces_i64)
            total_flips += n_flips
            if n_flips == 0:
                break
        faces = faces_i64.astype(np.int32)
        if verbose:
            print(
                f"        [flip polish] {total_flips} flips in "
                f"{_fp + 1} passes"
            )

    if verbose:
        ar_final, ma_final, _ = _triangle_quality(vertices, faces)
        print(
            f"        Refinement complete: {len(vertices)} verts, "
            f"{len(faces)} faces, "
            f"worst AR={ar_final.max():.2f}, "
            f"min angle={ma_final.min():.2f}°, "
            f"{len(gaussian_vertex_indices)} Gaussian verts retained"
        )

    return vertices, faces, gaussian_vertex_indices



# ── Save / load geodesic mesh ───────────────────────────────────────────────

def save_geodesic_mesh(
    output_dir: Path,
    vertices: np.ndarray,
    faces: np.ndarray,
    gaussian_vertex_indices: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save the geodesic mesh and Gaussian-to-vertex mapping.

    Saves two files under *output_dir*:
    * ``geodesic_mesh.ply`` — the triangulated mesh (trimesh export).
    * ``geodesic_mesh_data.npz`` — vertex array, face array,
      ``gaussian_vertex_indices`` (mapping Gaussian index → mesh vertex
      index), and optional metadata values.

    Parameters
    ----------
    output_dir : Path
        Directory to write into (created if missing).
    vertices, faces : ndarrays
        Mesh geometry.
    gaussian_vertex_indices : ndarray, shape ``(N_gaussians,)``
        For each original Gaussian, the index of its corresponding mesh
        vertex (projected position).  The first ``N_gaussians`` vertices
        in the mesh are guaranteed to be the projected Gaussians.
    metadata : dict, optional
        Extra key-value pairs to store in the ``.npz``.

    Returns
    -------
    npz_path : Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PLY mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ply_path = output_dir / "geodesic_mesh.ply"
    mesh.export(str(ply_path))

    # Save NPZ with mapping data
    npz_path = output_dir / "geodesic_mesh_data.npz"
    save_dict = dict(
        vertices=vertices,
        faces=faces,
        gaussian_vertex_indices=gaussian_vertex_indices,
    )
    if metadata:
        for k, v in metadata.items():
            save_dict[k] = v
    np.savez_compressed(str(npz_path), **save_dict)

    return npz_path


def load_geodesic_mesh(
    mesh_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a previously saved geodesic mesh.

    Parameters
    ----------
    mesh_dir : Path
        Directory containing ``geodesic_mesh_data.npz``.

    Returns
    -------
    vertices : ndarray, shape ``(V, 3)``
    faces : ndarray, shape ``(F, 3)``
    gaussian_vertex_indices : ndarray, shape ``(N_gaussians,)``
    """
    npz_path = Path(mesh_dir) / "geodesic_mesh_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Geodesic mesh data not found: {npz_path}")

    data = np.load(str(npz_path))
    return data["vertices"], data["faces"], data["gaussian_vertex_indices"]
