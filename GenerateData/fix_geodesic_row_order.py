#!/usr/bin/env python3
"""
Fix the row-ordering bug in gt_geodesic.npz files.

The bug: transfer_geodesic_to_gaussians() sorted the geodesic_distances rows by
argsort(source_gaussian_indices), but saved source_indices, source_positions, and
source_gaussian_indices in their ORIGINAL order. Then merge_partial_results sorted
everything again by source_gaussian_indices, double-sorting the distances.

This script:
1. Finds all partial result files (sources_range_*.npz)
2. Un-sorts the geodesic_distances rows to match the original source ordering
3. Re-merges partial results into a corrected gt_geodesic.npz

Usage:
    python GenerateData/fix_geodesic_row_order.py [--dry-run] [--verify-only]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def fix_partial_file(partial_path: Path, dry_run: bool = False) -> bool:
    """
    Fix row ordering in a single partial file.

    The bug sorted rows by argsort(source_gaussian_indices) inside
    transfer_geodesic_to_gaussians, but metadata was kept in original order.

    Strategy: try all candidate permutations and pick the one that makes
    self-distances (geodesic[i, sgi[i]]) closest to zero.

    Returns True if the file was (or would be) modified.
    """
    data = np.load(partial_path, allow_pickle=True)
    source_gaussian_indices = data['source_gaussian_indices']
    geodesic_distances = data['geodesic_distances']
    n = len(source_gaussian_indices)

    sort_order = np.argsort(source_gaussian_indices)
    inverse_sort = np.argsort(sort_order)

    # If already in order (sort_order == identity), nothing to fix
    if np.array_equal(sort_order, np.arange(n)):
        return False

    # Candidate permutations: identity (already correct), forward sort, inverse sort
    candidates = {
        'identity': np.arange(n),
        'forward_sort': sort_order,
        'inverse_sort': inverse_sort,
    }

    best_name = None
    best_perm = None
    best_max_self = float('inf')

    for name, perm in candidates.items():
        reordered = geodesic_distances[perm, :]
        self_dists = np.array([reordered[i, source_gaussian_indices[i]]
                               for i in range(n)])
        max_self = self_dists.max()
        if max_self < best_max_self:
            best_max_self = max_self
            best_name = name
            best_perm = perm

    if best_name == 'identity':
        return False  # Already correct

    fixed_geodesic = geodesic_distances[best_perm, :]

    if not dry_run:
        np.savez_compressed(
            partial_path,
            gaussian_positions=data['gaussian_positions'],
            source_indices=data['source_indices'],
            source_positions=data['source_positions'],
            geodesic_distances=fixed_geodesic,
            closest_mesh_indices=data['closest_mesh_indices'],
            closest_mesh_distances=data['closest_mesh_distances'],
            source_gaussian_indices=data['source_gaussian_indices'],
        )

    return True


def merge_fixed_partials(output_folder: Path, dry_run: bool = False):
    """
    Re-merge partial files into gt_geodesic.npz with consistent ordering.
    (Same logic as merge_partial_results but on already-fixed partials.)
    """
    partial_dir = output_folder / "geodesic_distance" / "gt_partial"
    partial_files = sorted(partial_dir.glob("sources_range_*.npz"))
    if not partial_files:
        return

    all_source_indices = []
    all_source_positions = []
    all_geodesic_distances = []
    all_source_gaussian_indices = []
    gaussian_positions = None
    closest_mesh_indices = None
    closest_mesh_distances = None

    for pf in partial_files:
        d = np.load(pf)
        if gaussian_positions is None:
            gaussian_positions = d['gaussian_positions']
            closest_mesh_indices = d['closest_mesh_indices']
            closest_mesh_distances = d['closest_mesh_distances']
        all_source_indices.append(d['source_indices'])
        all_source_positions.append(d['source_positions'])
        all_geodesic_distances.append(d['geodesic_distances'])
        all_source_gaussian_indices.append(d['source_gaussian_indices'])

    merged_si = np.concatenate(all_source_indices)
    merged_sp = np.concatenate(all_source_positions)
    merged_gd = np.concatenate(all_geodesic_distances)
    merged_sgi = np.concatenate(all_source_gaussian_indices)

    # Sort ALL arrays consistently by source_gaussian_indices
    order = np.argsort(merged_sgi)
    merged_si = merged_si[order]
    merged_sp = merged_sp[order]
    merged_gd = merged_gd[order]
    merged_sgi = merged_sgi[order]

    output_path = output_folder / "geodesic_distance" / "gt_geodesic.npz"

    if not dry_run:
        np.savez_compressed(
            output_path,
            gaussian_positions=gaussian_positions,
            source_indices=merged_si,
            source_positions=merged_sp,
            geodesic_distances=merged_gd,
            closest_mesh_indices=closest_mesh_indices,
            closest_mesh_distances=closest_mesh_distances,
            source_gaussian_indices=merged_sgi,
        )

    return merged_sgi, merged_gd


def verify_geodesic(gt_path: Path) -> dict:
    """Verify a gt_geodesic.npz has near-zero self-distances."""
    d = np.load(gt_path)
    sgi = d['source_gaussian_indices']
    gd = d['geodesic_distances']

    self_dists = np.array([gd[i, sgi[i]] for i in range(len(sgi))])
    return {
        'path': str(gt_path),
        'n_sources': len(sgi),
        'self_dist_mean': float(self_dists.mean()),
        'self_dist_max': float(self_dists.max()),
        'self_dist_min': float(self_dists.min()),
        'ok': self_dists.max() < 0.1,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be changed without writing")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing gt_geodesic.npz files")
    parser.add_argument("--base-dir", type=str,
                        default="TrainData/Polynomial/SyntheticColmapData",
                        help="Base directory to search for geodesic data")
    args = parser.parse_args()

    base = Path(args.base_dir)

    if args.verify_only:
        gt_files = sorted(base.rglob("gt_geodesic.npz"))
        print(f"Found {len(gt_files)} gt_geodesic.npz files\n")
        for f in gt_files:
            r = verify_geodesic(f)
            status = "OK" if r['ok'] else "BAD"
            print(f"[{status}] {r['path']}")
            print(f"       self-dist: mean={r['self_dist_mean']:.6f}  "
                  f"max={r['self_dist_max']:.6f}  min={r['self_dist_min']:.6f}")
        return

    # Find all output folders that have gt_partial directories
    partial_dirs = sorted(base.rglob("gt_partial"))
    print(f"Found {len(partial_dirs)} output folders with partial results\n")

    fixed_count = 0
    for pdir in tqdm(partial_dirs, desc="Fixing"):
        output_folder = pdir.parent.parent  # geodesic_distance -> output
        partial_files = sorted(pdir.glob("sources_range_*.npz"))
        if not partial_files:
            continue

        batch_fixed = False
        for pf in partial_files:
            if fix_partial_file(pf, dry_run=args.dry_run):
                batch_fixed = True

        if batch_fixed or True:  # Always re-merge to ensure consistency
            merge_fixed_partials(output_folder, dry_run=args.dry_run)
            fixed_count += 1

        # Verify
        gt_path = output_folder / "geodesic_distance" / "gt_geodesic.npz"
        if gt_path.exists() and not args.dry_run:
            r = verify_geodesic(gt_path)
            status = "OK" if r['ok'] else "BAD"
            tqdm.write(f"  [{status}] {r['path']}  self-dist max={r['self_dist_max']:.6f}")

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {fixed_count} datasets")


if __name__ == "__main__":
    main()
