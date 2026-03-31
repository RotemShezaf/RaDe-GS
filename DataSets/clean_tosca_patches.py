#!/usr/bin/env python3
"""
Post-hoc cleaning of TOSCA training patches.

Addresses two issues found during dataset verification:
1. Target outliers: examples with p_u far above the distribution tail
2. Fold-outlier neighbors: neighbors with geo/euc gradient indicating
   they come from a different surface fold

Actions:
- Removes examples with target > cap (default: per-file p99.5)
- For remaining examples, masks neighbors with geo/euc ratio > threshold
  (sets their geodesic column to MASK_CONSTANT)
- Saves cleaned files alongside originals (*_clean.npy)

Usage:
    python DataSets/clean_tosca_patches.py
    python DataSets/clean_tosca_patches.py --target-cap 40 --geo-euc-cap 50
    python DataSets/clean_tosca_patches.py --dry-run
    python DataSets/clean_tosca_patches.py --inplace  # overwrite originals
"""
import argparse
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

DATASETS_BASE = project_root / "TrainData" / "datasets" / "gaussian_patches"
MASK_CONSTANT = -10.0
TOSCA_ANIMALS = ["cat", "centaur", "david", "dog", "gorilla", "horse",
                 "michael", "victoria", "wolf"]
MAX_NBRS = 192
FEAT_PER_NBR = 4  # xyz(3) + geodesic(1)
EXPECTED_COLS = MAX_NBRS * FEAT_PER_NBR + 3 + 1 + 1  # 773


def clean_file(npy_path, target_cap_pct=99.5, target_cap_abs=None,
               geo_euc_cap=50.0, dry_run=False, inplace=False):
    """Clean a single .npy dataset file.

    Returns a stats dict.
    """
    data = np.load(str(npy_path))
    n_orig = data.shape[0]

    if data.shape[1] != EXPECTED_COLS:
        return {'file': npy_path.name, 'skipped': True,
                'reason': f'unexpected shape {data.shape}'}

    # Parse layout
    nbr_block = data[:, :MAX_NBRS * FEAT_PER_NBR].reshape(n_orig, MAX_NBRS, FEAT_PER_NBR)
    targets = data[:, -1]

    # -- Step 1: Remove examples with extreme targets --
    if target_cap_abs is not None:
        cap = target_cap_abs
    else:
        cap = np.percentile(targets, target_cap_pct)

    keep_mask = targets <= cap
    n_target_removed = int((~keep_mask).sum())

    # -- Step 2: Mask fold-outlier neighbors (geo/euc > threshold) --
    # Work on the kept examples
    data_clean = data[keep_mask].copy()
    nbr_clean = data_clean[:, :MAX_NBRS * FEAT_PER_NBR].reshape(
        data_clean.shape[0], MAX_NBRS, FEAT_PER_NBR)

    nbr_xyz = nbr_clean[:, :, :3]
    nbr_geo = nbr_clean[:, :, 3]
    is_already_masked = (nbr_geo == MASK_CONSTANT)

    euc_dist = np.linalg.norm(nbr_xyz, axis=2)  # (N, MAX_NBRS)
    # Only check real (non-masked) neighbors
    with np.errstate(divide='ignore', invalid='ignore'):
        geo_euc_ratio = np.abs(nbr_geo) / np.maximum(euc_dist, 1e-8)
    geo_euc_ratio[is_already_masked] = 0.0

    fold_outlier_mask = geo_euc_ratio > geo_euc_cap  # (N, MAX_NBRS)
    n_nbrs_masked = int(fold_outlier_mask.sum())
    n_examples_affected = int((fold_outlier_mask.any(axis=1)).sum())

    # Mask the fold-outlier neighbors' geodesic
    nbr_clean[:, :, 3][fold_outlier_mask] = MASK_CONSTANT
    # Write back to flat array
    data_clean[:, :MAX_NBRS * FEAT_PER_NBR] = nbr_clean.reshape(
        data_clean.shape[0], MAX_NBRS * FEAT_PER_NBR)

    n_final = data_clean.shape[0]

    stats = {
        'file': npy_path.name,
        'n_orig': n_orig,
        'n_final': n_final,
        'target_cap': float(cap),
        'n_target_removed': n_target_removed,
        'pct_target_removed': n_target_removed / n_orig * 100,
        'n_fold_nbrs_masked': n_nbrs_masked,
        'n_fold_examples_affected': n_examples_affected,
    }

    if not dry_run:
        if inplace:
            out_path = npy_path
        else:
            out_path = npy_path.with_name(
                npy_path.stem.rsplit('_n', 1)[0] + f'_n{n_final}_clean.npy')
        np.save(str(out_path), data_clean)
        stats['output_path'] = str(out_path)
        stats['output_size_mb'] = data_clean.nbytes / 1024 / 1024

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean TOSCA training patches")
    parser.add_argument('--target-cap-pct', type=float, default=99.5,
                        help='Percentile for target cap (default: 99.5)')
    parser.add_argument('--target-cap', type=float, default=None,
                        help='Absolute target cap (overrides percentile)')
    parser.add_argument('--geo-euc-cap', type=float, default=50.0,
                        help='Max geo/euc ratio before masking neighbor (default: 50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would be done without saving')
    parser.add_argument('--inplace', action='store_true',
                        help='Overwrite original files instead of creating _clean.npy')
    parser.add_argument('--animal', default=None,
                        help='Process single animal (default: all)')
    args = parser.parse_args()

    animals = [args.animal] if args.animal else TOSCA_ANIMALS

    print(f"\n{'#'*60}")
    print(f"# TOSCA Dataset Cleaning")
    print(f"# target_cap: {args.target_cap or f'p{args.target_cap_pct}'}")
    print(f"# geo_euc_cap: {args.geo_euc_cap}")
    print(f"# {'DRY RUN' if args.dry_run else 'WRITING' + (' (inplace)' if args.inplace else ' (_clean.npy)')}")
    print(f"{'#'*60}\n")

    all_stats = []

    for animal in animals:
        dataset_dir = DATASETS_BASE / f"tosca_{animal}"
        npy_files = sorted(dataset_dir.glob("gaussian_examples_ring3_*.npy"))
        # Skip existing _clean files
        npy_files = [f for f in npy_files if '_clean' not in f.name]

        for npy_file in npy_files:
            data_test = np.load(str(npy_file), mmap_mode='r')
            if data_test.shape[1] != EXPECTED_COLS:
                print(f"  Skipping {npy_file.name} (shape {data_test.shape})")
                continue

            print(f"  Processing {animal}/{npy_file.name}...")
            stats = clean_file(
                npy_file,
                target_cap_pct=args.target_cap_pct,
                target_cap_abs=args.target_cap,
                geo_euc_cap=args.geo_euc_cap,
                dry_run=args.dry_run,
                inplace=args.inplace,
            )
            all_stats.append(stats)

            print(f"    Target cap={stats['target_cap']:.2f}: "
                  f"removed {stats['n_target_removed']} "
                  f"({stats['pct_target_removed']:.3f}%)")
            print(f"    Fold-outlier neighbors masked: "
                  f"{stats['n_fold_nbrs_masked']} in "
                  f"{stats['n_fold_examples_affected']} examples")
            print(f"    {stats['n_orig']:,} -> {stats['n_final']:,} examples")
            if 'output_path' in stats:
                print(f"    Saved: {stats['output_path']} "
                      f"({stats['output_size_mb']:.1f} MB)")

    # Summary
    if all_stats:
        total_orig = sum(s['n_orig'] for s in all_stats)
        total_final = sum(s['n_final'] for s in all_stats)
        total_target_rm = sum(s['n_target_removed'] for s in all_stats)
        total_nbr_masked = sum(s['n_fold_nbrs_masked'] for s in all_stats)
        total_affected = sum(s['n_fold_examples_affected'] for s in all_stats)

        print(f"\n{'='*60}")
        print(f"  SUMMARY")
        print(f"{'='*60}")
        print(f"  Files processed: {len(all_stats)}")
        print(f"  Examples: {total_orig:,} -> {total_final:,} "
              f"(removed {total_target_rm:,}, {total_target_rm/total_orig*100:.3f}%)")
        print(f"  Fold-outlier neighbors masked: {total_nbr_masked:,} "
              f"in {total_affected:,} examples")
        print()


if __name__ == '__main__':
    sys.exit(main() or 0)
