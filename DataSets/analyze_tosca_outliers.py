#!/usr/bin/env python3
"""
Deep analysis of TOSCA dataset outliers and target distributions.

Reports:
1. Target (p_u) distribution with percentile analysis
2. Fold-outlier neighbor analysis (geo/euc gradient consistency)
3. Target vs neighbor-geodesic relationship ("does the target differ from surroundings?")
4. Quantifies how many examples would be affected by a target cap

Usage:
    python DataSets/analyze_tosca_outliers.py
    python DataSets/analyze_tosca_outliers.py --animal centaur
    python DataSets/analyze_tosca_outliers.py --target-cap 50
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


def analyze_animal(animal, target_cap=None, verbose=False):
    """Deeply analyze ring-3 dataset for one animal."""
    dataset_dir = DATASETS_BASE / f"tosca_{animal}"
    # Find the ring3 file with 192-wide layout (feat_per_nbr=4 → 773 cols)
    npy_files = sorted(dataset_dir.glob("gaussian_examples_ring3_*.npy"))
    target_file = None
    for f in npy_files:
        data_test = np.load(str(f), mmap_mode='r')
        if data_test.shape[1] == MAX_NBRS * FEAT_PER_NBR + 3 + 1 + 1:
            # 192*4 + 3 (point_feat xyz) + 1 (r1_min) + 1 (target) = 773
            target_file = f
            break

    if target_file is None:
        print(f"  {animal}: No ring3 file with expected layout found")
        return None

    data = np.load(str(target_file))
    n_examples = data.shape[0]

    # Parse layout
    nbr_block = data[:, :MAX_NBRS * FEAT_PER_NBR].reshape(n_examples, MAX_NBRS, FEAT_PER_NBR)
    point_features = data[:, MAX_NBRS * FEAT_PER_NBR:MAX_NBRS * FEAT_PER_NBR + 3]
    r1_min_vals = data[:, -2]
    targets = data[:, -1]

    nbr_xyz = nbr_block[:, :, :3]
    nbr_geo = nbr_block[:, :, 3]
    is_masked = (nbr_geo == MASK_CONSTANT)
    real_per_example = MAX_NBRS - np.sum(is_masked, axis=1)

    print(f"\n{'='*60}")
    print(f"  {animal.upper()}  ({n_examples:,} examples, file: {target_file.name})")
    print(f"{'='*60}")

    # -- 1. TARGET DISTRIBUTION --
    print(f"\n  1. TARGET (p_u) DISTRIBUTION")
    print(f"     mean={targets.mean():.4f}, std={targets.std():.4f}")
    print(f"     min={targets.min():.4f}, max={targets.max():.4f}")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]:
        print(f"     p{p}={np.percentile(targets, p):.4f}")

    # Show extreme targets
    extreme_thresholds = [50, 100, 200, 500]
    for t in extreme_thresholds:
        count = int(np.sum(targets > t))
        if count > 0:
            pct = count / n_examples * 100
            print(f"     targets > {t}: {count} ({pct:.4f}%)")

    # -- 2. TARGET vs NEIGHBOR GEODESIC RELATIONSHIP --
    print(f"\n  2. TARGET vs NEIGHBOR GEODESIC RELATIONSHIP")
    # For each example, compare target to neighbor geodesics
    # On same surface sheet, neighbors closer to source should have lower geo,
    # and the target should sit "among" the neighbor geodesics.

    # Subsample for speed
    rng = np.random.RandomState(42)
    sample_n = min(n_examples, 50000)
    si = rng.choice(n_examples, sample_n, replace=False) if sample_n < n_examples else np.arange(n_examples)

    s_targets = targets[si]
    s_nbr_geo = nbr_geo[si]
    s_is_masked = is_masked[si]
    s_real_per = real_per_example[si]
    s_nbr_xyz = nbr_xyz[si]

    # Per-example stats
    target_rank_pcts = []  # Where does target fall among its neighbor geodesics?
    nbr_mean_geos = []
    nbr_max_geos = []
    target_minus_nbrmax = []
    target_over_nbrmean = []
    outlier_examples_geo_euc = []  # examples with geo/euc > 50

    for i in range(sample_n):
        real = ~s_is_masked[i]
        n_real = int(real.sum())
        if n_real < 3:
            continue

        geos = s_nbr_geo[i, real]
        xyzs = s_nbr_xyz[i, real]
        target = s_targets[i]

        nbr_mean = geos.mean()
        nbr_max = geos.max()
        nbr_mean_geos.append(nbr_mean)
        nbr_max_geos.append(nbr_max)

        # Where does target rank among neighbor geodesics?
        rank_pct = float(np.sum(geos < target)) / n_real * 100
        target_rank_pcts.append(rank_pct)

        target_minus_nbrmax.append(target - nbr_max)
        if nbr_mean > 0:
            target_over_nbrmean.append(target / nbr_mean)

        # Check for high geo/euc neighbors
        euc_dists = np.linalg.norm(xyzs, axis=1)
        ratios = geos / np.maximum(euc_dists, 1e-8)
        if ratios.max() > 50:
            outlier_examples_geo_euc.append({
                'idx': int(si[i]),
                'target': float(target),
                'max_ratio': float(ratios.max()),
                'n_gt50': int(np.sum(ratios > 50)),
                'nbr_mean_geo': float(nbr_mean),
            })

    target_rank_pcts = np.array(target_rank_pcts)
    nbr_mean_geos = np.array(nbr_mean_geos)
    target_over_nbrmean = np.array(target_over_nbrmean)
    target_minus_nbrmax = np.array(target_minus_nbrmax)

    print(f"     Target rank among neighbors (pct of nbrs with geo < target):")
    print(f"       mean={target_rank_pcts.mean():.1f}%, p50={np.median(target_rank_pcts):.1f}%")
    print(f"       p5={np.percentile(target_rank_pcts, 5):.1f}%, "
          f"p95={np.percentile(target_rank_pcts, 95):.1f}%")
    print(f"       (Expected: target should be near TOP of neighbor geodesics, i.e. ~80-100%)")

    print(f"\n     target / mean_nbr_geo:")
    print(f"       mean={target_over_nbrmean.mean():.4f}, "
          f"p50={np.median(target_over_nbrmean):.4f}, "
          f"p5={np.percentile(target_over_nbrmean, 5):.4f}, "
          f"p95={np.percentile(target_over_nbrmean, 95):.4f}")

    print(f"\n     target - max_nbr_geo:")
    print(f"       mean={target_minus_nbrmax.mean():.4f}, "
          f"min={target_minus_nbrmax.min():.4f}, "
          f"p50={np.median(target_minus_nbrmax):.4f}")
    print(f"       (Should always be >= 0: neighbor geo <= p_u by construction)")
    neg_count = int(np.sum(target_minus_nbrmax < -1e-6))
    if neg_count > 0:
        print(f"       *** WARNING: {neg_count} examples have nbr_geo > target! ***")

    # -- 3. FOLD OUTLIER ANALYSIS (detailed) --
    print(f"\n  3. FOLD-OUTLIER NEIGHBOR ANALYSIS (geo/euc gradient)")
    n_outlier_examples = len(outlier_examples_geo_euc)
    pct_outlier = n_outlier_examples / sample_n * 100
    print(f"     Examples with any nbr geo/euc > 50: "
          f"{n_outlier_examples} ({pct_outlier:.3f}%)")

    if outlier_examples_geo_euc:
        max_ratios = [e['max_ratio'] for e in outlier_examples_geo_euc]
        targets_of_oe = [e['target'] for e in outlier_examples_geo_euc]
        print(f"     Max geo/euc ratio across these: {max(max_ratios):.1f}")
        print(f"     Their target distribution: "
              f"mean={np.mean(targets_of_oe):.2f}, "
              f"median={np.median(targets_of_oe):.2f}, "
              f"max={max(targets_of_oe):.2f}")
        print(f"     Correlation: high-ratio examples tend to have "
              f"{'high' if np.corrcoef(max_ratios, targets_of_oe)[0,1] > 0.3 else 'normal'} targets")

    # -- 4. TARGET CAP IMPACT --
    if target_cap is not None:
        cap = target_cap
    else:
        cap = np.percentile(targets, 99.5)

    print(f"\n  4. PROPOSED TARGET CAP = {cap:.2f} (p99.5)")
    n_capped = int(np.sum(targets > cap))
    pct_capped = n_capped / n_examples * 100
    print(f"     Examples that would be removed: {n_capped} ({pct_capped:.4f}%)")
    clean_targets = targets[targets <= cap]
    print(f"     After cap: mean={clean_targets.mean():.4f}, "
          f"std={clean_targets.std():.4f}, max={clean_targets.max():.4f}")

    return {
        'animal': animal,
        'n_examples': n_examples,
        'target_mean': float(targets.mean()),
        'target_std': float(targets.std()),
        'target_max': float(targets.max()),
        'target_p99': float(np.percentile(targets, 99)),
        'target_p995': float(np.percentile(targets, 99.5)),
        'n_gt_100': int(np.sum(targets > 100)),
        'n_geo_euc_outlier_examples': n_outlier_examples,
        'target_cap': cap,
        'n_capped': n_capped,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--animal', default=None,
                        help='Single animal to analyze (default: all)')
    parser.add_argument('--target-cap', type=float, default=None,
                        help='Fixed target cap value (default: p99.5 per animal)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    animals = [args.animal] if args.animal else TOSCA_ANIMALS
    results = []

    print(f"\n{'#'*60}")
    print(f"# TOSCA Outlier & Target Distribution Analysis")
    print(f"{'#'*60}")

    for animal in animals:
        r = analyze_animal(animal, target_cap=args.target_cap,
                          verbose=args.verbose)
        if r:
            results.append(r)

    # -- CROSS-ANIMAL SUMMARY --
    if len(results) > 1:
        print(f"\n{'#'*60}")
        print(f"# CROSS-ANIMAL SUMMARY")
        print(f"{'#'*60}")
        print(f"\n  {'Animal':<12} {'N':>8} {'t_mean':>8} {'t_std':>8} "
              f"{'t_max':>10} {'t_p99':>8} {'>100':>6} {'fold%':>8} {'cap_rm':>8}")
        print(f"  {'-'*82}")
        for r in results:
            print(f"  {r['animal']:<12} {r['n_examples']:>8,} "
                  f"{r['target_mean']:>8.2f} {r['target_std']:>8.2f} "
                  f"{r['target_max']:>10.1f} {r['target_p99']:>8.2f} "
                  f"{r['n_gt_100']:>6} "
                  f"{r['n_geo_euc_outlier_examples']/r['n_examples']*100:>7.3f}% "
                  f"{r['n_capped']:>8}")

        total = sum(r['n_examples'] for r in results)
        total_gt100 = sum(r['n_gt_100'] for r in results)
        total_capped = sum(r['n_capped'] for r in results)
        print(f"\n  Total examples: {total:,}")
        print(f"  Total with target > 100: {total_gt100} ({total_gt100/total*100:.4f}%)")
        print(f"  Total that would be capped (p99.5): {total_capped} ({total_capped/total*100:.4f}%)")

        # RECOMMENDATION
        print(f"\n  RECOMMENDATION:")
        print(f"  The target outliers (max up to {max(r['target_max'] for r in results):.0f}) are")
        print(f"  real geodesic distances to far-away points. They represent")
        print(f"  {total_gt100}/{total} ({total_gt100/total*100:.3f}%) of examples.")
        print(f"  These extreme targets will create noisy gradients during training.")
        print(f"  Recommended: post-hoc clean with target cap at p99.5 (~{np.mean([r['target_p995'] for r in results]):.1f})")
        print()


if __name__ == '__main__':
    sys.exit(main() or 0)
