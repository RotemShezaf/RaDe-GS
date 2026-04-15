#!/usr/bin/env python3
"""Choose the best sweep result for ALL TOSCA shapes.

Iterates over every shape in the SyntheticColmapData directory, runs
choose_sweep_results logic on each, and copies the best output to
<source_path>/best_output/ for every shape.

Usage:
    python scripts/tosca/benchmark/choose_sweep_results_all.py

    # Custom base path and targets:
    python scripts/tosca/benchmark/choose_sweep_results_all.py \
        --synth_data_base TrainData/TOSCA/SyntheticColmapData \
        --target_psnr 48.5 --target_chamfer 0.101 --target_msd 0.2

    # Dry run (rank all, don't copy):
    python scripts/tosca/benchmark/choose_sweep_results_all.py --dry_run
"""

import argparse
import json
import sys
from pathlib import Path

# Import from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from choose_sweep_results import (
    choose_best, print_report, copy_best,
    TARGET_PSNR, TARGET_CHAMFER, TARGET_MSD, G2S_MAX_SOFT, G2S_MAX_WEIGHT,
)


def find_all_source_paths(synth_data_base: Path, textures: list, resolutions: list,
                          use_decoupled: bool = True):
    """Yield (shape_name, source_path) for every valid shape directory."""
    for texture in textures:
        texture_dir = synth_data_base / f"{texture}_texture"
        if not texture_dir.exists():
            continue
        for shape_dir in sorted(texture_dir.iterdir()):
            if not shape_dir.is_dir():
                continue
            shape_name = shape_dir.name
            for resolution in resolutions:
                if use_decoupled:
                    src = shape_dir / resolution / "decoupled_appearance"
                else:
                    src = shape_dir / resolution / "light_0"
                if src.exists():
                    yield shape_name, resolution, src


def main():
    parser = argparse.ArgumentParser(
        description="Choose best sweep result for all TOSCA shapes")
    parser.add_argument("--synth_data_base", type=str,
                        default="TrainData/TOSCA/SyntheticColmapData",
                        help="Base directory for synthetic COLMAP data")
    parser.add_argument("--textures", type=str, default="blue",
                        help="Comma-separated texture names")
    parser.add_argument("--resolutions", type=str, default="high_res",
                        help="Comma-separated resolution levels")
    parser.add_argument("--use_decoupled_appearance", action="store_true", default=False)
    parser.add_argument("--target_psnr", type=float, default=TARGET_PSNR)
    parser.add_argument("--target_chamfer", type=float, default=TARGET_CHAMFER)
    parser.add_argument("--target_msd", type=float, default=TARGET_MSD)
    parser.add_argument("--g2s_max_soft", type=float, default=G2S_MAX_SOFT)
    parser.add_argument("--g2s_max_weight", type=float, default=G2S_MAX_WEIGHT)
    parser.add_argument("--dry_run", action="store_true",
                        help="Print rankings without copying")
    args = parser.parse_args()

    synth_base = Path(args.synth_data_base)
    if not synth_base.exists():
        print(f"ERROR: {synth_base} does not exist")
        sys.exit(1)

    textures = [t.strip() for t in args.textures.split(",")]
    resolutions = [r.strip() for r in args.resolutions.split(",")]

    score_kwargs = dict(
        target_psnr=args.target_psnr,
        target_chamfer=args.target_chamfer,
        target_msd=args.target_msd,
        g2s_max_soft=args.g2s_max_soft,
        g2s_max_weight=args.g2s_max_weight,
    )

    print(f"Synth data base: {synth_base}")
    print(f"Textures: {textures}, Resolutions: {resolutions}")
    print(f"Targets: PSNR≥{args.target_psnr}, Chamfer≤{args.target_chamfer}, "
          f"MSD≤{args.target_msd}, g2s_max soft≤{args.g2s_max_soft}")
    print()

    results_summary = []
    shapes_processed = 0
    shapes_skipped = 0

    for shape_name, resolution, source_path in find_all_source_paths(
            synth_base, textures, resolutions, args.use_decoupled_appearance):

        print(f"\n{'='*70}")
        print(f"Shape: {shape_name} / {resolution}")
        print(f"Source: {source_path}")
        print(f"{'='*70}")

        best_dir, best_report, best_score = choose_best(source_path, **score_kwargs)
        if best_dir is None:
            print("  No completed sweep runs found — skipping")
            shapes_skipped += 1
            continue

        print_report(source_path, **score_kwargs)
        shapes_processed += 1

        entry = {
            "shape": shape_name,
            "resolution": resolution,
            "best_config": best_dir.name,
            "score": best_score,
            "psnr": best_report.get("psnr", {}).get("test_psnr", 0),
            "chamfer": best_report.get("chamfer_distance", 999),
            "g2s_msq": best_report.get("gaussian_to_recon_mesh_surface", {}).get("mean_squared", 999),
            "g2s_max": best_report.get("gaussian_to_recon_mesh_surface", {}).get("max", 999),
        }
        results_summary.append(entry)

        if args.dry_run:
            print(f"\n  [DRY RUN] Would copy {best_dir.name} → best_output/")
        else:
            dest = source_path / "best_output"
            copy_best(best_dir, dest)
            print(f"\n  Best output copied to: {dest}")

    # ── Overall summary ─────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"  Shapes processed: {shapes_processed}")
    print(f"  Shapes skipped:   {shapes_skipped}")
    print()

    if results_summary:
        print(f"  {'Shape':18s}  {'Config':15s}  {'Score':>7s}  {'PSNR':>7s}  "
              f"{'Chamf':>7s}  {'g2s_msq':>8s}  {'g2s_max':>8s}")
        print("  " + "-" * 85)
        for e in sorted(results_summary, key=lambda x: x["score"]):
            print(f"  {e['shape']:18s}  {e['best_config']:15s}  {e['score']:7.4f}  "
                  f"{e['psnr']:7.2f}  {e['chamfer']:7.4f}  {e['g2s_msq']:8.4f}  "
                  f"{e['g2s_max']:8.2f}")

    # Save summary JSON
    if not args.dry_run and results_summary:
        summary_path = synth_base / "sweep_selection_summary.json"
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
