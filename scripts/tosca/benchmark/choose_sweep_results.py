#!/usr/bin/env python3
"""Choose the best sweep result for a single TOSCA shape based on a composite
quality formula.

The formula balances:
  - test_PSNR        (target ~48.5)
  - chamfer_distance  (target ~0.101)
  - g2s mean(d²)      (Gaussian-to-surface mean squared distance, target ~0.2)
  - g2s max           (max distance of any Gaussian from surface — penalty term)

Usage:
    python scripts/tosca/benchmark/choose_sweep_results.py \
        --source_path TrainData/TOSCA/SyntheticColmapData/blue_texture/cat0/high_res/decoupled_appearance

    # Override targets:
    python scripts/tosca/benchmark/choose_sweep_results.py \
        --source_path ... --target_psnr 48.5 --target_chamfer 0.101

The best output is **copied** into <source_path>/best_output/.
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


# ── Default target metrics ──────────────────────────────────────────────────
TARGET_PSNR = 48.5
TARGET_CHAMFER = 0.101
TARGET_MSD = 0.2       # mean squared distance (Gaussian → recon surface)
G2S_MAX_SOFT = 5.0     # g2s_max penalty kicks in above this
G2S_MAX_WEIGHT = 0.3   # weight of g2s_max penalty relative to other terms


def compute_score(report: dict,
                  target_psnr: float = TARGET_PSNR,
                  target_chamfer: float = TARGET_CHAMFER,
                  target_msd: float = TARGET_MSD,
                  g2s_max_soft: float = G2S_MAX_SOFT,
                  g2s_max_weight: float = G2S_MAX_WEIGHT) -> float:
    """Compute a composite quality score (lower is better).

    Components (each normalised so ≤1 when at target):
      1. PSNR error:    max(0, target - actual) / target
      2. Chamfer error: max(0, actual - target) / target
      3. MSD error:     max(0, actual - target) / target
      4. g2s_max pen.:  max(0, actual - soft_threshold) / 10 × weight
    """
    psnr = report.get("psnr", {}).get("test_psnr", 0)
    chamfer = report.get("chamfer_distance", 999)
    g2s = report.get("gaussian_to_recon_mesh_surface", {})
    g2s_msq = g2s.get("mean_squared", 999)
    g2s_max = g2s.get("max", 999)

    psnr_err = max(0, target_psnr - psnr) / target_psnr
    chamfer_err = max(0, chamfer - target_chamfer) / target_chamfer
    msd_err = max(0, g2s_msq - target_msd) / target_msd
    max_pen = max(0, g2s_max - g2s_max_soft) / 10.0 * g2s_max_weight

    return psnr_err + chamfer_err + msd_err + max_pen


def find_sweep_dirs(source_path: Path):
    """Yield (dir_path, report_dict) for every sweep_* subdir with a report."""
    for d in sorted(source_path.iterdir()):
        if not d.is_dir() or not d.name.startswith("sweep_"):
            continue
        report_path = d / "benchmark_report.json"
        if not report_path.exists():
            continue
        with open(report_path) as f:
            report = json.load(f)
        if "error" in report:
            continue
        yield d, report


def choose_best(source_path: Path, **kwargs) -> tuple:
    """Return (best_dir, best_report, best_score) or (None, None, inf)."""
    best_dir, best_report, best_score = None, None, float("inf")
    for d, report in find_sweep_dirs(source_path):
        score = compute_score(report, **kwargs)
        if score < best_score:
            best_dir, best_report, best_score = d, report, score
    return best_dir, best_report, best_score


def print_report(source_path: Path, **kwargs):
    """Print a ranked table of all sweep runs."""
    entries = []
    for d, report in find_sweep_dirs(source_path):
        score = compute_score(report, **kwargs)
        psnr = report.get("psnr", {}).get("test_psnr", 0)
        chamfer = report.get("chamfer_distance", 999)
        g2s = report.get("gaussian_to_recon_mesh_surface", {})
        entries.append({
            "name": d.name,
            "score": score,
            "psnr": psnr,
            "chamfer": chamfer,
            "g2s_msq": g2s.get("mean_squared", 999),
            "g2s_max": g2s.get("max", 999),
            "num_g": report.get("num_gaussians", 0),
        })

    entries.sort(key=lambda e: e["score"])

    print(f"\n{'Rank':>4s}  {'Config':15s}  {'Score':>7s}  {'PSNR':>7s}  {'Chamf':>7s}  "
          f"{'g2s_msq':>8s}  {'g2s_max':>8s}  {'#Gauss':>7s}")
    print("-" * 85)
    for i, e in enumerate(entries, 1):
        marker = " <-- BEST" if i == 1 else ""
        print(f"{i:4d}  {e['name']:15s}  {e['score']:7.4f}  {e['psnr']:7.2f}  "
              f"{e['chamfer']:7.4f}  {e['g2s_msq']:8.4f}  {e['g2s_max']:8.2f}  "
              f"{e['num_g']:7d}{marker}")
    return entries


def copy_best(best_dir: Path, dest_dir: Path):
    """Copy the best output to dest_dir, preserving key files."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)

    # Copy key output files (not the entire training checkpoint)
    files_to_copy = [
        "benchmark_report.json",
        "benchmark_args.txt",
        "recon.ply",
        "recon_clean.ply",
        "cameras.json",
        "cfg_args",
        "train.log",
        "eval.log",
    ]
    for fname in files_to_copy:
        src = best_dir / fname
        if src.exists():
            shutil.copy2(src, dest_dir / fname)

    # Copy point_cloud directory if exists
    pc_dir = best_dir / "point_cloud"
    if pc_dir.exists():
        shutil.copytree(pc_dir, dest_dir / "point_cloud")

    # Write metadata about selection
    meta = {
        "source_sweep_dir": str(best_dir),
        "selected_config": best_dir.name,
    }
    with open(dest_dir / "selection_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Choose the best sweep result for a TOSCA shape")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to the shape's data dir (e.g. .../cat0/high_res/decoupled_appearance)")
    parser.add_argument("--target_psnr", type=float, default=TARGET_PSNR)
    parser.add_argument("--target_chamfer", type=float, default=TARGET_CHAMFER)
    parser.add_argument("--target_msd", type=float, default=TARGET_MSD)
    parser.add_argument("--g2s_max_soft", type=float, default=G2S_MAX_SOFT)
    parser.add_argument("--g2s_max_weight", type=float, default=G2S_MAX_WEIGHT)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to copy the best output (default: <source_path>/best_output)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print ranking without copying")
    args = parser.parse_args()

    source_path = Path(args.source_path)
    if not source_path.exists():
        print(f"ERROR: {source_path} does not exist")
        sys.exit(1)

    score_kwargs = dict(
        target_psnr=args.target_psnr,
        target_chamfer=args.target_chamfer,
        target_msd=args.target_msd,
        g2s_max_soft=args.g2s_max_soft,
        g2s_max_weight=args.g2s_max_weight,
    )

    print(f"Source: {source_path}")
    print(f"Targets: PSNR≥{args.target_psnr}, Chamfer≤{args.target_chamfer}, "
          f"MSD≤{args.target_msd}, g2s_max soft≤{args.g2s_max_soft}")

    entries = print_report(source_path, **score_kwargs)
    if not entries:
        print("No completed sweep runs found.")
        sys.exit(1)

    best_dir, best_report, best_score = choose_best(source_path, **score_kwargs)
    print(f"\nBest: {best_dir.name} (score={best_score:.4f})")

    if args.dry_run:
        print("[DRY RUN] Would copy to best_output/")
        return

    dest = Path(args.output_dir) if args.output_dir else source_path / "best_output"
    copy_best(best_dir, dest)
    print(f"Best output copied to: {dest}")


if __name__ == "__main__":
    main()
