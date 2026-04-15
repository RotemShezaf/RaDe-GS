#!/usr/bin/env python3
"""Generate PNG benchmark reports for ALL TOSCA shapes that have sweep results.

Iterates over every shape in the SyntheticColmapData directory and generates
report_*.png files for each shape that has completed sweep runs.

Usage:
    python scripts/tosca/benchmark/generate_report_png_all.py

    # Custom paths:
    python scripts/tosca/benchmark/generate_report_png_all.py \
        --synth_data_base TrainData/TOSCA/SyntheticColmapData \
        --textures blue --resolutions high_res
"""

import argparse
import sys
from pathlib import Path

# Import from sibling module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_report_png import load_rows, make_table_figure, make_per_param_figures

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_all_source_paths(synth_data_base: Path, textures: list, resolutions: list,
                          use_decoupled: bool = True):
    """Yield (shape_name, resolution, source_path) for every valid shape directory."""
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


def generate_pngs_for_shape(shape_name: str, resolution: str,
                            source_path: Path, output_base: Path):
    """Generate report PNGs for a single shape. Returns count of PNGs saved."""
    rows = load_rows(source_path)
    if not rows:
        return 0

    output_dir = output_base / f"{shape_name}_{resolution}"
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    # Table sorted by Chamfer
    fig = make_table_figure(rows, f"{shape_name} — Sorted by Chamfer Distance",
                            sort_key="chamfer_distance", ascending=True)
    fig.savefig(output_dir / "report_chamfer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved += 1

    # Table sorted by PSNR
    fig = make_table_figure(rows, f"{shape_name} — Sorted by PSNR",
                            sort_key="test_psnr", ascending=False)
    fig.savefig(output_dir / "report_psnr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved += 1

    # Table sorted by G→Surf mean(d²)
    fig = make_table_figure(rows, f"{shape_name} — Sorted by G→Surf mean(d²)",
                            sort_key="g2s_mean_sq", ascending=True)
    fig.savefig(output_dir / "report_g2s_msq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved += 1

    # Per-parameter trend plots
    figures = make_per_param_figures(rows)
    for i, fig in enumerate(figures):
        fig.savefig(output_dir / f"report_param_{i:02d}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved += 1

    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Generate PNG benchmark reports for all TOSCA shapes")
    parser.add_argument("--synth_data_base", type=str,
                        default="TrainData/TOSCA/SyntheticColmapData",
                        help="Base directory for synthetic COLMAP data")
    parser.add_argument("--textures", type=str, default="blue",
                        help="Comma-separated texture names")
    parser.add_argument("--resolutions", type=str, default="high_res",
                        help="Comma-separated resolution levels")
    parser.add_argument("--use_decoupled_appearance", action="store_true", default=False)
    parser.add_argument("--output_base", type=str, default=None,
                        help="Output base dir for all PNGs (default: <synth_data_base>/sweep_reports)")
    args = parser.parse_args()

    synth_base = Path(args.synth_data_base)
    if not synth_base.exists():
        print(f"ERROR: {synth_base} does not exist")
        sys.exit(1)

    textures = [t.strip() for t in args.textures.split(",")]
    resolutions = [r.strip() for r in args.resolutions.split(",")]

    output_base = Path(args.output_base) if args.output_base else synth_base / "sweep_reports"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Synth data base: {synth_base}")
    print(f"Output base: {output_base}")
    print()

    total_shapes = 0
    total_pngs = 0

    for shape_name, resolution, source_path in find_all_source_paths(
            synth_base, textures, resolutions, args.use_decoupled_appearance):

        count = generate_pngs_for_shape(shape_name, resolution, source_path, output_base)
        if count > 0:
            print(f"  {shape_name}/{resolution}: {count} PNGs")
            total_shapes += 1
            total_pngs += count
        else:
            print(f"  {shape_name}/{resolution}: no sweep data — skipped")

    print(f"\nDone: {total_pngs} PNGs for {total_shapes} shapes in {output_base}")


if __name__ == "__main__":
    main()
