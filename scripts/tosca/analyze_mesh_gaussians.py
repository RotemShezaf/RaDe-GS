#!/usr/bin/env python3
"""
Analyze TOSCA meshes vs Gaussian splats: connected components and
Gaussian-to-mesh distances for each shape at high and low resolution.

Reads shapes from the tosca_all_blue.txt config file and reports:
  - Number of connected components per mesh (with vertex/face/area stats)
  - Gaussian-to-nearest-mesh-vertex distance statistics (percentiles)

Usage:
    python scripts/tosca/analyze_mesh_gaussians.py
    python scripts/tosca/analyze_mesh_gaussians.py --shapes cat0,dog0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import KDTree

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from GenerateData.utils.load_utils import load_gaussian_data_cpu, load_ply

TOSCA_PROCESSED = project_root / "TrainData" / "TOSCA" / "processed"
CONFIG_FILE = (
    project_root
    / "DataSets"
    / "configs"
    / "tosca"
    / "gaussian_sources"
    / "tosca_all_blue.txt"
)
RESOLUTIONS = ["high_res", "low_res"]


def parse_shapes_from_config(config_path: Path):
    """Extract shape names from the config file."""
    shapes = []
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # .../blue_texture/<shape>/high_res/...
        parts = Path(line).parts
        # Find the part after 'blue_texture'
        for i, p in enumerate(parts):
            if p == "blue_texture" and i + 1 < len(parts):
                shapes.append(parts[i + 1])
                break
    return sorted(set(shapes))


def analyze_mesh_components(vertices, faces):
    """Return per-component stats using trimesh."""
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    components = mesh.split(only_watertight=False)
    stats = []
    for comp in components:
        stats.append({
            "vertices": len(comp.vertices),
            "faces": len(comp.faces),
            "area": float(comp.area),
        })
    # Sort by area descending
    stats.sort(key=lambda s: s["area"], reverse=True)
    return stats


def analyze_gaussian_mesh_distance(gaussian_positions, mesh_vertices):
    """Compute distance from each Gaussian to its nearest mesh vertex."""
    tree = KDTree(mesh_vertices)
    dists, _ = tree.query(gaussian_positions)
    return dists


def main():
    parser = argparse.ArgumentParser(
        description="Analyze TOSCA mesh components and Gaussian-mesh distances"
    )
    parser.add_argument(
        "--shapes", type=str, default=None,
        help="Comma-separated shape filter (default: all from config)",
    )
    args = parser.parse_args()

    shapes = parse_shapes_from_config(CONFIG_FILE)
    if args.shapes:
        filt = set(args.shapes.split(","))
        shapes = [s for s in shapes if s in filt]

    if not shapes:
        print("No shapes found.")
        return

    print(f"\n{'='*90}")
    print(f"  TOSCA Mesh & Gaussian Analysis  ({len(shapes)} shapes × {len(RESOLUTIONS)} resolutions)")
    print(f"{'='*90}")

    pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    for shape in shapes:
        print(f"\n{'─'*80}")
        print(f"Shape: {shape}")

        # Load Gaussians (from high_res decoupled_appearance output)
        gauss_output = (
            project_root
            / "TrainData"
            / "TOSCA"
            / "SyntheticColmapData"
            / "blue_texture"
            / shape
            / "high_res"
            / "decoupled_appearance"
            / "output"
        )
        try:
            gdata = load_gaussian_data_cpu(gauss_output)
            gauss_pos = gdata.get_xyz()
            n_gauss = len(gauss_pos)
            print(f"  Gaussians: {n_gauss}")
        except Exception as e:
            print(f"  [SKIP] Cannot load Gaussians: {e}")
            continue

        for resolution in RESOLUTIONS:
            # Find mesh file
            shape_dir = TOSCA_PROCESSED / shape
            candidates = sorted(shape_dir.glob(f"mesh_{resolution}_*.ply"))
            if not candidates:
                print(f"\n  {resolution}: no mesh found")
                continue

            mesh_path = candidates[0]
            vertices, faces = load_ply(str(mesh_path))

            print(f"\n  {resolution}:  {mesh_path.name}")
            print(f"    Mesh: {len(vertices):,} vertices, {len(faces):,} faces")

            # ── Connected components ──
            comp_stats = analyze_mesh_components(vertices, faces)
            n_comp = len(comp_stats)
            print(f"    Connected components: {n_comp}")
            if n_comp > 1:
                for i, c in enumerate(comp_stats):
                    print(
                        f"      Component {i}: {c['vertices']:,} verts, "
                        f"{c['faces']:,} faces, area={c['area']:.4f}"
                    )
            else:
                c = comp_stats[0]
                print(
                    f"      Single component: {c['vertices']:,} verts, "
                    f"{c['faces']:,} faces, area={c['area']:.4f}"
                )

            # ── Gaussian-to-mesh distance ──
            dists = analyze_gaussian_mesh_distance(gauss_pos, vertices)
            pct_vals = np.percentile(dists, pcts)
            header = "  ".join(f"p{p:>3d}" for p in pcts)
            values = "  ".join(f"{v:5.4f}" for v in pct_vals)
            print(f"    Gaussian→mesh distance (nearest vertex):")
            print(f"      mean={dists.mean():.6f}  std={dists.std():.6f}")
            print(f"      {header}")
            print(f"      {values}")

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    main()
