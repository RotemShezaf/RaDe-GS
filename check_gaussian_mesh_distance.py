"""Check distance of Gaussian centers from the closest reconstructed mesh vertices.

For each TOSCA shape in TrainData/TOSCA/SyntheticColmapData/blue_texture/,
loads the trained Gaussian point cloud (point_cloud.ply) and the reconstructed
mesh (recon.ply), then measures Gaussian-to-nearest-recon-vertex distances.
"""

import numpy as np
import trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from plyfile import PlyData


def load_gaussian_centers(ply_path: Path) -> np.ndarray:
    """Load Gaussian centers (xyz) from a trained point_cloud.ply."""
    plydata = PlyData.read(str(ply_path))
    vertex = plydata['vertex']
    return np.stack([np.array(vertex['x']),
                     np.array(vertex['y']),
                     np.array(vertex['z'])], axis=-1)


def load_mesh_vertices(ply_path: Path) -> np.ndarray:
    """Load mesh vertices from a .ply file."""
    mesh = trimesh.load(str(ply_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return np.asarray(mesh.vertices)


def print_stats(label: str, distances: np.ndarray):
    print(f"\n  {label}:")
    print(f"    Count:   {len(distances)}")
    print(f"    Mean:    {distances.mean():.6f}")
    print(f"    Median:  {np.median(distances):.6f}")
    print(f"    Std:     {distances.std():.6f}")
    print(f"    Min:     {distances.min():.6f}")
    print(f"    Max:     {distances.max():.6f}")
    for pct in [90, 95, 99]:
        print(f"    P{pct}:     {np.percentile(distances, pct):.6f}")


def main():
    base_dir = Path("TrainData/TOSCA/SyntheticColmapData/blue_texture")

    shapes = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
    print(f"Found {len(shapes)} shapes: {shapes}\n")

    for shape in shapes:
        shape_dir = base_dir / shape
        for res_dir in sorted(shape_dir.iterdir()):
            if not res_dir.is_dir():
                continue
            for light_dir in sorted(res_dir.iterdir()):
                if not light_dir.is_dir():
                    continue

                output_dir = light_dir / "output"
                recon_ply = output_dir / "recon.ply"
                gauss_ply = output_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
                if not gauss_ply.exists():
                    gauss_ply = output_dir / "point_cloud" / "iteration_7000" / "point_cloud.ply"

                if not recon_ply.exists() or not gauss_ply.exists():
                    missing = []
                    if not recon_ply.exists(): missing.append("recon.ply")
                    if not gauss_ply.exists(): missing.append("point_cloud.ply")
                    print(f"[SKIP] {shape}/{res_dir.name}/{light_dir.name}: missing {', '.join(missing)}")
                    continue

                print(f"{'='*80}")
                print(f"Shape: {shape} | Res: {res_dir.name} | Light: {light_dir.name}")
                print(f"{'='*80}")

                # Load Gaussian centers
                gaussians = load_gaussian_centers(gauss_ply)
                print(f"  Gaussians: {len(gaussians)}")

                # Load reconstructed mesh vertices
                recon_verts = load_mesh_vertices(recon_ply)
                print(f"  Recon mesh vertices: {len(recon_verts)}")

                # Gaussian -> nearest recon vertex
                tree = cKDTree(recon_verts)
                dists, _ = tree.query(gaussians)
                print_stats("Gaussian center -> nearest recon mesh vertex", dists)

                # Threshold analysis
                print(f"\n  Gaussians within threshold of recon mesh vertex:")
                for thresh in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
                    pct = (dists < thresh).sum() / len(dists) * 100
                    print(f"    Within {thresh:.3f}: {pct:.1f}%")
                print()


if __name__ == "__main__":
    main()
