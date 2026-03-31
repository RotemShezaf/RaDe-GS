"""
Results saving and loading utilities for geodesic propagation.

This module provides utilities for:
- Saving propagation results to disk
- Loading results for analysis
- Exporting results in various formats
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import torch


class ResultsSaver:
    """
    Handler for saving and loading geodesic propagation results.
    
    Supports multiple output formats:
    - NPZ: NumPy compressed format (default)
    - JSON: For metadata and small results
    - PLY: For visualization in 3D software
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize the results saver.
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_propagation_results(
        self,
        distances: np.ndarray,
        source_indices: Union[List[int], np.ndarray],
        positions: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "propagation_results"
    ) -> Path:
        """
        Save complete propagation results.
        
        Args:
            distances: Array of computed geodesic distances (N,)
            source_indices: Indices of source points
            positions: Gaussian positions (N, 3)
            metadata: Additional metadata dict
            name: Base name for output files
            
        Returns:
            Path to the saved results file
        """
        # Prepare data
        if isinstance(source_indices, list):
            source_indices = np.array(source_indices)
        
        # Create metadata
        full_metadata = {
            'num_points': len(distances),
            'num_sources': len(source_indices),
            'source_indices': source_indices.tolist(),
            'timestamp': datetime.now().isoformat(),
            'distance_stats': {
                'min': float(np.min(distances[np.isfinite(distances)])) if np.any(np.isfinite(distances)) else None,
                'max': float(np.max(distances[np.isfinite(distances)])) if np.any(np.isfinite(distances)) else None,
                'mean': float(np.mean(distances[np.isfinite(distances)])) if np.any(np.isfinite(distances)) else None,
                'num_unreachable': int(np.sum(~np.isfinite(distances))),
            }
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # Save NPZ file
        npz_path = self.output_dir / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            distances=distances,
            source_indices=source_indices,
            positions=positions
        )
        
        # Save metadata as JSON
        json_path = self.output_dir / f"{name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        print(f"Results saved to: {npz_path}")
        print(f"Metadata saved to: {json_path}")
        
        return npz_path
    
    def save_evaluation_results(
        self,
        predicted_distances: np.ndarray,
        ground_truth_distances: np.ndarray,
        source_indices: Union[List[int], np.ndarray],
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        name: str = "evaluation_results"
    ) -> Path:
        """
        Save evaluation results comparing predictions to ground truth.
        
        Args:
            predicted_distances: Predicted geodesic distances (N,)
            ground_truth_distances: Ground truth distances (N,)
            source_indices: Indices of source points
            metrics: Dictionary of evaluation metrics
            metadata: Additional metadata
            name: Base name for output files
            
        Returns:
            Path to saved results
        """
        if isinstance(source_indices, list):
            source_indices = np.array(source_indices)
        
        # Compute errors
        errors = predicted_distances - ground_truth_distances
        valid_mask = np.isfinite(predicted_distances) & np.isfinite(ground_truth_distances)
        
        # Create metadata
        full_metadata = {
            'num_points': len(predicted_distances),
            'num_sources': len(source_indices),
            'source_indices': source_indices.tolist(),
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'error_stats': {
                'mae': float(np.mean(np.abs(errors[valid_mask]))) if np.any(valid_mask) else None,
                'rmse': float(np.sqrt(np.mean(errors[valid_mask] ** 2))) if np.any(valid_mask) else None,
                'max_error': float(np.max(np.abs(errors[valid_mask]))) if np.any(valid_mask) else None,
                'num_valid': int(np.sum(valid_mask)),
            }
        }
        
        if metadata:
            full_metadata.update(metadata)
        
        # Save NPZ file
        npz_path = self.output_dir / f"{name}.npz"
        np.savez_compressed(
            npz_path,
            predicted_distances=predicted_distances,
            ground_truth_distances=ground_truth_distances,
            source_indices=source_indices,
            errors=errors,
            valid_mask=valid_mask
        )
        
        # Save metadata as JSON
        json_path = self.output_dir / f"{name}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        print(f"Evaluation results saved to: {npz_path}")
        print(f"Evaluation metadata saved to: {json_path}")
        
        return npz_path
    
    def load_results(self, name: str = "propagation_results") -> Dict[str, Any]:
        """
        Load previously saved results.
        
        Args:
            name: Base name of the results files
            
        Returns:
            Dictionary with loaded data and metadata
        """
        npz_path = self.output_dir / f"{name}.npz"
        json_path = self.output_dir / f"{name}_metadata.json"
        
        if not npz_path.exists():
            raise FileNotFoundError(f"Results file not found: {npz_path}")
        
        # Load NPZ data
        data = dict(np.load(npz_path, allow_pickle=True))
        
        # Load metadata if exists
        if json_path.exists():
            with open(json_path, 'r') as f:
                data['metadata'] = json.load(f)
        
        return data
    
    def export_to_ply(
        self,
        positions: np.ndarray,
        distances: np.ndarray,
        output_name: str = "geodesic_visualization",
        colormap: str = "viridis"
    ) -> Path:
        """
        Export results to PLY format for 3D visualization.
        
        Colors points based on geodesic distance using a colormap.
        
        Args:
            positions: Point positions (N, 3)
            distances: Geodesic distances (N,)
            output_name: Output file name (without extension)
            colormap: Matplotlib colormap name
            
        Returns:
            Path to the PLY file
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
        except ImportError:
            print("matplotlib required for PLY export with colormap")
            raise
        
        # Normalize distances for colormap
        valid_mask = np.isfinite(distances)
        if np.any(valid_mask):
            min_dist = np.min(distances[valid_mask])
            max_dist = np.max(distances[valid_mask])
            normalized = (distances - min_dist) / (max_dist - min_dist + 1e-8)
        else:
            normalized = np.zeros_like(distances)
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colors = cmap(normalized)[:, :3]  # RGB only
        colors = (colors * 255).astype(np.uint8)
        
        # Handle infinite distances
        colors[~valid_mask] = [128, 128, 128]  # Gray for unreachable
        
        # Write PLY file
        ply_path = self.output_dir / f"{output_name}.ply"
        
        with open(ply_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(positions)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("property float distance\n")
            f.write("end_header\n")
            
            for i in range(len(positions)):
                f.write(f"{positions[i, 0]} {positions[i, 1]} {positions[i, 2]} "
                       f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]} {distances[i]}\n")
        
        print(f"PLY exported to: {ply_path}")
        return ply_path
    
    def generate_summary_report(
        self,
        results_name: str = "evaluation_results"
    ) -> str:
        """
        Generate a text summary report from saved results.
        
        Args:
            results_name: Name of the results to summarize
            
        Returns:
            Summary report as string
        """
        data = self.load_results(results_name)
        metadata = data.get('metadata', {})
        
        report = []
        report.append("=" * 60)
        report.append("Geodesic Distance Propagation - Evaluation Report")
        report.append("=" * 60)
        report.append("")
        
        if 'timestamp' in metadata:
            report.append(f"Timestamp: {metadata['timestamp']}")
        
        report.append(f"Number of points: {metadata.get('num_points', 'N/A')}")
        report.append(f"Number of sources: {metadata.get('num_sources', 'N/A')}")
        report.append("")
        
        if 'metrics' in metadata:
            report.append("Evaluation Metrics:")
            report.append("-" * 40)
            for key, value in metadata['metrics'].items():
                if isinstance(value, float):
                    report.append(f"  {key}: {value:.6f}")
                else:
                    report.append(f"  {key}: {value}")
        
        if 'error_stats' in metadata:
            report.append("")
            report.append("Error Statistics:")
            report.append("-" * 40)
            for key, value in metadata['error_stats'].items():
                if value is not None:
                    if isinstance(value, float):
                        report.append(f"  {key}: {value:.6f}")
                    else:
                        report.append(f"  {key}: {value}")
        
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report to file
        report_path = self.output_dir / f"{results_name}_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to: {report_path}")
        return report_text


if __name__ == "__main__":
    """Test the results saver."""
    import tempfile
    
    print("Testing ResultsSaver...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        saver = ResultsSaver(tmpdir)
        
        # Test data
        num_points = 100
        positions = np.random.randn(num_points, 3)
        distances = np.random.rand(num_points) * 10
        distances[0] = 0  # Source
        source_indices = [0]
        
        # Test save propagation results
        path = saver.save_propagation_results(
            distances=distances,
            source_indices=source_indices,
            positions=positions,
            metadata={'test': True},
            name='test_results'
        )
        print(f"Saved to: {path}")
        
        # Test load
        loaded = saver.load_results('test_results')
        print(f"Loaded keys: {list(loaded.keys())}")
        assert np.allclose(loaded['distances'], distances)
        print("Load verification passed!")
        
        # Test evaluation results
        gt_distances = distances + np.random.randn(num_points) * 0.1
        saver.save_evaluation_results(
            predicted_distances=distances,
            ground_truth_distances=gt_distances,
            source_indices=source_indices,
            metrics={'mae': 0.05, 'rmse': 0.07},
            name='test_eval'
        )
        
        # Test PLY export
        saver.export_to_ply(positions, distances, 'test_viz')
        
    print("\n✓ ResultsSaver tests passed!")
