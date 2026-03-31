"""
Geodesic Distance Propagation on Gaussian Splats

This module implements a Fast Marching Method-based algorithm for computing
geodesic distances on Gaussian splat representations using a learned model.

The algorithm maintains three disjoint sets:
1. Visited: Points where geodesic distance is finalized
2. Wavefront: Points where distance computation is in progress
3. Unvisited: Points where distance has not been computed yet

Main components:
- FastMarchingPropagator: Core algorithm implementation
- ModelHandler: Utilities for loading and using the prediction model
- PriorityQueue: Efficient wavefront management
- InputBuilder: Creates model inputs from Gaussian neighborhoods
"""

from .priority_queue import WavefrontPriorityQueue
from .fast_marching import FastMarchingPropagator
from .input_builder import GaussianInputBuilder

__all__ = [
    'WavefrontPriorityQueue',
    'FastMarchingPropagator', 
    'GaussianInputBuilder',
]
