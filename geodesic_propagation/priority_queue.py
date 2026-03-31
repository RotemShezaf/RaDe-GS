"""
Priority Queue implementation for the Fast Marching Method wavefront.

This module provides an efficient priority queue for managing the wavefront
of points being processed in the geodesic distance propagation algorithm.

Uses Python's heapq with indexed priority queue pattern for O(log n) operations.
"""

import heapq
import numpy as np
from typing import Set, Iterator, Optional, Tuple, List
from enum import IntEnum


class PointState(IntEnum):
    """State of a point in the Fast Marching algorithm."""
    UNVISITED = 0
    WAVEFRONT = 1
    VISITED = 2


class WavefrontPriorityQueue:
    """
    Indexed Priority Queue for Fast Marching Method wavefront management.
    
    This implementation uses:
    - heapq for O(log n) min extraction
    - Lazy deletion pattern for efficient distance updates
    - NumPy arrays for fast state/distance lookups
    
    Features:
    - O(log n) insertion and extraction of minimum
    - O(log n) decrease-key via lazy deletion
    - O(1) state and distance queries
    """
    
    __slots__ = ['num_points', '_heap', '_entry_finder', '_states', '_distances', '_counter']
    
    def __init__(self, num_points: int):
        """
        Initialize the priority queue.
        
        Args:
            num_points: Total number of points in the dataset
        """
        self.num_points = num_points
        self._heap: List[List] = []  # [distance, counter, point_idx, valid]
        self._entry_finder: dict = {}  # point_idx -> entry
        self._states = np.full(num_points, PointState.UNVISITED, dtype=np.int8)
        self._distances = np.full(num_points, np.inf, dtype=np.float64)
        self._counter = 0  # Tie-breaker for equal distances
    
    def is_empty(self) -> bool:
        """Check if the wavefront is empty."""
        self._cleanup_heap()
        return len(self._heap) == 0
    
    def _cleanup_heap(self):
        """Remove invalid entries from heap top."""
        while self._heap and not self._heap[0][3]:  # [3] is valid flag
            heapq.heappop(self._heap)
    
    def __len__(self) -> int:
        """Return the number of valid entries in the wavefront."""
        return int(np.sum(self._states == PointState.WAVEFRONT))
    
    def add_or_update(self, point_idx: int, distance: float, allow_increase: bool = False) -> bool:
        """
        Add a point to the wavefront or update its distance.

        By default (``allow_increase=False``), the distance is only updated if
        the new value is strictly *less* than the current one (standard FM
        monotone-decrease policy).  When ``allow_increase=True``, any finite
        new value replaces the current one.  This is needed when a learned
        model re-evaluates a wavefront point with more visited-neighbour
        context — the new prediction may be *higher* yet more accurate.
        
        Args:
            point_idx: Index of the point
            distance: Estimated geodesic distance
            allow_increase: If True, accept increases for existing wavefront
                points (used by learned-model re-evaluations).
            
        Returns:
            True if the point was added/updated, False otherwise
        """
        if point_idx < 0 or point_idx >= self.num_points:
            raise ValueError(f"Point index {point_idx} out of range [0, {self.num_points})")
        
        # Cannot update visited points
        if self._states[point_idx] == PointState.VISITED:
            return False
        
        # If point is unvisited, always add it to wavefront
        is_new_point = self._states[point_idx] == PointState.UNVISITED
        
        # For existing wavefront points: reject worse distances unless
        # allow_increase is set (learned-model re-evaluation).
        if not is_new_point:
            if not allow_increase and distance >= self._distances[point_idx]:
                return False
            if allow_increase and distance == self._distances[point_idx]:
                return False  # no-op
        
        # Invalidate old entry if exists
        if point_idx in self._entry_finder:
            self._entry_finder[point_idx][3] = False  # Mark invalid
        
        # Create new entry: [distance, counter, point_idx, valid]
        self._counter += 1
        entry = [distance, self._counter, point_idx, True]
        self._entry_finder[point_idx] = entry
        self._distances[point_idx] = distance
        self._states[point_idx] = PointState.WAVEFRONT
        heapq.heappush(self._heap, entry)
        
        return True
    
    def pop_min(self) -> Optional[Tuple[int, float]]:
        """
        Remove and return the point with minimum distance from the wavefront.
        
        Returns:
            Tuple of (point_idx, distance) or None if wavefront is empty
        """
        while self._heap:
            entry = heapq.heappop(self._heap)
            if entry[3]:  # Valid entry
                point_idx = entry[2]
                distance = entry[0]
                # Mark as visited
                self._states[point_idx] = PointState.VISITED
                # Remove from finder
                del self._entry_finder[point_idx]
                return point_idx, distance
        return None

    def pop_all_min(self) -> List[Tuple[int, float]]:
        """
        Remove and return ALL points that share the current minimum distance.

        Matches deep_eikonal's ``get_multiple_next_indexes()`` which extracts
        every tied-minimum entry in one go.

        Returns:
            List of (point_idx, distance) tuples.  Empty list when the
            wavefront is empty.
        """
        first = self.pop_min()
        if first is None:
            return []
        results = [first]
        min_dist = first[1]
        # Keep popping while the next entry has the exact same distance.
        while True:
            peeked = self.peek_min()
            if peeked is None or peeked[1] != min_dist:
                break
            results.append(self.pop_min())  # type: ignore[arg-type]
        return results
    
    def peek_min(self) -> Optional[Tuple[int, float]]:
        """
        Return the point with minimum distance without removing it.
        
        Returns:
            Tuple of (point_idx, distance) or None if wavefront is empty
        """
        self._cleanup_heap()
        if not self._heap:
            return None
        return self._heap[0][2], self._heap[0][0]  # point_idx, distance
    
    def get_state(self, point_idx: int) -> int:
        """Get the current state of a point."""
        return int(self._states[point_idx])
    
    def get_distance(self, point_idx: int) -> float:
        """Get the current distance estimate for a point."""
        return float(self._distances[point_idx])
    
    def set_visited(self, point_idx: int, distance: float = 0.0):
        """
        Mark a point as visited with a fixed distance.
        Used for source points initialization.
        
        Args:
            point_idx: Index of the point
            distance: Fixed distance value (default: 0.0 for sources)
        """
        if point_idx in self._entry_finder:
            self._entry_finder[point_idx][3] = False
            del self._entry_finder[point_idx]
        
        self._states[point_idx] = PointState.VISITED
        self._distances[point_idx] = distance
    
    def get_all_distances(self) -> np.ndarray:
        """Return all distance estimates."""
        return self._distances.copy()
    
    def get_visited_mask(self) -> np.ndarray:
        """Return a mask indicating which points are visited."""
        return self._states == PointState.VISITED
    
    def get_wavefront_points(self) -> Set[int]:
        """Return the set of point indices currently in the wavefront."""
        return set(np.where(self._states == PointState.WAVEFRONT)[0].tolist())
    
    def num_visited(self) -> int:
        """Return the number of visited points."""
        return int(np.sum(self._states == PointState.VISITED))
    
    def num_wavefront(self) -> int:
        """Return the number of points in the wavefront."""
        return int(np.sum(self._states == PointState.WAVEFRONT))
    
    def num_unvisited(self) -> int:
        """Return the number of unvisited points."""
        return int(np.sum(self._states == PointState.UNVISITED))
    
    def __iter__(self) -> Iterator[Tuple[int, float]]:
        """Iterate over wavefront points (point_idx, distance) in order."""
        # Get valid entries and sort by distance
        valid_entries = [(e[0], e[2]) for e in self._heap if e[3]]
        valid_entries.sort(key=lambda x: x[0])
        for distance, point_idx in valid_entries:
            yield point_idx, distance
    
    def reset(self):
        """Reset the queue to initial state."""
        self._heap.clear()
        self._entry_finder.clear()
        self._states.fill(PointState.UNVISITED)
        self._distances.fill(np.inf)
        self._counter = 0
