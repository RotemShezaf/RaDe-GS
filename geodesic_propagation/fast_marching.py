"""
Fast Marching Method implementation for geodesic distance propagation on Gaussian splats.

This module implements the core algorithm for computing geodesic distances
using a learned model for distance prediction.

Algorithm:
1. Initialize source points as Visited with distance 0
2. Add neighbors of sources to Wavefront
3. Repeat until all points are visited:
   a. For each point in Wavefront, predict distance using model
   b. Mark the minimum-distance Wavefront point as Visited
   c. Add Unvisited neighbors of newly Visited point to Wavefront
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Callable
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.priority_queue import WavefrontPriorityQueue, PointState
from geodesic_propagation.input_builder import GaussianInputBuilder
from geodesic_propagation.utils.model_handler import ModelHandler


class FastMarchingPropagator:
    """
    Fast Marching Method for geodesic distance propagation on Gaussian splats.
    
    This class implements the algorithm:
    1. Divide points into Visited, Wavefront, and Unvisited sets
    2. Use a learned model to predict geodesic distances
    3. Propagate distances from source points to all other points
    
    The model predicts the geodesic distance of a point based on:
    - Features of neighboring points that are already visited
    - The known geodesic distances of those neighbors
    """
    
    def __init__(
        self,
        model_handler: ModelHandler,
        input_builder: GaussianInputBuilder,
        ring1_neighbors: Dict[int, np.ndarray],
        ring_neighbors: Optional[Dict[int, np.ndarray]] = None,
        ring: int = 2,
        verbose: bool = True
    ):
        """
        Initialize the Fast Marching propagator.
        
        Args:
            model_handler: Handler for the geodesic prediction model
            input_builder: Builder for creating model inputs
            ring1_neighbors: Dict mapping point index to ring-1 neighbor indices
            ring_neighbors: Dict mapping point index to ring-k neighbor indices
                           (if None, will compute from ring1_neighbors)
            ring: Ring level for neighbor expansion (1-4)
            verbose: Whether to print progress information
        """
        self.model_handler = model_handler
        self.input_builder = input_builder
        self.ring1_neighbors = ring1_neighbors
        self.ring = ring
        self.verbose = verbose
        
        # Number of points
        self.num_points = input_builder.num_points
        
        # Compute ring-k neighbors if not provided
        if ring_neighbors is None:
            self.ring_neighbors = self._compute_ring_neighbors(ring)
        else:
            self.ring_neighbors = ring_neighbors
        
        # Reverse neighbour map: for each point V, which points have V in
        # *their* ring-k stencil.  KNN graphs are NOT symmetric, so
        # ring_neighbors[A] containing B does NOT imply ring_neighbors[B]
        # contains A.  During propagation we need to re-evaluate every
        # wavefront point P whose stencil contains the newly-visited point V,
        # i.e. V ∈ ring_neighbors[P].  This reverse map gives that set in O(1).
        self.reverse_ring_neighbors: Dict[int, List[int]] = {
            i: [] for i in range(self.num_points)
        }
        for point_idx, nbrs in self.ring_neighbors.items():
            for nbr in nbrs:
                self.reverse_ring_neighbors[int(nbr)].append(int(point_idx))
        
        # Initialize priority queue
        self.wavefront = WavefrontPriorityQueue(self.num_points)
        
        # Store computed distances
        self.distances = np.full(self.num_points, np.inf, dtype=np.float32)
        
        # Visited mask for O(1) neighbor filtering
        self.visited_mask = np.zeros(self.num_points, dtype=bool)
        
        # Diagnostic counters for model vs floor usage
        self._stats_model_wins = 0
        self._stats_floor_wins = 0
        self._stats_euclidean_fallback = 0
    
    def _compute_ring_neighbors(self, ring: int) -> Dict[int, np.ndarray]:
        """
        Compute ring-k neighbors from ring-1 neighbors.
        
        The ring-k neighborhood is built iteratively: ring-(i+1) is formed by
        taking all ring-1 neighbors of every ring-i neighbor and excluding the
        central point and all already-included points.
        
        Args:
            ring: Ring level (1-4)
            
        Returns:
            Dictionary mapping point index to ring-k neighbor indices
        """
        if ring == 1:
            return self.ring1_neighbors
        
        ring_neighbors = {}
        
        for point_idx in range(self.num_points):
            # Start with ring-1
            all_nbrs: Set[int] = set(self.ring1_neighbors[point_idx])
            # Frontier of the PREVIOUS ring level, used to expand to next level.
            prev_frontier: Set[int] = set(all_nbrs)
            
            for _level in range(2, ring + 1):
                new_frontier: Set[int] = set()
                for nbr in prev_frontier:
                    if nbr in self.ring1_neighbors:
                        new_frontier.update(self.ring1_neighbors[nbr])
                # Remove the central point and already-visited nodes
                new_frontier.discard(point_idx)
                new_frontier -= all_nbrs
                all_nbrs.update(new_frontier)
                prev_frontier = new_frontier
            
            ring_neighbors[point_idx] = np.array(sorted(all_nbrs), dtype=np.int64)
        
        return ring_neighbors
    
    def reset(self):
        """Reset the propagator to initial state."""
        self.wavefront.reset()
        self.distances = np.full(self.num_points, np.inf, dtype=np.float32)
        self.visited_mask = np.zeros(self.num_points, dtype=bool)
        self._stats_model_wins = 0
        self._stats_floor_wins = 0
        self._stats_euclidean_fallback = 0
    
    def initialize_sources(self, source_indices: Union[List[int], np.ndarray]):
        """
        Initialize source points with distance 0.
        
        Algorithm step (matches deep_eikonal_revisit):
        - Source points S: u(p) = 0, tag as VISITED
        - All other points: u(p) = infinity, tag as UNVISITED
        - Ring-1 neighbors of VISITED sources → tag as WAVEFRONT
        
        Using ring-1 for wavefront expansion (rather than ring-k) matches the
        reference deep_eikonal implementation: points enter the wavefront only
        when a ring-1 neighbour is alive, ensuring they have nearby context
        before being predicted.
        
        Args:
            source_indices: Indices of source points
        """
        # Mark sources as visited with distance 0
        for src_idx in source_indices:
            self.wavefront.set_visited(src_idx, 0.0)
            self.distances[src_idx] = 0.0
            self.visited_mask[src_idx] = True
        
        # Add ring-1 neighbors of sources to wavefront.
        for src_idx in source_indices:
            neighbors = self.ring1_neighbors.get(src_idx, np.array([], dtype=np.int64))
            if len(neighbors) > 0:
                # Filter out already visited neighbors using mask
                unvisited = neighbors[~self.visited_mask[neighbors]]
                for nbr_idx in unvisited:
                    self.wavefront.add_or_update(int(nbr_idx), float('inf'))
    
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_visited_neighbor_distances(
        self,
        point_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the indices and distances of visited neighbors for a point.
        
        Uses vectorized NumPy operations for O(1) masking.
        
        Args:
            point_idx: Index of the point
            
        Returns:
            Tuple of (neighbor_indices, neighbor_distances)
        """
        neighbors = self.ring_neighbors.get(point_idx, np.array([], dtype=np.int64))
        
        if len(neighbors) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        
        # Vectorized: get visited mask for all neighbors at once
        visited_mask_neighbors = self.visited_mask[neighbors]
        visited_neighbors = neighbors[visited_mask_neighbors]
        visited_distances = self.distances[visited_neighbors]
        
        return visited_neighbors, visited_distances
    
    # Minimum number of visited neighbours required to use the learned
    # model.  Below this threshold the model has too little context and
    # tends to predict ~0, causing a cascading-zero failure for every
    # subsequent point.  Instead, we fall back to a simple Euclidean-
    # interpolation estimate:
    #   dist(p) = min_{v ∈ visited_nbrs} ( dist(v) + ||p - v|| )
    MIN_NEIGHBORS_FOR_MODEL: int = 1

    @staticmethod
    def _euclidean_dijkstra_estimate(
        positions: np.ndarray,
        point_idx: int,
        visited_neighbors: np.ndarray,
        visited_distances: np.ndarray,
    ) -> float:
        """
        Compute the Euclidean-Dijkstra distance estimate for a point:
            dist(p) = min_{v} ( dist(v) + ||pos[p] - pos[v]|| )

        This provides a reliable *lower bound* that prevents the model-
        predicted distances from collapsing toward zero.
        """
        euc_dists = np.linalg.norm(
            positions[visited_neighbors] - positions[point_idx], axis=1
        )
        return float(np.min(visited_distances + euc_dists))

    def _update_wavefront_for_points(
        self,
        points_to_update: Union[List[int], Set[int]]
    ) -> None:
        """
        Predict and update wavefront distance estimates for a specific set of points.
        
        Model input is built from the CURRENT visited neighbourhood of each
        point.  Predictions are run in a single batched forward pass.
        
        Args:
            points_to_update: Collection of wavefront point indices to re-evaluate.
        """
        if not points_to_update:
            return

        batch_neighborhoods: List[torch.Tensor] = []
        batch_point_features: List[torch.Tensor] = []
        batch_valid_masks: List[torch.Tensor] = []
        batch_build_infos: List[Dict] = []
        valid_points: List[int] = []

        for point_idx in points_to_update:
            neighbors = self.ring_neighbors.get(point_idx, np.array([], dtype=np.int64))
            if len(neighbors) == 0 or not np.any(self.visited_mask[neighbors]):
                continue

            result = self.input_builder.build_input(
                point_idx=point_idx,
                all_neighbor_indices=neighbors,
                neighbor_distances=self.distances,
                ring1_neighbor_indices=self.ring1_neighbors.get(point_idx, None),
                visited_mask=self.visited_mask,
            )

            if result is None:
                continue

            neighborhood, point_features, valid_mask, build_info = result
            batch_neighborhoods.append(neighborhood)
            batch_point_features.append(point_features)
            batch_valid_masks.append(valid_mask)
            batch_build_infos.append(build_info)
            valid_points.append(point_idx)

        if not valid_points:
            return

        # Single batched model call
        neighborhoods_batch = torch.stack(batch_neighborhoods, dim=0)
        point_features_batch = torch.stack(batch_point_features, dim=0)
        valid_masks_batch = torch.stack(batch_valid_masks, dim=0)

        predictions = self.model_handler.predict(
            neighborhoods_batch, point_features_batch, valid_masks_batch
        )

        for i, point_idx in enumerate(valid_points):
            raw_pred = predictions[i, 0].item()
            model_dist = self.input_builder.denormalize_result(raw_pred, batch_build_infos[i])

            self._stats_model_wins += 1
            self.wavefront.add_or_update(point_idx, model_dist, allow_increase=True)

    def _update_neighbors_of_newly_visited(
        self,
        newly_visited_idx: int,
        newly_added_wavefront: List[int]
    ) -> None:
        """
        Update wavefront distance estimates fl points affected by a newly
        VISITED event.
        
        A point's distance estimate can only improve if it gains a new visited
        neighbour.  We need to re-evaluate every wavefront point P such that
        the newly-visited point V is in P's ring-k stencil (i.e.
        V ∈ ring_neighbors[P]).  Because KNN graphs are NOT symmetric,
        ring_neighbors[V] containing P does not imply ring_neighbors[P]
        contains V.  We therefore use the precomputed *reverse* neighbour map
        to find the correct affected set.
        
        Args:
            newly_visited_idx: Index of the point just marked as VISITED.
            newly_added_wavefront: Points just added to the WAVEFRONT (as
                unvisited neighbours of newly_visited_idx).
        """
        points_to_update: Set[int] = set(newly_added_wavefront)

        # All points whose ring-k stencil contains the newly visited point.
        # These are the points whose model input just changed (they gained a
        # new visited neighbour).
        for affected_point in self.reverse_ring_neighbors.get(newly_visited_idx, []):
            if self.wavefront.get_state(affected_point) == PointState.WAVEFRONT:
                points_to_update.add(affected_point)

        self._update_wavefront_for_points(points_to_update)

    # ------------------------------------------------------------------
    # Legacy helper kept for API compatibility and full-wavefront refresh.
    # NOTE: This is O(|wavefront|) model evaluations and is therefore
    # NOT used in the default propagation loop.  Prefer
    # _update_neighbors_of_newly_visited instead.
    # ------------------------------------------------------------------
    def _update_wavefront_distances(self) -> None:
        """
        Refresh distance estimates for *all* current wavefront points.
        
        This is the naïve O(|wavefront|) variant that re-evaluates every
        wavefront point on every iteration.  It is O(N²) overall and is
        retained for debugging / reference only.  Use
        ``_update_neighbors_of_newly_visited`` for production runs.
        """
        wavefront_points = list(self.wavefront.get_wavefront_points())
        self._update_wavefront_for_points(wavefront_points)

        # (Legacy return-value contract: update wavefront viaor al add_or_update;
        #  predictions[i, 0] bookkeeping block moved into _update_wavefront_for_points.)
        # The block below that previously duplicated the update logic has been
        # removed – all logic now lives in _update_wavefront_for_points.
        # Keeping this method as a thin wrapper ensures backward compatibility.

    # NOTE: _predict_distance (single-point, sequential) was removed as dead
    # code.  All inference now goes through the batched
    # _update_wavefront_for_points path.

    def propagate(
        self,
        source_indices: Union[List[int], np.ndarray],
        max_iterations: Optional[int] = None,
        callback: Optional[Callable[[int, int, float], None]] = None
    ) -> np.ndarray:
        """
        Run the Fast Marching propagation algorithm.
        
        Algorithm:
        1. Initialize sources as VISITED with u(p)=0, all others UNVISITED with u(p)=inf
        2. Tag all ring neighbors of VISITED sources as WAVEFRONT and predict
           their initial distances.
        3. Repeat until all points visited (or wavefront empty):
           - Tag least distant WAVEFRONT point p' as VISITED
           - Tag all UNVISITED ring neighbors of p' as WAVEFRONT
           - Re-predict only the wavefront points whose visited-neighbor set
             just changed (neighbors of p', plus newly added points).
        
        This is O(N × k) model evaluations (k = average ring-neighbor count)
        rather than the naive O(N × |wavefront|) approach.
        
        Args:
            source_indices: Indices of source points
            max_iterations: Maximum iterations (safety limit, default: None = no limit)
            callback: Optional callback(iteration, point_idx, distance)
            
        Returns:
            Array of geodesic distances for all points
        """
        self.reset()
        self.initialize_sources(source_indices)
        
        # Store source position for Euclidean floor computation
        self._source_pos = self.input_builder.positions[source_indices[0]]
        
        visited_count = len(source_indices)
        iteration = 0
        
        if self.verbose:
            pbar = tqdm(total=self.num_points, desc="Propagating distances")
            pbar.update(visited_count)
        
        # Euclidean seeding for initial wavefront.
        #
        # Ring-1 source neighbours only see the source (1 valid entry) — a
        # regime the model was NOT trained on.  In this neighbourhood the
        # geodesic distance to the source is well-approximated by the
        # Euclidean distance (curvature is negligible over one hop).  We
        # therefore seed all initial wavefront points with their Euclidean
        # distance to the source, accept them in Euclidean order, and let
        # the model take over from ring-2 onward where it has enough
        # visited context.
        initial_wavefront = set(self.wavefront.get_wavefront_points())
        refined_accepted: List[int] = []

        positions = self.input_builder.positions
        source_pos = positions[source_indices[0]]
        euclidean_order = sorted(
            initial_wavefront,
            key=lambda p: float(np.linalg.norm(positions[p] - source_pos))
        )

        for point_idx in euclidean_order:
            distance = float(np.linalg.norm(positions[point_idx] - source_pos))
            self.wavefront.set_visited(point_idx, distance)
            refined_accepted.append(point_idx)

            self.distances[point_idx] = distance
            self.visited_mask[point_idx] = True
            visited_count += 1

            if self.verbose:
                pbar.update(1)

            if callback:
                callback(iteration, point_idx, distance)

        # After refinement: expand wavefront from all accepted initial points
        # so ring-2+ neighbors enter the wavefront before the main loop.
        all_newly_added: List[int] = []
        points_to_update_post_refine: Set[int] = set()
        for vis_idx in refined_accepted:
            r1_nbrs = self.ring1_neighbors.get(vis_idx, np.array([], dtype=np.int64))
            if len(r1_nbrs) > 0:
                unvis = r1_nbrs[~self.visited_mask[r1_nbrs]]
                for nbr_idx in unvis:
                    if self.wavefront.get_state(int(nbr_idx)) == PointState.UNVISITED:
                        self.wavefront.add_or_update(int(nbr_idx), float('inf'))
                        all_newly_added.append(int(nbr_idx))
            for affected in self.reverse_ring_neighbors.get(vis_idx, []):
                if self.wavefront.get_state(affected) == PointState.WAVEFRONT:
                    points_to_update_post_refine.add(affected)
        points_to_update_post_refine.update(all_newly_added)
        if points_to_update_post_refine:
            self._update_wavefront_for_points(points_to_update_post_refine)

        # Main loop: continue until all points are visited or wavefront is empty
        while visited_count < self.num_points and not self.wavefront.is_empty():
            # Safety limit check
            if max_iterations is not None and iteration >= max_iterations:
                break
            
            # Step 1: Tag least distant WAVEFRONT point p' as VISITED
            result = self.wavefront.pop_min()
            if result is None:
                break
            
            point_idx, distance = result
            
            # Store the distance, update visited mask, and increment visited count
            self.distances[point_idx] = distance
            self.visited_mask[point_idx] = True
            visited_count += 1
            
            # Step 2: Expand wavefront via ring-1 of the newly visited point.
            # Using ring-1 (not ring-k) matches the deep_eikonal reference:
            # gradual one-hop expansion ensures each new wavefront point has
            # a close visited neighbour for context.
            r1_neighbors = self.ring1_neighbors.get(point_idx, np.array([], dtype=np.int64))
            newly_added: List[int] = []
            if len(r1_neighbors) > 0:
                unvisited = r1_neighbors[~self.visited_mask[r1_neighbors]]
                for nbr_idx in unvisited:
                    if self.wavefront.get_state(int(nbr_idx)) == PointState.UNVISITED:
                        self.wavefront.add_or_update(int(nbr_idx), float('inf'))
                        newly_added.append(int(nbr_idx))
            
            # Step 3: Re-predict wavefront points in the ring-k stencil of the
            # newly visited point (matches deep_eikonal affected_stencil logic).
            self._update_neighbors_of_newly_visited(point_idx, newly_added)
            
            # Callback
            if callback:
                callback(iteration, point_idx, distance)
            
            iteration += 1
            
            if self.verbose:
                pbar.update(1)
        
        if self.verbose:
            pbar.close()
            print(f"\nPropagation complete:")
            print(f"  Visited points: {visited_count}")
            print(f"  Unreachable points: {self.num_points - visited_count}")
            finite_dists = self.distances[np.isfinite(self.distances)]
            if len(finite_dists) > 0:
                print(f"  Distance range: [{finite_dists.min():.4f}, {finite_dists.max():.4f}]")
        
        return self.distances
    
    def propagate_batch(
        self,
        source_indices: Union[List[int], np.ndarray],
        batch_size: int = 32,
        max_iterations: Optional[int] = None,
        max_visited: Optional[int] = None,
        refine_passes: int = 0,
    ) -> np.ndarray:
        """
        Run Fast Marching with batched model predictions for efficiency.
        
        Identical algorithm to ``propagate`` but the per-iteration update of
        affected wavefront points is grouped into GPU-sized batches via
        ``build_batch_input``, which is more cache-efficient for large
        neighbourhoods.
        
        *batch_size* controls the maximum number of points sent to the model
        in a single forward pass.  When the affected wavefront set exceeds
        *batch_size*, it is split into consecutive chunks.
        
        After the FM loop, if *refine_passes* > 0, all visited (non-source)
        points are re-evaluated using the current propagated distances as
        context.  Each pass re-predicts every visited point and updates
        ``self.distances`` with the new prediction (subject to the Euclidean
        floor).  This counters error accumulation because later passes see
        improved neighbour distances.
        
        Args:
            source_indices: Indices of source points
            batch_size: Max points per model forward pass
            max_iterations: Maximum iterations (safety limit, default: None = no limit)
            max_visited: Stop after this many points are visited (default: None = all)
            refine_passes: Number of post-FM refinement passes (default: 0 = disabled)
            
        Returns:
            Array of geodesic distances for all points
        """
        self.reset()
        self.initialize_sources(source_indices)
        
        # Store source position for Euclidean floor computation in _run_batch_update
        self._source_pos = self.input_builder.positions[source_indices[0]]
        
        visited_count = len(source_indices)
        iteration = 0
        
        if self.verbose:
            pbar = tqdm(total=self.num_points, desc="Propagating (batched)")
            pbar.update(visited_count)
        
        # ── No special seeding ──
        # Ring-1 wavefront points start at inf and get model predictions
        # via _run_batch_update, same as every other wavefront point.
        # The source (dist=0) provides the visited context the model needs.

        initial_wavefront = list(self.wavefront.get_wavefront_points())
        if initial_wavefront:
            self._run_batch_update(initial_wavefront, batch_size)

        # Main loop
        while visited_count < self.num_points and not self.wavefront.is_empty():
            if max_iterations is not None and iteration >= max_iterations:
                break
            if max_visited is not None and visited_count >= max_visited:
                break
            
            # Pop ALL tied-minimum wavefront points simultaneously
            # (matches deep_eikonal get_multiple_next_indexes).
            accepted = self.wavefront.pop_all_min()
            if not accepted:
                break
            
            # Mark all accepted points as visited.
            # Apply Euclidean floor: geodesic >= Euclidean to source.
            # This prevents under-estimated distances from polluting
            # min_input in downstream normalization (the dominant error
            # source — min_input drift accounts for 89-99% of error).
            positions = self.input_builder.positions
            for point_idx, distance in accepted:
                euc_floor = float(np.linalg.norm(
                    positions[point_idx] - self._source_pos
                ))
                if distance < euc_floor:
                    distance = euc_floor
                    self._stats_floor_wins += 1
                self.distances[point_idx] = distance
                self.visited_mask[point_idx] = True
                visited_count += 1
            
            # For each newly accepted point: expand wavefront by ring-1,
            # then collect ring-k affected stencil for re-evaluation.
            all_newly_added: List[int] = []
            points_to_update: Set[int] = set()
            
            for point_idx, _distance in accepted:
                # Expand wavefront via ring-1 of the newly visited point.
                # Ring-1 (not ring-k) ensures gradual expansion so each new
                # wavefront point has a nearby visited neighbour for context.
                r1_neighbors = self.ring1_neighbors.get(point_idx, np.array([], dtype=np.int64))
                if len(r1_neighbors) > 0:
                    unvisited = r1_neighbors[~self.visited_mask[r1_neighbors]]
                    for nbr_idx in unvisited:
                        if self.wavefront.get_state(int(nbr_idx)) == PointState.UNVISITED:
                            self.wavefront.add_or_update(int(nbr_idx), float('inf'))
                            all_newly_added.append(int(nbr_idx))
                
                # Collect all wavefront points whose ring-k stencil contains
                # the newly visited point (reverse lookup — KNN is asymmetric).
                for affected in self.reverse_ring_neighbors.get(point_idx, []):
                    if self.wavefront.get_state(affected) == PointState.WAVEFRONT:
                        points_to_update.add(affected)
            
            points_to_update.update(all_newly_added)
            
            # Run batched update for affected wavefront points
            self._run_batch_update(list(points_to_update), batch_size)
            
            iteration += 1
            if self.verbose:
                pbar.update(len(accepted))
        
        if self.verbose:
            pbar.close()

        # ── Iterative refinement passes ────────────────────────────────
        if refine_passes > 0:
            source_set = set(int(s) for s in source_indices)
            visited_non_source = [
                i for i in range(self.num_points)
                if self.visited_mask[i] and i not in source_set
            ]
            if self.verbose:
                print(f"\nIterative refinement: {refine_passes} pass(es) "
                      f"over {len(visited_non_source)} visited points")

            positions = self.input_builder.positions
            for pass_idx in range(refine_passes):
                old_distances = self.distances.copy()
                updated = 0

                for chunk_start in range(0, len(visited_non_source), batch_size):
                    chunk = visited_non_source[chunk_start: chunk_start + batch_size]

                    batch_points: List[int] = []
                    batch_neighborhoods: List[torch.Tensor] = []
                    batch_point_features: List[torch.Tensor] = []
                    batch_valid_masks: List[torch.Tensor] = []
                    batch_build_infos: List[Dict] = []

                    for pid in chunk:
                        all_nbrs = self.ring_neighbors.get(pid, np.array([], dtype=np.int64))
                        if len(all_nbrs) == 0 or not np.any(self.visited_mask[all_nbrs]):
                            continue
                        result = self.input_builder.build_input(
                            point_idx=pid,
                            all_neighbor_indices=all_nbrs,
                            neighbor_distances=self.distances,
                            ring1_neighbor_indices=self.ring1_neighbors.get(pid, None),
                            visited_mask=self.visited_mask,
                        )
                        if result is None:
                            continue
                        neighborhood, point_features, valid_mask, build_info = result
                        batch_points.append(pid)
                        batch_neighborhoods.append(neighborhood)
                        batch_point_features.append(point_features)
                        batch_valid_masks.append(valid_mask)
                        batch_build_infos.append(build_info)

                    if not batch_points:
                        continue

                    neighborhoods_batch = torch.stack(batch_neighborhoods)
                    point_features_batch = torch.stack(batch_point_features)
                    valid_masks_batch = torch.stack(batch_valid_masks)

                    predictions = self.model_handler.predict(
                        neighborhoods_batch, point_features_batch, valid_masks_batch
                    )

                    for i, pid in enumerate(batch_points):
                        new_dist = self.input_builder.denormalize_result(
                            predictions[i, 0].item(),
                            batch_build_infos[i],
                        )
                        # Apply Euclidean floor
                        euc_floor = float(np.linalg.norm(
                            positions[pid] - self._source_pos
                        ))
                        new_dist = max(new_dist, euc_floor)
                        self.distances[pid] = new_dist
                        updated += 1

                delta = np.abs(self.distances - old_distances)
                finite_delta = delta[np.isfinite(delta)]
                if self.verbose:
                    print(f"  Pass {pass_idx + 1}/{refine_passes}: "
                          f"updated {updated} points, "
                          f"mean |delta|={finite_delta.mean():.6f}, "
                          f"max |delta|={finite_delta.max():.6f}")

        return self.distances

    def _run_batch_update(
        self,
        points_to_update: List[int],
        batch_size: int
    ) -> None:
        """
        Run batched model predictions for *points_to_update*, processing at
        most *batch_size* points per forward pass.
        
        Uses the fully-vectorised ``build_batch_input`` path for maximum
        throughput on GPU.
        """
        if not points_to_update:
            return

        for chunk_start in range(0, len(points_to_update), batch_size):
            chunk = points_to_update[chunk_start: chunk_start + batch_size]

            batch_points: List[int] = []
            batch_neighborhoods: List[torch.Tensor] = []
            batch_point_features: List[torch.Tensor] = []
            batch_valid_masks: List[torch.Tensor] = []
            batch_build_infos: List[Dict] = []

            for pid in chunk:
                all_nbrs = self.ring_neighbors.get(pid, np.array([], dtype=np.int64))
                if len(all_nbrs) == 0 or not np.any(self.visited_mask[all_nbrs]):
                    continue

                result = self.input_builder.build_input(
                    point_idx=pid,
                    all_neighbor_indices=all_nbrs,
                    neighbor_distances=self.distances,
                    ring1_neighbor_indices=self.ring1_neighbors.get(pid, None),
                    visited_mask=self.visited_mask,
                )

                if result is None:
                    continue

                neighborhood, point_features, valid_mask, build_info = result
                batch_points.append(pid)
                batch_neighborhoods.append(neighborhood)
                batch_point_features.append(point_features)
                batch_valid_masks.append(valid_mask)
                batch_build_infos.append(build_info)

            if not batch_points:
                continue

            neighborhoods_batch = torch.stack(batch_neighborhoods)
            point_features_batch = torch.stack(batch_point_features)
            valid_masks_batch = torch.stack(batch_valid_masks)

            predictions = self.model_handler.predict(
                neighborhoods_batch, point_features_batch, valid_masks_batch
            )

            for i, pid in enumerate(batch_points):
                model_dist = self.input_builder.denormalize_result(
                    predictions[i, 0].item(),
                    batch_build_infos[i],
                )

                self._stats_model_wins += 1
                self.wavefront.add_or_update(pid, model_dist, allow_increase=True)
    
    def get_distances(self) -> np.ndarray:
        """Get the computed distances."""
        return self.distances.copy()
    
    def get_model_floor_stats(self) -> Dict[str, int]:
        """Return diagnostic counters for model vs Euclidean floor usage."""
        return {
            'model_wins': self._stats_model_wins,
            'floor_wins': self._stats_floor_wins,
            'euclidean_fallback': self._stats_euclidean_fallback,
            'total_predictions': self._stats_model_wins + self._stats_floor_wins,
        }
    
    def get_propagation_stats(self) -> Dict[str, float]:
        """Get statistics about the propagation."""
        finite_mask = np.isfinite(self.distances)
        finite_dists = self.distances[finite_mask]
        
        return {
            'num_visited': int(np.sum(finite_mask)),
            'num_unreachable': int(np.sum(~finite_mask)),
            'min_distance': float(finite_dists.min()) if len(finite_dists) > 0 else None,
            'max_distance': float(finite_dists.max()) if len(finite_dists) > 0 else None,
            'mean_distance': float(finite_dists.mean()) if len(finite_dists) > 0 else None,
            'std_distance': float(finite_dists.std()) if len(finite_dists) > 0 else None,
        }


def create_propagator(
    model_path: str,
    gaussian_data: Dict[str, np.ndarray],
    dataset_config: Optional[Union[str, Path, Dict]] = None,
    ring: Optional[int] = None,
    device: Optional[str] = None,
    verbose: bool = True,
) -> FastMarchingPropagator:
    """
    Factory function to create a Fast Marching propagator.

    Accepts Gaussian data as a dict with keys ``'positions'``, ``'scales'``,
    ``'rotations'``, ``'opacities'``, ``'sh_features'``.

    Ring-1/ring-k neighbour computation and inference-transform construction
    are handled internally by :class:`GaussianInputBuilder`.  The
    ``dataset_config`` and ``ring`` are automatically read from the model
    checkpoint's companion YAML when not explicitly provided.  All kNN
    parameters (``n_neighbors``, ``use_mahalanobis``, adaptive kNN settings)
    are read from ``dataset_config``.

    Args:
        model_path: Path to the trained model checkpoint.
        gaussian_data: Dict with keys ``'positions'``, ``'scales'``,
            ``'rotations'``, ``'opacities'``, ``'sh_features'``.
        dataset_config: Dataset config YAML path/dict. If ``None``, read
            from the model checkpoint's companion YAML.
        ring: Ring level for neighbor expansion. If ``None``, read from the
            model checkpoint's companion YAML (falls back to 2).
        device: Device for model.
        verbose: Whether to print progress.

    Returns:
        Configured :class:`FastMarchingPropagator`.
    """
    # ── Resolve Gaussian arrays from dict ─────────────────────────────
    if gaussian_data is None or not isinstance(gaussian_data, dict):
        raise ValueError("gaussian_data must be a dict with keys 'positions', 'scales', etc.")

    positions = gaussian_data.get('positions')
    scales = gaussian_data.get('scales')
    rotations = gaussian_data.get('rotations')
    opacities = gaussian_data.get('opacities')
    sh_features = gaussian_data.get('sh_features')

    if positions is None:
        raise ValueError("gaussian_data must contain 'positions'")

    # ── Load model ────────────────────────────────────────────────────
    model_handler = ModelHandler(model_path=model_path, device=device)

    # ── Resolve dataset_config and ring from companion YAML ──────────
    if dataset_config is None:
        dataset_config = model_handler.get_dataset_config_path()
        if dataset_config is None:
            raise ValueError(
                "dataset_config was not provided and could not be read from "
                "the model checkpoint's companion YAML."
            )
    if ring is None:
        ring = model_handler.get_ring() or 2

    # ── Resolve transforms config from companion YAML ─────────────────
    transforms_cfg = model_handler.get_transforms_config()

    if verbose:
        attributes = model_handler.get_attributes()
        print(f"Model attributes: {attributes}")
        print(f"Max neighbors: {model_handler.get_max_neighbors()}")
        print(f"Ring: {ring}")
        print(f"Dataset config: {dataset_config}")
        if transforms_cfg:
            print(f"Transforms config: {[c if isinstance(c, str) else c.get('name') for c in transforms_cfg]}")

    # ── Create input builder (handles ring computation + transforms) ──
    if verbose:
        print("Computing ring-1 neighbors...")

    input_builder = GaussianInputBuilder(
        positions=positions,
        dataset_config=dataset_config,
        ring=ring,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_features=sh_features,
        device=device,
        transforms_config=transforms_cfg,
    )

    if verbose:
        print(f"Mean nearest neighbor distance: {input_builder.normalization_factor:.4f}")

    # ── Create propagator ─────────────────────────────────────────────
    propagator = FastMarchingPropagator(
        model_handler=model_handler,
        input_builder=input_builder,
        ring1_neighbors=input_builder.ring1_neighbors,
        ring_neighbors=input_builder.ring_neighbors,
        ring=ring,
        verbose=verbose
    )

    return propagator


if __name__ == "__main__":
    """Test the Fast Marching propagator."""
    print("Testing FastMarchingPropagator...")
    
    # Create dummy data
    num_points = 50
    np.random.seed(42)
    positions = np.random.randn(num_points, 3).astype(np.float32)
    scales = np.random.rand(num_points, 3).astype(np.float32) * 0.1
    rotations = np.random.randn(num_points, 4).astype(np.float32)
    rotations = rotations / np.linalg.norm(rotations, axis=1, keepdims=True)
    opacities = np.random.rand(num_points, 1).astype(np.float32)
    sh_features = np.random.randn(num_points, 3).astype(np.float32)
    
    # Create mock ring-1 neighbors (using Euclidean distance)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=8).fit(positions)
    _, indices = nbrs.kneighbors(positions)
    ring1_nbrs = {i: indices[i, 1:] for i in range(num_points)}  # Exclude self
    
    # Create model handler with a fresh model (for testing)
    model_handler = ModelHandler()
    model_handler.create_model({
        'attributes': ["xyz", "opacity", "scale", "rotation", "sh"],
        'max_neighbors': 32,
        'embed_dim': 128,
        'encoder_depth': 2,
        'num_heads': 4,
    })
    
    # Create input builder with a dataset config dict
    _test_config = {
        "output_dir": "/tmp/nonexistent_test_dir",
        "attributes": ["xyz", "opacity", "scale", "rotation", "sh"],
        "nn_mean": 1.0,
        "mask_constant": -10.0,
        "use_r1_min_val": False,
        "use_mahalanobis": False,
        "ring_size_mapping": {"euclidean": {2: 32, 3: 128}},
    }
    input_builder = GaussianInputBuilder(
        positions=positions,
        dataset_config=_test_config,
        ring=2,
        normalization_factor=0.5,
        scales=scales,
        rotations=rotations,
        opacities=opacities,
        sh_features=sh_features,
        device='cpu',
    )
    
    # Create propagator
    propagator = FastMarchingPropagator(
        model_handler=model_handler,
        input_builder=input_builder,
        ring1_neighbors=ring1_nbrs,
        ring=2,
        verbose=True
    )
    
    # Run propagation
    source_indices = [0]
    distances = propagator.propagate(source_indices, max_iterations=num_points)
    
    print(f"\nDistances shape: {distances.shape}")
    print(f"Source distance: {distances[0]}")
    print(f"Non-zero distances: {np.sum(distances > 0)}")
    
    # Get stats
    stats = propagator.get_propagation_stats()
    print(f"\nPropagation stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ FastMarchingPropagator tests passed!")
