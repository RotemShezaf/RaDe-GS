"""
Comprehensive tests for WavefrontPriorityQueue and PointState.

Covers:
- All state transitions (UNVISITED -> WAVEFRONT -> VISITED)
- Lazy-deletion correctness
- Decrease-key semantics
- Invariant: pop_min always returns minimum finite distance
- Edge cases: empty queue, out-of-range, duplicate updates
- Full propagation simulation on known graphs
"""

import pytest
import numpy as np
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geodesic_propagation.priority_queue import WavefrontPriorityQueue, PointState


# ---------------------------------------------------------------------------
# PointState
# ---------------------------------------------------------------------------

class TestPointState:
    def test_int_values(self):
        assert int(PointState.UNVISITED) == 0
        assert int(PointState.WAVEFRONT) == 1
        assert int(PointState.VISITED)   == 2

    def test_ordering(self):
        assert PointState.UNVISITED < PointState.WAVEFRONT < PointState.VISITED

    def test_all_distinct(self):
        states = [PointState.UNVISITED, PointState.WAVEFRONT, PointState.VISITED]
        assert len(set(states)) == 3


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_default_counts(self):
        q = WavefrontPriorityQueue(20)
        assert q.num_points    == 20
        assert q.is_empty()
        assert len(q)          == 0
        assert q.num_visited()   == 0
        assert q.num_wavefront() == 0
        assert q.num_unvisited() == 20

    def test_all_distances_inf(self):
        q = WavefrontPriorityQueue(5)
        assert np.all(np.isinf(q.get_all_distances()))

    def test_all_states_unvisited(self):
        q = WavefrontPriorityQueue(5)
        for i in range(5):
            assert q.get_state(i) == PointState.UNVISITED

    def test_zero_points(self):
        q = WavefrontPriorityQueue(0)
        assert q.is_empty()
        assert q.pop_min() is None

    def test_single_point(self):
        q = WavefrontPriorityQueue(1)
        assert q.num_unvisited() == 1


# ---------------------------------------------------------------------------
# add_or_update
# ---------------------------------------------------------------------------

class TestAddOrUpdate:
    def test_new_point_returns_true_and_sets_state(self):
        q = WavefrontPriorityQueue(10)
        assert q.add_or_update(3, 1.5) is True
        assert q.get_state(3)    == PointState.WAVEFRONT
        assert q.get_distance(3) == pytest.approx(1.5)

    def test_decrease_key_accepted(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(2, 5.0)
        assert q.add_or_update(2, 2.0) is True
        assert q.get_distance(2) == pytest.approx(2.0)

    def test_increase_key_rejected(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(2, 1.0)
        assert q.add_or_update(2, 9.0) is False
        assert q.get_distance(2) == pytest.approx(1.0)

    def test_equal_key_rejected(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(2, 3.0)
        assert q.add_or_update(2, 3.0) is False

    def test_update_visited_rejected(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(3, 2.0)
        q.pop_min()
        assert q.add_or_update(3, 0.1) is False
        assert q.get_state(3) == PointState.VISITED

    def test_inf_distance_allowed(self):
        """Initial wavefront entries may be inf (distance not yet predicted)."""
        q = WavefrontPriorityQueue(10)
        q.add_or_update(7, float('inf'))
        assert q.get_state(7) == PointState.WAVEFRONT
        assert np.isinf(q.get_distance(7))

    def test_out_of_range_raises(self):
        q = WavefrontPriorityQueue(5)
        with pytest.raises((ValueError, IndexError)):
            q.add_or_update(10, 1.0)
        with pytest.raises((ValueError, IndexError)):
            q.add_or_update(-1, 1.0)

    def test_update_does_not_double_count(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 5.0)
        q.add_or_update(0, 3.0)
        assert q.num_wavefront() == 1
        assert len(q) == 1

    def test_zero_distance(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 0.0)
        assert q.get_distance(0) == 0.0


# ---------------------------------------------------------------------------
# set_visited
# ---------------------------------------------------------------------------

class TestSetVisited:
    def test_from_unvisited(self):
        q = WavefrontPriorityQueue(10)
        q.set_visited(2, 0.5)
        assert q.get_state(2)    == PointState.VISITED
        assert q.get_distance(2) == pytest.approx(0.5)
        assert q.num_visited() == 1

    def test_from_wavefront(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(4, 1.0)
        q.set_visited(4, 1.0)
        assert q.get_state(4) == PointState.VISITED

    def test_not_in_wavefront_set(self):
        q = WavefrontPriorityQueue(10)
        q.set_visited(3, 0.0)
        assert 3 not in q.get_wavefront_points()


# ---------------------------------------------------------------------------
# pop_min
# ---------------------------------------------------------------------------

class TestPopMin:
    def test_empty_returns_none(self):
        q = WavefrontPriorityQueue(10)
        assert q.pop_min() is None

    def test_returns_global_minimum(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 3.0)
        q.add_or_update(1, 1.0)
        q.add_or_update(2, 2.0)
        idx, dist = q.pop_min()
        assert idx  == 1
        assert dist == pytest.approx(1.0)

    def test_transitions_to_visited(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(5, 0.5)
        q.pop_min()
        assert q.get_state(5) == PointState.VISITED

    def test_decrements_wavefront_count(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 1.0)
        q.add_or_update(1, 2.0)
        q.pop_min()
        assert q.num_wavefront() == 1

    def test_minimum_after_decrease_key(self):
        """Decrease-key before pop must be respected."""
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 5.0)
        q.add_or_update(1, 3.0)
        q.add_or_update(2, 4.0)
        q.add_or_update(2, 1.0)   # decrease key: 2 is now minimum
        idx, dist = q.pop_min()
        assert idx  == 2
        assert dist == pytest.approx(1.0)

    def test_consecutive_pops_non_decreasing(self):
        """Consecutive pops must yield non-decreasing distances."""
        q = WavefrontPriorityQueue(20)
        dists = [4.0, 1.0, 7.0, 2.0, 6.0, 0.5, 3.0]
        for i, d in enumerate(dists):
            q.add_or_update(i, d)
        prev = -np.inf
        while not q.is_empty():
            idx, dist = q.pop_min()
            assert dist >= prev - 1e-9
            prev = dist

    def test_multiple_decrease_keys_one_entry_popped(self):
        """Only one entry popped per point regardless of how many updates."""
        q = WavefrontPriorityQueue(10)
        q.add_or_update(3, 10.0)
        q.add_or_update(3, 7.0)
        q.add_or_update(3, 5.0)
        q.add_or_update(3, 8.0)   # ignored
        idx, dist = q.pop_min()
        assert idx  == 3
        assert dist == pytest.approx(5.0)
        assert q.is_empty()        # no stale entry left visible


# ---------------------------------------------------------------------------
# Lazy-deletion invariants
# ---------------------------------------------------------------------------

class TestLazyDeletion:
    def test_stale_entry_skipped(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 5.0)
        q.add_or_update(1, 4.0)
        q.add_or_update(0, 2.0)   # invalidates old (5.0) entry
        idx, dist = q.pop_min()
        assert idx  == 0
        assert dist == pytest.approx(2.0)

    def test_no_duplicate_visits(self):
        """Hammer decrease-key; each point must be popped exactly once."""
        q = WavefrontPriorityQueue(5)
        for i in range(5):
            q.add_or_update(i, 100.0)
        for _ in range(3):
            for i in range(5):
                q.add_or_update(i, float(i) + 1.0)
        visited = set()
        while not q.is_empty():
            idx, dist = q.pop_min()
            assert idx not in visited, f"Point {idx} visited twice"
            visited.add(idx)
        assert visited == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

class TestQueryMethods:
    def test_get_wavefront_points(self):
        q = WavefrontPriorityQueue(10)
        for i, d in [(1, 1.0), (2, 2.0), (3, 3.0)]:
            q.add_or_update(i, d)
        assert q.get_wavefront_points() == {1, 2, 3}

    def test_wavefront_excludes_visited(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 1.0)
        q.add_or_update(1, 2.0)
        q.pop_min()
        wf = q.get_wavefront_points()
        assert 0 not in wf
        assert 1 in wf

    def test_get_visited_mask(self):
        q = WavefrontPriorityQueue(5)
        q.set_visited(0, 0.0)
        q.add_or_update(1, 1.0)
        q.pop_min()
        mask = q.get_visited_mask()
        assert mask[0] and mask[1]
        assert not mask[2]

    def test_peek_min_non_destructive(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(3, 5.0)
        q.add_or_update(7, 2.0)
        q.add_or_update(1, 8.0)
        idx, dist = q.peek_min()
        assert idx  == 7
        assert dist == pytest.approx(2.0)
        assert q.num_wavefront() == 3   # not consumed

    def test_len_equals_wavefront_count(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 1.0)
        q.add_or_update(1, 2.0)
        q.add_or_update(2, 3.0)
        assert len(q) == q.num_wavefront() == 3


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

class TestIteration:
    def test_sorted_order(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(3, 3.0)
        q.add_or_update(1, 1.0)
        q.add_or_update(2, 2.0)
        items = list(q)
        assert items[0][0] == 1
        assert items[1][0] == 2
        assert items[2][0] == 3

    def test_non_destructive(self):
        q = WavefrontPriorityQueue(10)
        q.add_or_update(0, 1.0)
        q.add_or_update(1, 2.0)
        list(q)
        assert q.num_wavefront() == 2


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_clears_state(self):
        q = WavefrontPriorityQueue(10)
        q.set_visited(0, 0.0)
        q.add_or_update(1, 1.0)
        q.pop_min()
        q.reset()
        assert q.is_empty()
        assert q.num_visited()   == 0
        assert q.num_wavefront() == 0
        assert q.num_unvisited() == 10
        assert np.all(np.isinf(q.get_all_distances()))

    def test_allows_reuse(self):
        q = WavefrontPriorityQueue(5)
        q.set_visited(0, 0.0)
        q.add_or_update(1, 1.0)
        q.reset()
        q.add_or_update(3, 0.5)
        assert q.get_state(3) == PointState.WAVEFRONT
        assert q.get_state(0) == PointState.UNVISITED


# ---------------------------------------------------------------------------
# Full propagation simulations on known topologies
# ---------------------------------------------------------------------------

class TestPropagationSimulation:
    """Simulate a fast-marching loop to verify the queue semantics end-to-end."""

    @staticmethod
    def _run_dijkstra(n, adj, source=0, edge_weight_fn=None):
        """Run Dijkstra using WavefrontPriorityQueue; return distance array."""
        if edge_weight_fn is None:
            edge_weight_fn = lambda u, v: 1.0
        q = WavefrontPriorityQueue(n)
        dist = np.full(n, np.inf)
        dist[source] = 0.0
        q.set_visited(source, 0.0)
        for nbr in adj[source]:
            new_d = dist[source] + edge_weight_fn(source, nbr)
            q.add_or_update(nbr, new_d)
        while not q.is_empty():
            idx, d = q.pop_min()
            dist[idx] = d
            for nbr in adj[idx]:
                if q.get_state(nbr) != PointState.VISITED:
                    q.add_or_update(nbr, d + edge_weight_fn(idx, nbr))
        return dist

    def test_chain_graph_exact_distances(self):
        n = 12
        adj = {i: [] for i in range(n)}
        for i in range(n - 1):
            adj[i].append(i + 1)
            adj[i + 1].append(i)
        dist = self._run_dijkstra(n, adj)
        expected = np.arange(n, dtype=float)
        assert np.allclose(dist, expected), f"Expected {expected}, got {dist}"

    def test_ring_graph_all_visited(self):
        n = 15
        adj = {i: [(i + 1) % n, (i - 1) % n] for i in range(n)}
        dist = self._run_dijkstra(n, adj)
        assert np.all(np.isfinite(dist))

    def test_pop_order_non_decreasing(self):
        """Popped distances must be monotonically non-decreasing."""
        np.random.seed(7)
        n = 20
        # Complete graph with random weights
        W = np.abs(np.random.randn(n, n)) + 0.1
        W = (W + W.T) / 2
        adj = {i: list(range(n)) for i in range(n)}

        q = WavefrontPriorityQueue(n)
        dist = np.full(n, np.inf)
        dist[0] = 0.0
        q.set_visited(0, 0.0)
        for j in range(1, n):
            q.add_or_update(j, W[0, j])

        pop_dists = []
        while not q.is_empty():
            idx, d = q.pop_min()
            pop_dists.append(d)
            dist[idx] = d
            for j in range(n):
                if q.get_state(j) != PointState.VISITED:
                    q.add_or_update(j, d + W[idx, j])

        for a, b in zip(pop_dists, pop_dists[1:]):
            assert a <= b + 1e-9, f"Non-monotone: {a} then {b}"

    def test_source_always_visited_immediately(self):
        q = WavefrontPriorityQueue(10)
        q.set_visited(0, 0.0)
        assert q.get_state(0) == PointState.VISITED
        assert q.get_distance(0) == 0.0

    def test_disconnected_components(self):
        """Points in disconnected component should remain inf."""
        n = 6
        # Only 0-1-2 connected; 3-4-5 isolated
        adj = {0: [1], 1: [0, 2], 2: [1], 3: [], 4: [], 5: []}
        dist = self._run_dijkstra(n, adj, source=0)
        assert np.all(np.isfinite(dist[:3]))
        assert np.all(np.isinf(dist[3:]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
