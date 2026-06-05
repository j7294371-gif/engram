"""Spreading activation engine for associative memory retrieval.

Implements graph-based spreading activation: given seed nodes,
traverse associations breadth-first with activation decay per hop.
"""

from __future__ import annotations

from collections.abc import Callable


class SpreadingActivation:
    """Graph traversal engine for associative retrieval.

    Uses BFS-style spreading activation from seed node(s).
    Activation decays by a configurable factor per hop.
    """

    def __init__(
        self,
        decay_factor: float = 0.85,
        max_depth: int = 3,
        min_activation: float = 0.01,
    ) -> None:
        self.decay_factor = decay_factor
        self.max_depth = max_depth
        self.min_activation = min_activation

    def activate(
        self,
        seed_id: str,
        get_neighbors: Callable[[str], dict[str, float]],
    ) -> dict[str, float]:
        """Run spreading activation from a seed node.

        Args:
            seed_id: Starting node ID.
            get_neighbors: Function that returns {neighbor_id: edge_strength}
                for a given node ID.

        Returns:
            Dict of {node_id: activation_strength}, excluding the seed.
        """
        visited: dict[str, float] = {}
        queue: list[tuple[str, float, int]] = [(seed_id, 1.0, 0)]

        while queue:
            current_id, activation, depth = queue.pop(0)

            # Pruning
            if depth > self.max_depth:
                continue
            if activation < self.min_activation:
                continue
            if current_id in visited and visited[current_id] >= activation:
                continue

            visited[current_id] = activation

            if depth < self.max_depth:
                neighbors = get_neighbors(current_id)
                for neighbor_id, edge_strength in neighbors.items():
                    next_activation = activation * edge_strength * self.decay_factor
                    queue.append((neighbor_id, next_activation, depth + 1))

        visited.pop(seed_id, None)
        return visited

    def activate_multi(
        self,
        seed_ids: list[str],
        get_neighbors: Callable[[str], dict[str, float]],
    ) -> dict[str, float]:
        """Run spreading activation from multiple seed nodes.

        Activations from multiple seeds are summed (not maxed).
        """
        combined: dict[str, float] = {}
        for sid in seed_ids:
            result = self.activate(sid, get_neighbors)
            for node_id, activation in result.items():
                combined[node_id] = combined.get(node_id, 0.0) + activation
        return combined
