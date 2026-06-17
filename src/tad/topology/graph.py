"""Network topology graph (Section 11.3).

Nodes are linear maps; edges encode composition (output of map i feeds map j).
The graph supplies adjacency for the topology-aware predictor and for the
shuffled-graph control (Section 13.12 / interpretation rule 9).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class TopologyGraph:
    node_names: List[str]
    edges: List[Tuple[int, int]]  # (src, dst) composition edges
    node_shapes: List[Tuple[int, int]] = field(default_factory=list)
    residual_edges: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.node_names)

    def adjacency(self) -> np.ndarray:
        A = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for s, d in self.edges:
            A[s, d] = 1.0
        return A

    def shuffled(self, rng: np.random.Generator) -> "TopologyGraph":
        """Edge-shuffled control graph with the same number of edges."""
        n = self.num_nodes
        possible = [(i, j) for i in range(n) for j in range(n) if i != j]
        idx = rng.choice(len(possible), size=min(len(self.edges), len(possible)), replace=False)
        new_edges = [possible[i] for i in idx]
        return TopologyGraph(self.node_names, new_edges, self.node_shapes, [])


def from_model(model) -> TopologyGraph:
    """Build the sequential (or residual) composition graph from a TADModel."""
    names = list(model.layer_names)
    shapes = [tuple(model.weight(n).shape) for n in names]
    edges = [(i, i + 1) for i in range(len(names) - 1)]
    residual = []
    if getattr(model, "residual", False):
        residual = [(i, i) for i in range(len(names))]
    return TopologyGraph(names, edges, shapes, residual)
