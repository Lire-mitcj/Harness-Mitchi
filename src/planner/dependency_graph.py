from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class DependencyGraph:
    """DAG of step dependencies with topological ordering and cycle detection."""

    _adjacency: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _in_degree: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _nodes: set[str] = field(default_factory=set)

    def add_node(self, node_id: str) -> None:
        self._nodes.add(node_id)
        if node_id not in self._in_degree:
            self._in_degree[node_id] = 0

    def add_edge(self, from_id: str, to_id: str) -> None:
        self.add_node(from_id)
        self.add_node(to_id)
        self._adjacency[from_id].append(to_id)
        self._in_degree[to_id] += 1

    def topological_sort(self) -> list[str]:
        """Kahn's algorithm. Raises ValueError on cycle."""
        in_degree = dict(self._in_degree)
        queue: deque[str] = deque(n for n in self._nodes if in_degree.get(n, 0) == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for neighbor in self._adjacency.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self._nodes):
            raise ValueError("Cycle detected in dependency graph")
        return result

    def get_ready(self, completed: set[str]) -> list[str]:
        ready: list[str] = []
        for node in self._nodes:
            if node in completed:
                continue
            deps_met = all(
                dep in completed
                for dep in self._nodes
                if node in self._adjacency.get(dep, [])
            )
            if deps_met:
                ready.append(node)
        return ready

    def has_cycle(self) -> bool:
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True
