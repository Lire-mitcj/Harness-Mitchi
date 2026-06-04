from __future__ import annotations

from collections import defaultdict


def pagerank(
    edges: list[tuple[str, str]],
    *,
    nodes: list[str] | None = None,
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute PageRank scores for directed graph edges (src -> dst)."""
    out: dict[str, set[str]] = defaultdict(set)
    incoming: dict[str, set[str]] = defaultdict(set)
    all_nodes: set[str] = set(nodes or [])
    for src, dst in edges:
        if src == dst:
            continue
        out[src].add(dst)
        incoming[dst].add(src)
        all_nodes.add(src)
        all_nodes.add(dst)

    if not all_nodes:
        return {}

    scores = {n: 1.0 / len(all_nodes) for n in all_nodes}
    base = (1.0 - damping) / len(all_nodes)

    for _ in range(iterations):
        next_scores: dict[str, float] = {}
        for node in all_nodes:
            rank = base
            for src in incoming[node]:
                out_degree = len(out[src]) or 1
                rank += damping * scores[src] / out_degree
            next_scores[node] = rank
        scores = next_scores

    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}
