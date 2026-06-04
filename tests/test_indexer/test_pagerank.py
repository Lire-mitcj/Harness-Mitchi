from src.indexer.pagerank import pagerank


def test_pagerank_hub_gets_higher_score() -> None:
    edges = [
        ("a", "hub"),
        ("b", "hub"),
        ("c", "hub"),
        ("hub", "leaf"),
    ]
    scores = pagerank(edges)
    assert scores["hub"] > scores["a"]
    assert scores["hub"] > scores["b"]
    assert scores["hub"] > scores["c"]
    assert abs(sum(scores.values()) - 1.0) < 1e-6


def test_pagerank_empty_graph() -> None:
    assert pagerank([]) == {}
