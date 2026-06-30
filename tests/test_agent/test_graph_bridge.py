from __future__ import annotations

from pathlib import Path

import pytest

from src.tools.assembled.graph_bridge import CursorGraphQueryBridge
from src.tools.assembled.repo_map_lookup import CandidateSymbol
from src.indexer.repo_map import RankedSymbol, RepoMap


def _symbol(
    file_path: str,
    name: str,
    line: int,
    *,
    score: float = 0.05,
) -> RankedSymbol:
    return RankedSymbol(
        file_path=file_path,
        name=name,
        kind="function",
        start_line=line,
        end_line=line + 3,
        signature=f"def {name}():",
        score=score,
        symbol_id=f"{file_path}:{name}:{line}",
    )


class _RepoMapService:
    def __init__(self, repo_map: RepoMap) -> None:
        self.map = repo_map

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        return True


def _repo_map(tmp_path: Path) -> RepoMap:
    render_view = _symbol("app.py", "render_view", 10, score=0.2)
    profile = _symbol("app.py", "my_profile_page", 30, score=0.1)
    flight = _symbol("main.py", "flight_management", 20, score=0.08)
    helper = _symbol("main.py", "build_page", 50, score=0.04)
    symbols = [render_view, profile, flight, helper]
    return RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={
            "app.py": [render_view, profile],
            "main.py": [flight, helper],
        },
        symbols_by_id={symbol.symbol_id: symbol for symbol in symbols},
        file_scores={"app.py": 0.6, "main.py": 0.4},
        reference_edges=[
            (render_view.symbol_id, flight.symbol_id),
            (flight.symbol_id, helper.symbol_id),
            ("file:app.py", "file:main.py"),
        ],
    )


@pytest.mark.asyncio
async def test_graph_bridge_expands_symbol_subgraph(tmp_path: Path) -> None:
    repo_map = _repo_map(tmp_path)
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=2,
        top_symbols=4,
        top_files=2,
    )
    seed = repo_map.symbols_by_id["app.py:render_view:10"]

    result = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))

    assert "render_view" in result.expanded_symbols
    assert "flight_management" in result.expanded_symbols
    assert set(result.expanded_files) == {"main.py", "app.py"}
    assert any(edge.type == "reference" for edge in result.graph_edges)
    assert any(edge.type == "co_occurrence" for edge in result.graph_edges)
    assert any(edge.type == "import" for edge in result.graph_edges)


@pytest.mark.asyncio
async def test_graph_bridge_terminates_on_reference_cycles(tmp_path: Path) -> None:
    repo_map = _repo_map(tmp_path)
    render_id = "app.py:render_view:10"
    profile_id = "app.py:my_profile_page:30"
    repo_map.reference_edges.extend(((render_id, profile_id), (profile_id, render_id)))
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=4,
        top_symbols=4,
    )
    seed = repo_map.symbols_by_id[render_id]

    result = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))

    assert len(result.expanded_symbols) <= 4


@pytest.mark.asyncio
async def test_graph_bridge_is_bounded_and_deterministic(tmp_path: Path) -> None:
    repo_map = _repo_map(tmp_path)
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        top_files=1,
        max_seeds=1,
    )
    seed = repo_map.symbols_by_id["app.py:render_view:10"]

    first = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))
    second = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))

    assert first == second
    assert len(first.expanded_symbols) <= 2
    assert len(first.expanded_files) <= 1


@pytest.mark.asyncio
async def test_graph_bridge_without_repo_map_is_empty() -> None:
    graph = CursorGraphQueryBridge(None)

    result = await graph.expand_candidates(())

    assert result.expanded_symbols == ()
    assert result.expanded_files == ()


@pytest.mark.asyncio
async def test_graph_bridge_ignores_semantics_and_only_traverses_candidates(
    tmp_path: Path,
) -> None:
    repo_map = _repo_map(tmp_path)
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        top_symbols=4,
    )
    seed = repo_map.symbols_by_id["app.py:render_view:10"]

    first = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))
    second = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))

    assert "render_view" in first.expanded_symbols
    assert not any(edge.type == "semantic_alias" for edge in first.graph_edges)
    assert not any(edge.type == "concept_feedback" for edge in first.graph_edges)
    assert second == first


@pytest.mark.asyncio
async def test_graph_bridge_uses_distance_aware_decay(tmp_path: Path) -> None:
    first = _symbol("a.py", "render_view", 1, score=0.0)
    second = _symbol("b.py", "route_handler", 1, score=0.0)
    third = _symbol("c.py", "load_screen", 1, score=0.0)
    symbols = [first, second, third]
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={symbol.file_path: [symbol] for symbol in symbols},
        symbols_by_id={symbol.symbol_id: symbol for symbol in symbols},
        reference_edges=[
            (first.symbol_id, second.symbol_id),
            (second.symbol_id, third.symbol_id),
        ],
    )
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=2,
        top_symbols=3,
        max_seeds=1,
    )
    seed = repo_map.symbols_by_id[first.symbol_id]

    result = await graph.expand_candidates((CandidateSymbol(seed, 1.0),))
    scores = {
        node.name: node.score
        for node in result.graph_nodes
        if node.type != "concept"
    }

    assert set(scores) == {"render_view", "route_handler", "load_screen"}
    assert scores["load_screen"] / scores["route_handler"] == pytest.approx(
        0.7082,
        rel=1e-3,
    )


def test_edge_type_aware_decay() -> None:
    bridge = CursorGraphQueryBridge(None)
    assert bridge._decay("reference", 1) == pytest.approx(0.7408, rel=1e-3)
    assert bridge._decay("reference", 2) == pytest.approx(0.7082, rel=1e-3)
    assert bridge._decay("co_occurrence", 1) == pytest.approx(0.5220, rel=1e-3)
    assert bridge._decay("co_occurrence", 2) == pytest.approx(0.4735, rel=1e-3)

def test_embedder_provider_resolution() -> None:
    from src.indexer.embedder import Embedder
    emb1 = Embedder(model="Qwen/Qwen3-VL-Embedding-8B", provider="openai")
    assert emb1.model == "openai/Qwen/Qwen3-VL-Embedding-8B"

    emb2 = Embedder(model="openai/BAAI/bge-m3", provider="openai")
    assert emb2.model == "openai/BAAI/bge-m3"

    emb3 = Embedder(model="nomic-embed-text", provider="ollama")
    assert emb3.model == "ollama/nomic-embed-text"


def test_query_bridge_fallback_intent_heuristic() -> None:
    from src.tools.assembled.query_bridge import CursorQueryBridge
    bridge = CursorQueryBridge(None)

    assert bridge.fallback("登机牌查询的接口改成使用视图查询").intent == "modify"
    assert bridge.fallback("程序运行报错 ValueError").intent == "debug"
    assert bridge.fallback("列出所有的项目文件").intent == "query"
    assert bridge.fallback("解释一下什么是AST符号图").intent == "explain"


@pytest.mark.asyncio
async def test_graph_bridge_framework_dampening_and_frequency_penalty(tmp_path: Path) -> None:
    query_sym = _symbol("app.py", "api_get", 10, score=0.2)
    rare_sym = _symbol("app.py", "make_boarding_pass_pdf", 30, score=0.1)

    symbols = [query_sym, rare_sym]
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={"app.py": symbols},
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[],
    )

    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        max_seeds=2,
    )

    result = await graph.expand_candidates((
        CandidateSymbol(query_sym, 1.0),
        CandidateSymbol(rare_sym, 1.0),
    ))

    node_scores = {node.name: node.score for node in result.graph_nodes}
    assert node_scores["make_boarding_pass_pdf"] > node_scores.get("api_get", 0.0)


@pytest.mark.asyncio
async def test_graph_bridge_links_embedded_sql_to_view_dependencies(
    tmp_path: Path,
) -> None:
    query = RankedSymbol(
        file_path="service.py",
        name="SELECT:orders",
        kind="dml_select",
        start_line=3,
        end_line=5,
        signature="SELECT * FROM orders",
        score=0.2,
        symbol_id="service.py:SELECT:orders:3",
        tables_referenced=("orders",),
    )
    view = RankedSymbol(
        file_path="schema.sql",
        name="v_order_detail",
        kind="ddl_view",
        start_line=1,
        end_line=3,
        signature="CREATE VIEW v_order_detail AS SELECT * FROM orders",
        score=0.1,
        symbol_id="schema.sql:v_order_detail:1",
        tables_referenced=("orders",),
    )
    symbols = [query, view]
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={
            "service.py": [query],
            "schema.sql": [view],
        },
        symbols_by_id={symbol.symbol_id: symbol for symbol in symbols},
        reference_edges=[],
    )
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        max_seeds=1,
    )

    result = await graph.expand_candidates((CandidateSymbol(query, 1.0),))

    assert "v_order_detail" in result.expanded_symbols
    assert any(edge.type == "db_schema_affinity" for edge in result.graph_edges)


@pytest.mark.asyncio
async def test_graph_bridge_bm25_domain_exemption_and_framework_penalty(tmp_path: Path) -> None:
    api_get_sym = _symbol("app.py", "api_get", 10, score=0.0)
    order_detail_sym = _symbol("app.py", "build_order_detail_sql", 20, score=0.0)

    api_hubs = [
        _symbol("app.py", f"api_util_{i}", 100 + i, score=0.0)
        for i in range(100)
    ]
    order_hubs = [
        _symbol("app.py", f"order_processor_{i}", 300 + i, score=0.0)
        for i in range(100)
    ]

    symbols = [api_get_sym, order_detail_sym] + api_hubs + order_hubs
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={"app.py": symbols},
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[],
    )

    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        max_seeds=2,
    )

    class FakeBridge:
        def __init__(self, concepts):
            self.concepts = concepts

    result = await graph.expand_candidates(
        (
            CandidateSymbol(api_get_sym, 0.05),
            CandidateSymbol(order_detail_sym, 0.05),
        ),
        bridge=FakeBridge(["api", "order"])
    )

    node_scores = {node.name: node.score for node in result.graph_nodes}
    assert node_scores["build_order_detail_sql"] > node_scores.get("api_get", 0.0)
