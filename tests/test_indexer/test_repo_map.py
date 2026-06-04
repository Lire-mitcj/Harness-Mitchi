from pathlib import Path

import pytest

from src.indexer.repo_map import build_repo_map
from src.indexer.ctags import CtagsIndexResult, CtagsSymbol
from src.indexer.repo_map_service import RepoMapService
from src.tools.search.map_search import MapSearchTool

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "repo_map_sample"


def test_search_uses_all_symbols_not_only_top_k() -> None:
    repo_map = build_repo_map(FIXTURE_ROOT, top_k=2)
    assert len(repo_map.symbols) == 2
    assert len(repo_map.all_symbols) > len(repo_map.symbols)
    hits = repo_map.search("Helper", limit=5)
    assert hits
    assert hits[0].name == "Helper"


def test_build_repo_map_includes_sql_symbols() -> None:
    repo_map = build_repo_map(FIXTURE_ROOT, top_k=50)
    names = {s.name for s in repo_map.all_symbols}
    assert "v_boarding_pass" in names


def test_repo_map_skeleton_includes_search_modules() -> None:
    repo_map = build_repo_map(FIXTURE_ROOT, top_k=50)
    block = repo_map.to_skeleton_block(max_chars=12_000)
    assert "## Search modules" in block
    assert "one module" in block
    assert "patterns=" in block
    assert "boarding" in block or "v_boarding_pass" in block


def test_repo_map_expands_symbol_reference_edges() -> None:
    repo_map = build_repo_map(FIXTURE_ROOT, top_k=50)
    app = repo_map.search("AppService", limit=1)[0]

    edges = repo_map.expand_symbol_edges([app.symbol_id], depth=2)

    assert any(src.name == "AppService" and dst.name == "BaseService" for src, dst in edges)


def test_repo_map_expands_function_call_chain_from_indexed_references(tmp_path: Path) -> None:
    indexed = CtagsIndexResult(
        symbols=[
            CtagsSymbol("main.py", "query_orders", "function", 1, 2, "def query_orders()"),
            CtagsSymbol("main.py", "build_order_query", "function", 4, 5, "def build_order_query()"),
            CtagsSymbol("main.py", "format_order_response", "function", 7, 8, "def format_order_response()"),
        ],
        references=[
            ("main.py:query_orders:1", "build_order_query"),
            ("main.py:build_order_query:4", "format_order_response"),
        ],
        source="test",
    )
    repo_map = build_repo_map(tmp_path, top_k=50, indexed=indexed)
    focused = repo_map.search("query_orders", limit=1)[0]

    edges = repo_map.expand_symbol_edges([focused.symbol_id], depth=2)

    assert [(src.name, dst.name) for src, dst in edges] == [
        ("query_orders", "build_order_query"),
        ("build_order_query", "format_order_response"),
    ]


@pytest.mark.asyncio
async def test_map_search_tool(tmp_path: Path) -> None:
    service = RepoMapService(
        FIXTURE_ROOT,
        enabled=True,
        top_k=10,
        cache_path=tmp_path / "symbols.db",
    )
    service.initial_build()
    tool = MapSearchTool(service)
    result = await tool.execute(query="boarding")
    assert result.success
    assert "v_boarding_pass" in result.output


@pytest.mark.asyncio
async def test_repo_map_service_refresh_on_dirty(tmp_path: Path) -> None:
    service = RepoMapService(
        FIXTURE_ROOT,
        enabled=True,
        top_k=50,
        cache_path=tmp_path / "symbols.db",
    )
    first = service.initial_build()
    assert first is not None
    service.mark_dirty("schema.sql")
    second = service.refresh_if_dirty()
    assert second is not None
    assert second.build_ms >= 0
