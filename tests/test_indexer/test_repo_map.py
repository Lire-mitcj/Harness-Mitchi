from pathlib import Path

import pytest

from src.indexer.repo_map import build_repo_map
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
