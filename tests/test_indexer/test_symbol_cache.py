from pathlib import Path

from src.indexer.symbol_cache import SymbolCache
from src.indexer.repo_map_service import RepoMapService


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "repo_map_sample"


def test_symbol_cache_roundtrip(tmp_path: Path) -> None:
    cache = SymbolCache(tmp_path / "sym.db", FIXTURE)
    service = RepoMapService(FIXTURE, enabled=True, top_k=20, cache_path=tmp_path / "sym.db")
    first = service.initial_build()
    assert first is not None
    assert cache.stats()["symbols"] > 0

    loaded = cache.load_index()
    assert loaded is not None
    assert any(s.name == "AppService" for s in loaded.symbols)


def test_incremental_reindex_updates_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "sym.db"
    service = RepoMapService(FIXTURE, enabled=True, top_k=30, cache_path=cache_path)
    service.initial_build()
    service.mark_dirty("app.py")
    refreshed = service.refresh_if_dirty()
    assert refreshed is not None
    assert refreshed.symbol_count > 0
