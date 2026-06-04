from __future__ import annotations

import logging
import threading
from enum import StrEnum
from pathlib import Path

from src.config.settings import MitKIISettings, get_settings
from src.indexer.ctags import index_project
from src.indexer.repo_map import RankedSymbol, RepoMap, build_repo_map
from src.indexer.symbol_cache import SymbolCache

log = logging.getLogger(__name__)

_FULL_REINDEX_DIRTY_THRESHOLD = 25


class BuildState(StrEnum):
    IDLE = "idle"
    BUILDING = "building"
    READY = "ready"
    ERROR = "error"


class RepoMapService:
    """Repo map lifecycle: SymbolCache persistence + PageRank skeleton."""

    def __init__(
        self,
        project_root: Path,
        *,
        enabled: bool = True,
        top_k: int = 200,
        cache_path: Path | None = None,
        settings: MitKIISettings | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.enabled = enabled
        self.top_k = top_k
        self._settings = settings or get_settings()
        self._map: RepoMap | None = None
        self._dirty = False
        self._dirty_paths: set[str] = set()
        self._lock = threading.Lock()
        self._build_state = BuildState.IDLE
        self._build_error: str | None = None
        self._build_thread: threading.Thread | None = None
        if enabled:
            db = cache_path or (self._settings.data_dir / "repo_map" / "symbols.db")
            self._cache = SymbolCache(db, self.project_root)
        else:
            self._cache = None

    @property
    def build_state(self) -> BuildState:
        with self._lock:
            return self._build_state

    @property
    def build_error(self) -> str | None:
        with self._lock:
            return self._build_error

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._build_state == BuildState.READY and self._map is not None

    def start_background_build(self) -> None:
        """Build repo map on a daemon thread (non-blocking startup)."""
        if not self.enabled:
            return
        with self._lock:
            if self._build_state == BuildState.BUILDING:
                return
            if self._map is not None and not self._dirty:
                self._build_state = BuildState.READY
                return
            self._build_state = BuildState.BUILDING
            self._build_error = None
        thread = threading.Thread(
            target=self._background_build_worker,
            name="mitkii-repo-map",
            daemon=True,
        )
        self._build_thread = thread
        thread.start()

    def _background_build_worker(self) -> None:
        try:
            if self._dirty or self._map is None:
                self.refresh_if_dirty()
            else:
                self.initial_build()
            with self._lock:
                self._build_state = BuildState.READY
        except Exception as exc:
            log.exception("Repo map background build failed")
            with self._lock:
                self._build_state = BuildState.ERROR
                self._build_error = str(exc)

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until background build finishes (or map already ready)."""
        if not self.enabled:
            return True
        thread = self._build_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._lock:
            if self._build_state == BuildState.READY and self._map is not None:
                return True
            if self._build_state == BuildState.ERROR:
                return False
            if self._map is None and self._build_state != BuildState.BUILDING:
                return self.initial_build() is not None
            return self._map is not None

    def initial_build(self) -> RepoMap | None:
        if not self.enabled:
            return None
        indexed = None
        source = "parser"
        if self._cache is not None and not self._cache.is_empty():
            indexed = self._cache.load_index()
            if indexed is not None:
                source = indexed.source

        if indexed is None or not indexed.symbols:
            indexed = index_project(self.project_root)
            source = indexed.source
            if self._cache is not None:
                self._cache.replace_all(indexed)

        self._map = build_repo_map(
            self.project_root,
            top_k=self.top_k,
            indexed=indexed,
        )
        self._map.source = source
        self._dirty = False
        self._dirty_paths.clear()
        with self._lock:
            self._build_state = BuildState.READY
        log.debug(
            "Repo map built: %s symbols in %sms (%s)",
            self._map.symbol_count,
            self._map.build_ms,
            self._map.source,
        )
        return self._map

    def mark_dirty(self, path: str | None = None) -> None:
        if not self.enabled:
            return
        self._dirty = True
        with self._lock:
            if self._build_state == BuildState.READY:
                self._build_state = BuildState.IDLE
        if path:
            norm = path.replace("\\", "/").lstrip("./")
            self._dirty_paths.add(norm)
            log.debug("Repo map marked dirty after change: %s", norm)

    def refresh_if_dirty(self) -> RepoMap | None:
        if not self.enabled:
            return None
        if not self._dirty and self._map is not None:
            return self._map

        prev = self._map.symbol_count if self._map else 0
        indexed = self._reindex_dirty_or_full()
        self._map = build_repo_map(
            self.project_root,
            top_k=self.top_k,
            indexed=indexed,
        )
        self._map.source = indexed.source
        self._dirty = False
        self._dirty_paths.clear()
        with self._lock:
            self._build_state = BuildState.READY
        log.debug(
            "Repo map refreshed: %s → %s symbols (%sms, %s)",
            prev,
            self._map.symbol_count,
            self._map.build_ms,
            indexed.source,
        )
        return self._map

    def _reindex_dirty_or_full(self):
        if self._cache is None or not self._dirty_paths:
            indexed = index_project(self.project_root)
            if self._cache is not None:
                self._cache.replace_all(indexed)
            return indexed

        if len(self._dirty_paths) >= _FULL_REINDEX_DIRTY_THRESHOLD:
            indexed = index_project(self.project_root)
            self._cache.replace_all(indexed)
            return indexed

        self._cache.reindex_paths(self._dirty_paths)
        indexed = self._cache.load_index()
        if indexed is None or not indexed.symbols:
            indexed = index_project(self.project_root)
            self._cache.replace_all(indexed)
        indexed.source = "cache"
        return indexed

    @property
    def map(self) -> RepoMap | None:
        return self.refresh_if_dirty()

    def search(self, query: str, *, limit: int = 20) -> list[RankedSymbol]:
        repo_map = self.refresh_if_dirty()
        if repo_map is None:
            return []
        return repo_map.search(query, limit=limit)

    def to_planner_context(self, *, max_chars: int = 12_000) -> str | None:
        self.wait_until_ready(timeout=self._settings.repo_map_build_timeout)
        repo_map = self.refresh_if_dirty()
        if repo_map is None:
            return None
        return repo_map.to_planner_context(max_chars=max_chars)
