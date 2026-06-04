from __future__ import annotations

from pathlib import Path

from src.indexer.parser import CodeParser
from src.indexer.symbol_cache import SymbolCache


class IncrementalIndexer:
    """Incremental file reindexing via :class:`SymbolCache` (RepoMap backend)."""

    def __init__(
        self,
        cache: SymbolCache,
        parser: CodeParser | None = None,
    ) -> None:
        self.cache = cache
        self.parser = parser or CodeParser()

    def update(self, files: list[Path]) -> dict[str, int]:
        rel_paths: set[str] = set()
        stats = {"indexed": 0, "skipped": 0, "errors": 0}
        root = self.cache.project_root
        for path in files:
            try:
                rel = path.resolve().relative_to(root).as_posix()
            except ValueError:
                stats["errors"] += 1
                continue
            abs_path = root / rel
            if not abs_path.is_file():
                stats["errors"] += 1
                continue
            if self.cache.file_hash(abs_path) == self._stored_hash(rel):
                stats["skipped"] += 1
                continue
            rel_paths.add(rel)
        if rel_paths:
            self.cache.reindex_paths(rel_paths)
            stats["indexed"] = len(rel_paths)
        return stats

    def _stored_hash(self, rel: str) -> str | None:
        with self.cache._connect() as conn:
            row = conn.execute("SELECT hash FROM files WHERE path = ?", (rel,)).fetchone()
            return str(row[0]) if row else None
