from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import aiosqlite

from src.indexer.parser import Symbol

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    language TEXT,
    last_indexed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
"""


class IndexStore:
    """SQLite-backed symbol index.

    .. deprecated::
        Use :class:`src.indexer.symbol_cache.SymbolCache` with :class:`RepoMapService`
        instead. IndexStore remains for async callers/tests only.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(SCHEMA_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def upsert_file(
        self,
        path: str,
        file_hash: str,
        language: str | None,
        symbols: list[Symbol],
    ) -> None:
        assert self._db is not None
        import time

        await self._db.execute(
            "INSERT OR REPLACE INTO files (path, hash, language, last_indexed) VALUES (?, ?, ?, ?)",
            (path, file_hash, language, time.time()),
        )
        await self._db.execute("DELETE FROM symbols WHERE file_path = ?", (path,))
        for sym in symbols:
            await self._db.execute(
                "INSERT INTO symbols (file_path, name, kind, start_line, end_line, signature) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (path, sym.name, sym.kind, sym.start_line, sym.end_line, sym.signature),
            )
        await self._db.commit()

    async def search_symbols(self, query: str, limit: int = 20) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT file_path, name, kind, start_line, end_line, signature "
            "FROM symbols WHERE name LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [
            {"file_path": r[0], "name": r[1], "kind": r[2],
             "start_line": r[3], "end_line": r[4], "signature": r[5]}
            for r in rows
        ]

    async def get_file_symbols(self, path: str) -> list[Symbol]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT name, kind, start_line, end_line, signature "
            "FROM symbols WHERE file_path = ?",
            (path,),
        )
        rows = await cursor.fetchall()
        return [
            Symbol(name=r[0], kind=r[1], start_line=r[2], end_line=r[3], signature=r[4])
            for r in rows
        ]

    async def get_file_hash(self, path: str) -> str | None:
        assert self._db is not None
        cursor = await self._db.execute("SELECT hash FROM files WHERE path = ?", (path,))
        row = await cursor.fetchone()
        return row[0] if row else None

    async def get_stats(self) -> dict:
        assert self._db is not None
        files_count = (await (await self._db.execute("SELECT COUNT(*) FROM files")).fetchone())[0]
        symbols_count = (await (await self._db.execute("SELECT COUNT(*) FROM symbols")).fetchone())[0]
        return {"files": files_count, "symbols": symbols_count}
