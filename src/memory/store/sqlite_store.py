from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite


class SQLiteStore:
    """Generic async SQLite wrapper for local persistence."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self, schema_sql: str | None = None) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        self._db.row_factory = aiosqlite.Row
        if schema_sql:
            await self._db.executescript(schema_sql)
            await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def execute(self, sql: str, params: tuple = ()) -> None:
        assert self._db is not None
        await self._db.execute(sql, params)
        await self._db.commit()

    async def fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        assert self._db is not None
        cursor = await self._db.execute(sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
