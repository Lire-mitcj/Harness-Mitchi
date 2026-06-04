from __future__ import annotations

import time
from pathlib import Path

import aiosqlite

LONG_TERM_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project TEXT NOT NULL,
    summary TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sessions_project ON session_summaries(project);
"""


class LongTermMemory:
    """L3: Cross-project persistent memory backed by SQLite.

    Stores session summaries, user preferences, and learned patterns.
    Lives at ~/.mitkii/memory.db.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(LONG_TERM_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def save_session_summary(self, summary: str, project: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO session_summaries (project, summary, created_at) VALUES (?, ?, ?)",
            (project, summary, time.time()),
        )
        await self._db.commit()

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT project, summary, created_at FROM session_summaries "
            "WHERE summary LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [
            {"project": r[0], "content": r[1], "timestamp": r[2]}
            for r in rows
        ]

    async def get_recent_sessions(self, project: str | None = None, limit: int = 10) -> list[dict]:
        assert self._db is not None
        if project:
            cursor = await self._db.execute(
                "SELECT project, summary, created_at FROM session_summaries "
                "WHERE project = ? ORDER BY created_at DESC LIMIT ?",
                (project, limit),
            )
        else:
            cursor = await self._db.execute(
                "SELECT project, summary, created_at FROM session_summaries "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
        rows = await cursor.fetchall()
        return [
            {"project": r[0], "content": r[1], "timestamp": r[2]}
            for r in rows
        ]

    async def get_preference(self, key: str) -> str | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT value FROM user_preferences WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set_preference(self, key: str, value: str) -> None:
        assert self._db is not None
        await self._db.execute(
            "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        await self._db.commit()
