from __future__ import annotations

from pathlib import Path

import aiosqlite

PROJECT_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class ProjectMemory:
    """L2: Project-level memory backed by SQLite.

    Stores project structure summaries, rules, conventions, and indexed knowledge.
    Lives in `.mitkii/` within the project.
    """

    def __init__(self, mitkii_dir: Path) -> None:
        self.mitkii_dir = mitkii_dir
        self.db_path = mitkii_dir / "project_memory.db"
        self._db: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self.mitkii_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self.db_path))
        await self._db.executescript(PROJECT_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def load_rules(self) -> str | None:
        rules_path = self.mitkii_dir / "rules.md"
        if rules_path.exists():
            return rules_path.read_text(encoding="utf-8")
        return None

    async def get(self, key: str) -> str | None:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT value FROM project_facts WHERE key = ?", (key,)
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def set(self, key: str, value: str) -> None:
        assert self._db is not None
        import time
        await self._db.execute(
            "INSERT OR REPLACE INTO project_facts (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, time.time()),
        )
        await self._db.commit()

    async def search(self, query: str, limit: int = 5) -> list[dict]:
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT key, value FROM project_facts WHERE value LIKE ? LIMIT ?",
            (f"%{query}%", limit),
        )
        rows = await cursor.fetchall()
        return [{"key": r[0], "content": r[1]} for r in rows]

    async def get_project_summary(self) -> str | None:
        return await self.get("project_summary")

    async def set_project_summary(self, summary: str) -> None:
        await self.set("project_summary", summary)
