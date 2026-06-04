from __future__ import annotations

from pathlib import Path
from typing import Any


class VectorStore:
    """Local vector store placeholder.

    Will use sqlite-vec extension for production.
    Currently returns empty results as a stub.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        self._initialized = True

    async def add(self, item_id: str, text: str, embedding: list[float]) -> None:
        pass

    async def search(
        self, query_embedding: list[float], top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return []

    async def delete(self, item_id: str) -> None:
        pass

    async def close(self) -> None:
        self._initialized = False
