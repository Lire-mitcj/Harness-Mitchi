from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.context.builder import ContextBuilder


@dataclass(frozen=True, slots=True)
class ProjectContextSnapshot:
    """Stable repo_map + directory tree text for prompt-cache L1."""

    text: str
    fingerprint: str


async def get_project_context_snapshot(
    builder: ContextBuilder,
) -> ProjectContextSnapshot | None:
    """Return cached L1 project context (shared by Planner and session)."""
    return await builder.get_project_context_snapshot()
