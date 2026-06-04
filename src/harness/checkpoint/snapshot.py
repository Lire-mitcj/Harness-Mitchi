from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Snapshot:
    """Immutable record of agent state at a point in time.

    Designed for serialization to/from JSON on disk so the agent can
    roll back to any previous checkpoint.
    """

    id: str
    trigger: str
    timestamp: float
    messages: list[dict[str, Any]]
    file_changes: list[str]
    git_patch: str | None = None
    memory_snapshot: dict[str, Any] | None = None
    plan_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            id=data["id"],
            trigger=data["trigger"],
            timestamp=data["timestamp"],
            messages=data.get("messages", []),
            file_changes=data.get("file_changes", []),
            git_patch=data.get("git_patch"),
            memory_snapshot=data.get("memory_snapshot"),
            plan_state=data.get("plan_state"),
        )
