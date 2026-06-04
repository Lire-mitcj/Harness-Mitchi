from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum


class FileAction(StrEnum):
    READ = "read"
    WRITE = "write"
    EDIT = "edit"


@dataclass(slots=True)
class FileEvent:
    path: str
    action: FileAction
    timestamp: float = field(default_factory=time.time)


class FileTracker:
    """Tracks file operations within the current agent session.

    Maintains a chronological log of every read/write/edit and an ordered
    set of recently-touched files for quick context assembly.
    """

    def __init__(self) -> None:
        self._log: list[FileEvent] = []
        self._recent: OrderedDict[str, FileAction] = OrderedDict()
        self._on_change: Callable[[str, FileAction], None] | None = None

    def set_change_callback(
        self, callback: Callable[[str, FileAction], None] | None
    ) -> None:
        self._on_change = callback

    def record_read(self, path: str) -> None:
        self._record(path, FileAction.READ)

    def record_write(self, path: str) -> None:
        self._record(path, FileAction.WRITE)

    def record_edit(self, path: str) -> None:
        self._record(path, FileAction.EDIT)

    def _record(self, path: str, action: FileAction) -> None:
        self._log.append(FileEvent(path=path, action=action))
        # Move to end so most-recent is last
        self._recent.pop(path, None)
        self._recent[path] = action
        if action in (FileAction.WRITE, FileAction.EDIT) and self._on_change:
            self._on_change(path, action)

    def get_recent(self, limit: int = 5) -> list[str]:
        """Return the *limit* most recently touched file paths (newest first)."""
        items = list(self._recent.keys())
        items.reverse()
        return items[:limit]

    def get_changes(self) -> list[dict[str, str]]:
        """Return all recorded file events as a list of dicts."""
        return [
            {"path": e.path, "action": e.action.value, "timestamp": str(e.timestamp)}
            for e in self._log
        ]

    def get_modified_files(self) -> list[str]:
        """Return paths of files that were written or edited (no duplicates)."""
        seen: set[str] = set()
        result: list[str] = []
        for e in self._log:
            if e.action in (FileAction.WRITE, FileAction.EDIT) and e.path not in seen:
                seen.add(e.path)
                result.append(e.path)
        return result

    def has_been_read(self, path: str) -> bool:
        return path in self._recent

    def clear(self) -> None:
        self._log.clear()
        self._recent.clear()

    def to_dict(self) -> dict:
        return {
            "log": [
                {"path": e.path, "action": e.action.value, "ts": e.timestamp}
                for e in self._log
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> FileTracker:
        tracker = cls()
        for entry in data.get("log", []):
            tracker._record(entry["path"], FileAction(entry["action"]))
        return tracker
