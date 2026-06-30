from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SessionStorage:
    """Manages append-only JSONL transcripts for main agent loops, subagent sidechains, and global history."""

    def __init__(self, transcripts_dir: Path) -> None:
        self.transcripts_dir = transcripts_dir
        self.transcripts_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        return self.transcripts_dir / f"{session_id}.jsonl"

    def _get_sidechain_path(self, session_id: str, subtask_id: str) -> Path:
        d = self.transcripts_dir / "sidechains"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{session_id}_{subtask_id}.jsonl"

    def append_event(self, session_id: str, event_dict: dict[str, Any]) -> None:
        """Append a serialized AgentEvent (or dict) to the main session transcript."""
        p = self._get_session_path(session_id)
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_dict, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Failed to append event to session transcript %s: %s", session_id, exc)

    def append_sidechain_message(
        self,
        session_id: str,
        subtask_id: str,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a subagent or tool-specific LLM transaction message to its sidechain log."""
        p = self._get_sidechain_path(session_id, subtask_id)
        record = {
            "timestamp": time.time(),
            "role": role,
            "content": content,
            "tool_calls": tool_calls,
        }
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Failed to append to sidechain log: %s", exc)

    def load_session_events(self, session_id: str) -> list[dict[str, Any]]:
        """Load and deserialize all events from the main session transcript."""
        p = self._get_session_path(session_id)
        if not p.exists():
            return []
        events = []
        try:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except OSError as exc:
            log.warning("Failed to read session transcript %s: %s", session_id, exc)
        return events

    @staticmethod
    def append_global_history(user_query: str) -> None:
        """Append the query to the user's global REPL prompt history."""
        home_mitkii = Path.home() / ".mitkii"
        home_mitkii.mkdir(parents=True, exist_ok=True)
        p = home_mitkii / "history.jsonl"
        record = {
            "query": user_query,
            "timestamp": time.time(),
        }
        try:
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("Failed to append to global prompt history: %s", exc)
