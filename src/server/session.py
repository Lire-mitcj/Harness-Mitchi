from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4

log = logging.getLogger(__name__)

_MAX_SESSIONS = 16


class SessionManager:
    """Manages concurrent agent sessions.

    Each session has a unique ID and holds a bag of state (agent loop
    reference, tool registry, turn count, etc.). Old sessions are
    automatically evicted when the limit is reached.
    """

    def __init__(self, max_sessions: int = _MAX_SESSIONS) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._max_sessions = max_sessions

    def create_session(self, metadata: dict[str, Any] | None = None) -> str:
        """Create a new session and return its ID."""
        if len(self._sessions) >= self._max_sessions:
            self._evict_oldest()

        session_id = uuid4().hex[:12]
        self._sessions[session_id] = {
            "id": session_id,
            "created_at": time.time(),
            "status": "idle",
            "turn_count": 0,
            **(metadata or {}),
        }
        log.info("Session created: %s", session_id)
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        session = self._sessions.pop(session_id, None)
        if session is not None:
            log.info("Session ended: %s", session_id)
            return True
        return False

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return summary info for all active sessions."""
        return [
            {
                "id": s["id"],
                "created_at": s["created_at"],
                "status": s["status"],
                "turn_count": s.get("turn_count", 0),
            }
            for s in self._sessions.values()
        ]

    def update_session(self, session_id: str, **kwargs: Any) -> bool:
        """Update fields on an existing session."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.update(kwargs)
        return True

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest_id = min(self._sessions, key=lambda k: self._sessions[k]["created_at"])
        log.warning("Evicting oldest session %s (limit %d reached)", oldest_id, self._max_sessions)
        self._sessions.pop(oldest_id, None)
