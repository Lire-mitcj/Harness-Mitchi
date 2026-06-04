from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class EventType(StrEnum):
    """All event types emitted during an agent session."""

    THINKING = "thinking"
    STATUS = "status"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"
    FILE_EDIT = "file_edit"
    FINAL_ANSWER = "final_answer"
    ERROR = "error"
    COST_UPDATE = "cost_update"
    CHECKPOINT_SAVED = "checkpoint_saved"
    SCORE_RESULT = "score_result"
    PLAN_UPDATE = "plan_update"
    STREAM_START = "stream_start"
    STREAM_END = "stream_end"


@dataclass(slots=True)
class AgentEvent:
    """Single event produced by the agent runtime.

    Events are the primary mechanism for communicating agent activity to the UI
    layer and to any registered observers (loggers, scorers, etc.).
    """

    type: EventType
    content: str | None = None
    data: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid4().hex)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict suitable for JSON encoding."""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------


def thinking_event(content: str, **data: Any) -> AgentEvent:
    """The model is reasoning internally (optional data e.g. phase=planner)."""
    return AgentEvent(
        type=EventType.THINKING,
        content=content,
        data=dict(data) if data else None,
    )


def tool_call_event(
    name: str,
    params: dict[str, Any],
    *,
    phase: str | None = None,
) -> AgentEvent:
    """A tool invocation has been requested."""
    data: dict[str, Any] = {"tool": name, "params": params}
    if phase:
        data["phase"] = phase
    return AgentEvent(
        type=EventType.TOOL_CALL,
        content=name,
        data=data,
    )


def tool_result_event(
    name: str,
    result: str,
    *,
    success: bool = True,
    phase: str | None = None,
) -> AgentEvent:
    """A tool invocation has completed."""
    data: dict[str, Any] = {"tool": name, "success": success}
    if phase:
        data["phase"] = phase
    return AgentEvent(
        type=EventType.TOOL_RESULT,
        content=result,
        data=data,
    )


def approval_event(action: str, risk_level: str) -> AgentEvent:
    """The agent is requesting human approval before proceeding."""
    return AgentEvent(
        type=EventType.APPROVAL_REQUEST,
        content=action,
        data={"action": action, "risk_level": risk_level},
    )


def file_edit_event(path: str, diff: str, *, auto_apply: bool = False) -> AgentEvent:
    """A file modification is proposed or applied."""
    return AgentEvent(
        type=EventType.FILE_EDIT,
        content=diff,
        data={"path": path, "diff": diff, "auto_apply": auto_apply},
    )


def final_answer_event(content: str, **data: Any) -> AgentEvent:
    """The agent has produced its final answer for this turn."""
    return AgentEvent(
        type=EventType.FINAL_ANSWER,
        content=content,
        data=dict(data) if data else None,
    )


def error_event(message: str, details: dict[str, Any] | None = None) -> AgentEvent:
    """An error occurred during agent execution."""
    return AgentEvent(
        type=EventType.ERROR,
        content=message,
        data={"message": message, **(details or {})},
    )


def cost_event(
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
) -> AgentEvent:
    """Token-usage / cost update for the current session."""
    return AgentEvent(
        type=EventType.COST_UPDATE,
        data={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost": cost,
        },
    )
