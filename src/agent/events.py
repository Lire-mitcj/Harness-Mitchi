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


def get_tool_status_text(name: str, arguments: dict[str, Any]) -> str:
    """Map tool name and arguments to a user-friendly Chinese progress message."""
    match name:
        case "codebase_retrieve":
            query = arguments.get("query", "")
            return f"正在检索代码库: {query}…" if query else "正在检索代码库…"
        case "decision_edit":
            target = arguments.get("target_file", "")
            return f"正在编辑文件: {target}…" if target else "正在编辑文件…"
        case "read_file":
            path = arguments.get("path", "")
            return f"正在读取文件: {path}…" if path else "正在读取文件…"
        case "read_files":
            paths = arguments.get("paths", [])
            if isinstance(paths, list) and len(paths) == 1:
                return f"正在读取文件: {paths[0]}…"
            return "正在读取多个文件…"
        case "write_file":
            path = arguments.get("path", "")
            return f"正在写入文件: {path}…" if path else "正在写入文件…"
        case "edit_file":
            path = arguments.get("path", "")
            return f"正在修改文件: {path}…" if path else "正在修改文件…"
        case "delete_file":
            path = arguments.get("path", "")
            return f"正在删除文件: {path}…" if path else "正在删除文件…"
        case "grep_search":
            query = arguments.get("query", "")
            return f"正在搜索代码: {query}…" if query else "正在搜索代码…"
        case "glob_files":
            pattern = arguments.get("pattern", "")
            return f"正在匹配文件模式: {pattern}…" if pattern else "正在搜索文件…"
        case "shell_exec":
            cmd = arguments.get("command", "")
            if len(cmd) > 40:
                cmd = cmd[:37] + "..."
            return f"正在执行命令: {cmd}…" if cmd else "正在执行命令…"
        case "git_status":
            return "正在获取 Git 状态…"
        case "git_commit":
            return "正在提交 Git 变更…"
        case "git_stash":
            return "正在保存 Git 暂存…"
        case "map_search":
            return "正在搜索代码图…"
        case "context_search":
            return "正在搜索上下文…"
        case _:
            return f"running tool {name}..."


PARALLEL_RETRIEVAL_TOOLS = frozenset({
    "codebase_retrieve",
    "view_symbol_code",
    "grep_search",
})

