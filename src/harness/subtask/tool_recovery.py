from __future__ import annotations

from dataclasses import dataclass, field

from src.agent.types import Message, harness_nudge
from src.executor.edit_guard import (
    EDIT_IDENTICAL_HINT,
    edit_failure_hint,
    recent_edit_recoverable_failure,
)
from src.executor.policy import resolve_executor_tools
from src.harness.subtask.context_pipeline import ExecutorRuntimeState
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


@dataclass
class ToolRecoveryResult:
    """Harness post-tool-round recovery (nudges + runtime policy updates)."""

    nudges: list[Message] = field(default_factory=list)
    status_lines: list[str] = field(default_factory=list)
    runtime_tools: frozenset[str] | None = None
    explore_restricted: bool | None = None
    edit_read_fallback: bool | None = None


def apply_post_tool_recovery(
    *,
    subtask: SubTaskNode,
    runtime: ExecutorRuntimeState,
    error_trace: list[str],
    splice_edit: bool,
) -> ToolRecoveryResult:
    """After LLM tool round: Harness-owned retry hints and runtime tool adjustments."""
    out = ToolRecoveryResult()

    if splice_edit:
        if _recent_replace_symbol_anchor_failure(error_trace):
            out.nudges.append(harness_nudge(
                "replace_symbol anchor validation failed after Harness retries. "
                "Revise new_body from <edit_target> original, or report a blocker."
            ))
            out.status_lines.append("splice anchor failed after Harness retries")
        return out

    if subtask.kind == SubTaskKind.DIAGNOSE and _recent_duplicate_explore(error_trace):
        runtime.active_runtime_tools = frozenset()
        runtime.explore_restricted = True
        out.runtime_tools = runtime.active_runtime_tools
        out.explore_restricted = True
        out.nudges.append(harness_nudge(
            "Duplicate exploration was blocked. Stop searching and summarize from "
            "the session summary/tool outputs now."
        ))
        out.status_lines.append("duplicate context_search blocked")
        return out

    if not recent_edit_recoverable_failure(error_trace):
        return out

    hint = edit_failure_hint(error_trace)
    if hint:
        out.nudges.append(harness_nudge(hint))

    if runtime.edit_read_fallback:
        return out

    # Do not enable read_file fallback if the issue was identical strings
    # (i.e., the agent found the string perfectly, but didn't modify it).
    if hint == EDIT_IDENTICAL_HINT:
        return out

    runtime.edit_read_fallback = True
    runtime.explore_restricted = False
    runtime.active_runtime_tools = resolve_executor_tools(
        subtask,
        preloaded_paths=runtime.preloaded_paths,
        truncated_paths=runtime.truncated_paths,
        explore_restricted=False,
        edit_read_fallback=True,
    )
    out.edit_read_fallback = True
    out.explore_restricted = False
    out.runtime_tools = runtime.active_runtime_tools
    out.nudges.append(harness_nudge(
        "edit_file match failed — read_file/read_files enabled. "
        "Read the target lines, copy exact text into old_string, "
        "then call edit_file again."
    ))
    out.status_lines.append("edit failed — read_file enabled for exact copy")
    return out


def _recent_replace_symbol_anchor_failure(error_trace: list[str], *, lookback: int = 4) -> bool:
    for entry in reversed(error_trace[-lookback:]):
        lower = entry.lower()
        if "replace_symbol" in lower and (
            "anchor" in lower or "re-resolve" in lower or "attempt(s)" in lower
        ):
            return True
    return False


def _recent_duplicate_explore(error_trace: list[str], *, lookback: int = 4) -> bool:
    for entry in reversed(error_trace[-lookback:]):
        lower = entry.lower()
        if "blocked duplicate" in lower and (
            "context_search" in lower
        ):
            return True
    return False
