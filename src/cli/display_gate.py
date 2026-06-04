from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.events import AgentEvent, EventType
from src.cli.report_format import compact_diagnose_report, format_report_for_terminal, milestone_from_report

if TYPE_CHECKING:
    from src.cli.renderer import CLIRenderer

_SUMMARY_MAX = 140

# Executor tools the user must see before permission prompts / after run.
_EXECUTOR_VISIBLE_TOOLS = frozenset({
    "edit_file",
    "write_file",
    "delete_file",
    "shell_exec",
})

_EXECUTOR_ACTIVITY_TOOLS = frozenset({
    "read_file",
    "read_files",
    "grep_search",
    "map_search",
    "glob_files",
    "list_dir",
    "git_status",
})


def summarize_text(text: str, *, max_len: int = _SUMMARY_MAX) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1] + "…"


class OrchestratorDisplayGate:
    """Condense scout/executor noise; keep full plan and terminal answer."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self._pending_answer: dict[str, str] = {}

    def render(self, renderer: CLIRenderer, event: AgentEvent) -> None:
        if not self.enabled:
            renderer.render_event(event)
            return

        et = event.type
        data = event.data or {}

        if et == EventType.PLAN_UPDATE:
            renderer.render_event(event)
            return

        if et == EventType.FINAL_ANSWER:
            if data.get("terminal"):
                self._flush_pending(renderer)
                renderer.render_event(event)
                return
            if data.get("intermediate"):
                sid = str(data.get("subtask_id") or "?")
                self._pending_answer[sid] = event.content or ""
                return
            renderer.render_event(event)
            return

        if et == EventType.STATUS:
            if data.get("spinner_only"):
                if data.get("llm_loading"):
                    renderer.render_event(event)
                return
            milestone = data.get("milestone")
            if milestone == "subtask_start":
                sid = str(data.get("subtask_id") or "?")
                kind = str(data.get("kind") or "")
                desc = summarize_text(str(data.get("description") or ""), max_len=80)
                label = f"{kind}: {desc}" if kind else desc
                renderer.render_subtask_milestone(sid, "start", label)
                return
            if milestone == "subtask_done":
                sid = str(data.get("subtask_id") or "?")
                kind = str(data.get("kind") or "")
                summary = self._pending_answer.pop(sid, "")
                if not summary:
                    summary = str(event.content or "")
                if kind == "diagnose" and summary.strip():
                    renderer.render_subtask_milestone(
                        sid,
                        "done",
                        milestone_from_report(summary),
                        kind=kind,
                    )
                    compact = compact_diagnose_report(summary)
                    if compact:
                        renderer.render_plain_report(compact)
                    return
                renderer.render_subtask_milestone(
                    sid,
                    "done",
                    summarize_text(summary),
                    kind=kind,
                )
                return
            if milestone == "discovery_done":
                renderer.render_status(summarize_text(str(event.content or "Discovery complete")))
                return
            if data.get("executor_activity") and event.content:
                if "calling model" in str(event.content):
                    return
                renderer.render_status(str(event.content))
                return
            content = event.content or ""
            if any(
                token in content
                for token in (
                    "escalating to re-plan",
                    "attempt ",
                    "Preflight BLOCK",
                    "Manifest:",
                    "Orchestrator replan",
                )
            ):
                renderer.render_status(content)
            return

        if et in {EventType.TOOL_CALL, EventType.TOOL_RESULT}:
            phase = data.get("phase")
            tool = str(data.get("tool") or "")
            if phase == "scout":
                if et == EventType.TOOL_RESULT and not data.get("success", True):
                    renderer.render_tool_result(
                        str(data.get("tool", "?")),
                        event.content or "",
                        success=False,
                        phase=str(phase) if phase else None,
                    )
                return
            if phase == "executor":
                if et == EventType.TOOL_CALL and tool in (
                    _EXECUTOR_VISIBLE_TOOLS | _EXECUTOR_ACTIVITY_TOOLS
                ):
                    renderer.render_event(event)
                    return
                if et == EventType.TOOL_RESULT and tool in _EXECUTOR_ACTIVITY_TOOLS:
                    if not data.get("success", True):
                        renderer.render_tool_result(
                            tool or "?",
                            event.content or "",
                            success=False,
                            phase="executor",
                        )
                    return
                if tool in _EXECUTOR_VISIBLE_TOOLS:
                    renderer.render_event(event)
                    return
                if et == EventType.TOOL_RESULT and not data.get("success", True):
                    renderer.render_tool_result(
                        tool or "?",
                        event.content or "",
                        success=False,
                        phase="executor",
                    )
                return
            renderer.render_event(event)
            return

        if et == EventType.SCORE_RESULT:
            return

        if et == EventType.CHECKPOINT_SAVED:
            return

        if et == EventType.ERROR:
            self._flush_pending(renderer)
            renderer.render_event(event)
            return

        if et == EventType.APPROVAL_REQUEST:
            renderer.render_event(event)
            return

        if et == EventType.COST_UPDATE:
            return

        if et == EventType.FILE_EDIT:
            renderer.render_event(event)
            return

        renderer.render_event(event)

    def _flush_pending(self, renderer: CLIRenderer) -> None:
        for sid, text in sorted(self._pending_answer.items()):
            renderer.render_subtask_milestone(sid, "done", summarize_text(text))
        self._pending_answer.clear()
