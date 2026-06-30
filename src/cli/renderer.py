from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from src.agent.events import AgentEvent, EventType
from src.cli.theme import MitKIITheme, get_theme

_COMPACT_CALL_TOOLS = frozenset({
    "read_file",
    "read_files",
    "grep_search",
    "map_search",
    "glob_files",
    "list_dir",
    "git_status",
})

# Successful results for these tools are omitted (tool_call line is enough).
_COMPACT_RESULT_SKIP = frozenset({
    "read_file",
    "read_files",
    "list_dir",
    "glob_files",
    "git_status",
})


class CLIRenderer:
    """Routes :class:`AgentEvent` instances to the appropriate Rich renderer."""

    def __init__(self, console: Console | None = None, theme: MitKIITheme | None = None) -> None:
        self.theme = theme or get_theme()
        self.console = console or Console(theme=self.theme.to_rich_theme())
        self._thinking_buffer: list[str] = []

    # ------------------------------------------------------------------
    # Event dispatcher
    # ------------------------------------------------------------------

    def render_event(self, event: AgentEvent) -> None:
        handler = _EVENT_HANDLERS.get(event.type)
        if handler is not None:
            handler(self, event)

    # ------------------------------------------------------------------
    # Individual renderers
    # ------------------------------------------------------------------

    def render_thinking(self, content: str) -> None:
        self.console.print(
            Text("● Thinking  ", style=self.theme.thinking),
            Text(content, style="dim"),
            end="",
        )

    def render_tool_call(
        self,
        name: str,
        params: dict[str, Any],
        *,
        phase: str | None = None,
    ) -> None:
        if name in _COMPACT_CALL_TOOLS:
            self._render_tool_call_compact(name, params, phase=phase)
            return

        params_text = Text()
        for k, v in params.items():
            display_val = str(v)
            if len(display_val) > 120:
                display_val = display_val[:117] + "..."
            params_text.append(f"  {k}: ", style=self.theme.tool_param)
            params_text.append(f"{display_val}\n")

        phase_label = f"{phase} · " if phase else ""
        self.console.print(Panel(
            params_text,
            title=f"[{self.theme.tool}]⚡ {phase_label}{name}[/]",
            title_align="left",
            border_style="blue",
            padding=(0, 1),
        ))

    def _render_tool_call_compact(
        self,
        name: str,
        params: dict[str, Any],
        *,
        phase: str | None = None,
    ) -> None:
        detail = _format_compact_tool_call(name, params)
        phase_label = f"{phase} · " if phase else ""
        self.console.print(
            Text(f"  ⚡ {phase_label}{name}  ", style=self.theme.tool),
            Text(detail, style="dim"),
        )

    def render_tool_result(
        self,
        name: str,
        result: str,
        success: bool = True,
        *,
        phase: str | None = None,
    ) -> None:
        if success and name in _COMPACT_RESULT_SKIP:
            return

        icon = "✓" if success else "✗"
        style = self.theme.success if success else self.theme.error

        if success and name == "grep_search":
            output = _format_grep_count(result)
        else:
            summary = _format_compact_tool_result(name, result) if success else None
            output = summary if summary is not None else result
            if summary is None and len(output) > 500:
                output = output[:497] + "..."

        phase_label = f"{phase} · " if phase else ""
        self.console.print(
            Text(f"  {icon} {phase_label}{name}: ", style=style),
            Text(output, style="dim" if success else self.theme.error),
        )

    def render_approval_request(self, action: str, risk: str) -> None:
        risk_style = {
            "safe": self.theme.success,
            "moderate": self.theme.warning,
            "dangerous": self.theme.error,
        }.get(risk, self.theme.warning)

        self.console.print(Panel(
            Text.assemble(
                ("Action: ", "bold"),
                (action, ""),
                "\n",
                ("Risk: ", "bold"),
                (risk, risk_style),
                "\n\n",
                ("Allow this action? ", "bold"),
                ("[y]es / [n]o / [a]lways", "dim"),
            ),
            title="[bold yellow]⚠ Permission Required[/]",
            title_align="left",
            border_style="yellow",
            padding=(0, 1),
        ))

    def render_file_edit(self, path: str, diff: str) -> None:
        lines: list[str] = []
        for line in diff.splitlines():
            lines.append(line)

        self.console.print(Panel(
            Syntax("\n".join(lines), "diff", theme="monokai", line_numbers=False),
            title=f"[{self.theme.file_path}]{path}[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        ))

    def render_final_answer(self, content: str) -> None:
        self.console.print()
        self.console.print(Markdown(content))
        self.console.print()

    def render_plain_report(self, content: str) -> None:
        """Plain diagnose/edit report — no Markdown re-parse."""
        for line in (content or "").splitlines():
            style = self.theme.success if line.startswith(("✓", "✅")) else ""
            self.console.print(Text(line, style=style))
        self.console.print()

    def render_error(self, message: str) -> None:
        self.console.print(Panel(
            Text(message, style=self.theme.error),
            title="[bold red]Error[/]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        ))

    def render_cost(self, usage_summary: dict[str, Any]) -> None:
        table = Table(title="Token Usage", border_style="dim", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Prompt tokens", f"{usage_summary.get('prompt_tokens', 0):,}")
        table.add_row("Completion tokens", f"{usage_summary.get('completion_tokens', 0):,}")
        table.add_row("Total tokens", f"{usage_summary.get('total_tokens', 0):,}")
        cost = usage_summary.get("cost", 0.0)
        table.add_row("Cost", f"${cost:.4f}", style=self.theme.cost)
        self.console.print(table)

    def render_score(self, score_result: dict[str, Any]) -> None:
        table = Table(title="Quality Score", border_style="dim", show_header=True)
        table.add_column("Check", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details")

        for check in score_result.get("checks", []):
            passed = check.get("passed", False)
            icon = "✓" if passed else "✗"
            style = self.theme.score_pass if passed else self.theme.score_fail
            table.add_row(
                check.get("name", ""),
                Text(icon, style=style),
                check.get("message", ""),
            )

        self.console.print(table)
        gate = score_result.get("gate")
        if gate:
            style = self.theme.score_pass if gate == "PASS" else self.theme.score_fail
            self.console.print(Text(f"  Gate: {gate}", style=style))
        if "blocker_count" in score_result:
            self.console.print(Text(f"  Blockers: {score_result.get('blocker_count', 0)}", style="dim"))
        if score_result.get("needs_retry") is not None:
            retry_text = "yes" if score_result.get("needs_retry") else "no"
            self.console.print(Text(f"  Needs retry: {retry_text}", style="dim"))
        feedback = score_result.get("feedback")
        if feedback:
            self.console.print()
            self.console.print(Markdown(str(feedback)))

    def render_welcome(self) -> None:
        banner = Text.assemble(
            ("MitKII", "bold cyan"),
            (" — AI Code Agent", "dim"),
        )
        subtitle = Text.assemble(
            ("Type your request, or ", "dim"),
            ("/help", "bold"),
            (" for commands. ", "dim"),
            ("Ctrl+C", "bold"),
            (" to exit.", "dim"),
        )
        self.console.print()
        self.console.print(Panel(
            Text.assemble(banner, "\n", subtitle),
            border_style="cyan",
            padding=(1, 2),
        ))
        self.console.print()

    def render_status(self, message: str) -> None:
        self.console.print(Text(f"  ℹ {message}", style=self.theme.muted))

    def render_subtask_milestone(
        self,
        subtask_id: str,
        milestone: str,
        detail: str,
        *,
        kind: str = "",
    ) -> None:
        if milestone == "start":
            prefix = Text(f"  → [{subtask_id}] ", style=self.theme.muted)
            kind_part = Text(f"{kind} ", style="dim") if kind else Text("")
            self.console.print(prefix, kind_part, Text(detail, style="dim"))
            return
        icon = "✓" if milestone == "done" else "○"
        style = self.theme.success if milestone == "done" else self.theme.muted
        kind_suffix = f" ({kind})" if kind else ""
        self.console.print(
            Text(f"  {icon} [{subtask_id}]{kind_suffix} ", style=style),
            Text(detail, style="" if milestone == "done" else "dim"),
        )

    def render_plan(self, plan: str) -> None:
        self.console.print(Panel(
            Markdown(plan),
            title="[bold magenta]Plan[/]",
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
        ))


# ------------------------------------------------------------------
# Event-type → handler mapping
# ------------------------------------------------------------------


def _format_compact_tool_call(name: str, params: dict[str, Any]) -> str:
    if name == "read_file":
        path = params.get("path", "?")
        start = params.get("start_line")
        end = params.get("end_line")
        if start is not None and end is not None:
            return f"{path}:{start}-{end}"
        return str(path)
    if name == "read_files":
        paths = params.get("paths")
        if isinstance(paths, list):
            shown = [str(p) for p in paths[:4]]
            suffix = f" +{len(paths) - 4} more" if len(paths) > 4 else ""
            return ", ".join(shown) + suffix
        return "?"
    if name == "grep_search":
        pattern = params.get("pattern", "?")
        include = params.get("include") or params.get("path") or "*"
        return f'"{pattern}" in {include}'
    if name == "map_search":
        query = params.get("query", "?")
        limit = params.get("limit")
        if limit is not None:
            return f'"{query}" (limit={limit})'
        return f'"{query}"'
    if name == "glob_files":
        return str(params.get("pattern", "?"))
    if name == "list_dir":
        return str(params.get("path", "."))
    if name == "git_status":
        return str(params.get("subcommand", "status"))
    if name == "edit_file":
        path = params.get("path", "?")
        old = params.get("old_string", "")
        preview = str(old).replace("\n", " ")[:80]
        return f"{path}  replace: {preview}…" if preview else str(path)
    if name == "write_file":
        path = params.get("path", "?")
        content = str(params.get("content", ""))
        return f"{path}  ({len(content)} chars)"
    if name == "shell_exec":
        cmd = params.get("command", "?")
        wd = params.get("working_dir")
        if wd:
            return f"$ {cmd}  (cwd={wd})"
        return f"$ {cmd}"
    return str(params)[:120]


def _format_compact_tool_result(name: str, result: str) -> str | None:
    if name == "grep_search":
        return _format_grep_count(result)
    if name == "shell_exec":
        lines = result.strip().splitlines()
        if len(lines) > 8:
            head = "\n".join(lines[:4])
            return f"{head}\n  ... ({len(lines)} lines)"
    return None


def _format_grep_count(result: str) -> str:
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return "grep result"
    if isinstance(payload, dict):
        returned = int(payload.get("returned_matches", len(payload.get("matches") or [])))
        total = int(payload.get("total_matches", returned))
        suffix = f" of {total}" if total != returned else ""
        return f"{returned}{suffix} match(es)"
    if isinstance(payload, list):
        return f"{len(payload)} match(es)"
    return "0 match(es)"


def _on_thinking(r: CLIRenderer, e: AgentEvent) -> None:
    # LLM stream (incl. CoT traces) stays on the spinner — not printed.
    return


def _on_tool_call(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_tool_call(
            e.data.get("tool", "?"),
            e.data.get("params", {}),
            phase=e.data.get("phase"),
        )


def _on_tool_result(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_tool_result(
            e.data.get("tool", "?"),
            e.content or "",
            success=e.data.get("success", True),
            phase=e.data.get("phase"),
        )


def _on_approval(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_approval_request(
            e.data.get("action", "?"),
            e.data.get("risk_level", "moderate"),
        )


def _on_file_edit(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_file_edit(e.data.get("path", "?"), e.data.get("diff", ""))


def _on_final_answer(r: CLIRenderer, e: AgentEvent) -> None:
    if e.content:
        r.render_final_answer(e.content)


def _on_error(r: CLIRenderer, e: AgentEvent) -> None:
    r.render_error(e.content or "Unknown error")


def _on_cost(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_cost(e.data)


def _on_score(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data:
        r.render_score(e.data)


def _on_plan(r: CLIRenderer, e: AgentEvent) -> None:
    if e.content:
        r.render_plan(e.content)


def _on_status(r: CLIRenderer, e: AgentEvent) -> None:
    if e.content:
        r.render_status(e.content)


def _on_checkpoint_saved(r: CLIRenderer, e: AgentEvent) -> None:
    if e.data and e.data.get("checkpoint_id"):
        r.render_status(f"Checkpoint saved: {e.data['checkpoint_id']}")


_EVENT_HANDLERS: dict[EventType, Any] = {
    EventType.THINKING: _on_thinking,
    EventType.TOOL_CALL: _on_tool_call,
    EventType.TOOL_RESULT: _on_tool_result,
    EventType.APPROVAL_REQUEST: _on_approval,
    EventType.FILE_EDIT: _on_file_edit,
    EventType.FINAL_ANSWER: _on_final_answer,
    EventType.ERROR: _on_error,
    EventType.COST_UPDATE: _on_cost,
    EventType.SCORE_RESULT: _on_score,
    EventType.PLAN_UPDATE: _on_plan,
    EventType.STATUS: _on_status,
    EventType.CHECKPOINT_SAVED: _on_checkpoint_saved,
}
