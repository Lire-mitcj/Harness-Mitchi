from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.markup import escape

from src.agent.events import AgentEvent, EventType
from src.cli.display_gate import OrchestratorDisplayGate
from src.cli.renderer import CLIRenderer
from src.cli.spinner import thinking_spinner

if TYPE_CHECKING:
    from src.cli.permissions import CLIPermissionHandler


class AgentLoopProtocol(Protocol):
    """Minimal interface the REPL expects from the agent loop."""

    async def run_turn(self, user_input: str) -> list[AgentEvent]: ...
    def run_turn_stream(self, user_input: str) -> AsyncIterator[AgentEvent]: ...
    async def resolve_approval(self, action: str, approved: bool) -> None: ...
    async def list_checkpoints(self) -> list[dict[str, Any]]: ...
    def get_probe_metrics(self) -> dict[str, Any]: ...
    async def run_score_now(self) -> dict[str, Any] | None: ...
    def get_state(self) -> Any: ...
    def get_context_records(self) -> list[dict[str, Any]]: ...


SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show this help message",
    "/checkpoint": "Save a checkpoint of the current session",
    "/rollback": "Rollback to a previous checkpoint",
    "/history": "Show conversation history",
    "/compact": "Compress conversation to save context",
    "/plan": "Direct plan: /plan <task> skips Scout; /plan alone shows plan",
    "/score": "Run quality scoring on recent changes",
    "/probe": "Show context probe metrics",
    "/context": "Show Core LLM input context: /context [call|all]",
    "/clear": "Clear the terminal screen",
    "/config": "View or modify config values",
}


_SPINNER_KEEP = frozenset({
    EventType.STREAM_START,
    EventType.STREAM_END,
    EventType.THINKING,
    EventType.COST_UPDATE,
})

_EXECUTOR_SPINNER_KEEP = frozenset({
    EventType.TOOL_CALL,
    EventType.TOOL_RESULT,
    EventType.STREAM_START,
    EventType.STREAM_END,
    EventType.THINKING,
    EventType.STATUS,
})


class REPLSession:
    """Interactive read-eval-print loop for MitKII.

    Reads user input via prompt_toolkit, dispatches to the agent loop,
    and renders streaming events through the CLIRenderer.
    """

    def __init__(
        self,
        agent_loop: AgentLoopProtocol,
        renderer: CLIRenderer,
        permission_handler: CLIPermissionHandler,
        console: Console | None = None,
        *,
        condense_orchestrator: bool = False,
    ) -> None:
        self._agent = agent_loop
        self._renderer = renderer
        self._permissions = permission_handler
        self._console = console or renderer.console
        self._running = True
        self._display_gate = OrchestratorDisplayGate(enabled=condense_orchestrator)

        kb = KeyBindings()

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        self._prompt: PromptSession[str] = PromptSession(
            history=InMemoryHistory(),
            key_bindings=kb,
            multiline=False,
        )

    async def start(self) -> None:
        self._renderer.render_welcome()

        while self._running:
            try:
                user_input = await self._prompt.prompt_async(
                    HTML("<seagreen><b>❯</b></seagreen> "),
                )
            except KeyboardInterrupt:
                self._console.print("\n[dim]Interrupted. Press Ctrl+C again to exit.[/]")
                try:
                    await self._prompt.prompt_async(
                        HTML("<seagreen><b>❯</b></seagreen> "),
                    )
                except (KeyboardInterrupt, EOFError):
                    self._console.print("[dim]Goodbye![/]")
                    break
                continue
            except EOFError:
                self._console.print("[dim]Goodbye![/]")
                break

            text = user_input.strip()
            if not text:
                continue

            # Handle backslash continuation for multiline
            while text.endswith("\\"):
                text = text[:-1] + "\n"
                try:
                    continuation = await self._prompt.prompt_async(
                        HTML("<dim>… </dim>"),
                    )
                    text += continuation.strip()
                except (KeyboardInterrupt, EOFError):
                    break

            if text.startswith("/"):
                handled = await self._handle_slash_command(text)
                if handled:
                    continue

            await self._process_message(text)

    async def _handle_slash_command(self, command: str) -> bool:
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            self._show_help()
            return True

        if cmd == "/clear":
            self._console.clear()
            return True

        if cmd == "/history":
            state = self._agent.get_state()
            msgs = getattr(state, "messages", [])
            if not msgs:
                self._console.print("[dim]No conversation history.[/]")
                return True
            for msg in msgs[-20:]:
                role = getattr(msg, "role", "?")
                content = getattr(msg, "content", "")
                preview = (content[:100] + "...") if len(content) > 100 else content
                self._console.print(f"  [{self._renderer.theme.muted}]{role}:[/] {preview}")
            return True

        if cmd == "/compact":
            self._console.print("[dim]Compressing conversation history...[/]")
            # Delegate to agent loop if it supports compaction
            if hasattr(self._agent, "compact"):
                await self._agent.compact()
                self._console.print("[green]✓ Conversation compacted.[/]")
            else:
                self._console.print("[dim]Compact not yet available.[/]")
            return True

        if cmd == "/plan":
            if arg.strip():
                await self._process_message(f"/plan {arg.strip()}")
                return True
            state = self._agent.get_state()
            plan = getattr(state, "current_plan", None)
            if plan:
                self._renderer.render_plan(plan)
            else:
                self._console.print("[dim]No active plan.[/]")
            return True

        if cmd == "/checkpoint":
            checkpoints = await self._agent.list_checkpoints()
            if not checkpoints:
                self._console.print("[dim]No checkpoints yet.[/]")
                return True
            for cp in checkpoints[-10:]:
                ts = datetime.fromtimestamp(cp.get("timestamp", 0)).strftime("%H:%M:%S")
                self._console.print(
                    f"[dim]{ts}[/] id={cp.get('id')} trigger={cp.get('trigger')} "
                    f"files={cp.get('file_changes', 0)} patch={cp.get('has_patch')}"
                )
            return True

        if cmd == "/score":
            self._console.print("[dim]Running scorer...[/]")
            result = await self._agent.run_score_now()
            if result is None:
                self._console.print("[dim]Scorer unavailable.[/]")
                return True
            self._renderer.render_score(result)
            if result.get("feedback"):
                self._console.print(f"[dim]{result['feedback']}[/]")
            return True

        if cmd == "/probe":
            summary = self._agent.get_probe_metrics()
            if not summary:
                self._console.print("[dim]No probe metrics yet.[/]")
                return True
            self._console.print(
                f"LLM: calls={summary.get('total_calls', 0)} "
                f"tokens={summary.get('total_tokens', 0)} "
                f"cost=${summary.get('total_cost', 0.0):.4f} "
                f"session={summary.get('session_duration_s', 0)}s"
            )
            per_model = summary.get("per_model") or {}
            if per_model:
                self._console.print("[dim]Per model:[/]")
                for model, stats in per_model.items():
                    short = model.split("/")[-1] if "/" in model else model
                    self._console.print(
                        f"  {short}: calls={stats.get('calls', 0)} "
                        "tokens="
                        f"{stats.get('prompt_tokens', 0) + stats.get('completion_tokens', 0)} "
                        f"cost=${stats.get('cost', 0.0):.4f}"
                    )
            phases = summary.get("phases") or []
            if phases:
                self._console.print("[dim]Phases (last turn):[/]")
                for p in phases:
                    sid = p.get("subtask_id") or ""
                    label = f"{p.get('phase')}" + (f"[{sid}]" if sid else "")
                    verdict = p.get("verdict")
                    v = f" {verdict}" if verdict else ""
                    extra = ""
                    if p.get("estimated_tokens"):
                        extra = f" est={p['estimated_tokens']}/{p.get('budget_tokens', '?')}tok"
                    elif p.get("turns_used") is not None:
                        extra = f" turns={p['turns_used']}"
                    if p.get("source"):
                        extra += f" source={p['source']}"
                    self._console.print(
                        f"  {label}: {p.get('duration_ms', 0)}ms{v}{extra}"
                    )
                self._console.print(
                    f"[dim]Phase total: {summary.get('phase_total_ms', 0)}ms | "
                    f"Turn wall: {summary.get('turn_wall_ms', 0)}ms[/]"
                )
            return True

        if cmd == "/context":
            records = self._agent.get_context_records()
            if not records:
                self._console.print("[dim]No Core LLM context has been recorded yet.[/]")
                return True
            selected = records
            label = "all"
            if arg.strip().lower() != "all":
                if arg.strip():
                    try:
                        requested_step = int(arg.strip())
                    except ValueError:
                        self._console.print("[yellow]Usage: /context [call|all][/]")
                        return True
                    selected = [item for item in records if item.get("call") == requested_step]
                    label = f"call {requested_step}"
                    if not selected:
                        self._console.print(f"[dim]No context recorded for call {requested_step}.[/]")
                        return True
                else:
                    selected = [records[-1]]
                    label = f"latest call {records[-1].get('call')} (step {records[-1].get('step')})"
            import json
            self._console.print(f"[bold cyan]Core LLM context — {label}[/]")
            self._console.print_json(json=json.dumps(selected, ensure_ascii=False, default=str))
            return True

        if cmd in ("/rollback", "/config"):
            self._console.print(f"[dim]{cmd} — coming soon.[/]")
            return True

        self._console.print(f"[dim]Unknown command: {cmd}. Type /help for a list.[/]")
        return True

    def _show_help(self) -> None:
        from rich.table import Table

        table = Table(title="Commands", border_style="dim", show_header=False, padding=(0, 2))
        table.add_column("Command", style="bold cyan")
        table.add_column("Description")
        for cmd, desc in SLASH_COMMANDS.items():
            table.add_row(cmd, desc)
        self._console.print(table)

    async def _process_message(self, message: str) -> None:
        spinner_cm: Any = None
        status: Any = None
        spinner_active = False
        executor_spinner_msg = "Thinking..."
        executor_preview_shown = False
        parallel_llm_tasks: dict[str, str] = {}
        thinking_buffer = ""

        def _ensure_spinner(text: str) -> None:
            nonlocal spinner_cm, status, spinner_active, executor_spinner_msg
            executor_spinner_msg = text
            if not spinner_active:
                spinner_cm = thinking_spinner(self._console)
                status = spinner_cm.__enter__()
                spinner_active = True
            # decision_edit progress already carries Rich markup (bold blue path/stats).
            if "[/" in text and ("[bold" in text or "[dim" in text):
                display_text = text
            else:
                display_text = f"[dim cyan]{text}[/]"
            if thinking_buffer:
                import unicodedata
                console_width = self._console.width if self._console.width else 80
                width = min(80, console_width - 4)
                width = max(40, width)
                limit_width = width - 4

                raw_lines = thinking_buffer.splitlines()
                wrapped_lines = []
                for line in raw_lines:
                    if not line.strip():
                        wrapped_lines.append("")
                        continue
                    
                    current_line = []
                    current_width = 0
                    for char in line:
                        # CJK characters take 2 visual columns, ASCII takes 1
                        char_width = 2 if unicodedata.east_asian_width(char) in ('W', 'F', 'A') else 1
                        if current_width + char_width > limit_width:
                            if current_line:
                                wrapped_lines.append("".join(current_line))
                            current_line = [char]
                            current_width = char_width
                        else:
                            current_line.append(char)
                            current_width += char_width
                    if current_line:
                        wrapped_lines.append("".join(current_line))

                max_lines = 6
                total_lines = len(wrapped_lines)
                show_lines = wrapped_lines[-max_lines:]
                if total_lines > max_lines:
                    if len(show_lines) > 0 and show_lines[0] != "...":
                        show_lines = ["..."] + show_lines[1:] if len(show_lines) > 1 else ["..."]

                box_lines = []
                for line in show_lines:
                    box_lines.append(f"  [dim]{escape(line)}[/]")
                box_text = "\n".join(box_lines)
                display_text = f"{display_text}\n{box_text}"
            status.update(display_text)

        def _dismiss_spinner() -> None:
            nonlocal spinner_active
            if spinner_active and spinner_cm is not None:
                spinner_cm.__exit__(None, None, None)
                spinner_active = False

        def _update_parallel_spinner(event: AgentEvent) -> bool:
            data = event.data or {}
            task_id = str(data.get("parallel_task_id") or "")
            if not task_id:
                return False
            state = str(data.get("parallel_state") or "running")
            if state == "done":
                parallel_llm_tasks.pop(task_id, None)
            else:
                parallel_llm_tasks[task_id] = str(event.content or task_id)
            if parallel_llm_tasks:
                _ensure_spinner("  |  ".join(parallel_llm_tasks.values()))
            else:
                _dismiss_spinner()
            return True

        _ensure_spinner("Thinking...")

        def _executor_spinner_label(data: dict[str, Any]) -> str:
            sid = str(data.get("subtask_id") or "?")
            return f"Executor [{sid}]"

        def _reset_executor_preview() -> None:
            nonlocal executor_preview_shown
            executor_preview_shown = False

        def _should_dismiss_spinner(event: AgentEvent) -> bool:
            if event.type in _SPINNER_KEEP:
                return False
            phase = str((event.data or {}).get("phase") or "")
            if phase == "executor" and event.type in _EXECUTOR_SPINNER_KEEP:
                if event.type == EventType.STATUS and (event.data or {}).get("spinner_only"):
                    return False
                if event.type in {
                    EventType.TOOL_CALL,
                    EventType.TOOL_RESULT,
                    EventType.STREAM_START,
                    EventType.STREAM_END,
                    EventType.THINKING,
                }:
                    return False
            return True

        def _handle_llm_stream(event: AgentEvent) -> None:
            nonlocal executor_preview_shown, executor_spinner_msg
            phase = str((event.data or {}).get("phase") or "")
            if phase != "executor":
                return
            if not (event.data or {}).get("preview_line"):
                return
            if executor_preview_shown:
                return
            chunk = (event.content or "").strip()
            if not chunk:
                return
            executor_preview_shown = True
            label = _executor_spinner_label(event.data or {})
            line = chunk if len(chunk) <= 120 else chunk[:119] + "…"
            _dismiss_spinner()
            self._console.print(f"[dim cyan]● {label}[/] [dim]{line}[/]")
            _ensure_spinner(executor_spinner_msg)

        try:
            had_event = False
            async for event in self._agent.run_turn_stream(message):
                had_event = True

                if (
                    event.type == EventType.STATUS
                    and event.content
                    and event.data
                    and event.data.get("spinner_only")
                ):
                    if _update_parallel_spinner(event):
                        continue
                    phase = str(event.data.get("phase") or "")
                    if phase == "executor":
                        _reset_executor_preview()
                    _ensure_spinner(str(event.content))
                    continue

                phase = str((event.data or {}).get("phase") or "")
                if phase == "executor" or event.type in {EventType.TOOL_CALL, EventType.FINAL_ANSWER}:
                    thinking_buffer = ""

                if event.type == EventType.STREAM_START and phase == "executor":
                    _reset_executor_preview()

                if event.type == EventType.THINKING and phase == "executor":
                    if (event.data or {}).get("preview_line"):
                        _handle_llm_stream(event)
                    continue

                if event.type == EventType.STREAM_END and phase == "executor":
                    continue

                if event.type == EventType.COST_UPDATE:
                    continue

                if event.type == EventType.THINKING:
                    phase = str((event.data or {}).get("phase") or "")
                    if phase != "executor":
                        thinking_buffer += event.content or ""
                        _ensure_spinner(executor_spinner_msg)
                    continue

                if event.type == EventType.APPROVAL_REQUEST and event.data:
                    _dismiss_spinner()
                    action = str(event.data.get("action") or event.content or "").strip()
                    risk = event.data.get("risk_level", "moderate")
                    if self._permissions.is_always_approved(action):
                        await self._agent.resolve_approval(action, True)
                    else:
                        self._renderer.render_status(
                            "Waiting for approval input (y/n/a + Enter)..."
                        )
                        approved = await self._permissions.request_approval(action, risk)
                        await self._agent.resolve_approval(action, approved)
                    _ensure_spinner(executor_spinner_msg)
                    continue

                if _should_dismiss_spinner(event):
                    _dismiss_spinner()

                self._display_gate.render(self._renderer, event)

                if not spinner_active and event.type not in {EventType.STREAM_END, EventType.FINAL_ANSWER}:
                    _ensure_spinner(executor_spinner_msg)

            if not had_event:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            self._console.print("[dim]Cancelled.[/]")
            return
        except Exception as exc:
            self._renderer.render_error(f"Agent error: {exc}")
            return
        finally:
            _dismiss_spinner()

    def stop(self) -> None:
        self._running = False
