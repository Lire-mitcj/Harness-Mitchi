from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rich.console import Console

from src.agent.cursor_loop import CursorLoop
from src.agent.events import AgentEvent
from src.cli.permissions import CLIPermissionHandler
from src.cli.renderer import CLIRenderer
from src.cli.repl import REPLSession
from src.cli.theme import get_theme
from src.config.settings import get_settings
from src.orchestrator.orchestrator import OrchestratorLoop
from src.runtime.session_factory import MitKIISession, create_mitkii_session


class AgentLoopAdapter:
    """Adapter exposing run_turn/get_state expected by REPLSession."""

    def __init__(self, session: MitKIISession) -> None:
        self._orchestrator_loop = OrchestratorLoop(
            llm=session.llm,
            tools=session.tools,
            harness=session.harness,
            context=session.context_builder,
            permissions=session.permissions,
            settings=session.settings,
        )
        self._cursor_loop = CursorLoop(
            llm=session.cursor_decision_llm,
            inter_llm=session.cursor_inter_llm,
            decision_llm=session.cursor_decision_llm,
            tools=session.tools,
            harness=session.harness,
            context=session.context_builder,
            permissions=session.permissions,
            settings=session.settings,
        )
        self._current_loop: CursorLoop | OrchestratorLoop = self._cursor_loop

    async def run_turn_stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        stripped = user_input.strip().lower()
        if stripped.startswith("/plan ") or stripped.startswith("/plan:"):
            self._current_loop = self._orchestrator_loop
        else:
            self._current_loop = self._cursor_loop

        async for event in self._current_loop.run(user_input):
            yield event

    async def run_turn(self, user_input: str) -> list[AgentEvent]:
        events: list[AgentEvent] = []
        async for event in self.run_turn_stream(user_input):
            events.append(event)
        return events

    async def resolve_approval(self, action: str, approved: bool) -> None:
        await self._current_loop.resolve_approval(action, approved)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        return await self._current_loop.list_checkpoints()

    def get_probe_metrics(self) -> dict[str, Any]:
        return self._current_loop.get_probe_metrics()

    async def run_score_now(self) -> dict[str, Any] | None:
        return await self._current_loop.run_score_now()

    def get_state(self) -> Any:
        state = self._current_loop.state
        return state.agent_state if hasattr(state, "agent_state") else state


def run_chat(session_id: str | None = None, project_path: Path | None = None) -> None:
    """Set up and launch the interactive chat REPL."""
    settings = get_settings()
    theme = get_theme()
    console = Console(theme=theme.to_rich_theme())
    renderer = CLIRenderer(console=console, theme=theme)
    permission_handler = CLIPermissionHandler(renderer=renderer, console=console)

    session = create_mitkii_session(project_root=project_path)
    if session.repo_map_service is not None:
        console.print("[dim]Repo map building in background (ctags/parser + PageRank)...[/]")

    mode_label = "CursorLoop (/plan -> Orchestrator)"
    console.print(f"[dim]MitKII mode: {mode_label}[/]")
    console.print(f"[dim]Project root: {session.project_root}[/]")
    if settings.orchestrator_mode:
        console.print(
            f"[dim]Planner: {settings.effective_planner_model} | "
            f"Executor: {settings.model}[/]"
        )
        console.print(
            "[dim]Tip: /plan skips Scout; Planner picks subtask kinds - "
            "read/grep happen inside each step[/]"
        )

    agent_loop = AgentLoopAdapter(session)
    repl = REPLSession(
        agent_loop=agent_loop,
        renderer=renderer,
        permission_handler=permission_handler,
        console=console,
        condense_orchestrator=False,
    )

    try:
        asyncio.run(repl.start())
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/]")
