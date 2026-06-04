from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from rich.console import Console

from src.agent.loop import AgentLoop
from src.cli.permissions import CLIPermissionHandler
from src.cli.renderer import CLIRenderer
from src.cli.repl import REPLSession
from src.cli.theme import get_theme
from src.config.settings import get_settings
from src.orchestrator.orchestrator import OrchestratorLoop
from src.runtime.session_factory import create_mitkii_session


class AgentLoopAdapter:
    """Adapter exposing run_turn/get_state expected by REPLSession."""

    def __init__(self, loop: AgentLoop | OrchestratorLoop) -> None:
        self._loop = loop

    async def run_turn_stream(self, user_input: str) -> AsyncIterator:
        async for event in self._loop.run(user_input):
            yield event

    async def run_turn(self, user_input: str):
        events = []
        async for event in self.run_turn_stream(user_input):
            events.append(event)
        return events

    async def resolve_approval(self, action: str, approved: bool) -> None:
        await self._loop.resolve_approval(action, approved)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        return await self._loop.list_checkpoints()

    def get_probe_metrics(self) -> dict[str, Any]:
        return self._loop.get_probe_metrics()

    async def run_score_now(self) -> dict[str, Any] | None:
        return await self._loop.run_score_now()

    def get_state(self):
        state = self._loop.state
        return state.agent_state if hasattr(state, "agent_state") else state


def run_chat(session_id: str | None = None) -> None:
    """Set up and launch the interactive chat REPL."""
    settings = get_settings()
    theme = get_theme()
    console = Console(theme=theme.to_rich_theme())
    renderer = CLIRenderer(console=console, theme=theme)
    permission_handler = CLIPermissionHandler(renderer=renderer, console=console)

    session = create_mitkii_session()
    if session.repo_map_service is not None:
        console.print("[dim]Repo map building in background (ctags/parser + PageRank)...[/]")

    core_loop = session.create_core_loop()
    mode_label = (
        "Scout→Planner→Executor" if settings.orchestrator_mode else "legacy ReAct"
    )
    console.print(f"[dim]MitKII mode: {mode_label}[/]")
    if settings.orchestrator_mode:
        console.print(
            f"[dim]Planner: {settings.effective_planner_model} | "
            f"Executor: {settings.model}[/]"
        )
        console.print(
            "[dim]Tip: /plan skips Scout; Planner picks subtask kinds — read/grep happen inside each step[/]"
        )

    agent_loop = AgentLoopAdapter(core_loop)
    repl = REPLSession(
        agent_loop=agent_loop,
        renderer=renderer,
        permission_handler=permission_handler,
        console=console,
        condense_orchestrator=settings.orchestrator_mode,
    )

    try:
        asyncio.run(repl.start())
    except KeyboardInterrupt:
        console.print("\n[dim]Session ended.[/]")
