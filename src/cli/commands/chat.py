from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from rich.console import Console

from src.agent.events import AgentEvent
from src.cli.permissions import CLIPermissionHandler
from src.cli.renderer import CLIRenderer
from src.cli.repl import REPLSession
from src.cli.theme import get_theme
from src.config.settings import get_settings
from src.runtime.session_factory import MitKIISession, create_mitkii_session
from src.agent.state_assembled_loop import StateAssembledLoop


class AgentLoopAdapter:
    """Adapter exposing run_turn/get_state expected by REPLSession."""

    def __init__(self, session: MitKIISession) -> None:
        self.session = session
        self.settings = session.settings
        self._assembled_loop = StateAssembledLoop(
            llm=session.llm,
            tools=session.tools,
            harness=session.harness,
            context=session.context_builder,
            permissions=session.permissions,
            settings=session.settings,
        )
        self._current_loop = self._assembled_loop

    async def run_turn_stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
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
        return self._current_loop.agent_telemetry

    def get_run_state(self) -> Any:
        """Expose the reducer-owned flow state without mixing it with telemetry."""
        return self._current_loop.state.run_state

    def get_context_records(self) -> list[dict[str, Any]]:
        getter = getattr(self._current_loop, "get_context_records", None)
        if getter is not None:
            return cast(list[dict[str, Any]], getter())
        state = self._current_loop.state
        return list(getattr(state, "core_context_history", ()))


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

    mode_label = f"Mode: {settings.mitkii_mode}"
    console.print(f"[dim]MitKII mode: {mode_label}[/]")
    console.print(f"[dim]Project root: {session.project_root}[/]")
    if settings.mitkii_mode == "assembled":
        console.print(
            f"[dim]Coordinating LLM: {settings.model} | "
            f"Decision LLM: {settings.cursor_decision_model}[/]"
        )
        console.print(
            "[dim]Tip: codebase_retrieve and decision_edit will run dynamically "
            "based on coordinate state.[/]"
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
