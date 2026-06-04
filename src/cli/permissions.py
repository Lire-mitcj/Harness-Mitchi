from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from rich.console import Console

from src.agent.types import RiskLevel
from src.cli.renderer import CLIRenderer


class CLIPermissionHandler:
    """Interactive permission prompt for the CLI.

    Presents a Rich-formatted approval dialog and collects the user's
    response via prompt_toolkit.  Supports ``y`` (yes), ``n`` (no), and
    ``a`` (always allow for this action within the current session).
    """

    def __init__(
        self,
        renderer: CLIRenderer,
        console: Console | None = None,
    ) -> None:
        self._renderer = renderer
        self._console = console or renderer.console
        self._session_always: set[str] = set()
        self._prompt_session: PromptSession[str] = PromptSession()

    async def request_approval(self, action: str, risk_level: RiskLevel | str) -> bool:
        risk_str = risk_level if isinstance(risk_level, str) else risk_level.value

        if action in self._session_always:
            self._console.print(f"  [dim]↳ auto-approved (always): {action}[/]")
            return True

        self._renderer.render_approval_request(action, risk_str)

        while True:
            try:
                answer = await self._prompt_session.prompt_async(
                    HTML("<b>[y/n/a]</b> ▸ "),
                )
            except (EOFError, KeyboardInterrupt):
                return False

            answer = answer.strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            if answer in ("a", "always"):
                self._session_always.add(action)
                self._console.print(
                    f"  [dim]↳ will auto-approve '{action}' for this session[/]"
                )
                return True

            self._console.print("  [dim]Please enter y, n, or a[/]")

    def is_always_approved(self, action: str) -> bool:
        return action in self._session_always

    def reset(self) -> None:
        self._session_always.clear()
