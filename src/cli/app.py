from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

app = typer.Typer(
    name="mitkii",
    help="MitKII — AI Code Agent",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

_console = Console()


@app.command()
def chat(
    session_id: Optional[str] = typer.Option(  # noqa: UP007
        None, "--session", "-s", help="Resume a specific session by ID.",
    ),
) -> None:
    """Start an interactive coding session."""
    from src.cli.commands.chat import run_chat

    run_chat(session_id=session_id)


@app.command()
def init(
    path: Optional[str] = typer.Argument(  # noqa: UP007
        None, help="Project directory to initialise (defaults to cwd).",
    ),
) -> None:
    """Initialise MitKII in a project directory."""
    from src.cli.commands.init_cmd import run_init

    run_init(project_path=Path(path) if path else None)


@app.command()
def resume(
    session_id: Optional[str] = typer.Argument(  # noqa: UP007
        None, help="Session ID to resume. Omit to list recent sessions.",
    ),
) -> None:
    """Resume a previous coding session."""
    if session_id is None:
        _console.print("[dim]Recent sessions:[/]")
        _console.print("  [dim]No saved sessions yet.[/]")
        return

    from src.cli.commands.chat import run_chat

    run_chat(session_id=session_id)


@app.command()
def serve() -> None:
    """Start MitKII JSON-RPC server on stdio (same bootstrap as chat)."""
    from src.cli.commands.serve import run_server

    run_server()


@app.command(name="config")
def config_cmd(
    key: Optional[str] = typer.Argument(None, help="Config key to view or set."),  # noqa: UP007
    value: Optional[str] = typer.Argument(None, help="New value to set."),  # noqa: UP007
) -> None:
    """View or modify MitKII configuration."""
    from src.config.settings import get_settings

    settings = get_settings()

    if key is None:
        _console.print("[bold]Current configuration:[/]\n")
        for field_name in settings.model_fields:
            val = getattr(settings, field_name)
            _console.print(f"  [cyan]{field_name}[/] = {val!r}")
        return

    if not hasattr(settings, key):
        _console.print(f"[red]Unknown config key:[/] {key}")
        raise typer.Exit(code=1)

    if value is None:
        _console.print(f"  [cyan]{key}[/] = {getattr(settings, key)!r}")
    else:
        _console.print(f"[dim]Setting config in environment is not yet supported.[/]")
        _console.print(f"  Set [bold]MITKII_{key.upper()}={value}[/] in your .env file.")


def main() -> None:
    """CLI entry point (referenced by pyproject.toml ``[project.scripts]``)."""
    try:
        app()
    except KeyboardInterrupt:
        _console.print("\n[dim]Interrupted.[/]")


if __name__ == "__main__":
    main()
