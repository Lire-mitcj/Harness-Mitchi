from __future__ import annotations

from pathlib import Path

from rich.console import Console

from src.cli.spinner import ProgressTracker
from src.cli.theme import get_theme
from src.config.settings import get_settings


def run_init(project_path: Path | None = None) -> None:
    """Initialize a ``.mitkii`` directory in the given project.

    Creates the project-local config structure and performs a quick scan
    to build an initial file index.
    """
    theme = get_theme()
    console = Console(theme=theme.to_rich_theme())
    root = (project_path or Path.cwd()).resolve()
    mitkii_dir = root / ".mitkii"

    if mitkii_dir.exists():
        console.print(f"[dim]Already initialized: {mitkii_dir}[/]")
        return

    console.print(f"[bold cyan]Initializing MitKII[/] in [underline]{root}[/]")

    # Create directory structure
    (mitkii_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (mitkii_dir / "sessions").mkdir(parents=True, exist_ok=True)

    # Write default config
    config_path = mitkii_dir / "config.toml"
    config_path.write_text(
        '# MitKII project configuration\n'
        '# See https://github.com/mitkii/docs for all options.\n'
        '\n'
        '[project]\n'
        f'root = "{root}"\n'
        '\n'
        '[agent]\n'
        'auto_approve_safe = true\n',
        encoding="utf-8",
    )

    # Write a starter rules file
    rules_path = mitkii_dir / "rules.md"
    rules_path.write_text(
        "# Project Rules\n\n"
        "<!-- MitKII reads this file to learn project-specific conventions. -->\n\n"
        "- Follow existing code style and patterns.\n"
        "- Write tests for new functionality.\n"
        "- Keep functions small and focused.\n",
        encoding="utf-8",
    )

    # Quick scan for project overview
    file_count = _scan_project(root, console)

    console.print(f"\n[green]✓[/] Initialized .mitkii in [underline]{root}[/]")
    console.print(f"  Scanned [bold]{file_count}[/] files")
    console.print(f"  Config: [dim]{config_path}[/]")
    console.print(f"  Rules:  [dim]{rules_path}[/]")
    console.print()

    settings = get_settings()
    settings.ensure_dirs()


def _scan_project(root: Path, console: Console) -> int:
    """Walk the project tree, ignoring common non-source directories."""
    ignore_dirs = {
        ".git", ".mitkii", "__pycache__", "node_modules", ".venv", "venv",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
        ".eggs", "*.egg-info",
    }
    count = 0
    with ProgressTracker(console, "Scanning project") as tracker:
        for item in root.rglob("*"):
            if any(part in ignore_dirs for part in item.parts):
                continue
            if item.is_file():
                count += 1
                if count % 50 == 0:
                    tracker.update(advance=50)
    return count
