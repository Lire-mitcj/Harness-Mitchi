from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.status import Status


@contextmanager
def thinking_spinner(
    console: Console,
    message: str = "Thinking...",
) -> Generator[Status, None, None]:
    """Context manager that shows a pulsing spinner while the agent thinks."""
    with console.status(
        message,
        spinner="dots",
        spinner_style="cyan",
    ) as status:
        yield status


class ProgressTracker:
    """Multi-step progress bar for operations like indexing or scanning."""

    def __init__(self, console: Console, description: str = "Processing") -> None:
        self._console = console
        self._description = description
        self._progress: Progress | None = None

    def __enter__(self) -> ProgressTracker:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("[dim]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self._console,
            transient=True,
        )
        self._progress.__enter__()
        self._task_id = self._progress.add_task(self._description, total=None)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._progress is not None:
            self._progress.__exit__(*exc)

    def update(self, advance: int = 1, total: int | None = None, description: str | None = None) -> None:
        if self._progress is None:
            return
        kwargs: dict = {"advance": advance}
        if total is not None:
            kwargs["total"] = total
        if description is not None:
            kwargs["description"] = description
        self._progress.update(self._task_id, **kwargs)

    def set_total(self, total: int) -> None:
        if self._progress is not None:
            self._progress.update(self._task_id, total=total)
