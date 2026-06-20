from __future__ import annotations

import asyncio
import shutil
from pathlib import Path


class CursorHarnessContext:
    """Async copy-on-write transaction context for target files."""

    def __init__(self, project_root: Path, file_path: str) -> None:
        self.project_root = project_root
        self.file_path = file_path
        self.absolute_path = (project_root / file_path).resolve()
        self.checkpoint_path = self.absolute_path.with_suffix(
            self.absolute_path.suffix + ".checkpoint"
        )
        self.is_new_file = not self.absolute_path.exists()
        self.committed = False

    async def __aenter__(self) -> CursorHarnessContext:
        if not self.is_new_file:
            await asyncio.to_thread(shutil.copy2, self.absolute_path, self.checkpoint_path)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self.committed:
            await asyncio.to_thread(self.rollback)
        else:
            await asyncio.to_thread(self.cleanup)

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        if self.checkpoint_path.exists():
            try:
                shutil.move(self.checkpoint_path, self.absolute_path)
            except OSError:
                pass
        elif self.is_new_file and self.absolute_path.exists():
            try:
                self.absolute_path.unlink()
            except OSError:
                pass

    def cleanup(self) -> None:
        if self.checkpoint_path.exists():
            try:
                self.checkpoint_path.unlink()
            except OSError:
                pass
