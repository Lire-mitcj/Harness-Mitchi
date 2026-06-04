from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from src.harness.checkpoint.snapshot import Snapshot
from src.harness.checkpoint.store import CheckpointStore

log = logging.getLogger(__name__)


class RollbackManager:
    """Restores agent state from a checkpoint, optionally reversing file
    changes via the stored git patch."""

    async def rollback(self, checkpoint_id: str, store: CheckpointStore) -> Snapshot:
        snapshot = await store.load(checkpoint_id)

        if snapshot.git_patch:
            await self._apply_reverse_patch(snapshot.git_patch)
            log.info("Reversed git patch from checkpoint %s", checkpoint_id)

        log.info(
            "Rolled back to checkpoint %s (trigger=%s, %d messages)",
            checkpoint_id,
            snapshot.trigger,
            len(snapshot.messages),
        )
        return snapshot

    async def _apply_reverse_patch(self, git_patch: str) -> None:
        """Apply the stored patch in reverse to undo file changes."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".patch",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(git_patch)
            patch_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "apply", "--reverse", "--allow-empty", patch_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                err = stderr.decode(errors="replace").strip()
                log.warning("Reverse patch failed (rc=%d): %s", proc.returncode, err)
                raise RuntimeError(f"git apply --reverse failed: {err}")
        finally:
            Path(patch_path).unlink(missing_ok=True)
