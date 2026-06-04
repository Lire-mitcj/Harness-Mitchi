from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.harness.checkpoint.snapshot import Snapshot

log = logging.getLogger(__name__)

_AUTO_SAVE_INTERVAL_S = 120.0


class CheckpointStore:
    """Persists and retrieves :class:`Snapshot` objects as JSON files."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self._dir = checkpoint_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._last_auto_save: float = 0.0

    async def save(self, trigger: str, state: Any) -> str:
        """Serialise current agent state to a new checkpoint file.

        Returns the checkpoint id.
        """
        checkpoint_id = uuid4().hex[:12]
        messages = _extract_messages(state)
        file_changes = _extract_file_changes(state)
        git_patch = await self._create_git_patch()

        snapshot = Snapshot(
            id=checkpoint_id,
            trigger=trigger,
            timestamp=time.time(),
            messages=messages,
            file_changes=file_changes,
            git_patch=git_patch,
            memory_snapshot=_extract_memory(state),
            plan_state=_extract_plan(state),
        )

        path = self._dir / f"{checkpoint_id}.json"
        data = json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2)
        await asyncio.to_thread(path.write_text, data, encoding="utf-8")
        log.info("Checkpoint saved: %s (%s)", checkpoint_id, trigger)
        self._last_auto_save = time.time()
        return checkpoint_id

    async def load(self, checkpoint_id: str) -> Snapshot:
        path = self._dir / f"{checkpoint_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint_id} not found")
        raw = await asyncio.to_thread(path.read_text, "utf-8")
        return Snapshot.from_dict(json.loads(raw))

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for p in sorted(self._dir.glob("*.json"), key=lambda f: f.stat().st_mtime):
            try:
                raw = await asyncio.to_thread(p.read_text, "utf-8")
                data = json.loads(raw)
                entries.append({
                    "id": data["id"],
                    "trigger": data["trigger"],
                    "timestamp": data["timestamp"],
                    "file_changes": len(data.get("file_changes", [])),
                    "has_patch": data.get("git_patch") is not None,
                })
            except (json.JSONDecodeError, KeyError):
                log.warning("Skipping corrupt checkpoint file: %s", p.name)
        return entries

    async def delete(self, checkpoint_id: str) -> None:
        path = self._dir / f"{checkpoint_id}.json"
        if path.exists():
            await asyncio.to_thread(path.unlink)
            log.info("Deleted checkpoint %s", checkpoint_id)

    async def auto_save_if_needed(self, state: Any, trigger: str) -> str | None:
        elapsed = time.time() - self._last_auto_save
        if elapsed < _AUTO_SAVE_INTERVAL_S:
            return None
        return await self.save(trigger, state)

    async def _create_git_patch(self) -> str | None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git", "diff", "HEAD",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
            patch = stdout.decode(errors="replace").strip()
            return patch if patch else None
        except Exception:
            return None


# ------------------------------------------------------------------
# Helpers to extract state fields regardless of state object type
# ------------------------------------------------------------------

def _extract_messages(state: Any) -> list[dict[str, Any]]:
    if hasattr(state, "messages"):
        msgs = state.messages
        if msgs and hasattr(msgs[0], "to_dict"):
            return [m.to_dict() for m in msgs]
        return list(msgs)
    return []


def _extract_file_changes(state: Any) -> list[str]:
    return list(getattr(state, "file_changes", []))


def _extract_memory(state: Any) -> dict[str, Any] | None:
    mem = getattr(state, "memory_snapshot", None)
    if mem is not None:
        return dict(mem)
    return None


def _extract_plan(state: Any) -> dict[str, Any] | None:
    plan = getattr(state, "current_plan", None)
    if plan is None:
        return None
    if isinstance(plan, dict):
        return plan
    return {"plan_text": str(plan)}
