from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseRecord:
    phase: str
    duration_ms: float
    subtask_id: str | None = None
    verdict: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PhaseMetrics:
    """Wall-clock timing for orchestrator pipeline stages (no LLM cost)."""

    def __init__(self) -> None:
        self._records: list[PhaseRecord] = []
        self._active: dict[str, tuple[str, float, str | None]] = {}
        self._turn_start: float = time.monotonic()

    def start(self, phase: str, *, subtask_id: str | None = None) -> None:
        key = f"{phase}:{subtask_id or ''}"
        self._active[key] = (phase, time.monotonic(), subtask_id)

    def end(
        self,
        phase: str,
        *,
        subtask_id: str | None = None,
        verdict: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PhaseRecord:
        key = f"{phase}:{subtask_id or ''}"
        entry = self._active.pop(key, None)
        if entry is None:
            rec = PhaseRecord(
                phase=phase,
                duration_ms=0.0,
                subtask_id=subtask_id,
                verdict=verdict,
                metadata=dict(metadata or {}),
            )
        else:
            _, started, sid = entry
            rec = PhaseRecord(
                phase=phase,
                duration_ms=round((time.monotonic() - started) * 1000, 1),
                subtask_id=sid or subtask_id,
                verdict=verdict,
                metadata=dict(metadata or {}),
            )
        self._records.append(rec)
        return rec

    def reset_turn(self) -> None:
        """Clear records at the start of each user turn."""
        self._records.clear()
        self._active.clear()
        self._turn_start = time.monotonic()

    def get_summary(self) -> dict[str, Any]:
        phases = [
            {
                "phase": r.phase,
                "duration_ms": r.duration_ms,
                "subtask_id": r.subtask_id,
                "verdict": r.verdict,
                **r.metadata,
            }
            for r in self._records
        ]
        total_ms = round(sum(r.duration_ms for r in self._records), 1)
        return {
            "phases": phases,
            "phase_total_ms": total_ms,
            "turn_wall_ms": round((time.monotonic() - self._turn_start) * 1000, 1),
        }

    @property
    def records(self) -> list[PhaseRecord]:
        return list(self._records)
