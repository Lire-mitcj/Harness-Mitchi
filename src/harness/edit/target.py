from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditTarget:
    """Harness-resolved edit anchor: physical span + content hash."""

    path: str
    symbol: str
    kind: str
    start_line: int
    end_line: int
    original_source: str
    anchor_hash: str
    callee_signatures: tuple[str, ...] = ()


@dataclass(frozen=True)
class EditHandoff:
    """Structured diagnose → edit handoff for splice mode."""

    targets: tuple[EditTarget, ...]
    mode: str = "splice"

    @property
    def active(self) -> bool:
        return bool(self.targets)
