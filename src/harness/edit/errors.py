from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorError(Exception):
    """Anchor validation failed before or during splice."""

    code: str
    message: str

    def __str__(self) -> str:
        return self.message
