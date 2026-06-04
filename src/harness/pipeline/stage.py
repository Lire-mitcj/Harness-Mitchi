from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class StageResult:
    success: bool
    output: Any = None
    error: str | None = None

    @property
    def failed(self) -> bool:
        return not self.success


class StageHandler(ABC):
    """Abstract base class for a single pipeline stage."""

    @abstractmethod
    async def run(self, context: Any, inputs: dict[str, Any]) -> StageResult:
        """Execute this stage.

        *context* is the shared pipeline context; *inputs* holds outputs
        from upstream stages this stage depends on.
        """
        ...
