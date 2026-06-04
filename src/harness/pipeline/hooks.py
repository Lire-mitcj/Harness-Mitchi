from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

StageCallback = Callable[[str, Any], Awaitable[None]]
FailureCallback = Callable[[str, Exception], Awaitable[None]]


@dataclass
class PipelineHooks:
    """Lifecycle callbacks invoked by the pipeline executor."""

    before_stage: list[StageCallback] = field(default_factory=list)
    after_stage: list[StageCallback] = field(default_factory=list)
    on_failure: list[FailureCallback] = field(default_factory=list)

    async def fire_before(self, stage_name: str, context: Any) -> None:
        for cb in self.before_stage:
            await cb(stage_name, context)

    async def fire_after(self, stage_name: str, result: Any) -> None:
        for cb in self.after_stage:
            await cb(stage_name, result)

    async def fire_failure(self, stage_name: str, exc: Exception) -> None:
        for cb in self.on_failure:
            await cb(stage_name, exc)
