from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from src.harness.pipeline.hooks import PipelineHooks, FailureCallback
from src.harness.pipeline.stage import StageHandler


@dataclass
class Stage:
    name: str
    handler: StageHandler
    depends_on: list[str] = field(default_factory=list)
    parallel: bool = False
    retry_count: int = 0


class PipelineDefinition:
    """Declarative pipeline built via a fluent API.

    Example::

        pipeline = (
            PipelineDefinition("score")
            .stage("lint", LintStage())
            .stage("test", TestStage(), parallel=True)
            .stage("judge", JudgeStage(), depends_on=["lint", "test"])
            .on_failure(my_failure_handler)
        )
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.stages: list[Stage] = []
        self.hooks = PipelineHooks()

    def stage(
        self,
        name: str,
        handler: StageHandler,
        *,
        depends_on: list[str] | None = None,
        parallel: bool = False,
        retry_count: int = 0,
    ) -> PipelineDefinition:
        self.stages.append(Stage(
            name=name,
            handler=handler,
            depends_on=depends_on or [],
            parallel=parallel,
            retry_count=retry_count,
        ))
        return self

    def on_failure(self, callback: FailureCallback) -> PipelineDefinition:
        self.hooks.on_failure.append(callback)
        return self

    def on_before_stage(
        self,
        callback: Callable[[str, Any], Awaitable[None]],
    ) -> PipelineDefinition:
        self.hooks.before_stage.append(callback)
        return self

    def on_after_stage(
        self,
        callback: Callable[[str, Any], Awaitable[None]],
    ) -> PipelineDefinition:
        self.hooks.after_stage.append(callback)
        return self

    def get_stage(self, name: str) -> Stage | None:
        for s in self.stages:
            if s.name == name:
                return s
        return None


Pipeline = PipelineDefinition
