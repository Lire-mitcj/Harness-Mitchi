from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from src.harness.pipeline.definition import PipelineDefinition, Stage
from src.harness.pipeline.stage import StageResult

log = logging.getLogger(__name__)


@dataclass
class StageRecord:
    name: str
    result: StageResult | None = None
    attempts: int = 0
    elapsed_s: float = 0.0


@dataclass
class PipelineResult:
    pipeline_name: str
    success: bool
    stage_records: dict[str, StageRecord] = field(default_factory=dict)
    elapsed_s: float = 0.0

    @property
    def failed_stages(self) -> list[str]:
        return [
            name for name, rec in self.stage_records.items()
            if rec.result and rec.result.failed
        ]


class PipelineExecutor:
    """Executes a :class:`PipelineDefinition` respecting stage dependencies,
    parallelism, and retry policies."""

    async def execute(
        self,
        pipeline: PipelineDefinition,
        context: Any,
    ) -> PipelineResult:
        t0 = time.monotonic()
        order = self._topological_sort(pipeline.stages)
        outputs: dict[str, Any] = {}
        records: dict[str, StageRecord] = {}

        for batch in order:
            tasks = [
                self._run_stage(stage, context, outputs, pipeline)
                for stage in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for stage, result in zip(batch, results):
                if isinstance(result, BaseException):
                    rec = StageRecord(
                        name=stage.name,
                        result=StageResult(success=False, error=str(result)),
                    )
                    records[stage.name] = rec
                    await pipeline.hooks.fire_failure(stage.name, result if isinstance(result, Exception) else RuntimeError(str(result)))
                else:
                    record, output = result
                    records[stage.name] = record
                    outputs[stage.name] = output

        all_passed = all(
            r.result is not None and r.result.success
            for r in records.values()
        )
        return PipelineResult(
            pipeline_name=pipeline.name,
            success=all_passed,
            stage_records=records,
            elapsed_s=time.monotonic() - t0,
        )

    async def _run_stage(
        self,
        stage: Stage,
        context: Any,
        outputs: dict[str, Any],
        pipeline: PipelineDefinition,
    ) -> tuple[StageRecord, Any]:
        inputs = {dep: outputs.get(dep) for dep in stage.depends_on}
        record = StageRecord(name=stage.name)
        max_attempts = stage.retry_count + 1

        await pipeline.hooks.fire_before(stage.name, context)

        for attempt in range(1, max_attempts + 1):
            record.attempts = attempt
            t0 = time.monotonic()
            try:
                result = await stage.handler.run(context, inputs)
                record.elapsed_s = time.monotonic() - t0
                record.result = result
                await pipeline.hooks.fire_after(stage.name, result)

                if result.success or attempt == max_attempts:
                    return record, result.output
                log.info("Stage %s failed (attempt %d/%d), retrying", stage.name, attempt, max_attempts)
            except Exception as exc:
                record.elapsed_s = time.monotonic() - t0
                record.result = StageResult(success=False, error=str(exc))
                if attempt == max_attempts:
                    await pipeline.hooks.fire_failure(stage.name, exc)
                    return record, None
                log.info("Stage %s raised %s (attempt %d/%d), retrying", stage.name, exc, attempt, max_attempts)

        return record, None

    @staticmethod
    def _topological_sort(stages: list[Stage]) -> list[list[Stage]]:
        """Group stages into execution batches respecting dependencies.

        Stages within a batch have all their dependencies satisfied and can
        run in parallel.
        """
        by_name: dict[str, Stage] = {s.name: s for s in stages}
        in_degree: dict[str, int] = {s.name: 0 for s in stages}
        dependents: dict[str, list[str]] = defaultdict(list)

        for s in stages:
            for dep in s.depends_on:
                if dep in by_name:
                    in_degree[s.name] += 1
                    dependents[dep].append(s.name)

        batches: list[list[Stage]] = []
        ready = [name for name, deg in in_degree.items() if deg == 0]

        while ready:
            batch = [by_name[n] for n in ready]
            batches.append(batch)
            next_ready: list[str] = []
            for name in ready:
                for dep_name in dependents[name]:
                    in_degree[dep_name] -= 1
                    if in_degree[dep_name] == 0:
                        next_ready.append(dep_name)
            ready = next_ready

        scheduled = sum(len(b) for b in batches)
        if scheduled < len(stages):
            remaining = [s.name for s in stages if s.name not in {st.name for b in batches for st in b}]
            log.warning("Cycle detected in pipeline DAG, forcing remaining stages: %s", remaining)
            batches.append([by_name[n] for n in remaining])

        return batches
