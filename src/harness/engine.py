from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm

from src.agent.types import Message

if TYPE_CHECKING:
    from src.config.settings import MitKIISettings

from src.harness.checkpoint.rollback import RollbackManager
from src.harness.checkpoint.store import CheckpointStore
from src.harness.cursor.retrieval_guardrail import CursorRetrievalGuardrail
from src.harness.gates.phase_metrics import PhaseMetrics
from src.harness.pipeline.definition import PipelineDefinition
from src.harness.pipeline.executor import PipelineExecutor
from src.harness.probe.interceptor import ContextProbe
from src.harness.sandbox.executor import SandboxExecutor
from src.harness.sandbox.file_guard import FileGuard
from src.harness.sandbox.resource_limit import ResourceLimiter
from src.harness.scorer.engine import ScoringEngine
from src.harness.subtask.context_pipeline import (
    ContextPipelineResult,
    ExecutorContextConfig,
    ExecutorContextSession,
    ExecutorRuntimeState,
)
from src.harness.subtask.handoff import SubtaskHandoffBundle, prepare_executor_handoff
from src.harness.subtask.session_memory import ExploreSessionMemory

log = logging.getLogger(__name__)


class HarnessEngine:
    """Central orchestrator that wires together every harness subsystem.

    The agent loop holds a single ``HarnessEngine`` instance and delegates
    context management, checkpointing, scoring, pipeline execution, and
    sandboxed command runs through it.
    """

    def __init__(self, settings: MitKIISettings, project_root: Path | None = None) -> None:
        self.settings = settings
        self.project_root = (project_root or Path.cwd()).resolve()

        # --- Probe (context management) ---
        self.probe = ContextProbe(
            max_tokens=settings.max_context_tokens,
            budget_ratio=settings.context_budget_ratio,
        )
        self.phase_metrics = PhaseMetrics()
        self.cursor_retrieval_guardrail = CursorRetrievalGuardrail(
            query_cap=settings.cursor_retrieval_max_queries,
            fan_out=settings.cursor_retrieval_fan_out,
            timeout=settings.cursor_retrieval_timeout,
            early_stop_candidates=settings.cursor_retrieval_early_stop_candidates,
        )

        # --- Checkpoint ---
        checkpoint_dir = settings.data_dir / "checkpoints"
        self.checkpoint_store = CheckpointStore(checkpoint_dir)
        self.rollback = RollbackManager()

        # --- Scorer ---
        self.scorer = ScoringEngine(
            llm_client=self._judge_llm_call if settings.judge_model else None,
            project_root=self.project_root,
        )

        # --- Pipeline ---
        self.pipeline_executor = PipelineExecutor()

        # --- Sandbox ---
        self.file_guard = FileGuard(
            project_root=str(self.project_root),
        )
        self.resource_limiter = ResourceLimiter()
        self.sandbox = SandboxExecutor(
            file_guard=self.file_guard,
            limiter=self.resource_limiter,
        )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    async def before_llm_call(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return await self.probe.before_call(messages)

    async def before_executor_llm_call(
        self,
        session: ExecutorContextSession,
        messages: list[Message],
        error_trace: list[str],
    ) -> tuple[list[dict[str, Any]], ContextPipelineResult]:
        """Unified context pipeline: digest compact/fold, then probe trim."""
        cp = session.prepare_before_llm(messages, error_trace)
        msg_dicts = [m.to_dict() for m in cp.messages]
        if cp.token_est is not None and cp.token_est <= self.probe.budget:
            return msg_dicts, cp
        trimmed = await self.probe.before_call(msg_dicts)
        return trimmed, cp

    async def after_executor_tool_round(
        self,
        session: ExecutorContextSession,
        messages: list[Message],
        error_trace: list[str],
        *,
        explore_used: bool,
        explore_ok: bool,
    ) -> ContextPipelineResult:
        import asyncio
        import functools

        return await asyncio.to_thread(
            functools.partial(
                session.after_tool_round,
                messages,
                error_trace,
                explore_used=explore_used,
                explore_ok=explore_ok,
            )
        )

    async def after_llm_call(self, response: Any, usage: Any) -> None:
        await self.probe.after_call(response, usage)

    def create_executor_context_session(
        self,
        config: ExecutorContextConfig,
        memory: ExploreSessionMemory,
        runtime: ExecutorRuntimeState,
    ) -> ExecutorContextSession:
        return ExecutorContextSession(
            config=config,
            memory=memory,
            runtime=runtime,
        )

    def prepare_subtask_handoff(self, **kwargs: Any) -> SubtaskHandoffBundle:
        """Build Executor input (prompt + runtime policy)."""
        return prepare_executor_handoff(**kwargs)

    async def save_checkpoint(self, trigger: str, state: Any) -> str | None:
        return await self.checkpoint_store.auto_save_if_needed(state, trigger)

    async def run_pipeline(
        self,
        pipeline: PipelineDefinition,
        context: Any,
    ) -> Any:
        return await self.pipeline_executor.execute(pipeline, context)

    async def _judge_llm_call(self, *, messages: list[dict[str, Any]]) -> str:
        """LLM callback used by L1 rubric judge."""
        if not self.settings.judge_model:
            return ""
        try:
            resp = await litellm.acompletion(
                model=self.settings.judge_model,
                messages=messages,
                temperature=self.settings.judge_temperature,
                max_tokens=self.settings.judge_max_tokens,
                stream=False,
            )
            choice = resp.choices[0] if resp.choices else None
            content = choice.message.content if choice and choice.message else ""
            return content or ""
        except Exception as exc:
            log.warning("Judge LLM call failed: %s", exc)
            # Fail-safe: return invalid JSON so judge marks this as fail.
            return (
                '{"verdict":"fail","results":[],"blockers":'
                '["Judge LLM call failed"],"warnings":[]}'
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, settings: MitKIISettings, project_root: Path | None = None) -> HarnessEngine:
        """Build a fully-initialised ``HarnessEngine`` from application settings."""
        settings.ensure_dirs()
        return cls(settings, project_root=project_root)
