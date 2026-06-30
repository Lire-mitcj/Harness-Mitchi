from __future__ import annotations

import difflib
from dataclasses import replace
from typing import Literal

from src.agent.contracts import ExecutionResult, ValidationResult
from src.agent.state import CursorState, ExecutionTrace, PatchMemory


class CursorStateManager:
    """Owns the bounded but cumulative Cursor recovery state."""

    def __init__(self, *, max_bytes: int = 2048) -> None:
        self.max_bytes = max_bytes

    def initial(self, task: str, max_steps: int = 10) -> CursorState:
        return self._bounded(CursorState(
            task=task,
            current_file="",
            last_patch="",
            last_observation="",
            status="running",
            current_step=1,
            max_steps=max_steps,
            stage_completion=0.0,
            entropy_score=0.0,
        ))


    def after_execution(
        self,
        state: CursorState,
        target_file: str,
        patch: str,
        result: ExecutionResult,
    ) -> CursorState:
        observation = "patch applied" if result.success else result.error
        next_step = state.current_step + (0 if result.success else 1)
        status: Literal["running", "success", "failed"] = "running"
        if not result.success and next_step > state.max_steps:
            status = "failed"
        memories = state.patch_memory
        traces = state.execution_traces
        if not result.success:
            memories = (*memories, PatchMemory(
                step=state.current_step,
                target_file=target_file,
                patch_content=patch,
                rollback_reason=result.error or "patch application failed",
                diff_status="patch_application_rejected",
                base_snapshot=result.original_content,
                attempted_snapshot=result.attempted_content,
            ))
            traces = (*traces, ExecutionTrace(
                step=state.current_step,
                target_file=target_file,
                exception_type="PatchApplicationError",
                error_msg=result.error,
            ))
        return self._bounded(replace(
            state,
            current_file=target_file,
            last_patch=patch,
            last_observation=observation,
            current_step=next_step,
            status=status,
            execution_traces=traces,
            patch_memory=memories,
        ))

    def after_validation(
        self,
        state: CursorState,
        result: ValidationResult,
        suggested_completion: float = 0.0,
        *,
        patch: str = "",
        execution: ExecutionResult | None = None,
    ) -> CursorState:
        next_step = state.current_step + 1
        is_success = result.success
        status: Literal["running", "success", "failed"] = "running"

        if is_success:
            completion = suggested_completion if suggested_completion > 0.0 else 1.0
            if completion >= 1.0:
                status = "success"
                completion = 1.0
            else:
                if next_step > state.max_steps:
                    status = "failed"
        else:
            completion = state.stage_completion
            if next_step > state.max_steps:
                status = "failed"

        traces = state.execution_traces
        memories = state.patch_memory
        if not is_success:
            trace = _trace_from_validation(result, state.current_step, state.current_file)
            if trace is not None:
                traces = (*traces, trace)
            if patch:
                memories = (*memories, PatchMemory(
                    step=state.current_step,
                    target_file=state.current_file,
                    patch_content=patch,
                    rollback_reason=result.error or "validation rejected patch",
                    diff_status="partial_success_but_validation_rejected",
                    base_snapshot=execution.original_content if execution else "",
                    attempted_snapshot=execution.attempted_content if execution else "",
                ))

        return self._bounded(replace(
            state,
            last_observation="validation passed" if is_success else result.error,
            current_step=next_step,
            stage_completion=completion,
            status=status,
            execution_traces=traces,
            patch_memory=memories,
        ))

    def failed(self, state: CursorState, observation: str | None = None) -> CursorState:
        return self._bounded(replace(
            state,
            last_observation=observation or state.last_observation,
            status="failed",
        ))

    def observe(self, state: CursorState, observation: str) -> CursorState:
        return self._bounded(replace(
            state,
            last_observation=observation,
            status="running",
        ))

    def record_decision(
        self,
        state: CursorState,
        *,
        action: str,
        target_file: str = "",
        can_answer: bool,
    ) -> CursorState:
        """Record loop-only action costs; never rewrite the model's decision."""
        error_type = _error_type(state.last_observation)
        signature = f"{action}|{target_file or '-'}|{error_type}"
        repeated = signature in state.decision_signatures
        retry_bias = state.retry_bias + int(action == "ask_clarify" and can_answer) + int(repeated)
        cost = {"edit": 0.1, "answer": 0.2, "ask_clarify": 0.8}.get(action, 0.0)
        observation = state.last_observation
        if action == "ask_clarify" and can_answer:
            observation = (
                f"Decision selected ask_clarify despite grounded context; "
                f"retry_bias={retry_bias}, repeated_signature={repeated}."
            )
        return self._bounded(replace(
            state,
            decision_signatures=(*state.decision_signatures, signature),
            retry_bias=retry_bias,
            decision_cost_total=round(state.decision_cost_total + cost, 4),
            last_observation=observation,
            status="running",
        ))

    def mark_retry(self, state: CursorState, clarification: str = "") -> CursorState:
        """Advance a loop-only retry without converting clarify into edit."""
        next_step = state.current_step + 1
        observation = state.last_observation
        if clarification:
            observation = (
                f"Clarification requested: '{clarification}'. "
                f"However, since grounded context is already available, "
                f"please use the 'research' action to search for missing symbols, "
                f"or perform 'edit' on the current context."
            )
        return self._bounded(replace(
            state,
            last_observation=observation,
            current_step=next_step,
            status="failed" if next_step > state.max_steps else "running",
        ))

    def observe_failure_signature(
        self,
        state: CursorState,
        action: str,
        file: str,
        error_type: str,
    ) -> CursorState:
        increase = 1.0
        if error_type == "missing_info_signal":
            increase = 0.8
        elif "syntax" in str(error_type).lower():
            increase = 1.5

        from src.agent.state import ExecutionTrace
        traces = state.execution_traces + (ExecutionTrace(
            step=state.current_step,
            target_file=file,
            exception_type=error_type,
            error_msg=f"Failure observed: {action} on {file} with type {error_type}",
        ),)

        observation = f"Failure signature observed - action: {action}, file: {file}, error: {error_type}."
        return self._bounded(replace(
            state,
            entropy_score=round(state.entropy_score + increase, 4),
            last_observation=observation,
            execution_traces=traces,
        ))

    def apply_time_decay(self, state: CursorState) -> CursorState:
        decayed_entropy = round(state.entropy_score * 0.9, 4)
        return self._bounded(replace(
            state,
            entropy_score=decayed_entropy,
        ))


    def succeeded(self, state: CursorState, observation: str = "answered") -> CursorState:
        return self._bounded(replace(
            state,
            last_observation=observation,
            status="success",
        ))

    def format_for_prompt(self, state: CursorState) -> str:
        # 1. Truncate observation to prevent log poisoning (limit to ~400 chars)
        raw_obs = state.last_observation or 'None'
        if len(raw_obs) > 500:
            obs_trimmed = (
                raw_obs[:150]
                + "\n... [TRUNCATED SYSTEM LOGS] ...\n"
                + raw_obs[-250:]
            )
        else:
            obs_trimmed = raw_obs

        # 2. Keep a concise last-patch marker for the Kanban header.
        patch_summary = "None"
        if state.last_patch:
            patch_lines = state.last_patch.splitlines()
            anchor_line = next(
                (
                    line.strip()
                    for line in patch_lines
                    if "def " in line or "class " in line or "@app." in line
                ),
                "In-line diff adjustment",
            )
            patch_summary = f"[Modified around -> {anchor_line}]"

        trace_text = self._clip(
            _format_execution_traces(state.execution_traces), max(300, self.max_bytes // 5),
        )
        memory_text = self._clip(
            _format_patch_memory(state.patch_memory), max(300, self.max_bytes // 5),
        )
        evolution_text = self._clip(
            _format_diff_evolution(state.patch_memory), max(700, self.max_bytes // 2),
        )

        # 3. Compile the persistent recovery timeline into the next decision prompt.
        return (
            "### [KANBAN LOOP GLOBAL STATE] ###\n"
            f"Requirement  : {state.task}\n"
            f"Active File  : {state.current_file or 'None'}\n"
            f"Loop Progress: Step {state.current_step} / {state.max_steps}\n"
            f"Completion   : {int(state.stage_completion * 100)}%\n"
            f"Decision Stats: retry_bias={state.retry_bias}, "
            f"soft_cost={state.decision_cost_total:.1f}, "
            f"entropy={state.entropy_score:.2f}\n"

            f"Prior Patch  : {patch_summary}\n"
            "--- LAST RUNTIME OBSERVATION ---\n"
            f"{obs_trimmed.strip()}\n"
            "--- EXECUTION TRACE LAYER ---\n"
            f"{trace_text}\n"
            "--- PATCH MEMORY LAYER ---\n"
            f"{memory_text}\n"
            "--- STATE DIFF EVOLUTION LAYER ---\n"
            f"{evolution_text}\n"
            "#################################"
        )

    def serialized_size(self, state: CursorState) -> int:
        return len(self.format_for_prompt(state).encode("utf-8"))

    def _bounded(self, state: CursorState) -> CursorState:
        bounded = replace(
            state,
            task=self._clip(state.task, 512),
            current_file=self._clip(state.current_file, 256),
            last_patch=self._clip(state.last_patch, 700),
            last_observation=self._clip(state.last_observation, 700),
            execution_traces=tuple(
                replace(trace, error_msg=self._clip(trace.error_msg, 500))
                for trace in state.execution_traces
            ),
            patch_memory=tuple(
                replace(
                    memory,
                    patch_content=self._clip(memory.patch_content, 1600),
                    rollback_reason=self._clip(memory.rollback_reason, 600),
                    base_snapshot=self._clip(memory.base_snapshot, 2400),
                    attempted_snapshot=self._clip(memory.attempted_snapshot, 2400),
                )
                for memory in state.patch_memory
            ),
            decision_signatures=tuple(
                self._clip(signature, 300) for signature in state.decision_signatures
            ),
        )
        while self.serialized_size(bounded) > self.max_bytes:
            patch_limit = max(64, len(bounded.last_patch) - 128)
            observation_limit = max(64, len(bounded.last_observation) - 128)
            task_limit = max(64, len(bounded.task) - 64)
            file_limit = max(32, len(bounded.current_file) - 32)
            previous = bounded
            bounded = replace(
                bounded,
                task=self._clip(bounded.task, task_limit),
                current_file=self._clip(bounded.current_file, file_limit),
                last_patch=self._clip(bounded.last_patch, patch_limit),
                last_observation=self._clip(
                    bounded.last_observation,
                    observation_limit,
                ),
            )
            if bounded == previous:
                break
        return bounded

    @staticmethod
    def _clip(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        suffix = "...[truncated]"
        return value[: max(0, limit - len(suffix))] + suffix


def _trace_from_validation(
    result: ValidationResult,
    step: int,
    target_file: str,
) -> ExecutionTrace | None:
    ast_result = result.ast or {}
    raw = ast_result.get("trace") if isinstance(ast_result, dict) else None
    if isinstance(raw, dict):
        return ExecutionTrace(
            step=step,
            target_file=target_file,
            line_number=_as_int(raw.get("line_number")),
            offset=_as_int(raw.get("offset")),
            exception_type=str(raw.get("exception_type") or "ValidationError"),
            error_msg=str(raw.get("error_msg") or result.error),
        )
    if result.error:
        return ExecutionTrace(
            step=step,
            target_file=target_file,
            exception_type="ValidationError",
            error_msg=result.error,
        )
    return None


def _as_int(value: object) -> int | None:
    return int(value) if isinstance(value, int) else None


def _error_type(observation: str) -> str:
    if not observation:
        return "none"
    return observation.split(":", 1)[0].strip().replace(" ", "_")[:120] or "unknown"


def _format_execution_traces(traces: tuple[ExecutionTrace, ...]) -> str:
    if not traces:
        return "None"
    return "\n".join(
        f"Step {trace.step} | {trace.target_file} | {trace.exception_type} "
        f"line={trace.line_number if trace.line_number is not None else '-'} "
        f"offset={trace.offset if trace.offset is not None else '-'} | {trace.error_msg}"
        for trace in traces
    )


def _format_patch_memory(memories: tuple[PatchMemory, ...]) -> str:
    if not memories:
        return "None"
    return "\n".join(
        f"Step {memory.step} | {memory.target_file} | {memory.diff_status} | "
        f"rollback={memory.rollback_reason}"
        for memory in memories
    )


def _format_diff_evolution(memories: tuple[PatchMemory, ...]) -> str:
    if not memories:
        return "None"
    sections: list[str] = []
    for memory in memories:
        if not memory.base_snapshot and not memory.attempted_snapshot:
            continue
        diff = "\n".join(difflib.unified_diff(
            memory.base_snapshot.splitlines(),
            memory.attempted_snapshot.splitlines(),
            fromfile=f"{memory.target_file}@base",
            tofile=f"{memory.target_file}@attempt-step-{memory.step}",
            lineterm="",
        ))
        sections.append(
            f"[ATTEMPT STEP {memory.step} | {memory.diff_status}]\n"
            f"{diff or '(no textual diff captured)'}"
        )
    return "\n\n".join(sections) or "None"
