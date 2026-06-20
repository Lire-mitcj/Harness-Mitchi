from __future__ import annotations

import asyncio
from pathlib import Path

from src.agent.cursor_contracts import ExecutionResult, ValidationDecision, ValidationResult
from src.agent.cursor_evaluator import FullPipelineMetrics, Layer1Metrics, Layer2Metrics
from src.agent.cursor_harness_context import CursorHarnessContext
from src.agent.cursor_patch_applier import CursorPatchApplier
from src.agent.cursor_validator import CursorValidator


class CursorExecutor:
    def __init__(self, project_root: Path, patch_applier: CursorPatchApplier) -> None:
        self.project_root = project_root
        self.patch_applier = patch_applier

    def apply_patch(self, file_path: str, patch: str) -> ExecutionResult:
        success, error = self.patch_applier.apply_patch(file_path, patch)
        return ExecutionResult(success=success, file=file_path, error=error)

    async def execute_transaction(
        self,
        target_file: str,
        patch: str,
        validator: CursorValidator,
        *,
        step: int = 1,
        layer1: Layer1Metrics | None = None,
        user_intent: str = "",
    ) -> tuple[ExecutionResult, ValidationResult, FullPipelineMetrics]:
        layer1 = layer1 or Layer1Metrics()
        async with CursorHarnessContext(self.project_root, target_file) as harness:
            target_path = (self.project_root / target_file).resolve()
            try:
                original_content = target_path.read_text(encoding="utf-8")
            except OSError:
                original_content = ""

            # 1. Apply patch in a separate thread
            success, error = await asyncio.to_thread(
                self.patch_applier.apply_patch, target_file, patch
            )
            if not success:
                execution = ExecutionResult(
                    success=False,
                    file=target_file,
                    error=error,
                    original_content=original_content,
                    rolled_back=True,
                )
                validation = ValidationResult(
                    success=False,
                    error="validation skipped due to patch failure",
                )
                return (
                    execution,
                    validation,
                    _metrics(
                        step=step,
                        target_file=target_file,
                        layer1=layer1,
                        patch_correctness=0.0,
                        execution_success=0.0,
                        code_diff_correctness=0.0,
                        task_passed=False,
                        committed=harness.committed,
                        observation=error,
                        validation=_validation_payload(validation),
                    ),
                )

            try:
                patched_content = target_path.read_text(encoding="utf-8")
            except OSError:
                patched_content = ""

            # 2. Run layered validator checks
            val_result = await validator.validate(
                target_file=target_file,
                patch=patch,
                original_content=original_content,
                patched_content=patched_content,
                user_intent=user_intent,
            )
            validation_decision = _effective_decision(val_result)
            if validation_decision != "commit":
                # Truncate stacktrace output to protect metrics board space limit
                truncated_error = self._truncate_error(val_result.error)
                execution = ExecutionResult(
                    success=True,
                    file=target_file,
                    original_content=original_content,
                    attempted_content=patched_content,
                    rolled_back=True,
                )
                validation = ValidationResult(
                    success=False,
                    error=truncated_error,
                    ast=val_result.ast,
                    semantic=val_result.semantic,
                    execution=val_result.execution,
                    decision=validation_decision,
                    score=val_result.score,
                )
                return (
                    execution,
                    validation,
                    _metrics(
                        step=step,
                        target_file=target_file,
                        layer1=layer1,
                        patch_correctness=1.0,
                        execution_success=_execution_score(val_result),
                        code_diff_correctness=_ast_score(val_result),
                        task_passed=False,
                        committed=harness.committed,
                        observation=truncated_error,
                        validation=_validation_payload(validation),
                    ),
                )

            # 3. Layered validation committed: persist physical disk changes
            harness.commit()
            execution = ExecutionResult(
                success=True,
                file=target_file,
                original_content=original_content,
                attempted_content=patched_content,
                rolled_back=False,
            )
            return (
                execution,
                val_result,
                _metrics(
                    step=step,
                    target_file=target_file,
                    layer1=layer1,
                    patch_correctness=1.0,
                    execution_success=_execution_score(val_result),
                    code_diff_correctness=_ast_score(val_result),
                    task_passed=True,
                    committed=harness.committed,
                    observation="validation decision: commit",
                    validation=_validation_payload(val_result),
                ),
            )

    @staticmethod
    def _truncate_error(error: str) -> str:
        if len(error) <= 800:
            return error
        return error[:300] + "\n... [TRUNCATED STACKTRACE] ...\n" + error[-500:]


def _metrics(
    *,
    step: int,
    target_file: str,
    layer1: Layer1Metrics,
    patch_correctness: float,
    execution_success: float,
    code_diff_correctness: float,
    task_passed: bool,
    committed: bool,
    observation: str,
    validation: dict[str, object] | None = None,
) -> FullPipelineMetrics:
    return FullPipelineMetrics(
        step=step,
        target_file=target_file,
        layer1=layer1,
        layer2=Layer2Metrics(
            patch_correctness=patch_correctness,
            execution_success=execution_success,
            code_diff_correctness=code_diff_correctness,
            task_passed=task_passed,
            observation=observation,
            committed=committed,
            validation=validation or {},
        ),
    )


def _effective_decision(result: ValidationResult) -> ValidationDecision:
    has_layered_report = any(
        part is not None for part in (result.ast, result.semantic, result.execution)
    )
    if has_layered_report:
        return result.decision
    return "commit" if result.success else "retry"


def _execution_score(result: ValidationResult) -> float:
    if result.execution is None:
        return 1.0 if result.success else 0.0
    return 1.0 if bool(result.execution.get("pass")) else 0.0


def _ast_score(result: ValidationResult) -> float:
    if result.ast is None:
        return 1.0 if result.success else 0.0
    return 1.0 if bool(result.ast.get("pass")) else 0.0


def _validation_payload(result: ValidationResult) -> dict[str, object]:
    score = result.score
    if not any(part is not None for part in (result.ast, result.semantic, result.execution)):
        score = 1.0 if result.success else 0.0
    return {
        "ast": result.ast or {"pass": result.success, "issues": []},
        "semantic": result.semantic or {"score": 1.0 if result.success else 0.0},
        "execution": result.execution or {"pass": result.success, "error": result.error},
        "decision": _effective_decision(result),
        "score": score,
    }
