from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.cursor_contracts import ValidationResult
from src.agent.cursor_evaluator import CursorEvaluator, compute_layer1_metrics
from src.agent.cursor_executor import CursorExecutor
from src.agent.cursor_patch_applier import CursorPatchApplier
from src.agent.cursor_validator import CursorValidator


class _Validator(CursorValidator):
    def __init__(self, project_root: Path, result: ValidationResult) -> None:
        super().__init__(project_root, command=("pytest",))
        self.result = result

    async def validate(self, **_: object) -> ValidationResult:
        return self.result


@pytest.mark.asyncio
async def test_executor_metrics_patch_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = (
        "<<<<<<< SEARCH\n"
        "missing = 1\n"
        "=======\n"
        "value = 2\n"
        ">>>>>>> REPLACE"
    )
    executor = CursorExecutor(tmp_path, CursorPatchApplier(tmp_path))

    execution, validation, metrics = await executor.execute_transaction(
        "sample.py",
        patch,
        _Validator(tmp_path, ValidationResult(success=True)),
        step=2,
    )

    assert not execution.success
    assert not validation.success
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert metrics.layer2.patch_correctness == 0.0
    assert metrics.layer2.execution_success == 0.0
    assert metrics.layer2.code_diff_correctness == 0.0
    assert not metrics.layer2.committed


@pytest.mark.asyncio
async def test_executor_metrics_validation_failure_rolls_back(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = (
        "<<<<<<< SEARCH\n"
        "value = 1\n"
        "=======\n"
        "value = 2\n"
        ">>>>>>> REPLACE"
    )
    executor = CursorExecutor(tmp_path, CursorPatchApplier(tmp_path))

    execution, validation, metrics = await executor.execute_transaction(
        "sample.py",
        patch,
        _Validator(tmp_path, ValidationResult(success=False, error="pytest failed")),
        layer1=compute_layer1_metrics(
            ("sample.py:value:1-1",),
            ("sample.py:value:1-1",),
        ),
    )

    assert execution.success
    assert not validation.success
    assert target.read_text(encoding="utf-8") == "value = 1\n"
    assert metrics.layer2.patch_correctness == 1.0
    assert metrics.layer2.execution_success == 0.0
    assert metrics.layer2.code_diff_correctness == 0.0
    assert not metrics.layer2.committed
    assert metrics.layer1.recall == 1.0


@pytest.mark.asyncio
async def test_executor_metrics_success_commits_and_evaluator_persists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sample.py"
    target.write_text("value = 1\n", encoding="utf-8")
    patch = (
        "<<<<<<< SEARCH\n"
        "value = 1\n"
        "=======\n"
        "value = 2\n"
        ">>>>>>> REPLACE"
    )
    layer1 = compute_layer1_metrics(
        ("sample.py:value:1-1", "other.py:helper:1-1"),
        ("sample.py:value:1-1",),
    )
    executor = CursorExecutor(tmp_path, CursorPatchApplier(tmp_path))

    execution, validation, metrics = await executor.execute_transaction(
        "sample.py",
        patch,
        _Validator(tmp_path, ValidationResult(success=True)),
        step=3,
        layer1=layer1,
    )
    CursorEvaluator(tmp_path).record(metrics)

    assert execution.success
    assert validation.success
    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert metrics.layer2.patch_correctness == 1.0
    assert metrics.layer2.execution_success == 1.0
    assert metrics.layer2.code_diff_correctness == 1.0
    assert metrics.layer2.committed
    assert metrics.layer1.precision == 0.5
    assert metrics.layer1.recall == 1.0

    rows = (tmp_path / "harness_evaluation_metrics.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    payload = json.loads(rows[-1])
    assert payload["step"] == 3
    assert payload["layer2"]["task_passed"] is True
    assert payload["layer2"]["validation"]["decision"] == "commit"
    assert {"ast", "semantic", "execution", "decision", "score"} <= set(
        payload["layer2"]["validation"]
    )
    assert payload["layer1"]["hits"] == ["sample.py:value:1-1"]


def test_layer1_metrics_without_oracle_records_retrieved_only() -> None:
    metrics = compute_layer1_metrics(("a.py:foo:1-2",))

    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.retrieved == ("a.py:foo:1-2",)
    assert metrics.expected == ()
