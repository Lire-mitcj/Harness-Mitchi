from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.contracts import ValidationResult
from src.agent.evaluator import (
    CursorEvalHarnessV2,
    CursorEvaluator,
    FullPipelineMetrics,
    RetrievalTestCase,
    RetrievalTestCaseLoader,
    RetrievalTrace,
    compute_layer1_metrics,
    format_bi_report,
    resolve_cursor_eval_path,
)
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.validator import CursorValidator


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

    assert metrics.available is False
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.retrieved == ("a.py:foo:1-2",)
    assert metrics.expected == ()


def test_evaluator_maps_windows_output_path_under_wsl() -> None:
    assert resolve_cursor_eval_path(Path(r"D:\eval_json")) == Path("/mnt/d/eval_json")
    assert resolve_cursor_eval_path(Path(r"D:\eval_json\cases.jsonl")) == Path(
        "/mnt/d/eval_json/cases.jsonl"
    )
    assert resolve_cursor_eval_path("D:eval_jsonretrieval_test_cases.jsonl") == Path(
        "/mnt/d/eval_json/cases/retrieval_test_cases.jsonl"
    )


def test_bi_report_marks_layer1_unavailable_without_ground_truth() -> None:
    metrics = FullPipelineMetrics(step=1, target_file="sample.py")

    report = format_bi_report(metrics)

    assert "Precision : N/A (ground truth unavailable)" in report
    assert "Recall    : N/A (ground truth unavailable)" in report


def test_v2_evaluator_uses_bound_gt_and_cumulative_fusion_trace() -> None:
    harness = CursorEvalHarnessV2(RetrievalTestCase(
        name="passenger-n-plus-one",
        ground_truth=("api/list.py", "db/schema.sql"),
    ))
    harness.add_trace(RetrievalTrace(
        step=1,
        retrieved=("api/list.py",),
        fused_files=("api/list.py", "noise.py"),
    ))
    harness.add_trace(RetrievalTrace(
        step=2,
        retrieved=("db/schema.sql",),
        fused_files=("db/schema.sql",),
    ))

    metrics = harness.evaluate()

    assert metrics.expected == ("api/list.py", "db/schema.sql")
    assert metrics.retrieved == ("api/list.py", "noise.py", "db/schema.sql")
    assert metrics.hits == ("api/list.py", "db/schema.sql")
    assert metrics.precision == 0.6667
    assert metrics.recall == 1.0
    assert metrics.f1_score == 0.8


def test_v2_evaluator_rejects_empty_ground_truth() -> None:
    with pytest.raises(ValueError, match="GT cannot be empty"):
        CursorEvalHarnessV2(RetrievalTestCase(name="invalid", ground_truth=()))


def test_v2_test_case_loader_requires_explicit_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "case.json"
    path.write_text(
        '{"name":"case","ground_truth":["api.py"],"description":"fixture"}',
        encoding="utf-8",
    )

    case = RetrievalTestCaseLoader.load_json(path)

    assert case.name == "case"
    assert case.ground_truth == ("api.py",)


def test_v2_test_case_loader_selects_named_case_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name":"first","ground_truth":["first.py"]}\n'
        '{"name":"second","ground_truth":["second.py","schema.sql"]}\n',
        encoding="utf-8",
    )

    case = RetrievalTestCaseLoader.load_case(path, "second")

    assert case.name == "second"
    assert case.ground_truth == ("second.py", "schema.sql")
    with pytest.raises(ValueError, match="set case_name"):
        RetrievalTestCaseLoader.load_case(path)


def test_v2_test_case_loader_auto_selects_unique_name_match(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"name":"passenger-list-sensitive-fields","ground_truth":["list.py"]}\n'
        '{"name":"order-view-migration","ground_truth":["main.py"]}\n',
        encoding="utf-8",
    )

    case = RetrievalTestCaseLoader.select_case(
        path,
        ("passenger", "list", "sensitive", "fields"),
    )

    assert case.name == "passenger-list-sensitive-fields"
