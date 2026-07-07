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


class _CountingValidator(_Validator):
    def __init__(self, project_root: Path, result: ValidationResult) -> None:
        super().__init__(project_root, result)
        self.call_count = 0

    async def validate(self, **kwargs: object) -> ValidationResult:
        self.call_count += 1
        return self.result


@pytest.mark.asyncio
async def test_executor_batched_transaction_validates_once(tmp_path: Path) -> None:
    target = tmp_path / "sample.py"
    target.write_text("a = 1\nb = 1\n", encoding="utf-8")
    patch_a = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "=======\n"
        "a = 2\n"
        ">>>>>>> REPLACE"
    )
    patch_b = (
        "<<<<<<< SEARCH\n"
        "b = 1\n"
        "=======\n"
        "b = 2\n"
        ">>>>>>> REPLACE"
    )
    validator = _CountingValidator(tmp_path, ValidationResult(success=True))
    executor = CursorExecutor(tmp_path, CursorPatchApplier(tmp_path))

    execution, validation, metrics = await executor.execute_batched_transaction(
        "sample.py",
        [patch_a, patch_b],
        validator,
    )

    assert execution.success
    assert validation.success
    assert target.read_text(encoding="utf-8") == "a = 2\nb = 2\n"
    assert validator.call_count == 1
    assert metrics.layer2.committed


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


def test_patch_applier_creates_new_file_and_applies_empty_search(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path)
    new_file = "subdir/new_file.py"
    
    # 1. Test creating a brand-new file using empty SEARCH block
    patch = (
        "<<<<<<< SEARCH\n"
        "=======\n"
        "print('hello world')\n"
        ">>>>>>> REPLACE"
    )
    success, error = applier.apply_patch(new_file, patch)
    assert success is True
    assert (tmp_path / new_file).exists()
    assert (tmp_path / new_file).read_text(encoding="utf-8") == "print('hello world')\n"

    # 2. Test editing an empty file using empty SEARCH block
    empty_file = "empty.py"
    (tmp_path / empty_file).write_text("", encoding="utf-8")
    patch_empty = (
        "<<<<<<< SEARCH\n"
        "=======\n"
        "print('populated')\n"
        ">>>>>>> REPLACE"
    )
    success, error = applier.apply_patch(empty_file, patch_empty)
    assert success is True
    assert (tmp_path / empty_file).read_text(encoding="utf-8") == "print('populated')\n"


def test_patch_applier_applies_multiple_non_overlapping_blocks(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path)
    target = "routes.py"
    (tmp_path / target).write_text(
        "@router.get('/a')\ndef a():\n    return 1\n\n"
        "@router.get('/b')\ndef b():\n    return 2\n",
        encoding="utf-8",
    )
    patch = (
        "<<<<<<< SEARCH\n"
        "@router.get('/a')\n"
        "def a():\n"
        "=======\n"
        "@wrap\n"
        "@router.get('/a')\n"
        "def a():\n"
        ">>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n"
        "@router.get('/b')\n"
        "def b():\n"
        "=======\n"
        "@wrap\n"
        "@router.get('/b')\n"
        "def b():\n"
        ">>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is True, error
    updated = (tmp_path / target).read_text(encoding="utf-8")
    assert updated.count("@wrap") == 2


def test_patch_applier_mismatch_includes_search_and_closest_fragment(
    tmp_path: Path,
) -> None:
    applier = CursorPatchApplier(tmp_path)
    target = "list.py"
    (tmp_path / target).write_text(
        "    @router.get('/orders')\n"
        "    def list_orders():\n"
        "        return []\n",
        encoding="utf-8",
    )
    patch = (
        "<<<<<<< SEARCH\n"
        "    @_db_error_handler\n"
        "    @router.get('/orders')\n"
        "    def list_orders():\n"
        "=======\n"
        "    @_db_error_handler\n"
        "    @router.get('/orders')\n"
        "    def list_orders():\n"
        ">>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is False
    assert "mismatch: block 1 SEARCH code not found" in error
    assert "Diagnostic for block 1" in error
    assert "SEARCH (first lines" in error
    assert "@_db_error_handler" in error
    assert "Closest file fragment" in error
    assert "@router.get('/orders')" in error
    assert "SEARCH must be the current on-disk code" in error
