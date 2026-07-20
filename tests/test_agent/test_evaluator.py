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


def test_patch_applier_locates_search_when_unicode_escape_decoded_to_glyph(
    tmp_path: Path,
) -> None:
    """The Edit model single-escapes \\uXXXX over JSON, so on-disk source with a
    literal \\u4e00 arrives in SEARCH as the decoded glyph. The block must still
    locate via the escape-insensitive fallback instead of failing E2_LOCATE."""
    applier = CursorPatchApplier(tmp_path)
    target = "policy.py"
    (tmp_path / target).write_text(
        'PATTERN = r"^(?:[a-z]{1,3}|[\\u4e00-\\u9fa5]{1,2})$"\n'
        "VALUE = 1\n",
        encoding="utf-8",
    )
    patch = (
        "<<<<<<< SEARCH\n"
        'PATTERN = r"^(?:[a-z]{1,3}|[一-龥]{1,2})$"\n'
        "VALUE = 1\n"
        "=======\n"
        'PATTERN = r"^(?:[a-z]{1,3}|[一-龥]{1,2})$"\n'
        "VALUE = 2\n"
        ">>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is True, error
    assert "VALUE = 2" in (tmp_path / target).read_text(encoding="utf-8")


def test_escape_fold_equates_literal_escape_and_glyph() -> None:
    disk = 'r"[\\u4e00-\\u9fa5]"'
    search = 'r"[一-龥]"'
    assert CursorPatchApplier._escape_fold(disk) == CursorPatchApplier._escape_fold(
        search
    )
    # Case-insensitive on the hex, whitespace-insensitive overall.
    assert CursorPatchApplier._escape_fold(
        'r"[\\u4E00 - \\u9FA5]"'
    ) == CursorPatchApplier._escape_fold('r"[一-龥]"')


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


def test_patch_applier_unsticks_glued_block_boundaries(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path)
    target = "policy.py"
    (tmp_path / target).write_text(
        "    short_task_keywords: frozenset[str]\n"
        "    ingress_layer2_unique_types_max: int\n"
        "\n"
        "        short_task_keywords=frozenset(['a']),\n"
        "        ingress_layer2_unique_types_max=2,\n",
        encoding="utf-8",
    )
    # Model forgot the newline between REPLACE and the next SEARCH.
    patch = (
        "<<<<<<< SEARCH\n"
        "    short_task_keywords: frozenset[str]\n"
        "    ingress_layer2_unique_types_max: int\n"
        "=======\n"
        "    short_task_keywords: frozenset[str]\n"
        "    bot_nicknames: frozenset[str]\n"
        "    ingress_layer2_unique_types_max: int\n"
        ">>>>>>> REPLACE<<<<<<< SEARCH\n"
        "        short_task_keywords=frozenset(['a']),\n"
        "        ingress_layer2_unique_types_max=2,\n"
        "=======\n"
        "        short_task_keywords=frozenset(['a']),\n"
        "        bot_nicknames=frozenset(['b']),\n"
        "        ingress_layer2_unique_types_max=2,\n"
        ">>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is True, error
    text = (tmp_path / target).read_text(encoding="utf-8")
    assert "bot_nicknames: frozenset[str]" in text
    assert "bot_nicknames=frozenset(['b'])" in text
    assert ">>>>>>>" not in text
    assert "<<<<<<<" not in text


def test_patch_applier_rejects_markers_inside_replace_body(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path)
    target = "x.py"
    (tmp_path / target).write_text("def a():\n    return 1\n", encoding="utf-8")
    # Nested marker inside REPLACE survives boundary normalize.
    nested = (
        "<<<<<<< SEARCH\n"
        "def a():\n"
        "    return 1\n"
        "=======\n"
        "def a():\n"
        "<<<<<<< SEARCH\n"
        "    return 2\n"
        ">>>>>>> REPLACE\n"
        ">>>>>>> REPLACE"
    )
    success, error = applier.apply_patch(target, nested)
    assert success is False
    assert "invalid_patch:" in error
    assert "marker" in error.lower() or "SEARCH/REPLACE" in error


def test_patch_applier_rejects_more_than_max_blocks(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path, max_blocks=3)
    target = "routes.py"
    (tmp_path / target).write_text(
        "\n".join(f"def f{i}():\n    return {i}\n" for i in range(4)),
        encoding="utf-8",
    )
    parts = []
    for i in range(4):
        parts.append(
            "<<<<<<< SEARCH\n"
            f"def f{i}():\n"
            f"    return {i}\n"
            "=======\n"
            f"def f{i}():\n"
            f"    return {i + 10}\n"
            ">>>>>>> REPLACE"
        )
    success, error = applier.apply_patch(target, "\n".join(parts))
    assert success is False
    assert "too many SEARCH/REPLACE blocks (4 > 3)" in error
    assert "later Core decision_edit" in error
    assert (tmp_path / target).read_text(encoding="utf-8").count("return 10") == 0


def test_patch_applier_sequential_applies_top_to_bottom(tmp_path: Path) -> None:
    applier = CursorPatchApplier(tmp_path, sequential=True)
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


def test_patch_applier_mismatch_surfaces_the_single_diverging_line(
    tmp_path: Path,
) -> None:
    """A near-miss SEARCH (only one differing line) must expose that exact line
    verbatim, otherwise the inner retry regenerates the same failing patch."""
    applier = CursorPatchApplier(tmp_path)
    target = "noise.py"
    on_disk = [
        "def strip_chat_noise(raw: str) -> str:",
        '    """doc."""',
        "    text = raw.strip()",
        "    text = re.sub(r'\\[CQ:at,qq=\\d+\\]', '', text)",
        "    return text.strip()",
    ]
    (tmp_path / target).write_text("\n".join(on_disk) + "\n", encoding="utf-8")

    # Model's SEARCH matches every line except one (the regex line hallucinated).
    search = list(on_disk)
    search[3] = "    text = re.sub(r'@mention', '', text)"
    patch = (
        "<<<<<<< SEARCH\n"
        + "\n".join(search)
        + "\n=======\n"
        + "\n".join(on_disk)
        + "\n>>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is False
    # The exact on-disk line that the model got wrong must be visible verbatim.
    assert "text = re.sub(r'\\[CQ:at,qq=\\d+\\]', '', text)" in error
    # And it must be flagged as the diverging line, not silently truncated away.
    assert "✗" in error
    assert "your SEARCH had:" in error
    # Matching lines around it should still be present (full window, not 3 lines).
    assert "return text.strip()" in error


def test_patch_applier_mismatch_flags_phantom_search_and_correct_candidate(
    tmp_path: Path,
) -> None:
    """Target-state SEARCH (new line not on disk) must name the phantom and give
    a paste-ready CORRECT SEARCH candidate from the remaining real lines."""
    applier = CursorPatchApplier(tmp_path)
    target = "noise_policy.py"
    (tmp_path / target).write_text(
        "        ),\n"
        "        deictic_followup_patterns=_compile_deictic(deictic),\n"
        "    )\n",
        encoding="utf-8",
    )
    # Model put the not-yet-existing kwarg into SEARCH (classic insert failure).
    patch = (
        "<<<<<<< SEARCH\n"
        "        deictic_followup_patterns=_compile_deictic(deictic),\n"
        "        bot_nicknames=bot_nicknames_set,\n"
        "    )\n"
        "=======\n"
        "        deictic_followup_patterns=_compile_deictic(deictic),\n"
        "        bot_nicknames=bot_nicknames_set,\n"
        "    )\n"
        ">>>>>>> REPLACE"
    )

    success, error = applier.apply_patch(target, patch)

    assert success is False
    assert "mismatch: block 1 SEARCH code not found" in error
    assert "PHANTOM SEARCH lines" in error
    assert "bot_nicknames=bot_nicknames_set," in error
    assert "move them to REPLACE only" in error
    assert "CORRECT SEARCH candidate" in error
    # Candidate should be the contiguous on-disk span of the non-phantom lines.
    assert "deictic_followup_patterns=_compile_deictic(deictic)," in error
    assert "    )" in error
