from __future__ import annotations

from pathlib import Path

from src.executor.edit_guard import (
    EDIT_IDENTICAL_HINT,
    EDIT_NOT_FOUND_HINT,
    edit_failure_hint,
    is_edit_ambiguous_error,
    is_edit_identical_error,
    is_edit_not_found_error,
    should_skip_explore_fold,
)
from src.executor.retry_strategy import build_executor_retry_strategy, classify_failure_pattern
from src.orchestrator.escalation import EscalationAction, decide_subtask_escalation
from src.planner.kinds import SubTaskKind
from src.planner.prior_context import extract_line_refs_from_text
from src.planner.task_tree import SubTaskNode


def test_classify_edit_ambiguous() -> None:
    err = "edit_file: old_string appears 2 times in app.py"
    assert classify_failure_pattern([err]) == "edit_ambiguous"


def test_classify_edit_identical() -> None:
    err = "edit_file: old_string and new_string are identical"
    assert classify_failure_pattern([err]) == "edit_identical"


def test_classify_edit_not_found() -> None:
    err = "edit_file: old_string not found in app.py"
    assert classify_failure_pattern([err]) == "edit_not_found"


def test_edit_failure_hint_prefers_latest() -> None:
    assert edit_failure_hint(
        [
            "edit_file: old_string not found in app.py",
            "edit_file: old_string and new_string are identical",
        ]
    ) == EDIT_IDENTICAL_HINT


def test_escalation_replan_on_edit_not_found_turn_limit() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    verdict = decide_subtask_escalation(
        st,
        {
            "success": False,
            "failure_code": "turn_limit",
            "error_trace": [
                "edit_file: old_string not found in app.py",
            ],
            "changed_files": [],
        },
        attempt=1,
        max_subtask_retries=3,
    )
    assert verdict.action == EscalationAction.REPLAN


def test_retry_attempt2_with_digest_restricts_explore(tmp_path: Path) -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    s = build_executor_retry_strategy(
        st,
        subtask_attempt=2,
        prior_errors=["hit executor turn limit"],
        prior_exploration="Grep queries already run:\n  - 'x' in app.py\n",
    )
    assert s.restrict_explore is True
    assert s.paths_only is False
    assert "do not repeat the same context_search" in s.user_hint


def test_escalation_replan_on_first_ambiguous_turn_limit() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    verdict = decide_subtask_escalation(
        st,
        {
            "success": False,
            "failure_code": "turn_limit",
            "error_trace": [
                "edit_file: old_string appears 2 times in app.py",
            ],
            "changed_files": [],
        },
        attempt=1,
        max_subtask_retries=3,
    )
    assert verdict.action == EscalationAction.REPLAN


def test_extract_line_refs_from_diagnose_summary(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n")
    summary = (
        "Findings\n\n"
        "make_boarding_pass_pdf at app.py:1584-1655\n"
        "Also see app.py:1600-1620 for query.\n"
    )
    refs = extract_line_refs_from_text(summary, tmp_path)
    assert "app.py" in refs
    assert refs["app.py"][0] <= 1584


def test_should_skip_fold_after_ambiguous_edit() -> None:
    assert should_skip_explore_fold(
        ["edit_file: old_string appears 2 times in app.py"]
    )


def test_is_edit_ambiguous_error() -> None:
    assert is_edit_ambiguous_error(
        "old_string appears 2 times in app.py. Provide a more unique string"
    )


def test_is_edit_identical_error() -> None:
    assert is_edit_identical_error(
        "old_string and new_string are identical — provide the actual code change"
    )


def test_is_edit_not_found_error() -> None:
    assert is_edit_not_found_error("old_string not found in app.py")
    assert edit_failure_hint(["edit_file: old_string not found in app.py"]) == EDIT_NOT_FOUND_HINT
