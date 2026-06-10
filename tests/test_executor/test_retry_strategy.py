from __future__ import annotations

from src.executor.retry_strategy import (
    build_executor_retry_strategy,
    classify_failure_pattern,
    replan_revision_directive,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_attempt1_edit_no_strategy_change() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix sql")
    s = build_executor_retry_strategy(st, subtask_attempt=1, prior_errors=None)
    assert s.paths_only is False


def test_attempt2_edit_paths_only_with_digest_hint() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix sql")
    s = build_executor_retry_strategy(
        st,
        subtask_attempt=2,
        prior_errors=["read_file blocked: already preloaded"],
    )
    assert s.paths_only is True
    assert "digest" in s.user_hint.lower() or "context_search" in s.user_hint


def test_attempt3_includes_replan_hint() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix sql")
    s = build_executor_retry_strategy(
        st,
        subtask_attempt=3,
        prior_errors=["hit executor turn limit"],
    )
    assert s.replan_hint is not None


def test_classify_read_preload_loop() -> None:
    assert classify_failure_pattern(["read_file blocked: already preloaded"]) == "read_preload_loop"


def test_classify_strategy_mismatch() -> None:
    assert classify_failure_pattern(["missing_info=['diagnose_strategy_mismatch']"]) == "strategy_mismatch"
    assert classify_failure_pattern(["missing_info=['column_mapping']"]) == "strategy_mismatch"


def test_replan_directive_forbids_same_plan() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix boarding pass view")
    text = replan_revision_directive(
        failed_subtask=st,
        error_trace=["read_file blocked: already preloaded"],
    )
    assert "MUST change" in text
    assert "diagnose handoff" in text.lower()


def test_replan_directive_strategy_mismatch_reclassifies_task() -> None:
    st = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="refactor order detail")
    text = replan_revision_directive(
        failed_subtask=st,
        error_trace=["missing_info=['diagnose_strategy_mismatch']"],
    )
    assert "Do NOT keep searching for view columns" in text
    assert "function_refactor" in text
