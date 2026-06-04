from __future__ import annotations

from src.harness.subtask.context_pipeline import ExecutorRuntimeState
from src.harness.subtask.tool_recovery import apply_post_tool_recovery
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_post_tool_recovery_enables_read_after_edit_file_failure() -> None:
    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    runtime = ExecutorRuntimeState(
        paths_only_mode=False,
        use_paths_only=False,
        preloaded_paths=frozenset({"app.py"}),
        truncated_paths=frozenset(),
        active_runtime_tools=frozenset({"edit_file"}),
        explore_restricted=True,
    )
    recovery = apply_post_tool_recovery(
        subtask=subtask,
        runtime=runtime,
        error_trace=["edit_file: old_string not found in app.py"],
        splice_edit=False,
    )
    assert runtime.edit_read_fallback is True
    assert "read_file" in (recovery.runtime_tools or frozenset())
    assert recovery.nudges


def test_post_tool_recovery_splice_anchor_failure_nudge() -> None:
    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    runtime = ExecutorRuntimeState(
        paths_only_mode=False,
        use_paths_only=False,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset(),
        active_runtime_tools=frozenset({"replace_symbol"}),
        explore_restricted=True,
    )
    recovery = apply_post_tool_recovery(
        subtask=subtask,
        runtime=runtime,
        error_trace=[
            "replace_symbol failed after 2 attempt(s): anchor hash mismatch",
        ],
        splice_edit=True,
    )
    assert recovery.nudges
    assert runtime.edit_read_fallback is False


def test_post_tool_recovery_duplicate_diagnose_forces_summary_only() -> None:
    subtask = SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find")
    runtime = ExecutorRuntimeState(
        paths_only_mode=False,
        use_paths_only=False,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset(),
        active_runtime_tools=frozenset({"grep_search", "map_search"}),
        explore_restricted=False,
    )

    recovery = apply_post_tool_recovery(
        subtask=subtask,
        runtime=runtime,
        error_trace=["Blocked duplicate grep_search in diagnose: 'x' in main.py."],
        splice_edit=False,
    )

    assert runtime.active_runtime_tools == frozenset()
    assert runtime.explore_restricted is True
    assert recovery.nudges
