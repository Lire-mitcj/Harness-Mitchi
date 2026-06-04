from __future__ import annotations

from src.agent.framework_guard import (
    blocked_framework_browse,
    format_framework_browse_denial,
)
from src.agent.shell_guard import ShellCommandTracker, normalize_shell_command
from src.harness.gates.exit_gate import ExitCheckInput, validate_exit
from src.harness.gates.types import GateVerdict
from src.planner.task_tree import SubTaskKind, SubTaskNode, SubTaskStatus


def test_shell_tracker_blocks_duplicate_commands() -> None:
    tracker = ShellCommandTracker(dedup_limit=2, stagnant_limit=5)
    cmd = "docker exec db mysql -e 'SELECT 1'"
    assert tracker.check(cmd) is None
    tracker.record_run(cmd)
    assert tracker.check(cmd) is None
    tracker.record_run(cmd)
    deny = tracker.check(cmd)
    assert deny is not None
    assert "duplicate" in deny.lower()


def test_shell_tracker_blocks_stagnant_failures() -> None:
    tracker = ShellCommandTracker(dedup_limit=5, stagnant_limit=2)
    cmd = "docker exec db mysql -e 'INSERT'"
    tracker.record_run(cmd)
    tracker.record_outcome(cmd, success=False)
    assert tracker.check(cmd) is None
    tracker.record_run(cmd)
    tracker.record_outcome(cmd, success=False)
    deny = tracker.check(cmd)
    assert deny is not None
    assert "stagnant" in deny.lower()


def test_normalize_shell_command() -> None:
    assert normalize_shell_command("  docker   exec   db  ") == "docker exec db"


def test_exit_gate_blocks_empty_answer() -> None:
    node = SubTaskNode(id="st-1", description="diagnose", kind=SubTaskKind.DIAGNOSE)
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="",
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_edit_without_changes() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api.py",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="Done fixing the API handler.",
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_passes_edit_with_changes() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api.py",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="Updated api.py to handle null user_id.",
            error_trace=[],
            changed_files=["api.py"],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_subtask_effective_needs_l1() -> None:
    edit = SubTaskNode(id="a", description="x", kind=SubTaskKind.EDIT)
    diag = SubTaskNode(id="b", description="x", kind=SubTaskKind.DIAGNOSE)
    assert edit.effective_needs_l1() is True
    assert diag.effective_needs_l1() is False
    override = SubTaskNode(id="c", description="x", kind=SubTaskKind.DIAGNOSE, needs_l1=True)
    assert override.effective_needs_l1() is True


def test_framework_browse_blocks_harness_list_dir() -> None:
    blocked = blocked_framework_browse(
        "Create gate_demo.py",
        "list_dir",
        {"path": "src/harness", "recursive": True},
    )
    assert blocked == ["src/harness"]
    deny = format_framework_browse_denial(blocked, "list_dir")
    assert "list_dir blocked" in deny


def test_framework_browse_allows_when_user_names_path() -> None:
    msg = "Inspect src/harness/engine.py internals"
    blocked = blocked_framework_browse(
        msg,
        "grep_search",
        {"path": "src/harness", "pattern": "class"},
    )
    assert blocked == []
