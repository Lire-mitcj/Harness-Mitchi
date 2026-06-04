from __future__ import annotations

from src.orchestrator.escalation import EscalationAction, decide_subtask_escalation
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_turn_limit_retries_before_replan() -> None:
    node = SubTaskNode(id="st-2", description="edit", kind=SubTaskKind.EDIT)
    result = {
        "success": False,
        "failure_code": "turn_limit",
        "error_trace": ["turn limit"],
        "changed_files": [],
        "quality_gate_failures": 0,
    }
    v1 = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)
    assert v1.action == EscalationAction.RETRY_SUBTASK
    v3 = decide_subtask_escalation(node, result, attempt=3, max_subtask_retries=3)
    assert v3.action == EscalationAction.REPLAN


def test_whitelist_block_replans_immediately() -> None:
    node = SubTaskNode(id="st-2", description="edit", kind=SubTaskKind.EDIT)
    result = {
        "success": False,
        "failure_code": "turn_limit",
        "error_trace": ["edit_file blocked: path(s) [other.py] outside subtask edit scope"],
        "changed_files": [],
        "quality_gate_failures": 0,
    }
    v = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)
    assert v.action == EscalationAction.REPLAN


def test_quality_gate_exhausted_retries_first() -> None:
    node = SubTaskNode(id="st-2", description="edit", kind=SubTaskKind.EDIT)
    result = {
        "success": False,
        "failure_code": "quality_gate_exhausted",
        "error_trace": ["L0 fail"],
        "changed_files": ["main.py"],
        "quality_gate_failures": 3,
    }
    v = decide_subtask_escalation(node, result, attempt=2, max_subtask_retries=3)
    assert v.action == EscalationAction.RETRY_SUBTASK


def test_diagnose_turn_limit_retries_once_then_replans() -> None:
    node = SubTaskNode(id="st-1", description="diagnose", kind=SubTaskKind.DIAGNOSE)
    result = {
        "success": False,
        "failure_code": "turn_limit",
        "error_trace": ["turn limit"],
        "changed_files": [],
        "quality_gate_failures": 0,
    }
    v1 = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)
    assert v1.action == EscalationAction.RETRY_SUBTASK
    v2 = decide_subtask_escalation(node, result, attempt=2, max_subtask_retries=3)
    assert v2.action == EscalationAction.REPLAN


def test_diagnose_acceptance_unmet_replans_immediately() -> None:
    node = SubTaskNode(id="st-1", description="diagnose", kind=SubTaskKind.DIAGNOSE)
    result = {
        "success": False,
        "failure_code": "exit_gate",
        "error_trace": [
            "Diagnose summary indicates acceptance_criteria was not met — revise the plan",
        ],
        "changed_files": [],
        "quality_gate_failures": 0,
    }
    v = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)
    assert v.action == EscalationAction.REPLAN


def test_skill_executor_failure_aborts_without_retry_or_replan() -> None:
    node = SubTaskNode(id="st-2", description="edit", kind=SubTaskKind.EDIT)
    result = {
        "success": False,
        "failure_code": "skill_executor",
        "error_trace": ["Patch planner produced non-executable plan"],
        "changed_files": [],
        "quality_gate_failures": 0,
    }
    verdict = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)

    assert verdict.action == EscalationAction.ABORT
    assert "Patch planner produced non-executable plan" in verdict.reason


def test_success_must_not_escalate() -> None:
    node = SubTaskNode(id="st-1", description="diagnose", kind=SubTaskKind.DIAGNOSE)
    result = {
        "success": True,
        "failure_code": "",
        "error_trace": [],
        "changed_files": [],
        "summarized_after_limit": True,
    }
    v = decide_subtask_escalation(node, result, attempt=1, max_subtask_retries=3)
    assert v.action == EscalationAction.ABORT
