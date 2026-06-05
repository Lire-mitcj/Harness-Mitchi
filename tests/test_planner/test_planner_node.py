from __future__ import annotations

import json

from src.config.settings import MitKIISettings
from src.planner.planner_node import (
    _merge_replanned_tree,
    _parse_task_tree,
    extract_planning_trace,
    fallback_replan_for_failed_edit,
    planner_output_instruction,
)
from src.planner.task_tree import SubTaskKind, SubTaskNode, SubTaskStatus, TaskTree

_TRACE = """\
1. Task type: bug_fix — handler missing error case
2. Scout leverage: root_cause known, skip diagnose
3. Subtask sketch: st-1 edit; st-2 verify
4. Tool audit: edit has edit_file; verify has shell_exec
5. Ordering: st-2 depends st-1"""

_PAYLOAD = {
    "root_task": "Fix 500",
    "nodes": [
        {
            "id": "st-1",
            "kind": "edit",
            "description": "Patch handler",
            "acceptance_criteria": "503 on DB error",
            "allowed_tools": ["read_file", "edit_file"],
            "context_files": ["api/routes.py"],
            "depends_on": [],
            "needs_l1": True,
        },
        {
            "id": "st-2",
            "kind": "verify",
            "description": "Run tests",
            "acceptance_criteria": "pytest exits 0",
            "allowed_tools": ["shell_exec", "read_file"],
            "context_files": ["tests/test_users.py"],
            "depends_on": ["st-1"],
            "needs_l1": False,
        },
    ],
}

_BUG_FIX_SAMPLE = (
    f"<planning_trace>\n{_TRACE}\n</planning_trace>\n{json.dumps(_PAYLOAD)}"
)


def test_planner_output_instruction_no_trace() -> None:
    msg = planner_output_instruction(require_trace=False)
    assert "ONE raw TaskTree JSON" in msg
    assert "8 fields" in msg


def test_planner_output_instruction_with_trace() -> None:
    msg = planner_output_instruction(require_trace=True)
    assert "<planning_trace>" in msg


def test_effective_planner_model_prefers_planner_model() -> None:
    settings = MitKIISettings(
        model="openai/deepseek-ai/DeepSeek-V4-Flash",
        planner_model="openai/Qwen/Qwen2.5-7B-Instruct",
        scout_model="openai/other/scout",
    )
    assert settings.effective_planner_model == "openai/Qwen/Qwen2.5-7B-Instruct"


def test_effective_planner_model_falls_back_to_scout() -> None:
    settings = MitKIISettings(
        model="openai/deepseek-ai/DeepSeek-V4-Flash",
        planner_model=None,
        scout_model="openai/Qwen/Qwen2.5-7B-Instruct",
    )
    assert settings.effective_planner_model == "openai/Qwen/Qwen2.5-7B-Instruct"


def test_planner_defaults() -> None:
    assert MitKIISettings.model_fields["planner_max_tokens"].default == 1536
    assert MitKIISettings.model_fields["planner_json_mode"].default is True
    settings = MitKIISettings()
    assert settings.planner_trace is False
    assert settings.scout_trace is False
    assert settings.scout_max_tokens == 1024
    assert settings.executor_summary_max_tokens == 768
    assert settings.executor_summary_timeout == 30


def test_extract_planning_trace() -> None:
    trace = extract_planning_trace(_BUG_FIX_SAMPLE)
    assert trace is not None
    assert "bug_fix" in trace


def test_parse_task_tree_after_cot() -> None:
    tree = _parse_task_tree(_BUG_FIX_SAMPLE, fallback_task="Fix 500")
    assert tree.root_task == "Fix 500"
    assert len(tree.nodes) == 2
    assert tree.nodes[0].kind == SubTaskKind.EDIT
    assert tree.nodes[1].kind == SubTaskKind.VERIFY
    assert tree.nodes[1].depends_on == ["st-1"]


def test_parse_json_only_without_trace() -> None:
    raw = json.dumps({
        "root_task": "x",
        "nodes": [{
            "id": "st-1",
            "kind": "shell",
            "description": "run",
            "acceptance_criteria": "ok",
            "allowed_tools": ["shell_exec"],
            "context_files": [],
            "depends_on": [],
        }],
    })
    tree = _parse_task_tree(raw, fallback_task="x")
    assert len(tree.nodes) == 1
    assert tree.nodes[0].kind == SubTaskKind.SHELL


def test_merge_replanned_tree_replaces_failed_preserves_tail() -> None:
    current = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="done",
                status=SubTaskStatus.SUCCESS,
            ),
            SubTaskNode(
                id="st-2",
                description="failed edit",
                status=SubTaskStatus.FAILED,
            ),
            SubTaskNode(
                id="st-3",
                description="verify unchanged",
                status=SubTaskStatus.PENDING,
                depends_on=["st-2"],
            ),
        ],
    )
    revised = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-2a", kind=SubTaskKind.DIAGNOSE, description="grep first"),
            SubTaskNode(id="st-2b", kind=SubTaskKind.EDIT, description="narrow edit"),
        ],
    )
    merged = _merge_replanned_tree(current, revised, failed_subtask_id="st-2")
    ids = [n.id for n in merged.nodes]
    assert ids == ["st-1", "st-2a", "st-2b", "st-3"]
    assert merged.nodes[0].status == SubTaskStatus.SUCCESS
    assert merged.nodes[3].description == "verify unchanged"
    assert merged.nodes[3].depends_on == ["st-2b"]
    assert all(n.status == SubTaskStatus.PENDING for n in merged.nodes[1:3])


def test_merge_replanned_tree_skips_duplicate_completed_ids() -> None:
    current = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="done",
                status=SubTaskStatus.SUCCESS,
            ),
            SubTaskNode(
                id="st-2",
                description="failed edit",
                status=SubTaskStatus.FAILED,
            ),
        ],
    )
    revised = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-1", description="dup diagnose"),
            SubTaskNode(id="st-2b", description="retry edit"),
        ],
    )
    merged = _merge_replanned_tree(current, revised, failed_subtask_id="st-2")
    ids = [n.id for n in merged.nodes]
    assert ids == ["st-1", "st-2b"]
    assert merged.nodes[0].status == SubTaskStatus.SUCCESS
    assert merged.nodes[1].status == SubTaskStatus.PENDING


def test_fallback_replan_for_failed_edit_inserts_diagnose_and_edit() -> None:
    current = TaskTree(
        root_task="把登机牌查询接口改成用视图查询",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="修改 make_boarding_pass_pdf 函数以使用视图查询",
                context_files=["app.py"],
                allowed_tools=["context_search", "edit_file"],
                status=SubTaskStatus.FAILED,
            )
        ],
    )

    merged = fallback_replan_for_failed_edit(current, failed_subtask_id="st-1")

    assert merged is not None
    assert [node.kind for node in merged.nodes] == [
        SubTaskKind.DIAGNOSE,
        SubTaskKind.EDIT,
    ]
    assert merged.nodes[1].depends_on == ["st-1a"]
    assert "HANDOFF_CONTRACT_JSON" in merged.nodes[0].acceptance_criteria
