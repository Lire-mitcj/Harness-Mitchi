from __future__ import annotations

from pathlib import Path

from src.harness.gates.plan_gate import validate_plan
from src.harness.gates.types import GateVerdict
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree
from src.planner.tool_policy import (
    default_allowed_tools,
    effective_allowed_tools,
    normalize_allowed_tools,
    validate_node_tools,
)


def test_default_tools_by_kind() -> None:
    diag = default_allowed_tools(SubTaskKind.DIAGNOSE)
    assert "context_search" in diag
    assert "read_file" not in diag
    assert "map_search" not in diag
    assert "shell_exec" not in diag
    shell = default_allowed_tools(SubTaskKind.SHELL)
    assert "shell_exec" in shell
    assert "write_file" not in shell
    design = default_allowed_tools(SubTaskKind.DESIGN)
    assert design == ["context_search"]


def test_normalize_rejects_out_of_kind_tools() -> None:
    tools = normalize_allowed_tools(
        SubTaskKind.DIAGNOSE,
        ["context_search", "write_file", "shell_exec"],
    )
    assert "context_search" in tools
    assert "write_file" not in tools
    assert "shell_exec" not in tools


def test_validate_blocks_diagnose_with_shell() -> None:
    node = SubTaskNode(
        id="st-1",
        description="check schema",
        kind=SubTaskKind.DIAGNOSE,
        allowed_tools=["context_search", "shell_exec"],
        acceptance_criteria="schema listed",
    )
    blocks, _ = validate_node_tools(node)
    assert any("shell_exec" in b or "forbids" in b for b in blocks)


def test_validate_blocks_edit_without_write_tools() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api",
        kind=SubTaskKind.EDIT,
        allowed_tools=["context_search"],
        acceptance_criteria="api fixed",
    )
    blocks, _ = validate_node_tools(node)
    assert any("edit_file" in b or "write_file" in b for b in blocks)


def test_plan_gate_blocks_invalid_tool_policy(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="run mysql",
                kind=SubTaskKind.DIAGNOSE,
                allowed_tools=["shell_exec"],
                acceptance_criteria="done",
            )
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict == GateVerdict.BLOCK


def test_effective_allowed_tools_on_node() -> None:
    node = SubTaskNode(
        id="st-1",
        description="x",
        kind=SubTaskKind.VERIFY,
        allowed_tools=["shell_exec", "context_search"],
    )
    assert effective_allowed_tools(node) == frozenset({"context_search", "shell_exec"})


def test_verify_and_shell_can_declare_context_search_tools() -> None:
    verify = SubTaskNode(
        id="st-1",
        description="verify",
        kind=SubTaskKind.VERIFY,
        allowed_tools=["shell_exec", "context_search"],
        acceptance_criteria="verification output",
    )
    shell = SubTaskNode(
        id="st-2",
        description="shell",
        kind=SubTaskKind.SHELL,
        allowed_tools=["shell_exec", "context_search"],
        acceptance_criteria="command output",
    )

    assert validate_node_tools(verify)[0] == []
    assert validate_node_tools(shell)[0] == []
