from __future__ import annotations

from src.planner.kinds import SubTaskKind
from src.planner.strategy import (
    ExecutionStrategy,
    batch_parallelizable,
    ready_execution_batches,
    ready_pending_nodes,
    select_strategy,
)
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


def test_ready_execution_batches_groups_independent_nodes() -> None:
    tree = TaskTree(
        root_task="parallel discovery",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find api"),
            SubTaskNode(id="st-2", kind=SubTaskKind.DIAGNOSE, description="find schema"),
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.EDIT,
                description="patch api",
                depends_on=["st-1", "st-2"],
            ),
        ],
    )

    batches = ready_execution_batches(tree)

    assert [[node.id for node in batch] for batch in batches] == [["st-1", "st-2"]]
    assert [node.id for node in ready_pending_nodes(tree)] == ["st-1", "st-2"]
    assert select_strategy(tree) == ExecutionStrategy.MIXED
    assert batch_parallelizable(batches[0]) is True


def test_ready_execution_batches_splits_edit_write_conflicts() -> None:
    tree = TaskTree(
        root_task="two edits",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="patch route",
                context_files=["app.py"],
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="patch report",
                write_scope=["app.py"],
            ),
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.EDIT,
                description="patch tests",
                write_scope=["tests/test_app.py"],
            ),
        ],
    )

    batches = ready_execution_batches(tree)

    assert [[node.id for node in batch] for batch in batches] == [
        ["st-1", "st-3"],
        ["st-2"],
    ]
    assert batch_parallelizable(batches[0]) is False


def test_first_pending_uses_ready_frontier_after_dependency_success() -> None:
    tree = TaskTree(
        root_task="dependency",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="done",
                status=SubTaskStatus.SUCCESS,
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="patch",
                depends_on=["st-1"],
            ),
        ],
    )

    assert tree.first_pending() is not None
    assert tree.first_pending().id == "st-2"
