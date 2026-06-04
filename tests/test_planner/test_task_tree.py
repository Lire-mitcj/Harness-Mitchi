from __future__ import annotations

from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


def test_task_tree_first_pending_and_mark_success() -> None:
    tree = TaskTree(
        root_task="fix login",
        nodes=[
            SubTaskNode(id="st-1", description="read api"),
            SubTaskNode(id="st-2", description="patch handler"),
        ],
    )
    first = tree.first_pending()
    assert first is not None
    assert first.id == "st-1"

    tree.mark_success("st-1", checkpoint_id="cp-abc")
    assert tree.first_pending() is not None
    assert tree.first_pending().id == "st-2"
    assert tree.get("st-1").checkpoint_id == "cp-abc"
    assert tree.get("st-1").status == SubTaskStatus.SUCCESS


def test_task_tree_replace_remaining_keeps_completed() -> None:
    tree = TaskTree(
        root_task="db bug",
        nodes=[
            SubTaskNode(id="st-1", description="done", status=SubTaskStatus.SUCCESS),
            SubTaskNode(id="st-2", description="failed", status=SubTaskStatus.FAILED),
        ],
    )
    tree.replace_remaining(
        [
            SubTaskNode(id="st-3", description="describe table"),
            SubTaskNode(id="st-4", description="fix api"),
        ]
    )
    assert [n.id for n in tree.nodes] == ["st-1", "st-3", "st-4"]
    assert tree.version == 2


def test_first_pending_respects_depends_on() -> None:
    tree = TaskTree(
        root_task="db fix",
        nodes=[
            SubTaskNode(id="st-1", description="schema", status=SubTaskStatus.PENDING),
            SubTaskNode(
                id="st-2",
                description="patch api",
                status=SubTaskStatus.PENDING,
                depends_on=["st-1"],
            ),
        ],
    )
    assert tree.first_pending() is not None
    assert tree.first_pending().id == "st-1"
    tree.mark_success("st-1")
    assert tree.first_pending() is not None
    assert tree.first_pending().id == "st-2"


def test_task_tree_outline_contains_status_icons() -> None:
    tree = TaskTree(
        root_task="demo",
        nodes=[SubTaskNode(id="st-1", description="step", status=SubTaskStatus.PENDING)],
    )
    outline = tree.to_outline()
    assert "○ [st-1]" in outline
    assert "demo" in outline
