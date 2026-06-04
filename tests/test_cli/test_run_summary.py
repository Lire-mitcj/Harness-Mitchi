from __future__ import annotations

from src.cli.run_summary import build_terminal_run_summary
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


def test_build_terminal_run_summary_is_compact() -> None:
    tree = TaskTree(
        root_task="Find project views",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="Map-search view-related symbols",
                status=SubTaskStatus.SUCCESS,
            ),
        ],
    )
    summaries = {
        "st-1": "## Diagnosis\n\n| File | Symbol |\n|---|---|\n| a.sql | v_x |",
    }
    text = build_terminal_run_summary(tree, summaries)
    assert "Done (1/1 steps)" in text
    assert "Task: Find project views" in text
    assert "[st-1]" in text
    assert "a.sql" in text
    assert "## Step results" not in text
    assert "done when:" not in text
