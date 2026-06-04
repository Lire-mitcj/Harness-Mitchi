from __future__ import annotations

from pathlib import Path

from src.planner.prior_context import (
    extract_paths_from_text,
    format_prior_summaries_block,
    prior_summaries_for_node,
    propagate_diagnose_paths,
)
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree


def test_prior_summaries_transitive() -> None:
    tree = TaskTree(
        root_task="x",
        nodes=[
            SubTaskNode(id="st-1", description="search", kind=SubTaskKind.DIAGNOSE),
            SubTaskNode(
                id="st-2",
                description="edit",
                kind=SubTaskKind.EDIT,
                depends_on=["st-1"],
            ),
        ],
    )
    summaries = {"st-1": "Found views in app.py and db/schema.sql"}
    got = prior_summaries_for_node(tree, tree.nodes[1], summaries)
    assert got == summaries


def test_format_prior_summaries_block() -> None:
    text = format_prior_summaries_block({"st-1": "grep hit in app.py:120"})
    assert "Prior subtask results" in text
    assert "app.py" in text


def test_propagate_diagnose_paths(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("print('hi')", encoding="utf-8")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-1", description="search", kind=SubTaskKind.DIAGNOSE),
            SubTaskNode(
                id="st-2",
                description="patch",
                kind=SubTaskKind.EDIT,
                depends_on=["st-1"],
                context_files=[],
            ),
        ],
    )
    summary = "Relevant handler in app.py lines 10-40"
    propagate_diagnose_paths(tree, "st-1", summary, tmp_path)
    assert "app.py" in tree.nodes[1].context_files


def test_extract_paths_from_text(tmp_path: Path) -> None:
    f = tmp_path / "views" / "home.py"
    f.parent.mkdir()
    f.write_text("x", encoding="utf-8")
    paths = extract_paths_from_text("See views/home.py for page layout", tmp_path)
    assert "views/home.py" in paths
