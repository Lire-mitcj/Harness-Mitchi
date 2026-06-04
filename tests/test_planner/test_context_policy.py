from __future__ import annotations

from src.harness.gates.plan_gate import validate_plan
from src.planner.context_policy import effective_context_files, enrich_task_tree_context_files
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


def test_edit_inherits_diagnose_context_files() -> None:
    tree = TaskTree(
        root_task="fix registration transaction",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="diagnose",
                kind=SubTaskKind.DIAGNOSE,
                context_files=["app.py", "main.py", "db/init/01_schema.sql"],
            ),
            SubTaskNode(
                id="st-2",
                description="fix transaction in registration route",
                kind=SubTaskKind.EDIT,
                context_files=["app.py", "main.py"],
                depends_on=["st-1"],
            ),
        ],
    )
    enrich_task_tree_context_files(tree)
    assert tree.get("st-2").context_files == [
        "app.py",
        "main.py",
        "db/init/01_schema.sql",
    ]


def test_effective_context_files_without_mutating_plan() -> None:
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="d",
                kind=SubTaskKind.DIAGNOSE,
                context_files=["db/init/01_schema.sql"],
            ),
            SubTaskNode(
                id="st-2",
                description="e",
                kind=SubTaskKind.EDIT,
                context_files=["main.py"],
                depends_on=["st-1"],
            ),
        ],
    )
    effective = effective_context_files(tree, tree.get("st-2"))  # type: ignore[arg-type]
    assert effective == ["main.py", "db/init/01_schema.sql"]
    assert tree.get("st-2").context_files == ["main.py"]


def test_plan_gate_enriches_before_sql_warn() -> None:
    tree = TaskTree(
        root_task="fix sp_register_passenger transaction",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="Locate transaction schema target",
                kind=SubTaskKind.DIAGNOSE,
                context_files=["db/init/01_schema.sql", "main.py"],
                acceptance_criteria="Output file:line, symbol, and schema snippet/decision",
            ),
            SubTaskNode(
                id="st-2",
                description="Fix sp_register_passenger transaction handling",
                kind=SubTaskKind.EDIT,
                context_files=["main.py"],
                depends_on=["st-1"],
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="Transaction handling is fixed",
            ),
        ],
    )
    result = validate_plan(tree, project_root=__import__("pathlib").Path("."))
    assert result.passed
    assert "db/init/01_schema.sql" in tree.get("st-2").context_files  # type: ignore[union-attr]
