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


def test_resolve_project_paths_ambiguous_and_unique(tmp_path, caplog) -> None:
    from src.planner.context_policy import resolve_project_paths
    import logging

    # Setup temporary files:
    # 1. A unique file: src/utils/helper.py
    # 2. Ambiguous files: db/init/init.sql and src/init.sql
    unique_dir = tmp_path / "src" / "utils"
    unique_dir.mkdir(parents=True, exist_ok=True)
    unique_file = unique_dir / "helper.py"
    unique_file.touch()

    db_init_dir = tmp_path / "db" / "init"
    db_init_dir.mkdir(parents=True, exist_ok=True)
    init_sql1 = db_init_dir / "init.sql"
    init_sql1.touch()

    src_dir = tmp_path / "src"
    init_sql2 = src_dir / "init.sql"
    init_sql2.touch()

    # Unique match should resolve to full path
    paths = ["helper.py"]
    resolved = resolve_project_paths(tmp_path, paths)
    assert resolved == ["src/utils/helper.py"]

    # Ambiguous match should log a warning and retain original path
    with caplog.at_level(logging.WARNING):
        paths_ambiguous = ["init.sql"]
        resolved_ambiguous = resolve_project_paths(tmp_path, paths_ambiguous)
        assert resolved_ambiguous == ["init.sql"]
        assert any("ambiguous_path: init.sql matches multiple files" in record.message for record in caplog.records)

    # Missing file should retain original path
    paths_missing = ["nonexistent.py"]
    resolved_missing = resolve_project_paths(tmp_path, paths_missing)
    assert resolved_missing == ["nonexistent.py"]

