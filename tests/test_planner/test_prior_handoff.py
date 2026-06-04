from __future__ import annotations

from pathlib import Path

from src.planner.kinds import SubTaskKind
from src.planner.prior_context import (
    extract_symbol_hits_from_text,
    format_diagnose_handoff_block,
    propagate_diagnose_paths,
    rank_edit_relevant_paths,
)
from src.planner.task_tree import SubTaskNode, TaskTree


def test_extract_symbol_hits_from_map_table(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("def fn(): pass\n")
    text = "| app.py | 1584 | make_boarding_pass_pdf |\n"
    hits = extract_symbol_hits_from_text(text, tmp_path)
    assert hits == [("app.py", 1584, "make_boarding_pass_pdf")]


def test_format_diagnose_handoff_block(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x\n")
    block = format_diagnose_handoff_block(
        {
            "st-1": "Found make_boarding_pass_pdf at app.py:1584\n",
        },
        tmp_path,
    )
    assert "Structured locate results" in block
    assert "app.py:1584" in block


def test_propagate_diagnose_paths_from_line_refs(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x\n")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-1", description="diag", kind=SubTaskKind.DIAGNOSE),
            SubTaskNode(
                id="st-2",
                description="edit",
                kind=SubTaskKind.EDIT,
                depends_on=["st-1"],
            ),
        ],
    )
    propagate_diagnose_paths(
        tree,
        "st-1",
        "Target SQL at app.py:1600-1620 (not the PDF helper).",
        tmp_path,
    )
    edit = tree.get("st-2")
    assert edit is not None
    assert "app.py" in edit.context_files


def test_rank_edit_relevant_paths_filters_supporting_frontend_evidence(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("def query_orders():\n    pass\n")
    app = tmp_path / "app.py"
    app.write_text("def make_boarding_pass_pdf(order):\n    pass\n")
    init_dir = tmp_path / "db" / "init"
    init_dir.mkdir(parents=True)
    (init_dir / "init.sql").write_text("CREATE VIEW view_ticket_report_detail AS SELECT 1;\n")

    summary = """
接口端点: main.py:1625 — @app.get("/api/orders/query")
处理函数: main.py:1626 — def query_orders(...)
前端调用: app.py:1867 — api_get("/api/orders/query", params=params)
PDF生成: app.py:1584 — def make_boarding_pass_pdf(order)
目标视图: db/init/init.sql:394 — CREATE VIEW view_ticket_report_detail AS
"""

    ranked = rank_edit_relevant_paths(
        summary,
        ["main.py", "app.py", "db/init/init.sql"],
        intent_text="将登机牌查询接口改为使用视图",
    )

    assert set(ranked) == {"main.py", "db/init/init.sql"}
    assert "app.py" not in ranked
