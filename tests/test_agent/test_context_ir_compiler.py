from pathlib import Path

from src.agent.cursor_context_pack_builder import CursorContextPackBuilder
from src.agent.cursor_contracts import RetrievalResult, RetrievalSymbol
from src.agent.cursor_fusion import CursorFusionEngine
from src.agent.cursor_query_bridge import QueryBridgeResult


def test_three_layer_ir_keeps_core_exact_and_filters_header(tmp_path: Path) -> None:
    target = tmp_path / "list.py"
    target.write_text(
        "import logging\n"
        "from typing import Optional\n"
        "from sqlalchemy import text\n\n"
        "def unrelated():\n"
        "    return logging.getLogger(__name__)\n\n"
        "def passenger_snapshot(keyword: Optional[str] = None):\n"
        "    sql = text('SELECT ticket_order FROM ticket_order WHERE p_id = :p_id')\n"
        "    return sql\n",
        encoding="utf-8",
    )
    schema = tmp_path / "schema.sql"
    schema.write_text(
        "CREATE VIEW ticket_order AS SELECT p_id FROM ticket_order;\n", encoding="utf-8",
    )
    retrieval = RetrievalResult(symbols=(
        RetrievalSymbol(
            "list.py",
            "passenger_snapshot",
            8,
            10,
            kind="function",
            tables_referenced=("ticket_order",),
        ),
        RetrievalSymbol(
            "schema.sql",
            "ticket_order",
            1,
            1,
            kind="ddl_table",
            tables_referenced=("ticket_order",),
        ),
    ))

    pack = CursorContextPackBuilder(tmp_path).build_context(
        retrieval,
        final_context=(
            "FOCUS:list.py:passenger_snapshot:8-10",
            "schema.sql:ticket_order:1-1",
        ),
    )

    window = pack.windows[0]
    assert window.start_line == 8
    assert window.end_line == 10
    assert "[LAYER_1_CORE_SYMBOL]" in window.content
    assert "8: def passenger_snapshot" in window.content
    assert "1: import logging" not in window.content
    assert "2: from typing import Optional" in window.content
    assert "3: from sqlalchemy import text" in window.content
    assert "[SOFT_DEPENDENCY]" in window.content
    skeleton = window.content.split("[LAYER_3_GLOBAL_SKELETON]", 1)[1]
    assert "def unrelated():" in skeleton
    assert "conn.execute" not in skeleton


def test_fusion_never_marks_ddl_as_focus() -> None:
    bridge = QueryBridgeResult(
        intent="modify", expanded_terms=[], keywords=[], symbols=[], file_hints=[],
    )
    result = RetrievalResult(symbols=(
        RetrievalSymbol("api.py", "passenger_snapshot", 10, 30, score=0.9, kind="function"),
        RetrievalSymbol("schema.sql", "passenger_order_view", 1, 8, score=0.95, kind="ddl_view"),
    ))

    fused = CursorFusionEngine.decide(result, bridge, max_files=2, max_symbols=2)

    assert fused.final_context[0].startswith("FOCUS:api.py:passenger_snapshot")
    assert all("FOCUS:schema.sql" not in item for item in fused.final_context)


def test_context_coalescing_with_collapsed_placeholder(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text(
        "import os\n"
        "import sys\n"
        "def func1():\n"
        "    return 1\n"
        "def func2():\n"
        "    return 2\n"
        "def func3():\n"
        "    return 3\n",
        encoding="utf-8",
    )
    retrieval = RetrievalResult(symbols=(
        RetrievalSymbol("app.py", "func1", 3, 4, kind="function"),
        RetrievalSymbol("app.py", "func3", 7, 8, kind="function"),
    ))

    builder = CursorContextPackBuilder(tmp_path)
    from src.agent.cursor_contracts import ContextPack, ContextWindow
    initial_pack = ContextPack(windows=(
        ContextWindow(file="app.py", start_line=3, end_line=4, content="func1 content", symbols=("func1",)),
        ContextWindow(file="app.py", start_line=7, end_line=8, content="func3 content", symbols=("func3",)),
    ))
    
    pack = builder.merge_interval_subgraph(initial_pack, retrieval)
    assert len(pack.windows) == 1
    window = pack.windows[0]
    assert window.start_line == 3
    assert window.end_line == 8
    
    assert "INTERVAL_CHUNK RANGE 3-4" in window.content
    assert "INTERVAL_CHUNK RANGE 7-8" in window.content
    assert "[PHYSICAL LINE INTERVAL 5-6 COLLAPSED DUE TO PRUNING POLICY]" in window.content


def test_context_quota_groupBy_file_preemption_fix(tmp_path: Path) -> None:
    app_path = tmp_path / "app.py"
    app_path.write_text("def a(): pass\ndef b(): pass\ndef c(): pass\n", encoding="utf-8")
    
    other_path = tmp_path / "other.py"
    other_path.write_text("def d(): pass\n", encoding="utf-8")
    
    retrieval = RetrievalResult(symbols=(
        RetrievalSymbol("app.py", "a", 1, 1, kind="function"),
        RetrievalSymbol("app.py", "b", 2, 2, kind="function"),
        RetrievalSymbol("app.py", "c", 3, 3, kind="function"),
        RetrievalSymbol("other.py", "d", 1, 1, kind="function"),
    ))
    
    builder = CursorContextPackBuilder(tmp_path, max_files=2)
    pack = builder.build_context(
        retrieval,
        final_context=(
            "app.py:a:1-1",
            "app.py:b:2-2",
            "app.py:c:3-3",
            "other.py:d:1-1",
        )
    )
    
    files = [w.file for w in pack.windows]
    assert "app.py" in files
    assert "other.py" in files
    assert len(files) == 2


def test_context_coalescing_keeps_ddl_evidence(tmp_path: Path) -> None:
    app_path = tmp_path / "app.py"
    app_path.write_text(
        "def main_func():\n"
        "    pass\n",
        encoding="utf-8"
    )
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(
        "CREATE TABLE ticket_order (p_id INT);\n",
        encoding="utf-8"
    )
    
    retrieval = RetrievalResult(symbols=(
        RetrievalSymbol("app.py", "main_func", 1, 2, kind="function", tables_referenced=("ticket_order",)),
        RetrievalSymbol("schema.sql", "ticket_order", 1, 1, kind="ddl_table", tables_referenced=("ticket_order",)),
    ))
    
    builder = CursorContextPackBuilder(tmp_path, max_files=2)
    pack = builder.build_context(
        retrieval,
        final_context=(
            "FOCUS:app.py:main_func:1-2",
            "schema.sql:ticket_order:1-1",
        )
    )
    
    assert len(pack.windows) == 1
    window = pack.windows[0]
    assert window.file == "app.py"
    assert "schema.sql" in window.content
    assert "[SOFT_DEPENDENCY]" in window.content
    assert "ticket_order" in window.content
