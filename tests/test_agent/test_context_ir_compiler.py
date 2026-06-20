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
