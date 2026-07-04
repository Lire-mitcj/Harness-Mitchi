from __future__ import annotations

from src.tools.grep_match_symbols import (
    extract_symbol_from_match_line,
    suggested_views_from_matches,
)


def test_extract_sql_create_table_symbol() -> None:
    line = "CREATE TABLE IF NOT EXISTS `ticket_order` ("
    assert extract_symbol_from_match_line(line) == "ticket_order"


def test_extract_python_def_symbol() -> None:
    line = "async def build_router(engine: Engine) -> APIRouter:"
    assert extract_symbol_from_match_line(line) == "build_router"


def test_suggested_views_deduplicates_file_symbol() -> None:
    matches = [
        {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [10, 10]},
        {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [11, 11]},
        {"file": "list.py", "symbol": "build_router", "span": [16, 16]},
    ]
    views = suggested_views_from_matches(matches)
    assert len(views) == 2
    assert views[0]["symbol"] == "ticket_order"
