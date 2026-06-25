from __future__ import annotations

from src.agent.explore_guard import parse_read_path_with_lines

def test_parse_read_path_with_lines() -> None:
    path, start, end = parse_read_path_with_lines("db/init/init.sql:390-420")
    assert path == "db/init/init.sql"
    assert start == 390
    assert end == 420
