from __future__ import annotations

from pathlib import Path

from src.agent.explore_guard import parse_read_path_with_lines
from src.executor.preload_read import format_cached_read_from_policy
from dataclasses import replace

from src.harness.gates.types import TruncationPolicy


def test_parse_read_path_with_lines() -> None:
    path, start, end = parse_read_path_with_lines("db/init/init.sql:390-420")
    assert path == "db/init/init.sql"
    assert start == 390
    assert end == 420


def test_format_cached_read_from_policy(tmp_path: Path) -> None:
    sql = tmp_path / "db" / "init" / "init.sql"
    sql.parent.mkdir(parents=True)
    lines = [f"line {i}\n" for i in range(1, 501)]
    sql.write_text("".join(lines))
    policy = replace(
        TruncationPolicy.green(),
        line_slices={"db/init/init.sql": (390, 420)},
    )
    text = format_cached_read_from_policy(
        tmp_path,
        "db/init/init.sql",
        start_line=390,
        end_line=420,
        policy=policy,
    )
    assert text is not None
    assert "Already in context" in text
    assert "line 390" in text
