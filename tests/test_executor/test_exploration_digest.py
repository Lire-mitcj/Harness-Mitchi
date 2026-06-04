from __future__ import annotations

from src.agent.types import ToolCall, assistant_message, tool_message
from src.executor.exploration_digest import build_exploration_digest


def test_digest_records_read_line_ranges() -> None:
    messages = [
        assistant_message(
            "",
            [
                ToolCall(
                    id="t1",
                    name="read_file",
                    arguments={
                        "path": "/mnt/d/proj/main.py",
                        "start_line": 460,
                        "end_line": 510,
                    },
                ),
            ],
        ),
        tool_message("t1", "def build_order_detail_sql():\n    return text('SELECT ...')\n"),
    ]
    digest = build_exploration_digest(messages)
    assert "main.py:460-510" in digest
    assert "build_order_detail_sql" in digest


def test_digest_lists_grep_queries() -> None:
    messages = [
        assistant_message(
            "",
            [
                ToolCall(
                    id="g1",
                    name="grep_search",
                    arguments={"pattern": "boarding_pass", "path": "main.py"},
                ),
            ],
        ),
        tool_message("g1", "main.py:360:    text('SELECT ... boarding_pass ...')"),
    ]
    digest = build_exploration_digest(messages)
    assert "boarding_pass" in digest
    assert "main.py:360" in digest
