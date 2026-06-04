from __future__ import annotations

from src.agent.types import ToolCall, assistant_message, tool_message, user_message
from src.executor.context_compress import (
    merge_exploration_digests,
    rebuild_compacted_executor_messages,
)


def test_merge_digest_keeps_assistant_notes() -> None:
    messages = [
        assistant_message("Found boarding_pass query in main.py around line 360."),
        tool_message("t1", "===== main.py =====\nSELECT * FROM boarding_pass\n"),
    ]
    digest = merge_exploration_digests(None, messages)
    assert "main.py" in digest
    assert "boarding_pass" in digest.lower() or "360" in digest


def test_rebuild_compacted_includes_digest_block() -> None:
    base = [user_message("Execute subtask now.")]
    out = rebuild_compacted_executor_messages(
        base_messages=base,
        digest="Files already read: app.py",
        compact_reason="context size",
    )
    combined = "\n".join(m.content or "" for m in out)
    assert "SESSION_DIGEST_JSON" in combined
    assert "app.py" in combined
    assert '"event": "context_folded"' in combined


def test_merge_digest_extracts_context_search_snippet_ranges() -> None:
    messages = [
        assistant_message(
            "",
            [
                ToolCall(
                    id="tc1",
                    name="context_search",
                    arguments={
                        "query": "views used in project",
                        "need": "file:line and symbol evidence",
                    },
                )
            ],
        ),
        tool_message(
            "tc1",
            '<context_snippets>\n'
            '<snippet path="src/db/views.sql" lines="12-18">\n'
            "12: CREATE VIEW active_orders AS\n"
            "13: SELECT * FROM orders\n"
            "</snippet>\n"
            "</context_snippets>",
        ),
    ]

    digest = merge_exploration_digests(None, messages)

    assert "views used in project" in digest
    assert "src/db/views.sql:12-18" in digest
    assert "CREATE VIEW" in digest
