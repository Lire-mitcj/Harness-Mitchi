from __future__ import annotations

from src.agent.types import assistant_message, tool_message, user_message
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
    assert "Session exploration summary" in combined
    assert "app.py" in combined
    assert "folded" in combined.lower()
