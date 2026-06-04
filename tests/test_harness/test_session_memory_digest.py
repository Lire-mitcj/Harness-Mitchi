from __future__ import annotations

from src.agent.types import assistant_message, tool_message, user_message
from src.harness.subtask.session_memory import ExploreSessionMemory


def test_merge_digest_from_messages_incremental() -> None:
    memory = ExploreSessionMemory.create()
    batch1 = [
        user_message("go"),
        tool_message("t1", "app.py:10:boarding pass foo"),
    ]
    memory.merge_digest_from_messages(batch1)
    first = memory.running_digest

    batch2 = batch1 + [tool_message("t2", "app.py:20:boarding pass bar")]
    memory.merge_digest_from_messages(batch2)
    second = memory.running_digest

    assert "app.py:10" in first
    assert "app.py:20" in second
    assert memory._digest_scanned_len == len(batch2)

    memory.merge_digest_from_messages(batch2)
    assert memory.running_digest == second


def test_reset_digest_scan_after_compact() -> None:
    memory = ExploreSessionMemory.create()
    memory.merge_digest_from_messages([tool_message("t1", "x" * 100)])
    memory.reset_digest_scan(3)
    assert memory._digest_scanned_len == 3
    memory.merge_digest_from_messages([user_message("folded")])
    assert memory._digest_scanned_len == 1
