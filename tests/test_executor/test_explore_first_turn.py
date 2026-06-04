from __future__ import annotations

from src.executor.policy import explore_first_turn_tools


def test_explore_first_turn_prefers_batch_tools() -> None:
    runtime = frozenset({"grep_search", "read_files", "read_file", "edit_file"})
    assert explore_first_turn_tools(runtime) == frozenset({"grep_search", "read_files"})


def test_explore_first_turn_falls_back_to_read_file() -> None:
    runtime = frozenset({"read_file", "edit_file"})
    assert explore_first_turn_tools(runtime) == frozenset({"read_file"})
