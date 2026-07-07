from __future__ import annotations

import asyncio

import pytest

from src.hooks.before_tool import inspect_tool_request_async


@pytest.mark.asyncio
async def test_blocks_repeat_empty_grep_with_same_fingerprint() -> None:
    history = [
        {
            "file": ".",
            "pattern": "missing_symbol",
            "patterns": ["missing_symbol"],
            "mode": "default",
            "fingerprint": ".::::default::missing_symbol",
            "empty": True,
        }
    ]
    err = await inspect_tool_request_async(
        "grep_search",
        {"pattern": "missing_symbol", "path": ".", "mode": "default"},
        allowed_tools={"grep_search"},
        search_history=history,
    )
    assert err is not None
    assert err.startswith("BLOCK:")
    assert "empty grep_search" in err
