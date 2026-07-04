from __future__ import annotations

from src.hooks.preflight.static_constraints import inspect_static_constraints


def test_grep_without_pattern_is_blocked_with_hint() -> None:
    err = inspect_static_constraints(
        "grep_search",
        {"include": "*.sql", "mode": "default"},
        allowed_tools={"grep_search"},
    )
    assert err is not None
    assert "pattern" in err.casefold()
    assert "discovery_hints" in err or "include/mode" in err


def test_grep_with_patterns_passes_preflight() -> None:
    err = inspect_static_constraints(
        "grep_search",
        {"patterns": ["CREATE TABLE", "ticket_order"], "include": "*.sql"},
        allowed_tools={"grep_search"},
    )
    assert err is None
