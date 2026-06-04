from __future__ import annotations

from src.agent.framework_guard import (
    blocked_framework_reads,
    extract_framework_path_mentions,
    is_blocked_framework_path,
    user_allows_framework_path,
)


def test_is_blocked_framework_path() -> None:
    assert is_blocked_framework_path("src/harness/engine.py")
    assert is_blocked_framework_path("src/agent/loop.py")
    assert is_blocked_framework_path("src/cli/repl.py")
    assert is_blocked_framework_path("prompts/system_prompt.md")
    assert not is_blocked_framework_path("gate_l0_fail.py")
    assert not is_blocked_framework_path("src/tools/file/read.py")


def test_user_allows_when_path_explicitly_named() -> None:
    msg = "Fix the approval deadlock in src/agent/loop.py"
    assert user_allows_framework_path(msg, "src/agent/loop.py")
    assert not user_allows_framework_path(msg, "src/harness/engine.py")


def test_user_allows_parent_prefix() -> None:
    msg = "Refactor src/harness/scorer for better L0 handling"
    assert user_allows_framework_path(msg, "src/harness/scorer/code_quality.py")


def test_blocked_for_gate_demo_without_framework_path() -> None:
    msg = "Create gate_l0_fail.py with a syntax error to trigger L0 gate failure"
    blocked = blocked_framework_reads(
        msg,
        ["src/harness/engine.py", "src/agent/loop.py"],
    )
    assert blocked == ["src/harness/engine.py", "src/agent/loop.py"]


def test_extract_framework_path_mentions() -> None:
    msg = 'Please update src/agent/loop.py and prompts/system_prompt.md'
    mentions = extract_framework_path_mentions(msg)
    assert "src/agent/loop.py" in mentions
    assert "prompts/system_prompt.md" in mentions
