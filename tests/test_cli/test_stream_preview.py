from __future__ import annotations

from src.cli.stream_preview import (
    executor_reasoning_preview,
    executor_stream_hint,
    planner_status_hint,
    should_stream_executor_answer,
)


def test_planner_status_hint_kinds() -> None:
    partial = '{"root_task":"views","nodes":[{"kind":"diagnose"},{"kind":"verify"}'
    hint = planner_status_hint(partial)
    assert hint is not None
    assert "diagnose" in hint
    assert "verify" in hint


def test_planner_status_hint_no_raw_json() -> None:
    assert planner_status_hint('{"{"root_task') == "writing TaskTree JSON…"


def test_executor_reasoning_preview_skips_markdown() -> None:
    text = "## Title\n\nLet me grep for views.\n"
    assert executor_reasoning_preview(text) == "Let me grep for views."


def test_executor_reasoning_preview_skips_json() -> None:
    text = '{"kind":"diagnose"}\n'
    assert executor_reasoning_preview(text) == ""


def test_should_not_stream_long_markdown_answer() -> None:
    text = "## 诊断结果\n\n| a | b |\n|---|---|\n| x | y |"
    assert should_stream_executor_answer(text, has_tool_calls=False) is False


def test_executor_stream_hint_uses_heading() -> None:
    text = "## 诊断结果：登机牌相关视图搜索\n\n| 文件 | 行号 |\n"
    assert executor_stream_hint(text) == "诊断结果：登机牌相关视图搜索"
