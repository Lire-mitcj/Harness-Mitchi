from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.events import AgentEvent, EventType
from src.agent.loop import AgentLoop
from src.agent.types import LLMResponse, RiskLevel, TokenUsage, ToolCall
from src.config.permissions import PermissionConfig, PermissionManager
from src.config.settings import MitKIISettings
from src.tools.registry import ToolRegistry


def _make_settings(tmp_path) -> MitKIISettings:
    return MitKIISettings(data_dir=tmp_path / ".mitkii", max_turns=3)


def _make_harness() -> MagicMock:
    harness = MagicMock()
    harness.before_llm_call = AsyncMock(side_effect=lambda msgs: msgs)
    harness.after_llm_call = AsyncMock()
    harness.save_checkpoint = AsyncMock(return_value=None)
    harness.probe = MagicMock()
    harness.probe.metrics = MagicMock()
    record = MagicMock()
    record.cost = 0.001
    harness.probe.metrics.record = MagicMock(return_value=record)
    harness.scorer = MagicMock()
    harness.scorer.score = AsyncMock(return_value=None)
    return harness


@pytest.mark.asyncio
async def test_loop_yields_final_answer(tmp_path: Any) -> None:
    """AgentLoop should yield STREAM_START, THINKING, FINAL_ANSWER, STREAM_END for a simple message."""
    final_response = LLMResponse(
        content="The answer is 42.",
        tool_calls=None,
        usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
        model="test",
    )

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "content", "content": "The answer is 42."}
        yield {"type": "response", "response": final_response}

    llm = MagicMock()
    llm.chat_stream = MagicMock(side_effect=fake_stream)

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=[])

    loop = AgentLoop(
        llm=llm,
        tools=ToolRegistry(),
        harness=_make_harness(),
        context=context_builder,
        permissions=PermissionManager(),
        settings=_make_settings(tmp_path),
    )

    events: list[AgentEvent] = []
    async for event in loop.run("What is the meaning of life?"):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.STREAM_START in event_types
    assert EventType.THINKING in event_types
    assert EventType.FINAL_ANSWER in event_types
    assert EventType.STREAM_END in event_types

    final = next(e for e in events if e.type == EventType.FINAL_ANSWER)
    assert final.content == "The answer is 42."


@pytest.mark.asyncio
async def test_loop_handles_tool_calls(tmp_path: Any) -> None:
    """AgentLoop should process tool calls and yield tool events."""
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "/x"})],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        model="test",
    )
    final_response = LLMResponse(
        content="File content retrieved.",
        tool_calls=None,
        usage=TokenUsage(prompt_tokens=80, completion_tokens=15, total_tokens=95),
        model="test",
    )

    call_count = 0

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            yield {"type": "response", "response": tool_response}
        else:
            yield {"type": "content", "content": "File content retrieved."}
            yield {"type": "response", "response": final_response}

    llm = MagicMock()
    llm.chat_stream = MagicMock(side_effect=fake_stream)

    registry = ToolRegistry()
    # Register a mock read_file tool
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.risk_level = RiskLevel.SAFE
    mock_tool.to_schema.return_value = {"type": "function", "function": {"name": "read_file"}}
    registry._tools["read_file"] = mock_tool

    from src.agent.types import ToolResult

    registry.call = AsyncMock(return_value=ToolResult(success=True, output="file contents here"))

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=[])

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        harness=_make_harness(),
        context=context_builder,
        permissions=PermissionManager(),
        settings=_make_settings(tmp_path),
    )

    events: list[AgentEvent] = []
    async for event in loop.run("Read the file"):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.TOOL_CALL in event_types
    assert EventType.TOOL_RESULT in event_types
    assert EventType.FINAL_ANSWER in event_types


@pytest.mark.asyncio
async def test_loop_respects_max_turns(tmp_path: Any) -> None:
    """AgentLoop should emit an error when max_turns is exhausted."""
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "/x"})],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        model="test",
    )

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "response", "response": tool_response}

    llm = MagicMock()
    llm.chat_stream = MagicMock(side_effect=fake_stream)

    registry = ToolRegistry()
    from src.agent.types import ToolResult

    registry.call = AsyncMock(return_value=ToolResult(success=True, output="ok"))
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.risk_level = RiskLevel.SAFE
    mock_tool.to_schema.return_value = {"type": "function", "function": {"name": "read_file"}}
    registry._tools["read_file"] = mock_tool

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=[])

    settings = _make_settings(tmp_path)
    settings.max_turns = 2

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        harness=_make_harness(),
        context=context_builder,
        permissions=PermissionManager(),
        settings=settings,
    )

    events: list[AgentEvent] = []
    async for event in loop.run("Loop forever"):
        events.append(event)

    event_types = [e.type for e in events]
    assert EventType.ERROR in event_types
    error = next(e for e in events if e.type == EventType.ERROR)
    assert "maximum turns" in error.content.lower()
