from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.events import AgentEvent, EventType
from src.agent.loop import AgentLoop
from src.agent.types import LLMResponse, RiskLevel, TokenUsage, ToolCall
from src.config.permissions import PermissionManager
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
async def test_loop_tool_execution_metrics(tmp_path: Any) -> None:
    """AgentLoop should yield status updates and record phase metrics for tool execution."""
    tool_response = LLMResponse(
        content="",
        tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "/x"})],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
        model="test",
    )
    final_response = LLMResponse(
        content="Done.",
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
            yield {"type": "content", "content": "Done."}
            yield {"type": "response", "response": final_response}

    llm = MagicMock()
    llm.chat_stream = MagicMock(side_effect=fake_stream)

    registry = ToolRegistry()
    mock_tool = MagicMock()
    mock_tool.name = "read_file"
    mock_tool.risk_level = RiskLevel.SAFE
    mock_tool.to_schema.return_value = {"type": "function", "function": {"name": "read_file"}}
    registry._tools["read_file"] = mock_tool

    from src.agent.types import ToolResult
    registry.call = AsyncMock(return_value=ToolResult(success=True, output="file contents here"))

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=[])

    harness = _make_harness()
    from src.harness.gates.phase_metrics import PhaseMetrics
    harness.phase_metrics = PhaseMetrics()

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        harness=harness,
        context=context_builder,
        permissions=PermissionManager(),
        settings=_make_settings(tmp_path),
    )

    events: list[AgentEvent] = []
    async for event in loop.run("Read the file"):
        events.append(event)

    # Check status events for tool execution
    status_events = [e for e in events if e.type == EventType.STATUS]
    assert any("正在读取文件: /x" in str(e.content) for e in status_events)

    # Check that phase metrics tracked both core LLM and the tool separately
    records = harness.phase_metrics.records
    phases = [r.phase for r in records]
    assert "core_llm" in phases
    assert "tool_read_file" in phases
