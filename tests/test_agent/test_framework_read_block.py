from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.events import EventType
from src.agent.loop import AgentLoop
from src.agent.types import LLMResponse, TokenUsage, ToolCall, ToolResult, user_message
from src.config.permissions import PermissionManager
from src.config.settings import MitKIISettings
from src.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_loop_blocks_framework_read_without_user_path(tmp_path: Any) -> None:
    """read_file on src/harness/** is denied unless user named that path."""
    harness_path = tmp_path / "src" / "harness" / "engine.py"
    harness_path.parent.mkdir(parents=True)
    harness_path.write_text("# harness\n")

    tool_response = LLMResponse(
        content="",
        tool_calls=[
            ToolCall(id="tc_1", name="read_file", arguments={"path": "src/harness/engine.py"})
        ],
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
    mock_tool.risk_level = MagicMock()
    mock_tool.to_schema.return_value = {"type": "function", "function": {"name": "read_file"}}
    registry._tools["read_file"] = mock_tool
    registry.call = AsyncMock(return_value=ToolResult(success=True, output="should not run"))

    harness = MagicMock()
    harness.project_root = tmp_path
    harness.before_llm_call = AsyncMock(side_effect=lambda msgs: msgs)
    harness.after_llm_call = AsyncMock()
    harness.save_checkpoint = AsyncMock(return_value=None)
    harness.probe = MagicMock()
    harness.probe.metrics = MagicMock()
    harness.probe.metrics.record = MagicMock(return_value=MagicMock(cost=0.0))
    harness.scorer = MagicMock()
    harness.scorer.score = AsyncMock(return_value=None)

    context_builder = MagicMock()
    context_builder.build = AsyncMock(return_value=[user_message("Create gate_l0_fail.py")])

    loop = AgentLoop(
        llm=llm,
        tools=registry,
        harness=harness,
        context=context_builder,
        permissions=PermissionManager(),
        settings=MitKIISettings(data_dir=tmp_path / ".mitkii", max_turns=3),
    )

    tool_results: list[str] = []
    async for event in loop.run("Create gate_l0_fail.py"):
        if event.type == EventType.TOOL_RESULT:
            tool_results.append(str(event.content))

    registry.call.assert_not_awaited()
    assert any("Read blocked" in msg for msg in tool_results)
