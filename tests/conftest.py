from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.types import LLMResponse, Message, TokenUsage, ToolCall, ToolResult
from src.config.permissions import PermissionConfig, PermissionManager
from src.config.settings import MitKIISettings
from src.tools.base import Tool
from src.agent.types import RiskLevel


@pytest.fixture
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal project directory with a few sample files."""
    (tmp_path / "main.py").write_text("def main():\n    print('hello')\n")
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "helper.py").write_text("class Helper:\n    pass\n")
    (tmp_path / ".gitignore").write_text("__pycache__\n*.pyc\n")
    return tmp_path


@pytest.fixture
def settings(tmp_path: Path) -> MitKIISettings:
    """Settings instance with temp data directory."""
    return MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        max_turns=5,
        max_retries=2,
        auto_approve_edits=True,
    )


@pytest.fixture
def permission_manager() -> PermissionManager:
    return PermissionManager(PermissionConfig())


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """A simple LLM response with no tool calls."""
    return LLMResponse(
        content="Here is my answer.",
        tool_calls=None,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        model="test-model",
    )


@pytest.fixture
def mock_llm_tool_response() -> LLMResponse:
    """An LLM response that requests a tool call."""
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(
                id="call_001",
                name="read_file",
                arguments={"path": "/tmp/test.txt"},
            ),
        ],
        usage=TokenUsage(prompt_tokens=100, completion_tokens=30, total_tokens=130),
        model="test-model",
    )


@pytest.fixture
def mock_llm_client(mock_llm_response: LLMResponse) -> MagicMock:
    """A mock LLM client that returns a canned response."""
    client = MagicMock()
    client.chat = AsyncMock(return_value=mock_llm_response)

    async def fake_stream(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "content", "content": "Here is my answer."}
        yield {"type": "response", "response": mock_llm_response}

    client.chat_stream = MagicMock(side_effect=fake_stream)
    return client


@pytest.fixture
def mock_context_builder() -> MagicMock:
    """A mock context builder that returns a minimal message list."""
    builder = MagicMock()

    async def build(user_message: str) -> list[Message]:
        return [Message(role="system", content="You are a test assistant.")]

    builder.build = AsyncMock(side_effect=build)
    return builder


class FakeTool(Tool):
    """A simple tool for testing that always succeeds."""

    name = "fake_tool"
    description = "A fake tool for testing"
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "Test input"},
        },
        "required": ["input"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        return ToolResult(success=True, output=f"Processed: {params.get('input', '')}")


@pytest.fixture
def fake_tool() -> FakeTool:
    return FakeTool()
