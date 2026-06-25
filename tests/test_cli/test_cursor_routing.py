from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.agent.events import AgentEvent, EventType
from src.cli.commands import chat


class _FakeLoop:
    def __init__(self, **kwargs: object) -> None:
        self.state = SimpleNamespace(name=self.__class__.__name__)
        self.inputs: list[str] = []

    async def run(self, user_input: str):
        self.inputs.append(user_input)
        yield AgentEvent(type=EventType.STREAM_START)

    async def resolve_approval(self, action: str, approved: bool) -> None:
        return None

    async def list_checkpoints(self) -> list[dict[str, object]]:
        return []

    def get_probe_metrics(self) -> dict[str, object]:
        return {}

    async def run_score_now(self) -> dict[str, object] | None:
        return None


class _FakeCursor(_FakeLoop):
    pass


class _FakeAssembled(_FakeLoop):
    pass


@pytest.mark.asyncio
async def test_adapter_routes_by_mode(monkeypatch) -> None:
    monkeypatch.setattr(chat, "CursorLoop", _FakeCursor)
    
    # Mock StateAssembledLoop within src.agent.state_assembled_loop
    import sys
    mock_module = MagicMock()
    mock_module.StateAssembledLoop = _FakeAssembled
    sys.modules["src.agent.state_assembled_loop"] = mock_module

    # Test cursor mode
    session_cursor = SimpleNamespace(
        llm=MagicMock(),
        cursor_inter_llm=MagicMock(),
        cursor_decision_llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(),
        context_builder=MagicMock(),
        permissions=MagicMock(),
        settings=SimpleNamespace(mitkii_mode="cursor"),
    )
    adapter_cursor = chat.AgentLoopAdapter(session_cursor)
    _ = [event async for event in adapter_cursor.run_turn_stream("fix validator")]
    assert isinstance(adapter_cursor._current_loop, _FakeCursor)

    # Test assembled mode
    session_assembled = SimpleNamespace(
        llm=MagicMock(),
        cursor_inter_llm=MagicMock(),
        cursor_decision_llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(),
        context_builder=MagicMock(),
        permissions=MagicMock(),
        settings=SimpleNamespace(mitkii_mode="assembled"),
    )
    adapter_assembled = chat.AgentLoopAdapter(session_assembled)
    _ = [event async for event in adapter_assembled.run_turn_stream("fix validator")]
    assert isinstance(adapter_assembled._current_loop, _FakeAssembled)
