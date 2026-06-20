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


class _FakeOrchestrator(_FakeLoop):
    pass


@pytest.mark.asyncio
async def test_adapter_routes_default_to_cursor_and_plan_to_orchestrator(monkeypatch) -> None:
    monkeypatch.setattr(chat, "CursorLoop", _FakeCursor)
    monkeypatch.setattr(chat, "OrchestratorLoop", _FakeOrchestrator)
    session = SimpleNamespace(
        llm=MagicMock(),
        cursor_inter_llm=MagicMock(),
        cursor_decision_llm=MagicMock(),
        tools=MagicMock(),
        harness=MagicMock(),
        context_builder=MagicMock(),
        permissions=MagicMock(),
        settings=MagicMock(),
    )
    adapter = chat.AgentLoopAdapter(session)

    _ = [event async for event in adapter.run_turn_stream("fix validator")]
    assert isinstance(adapter._current_loop, _FakeCursor)
    assert adapter._cursor_loop.inputs == ["fix validator"]

    _ = [event async for event in adapter.run_turn_stream("/plan refactor validator")]
    assert isinstance(adapter._current_loop, _FakeOrchestrator)
    assert adapter._orchestrator_loop.inputs == ["/plan refactor validator"]
