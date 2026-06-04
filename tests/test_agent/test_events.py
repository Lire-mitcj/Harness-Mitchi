from __future__ import annotations

from src.agent.events import (
    AgentEvent,
    EventType,
    cost_event,
    error_event,
    final_answer_event,
    thinking_event,
    tool_call_event,
    tool_result_event,
)


class TestAgentEvents:
    def test_thinking_event(self) -> None:
        event = thinking_event("Let me look at the code...")
        assert event.type == EventType.THINKING
        assert event.content == "Let me look at the code..."
        assert event.id

    def test_tool_call_event(self) -> None:
        event = tool_call_event("read_file", {"path": "src/main.py"})
        assert event.type == EventType.TOOL_CALL
        assert event.data is not None
        assert event.data["tool"] == "read_file"

    def test_tool_result_event(self) -> None:
        event = tool_result_event("read_file", "file content", success=True)
        assert event.type == EventType.TOOL_RESULT
        assert event.data is not None
        assert event.data["success"] is True

    def test_final_answer_event(self) -> None:
        event = final_answer_event("Done! I've made the changes.")
        assert event.type == EventType.FINAL_ANSWER
        assert event.content == "Done! I've made the changes."

    def test_error_event(self) -> None:
        event = error_event("Something went wrong")
        assert event.type == EventType.ERROR
        assert event.content == "Something went wrong"

    def test_cost_event(self) -> None:
        event = cost_event(1000, 500, 0.005)
        assert event.type == EventType.COST_UPDATE
        assert event.data is not None
        assert event.data["prompt_tokens"] == 1000

    def test_event_has_unique_id(self) -> None:
        e1 = thinking_event("a")
        e2 = thinking_event("b")
        assert e1.id != e2.id

    def test_event_has_timestamp(self) -> None:
        event = thinking_event("test")
        assert event.timestamp > 0
