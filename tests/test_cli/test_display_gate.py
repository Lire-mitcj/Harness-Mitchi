from __future__ import annotations

from src.agent.events import AgentEvent, EventType, final_answer_event
from src.cli.display_gate import OrchestratorDisplayGate, summarize_text


class _FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def render_event(self, event: AgentEvent) -> None:
        self.calls.append(("event", event.type))

    def render_status(self, message: str) -> None:
        self.calls.append(("status", message))

    def render_subtask_milestone(self, sid: str, milestone: str, detail: str, **kw: object) -> None:
        self.calls.append(("milestone", (sid, milestone, detail)))

    def render_tool_result(self, name: str, result: str, **kw: object) -> None:
        self.calls.append(("tool_result", (name, result)))

    def render_final_answer(self, content: str) -> None:
        self.calls.append(("final_answer", content))

    def render_plain_report(self, content: str) -> None:
        self.calls.append(("plain_report", content))


def test_summarize_text_truncates() -> None:
    assert summarize_text("a " * 100, max_len=20).endswith("…")


def test_intermediate_final_answer_buffered_until_done() -> None:
    gate = OrchestratorDisplayGate(enabled=True)
    r = _FakeRenderer()
    gate.render(r, final_answer_event("long diagnose findings " * 5, intermediate=True, subtask_id="st-1"))
    assert r.calls == []
    gate.render(
        r,
        AgentEvent(
            type=EventType.STATUS,
            content="done",
            data={"milestone": "subtask_done", "subtask_id": "st-1", "kind": "diagnose"},
        ),
    )
    assert any(c[0] == "milestone" and c[1][0] == "st-1" for c in r.calls)
    assert any(c[0] == "plain_report" for c in r.calls)
    assert not any(c[0] == "final_answer" for c in r.calls)


def test_terminal_final_answer_rendered_full() -> None:
    gate = OrchestratorDisplayGate(enabled=True)
    r = _FakeRenderer()
    gate.render(r, final_answer_event("Done (1/1 steps)", terminal=True))
    assert ("event", EventType.FINAL_ANSWER) in r.calls


def test_map_search_tool_call_shown() -> None:
    gate = OrchestratorDisplayGate(enabled=True)
    r = _FakeRenderer()
    gate.render(
        r,
        AgentEvent(
            type=EventType.TOOL_CALL,
            data={"tool": "map_search", "params": {"query": "view"}, "phase": "executor"},
        ),
    )
    assert ("event", EventType.TOOL_CALL) in r.calls


def test_executor_activity_tool_calls_shown() -> None:
    gate = OrchestratorDisplayGate(enabled=True)
    r = _FakeRenderer()
    gate.render(
        r,
        AgentEvent(
            type=EventType.TOOL_CALL,
            data={"tool": "grep_search", "params": {}, "phase": "executor"},
        ),
    )
    assert ("event", EventType.TOOL_CALL) in r.calls


def test_llm_loading_spinner_status_is_not_swallowed() -> None:
    gate = OrchestratorDisplayGate(enabled=True)
    r = _FakeRenderer()
    gate.render(
        r,
        AgentEvent(
            type=EventType.STATUS,
            content="Planner",
            data={"spinner_only": True, "llm_loading": True, "phase": "planner"},
        ),
    )

    assert ("event", EventType.STATUS) in r.calls
