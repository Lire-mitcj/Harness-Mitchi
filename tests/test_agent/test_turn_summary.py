from __future__ import annotations

from types import SimpleNamespace

from src.agent.manifest import StepManifest, Sufficiency
from src.agent.run_state import RunState, RunPhase, start_run
from src.agent.turn_summary import build_turn_summary
from src.agent.types import Message, ToolCall


def _folded_messages_with_grep() -> list[Message]:
    messages: list[Message] = []
    for index in range(5):
        call = ToolCall(id=str(index), name="grep_search", arguments={"pattern": f"auth{index}"})
        messages.extend(
            [
                Message(
                    role="assistant",
                    content="Let me examine the current state more thoroughly.",
                    tool_calls=[call],
                ),
                Message(role="tool", content="[FILE FACT STORED]"),
            ]
        )
    return messages


def test_turn_summary_decision_uses_tool_calls_not_narration() -> None:
    summary = build_turn_summary(
        _folded_messages_with_grep(),
        run_state=start_run("task", edit_mode=True),
    )

    assert "Let me examine" not in summary
    assert "决策：grep_search('auth4' @ .)" in summary
    assert "STEP EVIDENCE" in summary


def test_turn_summary_includes_manifest_snapshot() -> None:
    run_state = RunState(
        phase=RunPhase.ACTING,
        task_mode="edit",
        step=4,
        max_steps=20,
        evidence=start_run("task", edit_mode=True).evidence,
        manifest=StepManifest(sufficiency=Sufficiency.INSUFFICIENT),
        retrieval_no_gain_rounds=2,
        grep_suggested_views=(
            {"file": "main.py", "symbol": "create_app", "span": [1, 40]},
        ),
    )
    summary = build_turn_summary(
        _folded_messages_with_grep(),
        run_state=run_state,
        tools_available=frozenset({"view_symbol_code"}),
    )

    assert "状态快照（折叠时）" in summary
    assert "edit_ready=no" in summary
    assert "no_gain=2" in summary
    assert "tools=view_symbol_code" in summary
    assert "suggested_views=main.py::create_app" in summary


def test_turn_summary_records_duplicate_block_with_reuse_hint() -> None:
    view_call = ToolCall(
        id="v1",
        name="view_symbol_code",
        arguments={"target_file": "./main.py", "symbol": "build_router"},
    )
    duplicate_body = (
        "[RETRIEVAL DUPLICATE — NO NEW EVIDENCE]\n"
        "Error: BLOCK: Symbol 'build_router' is already present in "
        "CURRENT_CONTEXT (./list.py:16-358). Reuse it; do not re-fetch."
    )
    messages = [
        Message(
            role="assistant",
            content="Let me load main.py to see how build_router is used.",
            tool_calls=[view_call],
        ),
        Message(role="tool", content=duplicate_body),
    ]

    summary = build_turn_summary(
        messages,
        run_state=start_run("task", edit_mode=True),
    )

    assert "Let me load main.py" not in summary
    assert "决策：view_symbol_code(./main.py::build_router)" in summary
    assert "检索结果：" in summary
    assert "BLOCK (reuse ./list.py:16-358)" in summary


def test_turn_summary_prefers_last_tool_action_in_fold_window() -> None:
    messages = [
        Message(
            role="assistant",
            content="First batch",
            tool_calls=[
                ToolCall(id="1", name="grep_search", arguments={"pattern": "router"}),
            ],
        ),
        Message(role="tool", content="matches"),
        Message(
            role="assistant",
            content="Now edit list.py",
            tool_calls=[
                ToolCall(
                    id="2",
                    name="decision_edit",
                    arguments={"target_file": "list.py", "intent": "add handler"},
                ),
            ],
        ),
        Message(role="tool", content="patch ok"),
    ]

    summary = build_turn_summary(
        messages,
        run_state=start_run("task", edit_mode=True),
    )

    assert "决策：decision_edit(list.py: add handler)" in summary
    assert "Now edit list.py" not in summary
