from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.contracts import ContextPack, Decision, ExecutionResult, ValidationResult
from src.agent.decision import CursorDecisionLLM, DecisionError
from src.tools.assembled.decision_edit import (
    DecisionEditTool,
    _build_state_text,
    build_patch_retry_state_text,
    decision_timeout_for_context,
    is_mechanical_patch_error,
    merge_context_spans,
)


def test_build_state_text_passes_core_plan_step() -> None:
    text = _build_state_text(
        "./list.py",
        "Add imports for SQLAlchemyError",
        ["passenger_snapshot", "archive_passenger"],
    )
    assert "Add imports for SQLAlchemyError" in text
    assert "passenger_snapshot" in text
    assert "Focus symbols for this plan step" in text


def test_decision_edit_only_mode_rejects_ask_clarify() -> None:
    with pytest.raises(DecisionError, match="edit-only mode forbids action 'ask_clarify'"):
        CursorDecisionLLM.parse(
            '{"action":"ask_clarify","answer":"","clarification":"need more context",'
            '"target_file":"","patch":"","suggested_completion":0}',
            ("db/init/init.sql",),
            edit_only=True,
        )


def test_decision_edit_only_prompt_mentions_edit_only_mode() -> None:
    messages = CursorDecisionLLM(MagicMock()).build_messages(
        state_text="apply change",
        context_pack=MagicMock(windows=(), candidate_files=()),
        hint=None,
        edit_only=True,
    )

    assert "EDIT_ONLY_MODE" in messages[0]["content"]
    assert '"action":"edit"' in messages[0]["content"]
    assert "Multiple non-overlapping" in messages[0]["content"]


def test_decision_prompt_allows_multiple_search_replace_blocks() -> None:
    from src.context.prompt_resources import load_internal_prompt

    prompt = load_internal_prompt("decision_prompt.md", fallback="")
    assert "one or more SEARCH/REPLACE blocks" in prompt
    assert "Multi-block example" in prompt
    assert "多处修改用多块" in prompt


def test_decision_edit_context_pack_includes_all_target_file_spans(tmp_path: Path) -> None:
    sql_path = tmp_path / "db" / "init" / "init.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"line {index}" for index in range(1, 401)]
    lines[142] = "CREATE TABLE order_timeline ("
    lines[327] = "CREATE TRIGGER tg_release_seat"
    sql_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(),
        decision_llm=MagicMock(),
        harness=MagicMock(project_root=tmp_path),
    )

    context_pack = tool._build_context_pack(
        "./db/init/init.sql",
        context_window=[
            {"file": "./db/init/init.sql", "span": [143, 155]},
            {"file": "./db/init/init.sql", "span": [311, 330]},
        ],
    )

    target_windows = [window for window in context_pack.windows if window.file.endswith("init.sql")]
    assert len(target_windows) == 2
    assert target_windows[0].role == "target"
    assert target_windows[1].role == "reference"
    assert "order_timeline" in target_windows[0].content
    assert "tg_release_seat" in target_windows[1].content


def test_merge_context_spans_merges_adjacent_same_file() -> None:
    merged = merge_context_spans(
        [
            ("list.py", 1, 15),
            ("list.py", 16, 17),
            ("list.py", 67, 70),
        ]
    )

    assert merged == [
        ("list.py", 1, 17),
        ("list.py", 67, 70),
    ]


def test_merge_context_spans_keeps_distant_spans_separate() -> None:
    merged = merge_context_spans(
        [
            ("db/init/init.sql", 143, 155),
            ("db/init/init.sql", 311, 330),
        ]
    )

    assert len(merged) == 2
    assert merged[0][1:] == (143, 155)
    assert merged[1][1:] == (311, 330)


def test_decision_edit_merges_adjacent_context_windows(tmp_path: Path) -> None:
    (tmp_path / "list.py").write_text(
        "\n".join(f"line {index}" for index in range(1, 21)) + "\n",
        encoding="utf-8",
    )
    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(),
        decision_llm=MagicMock(),
        harness=MagicMock(project_root=tmp_path),
    )

    context_pack = tool._build_context_pack(
        "list.py",
        context_window=[
            {"file": "list.py", "span": [1, 5]},
            {"file": "list.py", "span": [6, 10]},
            {"file": "list.py", "span": [18, 20]},
        ],
    )

    assert len(context_pack.windows) == 2
    assert context_pack.windows[0].start_line == 1
    assert context_pack.windows[0].end_line == 10
    assert context_pack.windows[1].start_line == 18
    assert context_pack.windows[1].end_line == 20


def test_build_evidence_flag_slim_when_context_window_provided() -> None:
    flag = DecisionEditTool._build_evidence_flag(
        "list.py",
        {
            "raw_evidence_store": [
                {
                    "file": "list.py",
                    "span": [1, 100],
                    "code": "x" * 5000,
                    "symbol": "build_router",
                }
            ]
        },
        ContextPack(windows=()),
        context_window=[
            {"file": "list.py", "span": [67, 70], "reason": "route decorator"},
        ],
    )

    assert flag["context_mode"] == "frozen_context_window"
    assert flag["context_window_spans"] == [
        {"file": "list.py", "span": [67, 70], "status": "frozen", "reason": "route decorator"}
    ]
    assert "target_code_anchors" not in flag
    assert "first_hop_functions" not in flag


def test_decision_timeout_scales_with_context_size() -> None:
    small = decision_timeout_for_context(120.0, context_chars=2000, span_count=2)
    large = decision_timeout_for_context(120.0, context_chars=11000, span_count=9)

    assert 120.0 <= small < large
    assert large > 200.0
    assert large <= 300.0


@pytest.mark.asyncio
async def test_generate_patch_stream_allows_long_generation_after_first_token(
    tmp_path: Path,
) -> None:
    """Streaming patch generation must not use connect budget as total wall-clock cap."""
    import asyncio
    from unittest.mock import AsyncMock
    from src.agent.types import LLMResponse

    target = tmp_path / "list.py"
    target.write_text("def route_a():\n    return 1\n", encoding="utf-8")

    edit_json = (
        '{"action":"edit","answer":"","clarification":"","target_file":"list.py",'
        '"patch":"<<<<<<< SEARCH\\ndef route_a():\\n=======\\ndef route_a():\\n    pass\\n>>>>>>> REPLACE",'
        '"suggested_completion":0}'
    )

    async def fake_chat_stream(messages, tools=None, timeout=None):
        yield ("{", None)
        await asyncio.sleep(0.15)
        yield ('"', None)
        await asyncio.sleep(0.15)
        yield ("", LLMResponse(content=edit_json, tool_calls=None, usage=None, model="test"))

    harness = MagicMock()
    harness.before_llm_call = AsyncMock(side_effect=lambda messages: messages)
    harness.after_llm_call = AsyncMock()

    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(cursor_decision_timeout=120.0),
        decision_llm=MagicMock(
            chat_stream=fake_chat_stream,
            stream_idle_timeout=60,
            model="test",
        ),
        harness=harness,
    )
    context_pack = tool._build_context_pack(
        "list.py",
        context_window=[{"file": "list.py", "span": [1, 2]}],
    )

    parsed, _response = await tool._generate_patch(
        target_file="list.py",
        state_text="apply change",
        context_pack=context_pack,
        evidence_flag={"can_edit": True},
        effective_timeout=0.05,
        started_at=asyncio.get_event_loop().time(),
    )

    assert parsed.action == "edit"
    assert parsed.target_file == "list.py"


def test_is_mechanical_patch_error() -> None:
    assert is_mechanical_patch_error("mismatch: block 1 SEARCH code not found")
    assert is_mechanical_patch_error("invalid_patch: block 3 overlaps another block")
    assert not is_mechanical_patch_error("SyntaxError: invalid syntax")
    assert not is_mechanical_patch_error("")


def test_build_patch_retry_state_text_includes_error_and_hint() -> None:
    base = _build_state_text("list.py", "add decorator", ["build_router"])
    text = build_patch_retry_state_text(
        base,
        attempt=1,
        max_attempts=3,
        error="invalid_patch: block 3 overlaps another block",
        failed_patch="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
    )
    assert "PATCH_RETRY_FEEDBACK (attempt 2/3)" in text
    assert "overlaps another block" in text
    assert "disjoint line ranges" in text
    assert "add decorator" in text


@pytest.mark.asyncio
async def test_execute_retries_mechanical_patch_error_then_succeeds(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    target = tmp_path / "list.py"
    target.write_text("def route_a():\n    return 1\n", encoding="utf-8")

    harness = MagicMock()
    harness.before_llm_call = AsyncMock(side_effect=lambda messages: messages)
    harness.after_llm_call = AsyncMock()

    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(
            cursor_decision_timeout=120.0,
            cursor_decision_patch_retries=2,
            cursor_validator_model="none",
            cursor_validator_command=["pytest"],
            cursor_validator_timeout=10.0,
            cursor_observation_max_chars=1000,
            cursor_validator_semantic_timeout=5.0,
            prompt_cache_ttl="5m",
        ),
        decision_llm=MagicMock(),
        harness=harness,
    )

    good_patch = (
        "<<<<<<< SEARCH\ndef route_a():\n    return 1\n=======\n"
        "def route_a():\n    return 2\n>>>>>>> REPLACE"
    )
    bad_patch = "<<<<<<< SEARCH\nmissing\n=======\nnew\n>>>>>>> REPLACE"
    decisions = [
        Decision(action="edit", target_file="list.py", patch=bad_patch, suggested_completion=0),
        Decision(action="edit", target_file="list.py", patch=good_patch, suggested_completion=0),
    ]
    generate_calls: list[str] = []

    async def fake_generate_patch(**kwargs: object) -> tuple[Decision, MagicMock]:
        generate_calls.append(str(kwargs.get("state_text") or ""))
        return decisions[len(generate_calls) - 1], MagicMock()

    exec_results = [
        (
            ExecutionResult(success=False, file="list.py", error="mismatch: block 1 SEARCH code not found"),
            ValidationResult(success=False),
            MagicMock(),
        ),
        (
            ExecutionResult(success=True, file="list.py"),
            ValidationResult(success=True),
            MagicMock(),
        ),
    ]

    with patch.object(tool, "_generate_patch", side_effect=fake_generate_patch), patch.object(
        tool.executor,
        "execute_transaction",
        new=AsyncMock(side_effect=exec_results),
    ):
        result = await tool.execute(
            target_file="list.py",
            intent="change route_a return value",
            context_window=[{"file": "list.py", "span": [1, 2]}],
        )

    assert result.success is True
    assert len(generate_calls) == 2
    assert "PATCH_RETRY_FEEDBACK" in generate_calls[1]
    assert result.metadata["patch_attempts"] == 2


@pytest.mark.asyncio
async def test_execute_does_not_retry_validator_failure(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    target = tmp_path / "list.py"
    target.write_text("def route_a():\n    return 1\n", encoding="utf-8")

    harness = MagicMock()
    harness.before_llm_call = AsyncMock(side_effect=lambda messages: messages)
    harness.after_llm_call = AsyncMock()

    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(
            cursor_decision_timeout=120.0,
            cursor_decision_patch_retries=2,
            cursor_validator_model="none",
            cursor_validator_command=["pytest"],
            cursor_validator_timeout=10.0,
            cursor_observation_max_chars=1000,
            cursor_validator_semantic_timeout=5.0,
            prompt_cache_ttl="5m",
        ),
        decision_llm=MagicMock(),
        harness=harness,
    )

    patch_text = (
        "<<<<<<< SEARCH\ndef route_a():\n    return 1\n=======\n"
        "def route_a():\n    return 2\n>>>>>>> REPLACE"
    )
    decision = Decision(action="edit", target_file="list.py", patch=patch_text, suggested_completion=0)

    generate_mock = AsyncMock(return_value=(decision, MagicMock()))

    with patch.object(
        tool,
        "_generate_patch",
        new=generate_mock,
    ), patch.object(
        tool.executor,
        "execute_transaction",
        new=AsyncMock(
            return_value=(
                ExecutionResult(success=True, file="list.py"),
                ValidationResult(success=False, error="pytest failed"),
                MagicMock(),
            )
        ),
    ):
        result = await tool.execute(
            target_file="list.py",
            intent="change route_a return value",
            context_window=[{"file": "list.py", "span": [1, 2]}],
        )

    assert result.success is False
    assert generate_mock.await_count == 1
    assert "pytest failed" in (result.error or "")
    assert result.metadata["patch_attempts"] == 1
