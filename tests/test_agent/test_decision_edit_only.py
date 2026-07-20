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
    count_stream_diff_lines,
    decision_timeout_for_context,
    format_edit_progress,
    is_format_validation_retry,
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


def test_count_stream_diff_lines_legacy_search_replace() -> None:
    patch = (
        "<<<<<<< SEARCH\n"
        "a = 1\n"
        "b = 2\n"
        "=======\n"
        "a = 1\n"
        "b = 99\n"
        "c = 3\n"
        ">>>>>>> REPLACE"
    )
    assert count_stream_diff_lines(patch) == (3, 2)


def test_count_stream_diff_lines_site_replace_only() -> None:
    patch = (
        "SITE: symbol=foo\n"
        "<<<<<<< REPLACE\n"
        "def foo():\n"
        "    return 2\n"
        ">>>>>>> REPLACE"
    )
    added, removed = count_stream_diff_lines(patch)
    assert added == 2
    assert removed == 0  # no span= yet; SEARCH filled later by harness


def test_count_stream_diff_lines_site_span_estimates_removed() -> None:
    patch = (
        "SITE: span=24-33\n"
        "<<<<<<< REPLACE\n"
        "def strip_chat_noise(raw: str) -> str:\n"
        "    return raw\n"
        ">>>>>>> REPLACE"
    )
    added, removed = count_stream_diff_lines(patch)
    assert added == 2
    assert removed == 10  # 24..33 inclusive


def test_format_edit_progress_marks_blue_stats() -> None:
    text = format_edit_progress("noise_policy.py", 5, 10)
    assert "[bold blue]noise_policy.py[/]" in text
    assert "[bold blue][+5 -10][/]" in text
    assert "正在编辑文件:" in text


def test_decision_edit_only_mode_rejects_ask_clarify() -> None:
    with pytest.raises(DecisionError, match="edit-only mode forbids action 'ask_clarify'"):
        CursorDecisionLLM.parse(
            '{"action":"ask_clarify","answer":"","clarification":"need more context",'
            '"target_file":"","patch":"","suggested_completion":0}',
            ("db/init/init.sql",),
            edit_only=True,
        )


def test_parse_block_format_edit_with_raw_code() -> None:
    """Legacy SEARCH/REPLACE still parses (backward compatible)."""
    raw = (
        "ACTION: edit\n"
        "TARGET_FILE: agentmesh/noise_policy.py\n"
        "COMPLETION: 50\n"
        "<<<<<<< SEARCH\n"
        '    deictic = doc.get("deictic_followup_patterns") or []\n'
        '    re_code = _rx("code", r"class\\s+\\w+")\n'
        "=======\n"
        '    deictic = doc.get("deictic_followup_patterns") or []\n'
        '    re_code = _rx("code", r"class\\s+\\w+")\n'
        '    bot_nicknames = frozenset()\n'
        ">>>>>>> REPLACE"
    )

    decision = CursorDecisionLLM.parse(raw, ("agentmesh/noise_policy.py",))

    assert decision.action == "edit"
    assert decision.target_file == "agentmesh/noise_policy.py"
    assert abs(decision.suggested_completion - 0.5) < 1e-9
    assert 'doc.get("deictic_followup_patterns")' in decision.patch
    assert r'r"class\s+\w+"' in decision.patch
    assert decision.patch.startswith("<<<<<<< SEARCH")
    assert decision.patch.rstrip().endswith(">>>>>>> REPLACE")


def test_parse_block_format_site_replace_only() -> None:
    """Preferred contract: SITE + REPLACE, no SEARCH."""
    raw = (
        "ACTION: edit\n"
        "TARGET_FILE: noise_policy.py\n"
        "COMPLETION: 40\n"
        "SITE: symbol=from_yaml\n"
        "<<<<<<< REPLACE\n"
        "def from_yaml(path: str) -> NoisePolicy:\n"
        '    bot_nicknames = frozenset()\n'
        "    return NoisePolicy(bot_nicknames=bot_nicknames)\n"
        ">>>>>>> REPLACE"
    )
    decision = CursorDecisionLLM.parse(raw, ("noise_policy.py",))
    assert decision.action == "edit"
    assert "SITE: symbol=from_yaml" in decision.patch
    assert "<<<<<<< SEARCH" not in decision.patch
    assert decision.patch.lstrip().startswith("SITE:")
    assert "def from_yaml" in decision.patch


def test_parse_block_format_edit_only_rejects_answer_action() -> None:
    raw = "ACTION: answer\nCOMPLETION: 100\nANSWER: all done"
    with pytest.raises(DecisionError, match="edit-only mode forbids"):
        CursorDecisionLLM.parse(raw, ("a.py",), edit_only=True)


def test_parse_block_format_answer() -> None:
    raw = "ACTION: answer\nCOMPLETION: 100\nANSWER: The bug is a missing null check."
    decision = CursorDecisionLLM.parse(raw, ("a.py",))
    assert decision.action == "answer"
    assert decision.answer == "The bug is a missing null check."
    assert decision.target_file == ""
    assert decision.patch == ""


def test_parse_block_format_ask_clarify() -> None:
    raw = "ACTION: ask_clarify\nCOMPLETION: 0\nCLARIFICATION: Which file holds the config?"
    decision = CursorDecisionLLM.parse(raw, ("a.py",))
    assert decision.action == "ask_clarify"
    assert decision.clarification == "Which file holds the config?"


def test_parse_block_format_tolerates_outer_markdown_fence() -> None:
    raw = (
        "```\n"
        "ACTION: edit\n"
        "TARGET_FILE: a.py\n"
        "COMPLETION: 30\n"
        "<<<<<<< SEARCH\n"
        "x = 1\n"
        "=======\n"
        "x = 2\n"
        ">>>>>>> REPLACE\n"
        "```"
    )
    decision = CursorDecisionLLM.parse(raw, ("a.py",))
    assert decision.action == "edit"
    assert "x = 1" in decision.patch and "x = 2" in decision.patch


def test_parse_salvages_invalid_regex_backslash_escapes() -> None:
    """Edit model writes regex like \\W into the patch without doubling the
    backslash; parse should salvage it instead of failing with Invalid \\escape."""
    raw = (
        '{"action":"edit","answer":"","clarification":"",'
        '"target_file":"db/init/init.sql",'
        '"patch":"<<<<<<< SEARCH\\nre_punctuation_only = r\\"^[\\W_]+$\\"\\n'
        '=======\\nre_punctuation_only = r\\"^[\\W_ ]+$\\"\\n>>>>>>> REPLACE",'
        '"suggested_completion":40}'
    )

    decision = CursorDecisionLLM.parse(raw, ("db/init/init.sql",))

    assert decision.action == "edit"
    assert "[\\W_]" in decision.patch
    assert "[\\W_ ]" in decision.patch


def test_parse_preserves_already_valid_escapes_when_salvaging() -> None:
    """A patch mixing valid escapes (\\n, \\", \\\\) with an invalid one (\\d)
    must survive without corrupting the already-correct escapes."""
    raw = (
        '{"action":"edit","answer":"","clarification":"",'
        '"target_file":"a.py",'
        '"patch":"<<<<<<< SEARCH\\nx = \\"a\\\\b\\"\\n=======\\n'
        'y = re.compile(r\\"\\d+\\")\\n>>>>>>> REPLACE",'
        '"suggested_completion":10}'
    )

    decision = CursorDecisionLLM.parse(raw, ("a.py",))

    assert decision.action == "edit"
    # The literal backslash-b escape stays a single backslash after JSON decode.
    assert 'x = "a\\b"' in decision.patch
    # The salvaged \d survives as a literal backslash-d.
    assert "\\d+" in decision.patch


def test_parse_salvages_raw_newlines_in_patch_string() -> None:
    """The Edit model often emits a multi-line patch with literal newlines instead
    of \\n, tripping json.loads with 'Invalid control character'. Salvage it."""
    raw = (
        '{"action":"edit","answer":"","clarification":"",'
        '"target_file":"a.py",\n'
        '"patch":"<<<<<<< SEARCH\n'
        "def f():\n"
        "    return 1\n"
        "=======\n"
        "def f():\n"
        "    return 2\n"
        '>>>>>>> REPLACE","suggested_completion":30}'
    )

    decision = CursorDecisionLLM.parse(raw, ("a.py",))

    assert decision.action == "edit"
    assert "<<<<<<< SEARCH\ndef f():\n    return 1" in decision.patch
    assert "return 2" in decision.patch


def test_parse_salvages_raw_newlines_mixed_with_escaped_quotes() -> None:
    """Real payloads mix literal newlines with correctly-escaped quotes and
    backslashes inside the regex; both must survive the same salvage pass."""
    raw = (
        '{"action":"edit","answer":"","clarification":"",'
        '"target_file":"n.py","patch":"<<<<<<< SEARCH\n'
        '    text = re.sub(r\\"\\\\s{2,}\\", \\" \\", text)\n'
        "=======\n"
        '    text = re.sub(r\\"\\\\s{2,}\\", \\" \\", text)\n'
        '    text = re.sub(r\\"@[^\\\\s]+\\", \\"\\", text)\n'
        '>>>>>>> REPLACE","suggested_completion":30}'
    )

    decision = CursorDecisionLLM.parse(raw, ("n.py",))

    assert decision.action == "edit"
    assert 'r"\\s{2,}"' in decision.patch
    assert 'r"@[^\\s]+"' in decision.patch


def test_parse_salvages_raw_quotes_and_newlines_via_markers() -> None:
    """Worst case: the patch body has BOTH raw double quotes and raw newlines
    (only backslashes escaped), which defeats quote-aware repair. The SEARCH/
    REPLACE sentinels must still recover it."""
    target = "agentmesh/app/noise_policy.py"
    raw = (
        '{"action":"edit","answer":"","clarification":"",'
        f'"target_file":"{target}\n'
        '","patch":"<<<<<<< SEARCH\n'
        '    deictic = doc.get("deictic_followup_patterns") or []\n'
        "    return NoisePolicy(\n"
        '        re_code=_rx("code", r"class\\\\s+\\\\w+"),\n'
        "        deictic_followup_patterns=_compile_deictic(deictic),\n"
        "    )=======\n"
        '    deictic = doc.get("deictic_followup_patterns") or []\n'
        "    return NoisePolicy(\n"
        '        re_code=_rx("code", r"class\\\\s+\\\\w+"),\n'
        "        deictic_followup_patterns=_compile_deictic(deictic),\n"
        '        bot_nicknames=frozenset(),\n'
        '    )>>>>>>> REPLACE","suggested_completion":50}'
    )

    decision = CursorDecisionLLM.parse(raw, (target,))

    assert decision.action == "edit"
    assert decision.target_file == target
    # Raw content quotes survive, backslash escapes decode to single, markers kept.
    assert 'doc.get("deictic_followup_patterns")' in decision.patch
    assert 'r"class\\s+\\w+"' in decision.patch
    assert decision.patch.startswith("<<<<<<< SEARCH")
    assert decision.patch.rstrip().endswith(">>>>>>> REPLACE")
    assert "bot_nicknames=frozenset()" in decision.patch


def test_parse_still_raises_on_unsalvageable_json() -> None:
    with pytest.raises(DecisionError, match="invalid JSON"):
        CursorDecisionLLM.parse(
            '{"action":"edit","answer":"" "clarification":"",'
            '"target_file":"a.py","patch":"x"',
            ("a.py",),
        )


def test_decision_edit_only_prompt_mentions_edit_only_mode() -> None:
    from src.agent.decision import _decision_system_prompt

    _decision_system_prompt.cache_clear()
    messages = CursorDecisionLLM(MagicMock()).build_messages(
        state_text="apply change",
        context_pack=MagicMock(windows=(), candidate_files=()),
        hint=None,
        edit_only=True,
    )

    assert "EDIT_ONLY_MODE" in messages[0]["content"]
    assert "ACTION: edit" in messages[0]["content"]
    assert "SITE" in messages[0]["content"]
    assert "do not emit search" in messages[0]["content"].casefold()
    assert "insert_after" in messages[0]["content"]


def test_decision_prompt_uses_site_replace_contract() -> None:
    from src.context.prompt_resources import load_internal_prompt

    prompt = load_internal_prompt("decision_prompt.md", fallback="")
    assert "SITE: symbol=" in prompt
    assert "Do NOT emit SEARCH" in prompt
    assert "insert_after" in prompt
    assert "ANCHOR:" in prompt
    assert "<<<<<<< REPLACE" in prompt


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


def test_empty_context_window_does_not_dump_whole_file(tmp_path: Path) -> None:
    """Regression: validate_params defaults context_window=[] which used to
    bypass span selection and load mode=full entire target files (~10k+ tokens).
    """
    huge = "\n".join(f"line_{index} = {index}" for index in range(1, 2001)) + "\n"
    (tmp_path / "huge.py").write_text(huge, encoding="utf-8")
    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(),
        decision_llm=MagicMock(),
        harness=MagicMock(project_root=tmp_path),
    )

    pack = tool._build_context_pack("huge.py", context_window=[])
    assert pack.windows
    total_chars = sum(len(window.content) for window in pack.windows)
    assert total_chars <= 12_000
    assert all(window.mode != "full" for window in pack.windows)

    focused = tool._build_context_pack(
        "huge.py",
        context_window=[],
        focus_symbols=["line_500"],
    )
    # focus_symbols only matches def/class — still must stay budgeted
    assert sum(len(window.content) for window in focused.windows) <= 12_000


def test_context_window_spans_are_clipped_to_edit_budget(tmp_path: Path) -> None:
    body = "\n".join(f"x = {index}" for index in range(1, 800)) + "\n"
    (tmp_path / "wide.py").write_text(body, encoding="utf-8")
    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(
            cursor_max_context_files=3,
            cursor_context_chars_per_file=50_000,
        ),
        decision_llm=MagicMock(),
        harness=MagicMock(project_root=tmp_path),
    )
    pack = tool._build_context_pack(
        "wide.py",
        context_window=[{"file": "wide.py", "span": [1, 800]}],
    )
    assert len(pack.windows) == 1
    assert len(pack.windows[0].content) <= 6_000 + 80  # clip marker allowance
    assert "truncated for Edit LLM" in pack.windows[0].content


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
    # Additive budget is capped so small packs are not misread as 160s+ waits.
    assert large <= 120.0 + 45.0 + 36.0
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
    assert is_mechanical_patch_error(
        "invalid_patch: too many SEARCH/REPLACE blocks (4 > 3). Keep at most 3"
    )
    assert not is_mechanical_patch_error("SyntaxError: invalid syntax")
    assert not is_mechanical_patch_error("")


def test_is_format_validation_retry_detects_marker_residue() -> None:
    assert is_format_validation_retry(
        "AST validation failed: python_syntax_error",
        attempted_content="x = 1\n>>>>>>> REPLACE<<<<<<< SEARCH\ny = 2\n",
    )
    assert not is_format_validation_retry(
        "AST validation failed: python_syntax_error",
        attempted_content="def broken(\n",
    )


def test_build_patch_retry_state_text_includes_error_and_hint() -> None:
    base = _build_state_text("list.py", "add decorator", ["build_router"])
    text = build_patch_retry_state_text(
        base,
        attempt=1,
        max_attempts=2,
        error="invalid_patch: block 3 overlaps another block",
        failed_patch="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
    )
    assert "PATCH_RETRY_FEEDBACK (attempt 2/2)" in text
    assert "overlaps another block" in text
    assert "disjoint" in text or "小步" in text
    assert "add decorator" in text
    assert "ACTION: edit" in text
    assert "action=edit JSON" not in text


def test_build_patch_retry_state_text_noop_omits_failed_echo_body() -> None:
    base = _build_state_text("noise_policy.py", "wire bot_nicknames", ["strip_chat_noise"])
    echoed = (
        "SITE: symbol=strip_chat_noise\n"
        "<<<<<<< REPLACE\n"
        "def strip_chat_noise(raw: str) -> str:\n"
        "    return raw\n"
        ">>>>>>> REPLACE"
    )
    text = build_patch_retry_state_text(
        base,
        attempt=1,
        max_attempts=2,
        error=(
            "E1_FORMAT: SITE 1 REPLACE equals on-disk span 24-33; produce a real change.\n"
            "REPLACE must be the AFTER-edit text for this SITE"
        ),
        failed_patch=echoed,
    )
    assert "Do NOT copy the previous REPLACE body" in text
    assert "Failed patch excerpt:\n```" not in text
    assert "def strip_chat_noise(raw: str) -> str:" not in text
    assert "AFTER-edit" in text
    assert "wire bot_nicknames" in text


def test_build_patch_retry_state_text_too_many_blocks_hint() -> None:
    base = _build_state_text("list.py", "decorate routes", ["a", "b", "c", "d"])
    text = build_patch_retry_state_text(
        base,
        attempt=1,
        max_attempts=2,
        error=(
            "invalid_patch: too many SEARCH/REPLACE blocks (4 > 3). "
            "Keep at most 3 blocks per decision_edit; prefer one block when possible."
        ),
        failed_patch="<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE",
    )
    assert "at most 3" in text
    assert "小步" in text


def test_build_patch_retry_state_text_e2_locate_mentions_phantom_and_action_edit() -> None:
    base = _build_state_text("noise_policy.py", "add bot_nicknames kwarg", ["build"])
    text = build_patch_retry_state_text(
        base,
        attempt=1,
        max_attempts=2,
        error=(
            "E2_LOCATE: SITE symbol='from_yaml' not found in noise_policy.py. "
            "Use a symbol that exists on disk."
        ),
        failed_patch=(
            "SITE: symbol=from_yaml\n"
            "<<<<<<< REPLACE\n"
            "def from_yaml():\n"
            "    return 1\n"
            ">>>>>>> REPLACE"
        ),
    )
    assert "ACTION: edit" in text
    assert "do NOT emit SEARCH" in text.casefold() or "Do NOT emit SEARCH" in text
    assert "SITE" in text
    assert "symbol=" in text


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
async def test_execute_retries_invalid_json_decision_schema(tmp_path: Path) -> None:
    from unittest.mock import AsyncMock, patch

    (tmp_path / "list.py").write_text("def route_a():\n    return 1\n", encoding="utf-8")
    harness = MagicMock()
    harness.before_llm_call = AsyncMock(side_effect=lambda messages: messages)
    harness.after_llm_call = AsyncMock()
    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(
            cursor_decision_timeout=120.0,
            cursor_decision_patch_retries=1,
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
    generate_calls: list[str] = []

    async def fake_generate_patch(**kwargs: object) -> tuple[Decision, MagicMock]:
        generate_calls.append(str(kwargs.get("state_text") or ""))
        if len(generate_calls) == 1:
            raise DecisionError(
                "decision_schema: invalid JSON: Expecting ',' delimiter: "
                "line 1 column 2062 (char 2061)\n"
                "Raw response excerpt:\n```\n{\"action\":\"edit\",\"patch\":\"broken\n```"
            )
        return (
            Decision(
                action="edit",
                target_file="list.py",
                patch=good_patch,
                suggested_completion=0,
            ),
            MagicMock(),
        )

    with patch.object(tool, "_generate_patch", side_effect=fake_generate_patch), patch.object(
        tool.executor,
        "execute_transaction",
        new=AsyncMock(
            return_value=(
                ExecutionResult(success=True, file="list.py"),
                ValidationResult(success=True),
                MagicMock(),
            )
        ),
    ):
        result = await tool.execute(
            target_file="list.py",
            intent="change route_a return value",
            context_window=[{"file": "list.py", "span": [1, 2]}],
        )

    assert result.success is True
    assert len(generate_calls) == 2
    assert "PATCH_RETRY_FEEDBACK" in generate_calls[1]
    assert "invalid JSON" in generate_calls[1]
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
