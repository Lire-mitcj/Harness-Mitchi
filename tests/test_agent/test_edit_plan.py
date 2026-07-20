from __future__ import annotations

from src.agent.edit_plan import (
    MAX_PATCH_BLOCKS_PER_MINOR_STEP,
    EditQueueStep,
    build_edit_queue,
    edit_plan_view_from_queue,
    edit_plan_view_halt,
    format_edit_plan_evidence_lines,
    format_edit_plan_runtime_card,
    major_minor_contract_text,
    open_major_steps,
    parse_edit_queue_from_text,
    parse_major_steps,
    sync_edit_queue_from_core_turn,
    validate_minor_step,
)


def test_parse_major_steps_from_checklist() -> None:
    majors = parse_major_steps(
        (
            "[√] Add bot_nicknames field",
            "[ ] Update YAML config",
            "[ ] Wire mention detection",
        )
    )
    assert len(majors) == 3
    assert majors[0].done is True
    assert majors[0].text == "Add bot_nicknames field"
    assert majors[1].open is True
    assert [step.text for step in majors if step.open] == [
        "Update YAML config",
        "Wire mention detection",
    ]


def test_open_major_steps() -> None:
    open_steps = open_major_steps(
        "- [√] done item\n- [ ] next item\n- [ ] later item\n"
    )
    assert [step.text for step in open_steps] == ["next item", "later item"]


def test_build_edit_queue_links_major_and_minor() -> None:
    checklist = (
        "[ ] Add nicknames to policy",
        "[ ] Wire mention path",
    )
    queue = build_edit_queue(
        [
            {
                "target_file": "noise_policy.py",
                "intent": "add bot_nicknames field",
                "focus_symbols": ["NoisePolicy"],
                "major_step_index": 0,
            },
            {
                "target_file": "noise_policy.yaml",
                "intent": "add bot_nicknames list",
                "focus_symbols": [],
                "major_step_index": 0,
            },
            {
                "file": "grpc_server.py",
                "intent": "use nicknames in mention check",
                "symbols": ["is_mentioned"],
                "major_step_index": 1,
            },
        ],
        checklist=checklist,
    )
    assert len(queue.major_steps) == 2
    assert len(queue.minor_steps) == 3
    assert queue.current is not None
    assert queue.current.target_file == "noise_policy.py"
    assert queue.current.major_step_text == "Add nicknames to policy"
    advanced = queue.advance().advance()
    assert advanced.current is not None
    assert advanced.current.target_file == "grpc_server.py"
    assert advanced.advance().drained is True


def test_validate_minor_step() -> None:
    ok, _ = validate_minor_step(
        EditQueueStep(
            target_file="a.py",
            intent="fix",
            focus_symbols=("f",),
            context_window=({"file": "a.py", "span": [1, 10]},),
        )
    )
    assert ok
    bad, reason = validate_minor_step(EditQueueStep(target_file="", intent="x"))
    assert not bad
    assert "E4_SPEC" in reason
    # focus_symbols and context_window are now required (no optional fields).
    no_focus, reason2 = validate_minor_step(
        EditQueueStep(target_file="a.py", intent="fix")
    )
    assert not no_focus
    assert "focus_symbols" in reason2
    no_ctx, reason3 = validate_minor_step(
        EditQueueStep(target_file="a.py", intent="fix", focus_symbols=("f",))
    )
    assert not no_ctx
    assert "context_window" in reason3


def test_contract_text_mentions_major_minor() -> None:
    text = major_minor_contract_text()
    assert "edit_plan" in text
    assert "focus_symbols" in text
    assert "context_window" in text
    assert str(MAX_PATCH_BLOCKS_PER_MINOR_STEP) in text


def test_mechanical_rules_gated_by_plan_status() -> None:
    from src.agent.edit_plan import EditPlanView, mechanical_rule_lines_for_turn

    emit = mechanical_rule_lines_for_turn(
        edit_ready=True,
        edit_plan_view=EditPlanView(status="empty"),
    )
    assert any(line.startswith("plan_contract:") for line in emit)

    drain = mechanical_rule_lines_for_turn(
        edit_ready=True,
        edit_plan_view=EditPlanView(status="draining", total=2, cursor=0),
        edit_burst=True,
    )
    assert any("draining" in line for line in drain)
    assert not any(line.startswith("plan_contract:") for line in drain)

    halt = mechanical_rule_lines_for_turn(
        edit_ready=True,
        edit_plan_view=EditPlanView(
            status="halted",
            total=1,
            error_class="E4_SPEC",
            remaining=("a.py — fix",),
        ),
    )
    assert any("E4_SPEC" in line for line in halt)
    assert any("remaining" in line for line in halt)

    quiet = mechanical_rule_lines_for_turn(
        edit_ready=False,
        edit_plan_view=EditPlanView(status="empty"),
    )
    assert quiet == []


def test_halt_runtime_card_includes_core_action() -> None:
    from src.agent.edit_plan import EditPlanView, format_edit_plan_runtime_card

    card = format_edit_plan_runtime_card(
        EditPlanView(
            status="halted",
            total=1,
            cursor=0,
            failed_file="a.py",
            failed_intent="fix",
            error_class="E5_EVIDENCE",
            fail_reason="missing symbol",
        )
    )
    assert "core_action:" in card
    assert "E5_EVIDENCE" in card


def test_parse_edit_queue_from_fence() -> None:
    text = """
Plan:
- [ ] Wire nicknames

```edit_queue
[
  {"target_file":"a.py","intent":"add field","focus_symbols":["A"],"major_step_index":0},
  {"target_file":"b.py","intent":"wire caller","focus_symbols":["B"],"major_step_index":0}
]
```
"""
    minors = parse_edit_queue_from_text(text)
    assert len(minors) == 2
    assert minors[0]["target_file"] == "a.py"


def test_sync_edit_queue_drops_in_turn_decision_edit() -> None:
    text = """
```edit_queue
[
  {"target_file":"a.py","intent":"step one","focus_symbols":["A"]},
  {"target_file":"b.py","intent":"step two","focus_symbols":["B"]}
]
```
"""
    queue = sync_edit_queue_from_core_turn(
        response_text=text,
        decision_edit_args=[{"target_file": "a.py", "intent": "step one"}],
        checklist=("[ ] Wire nicknames",),
    )
    assert len(queue.minor_steps) == 1
    assert queue.current is not None
    assert queue.current.target_file == "b.py"


def test_sync_auto_followon_when_core_omits_edit_queue_fence() -> None:
    # Two-verb same-file intent stays one step — no auto residual.
    queue = sync_edit_queue_from_core_turn(
        response_text="Fix strip then add helper.",
        decision_edit_args=[
            {
                "target_file": "noise_policy.py",
                "intent": (
                    "Update strip_chat_noise to only strip bot nicknames, "
                    "and add is_bot_mentioned helper"
                ),
                "focus_symbols": ["strip_chat_noise"],
            }
        ],
        checklist=(),
    )
    assert queue.drained is True

    # Three-focus compound expands: first runs in-turn, residual = 2.
    queue2 = sync_edit_queue_from_core_turn(
        response_text="Wire nicknames.",
        decision_edit_args=[
            {
                "target_file": "noise_policy.py",
                "intent": (
                    "Add bot_nicknames to NoisePolicy, update "
                    "load_noise_policy_from_path, and add strip helper"
                ),
                "focus_symbols": [
                    "NoisePolicy",
                    "load_noise_policy_from_path",
                    "strip_chat_noise",
                ],
                "context_window": [
                    {
                        "file": "noise_policy.py",
                        "span": [79, 146],
                        "reason": "NoisePolicy — add field",
                    },
                    {
                        "file": "noise_policy.py",
                        "span": [158, 223],
                        "reason": "load — parse YAML",
                    },
                    {
                        "file": "noise_policy.py",
                        "span": [24, 33],
                        "reason": "strip_chat_noise — helper",
                    },
                ],
            }
        ],
        checklist=(),
    )
    assert len(queue2.minor_steps) == 2
    assert queue2.current is not None
    assert queue2.current.focus_symbols == ("load_noise_policy_from_path",)

def test_edit_plan_runtime_card_halt_includes_remaining() -> None:
    queue = build_edit_queue(
        [
            {
                "target_file": "a.py",
                "intent": "fix A",
                "focus_symbols": ["A"],
                "context_window": [{"file": "a.py", "span": [1, 5]}],
            },
            {
                "target_file": "b.py",
                "intent": "fix B",
                "focus_symbols": ["B"],
                "context_window": [{"file": "b.py", "span": [1, 5]}],
            },
            {
                "target_file": "c.py",
                "intent": "fix C",
                "focus_symbols": ["C"],
                "context_window": [{"file": "c.py", "span": [1, 5]}],
            },
        ]
    )
    failed = queue.current
    assert failed is not None
    view = edit_plan_view_halt(
        queue,
        failed=failed,
        error_class="E2_LOCATE",
        fail_reason="SITE not found",
        completed=("applied_diff_summary: [prev.py] +1 -0",),
    )
    card = format_edit_plan_runtime_card(view)
    assert "plan: [1/3] halted" in card
    assert "ErrorClass=E2_LOCATE" in card
    assert "b.py" in card
    assert "c.py" in card
    assert "Preserved execution checklist" not in card
    assert "applied_diff_summary: [prev.py]" in card

    evidence = "\n".join(format_edit_plan_evidence_lines(view))
    assert "halted" in evidence
    assert "edit_plan_remaining: 2" in evidence


def test_edit_plan_runtime_card_draining() -> None:
    queue = build_edit_queue(
        [
            {
                "target_file": "a.py",
                "intent": "step one",
                "focus_symbols": ["A"],
                "context_window": [{"file": "a.py", "span": [1, 5]}],
            },
            {
                "target_file": "b.py",
                "intent": "step two",
                "focus_symbols": ["B"],
                "context_window": [{"file": "b.py", "span": [1, 5]}],
            },
        ]
    )
    view = edit_plan_view_from_queue(queue)
    card = format_edit_plan_runtime_card(view)
    assert "plan: [1/2] draining" in card
    assert "current: a.py" in card
    assert "upcoming:" in card
    assert "b.py" in card
