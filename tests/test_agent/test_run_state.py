from __future__ import annotations

from dataclasses import replace

from src.agent.run_state import (
    ArtifactRefs,
    Evidence,
    RunEvent,
    RunPhase,
    reduce_run_state,
    start_run,
)


def _evidence(slot: str, artifact_id: str) -> Evidence:
    return Evidence(
        slot=slot,
        artifact_id=artifact_id,
        file="main.py",
        symbol=slot,
        evidence_type="full_symbol",
    )


def test_diagnose_run_moves_to_responding_only_after_grounded_evidence() -> None:
    state = start_run("检查函数", edit_mode=False)
    assert state.phase == RunPhase.RETRIEVING

    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "a1"),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )

    assert state.phase == RunPhase.RESPONDING
    assert state.retrieval_complete is True
    assert state.allowed_tools == frozenset()


def test_evidence_without_artifact_cannot_ground_a_slot() -> None:
    state = start_run("检查函数", edit_mode=False)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "missing"),),
        ),
    )

    assert state.phase == RunPhase.RETRIEVING
    assert state.evidence.grounded == frozenset()


def test_reducer_reconciles_successful_symbol_observation_into_manifest() -> None:
    state = start_run("修改接口", edit_mode=True)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            observations=(
                {
                    "file": "list.py",
                    "span": [10, 20],
                    "symbol": "build_router",
                    "code": "def build_router():\n    pass",
                },
            ),
        ),
    )

    observed = next(
        item for item in state.manifest.required_items
        if item.id == "observed.symbol:list.py:build_router"
    )
    assert observed.role == "observed"
    assert observed.file == "list.py"
    assert observed.symbol == "build_router"
    assert observed.span == (10, 20)


def test_run_starts_with_empty_bootstrap_manifest() -> None:
    state = start_run("增加订单时间线功能接口和新建表", edit_mode=True)

    assert state.manifest.required_items == ()
    assert state.manifest.sufficiency == "INSUFFICIENT"
    assert state.evidence.required  # task hints live on EvidenceLedger, not manifest slots


def test_reducer_tracks_consecutive_retrieval_rounds_without_new_observations() -> None:
    state = start_run("修改接口", edit_mode=True)

    state, _ = reduce_run_state(
        state,
        RunEvent("tool_round_observed", retrieval_attempted=True),
    )
    assert state.retrieval_no_gain_rounds == 1

    state, _ = reduce_run_state(
        state,
        RunEvent("tool_round_observed", retrieval_attempted=True),
    )
    assert state.retrieval_no_gain_rounds == 2


def test_reducer_resets_rounds_since_last_edit_on_applied_edit() -> None:
    state = start_run("修改接口", edit_mode=True)
    state, _ = reduce_run_state(state, RunEvent("edit_applied", file="list.py"))
    state, _ = reduce_run_state(
        state,
        RunEvent("validation_finished"),
    )
    state, _ = reduce_run_state(
        state,
        RunEvent("tool_round_observed", edit_applied_this_round=True),
    )
    assert state.rounds_since_last_edit == 0

    state, _ = reduce_run_state(
        state,
        RunEvent("tool_round_observed", retrieval_attempted=True),
    )
    assert state.rounds_since_last_edit == 1


def test_reducer_records_all_duplicate_view_round() -> None:
    state = start_run("修改接口", edit_mode=True)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "tool_round_observed",
            retrieval_attempted=True,
            view_all_duplicate=True,
        ),
    )

    assert state.retrieval_no_gain_rounds == 1
    assert state.view_last_round_all_duplicate is True


def test_new_durable_observation_resets_retrieval_no_gain_rounds() -> None:
    state = replace(
        start_run("修改接口", edit_mode=True),
        retrieval_no_gain_rounds=2,
    )
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            observations=(
                {
                    "file": "list.py",
                    "span": [1, 2],
                    "symbol": "handler",
                    "code": "def handler():\n    pass",
                },
            ),
        ),
    )
    state, _ = reduce_run_state(
        state,
        RunEvent("tool_round_observed", retrieval_attempted=True),
    )

    assert state.retrieval_no_gain_rounds == 0


def test_reducer_records_grep_error_and_suggested_views() -> None:
    state = start_run("订单时间线", edit_mode=True)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "tool_round_observed",
            retrieval_attempted=True,
            grep_error="grep_search requires pattern",
        ),
    )
    assert state.last_grep_error == "grep_search requires pattern"

    state = replace(
        state,
        retrieval_gain_in_round=True,
    )
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "tool_round_observed",
            retrieval_attempted=True,
            grep_suggested_views=(
                {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [10, 10]},
            ),
        ),
    )
    assert state.last_grep_error == ""
    assert state.grep_suggested_views[0]["symbol"] == "ticket_order"


def test_responding_rejects_tool_calls_and_accepts_answer() -> None:
    state = start_run("检查函数", edit_mode=False)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "a1"),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )

    unchanged, effects = reduce_run_state(
        state,
        RunEvent("tool_requested", tool_name="grep_search"),
    )
    assert unchanged == state
    assert effects[0].kind == "reject_tool"

    state, _ = reduce_run_state(
        state,
        RunEvent("answer_proposed", answer="done"),
    )
    assert state.phase == RunPhase.TERMINAL
    assert state.terminal and state.terminal.answer == "done"


def test_edit_run_transitions_through_validation() -> None:
    state = start_run("修改函数", edit_mode=True)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "a1"),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )
    assert state.phase == RunPhase.ACTING
    assert state.allowed_tools == frozenset({"decision_edit"})

    state, effects = reduce_run_state(
        state,
        RunEvent("edit_applied", file="main.py"),
    )
    assert state.phase == RunPhase.VALIDATING
    assert effects[0].kind == "run_validation"

    state, _ = reduce_run_state(state, RunEvent("validation_finished"))
    assert state.phase == RunPhase.ACTING
    assert state.can_answer is True
    assert state.allowed_tools == frozenset({"decision_edit"})


def test_security_requirements_are_derived_once_at_run_start() -> None:
    state = start_run(
        "归档接口接入登录态、角色和数据归属校验",
        edit_mode=True,
    )
    assert state.evidence.required == frozenset(
        {
            "target_implementation",
            "endpoint_implementation",
            "integration_or_mount_point",
            "authentication_context",
            "authorization_policy",
            "ownership_relation",
            "relevant_schema",
        }
    )


def test_artifact_invalidation_removes_grounding_without_hidden_phase_change() -> None:
    state = start_run("修改函数", edit_mode=True)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "a1"),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )
    assert state.phase == RunPhase.ACTING

    state, _ = reduce_run_state(
        state,
        RunEvent("artifacts_invalidated", artifact_refs=ArtifactRefs(code=("a1",))),
    )

    assert state.evidence.grounded == frozenset()
    assert state.artifacts.all == frozenset()
    assert state.phase == RunPhase.ACTING


def test_permission_waiting_round_trip_is_reducer_owned() -> None:
    state = start_run("检查函数", edit_mode=False)
    state, effects = reduce_run_state(
        state,
        RunEvent("permission_required", reason="approval needed"),
    )
    assert state.phase == RunPhase.WAITING_USER
    assert effects[0].kind == "request_permission"

    state, _ = reduce_run_state(state, RunEvent("permission_granted"))
    assert state.phase == RunPhase.RETRIEVING


def test_terminal_state_ignores_late_events() -> None:
    state = start_run("检查函数", edit_mode=False)
    state, _ = reduce_run_state(
        state,
        RunEvent(
            "evidence_stored",
            evidence=(_evidence("target_implementation", "a1"),),
            artifact_refs=ArtifactRefs(code=("a1",)),
        ),
    )
    state, _ = reduce_run_state(state, RunEvent("answer_proposed", answer="done"))
    terminal = state

    state, effects = reduce_run_state(state, RunEvent("step_started"))

    assert state == terminal
    assert effects[0].kind == "ignored"


def test_explicit_run_failure_always_enters_terminal() -> None:
    state = start_run("检查函数", edit_mode=False)

    state, _ = reduce_run_state(
        state,
        RunEvent("run_failed", reason="maximum steps exhausted"),
    )

    assert state.phase == RunPhase.TERMINAL
    assert state.terminal and state.terminal.success is False
