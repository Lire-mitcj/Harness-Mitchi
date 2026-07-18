from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency, execution_card
from src.agent.run_state import RunPhase
from src.hooks.reallocate_tools import allocate_tools, determine_allowed_tools
from src.hooks.tool_scorer import (
    EDIT,
    GREP,
    RETRIEVE,
    VIEW,
    ToolScoringContext,
    allocate,
    compute_scores,
    softmax,
)

DEFAULT_TOOLS = frozenset({"grep_search", "decision_edit", "codebase_retrieve", "view_symbol_code"})


def test_softmax_normalizes_and_ranks() -> None:
    probs = softmax({GREP: 2.0, VIEW: 1.0, EDIT: 0.0}, temperature=0.6)

    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs[GREP] > probs[VIEW] > probs[EDIT]


def test_softmax_low_temperature_sharpens() -> None:
    scores = {GREP: 2.0, VIEW: 1.0}
    sharp = softmax(scores, temperature=0.1)
    flat = softmax(scores, temperature=2.0)

    assert sharp[GREP] > flat[GREP]


def test_softmax_empty_returns_empty() -> None:
    assert softmax({}, temperature=0.6) == {}


def test_verification_context_prefers_view_over_edit() -> None:
    ctx = ToolScoringContext(
        task_mode="edit",
        sufficiency=str(Sufficiency.SUFFICIENT_FOR_EDIT),
        coverage=1.0,
        stale_ratio=0.5,
        verification_mode=True,
        actionable_suggested_views=True,
    )
    scores = compute_scores(ctx)

    assert scores[VIEW] > scores[EDIT]
    assert scores[VIEW] > scores[RETRIEVE]


def test_edit_burst_context_prefers_edit() -> None:
    ctx = ToolScoringContext(
        task_mode="edit",
        sufficiency=str(Sufficiency.SUFFICIENT_FOR_EDIT),
        coverage=1.0,
        edit_burst=True,
    )
    scores = compute_scores(ctx)

    assert scores[EDIT] > scores[GREP]
    assert scores[EDIT] > scores[VIEW]


def test_bootstrap_missing_prefers_grep_and_retrieve() -> None:
    ctx = ToolScoringContext(
        task_mode="edit",
        sufficiency=str(Sufficiency.INSUFFICIENT),
        missing_ratio=1.0,
        has_missing=True,
        critical_missing=True,
        bootstrap=True,
    )
    scores = compute_scores(ctx)

    assert scores[GREP] > scores[EDIT]
    assert scores[RETRIEVE] > scores[EDIT]


def test_allocate_restricts_probabilities_to_allowed() -> None:
    ctx = ToolScoringContext(
        task_mode="edit",
        sufficiency=str(Sufficiency.SUFFICIENT_FOR_EDIT),
        coverage=1.0,
        verification_mode=True,
        stale_ratio=0.5,
    )
    allocation = allocate(ctx, frozenset({VIEW, GREP}))

    assert set(allocation.probabilities) == {VIEW, GREP}
    assert allocation.preferred in {VIEW, GREP}
    assert allocation.preferred == VIEW
    assert EDIT not in allocation.probabilities


def test_allocate_empty_allowed_has_no_preferred() -> None:
    allocation = allocate(ToolScoringContext(), frozenset())

    assert allocation.probabilities == {}
    assert allocation.preferred is None


def _verification_state() -> SimpleNamespace:
    run_state = SimpleNamespace(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="observed.symbol:main.py:create_app",
                    need="main wiring",
                    type="symbol",
                    role="observed",
                    file="main.py",
                    span=(1, 80),
                    symbol="create_app",
                    status="STALE",
                    stale_reason="file modified by decision_edit",
                ),
                EvidenceItem(
                    id="observed.symbol:list.py:build_router",
                    need="handler",
                    type="symbol",
                    role="observed",
                    file="list.py",
                    span=(16, 358),
                    symbol="build_router",
                    status="SATISFIED",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="passed"),
        changes=SimpleNamespace(files=("list.py",)),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=False,
        edit_patch_failed=False,
        rounds_since_last_edit=0,
    )
    return SimpleNamespace(run_state=run_state, checklist=())


def test_allocate_tools_matches_gate_and_prefers_view_in_verification() -> None:
    state = _verification_state()
    gated = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)
    allocation = allocate_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allocation.allowed == gated
    assert "view_symbol_code" in allocation.allowed
    assert allocation.preferred == "view_symbol_code"
    assert allocation.probabilities[allocation.preferred] == max(
        allocation.probabilities.values()
    )


def test_allocate_tools_edit_burst_prefers_edit() -> None:
    run_state = SimpleNamespace(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="grounded",
                    need="grounded target",
                    status="SATISFIED",
                    file="list.py",
                    symbol="build_router",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="passed"),
        changes=SimpleNamespace(files=("db/init/init.sql",)),
        edit_patch_failed=False,
        rounds_since_last_edit=0,
        retrieval_no_gain_rounds=0,
        view_last_round_all_duplicate=False,
    )
    state = SimpleNamespace(
        run_state=run_state,
        checklist=("[ ] Wire handler in list.py",),
    )
    allocation = allocate_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allocation.allowed == frozenset({"decision_edit"})
    assert allocation.preferred == "decision_edit"


def test_execution_card_renders_preferred_and_ranking() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="grounded",
                need="grounded target",
                status="SATISFIED",
                file="list.py",
                symbol="build_router",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    card = execution_card(
        manifest,
        ["view_symbol_code", "grep_search"],
        preferred_tool="view_symbol_code",
        tool_probabilities={"view_symbol_code": 0.72, "grep_search": 0.28},
    )

    assert "preferred_tool: view_symbol_code" in card
    assert "tool_ranking: view_symbol_code(0.72) > grep_search(0.28)" in card
