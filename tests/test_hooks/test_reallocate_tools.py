from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency
from src.agent.run_state import RunPhase
from src.hooks.reallocate_tools import determine_allowed_tools


def _make_state(
    *,
    task_mode: str,
    sufficiency: Sufficiency,
    missing: tuple[str, ...] = (),
    stale: tuple[str, ...] = (),
    changed_files: tuple[str, ...] = (),
    validation_status: str = "not_run",
    no_gain_rounds: int = 0,
    view_last_round_all_duplicate: bool = False,
    phase: str = "retrieving",
) -> SimpleNamespace:
    items = [EvidenceItem(id=name, need=name, status="MISSING") for name in missing]
    items += [
        EvidenceItem(id=name, need=name, status="STALE", file=name) for name in stale
    ]
    if not items:
        items.append(
            EvidenceItem(
                id="grounded",
                need="grounded target",
                status="SATISFIED",
                file="target.py",
                symbol="target",
            )
        )
    manifest = StepManifest(required_items=tuple(items), sufficiency=sufficiency)
    run_state = SimpleNamespace(
        task_mode=task_mode,
        phase=phase,
        manifest=manifest,
        validation=SimpleNamespace(status=validation_status),
        changes=SimpleNamespace(files=changed_files),
        retrieval_no_gain_rounds=no_gain_rounds,
        view_last_round_all_duplicate=view_last_round_all_duplicate,
    )
    return SimpleNamespace(run_state=run_state)


DEFAULT_TOOLS = frozenset({
    "grep_search",
    "decision_edit",
    "codebase_retrieve",
    "view_symbol_code",
})
PRIMARY_RETRIEVAL = frozenset({"grep_search", "view_symbol_code"})
RETRIEVAL_TOOLS = frozenset({"grep_search", "codebase_retrieve", "view_symbol_code"})
BOOTSTRAP_RETRIEVAL = frozenset({"grep_search", "view_symbol_code", "codebase_retrieve"})
EDIT_PRIMARY = frozenset({"decision_edit", "grep_search", "view_symbol_code"})


def test_insufficient_edit_mode_opens_retrieval_only() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.INSUFFICIENT,
        missing=("target_implementation",),
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == BOOTSTRAP_RETRIEVAL
    assert "decision_edit" not in allowed


def test_sufficient_for_edit_is_edit_only_when_no_gaps() -> None:
    state = _make_state(task_mode="edit", sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_sufficient_for_edit_keeps_primary_retrieval_open_after_edit() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("list.py",),
        validation_status="passed",
        phase=RunPhase.ACTING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_one_no_gain_round_is_edit_only_without_stale_gaps() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        no_gain_rounds=1,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_one_no_gain_all_view_duplicate_keeps_grep_only() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        no_gain_rounds=1,
        view_last_round_all_duplicate=True,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_stale_reopens_exact_reader_after_no_gain() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        stale=("list.py",),
        no_gain_rounds=2,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit", "view_symbol_code"})


def test_two_no_gain_rounds_force_grounded_edit() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        no_gain_rounds=2,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_sufficient_for_edit_with_stale_reopens_retrieval() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        stale=("db/init/init.sql",),
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "decision_edit" in allowed
    assert "view_symbol_code" in allowed


def test_diagnose_answer_when_sufficient_and_fresh() -> None:
    state = _make_state(
        task_mode="diagnose",
        sufficiency=Sufficiency.SUFFICIENT_FOR_ANSWER,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset()


def test_diagnose_insufficient_keeps_retrieval() -> None:
    state = _make_state(
        task_mode="diagnose",
        sufficiency=Sufficiency.INSUFFICIENT,
        missing=("target_implementation",),
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == BOOTSTRAP_RETRIEVAL


def test_diagnose_stale_reopens_retrieval_instead_of_answering() -> None:
    state = _make_state(
        task_mode="diagnose",
        sufficiency=Sufficiency.SUFFICIENT_FOR_ANSWER,
        stale=("list.py",),
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"view_symbol_code"})


def test_validation_failed_allows_edit_and_refresh() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        stale=("list.py",),
        validation_status="failed",
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "decision_edit" in allowed
    assert "view_symbol_code" in allowed


def test_dead_sql_alias_validation_recovery_is_edit_only() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("schema.sql",),
    )
    allowed = determine_allowed_tools(
        state,
        MagicMock(),
        DEFAULT_TOOLS,
        has_compile_error=True,
        validation_error="Schema validation failed: DEAD_SQL_ALIAS",
    )

    assert allowed == frozenset({"decision_edit"})


def test_local_validation_recovery_reopens_retrieval_for_stale_evidence() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        stale=("schema.sql",),
    )
    allowed = determine_allowed_tools(
        state,
        MagicMock(),
        DEFAULT_TOOLS,
        validation_error="Schema validation failed: DEAD_SQL_ALIAS",
    )

    assert allowed == frozenset({"decision_edit", "view_symbol_code"})


def test_missing_dependency_recovery_can_retrieve_with_missing_evidence() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.INSUFFICIENT,
        missing=("target_implementation",),
    )
    allowed = determine_allowed_tools(
        state,
        MagicMock(),
        DEFAULT_TOOLS,
        validation_error="NameError: helper is not defined",
    )

    assert allowed == frozenset({"decision_edit", "grep_search", "view_symbol_code", "codebase_retrieve"})


def test_bootstrap_observed_only_allows_edit_not_retrieval() -> None:
    """Observed-only bootstrap at SUFFICIENT_FOR_EDIT: edit only, no grep/view."""
    schema = EvidenceItem(
        id="observed.schema:db/init/init.sql:ticket_order",
        need="schema",
        type="schema",
        role="observed",
        file="db/init/init.sql",
        span=(99, 110),
        symbol="ticket_order",
        status="SATISFIED",
    )
    symbol = EvidenceItem(
        id="observed.symbol:list.py:order_timeline",
        need="handler",
        type="symbol",
        role="observed",
        file="list.py",
        span=(346, 400),
        symbol="order_timeline",
        status="SATISFIED",
    )
    manifest = StepManifest(
        required_items=(schema, symbol),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    run_state = SimpleNamespace(
        task_mode="edit",
        manifest=manifest,
        validation=SimpleNamespace(status="not_run"),
        changes=SimpleNamespace(files=()),
        retrieval_no_gain_rounds=0,
        view_last_round_all_duplicate=False,
    )
    state = SimpleNamespace(run_state=run_state)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_edit_burst_after_validated_edit_is_edit_only() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("db/init/init.sql",),
        validation_status="passed",
        phase=RunPhase.ACTING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_edit_burst_continues_edit_only_when_edited_file_is_stale() -> None:
    """Self-inflicted staleness on an edited file must not reopen retrieval mid-plan."""
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("db/init/init.sql",),
        validation_status="passed",
        stale=("db/init/init.sql",),
        phase=RunPhase.ACTING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_edit_burst_reopens_view_for_external_stale_file() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("db/init/init.sql",),
        validation_status="passed",
        stale=("list.py",),
        phase=RunPhase.ACTING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "decision_edit" in allowed
    assert "view_symbol_code" in allowed


def test_edit_burst_reopens_retrieval_on_validation_failure() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.INSUFFICIENT,
        changed_files=("db/init/init.sql",),
        validation_status="failed",
        stale=("db/init/init.sql",),
        phase=RunPhase.ACTING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "decision_edit" in allowed
    assert "view_symbol_code" in allowed


def test_edit_patch_failed_reopens_retrieval_during_edit_burst() -> None:
    """After a successful edit, a failed patch must reopen grep/view for recovery."""
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
                    symbol="order_timeline",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="passed"),
        changes=SimpleNamespace(files=("db/init/init.sql",)),
        retrieval_no_gain_rounds=0,
        view_last_round_all_duplicate=False,
        edit_patch_failed=True,
    )
    state = SimpleNamespace(run_state=run_state)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit", "grep_search", "view_symbol_code"})


def test_duplicate_view_round_closes_retrieval_when_edit_ready() -> None:
    run_state = SimpleNamespace(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="observed.symbol:list.py:build_router",
                    need="handler",
                    type="symbol",
                    role="observed",
                    file="list.py",
                    span=(16, 350),
                    symbol="build_router",
                    status="SATISFIED",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="not_run"),
        changes=SimpleNamespace(files=()),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=True,
        edit_patch_failed=False,
    )
    state = SimpleNamespace(run_state=run_state)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_duplicate_view_overrides_edit_patch_failed_recovery() -> None:
    run_state = SimpleNamespace(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="observed.symbol:list.py:build_router",
                    need="handler",
                    type="symbol",
                    role="observed",
                    file="list.py",
                    span=(16, 350),
                    symbol="build_router",
                    status="SATISFIED",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="not_run"),
        changes=SimpleNamespace(files=()),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=True,
        edit_patch_failed=True,
    )
    state = SimpleNamespace(run_state=run_state)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_duplicate_view_closes_retrieval_even_with_stale_manifest_items() -> None:
    """Duplicate view round must not stay open because unrelated STALE rows exist."""
    run_state = SimpleNamespace(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="observed.schema:db/init/init.sql:airport_info",
                    need="schema",
                    type="schema",
                    role="observed",
                    file="db/init/init.sql",
                    span=(7, 14),
                    symbol="airport_info",
                    status="STALE",
                    stale_reason="file modified by decision_edit",
                ),
                EvidenceItem(
                    id="observed.symbol:list.py:build_router",
                    need="handler",
                    type="symbol",
                    role="observed",
                    file="list.py",
                    span=(16, 350),
                    symbol="build_router",
                    status="SATISFIED",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="not_run"),
        changes=SimpleNamespace(files=()),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=True,
        edit_patch_failed=False,
    )
    state = SimpleNamespace(run_state=run_state)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_verification_mode_reopens_retrieval_on_responding() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("db/init/init.sql", "list.py"),
        validation_status="passed",
        phase=RunPhase.RESPONDING,
    )
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "grep_search" in allowed
    assert "view_symbol_code" in allowed
    assert "decision_edit" in allowed


def _post_edit_run_state(**overrides: object) -> SimpleNamespace:
    base = dict(
        task_mode="edit",
        phase=RunPhase.ACTING,
        manifest=StepManifest(
            required_items=(
                EvidenceItem(
                    id="observed.symbol:list.py:build_router",
                    need="handler",
                    type="symbol",
                    role="observed",
                    file="list.py",
                    span=(16, 350),
                    symbol="build_router",
                    status="SATISFIED",
                ),
            ),
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        ),
        validation=SimpleNamespace(status="passed"),
        changes=SimpleNamespace(files=("db/init/init.sql", "list.py")),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=False,
        edit_patch_failed=False,
        rounds_since_last_edit=1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_post_edit_verification_opens_view_after_non_edit_round() -> None:
    """A non-duplicate non-edit round after plan edits reopens verification tools."""
    run_state = _post_edit_run_state()
    state = SimpleNamespace(run_state=run_state, checklist=())
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "view_symbol_code" in allowed
    assert "grep_search" in allowed
    assert "decision_edit" in allowed


def test_verification_converges_to_edit_only_on_duplicate_view() -> None:
    """Once verification view rounds replay cached evidence, converge to edit/answer."""
    run_state = _post_edit_run_state(view_last_round_all_duplicate=True)
    state = SimpleNamespace(run_state=run_state, checklist=())
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_post_edit_verification_opens_when_checklist_complete() -> None:
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
        changes=SimpleNamespace(files=("db/init/init.sql", "list.py")),
        edit_patch_failed=False,
        rounds_since_last_edit=0,
    )
    checklist = (
        "[√] Add order_timeline table",
        "[√] Implement order_timeline endpoint in list.py",
    )
    state = SimpleNamespace(run_state=run_state, checklist=checklist)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert "view_symbol_code" in allowed
    assert allowed != frozenset({"decision_edit"})


def test_edit_burst_stays_edit_only_when_plan_still_open() -> None:
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
    )
    checklist = (
        "[√] Add order_timeline table",
        "[ ] Implement order_timeline endpoint in list.py",
    )
    state = SimpleNamespace(run_state=run_state, checklist=checklist)
    allowed = determine_allowed_tools(state, MagicMock(), DEFAULT_TOOLS)

    assert allowed == frozenset({"decision_edit"})


def test_missing_dependency_does_not_reopen_retrieval_without_manifest_gap() -> None:
    state = _make_state(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        changed_files=("list.py",),
    )
    allowed = determine_allowed_tools(
        state,
        MagicMock(),
        DEFAULT_TOOLS,
        validation_error="NameError: helper is not defined",
    )

    assert allowed == frozenset({"decision_edit"})
