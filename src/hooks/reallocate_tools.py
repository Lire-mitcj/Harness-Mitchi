from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.checklist import checklist_item_open, checklist_plan_complete
from src.agent.manifest import (
    StepManifest,
    Sufficiency,
    grep_pending_caller_loads,
    manifest_metrics,
    retrieval_profile,
    stale_needing_refresh,
    wiring_gap_lines,
    _norm_file,
)
from src.agent.run_state import RunPhase
from src.hooks.retrieval_convergence import (
    HEAVY_RETRIEVAL_TOOLS,
    PRIMARY_RETRIEVAL_TOOLS,
    RETRIEVAL_TOOLS,
)
from src.hooks.tool_scorer import (
    DEFAULT_TEMPERATURE,
    ToolAllocation,
    ToolScoringContext,
    allocate,
)
from src.tools.grep_match_symbols import has_actionable_suggested_views

if TYPE_CHECKING:
    from src.agent.state_assembled_loop import AssembledState


_EDIT_TOOLS = frozenset({"decision_edit"})
_GREP_TOOLS = frozenset({"grep_search"})
_VIEW_TOOLS = frozenset({"view_symbol_code"})

# After a landed edit, cap how many consecutive non-edit (retrieval) rounds the
# model may take while evidence is already sufficient. Without a checklist, the
# post-edit heuristic reopens retrieval and the model keeps loading *new* symbols
# every round, so no_gain/duplicate saturation never fires — it explores forever.
# This bounds that loop and forces the next edit.
_MAX_RETRIEVAL_ROUNDS_AFTER_EDIT = 3


def _split_tool_groups(default_tools: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    edit = _EDIT_TOOLS & default_tools
    retrieval = RETRIEVAL_TOOLS & default_tools
    return edit, retrieval


def _retrieval_tools_from_profile(
    manifest: StepManifest | None,
    default_tools: frozenset[str],
    *,
    allow_heavy: bool,
    task_text: str = "",
) -> frozenset[str]:
    """Map manifest gaps to the minimal justified retrieval tool set."""
    if manifest is None:
        return PRIMARY_RETRIEVAL_TOOLS & default_tools

    profile = retrieval_profile(manifest, task_text=task_text)
    tools: set[str] = set()
    if profile.needs_grep:
        tools.add("grep_search")
    if profile.needs_view:
        tools.add("view_symbol_code")
    if profile.needs_heavy and allow_heavy:
        tools.add("codebase_retrieve")
    return frozenset(tools) & default_tools


def _allow_heavy_retrieval(
    manifest: StepManifest | None,
    *,
    sufficiency: str,
    no_gain_rounds: int,
    task_text: str = "",
) -> bool:
    """Heavy semantic retrieval is a bootstrap / stuck-discovery fallback only."""
    if manifest is None or sufficiency != Sufficiency.INSUFFICIENT:
        return False
    profile = retrieval_profile(manifest, task_text=task_text)
    if not profile.needs_heavy:
        return False
    if profile.bootstrap:
        return True
    return no_gain_rounds >= 2


def _apply_retrieval_saturation(
    retrieval: frozenset[str],
    default_tools: frozenset[str],
    *,
    manifest: StepManifest | None,
    no_gain_rounds: int,
    view_last_round_all_duplicate: bool,
    has_open_gaps: bool,
    wiring_gap_open: bool = False,
) -> frozenset[str]:
    """Throttle exploratory retrieval once manifest obligations are satisfied."""
    tools = set(retrieval) - HEAVY_RETRIEVAL_TOOLS

    if has_open_gaps:
        if no_gain_rounds >= 2 and view_last_round_all_duplicate:
            return frozenset()
        if view_last_round_all_duplicate:
            tools -= _VIEW_TOOLS
            if wiring_gap_open:
                tools |= _GREP_TOOLS
        return frozenset(tools) & default_tools

    if no_gain_rounds >= 2:
        return frozenset()
    if no_gain_rounds >= 1:
        if view_last_round_all_duplicate:
            tools -= _VIEW_TOOLS
            tools |= _GREP_TOOLS
        return frozenset(tools) & default_tools
    return frozenset(tools) & default_tools


def _insufficient_retrieval_tools(
    manifest: StepManifest | None,
    default_tools: frozenset[str],
    *,
    no_gain_rounds: int,
    task_text: str = "",
) -> frozenset[str]:
    """Retrieval-only phase: manifest gaps plus the primary grep→view workflow."""
    profile_tools = _retrieval_tools_from_profile(
        manifest,
        default_tools,
        allow_heavy=_allow_heavy_retrieval(
            manifest,
            sufficiency=Sufficiency.INSUFFICIENT,
            no_gain_rounds=no_gain_rounds,
            task_text=task_text,
        ),
        task_text=task_text,
    )
    return profile_tools | (PRIMARY_RETRIEVAL_TOOLS & default_tools)


def _has_grounded_observations(manifest: StepManifest | None) -> bool:
    """True when bootstrap observed memory has at least one SATISFIED/STALE anchor."""
    if manifest is None:
        return False
    return any(
        item.role == "observed" and item.status in {"SATISFIED", "STALE"}
        for item in manifest.required_items
    )


def _has_wiring_gap(
    manifest: StepManifest | None,
    *,
    task_text: str = "",
) -> bool:
    return (
        bool(wiring_gap_lines(manifest, task_text=task_text))
        if manifest is not None
        else False
    )


def _task_text_from_run_state(run_state: Any) -> str:
    return str(getattr(run_state, "task_text", "") or "").strip()


def _insufficient_allowed_tools(
    *,
    task_mode: str,
    manifest: StepManifest | None,
    default_tools: frozenset[str],
    edit_tools: frozenset[str],
    no_gain_rounds: int,
    view_last_round_all_duplicate: bool,
    has_missing: bool,
    has_stale: bool,
    grep_suggested_views: tuple[Any, ...] = (),
    task_text: str = "",
) -> frozenset[str]:
    """INSUFFICIENT phase with saturation and duplicate-round convergence."""
    sufficiency = str(getattr(manifest, "sufficiency", Sufficiency.INSUFFICIENT))

    if (
        view_last_round_all_duplicate
        and task_mode == "edit"
        and sufficiency in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
    ):
        # wiring_gap is advisory once evidence is already sufficient.
        return edit_tools or default_tools

    retrieval = _insufficient_retrieval_tools(
        manifest,
        default_tools,
        no_gain_rounds=no_gain_rounds,
        task_text=task_text,
    )
    if no_gain_rounds == 0 and (has_missing or has_stale):
        return retrieval or default_tools

    saturated = _apply_retrieval_saturation(
        retrieval,
        default_tools,
        manifest=manifest,
        no_gain_rounds=no_gain_rounds,
        view_last_round_all_duplicate=view_last_round_all_duplicate,
        has_open_gaps=has_missing or has_stale,
    )
    if task_mode == "edit" and no_gain_rounds >= 2:
        if sufficiency in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }:
            return edit_tools or default_tools
        # Partial bootstrap: prefer view, but reopen grep when view is stuck
        # (duplicate replay, no actionable suggested_views, or symbol-not-found loop).
        if view_last_round_all_duplicate:
            return (_GREP_TOOLS & default_tools) or default_tools
        if not has_actionable_suggested_views(grep_suggested_views):
            return (PRIMARY_RETRIEVAL_TOOLS & default_tools) or default_tools
        return (_VIEW_TOOLS & default_tools) or default_tools
    return saturated or default_tools


def _recovery_allowed_tools(
    *,
    task_mode: str,
    manifest: StepManifest | None,
    default_tools: frozenset[str],
    edit_tools: frozenset[str],
    has_missing: bool,
    has_stale: bool,
    no_gain_rounds: int,
    view_last_round_all_duplicate: bool,
    needs_retrieval: bool,
    task_text: str = "",
) -> frozenset[str]:
    """Edit recovery after validation/tool failures, with duplicate-round convergence."""
    if not needs_retrieval:
        return edit_tools or default_tools
    if view_last_round_all_duplicate and task_mode == "edit":
        return edit_tools or default_tools
    if manifest is not None and manifest.missing_items:
        retrieval = _insufficient_retrieval_tools(
            manifest,
            default_tools,
            no_gain_rounds=no_gain_rounds,
            task_text=task_text,
        )
        if no_gain_rounds == 0 and not view_last_round_all_duplicate:
            return frozenset(edit_tools | retrieval) or default_tools
    else:
        retrieval = _retrieval_tools_from_profile(
            manifest,
            default_tools,
            allow_heavy=False,
            task_text=task_text,
        )
    saturated = _apply_retrieval_saturation(
        retrieval,
        default_tools,
        manifest=manifest,
        no_gain_rounds=no_gain_rounds,
        view_last_round_all_duplicate=view_last_round_all_duplicate,
        has_open_gaps=has_missing or has_stale,
    )
    return frozenset(edit_tools | saturated) or default_tools


def _validation_recovery_tools(
    default_tools: frozenset[str],
    manifest: StepManifest | None,
    *,
    needs_evidence_refresh: bool,
    no_gain_rounds: int,
    view_last_round_all_duplicate: bool = False,
    task_mode: str = "edit",
    has_missing: bool = False,
    has_stale: bool = False,
    task_text: str = "",
) -> frozenset[str]:
    """Allow repair, adding retrieval only for manifest-declared gaps."""
    edit, _ = _split_tool_groups(default_tools)
    return _recovery_allowed_tools(
        task_mode=task_mode,
        manifest=manifest,
        default_tools=default_tools,
        edit_tools=edit,
        has_missing=has_missing,
        has_stale=has_stale,
        no_gain_rounds=no_gain_rounds,
        view_last_round_all_duplicate=view_last_round_all_duplicate,
        needs_retrieval=needs_evidence_refresh,
        task_text=task_text,
    )


def _manifest_state(run_state: Any) -> tuple[str, bool, bool]:
    """Return (sufficiency, has_missing, has_actionable_stale).

    Stale taxonomy is owned by ``tool_gate.actionable_stale`` — self-stale and
    edit_ready advisory stale must not reopen retrieval.
    """
    from src.hooks.tool_gate import has_actionable_stale

    manifest = getattr(run_state, "manifest", None)
    sufficiency = str(getattr(manifest, "sufficiency", Sufficiency.INSUFFICIENT))
    has_missing = bool(getattr(manifest, "missing_items", ()))
    if manifest is None:
        return sufficiency, has_missing, False
    return (
        sufficiency,
        has_missing,
        has_actionable_stale(
            run_state,
            manifest,
            sufficiency=sufficiency,
            has_missing=has_missing,
        ),
    )


def _blocks_edit(
    *,
    has_missing: bool,
    metrics: Any,
) -> bool:
    if has_missing:
        return True
    if metrics is None:
        return False
    return metrics.critical_missing


def _has_retrieval_gaps(
    *,
    has_missing: bool,
    has_stale: bool,
    metrics: Any,
) -> bool:
    if has_missing or has_stale:
        return True
    if metrics is None:
        return False
    return metrics.coverage < 1.0 or metrics.critical_missing


def _edit_allowed(
    *,
    task_mode: str,
    sufficiency: str,
    has_missing: bool,
    metrics: Any,
) -> bool:
    if task_mode != "edit":
        return False
    if sufficiency not in {
        Sufficiency.SUFFICIENT_FOR_EDIT,
        Sufficiency.SUFFICIENT_FOR_VERIFY,
    }:
        return False
    return not _blocks_edit(has_missing=has_missing, metrics=metrics)


def _checklist_has_open_items(checklist: tuple[str, ...]) -> bool:
    return bool(checklist) and any(checklist_item_open(item) for item in checklist)


def _plan_edits_complete(
    run_state: Any,
    *,
    checklist: tuple[str, ...] = (),
) -> bool:
    """True when a non-empty checklist is fully done and no partner gaps remain.

    An empty checklist must not mean "plan complete" — Core often keeps the plan
    only in prose; assuming done after the first edit reopens verification too early.
    """
    if not checklist or _checklist_has_open_items(checklist):
        return False
    manifest = getattr(run_state, "manifest", None)
    changes = getattr(run_state, "changes", None)
    edited_files = getattr(changes, "files", ()) if changes is not None else ()
    if not edited_files:
        return False
    if manifest is not None and manifest.missing_items:
        return False
    task_text = _task_text_from_run_state(run_state)
    if _has_wiring_gap(manifest, task_text=task_text):
        return False
    if _external_stale_blocks_edit_burst(manifest, edited_files):
        return False
    return True


def _self_stale_only(run_state: Any, manifest: Any) -> bool:
    """True when every stale-needing-refresh item is on an already-edited file."""
    from src.hooks.tool_gate import self_stale_only

    return self_stale_only(run_state, manifest)


def _post_edit_verification_ready(
    run_state: Any,
    *,
    checklist: tuple[str, ...] = (),
) -> bool:
    """Reopen retrieval after planned edits finish — never mid open-checklist burst."""
    if str(getattr(run_state, "task_mode", "")) != "edit":
        return False
    phase = getattr(run_state, "phase", None)
    if phase == RunPhase.RESPONDING:
        return True
    changes = getattr(run_state, "changes", None)
    edited_files = getattr(changes, "files", ()) if changes is not None else ()
    if not edited_files:
        return False
    validation = getattr(run_state, "validation", None)
    if str(getattr(validation, "status", "not_run") or "not_run") != "passed":
        return False
    if getattr(run_state, "edit_patch_failed", False):
        return False
    # Open checklist items mean the multi-step plan is still editing — keep
    # edit_burst; do not reopen grep/view just because rounds_since_last_edit>0.
    if _checklist_has_open_items(checklist):
        return False
    if checklist_plan_complete(checklist):
        return True
    if _plan_edits_complete(run_state, checklist=checklist):
        return True
    # No structured checklist: allow next-step discovery after one non-edit round.
    rounds = int(getattr(run_state, "rounds_since_last_edit", 0) or 0)
    return rounds >= 1 and not checklist


def _external_stale_blocks_edit_burst(manifest: Any, edited_files: tuple[str, ...]) -> bool:
    """True when stale evidence on not-yet-edited files should reopen retrieval."""
    if manifest is None:
        return False
    edited = {_norm_file(path) for path in edited_files if path}
    for item in stale_needing_refresh(manifest):
        if item.file and _norm_file(item.file) in edited:
            continue
        return True
    return False


def _edit_burst_active(
    run_state: Any,
    *,
    validation_error: str | None,
    has_compile_error: bool,
) -> bool:
    """After a validated edit landed, keep the loop in edit-only mode.

    Retrieval reopens on validation/tool failures or explicit verification phase.
    """
    if validation_error or has_compile_error:
        return False
    if str(getattr(run_state, "task_mode", "")) != "edit":
        return False
    phase = getattr(run_state, "phase", None)
    if phase == RunPhase.RESPONDING:
        return False
    changes = getattr(run_state, "changes", None)
    edited_files = getattr(changes, "files", ()) if changes is not None else ()
    if not edited_files:
        return False
    validation = getattr(run_state, "validation", None)
    status = str(getattr(validation, "status", "not_run") or "not_run")
    if status != "passed":
        return False
    manifest = getattr(run_state, "manifest", None)
    if manifest is not None and manifest.failure_items:
        return False
    # Partner STALE is advisory under edit_ready (tool_gate.actionable_stale).
    # Do not kill edit_burst solely because another observed file was marked STALE.
    if _external_stale_blocks_edit_burst(manifest, edited_files):
        # Only block burst when sufficiency is still INSUFFICIENT.
        if manifest is not None and str(manifest.sufficiency) == Sufficiency.INSUFFICIENT:
            return False
    return True


def _verification_mode_active(
    run_state: Any,
    *,
    checklist: tuple[str, ...] = (),
) -> bool:
    """Post-plan verification: retrieval allowed again before final answer."""
    if _post_edit_verification_ready(run_state, checklist=checklist):
        return True
    return (
        str(getattr(run_state, "task_mode", "")) == "edit"
        and getattr(run_state, "phase", None) == RunPhase.RESPONDING
    )


def determine_allowed_tools(
    state: AssembledState | Any,
    gravity_controller: Any,
    default_tools: frozenset[str],
    has_compile_error: bool = False,
    *,
    validation_error: str | None = None,
) -> frozenset[str]:
    """Allocate capabilities from manifest gaps plus saturation signals.

    Priority (top wins):

      (A) validation / compile failure -> edit (+ manifest-justified refresh)
      (B) manifest failure items     -> edit (+ refresh when stale/missing)
      (C) INSUFFICIENT               -> manifest-justified retrieval only
      (D) diagnose + sufficient      -> no tools only once phase is RESPONDING
      (E) sufficient + no missing/stale -> edit only (wiring_gap is advisory)
      (F) open missing/stale gaps    -> retrieval profile (+ edit when allowed)
    """
    run_state = getattr(state, "run_state", None)
    checklist = tuple(getattr(state, "checklist", ()) or ())
    _ = gravity_controller
    task_text = _task_text_from_run_state(run_state) if run_state is not None else ""
    task_mode = str(getattr(run_state, "task_mode", "diagnose") or "diagnose")
    edit_tools, _ = _split_tool_groups(default_tools)
    manifest = getattr(run_state, "manifest", None)
    metrics = manifest_metrics(manifest) if manifest is not None else None
    sufficiency, has_missing, has_stale = _manifest_state(run_state)
    no_gain_rounds = int(getattr(run_state, "retrieval_no_gain_rounds", 0) or 0)
    view_last_round_all_duplicate = bool(
        getattr(run_state, "view_last_round_all_duplicate", False)
    )
    grep_suggested_views = tuple(
        getattr(run_state, "grep_suggested_views", ()) or ()
    )
    has_retrieval_gaps = _has_retrieval_gaps(
        has_missing=has_missing,
        has_stale=has_stale,
        metrics=metrics,
    )
    blocks_edit = _blocks_edit(has_missing=has_missing, metrics=metrics)

    validation = getattr(run_state, "validation", None)
    validation_status = str(getattr(validation, "status", "not_run") or "not_run")
    concrete_error = (validation_error or "").strip()

    if concrete_error or has_compile_error:
        # Validation/compile recovery may re-read even advisory STALE anchors.
        raw_stale = bool(
            stale_needing_refresh(manifest) if manifest is not None else ()
        )
        return _validation_recovery_tools(
            default_tools,
            manifest,
            needs_evidence_refresh=(has_missing or has_stale or raw_stale),
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            task_mode=task_mode,
            has_missing=has_missing,
            has_stale=has_stale or raw_stale,
            task_text=task_text,
        )

    if manifest is not None and manifest.failure_items:
        return _recovery_allowed_tools(
            task_mode=task_mode,
            manifest=manifest,
            default_tools=default_tools,
            edit_tools=edit_tools,
            has_missing=has_missing,
            has_stale=has_stale,
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            needs_retrieval=has_missing or has_stale,
            task_text=task_text,
        )

    if validation_status == "failed":
        # Validation failure always warrants refresh — including re-reading the
        # file we just patched (self-stale is still actionable here).
        return _recovery_allowed_tools(
            task_mode=task_mode,
            manifest=manifest,
            default_tools=default_tools,
            edit_tools=edit_tools,
            has_missing=has_missing,
            has_stale=True,
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            needs_retrieval=True,
            task_text=task_text,
        )

    # Bounded retrieval after a landed edit. When evidence is already sufficient
    # and the plan is not verifiably complete, do not let the model reopen
    # retrieval for unbounded rounds: it keeps loading NEW symbols each round so
    # the no_gain / duplicate saturation never fires. Force edit-only to make
    # progress. A completed plan (or RESPONDING phase) still gets verification.
    rounds_since_last_edit = int(
        getattr(run_state, "rounds_since_last_edit", 0) or 0
    )
    phase = getattr(run_state, "phase", None)
    if (
        task_mode == "edit"
        and phase != RunPhase.RESPONDING
        and rounds_since_last_edit >= _MAX_RETRIEVAL_ROUNDS_AFTER_EDIT
        and not has_missing
        and not has_stale
        and not blocks_edit
        and not checklist_plan_complete(checklist)
        and _edit_allowed(
            task_mode=task_mode,
            sufficiency=sufficiency,
            has_missing=has_missing,
            metrics=metrics,
        )
    ):
        return edit_tools or default_tools

    # Post-plan verification reopens retrieval, but yields to duplicate-round
    # convergence: once the last view round replayed only cached evidence, the
    # LLM has finished verifying — drop to edit-only so it answers instead of
    # looping re-views against fact-locking BLOCKs.
    if (
        _verification_mode_active(run_state, checklist=checklist)
        and not view_last_round_all_duplicate
    ):
        return frozenset(
            (PRIMARY_RETRIEVAL_TOOLS | _EDIT_TOOLS) & default_tools
        ) or default_tools

    if (
        sufficiency in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
        and not has_missing
        and view_last_round_all_duplicate
        and _edit_allowed(
            task_mode=task_mode,
            sufficiency=sufficiency,
            has_missing=has_missing,
            metrics=metrics,
        )
    ):
        # Duplicate replay after sufficient evidence → edit only.
        # wiring_gap / grep_pending are advisory and must not reopen retrieval.
        return edit_tools or default_tools

    if getattr(run_state, "edit_patch_failed", False):
        return frozenset(
            (edit_tools | PRIMARY_RETRIEVAL_TOOLS) & default_tools
        ) or default_tools

    if _edit_burst_active(
        run_state,
        validation_error=concrete_error,
        has_compile_error=has_compile_error,
    ):
        return edit_tools or default_tools

    if sufficiency == Sufficiency.INSUFFICIENT:
        return _insufficient_allowed_tools(
            task_mode=task_mode,
            manifest=manifest,
            default_tools=default_tools,
            edit_tools=edit_tools,
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            has_missing=has_missing,
            has_stale=has_stale,
            grep_suggested_views=grep_suggested_views,
            task_text=task_text,
        )

    if task_mode == "diagnose":
        if has_missing:
            return _insufficient_allowed_tools(
                task_mode=task_mode,
                manifest=manifest,
                default_tools=default_tools,
                edit_tools=edit_tools,
                no_gain_rounds=no_gain_rounds,
                view_last_round_all_duplicate=view_last_round_all_duplicate,
                has_missing=has_missing,
                has_stale=has_stale,
                grep_suggested_views=grep_suggested_views,
                task_text=task_text,
            )
        if has_stale:
            return (
                _retrieval_tools_from_profile(
                    manifest,
                    default_tools,
                    allow_heavy=False,
                    task_text=task_text,
                )
                or default_tools
            )
        phase = getattr(run_state, "phase", None)
        if phase != RunPhase.RESPONDING:
            if sufficiency == Sufficiency.INSUFFICIENT:
                return _insufficient_allowed_tools(
                    task_mode=task_mode,
                    manifest=manifest,
                    default_tools=default_tools,
                    edit_tools=edit_tools,
                    no_gain_rounds=no_gain_rounds,
                    view_last_round_all_duplicate=view_last_round_all_duplicate,
                    has_missing=has_missing,
                    has_stale=has_stale,
                    grep_suggested_views=grep_suggested_views,
                    task_text=task_text,
                )
            retrieval = PRIMARY_RETRIEVAL_TOOLS & default_tools
            return _apply_retrieval_saturation(
                retrieval,
                default_tools,
                manifest=manifest,
                no_gain_rounds=no_gain_rounds,
                view_last_round_all_duplicate=view_last_round_all_duplicate,
                has_open_gaps=False,
            ) or default_tools
        return frozenset()

    # Scheme B: sufficient + no missing/stale => edit-only. wiring_gap and
    # grep_pending stay on the STEP EVIDENCE card as soft hints only.
    edit_ready_closed = (
        sufficiency in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
        and not has_missing
        and not has_stale
        and not blocks_edit
        and _edit_allowed(
            task_mode=task_mode,
            sufficiency=sufficiency,
            has_missing=has_missing,
            metrics=metrics,
        )
    )
    if edit_ready_closed:
        return edit_tools or default_tools

    retrieval = _retrieval_tools_from_profile(
        manifest,
        default_tools,
        allow_heavy=False,
        task_text=task_text,
    )
    if not blocks_edit and not has_missing:
        # Only missing/stale reopen retrieval; wiring_gap is not a tool gate.
        retrieval = _apply_retrieval_saturation(
            retrieval or (PRIMARY_RETRIEVAL_TOOLS & default_tools),
            default_tools,
            manifest=manifest,
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            has_open_gaps=has_stale,
            wiring_gap_open=False,
        )

    allowed = set(retrieval)
    if _edit_allowed(
        task_mode=task_mode,
        sufficiency=sufficiency,
        has_missing=has_missing,
        metrics=metrics,
    ):
        allowed.update(edit_tools)
    elif blocks_edit or (has_retrieval_gaps and sufficiency == Sufficiency.INSUFFICIENT):
        return (
            _insufficient_retrieval_tools(
                manifest,
                default_tools,
                no_gain_rounds=no_gain_rounds,
                task_text=task_text,
            )
            or default_tools
        )

    return frozenset(allowed) or default_tools


def post_edit_verification_ready(
    run_state: Any,
    *,
    checklist: tuple[str, ...] = (),
) -> bool:
    """Public wrapper for execution-card and loop integration."""
    return _post_edit_verification_ready(run_state, checklist=checklist)


def _build_scoring_context(
    run_state: Any,
    *,
    checklist: tuple[str, ...],
    has_compile_error: bool,
    validation_error: str | None,
) -> ToolScoringContext:
    """Flatten manifest + runtime signals into the tool scorer's input."""
    task_mode = str(getattr(run_state, "task_mode", "diagnose") or "diagnose")
    task_text = _task_text_from_run_state(run_state)
    manifest = getattr(run_state, "manifest", None)
    metrics = manifest_metrics(manifest) if manifest is not None else None
    profile = (
        retrieval_profile(manifest, task_text=task_text)
        if manifest is not None
        else None
    )
    sufficiency, has_missing, has_stale = _manifest_state(run_state)
    grep_suggested_views = tuple(
        getattr(run_state, "grep_suggested_views", ()) or ()
    )
    grep_pending = (
        bool(grep_pending_caller_loads(manifest, grep_suggested_views))
        if manifest is not None
        else False
    )
    validation = getattr(run_state, "validation", None)
    validation_status = str(getattr(validation, "status", "not_run") or "not_run")
    concrete_error = bool((validation_error or "").strip()) or has_compile_error
    phase = getattr(run_state, "phase", None)
    has_failures = bool(getattr(manifest, "failure_items", ()))

    # Mirror the gate waterfall precedence: verification reopens retrieval ahead of
    # edit_burst, so the two phase flags must be mutually exclusive for the scorer
    # (otherwise edit_burst's edit-bias would drown out verification's view-bias).
    verification_mode = _verification_mode_active(run_state, checklist=checklist)
    edit_burst = _edit_burst_active(
        run_state,
        validation_error=(validation_error or "").strip(),
        has_compile_error=has_compile_error,
    ) and not verification_mode

    return ToolScoringContext(
        task_mode=task_mode,
        is_responding=phase == RunPhase.RESPONDING,
        sufficiency=sufficiency,
        coverage=metrics.coverage if metrics is not None else 0.0,
        missing_ratio=metrics.missing_ratio if metrics is not None else 0.0,
        stale_ratio=metrics.stale_ratio if metrics is not None else 0.0,
        critical_missing=bool(metrics.critical_missing) if metrics is not None else False,
        has_missing=has_missing,
        has_stale=has_stale,
        wiring_gap=_has_wiring_gap(manifest, task_text=task_text),
        grep_pending=grep_pending,
        validation_failed=validation_status == "failed" or concrete_error,
        edit_burst=edit_burst,
        verification_mode=verification_mode,
        no_gain_rounds=int(getattr(run_state, "retrieval_no_gain_rounds", 0) or 0),
        view_last_round_all_duplicate=bool(
            getattr(run_state, "view_last_round_all_duplicate", False)
        ),
        edit_patch_failed=bool(getattr(run_state, "edit_patch_failed", False)),
        has_failures=has_failures,
        actionable_suggested_views=has_actionable_suggested_views(grep_suggested_views),
        bootstrap=bool(profile.bootstrap) if profile is not None else False,
    )


def allocate_tools(
    state: AssembledState | Any,
    gravity_controller: Any,
    default_tools: frozenset[str],
    has_compile_error: bool = False,
    *,
    validation_error: str | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ToolAllocation:
    """Gate tools via the existing waterfall, then rank them with the scorer.

    ``determine_allowed_tools`` is the hard-gate authority (safety constraints),
    so its result becomes ``allowed``. The probabilistic scorer then ranks that
    gated set so the Core LLM is steered toward the highest-probability tool.
    """
    allowed = determine_allowed_tools(
        state,
        gravity_controller,
        default_tools,
        has_compile_error=has_compile_error,
        validation_error=validation_error,
    )
    run_state = getattr(state, "run_state", None)
    if run_state is None:
        return ToolAllocation(allowed=allowed, temperature=temperature)
    checklist = tuple(getattr(state, "checklist", ()) or ())
    ctx = _build_scoring_context(
        run_state,
        checklist=checklist,
        has_compile_error=has_compile_error,
        validation_error=validation_error,
    )
    return allocate(ctx, allowed, temperature=temperature)
