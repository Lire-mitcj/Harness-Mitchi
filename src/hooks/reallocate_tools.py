from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.agent.checklist import checklist_plan_complete
from src.agent.manifest import (
    StepManifest,
    Sufficiency,
    manifest_metrics,
    retrieval_profile,
    stale_needing_refresh,
    _norm_file,
)
from src.agent.run_state import RunPhase
from src.hooks.retrieval_convergence import (
    HEAVY_RETRIEVAL_TOOLS,
    PRIMARY_RETRIEVAL_TOOLS,
    RETRIEVAL_TOOLS,
)

if TYPE_CHECKING:
    from src.agent.state_assembled_loop import AssembledState


_EDIT_TOOLS = frozenset({"decision_edit"})
_GREP_TOOLS = frozenset({"grep_search"})
_VIEW_TOOLS = frozenset({"view_symbol_code"})


def _split_tool_groups(default_tools: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
    edit = _EDIT_TOOLS & default_tools
    retrieval = RETRIEVAL_TOOLS & default_tools
    return edit, retrieval


def _retrieval_tools_from_profile(
    manifest: StepManifest | None,
    default_tools: frozenset[str],
    *,
    allow_heavy: bool,
) -> frozenset[str]:
    """Map manifest gaps to the minimal justified retrieval tool set."""
    if manifest is None:
        return PRIMARY_RETRIEVAL_TOOLS & default_tools

    profile = retrieval_profile(manifest)
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
) -> bool:
    """Heavy semantic retrieval is a bootstrap / stuck-discovery fallback only."""
    if manifest is None or sufficiency != Sufficiency.INSUFFICIENT:
        return False
    profile = retrieval_profile(manifest)
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
) -> frozenset[str]:
    """Throttle exploratory retrieval once manifest obligations are satisfied."""
    tools = set(retrieval) - HEAVY_RETRIEVAL_TOOLS

    if has_open_gaps:
        profile = retrieval_profile(manifest) if manifest is not None else None
        if (
            view_last_round_all_duplicate
            and profile is not None
            and not profile.needs_view
        ):
            tools -= _VIEW_TOOLS
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
) -> frozenset[str]:
    """Retrieval-only phase: manifest gaps plus the primary grep→view workflow."""
    profile_tools = _retrieval_tools_from_profile(
        manifest,
        default_tools,
        allow_heavy=_allow_heavy_retrieval(
            manifest,
            sufficiency=Sufficiency.INSUFFICIENT,
            no_gain_rounds=no_gain_rounds,
        ),
    )
    return profile_tools | (PRIMARY_RETRIEVAL_TOOLS & default_tools)


def _validation_recovery_tools(
    default_tools: frozenset[str],
    manifest: StepManifest | None,
    *,
    needs_evidence_refresh: bool,
    no_gain_rounds: int,
) -> frozenset[str]:
    """Allow repair, adding retrieval only for manifest-declared gaps."""
    edit, _ = _split_tool_groups(default_tools)
    allowed = set(edit)
    if needs_evidence_refresh:
        if manifest is not None and manifest.missing_items:
            allowed.update(
                _insufficient_retrieval_tools(
                    manifest,
                    default_tools,
                    no_gain_rounds=no_gain_rounds,
                )
            )
        else:
            allowed.update(
                _retrieval_tools_from_profile(
                    manifest,
                    default_tools,
                    allow_heavy=False,
                )
            )
    return frozenset(allowed) or default_tools


def _manifest_state(run_state: Any) -> tuple[str, bool, bool]:
    manifest = getattr(run_state, "manifest", None)
    sufficiency = str(getattr(manifest, "sufficiency", Sufficiency.INSUFFICIENT))
    has_missing = bool(getattr(manifest, "missing_items", ()))
    has_stale = bool(stale_needing_refresh(manifest)) if manifest is not None else False
    return sufficiency, has_missing, has_stale


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


def _post_edit_verification_ready(
    run_state: Any,
    *,
    checklist: tuple[str, ...] = (),
) -> bool:
    """Reopen retrieval after planned edits land (checklist done or pause after edit)."""
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
    if checklist_plan_complete(checklist):
        return True
    rounds = int(getattr(run_state, "rounds_since_last_edit", 0) or 0)
    return rounds >= 1


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
    if _external_stale_blocks_edit_burst(manifest, edited_files):
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
      (D) diagnose + sufficient      -> no tools unless stale refresh needed
      (E) open manifest gaps         -> retrieval profile (+ edit when allowed)
      (F) saturated edit-ready       -> edit + saturation-throttled retrieval
    """
    run_state = getattr(state, "run_state", None)
    checklist = tuple(getattr(state, "checklist", ()) or ())
    _ = gravity_controller
    task_mode = str(getattr(run_state, "task_mode", "diagnose") or "diagnose")
    edit_tools, _ = _split_tool_groups(default_tools)
    manifest = getattr(run_state, "manifest", None)
    metrics = manifest_metrics(manifest) if manifest is not None else None
    sufficiency, has_missing, has_stale = _manifest_state(run_state)
    no_gain_rounds = int(getattr(run_state, "retrieval_no_gain_rounds", 0) or 0)
    view_last_round_all_duplicate = bool(
        getattr(run_state, "view_last_round_all_duplicate", False)
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
        return _validation_recovery_tools(
            default_tools,
            manifest,
            needs_evidence_refresh=(has_missing or has_stale),
            no_gain_rounds=no_gain_rounds,
        )

    if manifest is not None and manifest.failure_items:
        allowed = set(edit_tools)
        if has_missing or has_stale:
            allowed.update(
                _retrieval_tools_from_profile(
                    manifest,
                    default_tools,
                    allow_heavy=False,
                )
            )
        return frozenset(allowed) or default_tools

    if validation_status == "failed":
        allowed = set(edit_tools)
        if has_missing or has_stale:
            allowed.update(
                _retrieval_tools_from_profile(
                    manifest,
                    default_tools,
                    allow_heavy=False,
                )
            )
        return frozenset(allowed) or default_tools

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
        return _insufficient_retrieval_tools(
            manifest,
            default_tools,
            no_gain_rounds=no_gain_rounds,
        ) or default_tools

    if task_mode == "diagnose":
        if has_missing:
            return (
                _insufficient_retrieval_tools(
                    manifest,
                    default_tools,
                    no_gain_rounds=no_gain_rounds,
                )
                or default_tools
            )
        if has_stale:
            return (
                _retrieval_tools_from_profile(
                    manifest,
                    default_tools,
                    allow_heavy=False,
                )
                or default_tools
            )
        return frozenset()

    profile = retrieval_profile(manifest) if manifest is not None else None
    edit_ready_closed = (
        sufficiency in {
            Sufficiency.SUFFICIENT_FOR_EDIT,
            Sufficiency.SUFFICIENT_FOR_VERIFY,
        }
        and not has_missing
        and not blocks_edit
        and _edit_allowed(
            task_mode=task_mode,
            sufficiency=sufficiency,
            has_missing=has_missing,
            metrics=metrics,
        )
    )
    if (
        edit_ready_closed
        and not has_stale
        and profile is not None
        and not profile.needs_grep
        and not profile.needs_view
        and not profile.needs_heavy
    ):
        return edit_tools or default_tools

    retrieval = _retrieval_tools_from_profile(
        manifest,
        default_tools,
        allow_heavy=False,
    )
    if not blocks_edit and not has_missing:
        retrieval = _apply_retrieval_saturation(
            retrieval or (PRIMARY_RETRIEVAL_TOOLS & default_tools),
            default_tools,
            manifest=manifest,
            no_gain_rounds=no_gain_rounds,
            view_last_round_all_duplicate=view_last_round_all_duplicate,
            has_open_gaps=has_stale,
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
