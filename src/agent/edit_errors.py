"""Edit/Core retry ownership for decision_edit failures.

Error classes (stable contract for harness + prompts):

  E1 FORMAT   — marker glue / nested markers / invalid patch shape
  E2 LOCATE   — SEARCH mismatch, overlapping blocks
  E3 SYNTAX   — apply succeeded but AST/gofmt/syntax failed (non-format)
  E4 SPEC     — bad Core specs (missing file/symbol, empty intent, not edit)
  E5 EVIDENCE — need retrieval / unknown next file (Core or discovery)
  E6 MILESTONE— semantic/test failure after a plan milestone (Core)

Routing:
  Edit inner retry: E1, E2, and format-tainted E3 (marker residue).
  Core retry / replan: E2/E3 after Edit exhaustion, E4, E5, E6.
"""

from __future__ import annotations

from enum import Enum


class EditErrorClass(str, Enum):
    E1_FORMAT = "E1_FORMAT"
    E2_LOCATE = "E2_LOCATE"
    E3_SYNTAX = "E3_SYNTAX"
    E4_SPEC = "E4_SPEC"
    E5_EVIDENCE = "E5_EVIDENCE"
    E6_MILESTONE = "E6_MILESTONE"
    UNKNOWN = "UNKNOWN"


class RetryOwner(str, Enum):
    EDIT = "edit"  # Decision/Edit LLM inner loop
    CORE = "core"  # Core LLM replan / new decision_edit / retrieval
    NONE = "none"  # Terminal; surface only


def classify_edit_error(
    error: str,
    *,
    attempted_content: str = "",
    apply_succeeded: bool = False,
) -> EditErrorClass:
    """Classify a decision_edit / validator failure string."""
    text = (error or "").strip()
    if not text:
        return EditErrorClass.UNKNOWN
    lowered = text.casefold()
    attempted = attempted_content or ""

    if (
        "too many SEARCH/REPLACE blocks" in text
        or "nested SEARCH/REPLACE markers" in text
        or "stray ======= marker" in text
        or "expected SEARCH/REPLACE blocks only" in text
        or "expected SITE" in text
        or "empty patch" in lowered
        or "replace equals on-disk" in lowered
        or "patch produces no change" in text
        or "applied patch left SEARCH/REPLACE markers" in text
        or "decision_schema:" in lowered
        or "invalid json" in lowered
        or text.startswith("E1_FORMAT:")
        or "<<<<<<<" in attempted
        or ">>>>>>>" in attempted
    ):
        return EditErrorClass.E1_FORMAT

    if (
        text.startswith("mismatch:")
        or text.startswith("E2_LOCATE:")
        or "overlaps another block" in text
        or "not found in" in lowered and "symbol=" in lowered
        or "out of range for" in lowered
        or "has no symbol=" in lowered
    ):
        return EditErrorClass.E2_LOCATE

    if text.startswith("invalid_patch:"):
        return EditErrorClass.E1_FORMAT

    if apply_succeeded and (
        "python_syntax_error" in lowered
        or "syntaxerror" in lowered
        or "gofmt" in lowered
        or "ast validation failed" in lowered
    ):
        if "<<<<<<<" in attempted or ">>>>>>>" in attempted:
            return EditErrorClass.E1_FORMAT
        return EditErrorClass.E3_SYNTAX

    if (
        "action is not edit" in lowered
        or "target is outside project root" in lowered
        or text.startswith("scope:")
    ):
        return EditErrorClass.E4_SPEC

    if (
        "edit_ready: no" in lowered
        or "wiring_gap" in lowered
        or "bootstrap_gate" in lowered
        or "missing evidence" in lowered
    ):
        return EditErrorClass.E5_EVIDENCE

    if (
        "pytest" in lowered
        or "go test" in lowered
        or "mvn test" in lowered
        or "validation failed" in lowered
        and "ast validation" not in lowered
    ):
        return EditErrorClass.E6_MILESTONE

    if apply_succeeded and not text.startswith(("mismatch:", "invalid_patch:")):
        # Applied but validator rejected without clear syntax markers.
        return EditErrorClass.E3_SYNTAX

    return EditErrorClass.UNKNOWN


def retry_owner_for(
    error_class: EditErrorClass,
    *,
    edit_retries_remaining: bool,
) -> RetryOwner:
    """Who should act next for this error class."""
    if error_class in {EditErrorClass.E1_FORMAT, EditErrorClass.E2_LOCATE}:
        return RetryOwner.EDIT if edit_retries_remaining else RetryOwner.CORE
    if error_class is EditErrorClass.E3_SYNTAX:
        # One Edit chance only when retries remain; otherwise Core.
        return RetryOwner.EDIT if edit_retries_remaining else RetryOwner.CORE
    if error_class in {
        EditErrorClass.E4_SPEC,
        EditErrorClass.E5_EVIDENCE,
        EditErrorClass.E6_MILESTONE,
    }:
        return RetryOwner.CORE
    return RetryOwner.CORE if not edit_retries_remaining else RetryOwner.EDIT


def edit_inner_retry_allowed(
    error: str,
    *,
    attempted_content: str = "",
    apply_succeeded: bool = False,
    edit_retries_remaining: bool,
) -> bool:
    """True when Decision/Edit LLM should regenerate the patch in-process."""
    klass = classify_edit_error(
        error,
        attempted_content=attempted_content,
        apply_succeeded=apply_succeeded,
    )
    return (
        retry_owner_for(klass, edit_retries_remaining=edit_retries_remaining)
        is RetryOwner.EDIT
    )


def core_hint_for(error_class: EditErrorClass) -> str:
    """Short Core-facing guidance after Edit cannot fix the failure.

    Only reached when RetryOwner is CORE — i.e. Edit exhausted its inner retries
    (E1/E2/E3) or the error is Core-owned (E4/E5/E6). Fix the plan step, then
    re-emit an ``edit_plan`` for the remaining work.
    """
    if error_class is EditErrorClass.E1_FORMAT:
        return (
            "ErrorClass=E1_FORMAT: Edit could not produce a valid patch shape for "
            "this step. Shrink the step (fewer SITE blocks) or tighten intent, "
            "then re-emit edit_plan for the remaining steps."
        )
    if error_class is EditErrorClass.E2_LOCATE:
        return (
            "ErrorClass=E2_LOCATE: SITE symbol/anchor not on disk — the plan step "
            "targets a wrong/absent symbol. Fix focus_symbols to on-disk names "
            "(available_symbols / Did you mean…), then re-emit edit_plan."
        )
    if error_class is EditErrorClass.E3_SYNTAX:
        return (
            "ErrorClass=E3_SYNTAX: patch applied but broke syntax — the change was "
            "wrong for this step. Revise the step's intent/spans, then re-emit "
            "edit_plan; do not enlarge the patch."
        )
    if error_class is EditErrorClass.E4_SPEC:
        return (
            "ErrorClass=E4_SPEC: malformed plan step. Each step needs target_file, "
            "a single coherent intent, focus_symbols (1–3 on-disk), and "
            "context_window (≥1 span). Re-emit a corrected edit_plan."
        )
    if error_class is EditErrorClass.E5_EVIDENCE:
        return (
            "ErrorClass=E5_EVIDENCE: needed code is not loaded. Retrieve the "
            "missing file/symbol (grep/view), then re-emit edit_plan for the step."
        )
    if error_class is EditErrorClass.E6_MILESTONE:
        return (
            "ErrorClass=E6_MILESTONE: tests/semantics failed after edits. Inspect "
            "validator output and plan a corrective step; do not blindly re-run."
        )
    return (
        "ErrorClass=UNKNOWN: shrink the failing plan step and re-emit edit_plan; "
        "if it persists, retrieve evidence then replan."
    )
