from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from src.executor.edit_guard import is_edit_ambiguous_error, is_edit_recoverable_error
from src.planner.task_tree import SubTaskKind, SubTaskNode


class EscalationAction(StrEnum):
    RETRY_SUBTASK = "retry_subtask"
    REPLAN = "re_plan"
    ABORT = "abort"


@dataclass(frozen=True)
class EscalationVerdict:
    action: EscalationAction
    reason: str


def decide_subtask_escalation(
    subtask: SubTaskNode,
    exec_result: dict,
    *,
    attempt: int,
    max_subtask_retries: int,
) -> EscalationVerdict:
    """Decide whether to retry the same subtask or escalate to Planner re-plan.

    Policy (user-aligned):
    - Retry same subtask when the executor ran out of turns or exit-gate failed
      but the approach may still work (exploration / fix attempts).
    - Re-plan only after subtask retries are exhausted, or when the plan itself
      is wrong (whitelist / missing context_files).
    - Quality-gate exhaustion inside executor counts as a scored fix attempt.
    """
    if exec_result.get("success"):
        return EscalationVerdict(
            EscalationAction.ABORT,
            "internal error: successful subtask must not escalate",
        )

    errors = list(exec_result.get("error_trace") or [])
    failure_code = str(exec_result.get("failure_code") or "unknown")
    qg_fails = int(exec_result.get("quality_gate_failures") or 0)
    changed = list(exec_result.get("changed_files") or [])

    if _has_whitelist_block(errors):
        return EscalationVerdict(
            EscalationAction.REPLAN,
            "context_files whitelist blocked required paths — Planner must widen scope",
        )

    if attempt >= max_subtask_retries:
        return EscalationVerdict(
            EscalationAction.REPLAN,
            f"subtask [{subtask.id}] failed after {attempt} attempt(s) "
            f"(last: {failure_code}, quality_gate_fails={qg_fails})",
        )

    if failure_code == "quality_gate_exhausted":
        return EscalationVerdict(
            EscalationAction.RETRY_SUBTASK,
            f"quality gate failed {qg_fails} time(s) — retry subtask "
            f"({attempt}/{max_subtask_retries})",
        )

    if failure_code == "turn_limit":
        if subtask.kind == SubTaskKind.DIAGNOSE:
            if exec_result.get("summarized_after_limit"):
                return EscalationVerdict(
                    EscalationAction.ABORT,
                    "internal error: summarized diagnose should be success",
                )
            if attempt >= 2:
                return EscalationVerdict(
                    EscalationAction.REPLAN,
                    f"diagnose [{subtask.id}] exhausted turns after {attempt} attempt(s)",
                )
            return EscalationVerdict(
                EscalationAction.RETRY_SUBTASK,
                f"diagnose turn limit — retry once ({attempt}/2)",
            )
        if subtask.kind == SubTaskKind.EDIT and not changed:
            if _has_edit_match_failure(errors):
                return EscalationVerdict(
                    EscalationAction.REPLAN,
                    f"edit [{subtask.id}] hit turn limit with edit_file match failures — "
                    "Planner must add diagnose with exact snippet or narrow target lines",
                )
            return EscalationVerdict(
                EscalationAction.RETRY_SUBTASK,
                f"turn limit with no edits — retry subtask ({attempt}/{max_subtask_retries})",
            )
        return EscalationVerdict(
            EscalationAction.RETRY_SUBTASK,
            f"turn limit — retry subtask ({attempt}/{max_subtask_retries})",
        )

    if failure_code == "exit_gate" and subtask.kind == SubTaskKind.DIAGNOSE:
        if _diagnose_acceptance_unmet(errors):
            return EscalationVerdict(
                EscalationAction.REPLAN,
                f"diagnose [{subtask.id}] did not meet acceptance — Planner must revise search",
            )
        if attempt >= 2:
            return EscalationVerdict(
                EscalationAction.REPLAN,
                f"diagnose exit gate failed after {attempt} attempt(s)",
            )
        return EscalationVerdict(
            EscalationAction.RETRY_SUBTASK,
            f"exit gate failed — retry diagnose ({attempt}/2)",
        )

    if failure_code == "exit_gate":
        return EscalationVerdict(
            EscalationAction.RETRY_SUBTASK,
            f"exit gate failed — retry subtask ({attempt}/{max_subtask_retries})",
        )

    if failure_code == "skill_executor":
        detail = errors[-1] if errors else "no detail"
        return EscalationVerdict(
            EscalationAction.ABORT,
            f"skill executor failed: {detail}",
        )

    return EscalationVerdict(
        EscalationAction.RETRY_SUBTASK,
        f"executor failed ({failure_code}) — retry subtask ({attempt}/{max_subtask_retries})",
    )


def _has_edit_match_failure(errors: list[str]) -> bool:
    return any(is_edit_recoverable_error(e) for e in errors)


def _has_edit_ambiguous(errors: list[str]) -> bool:
    return any(is_edit_ambiguous_error(e) for e in errors)


def _diagnose_acceptance_unmet(errors: list[str]) -> bool:
    needle = "acceptance_criteria was not met"
    return any(needle in e.lower() for e in errors)


def _has_whitelist_block(errors: list[str]) -> bool:
    return any(
        "outside subtask edit scope" in e
        or "outside subtask whitelist" in e
        or "not in context_files" in e
        or "outside the project" in e
        for e in errors
    )
