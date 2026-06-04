from __future__ import annotations

from dataclasses import dataclass

from src.executor.edit_guard import (
    EDIT_AMBIGUOUS_HINT,
    EDIT_IDENTICAL_HINT,
    EDIT_NOT_FOUND_HINT,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


@dataclass(frozen=True)
class ExecutorRetryStrategy:
    """Structural differences per subtask attempt — not tool micromanagement."""

    attempt: int
    paths_only: bool
    user_hint: str
    replan_hint: str | None = None
    restrict_explore: bool = False


def classify_failure_pattern(errors: list[str]) -> str:
    text = " ".join(errors).lower()
    if "edit_file" in text and "old_string and new_string are identical" in text:
        return "edit_identical"
    if "edit_file" in text and "old_string not found" in text:
        return "edit_not_found"
    if "edit_file" in text and "old_string" in text and "appears" in text and "times" in text:
        return "edit_ambiguous"
    if "/tmp" in text or "outside the project" in text:
        return "scratch_path"
    if "already preloaded" in text or "read_file blocked" in text:
        return "read_preload_loop"
    if "duplicate context_search" in text or "blocked duplicate context_search" in text:
        return "context_search_loop"
    if "turn limit" in text or "hit executor turn limit" in text:
        return "turn_limit"
    if "quality gate" in text:
        return "quality_gate"
    if "exit gate" in text or "empty final answer" in text:
        return "exit_gate"
    if "not in context_files" in text:
        return "whitelist"
    return "unknown"


def build_executor_retry_strategy(
    subtask: SubTaskNode,
    *,
    subtask_attempt: int,
    prior_errors: list[str] | None,
    prior_exploration: str | None = None,
) -> ExecutorRetryStrategy:
    """Map subtask attempt number → enforced executor mode."""
    attempt = max(1, subtask_attempt)
    errors = list(prior_errors or [])
    pattern = classify_failure_pattern(errors)
    has_digest = bool(prior_exploration and prior_exploration.strip())

    if attempt <= 1 or subtask.kind != SubTaskKind.EDIT:
        return ExecutorRetryStrategy(
            attempt=attempt,
            paths_only=False,
            user_hint="",
        )

    if attempt == 2:
        restrict = has_digest or pattern in {
            "turn_limit",
            "edit_ambiguous",
            "read_preload_loop",
            "context_search_loop",
        }
        if pattern in {"edit_not_found", "edit_identical"}:
            restrict = False
        hint_parts = [
            f"Subtask attempt {attempt}: prior failure ({pattern}).",
        ]
        if pattern == "edit_ambiguous":
            hint_parts.append(EDIT_AMBIGUOUS_HINT)
        elif pattern == "edit_identical":
            hint_parts.append(EDIT_IDENTICAL_HINT)
        elif pattern == "edit_not_found":
            hint_parts.append(EDIT_NOT_FOUND_HINT)
        if restrict:
            hint_parts.append(
                "Prior handoff evidence is in context — do not repeat the same "
                "context_search. Use read_files once only if fallback is exposed, "
                "then edit_file with a unique old_string."
            )
        elif has_digest:
            hint_parts.append(
                "Prior exploration digest is in context. Prefer edit_file; avoid re-reading "
                "files already listed in the digest. Never /tmp."
            )
        return ExecutorRetryStrategy(
            attempt=attempt,
            paths_only=not has_digest and pattern != "edit_ambiguous",
            restrict_explore=restrict,
            user_hint=" ".join(hint_parts),
            replan_hint=None,
        )

    hint = (
        f"Subtask attempt {attempt}: repeated failure ({pattern}). "
        "Use the exploration digest; edit or report a blocker. Never /tmp."
    )
    if pattern == "edit_ambiguous":
        hint += " " + EDIT_AMBIGUOUS_HINT
    elif pattern == "edit_identical":
        hint += " " + EDIT_IDENTICAL_HINT
    elif pattern == "edit_not_found":
        hint += " " + EDIT_NOT_FOUND_HINT
    return ExecutorRetryStrategy(
        attempt=attempt,
        paths_only=True,
        restrict_explore=True,
        user_hint=hint,
        replan_hint=(
            "Executor failed twice on the same edit subtask. "
            "Revise the plan: split diagnose (grep) and edit, or narrow context_files."
        ),
    )


def replan_revision_directive(
    *,
    failed_subtask: SubTaskNode,
    error_trace: list[str],
    strategy: ExecutorRetryStrategy | None = None,
) -> str:
    """Planner-facing instructions so re-plan must differ from the failed tail."""
    pattern = classify_failure_pattern(error_trace)
    lines = [
        f"Failed subtask [{failed_subtask.id}] kind={failed_subtask.kind.value} "
        f"pattern={pattern}.",
        "Revise ONLY this failed step — completed steps and later pending steps stay fixed.",
        "You MUST change the replacement subtask(s) — do NOT repeat the same description, "
        "same single edit step, or same context_files-only-on-failed-paths approach.",
    ]
    if pattern == "read_preload_loop":
        lines.append(
            "Add a dedicated diagnose handoff BEFORE edit, or list explicit "
            "SQL/search evidence requirements in acceptance_criteria."
        )
    elif pattern == "scratch_path":
        lines.append(
            "Edit subtasks must stay within context_files — remind Planner that /tmp "
            "is forbidden; split exploration into a diagnose handoff."
        )
    elif pattern == "turn_limit":
        lines.append(
            "Break work into smaller subtasks (diagnose → narrow edit → verify)."
        )
    elif pattern == "edit_ambiguous":
        lines.append(
            "Add a diagnose subtask that cites exact file:line and SQL snippet, "
            "or set context_files to the target path with acceptance_criteria naming the symbol."
        )
    elif pattern in {"edit_not_found", "edit_identical"}:
        lines.append(
            "Diagnose must paste the exact multi-line code/SQL snippet into acceptance_criteria "
            "or context_files preload; edit subtask should not guess function signatures alone."
        )
    if strategy and strategy.replan_hint:
        lines.append(strategy.replan_hint)
    return "\n".join(lines)
