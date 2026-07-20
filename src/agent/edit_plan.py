"""Core plan: a single flat ``edit_plan`` of equal-sized steps.

One ``edit_plan`` step == one ``decision_edit`` consumption unit == what EditLLM
can apply in a single call:

  - Exactly one ``target_file``
  - ``intent`` — one coherent change (may span ≤3 delta SITE blocks in one file)
  - ``focus_symbols`` — 1–3 on-disk symbols (required)
  - ``context_window`` — at least one span (required; harness may widen it)

Core emits the whole plan once; the harness freezes it and drains step-by-step
via ``decision_edit`` WITHOUT another Core turn between successful steps. Only an
error whose RetryOwner is CORE (see edit_errors.py) halts the drain and hands the
failed step back for replanning. There is no separate 大步/小步 split and no
Core-maintained ``[ ]`` checklist — the plan is the single control artifact.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent.checklist import checklist_item_done, checklist_item_open, parse_checklist_lines

# Accept ``edit_plan`` (canonical) and legacy ``edit_queue`` fences/keys.
_EDIT_QUEUE_FENCE = re.compile(
    r"```(?:edit_plan|edit_queue|json)?\s*\n(?P<body>\[.*?\])\s*\n```",
    re.DOTALL | re.IGNORECASE,
)
_EDIT_QUEUE_KEY = re.compile(
    r"['\"](?:edit_plan|edit_queue)['\"]\s*:\s*(\[.*?\])",
    re.DOTALL,
)

# Hard caps — keep in sync with CursorPatchApplier / decision_edit settings.
MAX_PATCH_BLOCKS_PER_MINOR_STEP = 3
PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP = 1
# Core-facing required fields for each plan step (no optional fields — an optional
# field is one Core never fills).
MAX_FOCUS_SYMBOLS_PER_STEP = 3


@dataclass(frozen=True, slots=True)
class ChecklistMajorStep:
    """大步 — one checklist line (outcome-level)."""

    text: str
    done: bool = False
    index: int = 0

    @property
    def open(self) -> bool:
        return not self.done


@dataclass(frozen=True, slots=True)
class EditQueueStep:
    """小步 — one decision_edit job for the Edit worker queue."""

    target_file: str
    intent: str
    focus_symbols: tuple[str, ...] = ()
    context_window: tuple[dict[str, Any], ...] = ()
    major_step_index: int | None = None
    major_step_text: str = ""
    step_id: str = ""
    max_blocks: int = MAX_PATCH_BLOCKS_PER_MINOR_STEP

    def to_decision_edit_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "target_file": self.target_file,
            "intent": self.intent,
            "focus_symbols": list(self.focus_symbols),
        }
        if self.context_window:
            args["context_window"] = [dict(item) for item in self.context_window]
        if self.step_id:
            args["task_id"] = self.step_id
        return args


@dataclass(frozen=True, slots=True)
class EditPlanQueue:
    """In-process queue of 小步 derived from Core 大步."""

    major_steps: tuple[ChecklistMajorStep, ...] = ()
    minor_steps: tuple[EditQueueStep, ...] = ()
    cursor: int = 0

    @property
    def remaining(self) -> tuple[EditQueueStep, ...]:
        return self.minor_steps[self.cursor :]

    @property
    def current(self) -> EditQueueStep | None:
        if self.cursor < 0 or self.cursor >= len(self.minor_steps):
            return None
        return self.minor_steps[self.cursor]

    @property
    def drained(self) -> bool:
        return self.cursor >= len(self.minor_steps)

    def advance(self) -> EditPlanQueue:
        if self.drained:
            return self
        return EditPlanQueue(
            major_steps=self.major_steps,
            minor_steps=self.minor_steps,
            cursor=self.cursor + 1,
        )


@dataclass(frozen=True, slots=True)
class EditPlanView:
    """Core-facing snapshot of edit_plan progress (replaces checklist in RUNTIME STATE).

    Harness owns cursor/status; Core only rewrites step content by emitting a new plan.
    """

    status: str = "empty"  # empty | draining | halted | complete
    cursor: int = 0  # 0-based index of next step to run (or failed step when halted)
    total: int = 0
    current_file: str = ""
    current_intent: str = ""
    failed_file: str = ""
    failed_intent: str = ""
    error_class: str = ""
    fail_reason: str = ""
    remaining: tuple[str, ...] = ()  # short "file — intent" lines
    completed: tuple[str, ...] = ()  # applied_diff_summary one-liners


_MAX_COMPLETED_IN_CARD = 8
_MAX_REMAINING_IN_CARD = 6
_MAX_INTENT_IN_CARD = 72


def _short_intent(intent: str) -> str:
    text = " ".join(str(intent or "").split())
    if len(text) <= _MAX_INTENT_IN_CARD:
        return text
    return text[: _MAX_INTENT_IN_CARD - 1] + "…"


def _step_one_liner(step: EditQueueStep) -> str:
    return f"{step.target_file} — {_short_intent(step.intent)}"


def edit_plan_view_from_queue(
    queue: EditPlanQueue,
    *,
    completed: Sequence[str] = (),
    status: str | None = None,
) -> EditPlanView:
    """Build a live view from the frozen queue (draining / empty / complete)."""
    total = len(queue.minor_steps)
    done = list(completed)[-_MAX_COMPLETED_IN_CARD:]
    if total == 0:
        return EditPlanView(status="empty", completed=tuple(done))
    if queue.drained:
        return EditPlanView(
            status="complete",
            cursor=total,
            total=total,
            completed=tuple(done),
        )
    current = queue.current
    remaining = tuple(
        _step_one_liner(step) for step in queue.remaining[1:_MAX_REMAINING_IN_CARD + 1]
    )
    return EditPlanView(
        status=status or "draining",
        cursor=queue.cursor,
        total=total,
        current_file=current.target_file if current else "",
        current_intent=_short_intent(current.intent) if current else "",
        remaining=remaining,
        completed=tuple(done),
    )


def edit_plan_view_halt(
    queue: EditPlanQueue,
    *,
    failed: EditQueueStep | Mapping[str, Any] | None,
    error_class: str = "",
    fail_reason: str = "",
    completed: Sequence[str] = (),
    failed_was_queue_current: bool | None = None,
) -> EditPlanView:
    """Snapshot remaining steps before the harness clears the queue on Core halt.

    When ``failed_was_queue_current`` is True (or failed is the queue's current
    EditQueueStep), remaining = steps after cursor. Otherwise the failed step was
    an in-turn decision_edit and the whole residual queue is still pending.
    """
    if isinstance(failed, EditQueueStep):
        failed_file = failed.target_file
        failed_intent = failed.intent
        on_queue = True if failed_was_queue_current is None else failed_was_queue_current
    elif isinstance(failed, Mapping):
        failed_file = str(failed.get("target_file") or failed.get("file") or "").strip()
        failed_intent = str(failed.get("intent") or failed.get("goal") or "").strip()
        on_queue = (
            False
            if failed_was_queue_current is None
            else failed_was_queue_current
        )
    else:
        failed_file = ""
        failed_intent = ""
        on_queue = True if failed_was_queue_current is None else failed_was_queue_current

    if on_queue and queue.minor_steps:
        rest = queue.minor_steps[queue.cursor + 1 :]
        cursor = queue.cursor
        total = len(queue.minor_steps)
    else:
        rest = queue.remaining
        cursor = 0
        total = 1 + len(rest) if (failed_file or rest) else len(queue.minor_steps)

    remaining = tuple(_step_one_liner(step) for step in rest[:_MAX_REMAINING_IN_CARD])
    reason = " ".join(str(fail_reason or "").split())
    if len(reason) > 160:
        reason = reason[:159] + "…"
    return EditPlanView(
        status="halted",
        cursor=cursor,
        total=total,
        failed_file=failed_file,
        failed_intent=_short_intent(failed_intent),
        error_class=str(error_class or "").strip(),
        fail_reason=reason,
        remaining=remaining,
        completed=tuple(list(completed)[-_MAX_COMPLETED_IN_CARD:]),
    )


def format_edit_plan_runtime_card(view: EditPlanView) -> str:
    """Multi-line edit_plan status for RUNTIME STATE (replaces checklist)."""
    total = view.total
    if view.status == "empty" and total == 0 and not view.completed:
        return "plan: empty (no frozen edit_plan)"

    if view.status == "complete":
        pos = total
    elif view.status == "halted":
        pos = (view.cursor + 1) if total else 0
    else:
        # draining: 1-based index of the current (next) step
        pos = (view.cursor + 1) if total and view.cursor < total else view.cursor

    lines = [f"plan: [{pos}/{total}] {view.status}"]
    if view.status == "halted":
        fail_bits = [
            f"file={view.failed_file or '?'}",
            f"intent={view.failed_intent or '?'}",
        ]
        if view.error_class:
            fail_bits.append(f"ErrorClass={view.error_class}")
        lines.append("failed_step: " + " | ".join(fail_bits))
        if view.fail_reason:
            lines.append(f"fail_reason: {view.fail_reason}")
        if view.error_class:
            from src.agent.edit_errors import EditErrorClass, core_hint_for

            try:
                klass = EditErrorClass(view.error_class)
            except ValueError:
                klass = EditErrorClass.UNKNOWN
            lines.append(f"core_action: {core_hint_for(klass)}")
        if view.remaining:
            lines.append("remaining (not run — re-emit in next edit_plan):")
            for item in view.remaining:
                lines.append(f"  - {item}")
        else:
            lines.append("remaining: none")
    elif view.status == "draining":
        if view.current_file:
            lines.append(
                f"current: {view.current_file} — {view.current_intent or '?'}"
            )
        if view.remaining:
            lines.append("upcoming:")
            for item in view.remaining:
                lines.append(f"  - {item}")
    if view.completed:
        lines.append("completed:")
        for item in view.completed:
            lines.append(f"  - {item}")
    return "\n".join(lines)


def format_edit_plan_evidence_lines(view: EditPlanView) -> list[str]:
    """Compact STEP EVIDENCE lines for plan halt / progress."""
    if view.status == "empty" and not view.completed:
        return []
    total = view.total
    if view.status == "halted":
        pos = (view.cursor + 1) if total else 0
        lines = [
            f"edit_plan: [{pos}/{total}] halted — Core must revise and re-emit edit_plan"
        ]
        if view.failed_file or view.error_class:
            bits = [f"file={view.failed_file or '?'}"]
            if view.error_class:
                bits.append(f"ErrorClass={view.error_class}")
            if view.failed_intent:
                bits.append(f"intent={view.failed_intent}")
            lines.append("edit_plan_failed: " + " | ".join(bits))
        if view.remaining:
            preview = "; ".join(view.remaining[:3])
            lines.append(
                f"edit_plan_remaining: {len(view.remaining)} step(s) — {preview}"
            )
        return lines
    if view.status == "draining" and total:
        pos = view.cursor + 1
        lines = [
            f"edit_plan: [{pos}/{total}] draining — harness consumes steps; "
            "do not re-plan mid-drain"
        ]
        if view.current_file:
            lines.append(
                f"edit_plan_current: {view.current_file} — {view.current_intent}"
            )
        return lines
    if view.status == "complete" and total:
        return [
            f"edit_plan: [{total}/{total}] complete — verify if needed; "
            "do not re-emit the same steps"
        ]
    return []


def parse_major_steps(checklist: Sequence[str] | str) -> tuple[ChecklistMajorStep, ...]:
    """Parse checklist lines into 大步."""
    if isinstance(checklist, str):
        items = parse_checklist_lines(checklist)
    else:
        items = tuple(str(item).strip() for item in checklist if str(item).strip())
    steps: list[ChecklistMajorStep] = []
    for index, item in enumerate(items):
        if checklist_item_done(item):
            text = item.split("]", 1)[-1].strip()
            steps.append(ChecklistMajorStep(text=text, done=True, index=index))
        elif checklist_item_open(item):
            text = item.split("]", 1)[-1].strip()
            steps.append(ChecklistMajorStep(text=text, done=False, index=index))
        else:
            # Bare line without marker — treat as open 大步.
            steps.append(ChecklistMajorStep(text=item, done=False, index=index))
    return tuple(steps)


def open_major_steps(
    checklist: Sequence[str] | str,
) -> tuple[ChecklistMajorStep, ...]:
    return tuple(step for step in parse_major_steps(checklist) if step.open)


def edit_queue_step_from_mapping(
    raw: Mapping[str, Any],
    *,
    major: ChecklistMajorStep | None = None,
    index: int = 0,
) -> EditQueueStep | None:
    """Build a 小步 from a Core-emitted JSON-like mapping."""
    target = str(raw.get("target_file") or raw.get("file") or "").strip()
    intent = str(raw.get("intent") or raw.get("goal") or "").strip()
    if not target or not intent:
        return None
    symbols_raw = raw.get("focus_symbols") or raw.get("symbols") or ()
    if isinstance(symbols_raw, str):
        symbols = tuple(part.strip() for part in symbols_raw.split(",") if part.strip())
    elif isinstance(symbols_raw, Sequence):
        symbols = tuple(str(item).strip() for item in symbols_raw if str(item).strip())
    else:
        symbols = ()
    windows_raw = raw.get("context_window") or ()
    windows: list[dict[str, Any]] = []
    if isinstance(windows_raw, Sequence) and not isinstance(windows_raw, (str, bytes)):
        for item in windows_raw:
            if isinstance(item, Mapping):
                windows.append(dict(item))
    max_blocks = int(raw.get("max_blocks") or MAX_PATCH_BLOCKS_PER_MINOR_STEP)
    max_blocks = max(1, min(max_blocks, MAX_PATCH_BLOCKS_PER_MINOR_STEP))
    step_id = str(raw.get("step_id") or raw.get("id") or f"step-{index}").strip()
    major_index: int | None
    if major is not None:
        major_index = major.index
    else:
        raw_idx = raw.get("major_step_index")
        major_index = int(raw_idx) if isinstance(raw_idx, int) else None
    return EditQueueStep(
        target_file=target,
        intent=intent,
        focus_symbols=symbols,
        context_window=tuple(windows),
        major_step_index=major_index,
        major_step_text=(
            major.text if major is not None else str(raw.get("major_step_text") or "")
        ),
        step_id=step_id,
        max_blocks=max_blocks,
    )


def build_edit_queue(
    minor_steps: Sequence[Mapping[str, Any] | EditQueueStep],
    *,
    checklist: Sequence[str] | str = (),
) -> EditPlanQueue:
    """Assemble an EditPlanQueue from Core-expanded 小步 (+ optional 大步 checklist)."""
    majors = parse_major_steps(checklist) if checklist else ()
    majors_by_index = {step.index: step for step in majors}
    built: list[EditQueueStep] = []
    for index, raw in enumerate(minor_steps):
        if isinstance(raw, EditQueueStep):
            built.append(raw)
            continue
        if not isinstance(raw, Mapping):
            continue
        major_idx = raw.get("major_step_index")
        major = None
        if isinstance(major_idx, int):
            major = majors_by_index.get(major_idx)
        step = edit_queue_step_from_mapping(raw, major=major, index=index)
        if step is not None:
            built.append(step)
    return EditPlanQueue(major_steps=majors, minor_steps=tuple(built), cursor=0)


def validate_minor_step(step: EditQueueStep) -> tuple[bool, str]:
    """Return (ok, reason) for one plan step before decision_edit.

    All Core-facing fields are required (no optional fields).
    """
    if not step.target_file.strip():
        return False, "E4_SPEC: plan step missing target_file"
    if not step.intent.strip():
        return False, "E4_SPEC: plan step missing intent"
    if not step.focus_symbols:
        return False, "E4_SPEC: plan step missing focus_symbols (≥1 on-disk symbol)"
    if not step.context_window:
        return False, "E4_SPEC: plan step missing context_window (≥1 span)"
    if step.max_blocks > MAX_PATCH_BLOCKS_PER_MINOR_STEP:
        return (
            False,
            f"E4_SPEC: plan step max_blocks={step.max_blocks} exceeds "
            f"{MAX_PATCH_BLOCKS_PER_MINOR_STEP}",
        )
    from src.agent.edit_brief import intent_needs_split

    # Overload is expanded by expand_decision_edit_to_steps before execute;
    # only reject if a fat intent somehow entered the frozen queue unsplit.
    if intent_needs_split(step.intent) and len(step.focus_symbols) >= 3:
        return False, "E4_SPEC: plan step intent is overloaded — split into more steps"
    if len(step.focus_symbols) > MAX_FOCUS_SYMBOLS_PER_STEP:
        return (
            False,
            f"E4_SPEC: focus_symbols>{MAX_FOCUS_SYMBOLS_PER_STEP} — split plan step",
        )
    return True, ""


def major_minor_contract_text() -> str:
    """Compact plan-contract blurb for prompts / STEP EVIDENCE."""
    return (
        "Emit one `edit_plan` (JSON array). Each step = one decision_edit "
        "(one target_file, one coherent change, "
        f"max {MAX_PATCH_BLOCKS_PER_MINOR_STEP} SITE blocks). "
        "Required per step: target_file, intent, focus_symbols "
        f"(1–{MAX_FOCUS_SYMBOLS_PER_STEP} on-disk symbols), context_window (≥1 span). "
        "Harness freezes the plan and drains it; it only returns to Core when an "
        "ErrorClass routes back (E4/E5/E6 or Edit retries exhausted)."
    )


def mechanical_rule_lines_for_turn(
    *,
    edit_ready: bool,
    edit_plan_view: EditPlanView | None = None,
    edit_burst: bool = False,
) -> list[str]:
    """Event-gated mechanical rules for STEP EVIDENCE (not the system prompt).

    Full plan/ErrorClass rulebooks stay in harness code; only the card that matches
    this turn's gate is spliced into the user message.
    """
    view = edit_plan_view
    status = view.status if view is not None else "empty"
    lines: list[str] = []

    # Gate A — need a plan: emit contract once when ready and no frozen queue.
    if edit_ready and status in {"empty", "complete"} and not edit_burst:
        lines.append(f"plan_contract: {major_minor_contract_text()}")
        return lines

    # Gate B — mid-drain: harness owns cursor; Core should not replan.
    if edit_burst or status == "draining":
        lines.append(
            "plan_rule: edit_plan is frozen/draining — harness runs decision_edit; "
            "do not re-emit edit_plan or re-view completed steps"
        )
        return lines

    # Gate C — halt: inject only the matching ErrorClass action card.
    if status == "halted" and view is not None:
        from src.agent.edit_errors import EditErrorClass, core_hint_for

        raw = str(view.error_class or "").strip()
        try:
            klass = EditErrorClass(raw) if raw else EditErrorClass.UNKNOWN
        except ValueError:
            klass = EditErrorClass.UNKNOWN
        lines.append(f"plan_rule: {core_hint_for(klass)}")
        if view.remaining:
            lines.append(
                "plan_rule: include failed + remaining steps in the next edit_plan"
            )
        return lines

    return lines


def _norm_step_key(file_path: str, intent: str) -> tuple[str, str]:
    return (
        file_path.replace("\\", "/").lstrip("./").strip().casefold(),
        " ".join(intent.strip().split()).casefold(),
    )


def parse_edit_queue_from_text(text: str) -> tuple[dict[str, Any], ...]:
    """Extract Core-emitted edit_plan (or legacy edit_queue) step mappings."""
    if not text or not text.strip():
        return ()
    candidates: list[str] = []
    for match in _EDIT_QUEUE_FENCE.finditer(text):
        body = (match.group("body") or "").strip()
        if body.startswith("["):
            candidates.append(body)
    for match in _EDIT_QUEUE_KEY.finditer(text):
        candidates.append(match.group(1))
    # Prefer the last explicit payload in the turn (latest plan revision).
    for payload in reversed(candidates):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return tuple(
                dict(item) for item in parsed if isinstance(item, Mapping)
            )
        if isinstance(parsed, Mapping) and isinstance(parsed.get("edit_queue"), list):
            return tuple(
                dict(item)
                for item in parsed["edit_queue"]
                if isinstance(item, Mapping)
            )
    return ()


def sync_edit_queue_from_core_turn(
    *,
    response_text: str,
    decision_edit_args: Sequence[Mapping[str, Any]],
    checklist: Sequence[str],
    previous: EditPlanQueue | None = None,
    project_root: Path | None = None,
) -> EditPlanQueue:
    """Build the frozen plan queue after Core's turn.

    Core emits one ``edit_plan`` JSON block listing every step. Steps that match
    this turn's in-turn ``decision_edit`` tool calls are dropped (already run);
    the remainder is frozen for auto-drain without another Core turn.

    If Core forgets the fence but packs ``Update X and add Y`` into one
    decision_edit, harness derives follow-on steps so Edit can still drain.
    """
    from src.agent.edit_brief import derive_edit_queue_followons

    minors = list(parse_edit_queue_from_text(response_text))
    if not minors:
        minors = derive_edit_queue_followons(
            decision_edit_args, project_root=project_root
        )

    if not minors and previous is not None and not previous.drained:
        # Keep an unfinished queue unless Core issued a fresh one.
        return previous

    if not minors:
        return EditPlanQueue(
            major_steps=parse_major_steps(checklist) if checklist else (),
        )

    consumed = [
        _norm_step_key(
            str(args.get("target_file") or args.get("file") or ""),
            str(args.get("intent") or args.get("goal") or ""),
        )
        for args in decision_edit_args
        if isinstance(args, Mapping)
    ]
    # Follow-ons derived from a compound intent are never "consumed" by the
    # parent tool call (keys differ); keep them all.
    residual: list[Mapping[str, Any]] = []
    for item in minors:
        key = _norm_step_key(
            str(item.get("target_file") or item.get("file") or ""),
            str(item.get("intent") or item.get("goal") or ""),
        )
        if key != ("", "") and key in consumed:
            consumed.remove(key)
            continue
        residual.append(item)
    return build_edit_queue(residual, checklist=checklist)
