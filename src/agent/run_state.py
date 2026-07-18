from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Literal

from src.agent.evidence_slots import slots_for_task
from src.agent.manifest import (
    EvidenceItem,
    StepManifest,
    reconcile_observations,
)


class RunPhase(StrEnum):
    RETRIEVING = "retrieving"
    ACTING = "acting"
    VALIDATING = "validating"
    RESPONDING = "responding"
    WAITING_USER = "waiting_user"
    TERMINAL = "terminal"


TaskMode = Literal["diagnose", "edit"]


@dataclass(frozen=True, slots=True)
class Evidence:
    slot: str
    artifact_id: str
    file: str
    symbol: str | None
    evidence_type: str
    confidence: Literal["low", "medium", "high"] = "high"


@dataclass(frozen=True, slots=True)
class EvidenceLedger:
    required: frozenset[str] = frozenset({"target_implementation"})
    entries: tuple[Evidence, ...] = ()
    candidates: tuple[str, ...] = ()

    @property
    def grounded(self) -> frozenset[str]:
        return frozenset(
            item.slot
            for item in self.entries
            if item.artifact_id and item.confidence in {"medium", "high"}
        )

    @property
    def missing(self) -> frozenset[str]:
        return self.required - self.grounded

    @property
    def complete(self) -> bool:
        return not self.missing


@dataclass(frozen=True, slots=True)
class ArtifactRefs:
    code: tuple[str, ...] = ()
    schemas: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    summaries: tuple[str, ...] = ()

    @property
    def all(self) -> frozenset[str]:
        return frozenset((*self.code, *self.schemas, *self.facts, *self.summaries))


@dataclass(frozen=True, slots=True)
class ChangeState:
    files: tuple[str, ...] = ()
    committed: bool = False


@dataclass(frozen=True, slots=True)
class ValidationState:
    status: Literal["not_run", "passed", "failed"] = "not_run"
    issues: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetryBudget:
    failures: int = 0
    limit: int = 3
    last_fingerprint: str | None = None
    repeated: int = 0

    @property
    def exhausted(self) -> bool:
        return self.failures >= self.limit


@dataclass(frozen=True, slots=True)
class WaitingState:
    reason: str
    resume_phase: RunPhase


@dataclass(frozen=True, slots=True)
class TerminalResult:
    success: bool
    answer: str
    reason: str


@dataclass(frozen=True, slots=True)
class RunState:
    phase: RunPhase
    task_mode: TaskMode
    step: int
    max_steps: int
    evidence: EvidenceLedger
    artifacts: ArtifactRefs = field(default_factory=ArtifactRefs)
    changes: ChangeState = field(default_factory=ChangeState)
    validation: ValidationState = field(default_factory=ValidationState)
    retry_budget: RetryBudget = field(default_factory=RetryBudget)
    waiting: WaitingState | None = None
    terminal: TerminalResult | None = None
    transition_reason: str = "request_started"
    manifest: StepManifest = field(default_factory=StepManifest)
    retrieval_gain_in_round: bool = False
    retrieval_no_gain_rounds: int = 0
    view_last_round_all_duplicate: bool = False
    last_grep_error: str = ""
    last_view_error: str = ""
    grep_suggested_views: tuple[dict[str, Any], ...] = ()
    edit_patch_failed: bool = False
    rounds_since_last_edit: int = 0
    task_text: str = ""

    @property
    def retrieval_complete(self) -> bool:
        return self.evidence.complete

    @property
    def can_answer(self) -> bool:
        return self.phase == RunPhase.RESPONDING or (
            self.task_mode == "edit"
            and self.phase == RunPhase.ACTING
            and self.validation.status == "passed"
        )

    @property
    def active_files(self) -> tuple[str, ...]:
        files = [item.file for item in self.evidence.entries if item.file]
        files.extend(self.changes.files)
        return tuple(dict.fromkeys(files))

    @property
    def allowed_tools(self) -> frozenset[str]:
        if self.phase == RunPhase.RETRIEVING:
            return frozenset(
                {"codebase_retrieve", "grep_search", "view_symbol_code"}
            )
        if self.phase == RunPhase.ACTING:
            return frozenset({"decision_edit"})
        return frozenset()


@dataclass(frozen=True, slots=True)
class RunEvent:
    kind: str
    evidence: tuple[Evidence, ...] = ()
    candidates: tuple[str, ...] = ()
    artifact_refs: ArtifactRefs = field(default_factory=ArtifactRefs)
    tool_name: str | None = None
    file: str | None = None
    issues: tuple[str, ...] = ()
    answer: str = ""
    reason: str = ""
    fingerprint: str | None = None
    observations: tuple[dict[str, Any], ...] = ()
    retrieval_attempted: bool = False
    view_all_duplicate: bool = False
    grep_error: str = ""
    view_error: str = ""
    grep_suggested_views: tuple[dict[str, Any], ...] = ()
    edit_applied_this_round: bool = False


@dataclass(frozen=True, slots=True)
class RunEffect:
    kind: str
    reason: str = ""
    allowed_tools: frozenset[str] = frozenset()


def detect_edit_mode(task_text: str) -> bool:
    """Classify implementation/edit tasks vs diagnose-only questions."""
    return bool(
        re.search(
            r"\b(?:fix|implement|change|edit|add|remove|refactor|supplement|support|optimize|adjust|create|build)\b|"
            r"修复|修改|实现|新增|删除|重构|补充|完善|加上|添加|引入|支持|调整|优化|创建|构建|开发|"
            r"接到|对接|连通|连接|统一|挂接|串联|接入|挂载",
            task_text,
            re.IGNORECASE,
        )
    )


def requirements_for_task(task_text: str) -> frozenset[str]:
    """Evidence slots a task activates. Backed by the shared slot registry."""
    return slots_for_task(task_text)


def manifest_template_for_task(task_text: str, task_mode: TaskMode) -> StepManifest:
    """Create an empty bootstrap manifest; concrete targets are discovered dynamically."""
    step_kind = "edit" if task_mode == "edit" else "retrieval"
    return StepManifest(step_id="task.default", step_kind=step_kind)


def start_run(
    task_text: str,
    *,
    edit_mode: bool,
    retry_limit: int = 3,
    max_steps: int = 20,
) -> RunState:
    task_mode: TaskMode = "edit" if edit_mode else "diagnose"
    return RunState(
        phase=RunPhase.RETRIEVING,
        task_mode=task_mode,
        step=0,
        max_steps=max_steps,
        evidence=EvidenceLedger(required=requirements_for_task(task_text)),
        retry_budget=RetryBudget(limit=retry_limit),
        manifest=manifest_template_for_task(task_text, task_mode),
        task_text=task_text.strip(),
    )


def reduce_run_state(
    state: RunState,
    event: RunEvent,
) -> tuple[RunState, tuple[RunEffect, ...]]:
    if state.phase == RunPhase.TERMINAL:
        return state, (RunEffect("ignored", "run is terminal"),)

    if event.kind == "step_started":
        return replace(state, step=state.step + 1, retrieval_gain_in_round=False), ()

    if event.kind == "evidence_stored":
        known_artifacts = state.artifacts.all | event.artifact_refs.all
        accepted = tuple(
            item for item in event.evidence if item.artifact_id in known_artifacts
        )
        entries = _dedupe_evidence((*state.evidence.entries, *accepted))
        candidates = tuple(
            dict.fromkeys((*state.evidence.candidates, *event.candidates))
        )[-16:]
        evidence = replace(state.evidence, entries=entries, candidates=candidates)
        manifest = reconcile_observations(state.manifest, event.observations)
        gained = _retrieval_gain_from_observations(event.observations)
        next_state = replace(
            state,
            evidence=evidence,
            manifest=manifest,
            retrieval_gain_in_round=(state.retrieval_gain_in_round or gained),
            edit_patch_failed=False if gained else state.edit_patch_failed,
            artifacts=_merge_artifact_refs(state.artifacts, event.artifact_refs),
            transition_reason=event.reason or "evidence_stored",
        )
        if evidence.complete:
            if state.task_mode == "diagnose":
                # The (slot-filtered) evidence ledger is the reducer-time signal.
                # Concrete required obligations (core_items) still block answering;
                # observed anchors are projected to SATISFIED later in the loop, so
                # they must NOT be read here (they are MISSING at reducer time).
                if manifest.has_missing:
                    return next_state, ()
                phase = RunPhase.RESPONDING
            else:
                phase = RunPhase.ACTING
            next_state = replace(
                next_state,
                phase=phase,
                transition_reason="required_evidence_grounded",
            )
        return next_state, ()

    if event.kind == "tool_round_observed":
        no_gain_rounds = state.retrieval_no_gain_rounds
        if event.retrieval_attempted:
            from src.tools.grep_match_symbols import has_actionable_suggested_views

            if has_actionable_suggested_views(event.grep_suggested_views):
                # Productive grep: concrete handler/route targets — not a no-gain round.
                no_gain_rounds = 0
            else:
                no_gain_rounds = (
                    0 if state.retrieval_gain_in_round else no_gain_rounds + 1
                )
        grep_error = state.last_grep_error
        view_error = state.last_view_error
        grep_views = state.grep_suggested_views
        if event.grep_error:
            grep_error = event.grep_error
        elif state.retrieval_gain_in_round:
            grep_error = ""
        if event.view_error:
            view_error = event.view_error
        elif state.retrieval_gain_in_round:
            view_error = ""
        if event.grep_suggested_views:
            grep_views = event.grep_suggested_views
        edit_patch_failed = state.edit_patch_failed
        if event.view_all_duplicate and event.retrieval_attempted:
            edit_patch_failed = False
        if event.edit_applied_this_round:
            rounds_since_last_edit = 0
        elif state.changes.files:
            rounds_since_last_edit = state.rounds_since_last_edit + 1
        else:
            rounds_since_last_edit = 0
        return (
            replace(
                state,
                retrieval_no_gain_rounds=no_gain_rounds,
                view_last_round_all_duplicate=bool(event.view_all_duplicate),
                edit_patch_failed=edit_patch_failed,
                last_grep_error=grep_error,
                last_view_error=view_error,
                grep_suggested_views=grep_views,
                rounds_since_last_edit=rounds_since_last_edit,
                transition_reason=event.reason or "tool_round_observed",
            ),
            (),
        )

    if event.kind == "artifacts_invalidated":
        removed = event.artifact_refs.all
        evidence = replace(
            state.evidence,
            entries=tuple(
                item for item in state.evidence.entries if item.artifact_id not in removed
            ),
        )
        stale_files = {_file_from_artifact_id(ref) for ref in removed}
        stale_files.discard("")
        manifest = _mark_stale_for_files(
            state.manifest, stale_files, "artifacts invalidated"
        )
        return (
            replace(
                state,
                evidence=evidence,
                manifest=manifest,
                artifacts=_subtract_artifact_refs(state.artifacts, removed),
                transition_reason=event.reason or "artifacts_invalidated",
            ),
            (),
        )

    if event.kind == "tool_requested":
        if event.tool_name not in state.allowed_tools:
            return state, (
                RunEffect(
                    "reject_tool",
                    reason=(
                        f"tool {event.tool_name!r} is not allowed during {state.phase}"
                    ),
                    allowed_tools=state.allowed_tools,
                ),
            )
        return state, (RunEffect("execute_tool", allowed_tools=state.allowed_tools),)

    if event.kind == "edit_applied":
        files = tuple(dict.fromkeys((*state.changes.files, event.file or "")))
        files = tuple(item for item in files if item)
        edited = _norm_path(event.file or "")
        stale_targets = {edited} if edited else set()
        from src.agent.manifest import cross_file_partner_files

        stale_targets |= set(cross_file_partner_files(state.manifest, edited))
        manifest = _mark_stale_for_files(
            state.manifest,
            stale_targets,
            "file modified by decision_edit",
        )
        return (
            replace(
                state,
                phase=RunPhase.VALIDATING,
                changes=replace(state.changes, files=files),
                validation=ValidationState(),
                manifest=manifest,
                transition_reason="edit_applied",
            ),
            (RunEffect("run_validation"),),
        )

    if event.kind == "validation_finished":
        if not event.issues:
            return (
                replace(
                    state,
                    phase=RunPhase.ACTING,
                    validation=ValidationState(status="passed"),
                    manifest=_clear_failure_items(state.manifest),
                    edit_patch_failed=False,
                    transition_reason="validation_passed_choose_edit_or_answer",
                ),
                (),
            )
        budget = _record_failure(state.retry_budget, event.fingerprint or "validation")
        manifest = _append_failure_items(state.manifest, event.issues, "test_failure")
        if budget.exhausted:
            terminal = TerminalResult(False, "", "validation retry budget exhausted")
            return (
                replace(
                    state,
                    phase=RunPhase.TERMINAL,
                    validation=ValidationState(status="failed", issues=event.issues),
                    retry_budget=budget,
                    manifest=manifest,
                    terminal=terminal,
                    transition_reason=terminal.reason,
                ),
                (),
            )
        return (
            replace(
                state,
                phase=RunPhase.ACTING,
                validation=ValidationState(status="failed", issues=event.issues),
                retry_budget=budget,
                manifest=manifest,
                transition_reason="validation_failed_retry_edit",
            ),
            (RunEffect("retry_edit", event.reason),),
        )

    if event.kind == "permission_required":
        return (
            replace(
                state,
                phase=RunPhase.WAITING_USER,
                waiting=WaitingState(event.reason, state.phase),
                transition_reason="permission_required",
            ),
            (RunEffect("request_permission", event.reason),),
        )

    if event.kind == "clarification_requested":
        return (
            replace(
                state,
                phase=RunPhase.WAITING_USER,
                waiting=WaitingState(event.reason or "clarification required", state.phase),
                transition_reason="clarification_requested",
            ),
            (),
        )

    if event.kind in {"permission_granted", "permission_denied"}:
        if state.waiting is None:
            return state, (RunEffect("ignored", "no permission request is pending"),)
        if event.kind == "permission_granted":
            return (
                replace(
                    state,
                    phase=state.waiting.resume_phase,
                    waiting=None,
                    transition_reason="permission_granted",
                ),
                (),
            )
        terminal = TerminalResult(False, "", event.reason or "permission denied")
        return (
            replace(
                state,
                phase=RunPhase.TERMINAL,
                waiting=None,
                terminal=terminal,
                transition_reason=terminal.reason,
            ),
            (),
        )

    if event.kind == "answer_proposed":
        if not state.can_answer:
            return state, (
                RunEffect("reject_answer", f"answer is not allowed during {state.phase}"),
            )
        terminal = TerminalResult(True, event.answer, event.reason or "completed")
        return (
            replace(
                state,
                phase=RunPhase.TERMINAL,
                terminal=terminal,
                transition_reason=terminal.reason,
            ),
            (),
        )

    if event.kind == "preflight_blocked":
        edit_patch_failed = state.edit_patch_failed
        if event.tool_name == "decision_edit":
            edit_patch_failed = True
        return (
            replace(
                state,
                edit_patch_failed=edit_patch_failed,
                transition_reason=event.reason or "preflight_blocked",
            ),
            (RunEffect("retry", event.reason),),
        )

    if event.kind == "tool_failed":
        fingerprint = event.fingerprint or _failure_fingerprint(event.reason)
        budget = _record_failure(state.retry_budget, fingerprint)
        if budget.exhausted:
            terminal = TerminalResult(False, "", event.reason or "tool retry exhausted")
            return (
                replace(
                    state,
                    phase=RunPhase.TERMINAL,
                    retry_budget=budget,
                    terminal=terminal,
                    transition_reason=terminal.reason,
                ),
                (),
            )
        edit_patch_failed = state.edit_patch_failed
        if event.tool_name == "decision_edit":
            edit_patch_failed = True
        return (
            replace(
                state,
                retry_budget=budget,
                edit_patch_failed=edit_patch_failed,
                transition_reason=event.reason or "tool_failed",
            ),
            (RunEffect("retry", event.reason),),
        )

    if event.kind == "run_failed":
        terminal = TerminalResult(False, "", event.reason or "run failed")
        return (
            replace(
                state,
                phase=RunPhase.TERMINAL,
                terminal=terminal,
                transition_reason=terminal.reason,
            ),
            (),
        )

    return state, (RunEffect("ignored", f"unknown event: {event.kind}"),)


def _retrieval_gain_from_observations(observations: tuple[Any, ...]) -> bool:
    """True when a durable code/schema observation was stored (not locator-only)."""
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        if observation.get("locator_only") or observation.get("match_line"):
            continue
        code = (
            observation.get("code")
            or observation.get("observation_code")
            or observation.get("verbatim_code")
            or ""
        )
        if str(code).strip() and observation.get("file"):
            return True
    return False


def _dedupe_evidence(items: tuple[Evidence, ...]) -> tuple[Evidence, ...]:
    by_key: dict[tuple[str, str], Evidence] = {}
    for item in items:
        by_key[(item.slot, item.artifact_id)] = item
    return tuple(by_key.values())


def _merge_artifact_refs(left: ArtifactRefs, right: ArtifactRefs) -> ArtifactRefs:
    return ArtifactRefs(
        code=tuple(dict.fromkeys((*left.code, *right.code))),
        schemas=tuple(dict.fromkeys((*left.schemas, *right.schemas))),
        facts=tuple(dict.fromkeys((*left.facts, *right.facts))),
        summaries=tuple(dict.fromkeys((*left.summaries, *right.summaries))),
    )


def _subtract_artifact_refs(refs: ArtifactRefs, removed: frozenset[str]) -> ArtifactRefs:
    return ArtifactRefs(
        code=tuple(item for item in refs.code if item not in removed),
        schemas=tuple(item for item in refs.schemas if item not in removed),
        facts=tuple(item for item in refs.facts if item not in removed),
        summaries=tuple(item for item in refs.summaries if item not in removed),
    )


def _record_failure(budget: RetryBudget, fingerprint: str) -> RetryBudget:
    repeated = budget.repeated + 1 if budget.last_fingerprint == fingerprint else 1
    return replace(
        budget,
        failures=budget.failures + 1,
        last_fingerprint=fingerprint,
        repeated=repeated,
    )


def _failure_fingerprint(reason: str) -> str:
    return re.sub(r"\d+", "#", reason.casefold()).strip()[:160]


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def _file_from_artifact_id(artifact_id: str) -> str:
    """Artifact ids look like 'path/to/file.py:12-40'; recover the file part."""
    match = re.match(r"^(?P<file>.+):\d+-\d+$", artifact_id)
    return _norm_path(match.group("file")) if match else ""


def _mark_stale_for_files(
    manifest: StepManifest,
    files: set[str],
    reason: str,
) -> StepManifest:
    targets = {_norm_path(item) for item in files if item}
    if not targets:
        return manifest
    updated = []
    for item in manifest.required_items:
        if item.is_failure:
            updated.append(item)
            continue
        if item.file and _norm_path(item.file) in targets and item.status != "MISSING":
            updated.append(replace(item, status="STALE", stale_reason=reason))
        else:
            updated.append(item)
    return replace(manifest, required_items=tuple(updated))


def _append_failure_items(
    manifest: StepManifest,
    issues: tuple[str, ...],
    failure_type: str,
) -> StepManifest:
    existing = {item.id for item in manifest.required_items}
    additions: list[EvidenceItem] = []
    for index, issue in enumerate(issues):
        text = " ".join(str(issue).split())[:200]
        if not text:
            continue
        item_id = f"{failure_type}.{_failure_fingerprint(text)[:48]}"
        if item_id in existing:
            continue
        existing.add(item_id)
        additions.append(
            EvidenceItem(
                id=item_id,
                need=text,
                type=failure_type,  # type: ignore[arg-type]
                status="MISSING",
                provenance="validator",
            )
        )
    if not additions:
        return manifest
    return replace(
        manifest,
        required_items=(*manifest.required_items, *tuple(additions)),
    )


def _clear_failure_items(manifest: StepManifest) -> StepManifest:
    remaining = tuple(item for item in manifest.required_items if not item.is_failure)
    if len(remaining) == len(manifest.required_items):
        return manifest
    return replace(manifest, required_items=remaining)
