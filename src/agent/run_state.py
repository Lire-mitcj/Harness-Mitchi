from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Literal


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


@dataclass(frozen=True, slots=True)
class RunEffect:
    kind: str
    reason: str = ""
    allowed_tools: frozenset[str] = frozenset()


def requirements_for_task(task_text: str) -> frozenset[str]:
    lowered = task_text.casefold()
    required = {"target_implementation"}
    if any(word in lowered for word in ("endpoint", "route", "接口", "路由")):
        required.add("endpoint_implementation")
    if any(
        word in lowered
        for word in ("integrate", "integration", "mount", "caller", "接入", "挂载", "调用")
    ):
        required.add("integration_or_mount_point")
    if any(
        word in lowered
        for word in (
            "auth",
            "login",
            "token",
            "current_user",
            "登录态",
            "认证",
            "鉴权",
        )
    ):
        required.add("authentication_context")
    if any(
        word in lowered
        for word in ("role", "permission", "authorize", "角色", "权限", "授权")
    ):
        required.add("authorization_policy")
    if any(
        word in lowered
        for word in ("owner", "ownership", "tenant", "归属", "数据范围")
    ):
        required.update(("ownership_relation", "relevant_schema"))
    if any(
        word in lowered
        for word in ("sql", "schema", "table", "database", "数据库", "表结构")
    ):
        required.add("relevant_schema")
    if any(word in lowered for word in ("test", "verify", "验证", "测试")):
        required.add("test_or_validation_path")
    return frozenset(required)


def start_run(
    task_text: str,
    *,
    edit_mode: bool,
    retry_limit: int = 3,
    max_steps: int = 20,
) -> RunState:
    return RunState(
        phase=RunPhase.RETRIEVING,
        task_mode="edit" if edit_mode else "diagnose",
        step=0,
        max_steps=max_steps,
        evidence=EvidenceLedger(required=requirements_for_task(task_text)),
        retry_budget=RetryBudget(limit=retry_limit),
    )


def reduce_run_state(
    state: RunState,
    event: RunEvent,
) -> tuple[RunState, tuple[RunEffect, ...]]:
    if state.phase == RunPhase.TERMINAL:
        return state, (RunEffect("ignored", "run is terminal"),)

    if event.kind == "step_started":
        return replace(state, step=state.step + 1), ()

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
        next_state = replace(
            state,
            evidence=evidence,
            artifacts=_merge_artifact_refs(state.artifacts, event.artifact_refs),
            transition_reason=event.reason or "evidence_stored",
        )
        if evidence.complete:
            phase = (
                RunPhase.RESPONDING
                if state.task_mode == "diagnose"
                else RunPhase.ACTING
            )
            next_state = replace(
                next_state,
                phase=phase,
                transition_reason="required_evidence_grounded",
            )
        return next_state, ()

    if event.kind == "artifacts_invalidated":
        removed = event.artifact_refs.all
        evidence = replace(
            state.evidence,
            entries=tuple(
                item for item in state.evidence.entries if item.artifact_id not in removed
            ),
        )
        return (
            replace(
                state,
                evidence=evidence,
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
        return (
            replace(
                state,
                phase=RunPhase.VALIDATING,
                changes=replace(state.changes, files=files),
                validation=ValidationState(),
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
                    transition_reason="validation_passed_choose_edit_or_answer",
                ),
                (),
            )
        budget = _record_failure(state.retry_budget, event.fingerprint or "validation")
        if budget.exhausted:
            terminal = TerminalResult(False, "", "validation retry budget exhausted")
            return (
                replace(
                    state,
                    phase=RunPhase.TERMINAL,
                    validation=ValidationState(status="failed", issues=event.issues),
                    retry_budget=budget,
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
        return (
            replace(
                state,
                retry_budget=budget,
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
