from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.agent.events import (
    AgentEvent,
    EventType,
    error_event,
    final_answer_event,
)
from src.agent.types import AgentState
from src.cli.run_summary import build_terminal_run_summary
from src.cli.stream_preview import planner_status_hint
from src.context.retriever import ContextRetriever
from src.executor.subtask_executor import SubTaskExecutor
from src.harness.discovery.input_parser import parse_turn_input
from src.harness.discovery.manifest import DiagnosticsManifest, manifest_actionable
from src.harness.discovery.manifest_gate import validate_manifest
from src.harness.discovery.scout_agent import ScoutAgent
from src.harness.gates.plan_gate import ReplanGateContext, validate_plan
from src.harness.gates.preflight_probe import assess_preflight
from src.harness.gates.types import GateVerdict, PreflightResult, TruncationPolicy
from src.harness.subtask.handoff import (
    collect_prior_summaries,
    commit_subtask_failure,
    commit_subtask_success,
)
from src.llm.client import LLMClient
from src.orchestrator.escalation import EscalationAction, decide_subtask_escalation
from src.orchestrator.evidence import EvidencePack
from src.planner.context_policy import effective_context_files
from src.planner.kinds import SubTaskKind
from src.planner.patch_plan_parse import parse_patch_plan_output
from src.planner.planner_node import (
    LiteLLMPlannerClient,
    PlannerNode,
    fallback_task_tree,
    parse_planner_output,
)
from src.planner.scout_skip import apply_scout_discovery_to_plan
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree
from src.skills import (
    CodeEditSkill,
    CodeSearchSkill,
    SkillContext,
    SkillExecutor,
    ValidatorSkill,
)

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.context.builder import ContextBuilder
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


def _cache_ttl(value: str) -> str:
    return "1h" if value.strip().lower() in {"1h", "hour", "60m"} else "5m"


def plan_update_event(task_tree: TaskTree) -> AgentEvent:
    return AgentEvent(
        type=EventType.PLAN_UPDATE,
        content=task_tree.to_outline(),
        data=task_tree.to_dict(),
    )


def _compact_preview(text: str, *, limit: int = 500) -> str:
    preview = " ".join((text or "").strip().split())
    if len(preview) <= limit:
        return preview
    return preview[: limit - 3] + "..."


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return max(1, len(text) // 4)


def _executor_status_data(**data: Any) -> dict[str, Any]:
    return {"phase": "executor", "executor_activity": True, **data}


@dataclass
class OrchestratorState:
    """Runtime bus state for the Planner-driven control loop."""

    task_tree: TaskTree | None = None
    agent_state: AgentState = field(default_factory=AgentState)
    replan_count: int = 0
    plan_gate_replans: int = 0
    discovery_manifest: DiagnosticsManifest | None = None
    subtask_attempts: dict[str, int] = field(default_factory=dict)
    subtask_summaries: dict[str, str] = field(default_factory=dict)
    subtask_exploration_digests: dict[str, str] = field(default_factory=dict)


class OrchestratorLoop:
    """Harness: Scout → Planner → PlanGate → Preflight → Executor."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        harness: HarnessEngine,
        context: ContextBuilder,
        permissions: PermissionManager,
        settings: MitKIISettings,
        *,
        planner: PlannerNode | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.harness = harness
        self.context = context
        self.permissions = permissions
        self.settings = settings
        self.state = OrchestratorState()
        self._executor = SubTaskExecutor(
            llm=llm,
            tools=tools,
            harness=harness,
            permissions=permissions,
            settings=settings,
            max_turns=settings.orchestrator_executor_max_turns,
        )
        self._planner = planner or PlannerNode(
            LiteLLMPlannerClient(
                model=settings.effective_planner_model,
                timeout=settings.llm_request_timeout,
                max_tokens=settings.planner_max_tokens,
                metrics=harness.probe.metrics,
                json_mode=settings.planner_json_mode,
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=_cache_ttl(settings.prompt_cache_ttl),
            ),
            require_trace=settings.planner_trace,
        )
        self._skill_executor = SkillExecutor([
            CodeEditSkill(
                project_root=harness.project_root,
                llm_complete=self._planner.client.complete,
            ),
            CodeSearchSkill(project_root=harness.project_root, tools=tools),
            ValidatorSkill(project_root=harness.project_root),
        ])
        scout_model = settings.scout_model
        self._scout = ScoutAgent(
            llm=LLMClient(
                model=scout_model,
                max_tokens=settings.scout_max_tokens,
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=_cache_ttl(settings.prompt_cache_ttl),
            ),
            tools=tools,
            harness=harness,
            permissions=permissions,
            settings=settings,
        )

    async def resolve_approval(self, action: str, approved: bool) -> None:
        await self._executor.resolve_approval(action, approved)
        await self._scout.resolve_approval(action, approved)

    async def run(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        self.harness.phase_metrics.reset_turn()
        user_msg, skip_discovery = parse_turn_input(user_msg)
        if not user_msg.strip():
            yield error_event("Empty request after parsing /plan prefix.")
            yield AgentEvent(type=EventType.STREAM_END)
            return

        project_structure = await self._project_structure()
        repo_map = await self._repo_map_for_preflight()

        yield AgentEvent(
            type=EventType.STATUS,
            content="Scanning project structure...",
            data={"spinner_only": True},
        )

        async for event in self._discovery_phase(user_msg, project_structure, skip_discovery):
            yield event

        manifest = self.state.discovery_manifest
        assert manifest is not None
        context_pack = await self._context_pack_for_planner(user_msg)
        discovery_block: str | None = None
        if self.settings.scout_enabled:
            discovery_block = manifest.to_planner_block(compact=True)
            if manifest_actionable(manifest):
                discovery_block += (
                    "\n<harness_directive>\n"
                    "Scout diagnosis is ACTIONABLE. Do NOT add kind=diagnose subtasks — "
                    "Executor must NOT re-explore. Start with edit or shell; use victim_files "
                    "for context_files.\n</harness_directive>"
                )
        if context_pack is not None:
            context_block = context_pack.to_planner_block()
            discovery_block = (
                f"{discovery_block}\n\n{context_block}"
                if discovery_block
                else context_block
            )
            yield AgentEvent(
                type=EventType.STATUS,
                content=(
                    "ContextRetriever: "
                    f"confidence={context_pack.confidence:.2f}, "
                    f"files={len(context_pack.relevant_files)}, "
                    f"symbols={len(context_pack.symbols)}"
                ),
                data={
                    "phase": "context_retriever",
                    "confidence": context_pack.confidence,
                    "files": list(context_pack.relevant_files),
                    "missing_info": list(context_pack.missing_info),
                },
            )
            if self.settings.patch_plan_enabled:
                async for event in self._try_patch_plan_skill_path(
                    user_msg=user_msg,
                    project_structure=project_structure,
                    context_pack_block=context_block,
                    context_pack=context_pack,
                ):
                    if event.type == EventType.FINAL_ANSWER:
                        yield event
                        yield AgentEvent(type=EventType.STREAM_END)
                        return
                    if (
                        event.type == EventType.ERROR
                        and event.data
                        and event.data.get("terminal")
                    ):
                        yield event
                        yield AgentEvent(type=EventType.STREAM_END)
                        return
                    yield event
            else:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content="PatchPlan experimental path disabled; using TaskTree planner.",
                    data={"phase": "patch_plan", "disabled": True},
                )
        elif not self.settings.patch_plan_enabled:
            yield AgentEvent(
                type=EventType.STATUS,
                content="PatchPlan experimental path disabled; using TaskTree planner.",
                data={"phase": "patch_plan", "disabled": True},
            )
        elif self.settings.patch_plan_enabled:
            event = self._patch_plan_failure_event("ContextRetriever produced no ContextPack")
            yield event
            if event.type == EventType.ERROR and event.data and event.data.get("terminal"):
                yield AgentEvent(type=EventType.STREAM_END)
                return
        yield AgentEvent(
            type=EventType.STREAM_START,
            data={"phase": "planner"},
        )
        plan_outcome: list[tuple[TaskTree | None, str | None]] = []
        async for ev in self._plan_with_gate(
            user_msg, project_structure, discovery_block, plan_outcome
        ):
            yield ev
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={"phase": "planner"},
        )
        if not plan_outcome:
            yield error_event("Planner produced no outcome")
            return
        task_tree, plan_error = plan_outcome[0]
        if plan_error:
            yield error_event(plan_error)
            return

        task_tree, scout_skipped = apply_scout_discovery_to_plan(task_tree, manifest)
        _apply_context_pack_to_task_tree(task_tree, context_pack)
        self.state.subtask_summaries.update(scout_skipped)
        for sid, summary in scout_skipped.items():
            node = task_tree.get(sid)
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Subtask [{sid}] skipped — Scout already diagnosed",
                data={
                    "milestone": "subtask_done",
                    "subtask_id": sid,
                    "kind": node.kind.value if node else "diagnose",
                    "scout_skip": True,
                },
            )
            self.state.subtask_summaries[sid] = summary

        self.state.task_tree = task_tree
        self.state.agent_state.current_plan = task_tree.to_json()
        yield plan_update_event(task_tree)

        while task_tree.has_pending():
            node = task_tree.first_pending()
            assert node is not None

            preflight = assess_preflight(
                subtask=node,
                task_tree=task_tree,
                project_root=self.harness.project_root,
                settings=self.settings,
                repo_map=repo_map,
            )
            pm = self.harness.phase_metrics
            pm.start("preflight", subtask_id=node.id)
            pm.end(
                "preflight",
                subtask_id=node.id,
                verdict=preflight.verdict.value,
                metadata={
                    "estimated_tokens": preflight.estimated_tokens,
                    "budget_tokens": preflight.budget_tokens,
                    "tier": preflight.policy.tier,
                },
            )

            policy = preflight.policy
            if preflight.verdict == GateVerdict.WARN and preflight.messages:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content="Preflight: " + "; ".join(preflight.messages),
                    data={"subtask_id": node.id, "preflight": preflight.verdict.value},
                )

            if not preflight.passed:
                replanned = await self._handle_preflight_block(
                    user_msg, task_tree, node, preflight, project_structure
                )
                if replanned is None:
                    policy = TruncationPolicy.red_fallback()
                    yield AgentEvent(
                        type=EventType.STATUS,
                        content=(
                            f"Preflight BLOCK for [{node.id}] — dispatching without "
                            "preloaded context (tool discovery fallback)."
                        ),
                    )
                else:
                    task_tree = replanned
                    self.state.task_tree = task_tree
                    self.state.agent_state.current_plan = task_tree.to_json()
                    yield plan_update_event(task_tree)
                    continue

            task_tree.mark_running(node.id)
            yield AgentEvent(
                type=EventType.STATUS,
                content=node.description,
                data={
                    "milestone": "subtask_start",
                    "subtask_id": node.id,
                    "kind": node.kind.value,
                    "description": node.description,
                },
            )

            pm.start("executor", subtask_id=node.id)
            exec_result: dict[str, Any] | None = None
            executor_fallback_errors: list[str] = []
            executor_fallback_digest: str | None = None
            if self.settings.executor_skill_enabled and node.kind == SubTaskKind.EDIT:
                async for event in self._run_edit_skill_executor(
                    user_msg=user_msg,
                    task_tree=task_tree,
                    node=node,
                    project_structure=project_structure,
                    context_pack=context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if "success" in data or data.get("failure_code"):
                            exec_result = data
                    yield event
                if exec_result and exec_result.get("requires_executor_fallback"):
                    executor_fallback_errors = list(exec_result.get("error_trace") or [])
                    executor_fallback_digest = exec_result.get("exploration_digest") or None
                    exec_result = None

            if exec_result is None:
                prior_errors = list(node.error_trace)
                prior_errors.extend(executor_fallback_errors)
                prior_ctx = collect_prior_summaries(
                    task_tree, node, self.state.subtask_summaries
                )
                attempt_num = self.state.subtask_attempts.get(node.id, 0) + 1
                prior_exploration = (
                    executor_fallback_digest
                    or self.state.subtask_exploration_digests.get(node.id)
                )
                async for event in self._executor.run(
                    root_task=task_tree.root_task,
                    task_tree=task_tree,
                    subtask=node,
                    truncation_policy=policy,
                    retry_feedback=prior_errors or None,
                    quality_gate_retry_limit=self.settings.subtask_quality_gate_retries,
                    prior_summaries=prior_ctx or None,
                    subtask_attempt=attempt_num,
                    prior_exploration=prior_exploration,
                    context_pack=context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if not data.get("heartbeat") and (
                            "success" in data or data.get("failure_code")
                        ):
                            exec_result = data
                    yield event

            if exec_result:
                pm.end(
                    "executor",
                    subtask_id=node.id,
                    verdict="SUCCESS" if exec_result.get("success") else "FAIL",
                    metadata={
                        "turns_used": exec_result.get("turns_used"),
                    },
                )

            if exec_result and exec_result.get("success"):
                self._sync_repo_map_after_exec(node, exec_result)
                commit_subtask_success(
                    task_tree=task_tree,
                    node=node,
                    exec_result=exec_result,
                    project_root=self.harness.project_root,
                    subtask_summaries=self.state.subtask_summaries,
                    subtask_attempts=self.state.subtask_attempts,
                    subtask_exploration_digests=self.state.subtask_exploration_digests,
                )
                cp_id = await self._save_subtask_checkpoint(node.id, task_tree)
                node.checkpoint_id = cp_id
                task_tree.mark_success(node.id, checkpoint_id=cp_id)
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=f"Subtask [{node.id}] SUCCESS",
                    data={
                        "milestone": "subtask_done",
                        "subtask_id": node.id,
                        "kind": node.kind.value,
                        "checkpoint_id": cp_id,
                    },
                )
                continue

            errors = (
                list(exec_result.get("error_trace") or [])
                if exec_result
                else ["Unknown executor failure"]
            )

            attempt = commit_subtask_failure(
                node=node,
                exec_result=exec_result,
                subtask_attempts=self.state.subtask_attempts,
                subtask_exploration_digests=self.state.subtask_exploration_digests,
            )
            verdict = decide_subtask_escalation(
                node,
                exec_result or {"success": False, "error_trace": errors},
                attempt=attempt,
                max_subtask_retries=self.settings.subtask_max_retries,
            )

            if verdict.action == EscalationAction.RETRY_SUBTASK:
                node.status = SubTaskStatus.PENDING
                node.error_trace = list(dict.fromkeys(errors))
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        f"Subtask [{node.id}] attempt {attempt}/"
                        f"{self.settings.subtask_max_retries} — {verdict.reason}"
                    ),
                    data={"subtask_id": node.id, "escalation": verdict.action.value},
                )
                continue

            task_tree.mark_failed(node.id, errors=errors)

            if verdict.action == EscalationAction.ABORT:
                yield error_event(verdict.reason, {"subtask_id": node.id})
                break

            if self.state.replan_count >= self.settings.orchestrator_max_replans:
                yield error_event(
                    f"Orchestrator replan limit reached ({self.settings.orchestrator_max_replans})",
                    {"subtask_id": node.id},
                )
                break

            replan_ctx = ReplanGateContext.from_node(node)
            gate_feedback: list[str] = []
            max_gate_attempts = self.settings.plan_gate_max_replans + 1
            plan_error: str | None = None

            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Subtask [{node.id}] escalating to re-plan — {verdict.reason}",
                data={"subtask_id": node.id, "escalation": EscalationAction.REPLAN.value},
            )

            for gate_attempt in range(max_gate_attempts):
                replan_errors = list(errors) + gate_feedback
                replan_evidence = EvidencePack(
                    subtask_id=node.id,
                    subtask_description=node.description,
                    error_trace=replan_errors,
                    file_diffs=(
                        dict(exec_result.get("file_diffs") or {})
                        if exec_result
                        else {}
                    ),
                    executor_turns_used=(
                        int(exec_result.get("turns_used") or 0)
                        if exec_result
                        else 0
                    ),
                    last_assistant_message=(
                        exec_result.get("final_message") if exec_result else None
                    ),
                    changed_files=(
                        list(exec_result.get("changed_files") or [])
                        if exec_result
                        else []
                    ),
                    context_files=effective_context_files(task_tree, node),
                    subtask_attempts=attempt,
                )
                if gate_attempt > 0:
                    yield AgentEvent(
                        type=EventType.STATUS,
                        content=f"Re-plan · PlanGate retry {gate_attempt + 1}",
                        data={"phase": "planner", "subtask_id": node.id},
                    )

                pm.start("replan")
                fresh_structure = await self._project_structure()
                task_tree = await self._planner.re_plan(
                    user_msg,
                    task_tree,
                    replan_evidence,
                    fresh_structure,
                    discovery_manifest=self._discovery_block(),
                )
                pm.end(
                    "replan",
                    metadata={
                        "trigger": "executor_fail",
                        "subtask_id": node.id,
                        "gate_attempt": gate_attempt + 1,
                    },
                )

                task_tree, plan_error = await self._apply_plan_gate(
                    task_tree,
                    replan_context=replan_ctx,
                )
                if not plan_error:
                    break
                gate_feedback.append(f"PlanGate rejected re-plan: {plan_error}")

            if plan_error:
                yield error_event(plan_error, {"subtask_id": node.id})
                break

            self.state.replan_count += 1
            self.state.subtask_attempts.clear()
            self.state.subtask_exploration_digests.clear()
            self.state.task_tree = task_tree
            self.state.agent_state.current_plan = task_tree.to_json()
            yield plan_update_event(task_tree)

        summary = self._build_summary(task_tree)
        yield final_answer_event(summary, terminal=True)
        yield AgentEvent(type=EventType.STREAM_END)

    async def _plan_with_gate(
        self,
        user_msg: str,
        project_structure: str,
        discovery_block: str | None,
        outcome: list[tuple[TaskTree | None, str | None]],
    ) -> AsyncIterator[AgentEvent]:
        pm = self.harness.phase_metrics
        last_err = ""
        previous_raw = ""
        parse_result = None
        tree: TaskTree | None = None
        max_attempts = self.settings.plan_gate_max_replans + 1
        client = self._planner.client

        for attempt in range(max_attempts):
            spinner_label = "Planner" if attempt == 0 else f"Planner · retry {attempt + 1}"
            yield AgentEvent(
                type=EventType.STATUS,
                content=spinner_label,
                data={"spinner_only": True, "phase": "planner"},
            )

            if attempt == 0:
                messages = self._planner.plan_messages(
                    user_msg,
                    project_structure,
                    discovery_manifest=discovery_block,
                )
            else:
                messages = self._planner.rewrite_messages(
                    user_msg,
                    project_structure,
                    gate_errors=last_err,
                    previous_raw=previous_raw,
                    discovery_manifest=discovery_block,
                )

            pm.start("plan")
            plan_meta: dict[str, Any] = {"attempt": attempt + 1, "source": "llm"}
            if attempt > 0:
                plan_meta["source"] = "llm_rewrite"

            parts: list[str] = []
            last_hint = ""
            if hasattr(client, "stream_complete"):
                async for delta in client.stream_complete(messages):
                    parts.append(delta)
                    hint = planner_status_hint("".join(parts))
                    if hint and hint != last_hint:
                        last_hint = hint
                        yield AgentEvent(
                            type=EventType.STATUS,
                            content=f"Planner · {hint}",
                            data={"spinner_only": True, "phase": "planner"},
                        )
            else:
                raw_once = await client.complete(messages)
                parts.append(raw_once)
                hint = planner_status_hint(raw_once)
                if hint:
                    yield AgentEvent(
                        type=EventType.STATUS,
                        content=f"Planner · {hint}",
                        data={"spinner_only": True, "phase": "planner"},
                    )

            raw = "".join(parts)
            previous_raw = raw
            parse_result = parse_planner_output(raw, fallback_task=user_msg)
            pm.end("plan", metadata=plan_meta)

            assert parse_result is not None
            tree = parse_result.tree

            if not parse_result.ok:
                last_err = "; ".join(parse_result.all_errors)
                log.debug("Planner parse/schema attempt %d failed: %s", attempt + 1, last_err)
                continue

            tree, err = await self._apply_plan_gate(tree)
            if err is None:
                self.state.plan_gate_replans = attempt
                outcome.append((tree, None))
                return
            last_err = err
            log.debug("PlanGate attempt %d failed: %s", attempt + 1, err)

        log.debug("PlanGate exhausted — using diagnose fallback: %s", last_err)
        fallback = fallback_task_tree(user_msg)
        tree, err = await self._apply_plan_gate(fallback)
        if err is None:
            self.state.plan_gate_replans = max_attempts
            outcome.append((fallback, None))
            return
        outcome.append((
            None,
            f"PlanGate failed after {max_attempts} attempts: {last_err}; fallback: {err}",
        ))

    async def _try_patch_plan_skill_path(
        self,
        *,
        user_msg: str,
        project_structure: str,
        context_pack_block: str,
        context_pack: Any,
    ) -> AsyncIterator[AgentEvent]:
        if not self.settings.patch_plan_enabled:
            return
        if not context_pack.is_high_confidence():
            yield self._patch_plan_failure_event(
                "ContextPack confidence is low or missing_info exists",
            )
            return

        yield AgentEvent(
            type=EventType.STATUS,
            content="PatchPlan: generating executable patch plan",
            data={"spinner_only": True, "phase": "patch_plan"},
        )
        messages = self._planner.patch_plan_messages(
            user_msg,
            project_structure,
            context_pack=context_pack_block,
        )
        raw = await self._planner.client.complete(messages)
        parsed = parse_patch_plan_output(raw)
        if not parsed.ok or parsed.patch_plan is None:
            yield self._patch_plan_failure_event(
                "; ".join(parsed.all_errors or ["planner returned no patch_plan"]),
                raw_preview=raw,
            )
            return
        patch_plan = parsed.patch_plan
        if not patch_plan.is_executable():
            yield self._patch_plan_failure_event(
                "plan is not executable "
                f"(confidence={patch_plan.confidence:.2f}, "
                f"edits={len(patch_plan.edits)}, "
                f"missing_info={list(patch_plan.missing_info)})",
            )
            return

        result = await self._skill_executor.run(
            "code_edit",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
                patch_plan=patch_plan,
            ),
        )
        if not result.success:
            yield self._patch_plan_failure_event(result.summary)
            return

        validation = await self._skill_executor.run(
            "validator",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
                patch_plan=patch_plan,
            ),
            changed_files=result.changed_files,
        )
        if not validation.success:
            yield self._patch_plan_failure_event(validation.summary)
            return

        for path in result.changed_files:
            service = getattr(self.context, "repo_map_service", None)
            if service is not None:
                service.mark_dirty(path)
            self.context.file_tracker.record_edit(path)

        yield AgentEvent(
            type=EventType.STATUS,
            content=f"PatchPlan SUCCESS: {result.summary} {validation.summary}",
            data={
                "phase": "patch_plan",
                "changed_files": list(result.changed_files),
                "validation": validation.validation_result,
            },
        )
        yield final_answer_event(
            "PatchPlan applied successfully.\n"
            f"Changed files: {', '.join(result.changed_files) or '(none)'}\n"
            f"Validation: {validation.summary}",
            terminal=True,
        )

    async def _run_edit_skill_executor(
        self,
        *,
        user_msg: str,
        task_tree: TaskTree,
        node: SubTaskNode,
        project_structure: str,
        context_pack: Any,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · code_search",
            data=_executor_status_data(
                subtask_id=node.id,
                spinner_only=True,
                skill="code_search",
            ),
        )
        skill_context = SkillContext(
            user_request=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            context_pack=context_pack,
        )
        search_started = time.perf_counter()
        search = await self._skill_executor.run(
            "code_search",
            skill_context,
            extra_query=f"{task_tree.root_task} {node.description}",
        )
        search_elapsed = time.perf_counter() - search_started
        search_output = search.metadata.get("search_output", "")
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · code_search took {search_elapsed:.1f}s "
                f"(success={search.success}, output={len(search_output)} chars)"
            ),
            data={
                "phase": "executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "code_search",
                "elapsed": search_elapsed,
                "success": search.success,
                "output_chars": len(search_output),
                "summary": search.summary,
            },
        )
        if not search.success:
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"SkillExecutor [{node.id}] · code_search failed: {search.summary}",
                data=_executor_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(node, search.summary)
            return

        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · code_edit",
            data=_executor_status_data(
                subtask_id=node.id,
                spinner_only=True,
                skill="code_edit",
            ),
        )
        edit_started = time.perf_counter()
        edit = await self._skill_executor.run(
            "code_edit",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
            ),
            instruction=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            search_output=search_output,
        )
        edit_elapsed = time.perf_counter() - edit_started
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · code_edit took {edit_elapsed:.1f}s "
                f"(success={edit.success}, changed={list(edit.changed_files)})"
            ),
            data={
                "phase": "executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "code_edit",
                "elapsed": edit_elapsed,
                "success": edit.success,
                "changed_files": list(edit.changed_files),
                "summary": edit.summary,
                "raw_preview": edit.metadata.get("raw_preview", ""),
            },
        )
        if not edit.success:
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"SkillExecutor [{node.id}] · code_edit failed: {edit.summary}",
                data=_executor_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(
                node,
                edit.summary,
                exploration_digest=str(search_output),
            )
            return

        validation_started = time.perf_counter()
        validation = await self._skill_executor.run(
            "validator",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
            ),
            changed_files=edit.changed_files,
        )
        validation_elapsed = time.perf_counter() - validation_started
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · validator took {validation_elapsed:.1f}s "
                f"(success={validation.success}, result={validation.validation_result})"
            ),
            data={
                "phase": "executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "validator",
                "elapsed": validation_elapsed,
                "success": validation.success,
                "validation_result": validation.validation_result,
                "summary": validation.summary,
            },
        )
        if not validation.success:
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"SkillExecutor [{node.id}] · validator failed: {validation.summary}",
                data=_executor_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(
                node,
                validation.summary,
                requires_executor_fallback=False,
            )
            return

        final_message = (
            f"SkillExecutor completed edit.\n"
            f"Changed files: {', '.join(edit.changed_files)}\n"
            f"Validation: {validation.summary}"
        )
        yield final_answer_event(
            final_message,
            intermediate=True,
            subtask_id=node.id,
        )
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "turns_used": 0,
                "success": True,
                "changed_files": list(edit.changed_files),
                "file_diffs": {},
                "error_trace": [],
                "final_message": final_message,
                "failure_code": "",
                "quality_gate_failures": 0,
            },
        )

    def _skill_failure_stream_end(
        self,
        node: SubTaskNode,
        reason: str,
        *,
        requires_executor_fallback: bool = True,
        exploration_digest: str | None = None,
    ) -> AgentEvent:
        return AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "turns_used": 0,
                "success": False,
                "changed_files": [],
                "file_diffs": {},
                "error_trace": [reason],
                "final_message": reason,
                "failure_code": "skill_executor",
                "quality_gate_failures": 0,
                "requires_executor_fallback": requires_executor_fallback,
                "exploration_digest": exploration_digest,
            },
        )

    def _patch_plan_failure_event(
        self,
        reason: str,
        *,
        raw_preview: str = "",
    ) -> AgentEvent:
        preview = _compact_preview(raw_preview)
        suffix = f" Raw preview: {preview}" if preview else ""
        return error_event(
            f"PatchPlan failed: {reason}{suffix}",
            {
                "phase": "patch_plan",
                "terminal": True,
                "fallback": "disabled",
                "raw_preview": preview,
            },
        )

    async def _apply_plan_gate(
        self,
        tree: TaskTree,
        *,
        replan_context: ReplanGateContext | None = None,
    ) -> tuple[TaskTree, str | None]:
        pm = self.harness.phase_metrics
        pm.start("plan_gate")
        result = validate_plan(
            tree,
            self.harness.project_root,
            max_nodes=self.settings.plan_gate_max_nodes,
            replan_context=replan_context,
        )
        pm.end("plan_gate", verdict=result.verdict.value, metadata=result.metadata)

        if result.verdict == GateVerdict.WARN:
            yield_msg = "; ".join(result.messages)
            log.info("PlanGate WARN: %s", yield_msg)

        if result.verdict == GateVerdict.BLOCK:
            return tree, "; ".join(result.messages)
        return tree, None

    async def _handle_preflight_block(
        self,
        user_msg: str,
        task_tree: TaskTree,
        node: SubTaskNode,
        preflight: PreflightResult,
        project_structure: str,
    ) -> TaskTree | None:
        if self.state.replan_count >= self.settings.orchestrator_max_replans:
            return None

        replan_ctx = ReplanGateContext.from_node(node)
        gate_feedback: list[str] = []
        max_gate_attempts = self.settings.plan_gate_max_replans + 1
        new_tree: TaskTree | None = None

        for gate_attempt in range(max_gate_attempts):
            replan_evidence = EvidencePack(
                subtask_id=node.id,
                subtask_description=node.description,
                error_trace=list(preflight.messages) + gate_feedback,
                executor_turns_used=0,
            )
            pm = self.harness.phase_metrics
            pm.start("replan")
            fresh_structure = await self._project_structure()
            candidate = await self._planner.re_plan(
                user_msg,
                task_tree,
                replan_evidence,
                fresh_structure,
                discovery_manifest=self._discovery_block(),
            )
            pm.end(
                "replan",
                metadata={
                    "trigger": "preflight_block",
                    "subtask_id": node.id,
                    "gate_attempt": gate_attempt + 1,
                },
            )

            candidate, err = await self._apply_plan_gate(
                candidate,
                replan_context=replan_ctx,
            )
            if not err:
                new_tree = candidate
                break
            gate_feedback.append(f"PlanGate rejected re-plan: {err}")

        if new_tree is None:
            log.warning(
                "Re-plan after preflight failed PlanGate: %s",
                gate_feedback[-1] if gate_feedback else "unknown",
            )
            return None

        self.state.replan_count += 1
        return new_tree

    async def _project_structure(self) -> str:
        await self._await_repo_map()
        ctx = await self.context._build_project_context()
        return ctx or f"Root: {self.harness.project_root}"

    async def _await_repo_map(self) -> None:
        service = getattr(self.context, "repo_map_service", None)
        if service is None:
            return
        timeout = self.settings.repo_map_build_timeout
        ready = await asyncio.to_thread(service.wait_until_ready, timeout)
        if not ready and service.build_error:
            log.warning("Repo map not ready: %s", service.build_error)

    async def _repo_map_for_preflight(self):
        await self._await_repo_map()
        service = getattr(self.context, "repo_map_service", None)
        if service is None:
            return None
        return service.map

    async def _context_pack_for_planner(self, user_msg: str):
        if not self.settings.context_retriever_enabled:
            return None
        await self._await_repo_map()
        service = getattr(self.context, "repo_map_service", None)
        retriever = ContextRetriever(
            project_root=self.harness.project_root,
            repo_map=service,
        )
        return await asyncio.to_thread(
            retriever.retrieve,
            user_msg,
            task_template="planner",
        )

    def _sync_repo_map_after_exec(
        self,
        node: SubTaskNode,
        exec_result: dict[str, Any],
    ) -> None:
        service = getattr(self.context, "repo_map_service", None)
        if service is None:
            return
        changed = [
            p for p in (exec_result.get("changed_files") or []) if isinstance(p, str)
        ]
        if changed:
            for path in changed:
                service.mark_dirty(path)
                self.context.file_tracker.record_edit(path)
            return
        if node.kind == SubTaskKind.EDIT:
            service.mark_dirty()

    def _discovery_block(self) -> str | None:
        if self.state.discovery_manifest is None:
            return None
        return self.state.discovery_manifest.to_planner_block()

    async def _discovery_phase(
        self,
        user_msg: str,
        project_structure: str,
        skip_discovery: bool,
    ) -> AsyncIterator[AgentEvent]:
        """Harness AOP pre-hook: Scout runs before Planner unless /plan skip."""
        pm = self.harness.phase_metrics
        pm.start("scout")

        if skip_discovery or not self.settings.scout_enabled:
            reason = (
                "direct /plan mode"
                if skip_discovery
                else "scout disabled (Planner uses repo context per subtask)"
            )
            manifest = DiagnosticsManifest.skipped_manifest(user_msg, project_structure)
            self.state.discovery_manifest = manifest
            pm.end("scout", verdict="SKIP", metadata={"reason": reason})
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Discovery skipped ({reason}).",
            )
            return

        async for event in self._scout.run(user_msg, project_structure):
            yield event

        manifest = self._scout.last_manifest
        if manifest is None:
            manifest = DiagnosticsManifest(
                user_request=user_msg,
                uncertainties=["Scout produced no manifest."],
            )
        self.state.discovery_manifest = manifest

        gate = validate_manifest(manifest)
        pm.end(
            "scout",
            verdict=gate.verdict.value,
            metadata={
                "turns": manifest.scout_turns_used,
                "victim_files": len(manifest.victim_files),
            },
        )
        if gate.verdict == GateVerdict.WARN and gate.messages:
            yield AgentEvent(
                type=EventType.STATUS,
                content="Manifest: " + "; ".join(gate.messages),
            )

        cp_id = await self._save_discovery_checkpoint(manifest)
        if cp_id:
            yield AgentEvent(
                type=EventType.CHECKPOINT_SAVED,
                content=f"Discovery checkpoint {cp_id}",
                data={"checkpoint_id": cp_id, "trigger": "discovery"},
            )

    async def _save_discovery_checkpoint(self, manifest: DiagnosticsManifest) -> str:
        import json

        self.state.agent_state.current_plan = json.dumps(
            manifest.to_dict(), ensure_ascii=False, indent=2
        )
        return await self.harness.save_checkpoint("discovery", self.state.agent_state) or ""

    async def _save_subtask_checkpoint(self, subtask_id: str, task_tree: TaskTree) -> str:
        self.state.agent_state.current_plan = task_tree.to_json()
        cp_id = await self.harness.save_checkpoint(
            f"subtask_success:{subtask_id}",
            self.state.agent_state,
        )
        return cp_id or ""

    def _build_summary(self, task_tree: TaskTree) -> str:
        return build_terminal_run_summary(task_tree, self.state.subtask_summaries)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        return await self.harness.checkpoint_store.list_checkpoints()

    def get_probe_metrics(self) -> dict[str, Any]:
        usage = self.harness.probe.metrics.get_summary()
        phases = self.harness.phase_metrics.get_summary()
        return {**usage, **phases}

    async def run_score_now(self) -> dict[str, Any] | None:
        from src.harness.quality_gate import evaluate_quality_gate

        changed = self.state.agent_state.file_changes[-10:]
        if not changed:
            return None
        root = self.state.task_tree.root_task if self.state.task_tree else "manual /score"
        return await evaluate_quality_gate(
            self.harness,
            user_msg=root,
            changed_files=changed,
        )


def _apply_context_pack_to_task_tree(task_tree: TaskTree, context_pack: Any | None) -> None:
    if context_pack is None:
        return
    files = [
        str(item.get("file") or "").strip()
        for item in getattr(context_pack, "candidate_files", ())[:5]
        if isinstance(item, dict) and str(item.get("file") or "").strip()
    ]
    if not files:
        files = list(getattr(context_pack, "relevant_files", ())[:5])
    if not files:
        return
    for node in task_tree.nodes:
        if node.context_files:
            continue
        if node.kind in {SubTaskKind.DIAGNOSE, SubTaskKind.EDIT}:
            node.context_files = list(files)
