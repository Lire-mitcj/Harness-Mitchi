from __future__ import annotations

import asyncio
import re
import json
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
from src.executor.final_output import parse_executor_final
from src.harness.discovery.input_parser import parse_turn_input
from src.harness.discovery.manifest import DiagnosticsManifest, manifest_actionable
from src.harness.discovery.manifest_gate import validate_manifest
from src.harness.discovery.scout_agent import ScoutAgent
from src.harness.gates.exit_gate import ExitCheckInput, validate_exit
from src.harness.gates.plan_gate import ReplanGateContext, validate_plan
from src.harness.gates.preflight_probe import assess_preflight
from src.harness.gates.types import GateVerdict, PreflightResult, TruncationPolicy
from src.harness.task_analysis import HarnessTaskAnalysis, analyze_task
from src.harness.subtask.handoff import (
    collect_prior_summaries,
    commit_subtask_failure,
    commit_subtask_success,
)
from src.llm.client import LLMClient
from src.orchestrator.escalation import EscalationAction, decide_subtask_escalation
from src.orchestrator.evidence import EvidencePack
from src.orchestrator.final_summarizer import (
    FinalSummarizer,
    build_deterministic_user_summary,
)
from src.orchestrator.handoff_contract import (
    build_handoff_contract,
    extract_handoff_contract,
    format_handoff_contract,
    merge_handoff_contracts,
)
from src.planner.context_policy import effective_context_files
from src.planner.kinds import SubTaskKind
from src.planner.patch_plan_parse import parse_patch_plan_output
from src.planner.planner_node import (
    LiteLLMPlannerClient,
    PlannerNode,
    fallback_replan_for_failed_edit,
    fallback_task_tree,
    parse_planner_output,
)
from src.planner.scout_skip import apply_scout_discovery_to_plan
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree
from src.skills import (
    CodeEditSkill,
    CodeSearchSkill,
    DesignSkill,
    SkillContext,
    SkillExecutor,
    ValidatorSkill,
    VerifySkill,
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


def _primary_edit_target_from_metadata(
    metadata: dict[str, str],
) -> tuple[str, str]:
    raw = str(metadata.get("edit_context_json") or "").strip()
    if not raw:
        return "", ""
    try:
        ctx = json.loads(raw)
    except json.JSONDecodeError:
        return "", ""
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("edit_targets")
    if not isinstance(targets, list) or not targets:
        return "", ""
    first = targets[0]
    if not isinstance(first, dict):
        return "", ""
    return (
        str(first.get("file") or "").strip(),
        str(first.get("symbol") or first.get("name") or "").strip(),
    )


def _restore_original_files(project_root: Any, raw: str) -> str:
    if not raw:
        return ""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "failed to parse original file snapshot"
    if not isinstance(payload, dict) or not payload:
        return ""
    root = project_root.resolve()
    restored: list[str] = []
    errors: list[str] = []
    for rel, content in payload.items():
        if not isinstance(rel, str) or not isinstance(content, str):
            continue
        try:
            path = (root / rel.replace("\\", "/").lstrip("./")).resolve()
            path.relative_to(root)
            path.write_text(content, encoding="utf-8")
            restored.append(rel)
        except (OSError, ValueError) as exc:
            errors.append(f"{rel}: {exc}")
    if errors:
        return "restored " + ", ".join(restored) + "; errors: " + "; ".join(errors)
    return "restored " + ", ".join(restored)


def _estimate_prompt_tokens(messages: list[dict[str, Any]]) -> int:
    text = "\n".join(str(message.get("content") or "") for message in messages)
    return max(1, len(text) // 4)


def _skill_status_data(**data: Any) -> dict[str, Any]:
    return {"phase": "skill_executor", "executor_activity": True, **data}


@dataclass
class OrchestratorState:
    """Runtime bus state for the Planner-driven control loop."""

    task_tree: TaskTree | None = None
    agent_state: AgentState = field(default_factory=AgentState)
    replan_count: int = 0
    plan_gate_replans: int = 0
    discovery_manifest: DiagnosticsManifest | None = None
    subtask_attempts: dict[str, int] = field(default_factory=dict)
    failure_fingerprints: dict[str, int] = field(default_factory=dict)
    subtask_summaries: dict[str, str] = field(default_factory=dict)
    subtask_exploration_digests: dict[str, str] = field(default_factory=dict)
    task_analysis: HarnessTaskAnalysis | None = None


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
        self._final_summarizer = FinalSummarizer(
            LLMClient(
                model=settings.effective_final_summary_model,
                max_tokens=settings.final_summary_max_tokens,
                request_timeout=settings.final_summary_timeout,
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=_cache_ttl(settings.prompt_cache_ttl),
            ),
            max_tokens=settings.final_summary_max_tokens,
            timeout=float(settings.final_summary_timeout),
        )
        self._skill_executor = SkillExecutor([
            CodeEditSkill(
                project_root=harness.project_root,
                llm_complete=self._planner.client.complete,
            ),
            CodeSearchSkill(project_root=harness.project_root, tools=tools),
            DesignSkill(),
            ValidatorSkill(project_root=harness.project_root),
            VerifySkill(project_root=harness.project_root),
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
        self.state.task_analysis = analyze_task(user_msg, context_pack)
        analysis_block = self.state.task_analysis.to_planner_block()
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
        discovery_block = (
            f"{discovery_block}\n\n{analysis_block}"
            if discovery_block
            else analysis_block
        )
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
            subtask_context_pack = await self._context_pack_for_subtask(
                task_tree=task_tree,
                node=node,
            )
            _apply_context_pack_to_subtask(node, subtask_context_pack)
            if subtask_context_pack is not None:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        f"ContextBuilder [{node.id}]: "
                        f"confidence={subtask_context_pack.confidence:.2f}, "
                        f"files={len(subtask_context_pack.relevant_files)}, "
                        f"snippets={len(subtask_context_pack.focused_snippets or subtask_context_pack.snippets)}"
                    ),
                    data={
                        "phase": "context_builder",
                        "subtask_id": node.id,
                        "confidence": subtask_context_pack.confidence,
                        "files": list(subtask_context_pack.relevant_files),
                        "missing_info": list(subtask_context_pack.missing_info),
                    },
                )

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
            prior_errors = list(node.error_trace)
            prior_ctx = collect_prior_summaries(
                task_tree, node, self.state.subtask_summaries
            )
            attempt_num = self.state.subtask_attempts.get(node.id, 0) + 1
            prior_exploration = self.state.subtask_exploration_digests.get(node.id)
            from src.planner.kinds import SubTaskKind

            if node.kind == SubTaskKind.DIAGNOSE:
                async for event in self._run_diagnose_skill_executor(
                    task_tree=task_tree,
                    node=node,
                    context_pack=subtask_context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if "success" in data or data.get("failure_code"):
                            exec_result = data
                    yield event
            elif node.kind == SubTaskKind.DESIGN:
                async for event in self._run_design_skill_executor(
                    task_tree=task_tree,
                    node=node,
                    context_pack=subtask_context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if "success" in data or data.get("failure_code"):
                            exec_result = data
                    yield event
            elif node.kind == SubTaskKind.EDIT:
                async for event in self._run_edit_skill_executor(
                    user_msg=user_msg,
                    task_tree=task_tree,
                    node=node,
                    project_structure=project_structure,
                    context_pack=subtask_context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if "success" in data or data.get("failure_code"):
                            exec_result = data
                    yield event
            elif node.kind == SubTaskKind.VERIFY:
                async for event in self._run_verify_skill_executor(
                    user_msg=user_msg,
                    task_tree=task_tree,
                    node=node,
                    context_pack=subtask_context_pack,
                ):
                    if event.type == EventType.STREAM_END and event.data:
                        data = event.data
                        if "success" in data or data.get("failure_code"):
                            exec_result = data
                    yield event
            else:
                yield error_event(
                    f"Unsupported subtask kind: {node.kind}",
                    {"subtask_id": node.id},
                )
                break

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
            base_tree = task_tree
            accepted_replan: TaskTree | None = None

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
                    context_files=effective_context_files(base_tree, node),
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
                candidate_tree = await self._planner.re_plan(
                    user_msg,
                    base_tree,
                    replan_evidence,
                    fresh_structure,
                    discovery_manifest=self._planner_context_block(),
                )
                pm.end(
                    "replan",
                    metadata={
                        "trigger": "executor_fail",
                        "subtask_id": node.id,
                        "gate_attempt": gate_attempt + 1,
                    },
                )

                candidate_tree, plan_error = await self._apply_plan_gate(
                    candidate_tree,
                    replan_context=replan_ctx,
                )
                if not plan_error:
                    accepted_replan = candidate_tree
                    break
                gate_feedback.append(f"PlanGate rejected re-plan: {plan_error}")

            if plan_error:
                fallback_tree = fallback_replan_for_failed_edit(
                    base_tree,
                    failed_subtask_id=node.id,
                    task_analysis=self.state.task_analysis,
                    error_trace=replan_errors,
                )
                if fallback_tree is not None:
                    fallback_tree, fallback_error = await self._apply_plan_gate(
                        fallback_tree,
                        replan_context=replan_ctx,
                    )
                    if not fallback_error:
                        accepted_replan = fallback_tree
                        plan_error = None
                        yield AgentEvent(
                            type=EventType.STATUS,
                            content=(
                                f"Re-plan fallback inserted diagnose+edit for [{node.id}]"
                            ),
                            data={"phase": "planner", "subtask_id": node.id},
                        )
                    else:
                        plan_error = f"{plan_error}; fallback: {fallback_error}"
                if plan_error:
                    yield error_event(plan_error, {"subtask_id": node.id})
                    break

            if accepted_replan is None:
                yield error_event(
                    "internal error: re-plan finished without accepted TaskTree",
                    {"subtask_id": node.id},
                )
                break
            task_tree = accepted_replan
            self.state.replan_count += 1
            self.state.subtask_attempts.clear()
            self.state.subtask_exploration_digests.clear()
            self.state.task_tree = task_tree
            self.state.agent_state.current_plan = task_tree.to_json()
            yield plan_update_event(task_tree)

        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                "FinalSummarizer · calling model "
                f"({self.settings.effective_final_summary_model})"
            ),
            data={
                "phase": "final_summarizer",
                "spinner_only": True,
                "llm_loading": True,
                "model": self.settings.effective_final_summary_model,
            },
        )
        summary_start = time.perf_counter()
        summary, summary_fallback = await self._build_summary(user_msg, task_tree)
        summary_elapsed = time.perf_counter() - summary_start
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"FinalSummarizer · model call took {summary_elapsed:.1f}s "
                f"(model={self.settings.effective_final_summary_model}, "
                f"output={len(summary)} chars, fallback={summary_fallback})"
            ),
            data={
                "phase": "final_summarizer",
                "model": self.settings.effective_final_summary_model,
                "elapsed": summary_elapsed,
                "output_chars": len(summary),
                "fallback": summary_fallback,
            },
        )
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
                data={"spinner_only": True, "llm_loading": True, "phase": "planner"},
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
        fallback = fallback_task_tree(user_msg, task_analysis=self.state.task_analysis)
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
            data={"spinner_only": True, "llm_loading": True, "phase": "patch_plan"},
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

    async def _run_diagnose_skill_executor(
        self,
        *,
        task_tree: TaskTree,
        node: SubTaskNode,
        context_pack: Any | None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · code_search",
            data=_skill_status_data(
                subtask_id=node.id,
                spinner_only=True,
                skill="code_search",
            ),
        )
        skill_context = SkillContext(
            user_request=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            context_pack=context_pack,
        )
        started = time.perf_counter()
        search = await self._skill_executor.run(
            "code_search",
            skill_context,
            extra_query=f"{task_tree.root_task} {node.description} {node.acceptance_criteria}",
        )
        elapsed = time.perf_counter() - started
        search_output = str(search.metadata.get("search_output", ""))
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · code_search took {elapsed:.1f}s "
                f"(success={search.success}, output={len(search_output)} chars, "
                f"edit_context_targets={search.metadata.get('edit_context_targets', '0')}, "
                f"hydration_hits={search.metadata.get('hydration_hits', '0')}, "
                f"hydration_version={search.metadata.get('hydration_version', 'unknown')}"
                + _hydration_hit_paths_suffix(search.metadata)
                + _hydration_failure_suffix(search.metadata)
                + ")"
            ),
            data={
                "phase": "skill_executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "code_search",
                "elapsed": elapsed,
                "success": search.success,
                "output_chars": len(search_output),
                "edit_context_targets": search.metadata.get("edit_context_targets", "0"),
                "hydration_hits": search.metadata.get("hydration_hits", "0"),
                "hydration_root": search.metadata.get("hydration_root", ""),
                "hydration_failures": search.metadata.get("hydration_failures", ""),
                "hydration_version": search.metadata.get("hydration_version", ""),
                "hydration_hit_paths": search.metadata.get("hydration_hit_paths", ""),
                "summary": search.summary,
            },
        )
        if not search.success:
            yield self._skill_failure_stream_end(
                node,
                search.summary,
                exploration_digest=search_output,
            )
            return

        final_message = _diagnose_summary_from_digest(node, search_output, [])
        final_message = _append_current_edit_context(final_message, search)
        contract = build_handoff_contract(
            user_request=task_tree.root_task,
            subtask_id=node.id,
            summary=final_message,
            search_output=search_output,
        )
        final_message = final_message + "\n\n" + format_handoff_contract(contract)
        exit_result = validate_exit(
            ExitCheckInput(
                subtask=node,
                final_message=final_message,
                error_trace=[],
                changed_files=[],
                turns_used=0,
                tool_failure_count=0,
            )
        )
        if not exit_result.passed:
            yield self._skill_failure_stream_end(
                node,
                "; ".join(exit_result.messages),
                exploration_digest=search_output,
                final_message=final_message,
            )
            return

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
                "changed_files": [],
                "file_diffs": {},
                "error_trace": [],
                "final_message": final_message,
                "failure_code": "",
                "quality_gate_failures": 0,
                "exploration_digest": search_output,
                "skill_executor": True,
            },
        )

    async def _run_design_skill_executor(
        self,
        *,
        task_tree: TaskTree,
        node: SubTaskNode,
        context_pack: Any | None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · code_search",
            data=_skill_status_data(
                subtask_id=node.id,
                spinner_only=True,
                skill="code_search",
            ),
        )
        skill_context = SkillContext(
            user_request=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            context_pack=context_pack,
        )
        started = time.perf_counter()
        search = await self._skill_executor.run(
            "code_search",
            skill_context,
            extra_query=f"{task_tree.root_task} {node.description} {node.acceptance_criteria}",
            task_analysis=(
                self.state.task_analysis.to_dict()
                if self.state.task_analysis is not None
                else {}
            ),
        )
        elapsed = time.perf_counter() - started
        search_output = str(search.metadata.get("search_output", ""))
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · code_search took {elapsed:.1f}s "
                f"(success={search.success}, output={len(search_output)} chars)"
            ),
            data=_skill_status_data(
                subtask_id=node.id,
                skill="code_search",
                elapsed=elapsed,
                success=search.success,
            ),
        )
        if not search.success:
            yield self._skill_failure_stream_end(
                node,
                search.summary,
                exploration_digest=search_output,
            )
            return

        prior_ctx = collect_prior_summaries(
            task_tree,
            node,
            self.state.subtask_summaries,
        )
        search_output = _append_prior_edit_context(search_output, prior_ctx)
        contract = merge_handoff_contracts(
            user_request=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            prior_summaries=prior_ctx,
            current_search_output=search_output,
            global_summaries=self.state.subtask_summaries,
        )
        analysis_dict = (
            self.state.task_analysis.to_dict()
            if self.state.task_analysis is not None
            else {}
        )
        design = await self._skill_executor.run(
            "design",
            SkillContext(
                user_request=task_tree.root_task,
                context_pack=context_pack,
            ),
            search_output=search_output,
            handoff_contract=contract,
            task_analysis=analysis_dict,
        )
        final_message = design.metadata.get("final_message", design.summary)
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · design (success={design.success})",
            data=_skill_status_data(
                subtask_id=node.id,
                skill="design",
                success=design.success,
                summary=design.summary,
            ),
        )
        if not design.success:
            yield self._skill_failure_stream_end(
                node,
                design.summary,
                exploration_digest=search_output,
                final_message=final_message,
            )
            return

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
                "changed_files": [],
                "file_diffs": {},
                "error_trace": [],
                "final_message": final_message,
                "failure_code": "",
                "quality_gate_failures": 0,
                "exploration_digest": search_output,
                "skill_executor": True,
            },
        )

    async def _run_edit_skill_executor(
        self,
        *,
        user_msg: str,
        task_tree: TaskTree,
        node: SubTaskNode,
        project_structure: str,
        context_pack: Any | None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · code_search",
            data=_skill_status_data(
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
                f"(success={search.success}, output={len(search_output)} chars, "
                f"edit_context_targets={search.metadata.get('edit_context_targets', '0')}, "
                f"hydration_hits={search.metadata.get('hydration_hits', '0')}, "
                f"hydration_version={search.metadata.get('hydration_version', 'unknown')}"
                + _hydration_hit_paths_suffix(search.metadata)
                + _hydration_failure_suffix(search.metadata)
                + ")"
            ),
            data={
                "phase": "skill_executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "code_search",
                "elapsed": search_elapsed,
                "success": search.success,
                "output_chars": len(search_output),
                "edit_context_targets": search.metadata.get("edit_context_targets", "0"),
                "hydration_hits": search.metadata.get("hydration_hits", "0"),
                "hydration_root": search.metadata.get("hydration_root", ""),
                "hydration_failures": search.metadata.get("hydration_failures", ""),
                "hydration_version": search.metadata.get("hydration_version", ""),
                "hydration_hit_paths": search.metadata.get("hydration_hit_paths", ""),
                "summary": search.summary,
            },
        )
        if not search.success:
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"SkillExecutor [{node.id}] · code_search failed: {search.summary}",
                data=_skill_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(node, search.summary)
            return
        if str(search.metadata.get("edit_context_targets", "0")) == "0":
            reason = (
                "code_search found no editable SQL/query target for this edit. "
                "Known negative: the current hits are not valid edit targets for "
                "a view-query change; do not re-edit those symbols. Re-plan must "
                "locate the actual query/API function containing SELECT/FROM/JOIN "
                "before scheduling edit_file."
            )
            failures = str(search.metadata.get("hydration_failures") or "").strip()
            if failures:
                reason += f" Hydration failures: {failures}"
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"SkillExecutor [{node.id}] · code_search no edit target: {reason}",
                data=_skill_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(
                node,
                reason,
                exploration_digest=str(search_output),
            )
            return

        target_file, target_symbol = _primary_edit_target_from_metadata(search.metadata)
        edit_label = f"code_edit · {target_file}" if target_file else "code_edit"
        if target_symbol and target_file:
            edit_label += f" · {target_symbol}"
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · {edit_label}",
            data=_skill_status_data(
                subtask_id=node.id,
                spinner_only=True,
                llm_loading=True,
                skill="code_edit",
                target_file=target_file,
                target_symbol=target_symbol,
            ),
        )
        edit_started = time.perf_counter()
        prior_ctx = collect_prior_summaries(
            task_tree,
            node,
            self.state.subtask_summaries,
        )
        search_output = _append_prior_edit_context(str(search_output), prior_ctx)
        contract = merge_handoff_contracts(
            user_request=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            prior_summaries=prior_ctx,
            current_search_output=str(search_output),
        )
        edit = await self._skill_executor.run(
            "code_edit",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
            ),
            instruction=f"{task_tree.root_task}\n\nSubtask: {node.description}",
            search_output=search_output,
            handoff_contract=contract,
        )
        edit_elapsed = time.perf_counter() - edit_started
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · code_edit took {edit_elapsed:.1f}s "
                f"(success={edit.success}, changed={list(edit.changed_files)})"
            ),
            data={
                "phase": "skill_executor",
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
            failure_type = str(edit.missing_info[0] if edit.missing_info else "code_edit")
            fingerprint = "|".join([
                target_symbol,
                _edit_operation_from_search_output(str(search_output)),
                failure_type,
                ",".join(edit.changed_files),
            ])
            fingerprint_count = self.state.failure_fingerprints.get(fingerprint, 0) + 1
            self.state.failure_fingerprints[fingerprint] = fingerprint_count
            edit_preview = _compact_preview(edit.metadata.get("raw_preview", ""))
            preview_suffix = f" raw_preview={edit_preview}" if edit_preview else ""
            yield AgentEvent(
                type=EventType.STATUS,
                content=(
                    f"SkillExecutor [{node.id}] · code_edit failed: "
                    f"{edit.summary}{preview_suffix}"
                ),
                data=_skill_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(
                node,
                edit.summary,
                requires_executor_fallback=fingerprint_count < 2,
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
            handoff_contract=contract,
        )
        validation_elapsed = time.perf_counter() - validation_started
        content = (
            f"SkillExecutor [{node.id}] · validator took {validation_elapsed:.1f}s "
            f"(success={validation.success}, result={validation.validation_result})"
        )
        if validation.success and validation.summary:
            content += f" — {validation.summary}"
        yield AgentEvent(
            type=EventType.STATUS,
            content=content,
            data={
                "phase": "skill_executor",
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
            rollback_summary = _restore_original_files(
                self.harness.project_root,
                edit.metadata.get("original_files_json", ""),
            )
            rollback_suffix = f"; rollback: {rollback_summary}" if rollback_summary else ""
            yield AgentEvent(
                type=EventType.STATUS,
                content=(
                    f"SkillExecutor [{node.id}] · validator failed: "
                    f"{validation.summary}{rollback_suffix}"
                ),
                data=_skill_status_data(subtask_id=node.id, skill_error=True),
            )
            yield self._skill_failure_stream_end(
                node,
                validation.summary + rollback_suffix,
                requires_executor_fallback=validation.requires_fallback,
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

    async def _run_verify_skill_executor(
        self,
        *,
        user_msg: str,
        task_tree: TaskTree,
        node: SubTaskNode,
        context_pack: Any | None,
    ) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STATUS,
            content=f"SkillExecutor [{node.id}] · verify",
            data=_skill_status_data(
                subtask_id=node.id,
                spinner_only=True,
                skill="verify",
            ),
        )
        changed_files = self.context.file_tracker.get_modified_files()
        started = time.perf_counter()
        verify_result = await self._skill_executor.run(
            "verify",
            SkillContext(
                user_request=user_msg,
                context_pack=context_pack,
            ),
            changed_files=changed_files,
        )
        elapsed = time.perf_counter() - started
        yield AgentEvent(
            type=EventType.STATUS,
            content=(
                f"SkillExecutor [{node.id}] · verify took {elapsed:.1f}s "
                f"(success={verify_result.success}, result={verify_result.validation_result})"
            ),
            data={
                "phase": "skill_executor",
                "executor_activity": True,
                "subtask_id": node.id,
                "skill": "verify",
                "elapsed": elapsed,
                "success": verify_result.success,
                "summary": verify_result.summary,
            },
        )
        if not verify_result.success:
            yield self._skill_failure_stream_end(
                node,
                verify_result.summary,
                requires_executor_fallback=False,
            )
            return

        final_message = (
            f"SkillExecutor completed verification.\n"
            f"Validation: {verify_result.summary}"
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
                "changed_files": [],
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
        final_message: str | None = None,
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
                "final_message": final_message or reason,
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
            task_analysis=self.state.task_analysis,
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
                discovery_manifest=self._planner_context_block(),
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

    async def _context_pack_for_subtask(
        self,
        *,
        task_tree: TaskTree,
        node: SubTaskNode,
    ):
        if not self.settings.context_retriever_enabled:
            return None
        await self._await_repo_map()
        service = getattr(self.context, "repo_map_service", None)
        retriever = ContextRetriever(
            project_root=self.harness.project_root,
            repo_map=service,
        )
        prior = collect_prior_summaries(
            task_tree,
            node,
            self.state.subtask_summaries,
        )
        request = _subtask_context_request(task_tree.root_task, node)
        return await asyncio.to_thread(
            retriever.build,
            user_request=request,
            task_template=f"subtask:{node.kind.value}",
            current_files=tuple(effective_context_files(task_tree, node)),
            recent_files=tuple(self.state.agent_state.file_changes[-5:]),
            previous_handoff=_previous_handoff_from_summaries(prior),
            mode=node.kind.value,
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

    def _planner_context_block(self) -> str | None:
        blocks: list[str] = []
        discovery = self._discovery_block()
        if discovery:
            blocks.append(discovery)
        if self.state.task_analysis is not None:
            blocks.append(self.state.task_analysis.to_planner_block())
        return "\n\n".join(blocks) if blocks else None

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

    async def _build_summary(self, user_msg: str, task_tree: TaskTree) -> tuple[str, bool]:
        fallback = build_deterministic_user_summary(
            user_request=user_msg,
            task_tree=task_tree,
            subtask_summaries=self.state.subtask_summaries,
        )
        if not fallback.strip():
            fallback = build_terminal_run_summary(task_tree, self.state.subtask_summaries)
        try:
            summary = await self._final_summarizer.summarize(
                user_request=user_msg,
                task_tree=task_tree,
                subtask_summaries=self.state.subtask_summaries,
            )
        except Exception as exc:
            log.warning("Final summarizer failed; using template summary: %s", exc)
            return fallback, True
        if not summary:
            return fallback, True
        return summary, False

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


def _subtask_context_request(root_task: str, node: SubTaskNode) -> str:
    parts = [
        f"Root task: {root_task}",
        f"Subtask [{node.id}] kind={node.kind.value}: {node.description}",
    ]
    if node.acceptance_criteria:
        parts.append(f"Acceptance: {node.acceptance_criteria}")
    if node.context_files:
        parts.append("Context files: " + ", ".join(node.context_files))
    if node.depends_on:
        parts.append("Depends on: " + ", ".join(node.depends_on))
    return "\n".join(parts)

def _previous_handoff_from_summaries(summaries: dict[str, str]) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    known_negatives: list[dict[str, Any]] = []
    next_focus: list[dict[str, Any]] = []
    for sid, text in summaries.items():
        parsed = parse_executor_final(text)
        if parsed is not None:
            for item in parsed.evidence:
                evidence.append({"source_subtask": sid, **item})
            if parsed.handoff:
                for item in parsed.handoff.get("facts", []) or []:
                    if isinstance(item, dict):
                        facts.append({"source_subtask": sid, **item})
                    elif str(item).strip():
                        facts.append({"source_subtask": sid, "fact": str(item)})
                for item in parsed.handoff.get("known_negatives", []) or []:
                    if isinstance(item, dict):
                        known_negatives.append({"source_subtask": sid, **item})
                for item in parsed.handoff.get("next_focus", []) or []:
                    if isinstance(item, dict):
                        next_focus.append({"source_subtask": sid, **item})
                    elif str(item).strip():
                        next_focus.append({"source_subtask": sid, "focus": str(item)})
            if parsed.blocker:
                known_negatives.append({"source_subtask": sid, "reason": parsed.blocker})
            continue
        lower = text.lower()
        if any(marker in lower for marker in ("no matches", "not found", "没有命中", "未找到")):
            known_negatives.append({"source_subtask": sid, "reason": text[:500]})
    return {
        "facts": facts[:20],
        "evidence": evidence[:30],
        "known_negatives": known_negatives[:12],
        "next_focus": next_focus[:20],
    }


def _hydration_failure_suffix(metadata: dict[str, str]) -> str:
    failures = str(metadata.get("hydration_failures") or "").strip()
    if not failures:
        return ""
    compact = " ".join(failures.split())
    if len(compact) > 180:
        compact = compact[:177] + "..."
    return f", hydration_failures={compact}"


def _hydration_hit_paths_suffix(metadata: dict[str, str]) -> str:
    hit_paths = str(metadata.get("hydration_hit_paths") or "").strip()
    if not hit_paths:
        return ""
    compact = " ".join(hit_paths.split())
    if len(compact) > 160:
        compact = compact[:157] + "..."
    return f", hydration_paths={compact}"


def _append_current_edit_context(final_message: str, search_result: Any) -> str:
    metadata = getattr(search_result, "metadata", {}) or {}
    edit_context = str(metadata.get("edit_context_json") or "").strip()
    if not edit_context or "EDIT_CONTEXT_JSON" in final_message:
        return final_message
    return final_message + "\n\nEDIT_CONTEXT_JSON\n" + edit_context


def _append_prior_edit_context(search_output: str, prior_summaries: dict[str, str]) -> str:
    prior_block = ""
    for text in prior_summaries.values():
        pb = _extract_marker_json_block(text, "EDIT_CONTEXT_JSON")
        if pb:
            prior_block = pb
            break
            
    if not prior_block:
        return search_output

    try:
        prior_ctx = json.loads(prior_block)
    except Exception:
        return search_output

    if "EDIT_CONTEXT_JSON" not in search_output:
        return search_output + "\n\nEDIT_CONTEXT_JSON\n" + prior_block

    current_block = _extract_marker_json_block(search_output, "EDIT_CONTEXT_JSON")
    if not current_block:
        return search_output

    try:
        current_ctx = json.loads(current_block)
        modified = False
        if not current_ctx.get("target_view") and prior_ctx.get("target_view"):
            current_ctx["target_view"] = prior_ctx["target_view"]
            modified = True
        if not current_ctx.get("available_views") and prior_ctx.get("available_views"):
            current_ctx["available_views"] = prior_ctx["available_views"]
            modified = True
        
        if modified:
            new_block = json.dumps(current_ctx, ensure_ascii=False, indent=2)
            prefix = search_output.split("EDIT_CONTEXT_JSON", 1)[0]
            return prefix + "EDIT_CONTEXT_JSON\n" + new_block
    except Exception:
        pass

    return search_output


def _append_effective_edit_context(
    search_output: str,
    *,
    handoff_contract: dict[str, Any],
    edit_analysis: dict[str, Any],
) -> str:
    current_block = _extract_marker_json_block(search_output, "EDIT_CONTEXT_JSON")
    if not current_block:
        return search_output
    try:
        current_ctx = json.loads(current_block)
    except Exception:
        return search_output
    if not isinstance(current_ctx, dict):
        return search_output

    patch_intent = edit_analysis.get("patch_intent")
    patch = patch_intent if isinstance(patch_intent, dict) else {}
    contract = handoff_contract if isinstance(handoff_contract, dict) else {}
    effective = dict(current_ctx)

    def first_non_empty(*values: object) -> object:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value
            if isinstance(value, list) and value:
                return value
            if isinstance(value, dict) and value:
                return value
            if value not in (None, "", [], {}):
                return value
        return ""

    target_view = first_non_empty(
        patch.get("target_view"),
        contract.get("target_view"),
        edit_analysis.get("target_view"),
        effective.get("target_view"),
    )
    if isinstance(target_view, str):
        effective["target_view"] = target_view.strip()

    strategy = first_non_empty(
        patch.get("edit_strategy"),
        edit_analysis.get("edit_strategy"),
        effective.get("edit_strategy"),
    )
    if isinstance(strategy, str) and strategy.strip():
        effective["edit_strategy"] = strategy.strip()

    dependencies = first_non_empty(
        patch.get("dependencies"),
        patch.get("dependencies_to_use"),
        patch.get("resolved_dependencies"),
        contract.get("dependencies"),
        contract.get("dependencies_to_use"),
        contract.get("resolved_dependencies"),
        effective.get("resolved_dependencies"),
    )
    if isinstance(dependencies, list):
        effective["resolved_dependencies"] = dependencies
        effective["dependencies"] = dependencies
        effective["dependencies_resolved"] = bool(dependencies)

    available_views = first_non_empty(
        patch.get("available_views"),
        contract.get("available_views"),
        effective.get("available_views"),
    )
    if isinstance(available_views, list):
        effective["available_views"] = available_views

    edit_targets = first_non_empty(
        patch.get("edit_targets"),
        contract.get("edit_targets"),
        edit_analysis.get("edit_targets"),
        effective.get("edit_targets"),
    )
    if isinstance(edit_targets, list):
        effective["edit_targets"] = edit_targets
        _merge_edit_target_metadata(effective, edit_targets)

    acceptance = first_non_empty(
        patch.get("acceptance_criteria"),
        patch.get("acceptance"),
        edit_analysis.get("acceptance_criteria"),
        edit_analysis.get("acceptance"),
        edit_analysis.get("acceptance_contract"),
        contract.get("acceptance_criteria"),
        contract.get("acceptance"),
        contract.get("acceptance_contract"),
        effective.get("acceptance_criteria"),
        effective.get("acceptance"),
    )
    if acceptance:
        effective["acceptance_criteria"] = acceptance
        effective["acceptance"] = acceptance

    if edit_analysis.get("edit_ready") is not None:
        effective["harness_edit_ready"] = bool(edit_analysis.get("edit_ready"))
    if patch:
        effective["patch_intent"] = patch

    new_block = json.dumps(effective, ensure_ascii=False, indent=2)
    prefix = search_output.split("EDIT_CONTEXT_JSON", 1)[0]
    return prefix.rstrip() + "\n\nEDIT_CONTEXT_JSON\n" + new_block


def _merge_edit_target_metadata(
    edit_context: dict[str, Any],
    edit_targets: list[object],
) -> None:
    targets = edit_context.get("editable_targets")
    if not isinstance(targets, list):
        targets = edit_context.get("snippets")
    if not isinstance(targets, list):
        return

    for hydrated in targets:
        if not isinstance(hydrated, dict):
            continue
        hydrated_file = str(hydrated.get("file") or "")
        hydrated_start = hydrated.get("start_line")
        hydrated_end = hydrated.get("end_line")
        for target in edit_targets:
            if not isinstance(target, dict):
                continue
            target_file = str(target.get("file") or "")
            target_start = target.get("line_start") or target.get("start_line")
            target_end = target.get("line_end") or target.get("end_line")
            same_file = target_file and target_file == hydrated_file
            overlaps = (
                isinstance(hydrated_start, int)
                and isinstance(hydrated_end, int)
                and isinstance(target_start, int)
                and isinstance(target_end, int)
                and target_start <= hydrated_end
                and hydrated_start <= target_end
            )
            if same_file and (overlaps or not target_start or not target_end):
                if target.get("symbol") and not hydrated.get("symbol"):
                    hydrated["symbol"] = target.get("symbol")
                if target.get("decision") and not hydrated.get("intended_change"):
                    hydrated["intended_change"] = target.get("decision")
                if target.get("acceptance_criteria") and not hydrated.get("acceptance_criteria"):
                    hydrated["acceptance_criteria"] = target.get("acceptance_criteria")
                break


def _extract_marker_json_block(text: str, marker: str) -> str:
    if marker not in text:
        return ""
    payload = text.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        return ""
    depth = 0
    end = -1
    for idx, ch in enumerate(payload[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    return payload[start : end + 1] if end >= start else ""


def _edit_operation_from_search_output(search_output: str) -> str:
    block = _extract_marker_json_block(search_output, "EDIT_CONTEXT_JSON")
    if not block:
        return ""
    try:
        payload = json.loads(block)
    except json.JSONDecodeError:
        return ""
    task_intent = payload.get("task_intent") if isinstance(payload, dict) else None
    if not isinstance(task_intent, dict):
        return ""
    return str(task_intent.get("operation") or "")


def _apply_context_pack_to_subtask(node: SubTaskNode, context_pack: Any | None) -> None:
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
    if node.context_files:
        return
    if node.kind in {SubTaskKind.DIAGNOSE, SubTaskKind.DESIGN, SubTaskKind.EDIT}:
        node.context_files = list(files)


def _analysis_for_edit(
    analysis: HarnessTaskAnalysis | None,
    prior_summaries: dict[str, str],
    global_summaries: dict[str, str] | None = None,
) -> dict[str, Any]:
    data = analysis.to_dict() if analysis is not None else {}
    patch_intent = _latest_patch_intent(prior_summaries)
    if patch_intent is None and global_summaries is not None:
        patch_intent = _latest_patch_intent(global_summaries)

    if _patch_intent_resolves_readiness(data, patch_intent) and patch_intent is not None:
        targets = patch_intent.get("edit_targets") or []
        targets_resolved = all(
            isinstance(t, dict) and bool(t.get("file"))
            for t in targets
        )
            
        strategy = patch_intent.get("edit_strategy") or data.get("edit_strategy") or ""
        dependencies_resolved = True
        if strategy in ("function_refactor", "sql_view_rewrite"):
            deps = patch_intent.get("dependencies") or patch_intent.get("dependencies_to_use") or []
            dependencies_resolved = all(
                isinstance(dep, dict) and bool(dep.get("name"))
                for dep in deps
            )
                
        acceptance_contract = patch_intent.get("acceptance_contract")
        acceptance_resolved = bool(
            patch_intent.get("acceptance_contract")
            or patch_intent.get("acceptance_criteria")
            or patch_intent.get("acceptance")
        )
        
        intent_resolved = strategy != "unknown" and strategy != ""
        
        unique_files = {
            t.get("file") for t in targets
            if isinstance(t, dict) and t.get("file")
        }
        edit_scope_bounded = len(unique_files) > 0 and len(unique_files) <= 2
        
        checks = {
            "intent_resolved": intent_resolved,
            "targets_resolved": targets_resolved,
            "dependencies_resolved": dependencies_resolved,
            "acceptance_resolved": acceptance_resolved,
            "edit_scope_bounded": edit_scope_bounded,
        }
        data["readiness_checks"] = checks
        data["edit_ready"] = all(checks.values())
        data["patch_intent_resolved"] = True
        data["patch_intent"] = patch_intent
        data["target_view"] = patch_intent.get("target_view") or data.get("target_view") or ""
        
        if strategy:
            data["edit_strategy"] = strategy
        if targets_resolved:
            data["editable_targets"] = targets
            data["edit_targets"] = targets
        if dependencies_resolved:
            dependencies = patch_intent.get("dependencies") or patch_intent.get("dependencies_to_use") or []
            data["resolved_dependencies"] = dependencies
            data["dependencies"] = dependencies
            data["dependencies_to_use"] = dependencies
        if acceptance_resolved:
            if isinstance(acceptance_contract, dict) and acceptance_contract:
                data["acceptance_contract"] = acceptance_contract
            elif patch_intent.get("acceptance_criteria") is not None:
                criteria = patch_intent.get("acceptance_criteria")
                if isinstance(criteria, list):
                    data["acceptance_contract"] = {"criteria": criteria}
                else:
                    data["acceptance_contract"] = {"criteria": [str(criteria)]}
            elif patch_intent.get("acceptance") is not None:
                acc = patch_intent.get("acceptance")
                if isinstance(acc, list):
                    data["acceptance_contract"] = {"criteria": acc}
                else:
                    data["acceptance_contract"] = {"criteria": [str(acc)]}
    else:
        intent_resolved = False
        targets_resolved = False
        dependencies_resolved = False
        acceptance_resolved = False
        edit_scope_bounded = False
        
        if patch_intent is not None:
            strategy = patch_intent.get("edit_strategy") or data.get("edit_strategy") or ""
            intent_resolved = strategy != "unknown" and strategy != ""
            targets = patch_intent.get("edit_targets") or []
            targets_resolved = bool(targets) and all(
                isinstance(t, dict) and bool(t.get("file"))
                for t in targets
            )
            dependencies_resolved = True
            if strategy in ("function_refactor", "sql_view_rewrite"):
                deps = patch_intent.get("dependencies") or patch_intent.get("dependencies_to_use") or []
                dependencies_resolved = all(
                    isinstance(dep, dict) and bool(dep.get("name"))
                    for dep in deps
                )
            acceptance_contract = patch_intent.get("acceptance_contract")
            acceptance_resolved = bool(
                patch_intent.get("acceptance_contract")
                or patch_intent.get("acceptance_criteria")
                or patch_intent.get("acceptance")
            )
            unique_files = {
                t.get("file") for t in targets
                if isinstance(t, dict) and t.get("file")
            }
            edit_scope_bounded = len(unique_files) > 0 and len(unique_files) <= 2
            
        data["readiness_checks"] = {
            "intent_resolved": intent_resolved,
            "targets_resolved": targets_resolved,
            "dependencies_resolved": dependencies_resolved,
            "acceptance_resolved": acceptance_resolved,
            "edit_scope_bounded": edit_scope_bounded,
        }
        data["edit_ready"] = False
        data["patch_intent_resolved"] = patch_intent is not None
        if patch_intent is not None:
            data["patch_intent"] = patch_intent
            data["target_view"] = patch_intent.get("target_view") or data.get("target_view") or ""
            strategy = patch_intent.get("edit_strategy") or data.get("edit_strategy") or ""
            if strategy:
                data["edit_strategy"] = strategy
            targets = patch_intent.get("edit_targets") or []
            if targets_resolved:
                data["editable_targets"] = targets
                data["edit_targets"] = targets
            dependencies = patch_intent.get("dependencies") or patch_intent.get("dependencies_to_use") or []
            if dependencies_resolved:
                data["resolved_dependencies"] = dependencies
                data["dependencies"] = dependencies
                data["dependencies_to_use"] = dependencies
        if patch_intent is not None and acceptance_resolved:
            if isinstance(acceptance_contract, dict) and acceptance_contract:
                data["acceptance_contract"] = acceptance_contract
            elif patch_intent.get("acceptance_criteria") is not None:
                criteria = patch_intent.get("acceptance_criteria")
                if isinstance(criteria, list):
                    data["acceptance_contract"] = {"criteria": criteria}
                else:
                    data["acceptance_contract"] = {"criteria": [str(criteria)]}
            elif patch_intent.get("acceptance") is not None:
                acc = patch_intent.get("acceptance")
                if isinstance(acc, list):
                    data["acceptance_contract"] = {"criteria": acc}
                else:
                    data["acceptance_contract"] = {"criteria": [str(acc)]}
            
    return data


def _latest_patch_intent(prior_summaries: dict[str, str]) -> dict[str, Any] | None:
    for summary in reversed(list(prior_summaries.values())):
        payload = extract_handoff_contract(summary)
        if payload is not None:
            return payload
    return None


def _patch_intent_resolves_readiness(
    analysis: dict[str, Any],
    patch_intent: dict[str, Any] | None,
) -> bool:
    if not isinstance(patch_intent, dict):
        return False
    if not patch_intent.get("edit_ready", False):
        return False
    strategy = patch_intent.get("edit_strategy") or str(analysis.get("edit_strategy") or analysis.get("intent") or "")
    if not strategy:
        return False
    targets = patch_intent.get("edit_targets")
    if not isinstance(targets, list) or not targets:
        return False
    dependencies_required = strategy in {"function_refactor", "sql_view_rewrite"}
    dependencies = patch_intent.get("dependencies_to_use") or patch_intent.get("dependencies")
    if dependencies_required and not (
        isinstance(dependencies, list) and dependencies
    ):
        return False
    if strategy == "sql_view_rewrite" and not str(patch_intent.get("target_view") or "").strip():
        return False
    return True


def _diagnose_summary_from_digest(
    subtask: SubTaskNode,
    digest: str,
    error_trace: list[str],
) -> str:
    body = digest.strip()
    if not body:
        recent = "\n".join(f"- {e}" for e in error_trace[-5:])
        body = recent or "The diagnose step reached summary mode before more tool output."
    findings = _diagnose_findings_from_digest(subtask, body)
    if findings:
        lines = [
            f"Result: 已定位 {len(findings)} 个相关代码位置。",
            "Evidence:",
        ]
        lines.extend(f"- {item}" for item in findings)
        lines.append(
            f"Conclusion: acceptance met。以上路径和行号可作为 {subtask.id} 的交接证据。"
        )
        return "\n".join(lines)
    searched = _diagnose_searches_from_digest(body)
    lines = [
        "Result: 未定位到可交接的具体路径和行号。",
        "Evidence:",
    ]
    if searched:
        lines.append("已执行的搜索:")
        lines.extend(f"- {item}" for item in searched[:6])
    else:
        lines.append("- 未记录到有效的 context_search/map_search/grep 命中。")
    if error_trace:
        lines.append("近期错误:")
        lines.extend(f"- {item}" for item in error_trace[-3:])
    lines.append(
        "Conclusion: 当前证据不足，不能作为后续 edit 的 handoff；需要扩大或改写搜索策略。"
    )
    return "\n".join(lines)


def _diagnose_findings_from_digest(subtask: SubTaskNode, digest: str) -> list[str]:
    intent = f"{subtask.description} {subtask.acceptance_criteria}".lower()
    wants_view_definition = (
        ("视图" in intent or "view" in intent)
        and ("定义" in intent or "definition" in intent or "create" in intent)
    )
    snippets = _code_snippets_from_digest(digest)
    candidates: list[tuple[int, str]] = []
    for raw in digest.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            line = line[2:].strip()
        if not _looks_like_location_hit(line):
            continue
        score = _finding_score(line, wants_view_definition=wants_view_definition)
        if score <= 0:
            continue
        candidates.append((
            score,
            _format_finding_line(
                line,
                wants_view_definition,
                fallback_snippet=snippets[0] if snippets else "",
            ),
        ))
    candidates.sort(key=lambda item: item[0], reverse=True)
    out: list[str] = []
    seen_locations: set[str] = set()
    for _score, text in candidates:
        location = text.split(" | ", 1)[0]
        if location in seen_locations:
            continue
        seen_locations.add(location)
        out.append(text)
        if len(out) >= 8:
            break
    return out


def _looks_like_location_hit(line: str) -> bool:
    return bool(re.match(r"[\w./-]+\.(?:py|sql|md|tsx?|jsx?):\d+", line))


def _diagnose_searches_from_digest(digest: str) -> list[str]:
    searches: list[str] = []
    in_search_section = False
    for raw in digest.splitlines():
        line = raw.strip()
        if line in {
            "Context/map searches already run:",
            "Grep queries already run:",
        }:
            in_search_section = True
            continue
        if in_search_section and line.startswith("- "):
            value = line[2:].strip()
            if value and value not in searches:
                searches.append(value)
            continue
        if in_search_section and line and not line.startswith("- "):
            in_search_section = False
    return searches


def _finding_score(line: str, *, wants_view_definition: bool) -> int:
    lower = line.lower()
    score = 1
    if ".sql:" in lower:
        score += 3
    if "create view" in lower or "create or replace view" in lower:
        score += 8
    if "视图" in line:
        score += 3
    if "-- 视图" in line or "视图：" in line or "视图:" in line:
        score += 5
    if wants_view_definition and not (
        ".sql:" in lower
        or "create view" in lower
        or "-- 视图" in line
        or "视图：" in line
        or "视图:" in line
    ):
        return 0
    return score


def _format_finding_line(
    line: str,
    wants_view_definition: bool,
    *,
    fallback_snippet: str = "",
) -> str:
    location, _, snippet = line.partition(":")
    line_no, _, rest = snippet.partition(":")
    loc = f"{location}:{line_no}" if line_no else location
    symbol = "视图定义" if wants_view_definition else "目标代码"
    evidence = rest.strip() or fallback_snippet or line
    return f"{loc} | {symbol} | {evidence[:220]}"


def _code_snippets_from_digest(digest: str) -> list[str]:
    snippets: list[str] = []
    in_section = False
    for raw in digest.splitlines():
        line = raw.strip()
        if line == "Code / SQL seen (snippets):":
            in_section = True
            continue
        if in_section and line and not line.startswith("- "):
            break
        if in_section and line.startswith("- "):
            snippet = line[2:].strip()
            if snippet:
                snippets.append(snippet)
    return snippets
