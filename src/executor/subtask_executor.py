from __future__ import annotations

import asyncio
import difflib
import logging
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.events import (
    AgentEvent,
    EventType,
    cost_event,
    error_event,
    final_answer_event,
    thinking_event,
    tool_call_event,
    tool_result_event,
)
from src.agent.shell_guard import ShellCommandTracker
from src.agent.types import (
    AgentState,
    LLMResponse,
    ToolCall,
    assistant_message,
    harness_nudge,
    tool_message,
)
from src.cli.stream_preview import (
    executor_reasoning_preview,
    executor_stream_hint,
    should_stream_executor_answer,
)
from src.context.retriever import build_context_queries
from src.executor.context_compress import merge_exploration_digests
from src.harness.gates.exit_gate import ExitCheckInput, validate_exit
from src.harness.gates.types import GateVerdict, TruncationPolicy
from src.harness.quality_gate import evaluate_quality_gate
from src.harness.subtask.handoff import (
    prepare_executor_handoff,
    resolve_turn_tools,
    turn_control_nudges,
)
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.harness.subtask.tool_pipeline import (
    ExecutorToolPipeline,
    build_tool_pipeline_context,
)
from src.llm.dsml import strip_dsml_text
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree

if TYPE_CHECKING:
    from src.agent.loop import LLMClient
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

EDIT_TOOLS = frozenset({"edit_file", "write_file", "delete_file", "replace_symbol"})
DIAGNOSE_SEED_TOOLS = frozenset({"grep_search", "map_search"})
_HANDOFF_FILE_LINE = re.compile(r"\bfile\s*:\s*line\b|\bline\b|行号|路径")
_HANDOFF_SYMBOL = re.compile(r"\bsymbol\b|\bfunction\b|\bmethod\b|符号|函数|类|方法|接口|端点")
_HANDOFF_SNIPPET = re.compile(r"\bsnippet\b|\bdecision\b|片段|代码|证据|结论|决策")


@dataclass
class SubTaskResult:
    success: bool
    subtask_id: str
    turns_used: int = 0
    final_message: str | None = None
    error_trace: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    file_diffs: dict[str, str] = field(default_factory=dict)


class SubTaskExecutor:
    """ReAct loop scoped to a single SubTaskNode (default max 3 turns)."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        harness: HarnessEngine,
        permissions: PermissionManager,
        settings: MitKIISettings,
        *,
        max_turns: int = 3,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.harness = harness
        self.permissions = permissions
        self.settings = settings
        self.max_turns = max_turns
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._tool_pipeline = ExecutorToolPipeline(
            tools,
            permissions,
            approval_waiter=self._wait_tool_approval,
            prepare_approval=self._prepare_tool_approval,
            normalize_path=SubTaskExecutor._normalize_path,
            snapshot_before_edit=_snapshot_before_edit,
            collect_diff=_collect_diffs,
        )

    def _prepare_tool_approval(self, action: str) -> None:
        self._approval_futures[action] = asyncio.get_running_loop().create_future()

    async def _wait_tool_approval(self, action: str) -> bool:
        decision = self.permissions.session_decision(action)
        if decision is not None:
            self._approval_futures.pop(action, None)
            return decision

        fut = self._approval_futures.get(action)
        if fut is None:
            self._prepare_tool_approval(action)
            fut = self._approval_futures[action]
        if fut.done():
            try:
                return bool(fut.result())
            finally:
                self._approval_futures.pop(action, None)
        try:
            return await asyncio.wait_for(fut, timeout=300.0)
        except TimeoutError:
            return False
        finally:
            self._approval_futures.pop(action, None)

    async def resolve_approval(self, action: str, approved: bool) -> None:
        self.permissions.record_decision(action, approved)
        fut = self._approval_futures.get(action)
        if fut is None:
            return
        if fut.done():
            return
        fut.set_result(approved)

    async def run(
        self,
        *,
        root_task: str,
        task_tree: TaskTree,
        subtask: SubTaskNode,
        truncation_policy: TruncationPolicy | None = None,
        retry_feedback: list[str] | None = None,
        quality_gate_retry_limit: int | None = None,
        prior_summaries: dict[str, str] | None = None,
        subtask_attempt: int = 1,
        prior_exploration: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._run_inner(
            root_task,
            task_tree,
            subtask,
            truncation_policy=truncation_policy,
            retry_feedback=retry_feedback,
            quality_gate_retry_limit=quality_gate_retry_limit,
            prior_summaries=prior_summaries,
            subtask_attempt=subtask_attempt,
            prior_exploration=prior_exploration,
        ):
            yield event

    async def run_collect(
        self,
        *,
        root_task: str,
        task_tree: TaskTree,
        subtask: SubTaskNode,
    ) -> SubTaskResult:
        result = SubTaskResult(success=False, subtask_id=subtask.id)
        async for event in self._run_inner(root_task, task_tree, subtask):
            if event.type == EventType.ERROR and event.content:
                result.error_trace.append(event.content)
            if (
                event.type == EventType.TOOL_RESULT
                and event.data
                and not event.data.get("success", True)
            ):
                result.error_trace.append(str(event.content))
            if event.type == EventType.FINAL_ANSWER:
                result.success = True
                result.final_message = event.content
        return result

    async def _run_inner(
        self,
        root_task: str,
        task_tree: TaskTree,
        subtask: SubTaskNode,
        *,
        truncation_policy: TruncationPolicy | None = None,
        retry_feedback: list[str] | None = None,
        quality_gate_retry_limit: int | None = None,
        prior_summaries: dict[str, str] | None = None,
        subtask_attempt: int = 1,
        prior_exploration: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        project_root = self.harness.project_root
        bundle = prepare_executor_handoff(
            root_task=root_task,
            task_tree=task_tree,
            subtask=subtask,
            project_root=project_root,
            settings=self.settings,
            truncation_policy=truncation_policy,
            prior_summaries=prior_summaries,
            retry_feedback=retry_feedback,
            prior_exploration=prior_exploration,
            subtask_attempt=subtask_attempt,
            quality_gate_retry_limit=quality_gate_retry_limit,
            map_search_in_registry="map_search" in self.tools,
        )
        session_memory = bundle.session_memory
        ctx_config = bundle.ctx_config
        ctx_runtime = bundle.ctx_runtime
        assert ctx_config is not None and ctx_runtime is not None
        ctx_session = self.harness.create_executor_context_session(
            ctx_config, session_memory, ctx_runtime
        )
        policy = bundle.policy
        qg_limit = bundle.qg_limit
        diag_handoff = bundle.diag_handoff
        paths_only_mode = ctx_runtime.paths_only_mode
        preloaded_paths = ctx_runtime.preloaded_paths
        truncated_paths = ctx_runtime.truncated_paths
        active_runtime_tools = ctx_runtime.active_runtime_tools
        turn_cap = bundle.turn_cap
        is_diagnose = subtask.kind == SubTaskKind.DIAGNOSE

        yield AgentEvent(type=EventType.STREAM_START)
        for status in bundle.startup_status:
            yield AgentEvent(
                type=EventType.STATUS,
                content=status,
                data={
                    "phase": "executor",
                    "subtask_id": subtask.id,
                    "spinner_only": True,
                },
            )
            await asyncio.sleep(0)

        state = AgentState()
        state.messages = list(bundle.initial_messages)
        ctx_session.seed_prefix(state.messages)

        pre_edit_snapshots: dict[str, str] = {}
        error_trace: list[str] = []
        turns_used = 0
        tool_rounds = 0
        final_message: str | None = None
        tool_failures = [0]
        quality_gate_failures = 0
        shell_tracker = ShellCommandTracker(
            dedup_limit=self.settings.shell_dedup_limit,
            stagnant_limit=self.settings.shell_stagnant_limit,
        )
        diagnose_summary_hint_sent = False

        if _should_seed_diagnose_search(
            subtask=subtask,
            active_runtime_tools=active_runtime_tools,
            prior_summaries=prior_summaries,
            subtask_attempt=subtask_attempt,
        ):
            async for seed_event in self._run_diagnose_seed_search(
                root_task=root_task,
                subtask=subtask,
                state=state,
                session_memory=session_memory,
            ):
                yield seed_event
            session_memory.merge_digest_from_messages(state.messages)
            tool_rounds = bundle.diagnose_tool_rounds

        for turns_used in range(1, turn_cap + 1):
            turn_changes_start = len(state.file_changes)
            state.advance_turn()

            for nudge in turn_control_nudges(
                bundle,
                turns_used=turns_used,
                tool_rounds=tool_rounds,
                file_changes=state.file_changes,
                diagnose_summary_hint_sent=diagnose_summary_hint_sent,
                error_trace=error_trace,
            ):
                state.add_message(nudge)
                if "Exploration budget used" in (nudge.content or ""):
                    diagnose_summary_hint_sent = True

            turn_tools = resolve_turn_tools(
                bundle,
                turns_used=turns_used,
                tool_rounds=tool_rounds,
                file_changes=state.file_changes,
                active_runtime_tools=active_runtime_tools,
            )

            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Executor [{subtask.id}] · preparing context…",
                data={
                    "spinner_only": True,
                    "phase": "executor",
                    "subtask_id": subtask.id,
                },
            )
            await asyncio.sleep(0)
            yield AgentEvent(
                type=EventType.STREAM_START,
                data={"phase": "executor", "subtask_id": subtask.id},
            )

            trimmed, pre_cp = await self.harness.before_executor_llm_call(
                ctx_session,
                state.messages,
                error_trace,
            )
            if pre_cp.changed:
                state.messages = pre_cp.messages
                ctx_runtime = pre_cp.runtime
                bundle.ctx_runtime = pre_cp.runtime
                paths_only_mode = pre_cp.runtime.paths_only_mode
                preloaded_paths = pre_cp.runtime.preloaded_paths
                truncated_paths = pre_cp.runtime.truncated_paths
                active_runtime_tools = pre_cp.runtime.active_runtime_tools
                turn_tools = resolve_turn_tools(
                    bundle,
                    turns_used=turns_used,
                    tool_rounds=tool_rounds,
                    file_changes=state.file_changes,
                    active_runtime_tools=active_runtime_tools,
                )
                for pipe_ev in pre_cp.events:
                    yield AgentEvent(
                        type=EventType.STATUS,
                        content=(
                            f"Executor [{subtask.id}] · {pipe_ev.content}"
                        ),
                        data={
                            "phase": "executor",
                            "subtask_id": subtask.id,
                            "executor_activity": True,
                        },
                    )
            yield AgentEvent(
                type=EventType.STATUS,
                content=(
                    f"Executor [{subtask.id}] · turn {turns_used}/{turn_cap} · calling model…"
                ),
                data={
                    "spinner_only": True,
                    "phase": "executor",
                    "subtask_id": subtask.id,
                },
            )
            await asyncio.sleep(0)
            tool_schemas = self.tools.get_schemas(include=turn_tools)
            prompt_tokens_est = pre_cp.token_est
            tool_names = ", ".join(sorted(turn_tools)) or "none"
            context_mode = (
                "paths_only"
                if paths_only_mode
                else "diag_handoff"
                if diag_handoff
                else "full"
            )
            call_detail = (
                f"Executor [{subtask.id}] · model input: "
                f"messages={len(trimmed)}, "
                f"prompt≈{prompt_tokens_est if prompt_tokens_est is not None else 'unknown'} tok, "
                f"tools={len(turn_tools)} [{tool_names}], "
                f"mode={context_mode}, "
                f"preloaded={len(preloaded_paths)}, truncated={len(truncated_paths)}"
            )
            log.info(call_detail)
            yield AgentEvent(
                type=EventType.STATUS,
                content=call_detail,
                data={
                    "phase": "executor",
                    "subtask_id": subtask.id,
                    "executor_activity": True,
                },
            )

            response: LLMResponse | None = None
            response_text = ""

            stream_parts: list[str] = []
            first_chunk_elapsed: float | None = None
            model_start = time.perf_counter()
            async for chunk in self._stream_llm(trimmed, tool_schemas):
                if chunk.get("type") == "content":
                    delta = chunk.get("content", "")
                    if delta:
                        if first_chunk_elapsed is None:
                            first_chunk_elapsed = time.perf_counter() - model_start
                        stream_parts.append(delta)
                elif chunk.get("type") == "response":
                    if first_chunk_elapsed is None:
                        first_chunk_elapsed = time.perf_counter() - model_start
                    response = chunk["response"]
            model_elapsed = time.perf_counter() - model_start

            response_text = "".join(stream_parts)
            yield AgentEvent(
                type=EventType.STREAM_END,
                data={
                    "phase": "executor",
                    "subtask_id": subtask.id,
                    "heartbeat": True,
                },
            )
            if model_elapsed >= 5.0:
                response_tool_names: list[str] = []
                response_tool_arg_chars = 0
                if response is not None and response.tool_calls:
                    response_tool_names = [tc.name for tc in response.tool_calls]
                    response_tool_arg_chars = sum(
                        len(str(tc.arguments)) for tc in response.tool_calls
                    )
                usage_bits = ""
                if response is not None and response.usage:
                    usage_bits = (
                        f", usage={response.usage.prompt_tokens}/"
                        f"{response.usage.completion_tokens} tok"
                    )
                slow_detail = (
                    f"Executor [{subtask.id}] · model call took {model_elapsed:.1f}s "
                    f"(first chunk {first_chunk_elapsed or 0:.1f}s, "
                    f"{len(turn_tools)} runtime tool(s), output={len(response_text)} chars, "
                    f"tool_calls={response_tool_names or 'none'}, "
                    f"tool_arg_chars={response_tool_arg_chars}{usage_bits})"
                )
                log.info(slow_detail)
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=slow_detail,
                    data={
                        "phase": "executor",
                        "subtask_id": subtask.id,
                        "executor_activity": True,
                    },
                )

            if response is not None and response.tool_calls:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        f"Executor [{subtask.id}] · turn {turns_used}/{turn_cap} · tools…"
                    ),
                    data={
                        "spinner_only": True,
                        "phase": "executor",
                        "subtask_id": subtask.id,
                    },
                )
            elif response is not None and response_text.strip() and not is_diagnose:
                if should_stream_executor_answer(response_text, has_tool_calls=False):
                    preview = executor_reasoning_preview(response_text)
                    if preview:
                        yield thinking_event(
                            preview,
                            phase="executor",
                            subtask_id=subtask.id,
                            preview_line=True,
                        )
                else:
                    hint = executor_stream_hint(response_text)
                    if hint:
                        yield thinking_event(
                            hint,
                            phase="executor",
                            subtask_id=subtask.id,
                            preview_line=True,
                        )

            if response is None:
                error_trace.append("LLM returned no response")
                yield error_event("LLM returned no response")
                break

            from src.harness.probe.llm_usage import (
                estimate_cost_for_model,
                estimate_usage_from_text,
                record_litellm_completion,
            )

            trimmed_msgs = [m.to_dict() for m in state.messages]
            if response.usage:
                record_litellm_completion(
                    self.harness.probe.metrics,
                    response,
                    model=response.model or self.llm.model,
                    messages=trimmed_msgs,
                    completion_text=response.content or response_text,
                )
                cost = (
                    self.harness.probe.metrics.last_record.cost
                    if self.harness.probe.metrics.last_record
                    else 0.0
                )
                state.record_usage(response.usage, cost)
                yield cost_event(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )
            else:
                est = estimate_usage_from_text(
                    trimmed_msgs,
                    response.content or response_text,
                    model=self.llm.model,
                )
                cost = estimate_cost_for_model(
                    est.prompt_tokens,
                    est.completion_tokens,
                    self.llm.model,
                )
                self.harness.probe.metrics.record(
                    self.llm.model,
                    est.prompt_tokens,
                    est.completion_tokens,
                    cost=cost,
                )
                state.record_usage(est, cost)
                yield cost_event(est.prompt_tokens, est.completion_tokens, cost)

            await self.harness.after_llm_call(response, response.usage)

            if is_diagnose and not turn_tools and response.tool_calls:
                digest = merge_exploration_digests(
                    session_memory.running_digest,
                    state.messages,
                )
                final_message = _diagnose_summary_from_digest(
                    subtask,
                    digest,
                    error_trace,
                    blocked_extra_tools=True,
                )
                response.content = final_message
                response.tool_calls = None
                response_text = final_message
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        f"Executor [{subtask.id}] · tools disabled; "
                        "converted extra tool request into diagnose summary"
                    ),
                    data={
                        "phase": "executor",
                        "subtask_id": subtask.id,
                        "executor_activity": True,
                    },
                )

            if response.tool_calls:
                tool_rounds += 1
                state.add_message(
                    assistant_message(response.content or "", response.tool_calls)
                )
                pipeline_ctx = build_tool_pipeline_context(
                    bundle=bundle,
                    subtask=subtask,
                    root_task=root_task,
                    project_root=project_root,
                    turn_tools=turn_tools,
                    runtime=ctx_runtime,
                    policy=policy,
                    memory=session_memory,
                    messages=state.messages,
                    error_trace=error_trace,
                    tool_failures=tool_failures,
                    shell_tracker=shell_tracker,
                    pre_edit_snapshots=pre_edit_snapshots,
                    file_changes=state.file_changes,
                    max_tool_output_chars=self.settings.preflight_max_chars_per_file * 2,
                )

                async for event in self._tool_pipeline.process_tool_round(
                    response,
                    pipeline_ctx,
                ):
                    yield event

                if pipeline_ctx.runtime is not None:
                    ctx_runtime = pipeline_ctx.runtime
                    bundle.ctx_runtime = ctx_runtime
                    active_runtime_tools = ctx_runtime.active_runtime_tools

                round_stats = self._tool_pipeline.last_round_stats(pipeline_ctx)
                cp = await self.harness.after_executor_tool_round(
                    ctx_session,
                    state.messages,
                    error_trace,
                    explore_used=round_stats.explore_used,
                    explore_ok=round_stats.explore_ok,
                )
                if cp.changed:
                    state.messages = cp.messages
                    ctx_runtime = cp.runtime
                    bundle.ctx_runtime = cp.runtime
                    paths_only_mode = cp.runtime.paths_only_mode
                    preloaded_paths = cp.runtime.preloaded_paths
                    truncated_paths = cp.runtime.truncated_paths
                    active_runtime_tools = cp.runtime.active_runtime_tools
                    turn_tools = active_runtime_tools
                    for pipe_ev in cp.events:
                        yield AgentEvent(
                            type=EventType.STATUS,
                            content=(
                                f"Executor [{subtask.id}] · {pipe_ev.content}"
                            ),
                            data={
                                "phase": "executor",
                                "subtask_id": subtask.id,
                                "executor_activity": True,
                            },
                        )

                if len(state.file_changes) > turn_changes_start:
                    changed = state.file_changes[turn_changes_start:]
                    pm = self.harness.phase_metrics
                    pm.start("quality_gate", subtask_id=subtask.id)
                    skip_l1 = not subtask.effective_needs_l1()
                    gate = await evaluate_quality_gate(
                        self.harness,
                        user_msg=f"{root_task}\n\nSubtask: {subtask.description}",
                        changed_files=changed,
                        skip_l1=skip_l1,
                        acceptance_criteria=subtask.acceptance_criteria,
                    )
                    if gate:
                        pm.end(
                            "quality_gate",
                            subtask_id=subtask.id,
                            verdict=str(gate.get("gate")),
                            metadata={
                                "l0_passed": gate.get("l0_passed"),
                                "l1_passed": gate.get("l1_passed"),
                                "l1_skipped": gate.get("l1_skipped"),
                                "kind": subtask.kind.value,
                            },
                        )
                        yield AgentEvent(type=EventType.SCORE_RESULT, data=gate)
                        if gate.get("gate") == "FAIL":
                            quality_gate_failures += 1
                            feedback = gate.get("feedback") or "L0/L1 quality gate failed"
                            error_trace.append(str(feedback))
                            state.add_message(
                                harness_nudge(
                                    "Quality gate FAIL after your edit. "
                                    "Fix with edit_file first; use write_file only for "
                                    "new files or complete-file rewrites.\n"
                                    f"{feedback}"
                                )
                            )
                            if quality_gate_failures >= qg_limit:
                                yield error_event(
                                    f"Quality gate failed {quality_gate_failures} time(s) "
                                    f"on subtask [{subtask.id}]",
                                    {"subtask_id": subtask.id},
                                )
                                yield AgentEvent(
                                    type=EventType.STREAM_END,
                                    data=_failure_payload(
                                        subtask_id=subtask.id,
                                        turns_used=turns_used,
                                        error_trace=error_trace,
                                        final_message=final_message,
                                        pre_edit_snapshots=pre_edit_snapshots,
                                        project_root=project_root,
                                        state=state,
                                        failure_code="quality_gate_exhausted",
                                        quality_gate_failures=quality_gate_failures,
                                        session_memory=session_memory,
                                    ),
                                )
                                return
                continue

            final_message = response.content or response_text

            if (
                subtask.kind == SubTaskKind.EDIT
                and not state.file_changes
                and turns_used < turn_cap
            ):
                nudge = (
                    "No file edits yet. Use evidence from the session summary, "
                    "then edit_file on a context_files path. Never /tmp."
                )
                state.add_message(harness_nudge(
                    "No file edits yet. Use evidence from the session summary, "
                    "then edit_file on a context_files path. Never /tmp."
                ))
                continue

            state.add_message(assistant_message(final_message))

            pm = self.harness.phase_metrics
            pm.start("exit_gate", subtask_id=subtask.id)
            exit_result = validate_exit(
                ExitCheckInput(
                    subtask=subtask,
                    final_message=final_message,
                    error_trace=error_trace,
                    changed_files=list(state.file_changes),
                    turns_used=turns_used,
                    tool_failure_count=tool_failures[0],
                )
            )
            pm.end(
                "exit_gate",
                subtask_id=subtask.id,
                verdict=exit_result.verdict.value,
                metadata=exit_result.metadata,
            )

            if exit_result.verdict == GateVerdict.WARN and exit_result.messages:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content="Exit gate: " + "; ".join(exit_result.messages),
                    data={"subtask_id": subtask.id},
                )

            if not exit_result.passed:
                err = "; ".join(exit_result.messages)
                error_trace.extend(exit_result.messages)
                yield error_event(err, {"subtask_id": subtask.id, "gate": "exit_gate"})
                yield AgentEvent(
                    type=EventType.STREAM_END,
                    data=_failure_payload(
                        subtask_id=subtask.id,
                        turns_used=turns_used,
                        error_trace=error_trace,
                        final_message=final_message,
                        pre_edit_snapshots=pre_edit_snapshots,
                        project_root=project_root,
                        state=state,
                        failure_code="exit_gate",
                        quality_gate_failures=quality_gate_failures,
                        session_memory=session_memory,
                    ),
                )
                return

            yield final_answer_event(
                final_message,
                intermediate=True,
                subtask_id=subtask.id,
            )
            success_payload: dict[str, Any] = {
                "subtask_id": subtask.id,
                "turns_used": turns_used,
                "success": True,
                "changed_files": list(state.file_changes),
                "file_diffs": _collect_diffs(
                    pre_edit_snapshots, project_root, state.file_changes
                ),
                "error_trace": error_trace,
                "final_message": final_message,
                "exit_gate": exit_result.verdict.value,
                "failure_code": "",
                "quality_gate_failures": quality_gate_failures,
            }
            digest = merge_exploration_digests(session_memory.running_digest, state.messages)
            if digest:
                success_payload["exploration_digest"] = digest
            if is_diagnose and turns_used > bundle.max_turns:
                success_payload["summarized_after_limit"] = True
            yield AgentEvent(type=EventType.STREAM_END, data=success_payload)
            return

        msg = f"Subtask [{subtask.id}] hit executor turn limit ({turn_cap})"
        error_trace.append(msg)
        yield error_event(msg, {"turns_used": turns_used, "subtask_id": subtask.id})
        yield AgentEvent(
            type=EventType.STREAM_END,
            data=_failure_payload(
                subtask_id=subtask.id,
                turns_used=turns_used,
                error_trace=error_trace,
                final_message=final_message,
                pre_edit_snapshots=pre_edit_snapshots,
                project_root=project_root,
                state=state,
                failure_code="turn_limit",
                quality_gate_failures=quality_gate_failures,
                session_memory=session_memory,
            ),
        )

    async def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async for content_chunk, final_response in self.llm.chat_stream(messages, tools=tools):
                clean_chunk = strip_dsml_text(content_chunk)
                if clean_chunk:
                    yield {"type": "content", "content": clean_chunk}
                if final_response is not None:
                    yield {"type": "response", "response": final_response}
        except Exception as exc:
            from src.agent.error_recovery import ErrorRecovery

            recovery = ErrorRecovery()
            yield {
                "type": "response",
                "response": LLMResponse(
                    content=recovery.handle_tool_error(exc, "llm_call"),
                    tool_calls=None,
                    usage=None,
                    model="error",
                ),
            }

    async def _run_diagnose_seed_search(
        self,
        *,
        root_task: str,
        subtask: SubTaskNode,
        state: AgentState,
        session_memory: ExploreSessionMemory,
    ) -> AsyncIterator[AgentEvent]:
        available = frozenset(
            name for name in DIAGNOSE_SEED_TOOLS if self.tools.get(name) is not None
        )
        calls = _diagnose_seed_tool_calls(
            root_task=root_task,
            subtask=subtask,
            available_tools=available,
        )
        if not calls:
            return

        detail = (
            f"Executor [{subtask.id}] · harness pre-search: "
            f"{len(calls)} batched map/grep call(s); next model turn summarizes"
        )
        log.info(detail)
        yield AgentEvent(
            type=EventType.STATUS,
            content=detail,
            data={
                "phase": "executor",
                "subtask_id": subtask.id,
                "executor_activity": True,
            },
        )
        state.add_message(assistant_message("", calls))

        for tc in calls:
            yield tool_call_event(tc.name, tc.arguments, phase="executor")
            started = time.perf_counter()
            result = await self.tools.call(tc.name, tc.arguments)
            elapsed = time.perf_counter() - started
            body = result.output if result.success else (result.error or result.output)
            body = session_memory.truncate_output(body, max_chars=12_000)
            key = session_memory.explore_key(tc.name, tc.arguments)
            if key:
                session_memory.put_output(key, body)
            session_memory.record_explore(tc.name, tc.arguments)
            state.add_message(tool_message(tc.id, body))
            yield tool_result_event(
                tc.name,
                body,
                success=result.success,
                phase="executor",
            )
            status = (
                f"{tc.name} seed call completed in {elapsed:.1f}s"
                if result.success
                else f"{tc.name} seed call failed in {elapsed:.1f}s"
            )
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Executor [{subtask.id}] · {status}",
                data={
                    "phase": "executor",
                    "subtask_id": subtask.id,
                    "executor_activity": True,
                },
            )

    @staticmethod
    def _normalize_path(project_root: Path, path: object) -> str | None:
        if not isinstance(path, str) or not path.strip():
            return None
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            else:
                p = p.resolve()
            return str(p.relative_to(project_root.resolve()))
        except Exception:
            return path.strip().replace("\\", "/")


def _failure_payload(
    *,
    subtask_id: str,
    turns_used: int,
    error_trace: list[str],
    final_message: str | None,
    pre_edit_snapshots: dict[str, str],
    project_root: Path,
    state: AgentState,
    failure_code: str,
    quality_gate_failures: int,
    session_memory: ExploreSessionMemory | None = None,
    prior_exploration: str | None = None,
) -> dict[str, Any]:
    base = (
        session_memory.running_digest
        if session_memory is not None
        else (prior_exploration or "")
    )
    digest = merge_exploration_digests(base, state.messages)
    return {
        "subtask_id": subtask_id,
        "turns_used": turns_used,
        "success": False,
        "changed_files": list(state.file_changes),
        "file_diffs": _collect_diffs(
            pre_edit_snapshots, project_root, state.file_changes
        ),
        "error_trace": error_trace,
        "final_message": final_message,
        "failure_code": failure_code,
        "quality_gate_failures": quality_gate_failures,
        "exploration_digest": digest or None,
    }


def _should_seed_diagnose_search(
    *,
    subtask: SubTaskNode,
    active_runtime_tools: frozenset[str],
    prior_summaries: dict[str, str] | None,
    subtask_attempt: int,
) -> bool:
    if subtask.kind != SubTaskKind.DIAGNOSE:
        return False
    if subtask_attempt != 1 or prior_summaries:
        return False
    if subtask.context_files:
        return False
    if not (active_runtime_tools & DIAGNOSE_SEED_TOOLS):
        return False
    criteria = subtask.acceptance_criteria or ""
    return bool(
        _HANDOFF_FILE_LINE.search(criteria)
        and _HANDOFF_SYMBOL.search(criteria)
        and _HANDOFF_SNIPPET.search(criteria)
    )


def _diagnose_seed_tool_calls(
    *,
    root_task: str,
    subtask: SubTaskNode,
    available_tools: frozenset[str],
) -> list[ToolCall]:
    text = " ".join(
        part
        for part in (root_task, subtask.description, subtask.acceptance_criteria)
        if part
    )
    map_queries, grep_pattern = _diagnose_seed_queries(text)

    calls: list[ToolCall] = []
    call_index = 1
    if "map_search" in available_tools:
        for query in map_queries:
            calls.append(
                ToolCall(
                    id=f"harness-diagnose-seed-{call_index}",
                    name="map_search",
                    arguments={"query": query, "limit": 20},
                )
            )
            call_index += 1
    if "grep_search" in available_tools and grep_pattern:
        for include in ("*.py", "*.sql"):
            calls.append(
                ToolCall(
                    id=f"harness-diagnose-seed-{call_index}",
                    name="grep_search",
                    arguments={
                        "pattern": grep_pattern,
                        "path": ".",
                        "include": include,
                        "max_results": 80,
                    },
                )
            )
            call_index += 1
    return calls


def _diagnose_seed_queries(text: str) -> tuple[list[str], str]:
    map_terms: list[str] = build_context_queries(text, limit=3)
    grep_terms: list[str] = [re.escape(term) for term in build_context_queries(text, limit=12)]

    def add_map(term: str) -> None:
        if term and term not in map_terms:
            map_terms.append(term)

    def add_grep(term: str, *, regex: bool = False) -> None:
        value = term if regex else re.escape(term)
        if value and value not in grep_terms:
            grep_terms.append(value)

    if "视图" in text or "view" in text.lower():
        add_grep(r"CREATE\s+VIEW", regex=True)
        add_grep(r"create\s+view", regex=True)
    if "接口" in text or "端点" in text or "api" in text.lower():
        add_grep(r"@app\.(get|post|put|delete|patch)", regex=True)

    for endpoint in re.findall(r"/[A-Za-z0-9_./{}-]+", text):
        add_map(endpoint.strip("/"))
        add_grep(endpoint)

    if not grep_terms:
        return map_terms[:3], ""

    return map_terms[:3], "|".join(grep_terms[:18])


def _diagnose_summary_from_digest(
    subtask: SubTaskNode,
    digest: str,
    error_trace: list[str],
    *,
    blocked_extra_tools: bool = False,
) -> str:
    body = digest.strip()
    if not body:
        recent = "\n".join(f"- {e}" for e in error_trace[-5:])
        body = recent or "The diagnose step reached summary mode before more tool output."
    if blocked_extra_tools:
        return (
            "Result: 诊断证据不足，acceptance criteria not yet met.\n"
            "Evidence:\n"
            f"{body[:6000]}\n"
            "Conclusion: 工具预算已用完，模型仍请求更多工具；当前证据不足以安全交付"
            f" {subtask.id}，需要重规划或扩大/调整搜索模块。"
        )
    return (
        "Result: 已根据已有工具证据生成诊断摘要。\n"
        "Evidence:\n"
        f"{body[:6000]}\n"
        "Conclusion: 使用上面的路径、行号和 grep/map 命中作为 "
        f"{subtask.id} 的交接证据。"
    )


def _snapshot_before_edit(
    project_root: Path,
    tc: ToolCall,
    snapshots: dict[str, str],
) -> None:
    path = tc.arguments.get("path")
    if not isinstance(path, str):
        return
    try:
        p = (
            (project_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
        if p.is_file() and str(p) not in snapshots:
            snapshots[str(p)] = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass


def _collect_diffs(
    snapshots: dict[str, str],
    project_root: Path,
    changed_files: list[str],
) -> dict[str, str]:
    diffs: dict[str, str] = {}
    for raw in changed_files:
        try:
            p = (
                (project_root / raw).resolve()
                if not Path(raw).is_absolute()
                else Path(raw).resolve()
            )
        except OSError:
            continue
        old = snapshots.get(str(p), "")
        try:
            new = p.read_text(encoding="utf-8", errors="replace") if p.is_file() else ""
        except OSError:
            new = ""
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                fromfile=f"{raw} (before)",
                tofile=f"{raw} (after)",
                lineterm="",
            )
        )
        if diff:
            diffs[str(raw)] = diff
    return diffs
