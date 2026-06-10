from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import litellm

from src.llm.prompt_cache import CacheTTL, apply_prompt_cache, mark_cache_breakpoint
from src.orchestrator.evidence import EvidencePack
from src.planner.planner_parse import (
    PlannerParseResult,
    build_task_tree_from_payload,
    normalize_planner_payload,
    validate_planner_payload,
)
from src.planner.task_templates import select_task_template, task_tree_from_template, calculate_edit_complexity_score
from src.planner.task_tree import SubTaskKind, SubTaskNode, SubTaskStatus, TaskTree
from src.planner.tool_policy import default_allowed_tools

_GATE_REWRITE_NUDGE = (
    "PlanGate / schema rejected your TaskTree:\n{errors}\n\n"
    "Reply with ONE corrected raw JSON object only (double-quoted keys, no fences).\n"
    "Checklist: st-1 kind matches intent (edit/diagnose/verify/shell); "
    "st-2+ depends_on when ordered; every node has all 8 fields; "
    "allowed_tools match kind table; edit includes edit_file or write_file."
)

log = logging.getLogger(__name__)

_PROJECT_CONTEXT_LABEL = "Project context (repo_map + directory tree)"

_PLANNER_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent / "prompts" / "planner_prompt.md"
)
_PATCH_PLAN_SCHEMA = (
    '{"patch_plan":{"files_to_edit":["path"],"target_symbols":["symbol"],'
    '"intended_changes":["short change"],"edits":[{"path":"path",'
    '"symbol":"optional","old_string":"exact text from context",'
    '"new_string":"replacement text"}],"validation_plan":["command or static check"],'
    '"requires_confirmation":false,"confidence":0.0,"missing_info":[]}}'
)
_PATCH_PLAN_SYSTEM_PROMPT = (
    "You are MitKII PatchPlan Planner.\n"
    "Output ONE raw JSON object only. No markdown, no prose, no TaskTree, "
    "no nodes array.\n\n"
    "Schema:\n"
    f"{_PATCH_PLAN_SCHEMA}\n\n"
    "Rules:\n"
    "- Only use evidence from <context_pack>.\n"
    "- old_string must be copied exactly from snippets.\n"
    "- If exact edit text is not known, return edits=[], confidence<0.75, "
    "and missing_info.\n"
    "- Never output TaskTree fields such as root_task, nodes, kind, "
    "allowed_tools, or depends_on.\n"
)

_TRACE_STEPS = (
    "1. Task type\n2. Scout leverage\n3. Subtask sketch\n4. Tool audit\n5. Ordering"
)


def load_planner_system_prompt() -> str:
    if _PLANNER_PROMPT_PATH.is_file():
        return _PLANNER_PROMPT_PATH.read_text(encoding="utf-8").strip()
    log.warning("planner_prompt.md missing, using embedded fallback")
    return "You are MitKII Planner. Output ONLY a JSON TaskTree."


def planner_output_instruction(*, require_trace: bool) -> str:
    if require_trace:
        return (
            "First write <planning_trace> (5 numbered one-liners: "
            f"{_TRACE_STEPS}), then ONE raw TaskTree JSON. "
            "Each node: kind, allowed_tools, acceptance_criteria, context_files."
        )
    return (
        "Output ONE raw TaskTree JSON only (no planning_trace, no markdown, no prose). "
        "Copy the example structure from the system prompt — all 8 fields on every node."
    )


@runtime_checkable
class PlannerClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


class StreamingPlannerClient(Protocol):
    async def complete(self, messages: list[dict[str, Any]]) -> str: ...

    def stream_complete(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[str]: ...


class LiteLLMPlannerClient:
    """Tool-free LLM backend for the Planner node."""

    def __init__(
        self,
        model: str,
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: int = 180,
        metrics: Any | None = None,
        json_mode: bool = True,
        prompt_cache_enabled: bool = True,
        prompt_cache_min_tokens: int = 1024,
        prompt_cache_ttl: CacheTTL = "5m",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._metrics = metrics
        self.json_mode = json_mode
        self.prompt_cache_enabled = prompt_cache_enabled
        self.prompt_cache_min_tokens = prompt_cache_min_tokens
        self.prompt_cache_ttl = prompt_cache_ttl

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return apply_prompt_cache(
            messages,
            model=self.model,
            enabled=self.prompt_cache_enabled,
            min_tokens=self.prompt_cache_min_tokens,
            ttl=self.prompt_cache_ttl,
        )

    async def _call(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool,
        stream_options: dict[str, Any] | None = None,
    ) -> Any:
        prepared = self._prepare_messages(messages)
        base: dict[str, Any] = {
            "model": self.model,
            "messages": prepared,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "stream": stream,
        }
        if stream and stream_options:
            base["stream_options"] = stream_options
        if not self.json_mode:
            return await litellm.acompletion(**base)
        try:
            return await litellm.acompletion(
                **base,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            log.debug("Planner JSON mode unsupported, falling back: %s", exc)
            return await litellm.acompletion(**base)

    async def complete(self, messages: list[dict[str, Any]]) -> str:
        response = await self._call(messages, stream=False)
        if self._metrics is not None:
            from src.harness.probe.llm_usage import record_litellm_completion

            record_litellm_completion(
                self._metrics,
                response,
                model=self.model,
                messages=messages,
                completion_text=response.choices[0].message.content or "",
            )
        content = response.choices[0].message.content
        return content or ""

    async def stream_complete(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Stream completion deltas; records usage after the stream finishes."""
        response = await self._call(
            messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        last_chunk: Any = None
        async for chunk in response:
            last_chunk = chunk
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                parts.append(delta.content)
                yield delta.content
        if self._metrics is not None and last_chunk is not None:
            from src.harness.probe.llm_usage import record_litellm_completion

            completion_text = "".join(parts)
            record_litellm_completion(
                self._metrics,
                last_chunk,
                model=self.model,
                messages=messages,
                completion_text=completion_text,
            )


class PlannerNode:
    """High-level planner: produces and revises TaskTree without tool access."""

    def __init__(
        self,
        client: PlannerClient,
        *,
        require_trace: bool = False,
    ) -> None:
        self._client = client
        self._require_trace = require_trace
        self._system_prompt = load_planner_system_prompt()

    @property
    def client(self) -> PlannerClient:
        return self._client

    def plan_messages(
        self,
        user_request: str,
        project_context: str,
        *,
        discovery_manifest: str | None = None,
    ) -> list[dict[str, Any]]:
        messages = self._cached_prefix_messages(project_context)
        messages.append({
            "role": "user",
            "content": self._build_user_plan_prompt(
                user_request,
                discovery_manifest=discovery_manifest,
            ),
        })
        return messages

    def rewrite_messages(
        self,
        user_request: str,
        project_context: str,
        *,
        gate_errors: str,
        previous_raw: str,
        discovery_manifest: str | None = None,
    ) -> list[dict[str, Any]]:
        messages = self._cached_prefix_messages(project_context)
        messages.append({
            "role": "user",
            "content": self._build_user_plan_prompt(
                user_request,
                discovery_manifest=discovery_manifest,
            ),
        })
        messages.extend([
            {"role": "assistant", "content": previous_raw},
            {
                "role": "user",
                "content": _GATE_REWRITE_NUDGE.format(errors=gate_errors),
            },
        ])
        return messages

    def patch_plan_messages(
        self,
        user_request: str,
        project_context: str,
        *,
        context_pack: str,
    ) -> list[dict[str, Any]]:
        messages = [
            mark_cache_breakpoint({
                "role": "system",
                "content": _PATCH_PLAN_SYSTEM_PROMPT,
            })
        ]
        if project_context.strip():
            messages.append(
                mark_cache_breakpoint({
                    "role": "system",
                    "content": f"{_PROJECT_CONTEXT_LABEL}:\n{project_context}",
                })
            )
        messages.append({
            "role": "user",
            "content": (
                f"User request:\n{user_request}\n\n"
                f"{context_pack}\n\n"
                "Output ONE raw JSON object only with this schema:\n"
                '{"patch_plan":{"files_to_edit":["path"],'
                '"target_symbols":["symbol"],'
                '"intended_changes":["short change"],'
                '"edits":[{"path":"path","symbol":"optional",'
                '"old_string":"exact text from context","new_string":"replacement text"}],'
                '"validation_plan":["command or static check"],'
                '"requires_confirmation":false,'
                '"confidence":0.0,'
                '"missing_info":[]}}\n'
                "Do NOT output a TaskTree or nodes array in this mode. "
                "Only include edits when old_string is copied exactly from <context_pack> "
                "snippets and new_string is the precise replacement. If exact edit text "
                "is not known, return an empty edits array, confidence below 0.75, "
                "and list missing_info. Never invent old_string."
            ),
        })
        return messages

    def _cached_prefix_messages(self, project_context: str) -> list[dict[str, Any]]:
        messages = [
            mark_cache_breakpoint({"role": "system", "content": self._system_prompt}),
        ]
        if project_context.strip():
            messages.append(
                mark_cache_breakpoint({
                    "role": "system",
                    "content": f"{_PROJECT_CONTEXT_LABEL}:\n{project_context}",
                })
            )
        return messages

    def _build_user_plan_prompt(
        self,
        user_request: str,
        *,
        discovery_manifest: str | None,
    ) -> str:
        parts = [f"User request:\n{user_request}\n"]
        if discovery_manifest:
            parts.append(f"{discovery_manifest}\n")
        parts.append(planner_output_instruction(require_trace=self._require_trace))
        return "\n".join(parts)

    async def _complete_raw(self, messages: list[dict[str, Any]]) -> str:
        client = self._client
        if hasattr(client, "stream_complete"):
            parts: list[str] = []
            async for delta in client.stream_complete(messages):
                parts.append(delta)
            return "".join(parts)
        return await client.complete(messages)

    async def plan(
        self,
        user_request: str,
        project_context: str,
        *,
        discovery_manifest: str | None = None,
    ) -> PlannerParseResult:
        messages = self.plan_messages(
            user_request,
            project_context,
            discovery_manifest=discovery_manifest,
        )
        raw = await self._complete_raw(messages)
        return parse_planner_output(raw, fallback_task=user_request)

    async def rewrite_plan(
        self,
        user_request: str,
        project_context: str,
        *,
        gate_errors: str,
        previous_raw: str,
        discovery_manifest: str | None = None,
    ) -> PlannerParseResult:
        messages = self.rewrite_messages(
            user_request,
            project_context,
            gate_errors=gate_errors,
            previous_raw=previous_raw,
            discovery_manifest=discovery_manifest,
        )
        raw = await self._complete_raw(messages)
        return parse_planner_output(raw, fallback_task=user_request)

    def _build_plan_prompt(
        self,
        user_request: str,
        project_context: str,
        *,
        discovery_manifest: str | None,
    ) -> str:
        """Legacy combined prompt (project_context appended for callers that need one block)."""
        parts = [self._build_user_plan_prompt(user_request, discovery_manifest=discovery_manifest)]
        if project_context.strip():
            parts.append(f"{_PROJECT_CONTEXT_LABEL}:\n{project_context}\n")
        return "\n".join(parts)

    async def re_plan(
        self,
        user_request: str,
        current_tree: TaskTree,
        evidence: EvidencePack,
        project_context: str,
        *,
        discovery_manifest: str | None = None,
    ) -> TaskTree:
        completed_summary = "\n".join(
            f"  ✓ [{n.id}] {n.description}" for n in current_tree.completed_nodes()
        ) or "  (none)"

        prompt_parts = [
            f"Original request:\n{user_request}\n",
        ]
        if discovery_manifest:
            prompt_parts.append(f"{discovery_manifest}\n")
        completed_ids = ", ".join(n.id for n in current_tree.completed_nodes()) or "(none)"
        prompt_parts.extend([
            f"Current plan (v{current_tree.version}):\n{current_tree.to_outline()}\n",
            f"Completed:\n{completed_summary}\n",
            f"Completed ids (frozen — do not replan): {completed_ids}\n",
            f"Failure evidence:\n{evidence.to_prompt_block()}\n",
        ])
        from src.executor.retry_strategy import replan_revision_directive

        failed = current_tree.get(evidence.subtask_id)
        pending_tail = [
            n
            for n in current_tree.nodes
            if n.status == SubTaskStatus.PENDING and n.id != evidence.subtask_id
        ]
        if pending_tail:
            tail_lines = "\n".join(
                f"  - [{n.id}] kind={n.kind.value}: {n.description}"
                for n in pending_tail
            )
            prompt_parts.append(
                "Later pending steps (orchestrator keeps these unchanged — "
                "do NOT include in output):\n"
                f"{tail_lines}\n"
            )
        if failed is not None:
            prompt_parts.append(
                replan_revision_directive(
                    failed_subtask=failed,
                    error_trace=evidence.error_trace,
                )
                + "\n"
            )
            if failed.kind == SubTaskKind.EDIT:
                prompt_parts.append(
                    "The failed subtask was kind=edit. Replacement nodes MUST still "
                    "complete the edit objective. You may output diagnose + edit "
                    "when the failure needs better evidence, but do NOT output only "
                    "diagnose/verify. The final replacement node for this failure "
                    "must be kind=edit with edit_file allowed.\n"
                )
        prompt_parts.extend([
            f"Replace ONLY failed subtask [{evidence.subtask_id}]. "
            + planner_output_instruction(require_trace=self._require_trace)
            + "\n- Output nodes array with ONLY replacement subtask(s) for that step.\n"
            "- Do NOT output completed ids or later pending steps listed above.\n"
            "- Use new ids (e.g. st-2a, st-2b) that do not collide with completed ids.\n"
            "- edit context_files must list files to touch.",
        ])
        prompt = "\n".join(prompt_parts)
        messages = self._cached_prefix_messages(project_context)
        messages.append({"role": "user", "content": prompt})
        raw = await self._client.complete(messages)
        trace = extract_planning_trace(raw)
        if trace:
            log.debug("Planner re-plan trace:\n%s", trace)
        revised = parse_planner_output(raw, fallback_task=user_request)
        if not revised.ok:
            log.debug("Planner re-plan parse/schema failed: %s", "; ".join(revised.all_errors))

        merged = _merge_replanned_tree(
            current_tree,
            revised.tree,
            failed_subtask_id=evidence.subtask_id,
        )
        log.info("Planner re-plan → %d total nodes", len(merged.nodes))
        return merged


def _merge_replanned_tree(
    current_tree: TaskTree,
    revised: TaskTree,
    *,
    failed_subtask_id: str,
) -> TaskTree:
    """Replace the failed subtask only; keep completed and later pending steps."""
    merged = TaskTree(root_task=current_tree.root_task, version=current_tree.version)
    failed_idx: int | None = None
    for i, node in enumerate(current_tree.nodes):
        if node.id == failed_subtask_id:
            failed_idx = i
            break

    if failed_idx is None:
        merged.nodes = list(current_tree.completed_nodes())
        merged_ids = {n.id for n in merged.nodes}
        for node in revised.nodes:
            if node.id in merged_ids:
                continue
            node.status = SubTaskStatus.PENDING
            node.checkpoint_id = None
            merged.nodes.append(node)
            merged_ids.add(node.id)
    else:
        merged.nodes = list(current_tree.nodes[:failed_idx])
        merged_ids = {n.id for n in merged.nodes}

        for node in revised.nodes:
            if node.id in merged_ids:
                continue
            node.status = SubTaskStatus.PENDING
            node.checkpoint_id = None
            merged.nodes.append(node)
            merged_ids.add(node.id)

        for node in current_tree.nodes[failed_idx + 1 :]:
            merged.nodes.append(node)

    replacement_id = revised.nodes[-1].id if revised.nodes else None
    for node in merged.nodes:
        if node in revised.nodes:
            continue
        if failed_subtask_id in node.depends_on:
            if replacement_id:
                node.depends_on = [
                    replacement_id if dep == failed_subtask_id else dep
                    for dep in node.depends_on
                ]
            else:
                node.depends_on = [dep for dep in node.depends_on if dep != failed_subtask_id]

    merged.version = current_tree.version + 1
    return merged


def fallback_replan_for_failed_edit(
    current_tree: TaskTree,
    *,
    failed_subtask_id: str,
    task_analysis: Any | None = None,
    error_trace: list[str] | None = None,
) -> TaskTree | None:
    """Deterministic replacement when Planner drops the edit during re-plan."""
    failed = current_tree.get(failed_subtask_id)
    if failed is None or failed.kind != SubTaskKind.EDIT:
        return None

    from src.executor.retry_strategy import parse_validation_error_details

    details = parse_validation_error_details(error_trace or [])
    expected_syms = details["expected_target_symbols"]
    wrong_syms = details["wrong_modified_symbols"]
    wrong_lines = details["wrong_modified_lines"]

    suffix = f" (re-plan {failed_subtask_id})"
    custom_hint_diag = ""
    custom_hint_crit = ""
    custom_hint_design = ""
    custom_hint_edit_crit = ""

    if expected_syms or wrong_syms:
        expected_str = ", ".join(expected_syms) if expected_syms else "原目标符号"
        wrong_str = ", ".join(wrong_syms) if wrong_syms else "误修改符号"
        
        custom_hint_diag = f"，定位原目标符号 {expected_str} 的正确位置并避开错误修改的符号 {wrong_str}"
        custom_hint_crit = f"，需输出正确目标符号 {expected_str} 的行范围且不包含错误修改的符号 {wrong_str}"
        custom_hint_design = f"，仅针对原目标符号 {expected_str} 设计补丁，严禁包含错误符号 {wrong_str}"
        custom_hint_edit_crit = f"，必须且仅修改原目标符号 {expected_str} 且限制在其正确允许范围内，不得修改错误符号 {wrong_str}"

    diag_id = f"{failed.id}a"
    design_id = f"{failed.id}b"
    edit_id = f"{failed.id}c"
    desc = failed.description.strip() or current_tree.root_task
    strategy = _analysis_strategy(task_analysis)
    complexity = "medium"
    if hasattr(task_analysis, "complexity"):
        complexity = str(getattr(task_analysis, "complexity") or "medium")
    elif isinstance(task_analysis, dict):
        complexity = str(task_analysis.get("complexity") or "medium")

    if complexity == "high" or strategy in ("sql_view_rewrite", "function_refactor"):
        if strategy == "sql_view_rewrite":
            diag_desc = f"定位{desc}的目标代码、当前 SQL 和 Harness 解析的视图依赖{custom_hint_diag}{suffix}"[:120]
            diag_crit = f"输出 HANDOFF_CONTRACT_JSON：file:line、symbol、SQL、resolved dependency、片段/决策{custom_hint_crit}"
            design_desc = f"设计{desc}的视图替换补丁意图{custom_hint_design}{suffix}"[:120]
            edit_crit = f"仅修改设计指定目标，使用 Harness 批准的视图依赖且 old/new 不同{custom_hint_edit_crit or suffix}"
        elif strategy == "function_refactor":
            diag_desc = f"定位{desc}的调用点、旧逻辑和可复用依赖{custom_hint_diag}{suffix}"[:120]
            diag_crit = f"输出 HANDOFF_CONTRACT_JSON：file:line、symbol、调用点、可复用依赖、片段/决策{custom_hint_crit}"
            design_desc = f"设计{desc}的函数重构补丁意图{custom_hint_design}{suffix}"[:120]
            edit_crit = f"仅修改设计指定目标，复用 Harness 批准的 helper/function 并移除旧重复逻辑{custom_hint_edit_crit or suffix}"
        else:
            diag_desc = f"定位{desc}的目标代码和当前实现{custom_hint_diag}{suffix}"[:120]
            diag_crit = f"输出 HANDOFF_CONTRACT_JSON：file:line、symbol、当前片段、片段/决策{custom_hint_crit}"
            design_desc = f"设计{desc}的修改补丁意图{custom_hint_design}{suffix}"[:120]
            edit_crit = f"仅修改设计指定目标，按 Harness/Design 批准的策略完成修改{custom_hint_edit_crit or suffix}"

        nodes = [
            SubTaskNode(
                id=diag_id,
                kind=SubTaskKind.DIAGNOSE,
                description=diag_desc,
                acceptance_criteria=diag_crit,
                allowed_tools=["context_search"],
                context_files=list(failed.context_files),
                needs_l1=False,
            ),
            SubTaskNode(
                id=design_id,
                kind=SubTaskKind.DESIGN,
                description=design_desc,
                acceptance_criteria=(
                    f"输出完整 PATCH_INTENT_JSON：edit_ready=true、edit_strategy、"
                    f"edit_targets、dependencies、acceptance_criteria、target_view{suffix}"
                ),
                allowed_tools=["context_search"],
                context_files=list(failed.context_files),
                depends_on=[diag_id],
                needs_l1=False,
                handoff_outputs=["PATCH_INTENT_JSON"],
            ),
        ]

        editable_targets = []
        if task_analysis is not None:
            if hasattr(task_analysis, "editable_targets"):
                editable_targets = list(getattr(task_analysis, "editable_targets") or [])
            elif isinstance(task_analysis, dict):
                editable_targets = list(task_analysis.get("editable_targets") or [])

        score = calculate_edit_complexity_score(task_analysis)

        if len(editable_targets) > 1 and score >= 2.0:
            for idx, t_item in enumerate(editable_targets):
                symbol_name = t_item.get("symbol") or f"target_{idx+1}"
                file_name = t_item.get("file") or ""
                sub_id = f"{failed.id}c_{idx+1}"
                dep = [nodes[-1].id]
                nodes.append(
                    SubTaskNode(
                        id=sub_id,
                        kind=SubTaskKind.EDIT,
                        description=f"分块修改第{idx+1}个目标 {symbol_name} ({file_name}){suffix}"[:120],
                        acceptance_criteria=edit_crit,
                        allowed_tools=["context_search", "edit_file"],
                        context_files=[file_name] if file_name else list(failed.context_files),
                        depends_on=dep,
                        needs_l1=failed.needs_l1 if isinstance(failed.needs_l1, bool) else True,
                        requires_handoff=["PATCH_INTENT_JSON"],
                    )
                )
        else:
            nodes.append(
                SubTaskNode(
                    id=edit_id,
                    kind=SubTaskKind.EDIT,
                    description=f"基于 PATCH_INTENT_JSON 分块修改单个 edit_target：{desc}{suffix}"[:120],
                    acceptance_criteria=edit_crit,
                    allowed_tools=["context_search", "edit_file"],
                    context_files=list(failed.context_files),
                    depends_on=[design_id],
                    needs_l1=failed.needs_l1 if isinstance(failed.needs_l1, bool) else True,
                    requires_handoff=["PATCH_INTENT_JSON"],
                )
            )
    else:
        edit_id = f"{failed.id}b"
        diag_desc = f"定位{desc}的目标代码和当前实现{custom_hint_diag}{suffix}"[:120]
        diag_crit = f"输出 HANDOFF_CONTRACT_JSON：file:line、symbol、当前片段、片段/决策{custom_hint_crit}"
        edit_crit = f"仅修改 handoff 指定代码，按 Harness 批准的策略完成修改{custom_hint_edit_crit or suffix}"

        nodes = [
            SubTaskNode(
                id=diag_id,
                kind=SubTaskKind.DIAGNOSE,
                description=diag_desc,
                acceptance_criteria=diag_crit,
                allowed_tools=["context_search"],
                context_files=list(failed.context_files),
                needs_l1=False,
            ),
            SubTaskNode(
                id=edit_id,
                kind=SubTaskKind.EDIT,
                description=f"基于 handoff 完成：{desc}{suffix}"[:120],
                acceptance_criteria=edit_crit,
                allowed_tools=["context_search", "edit_file"],
                context_files=list(failed.context_files),
                depends_on=[diag_id],
                needs_l1=failed.needs_l1 if isinstance(failed.needs_l1, bool) else True,
            ),
        ]
    revised = TaskTree(
        root_task=current_tree.root_task,
        nodes=nodes,
    )
    return _merge_replanned_tree(
        current_tree,
        revised,
        failed_subtask_id=failed_subtask_id,
    )


def parse_planner_output(raw: str, *, fallback_task: str) -> PlannerParseResult:
    """Parse Planner LLM text → TaskTree + schema status (no silent fallback)."""
    trace = extract_planning_trace(raw)
    if trace:
        log.debug("Planner trace:\n%s", trace)
    payload, json_ok = _extract_json_payload(raw)
    if json_ok:
        payload = normalize_planner_payload(payload)
    schema_errors = validate_planner_payload(payload) if json_ok else []
    tree = build_task_tree_from_payload(payload, fallback_task=fallback_task)
    if not tree.nodes and json_ok:
        schema_errors.append("nodes array produced no valid subtasks.")
    result = PlannerParseResult(
        raw=raw,
        payload=payload,
        tree=tree,
        json_ok=json_ok,
        schema_errors=schema_errors,
    )
    if result.ok:
        log.info("Planner produced %d subtasks", len(tree.nodes))
    else:
        log.debug("Planner output invalid: %s", "; ".join(result.all_errors))
    return result


def fallback_task_tree(fallback_task: str, *, task_analysis: Any | None = None) -> TaskTree:
    """Last-resort plan when PlanGate exhausts rewrites."""
    return task_tree_from_template(
        fallback_task,
        select_task_template(fallback_task),
        task_analysis=task_analysis,
    )


def _analysis_strategy(task_analysis: Any | None) -> str:
    if hasattr(task_analysis, "edit_strategy"):
        return str(getattr(task_analysis, "edit_strategy") or "")
    if isinstance(task_analysis, dict):
        return str(task_analysis.get("edit_strategy") or task_analysis.get("intent") or "")
    return ""


def _parse_task_tree(raw: str, *, fallback_task: str) -> TaskTree:
    """Legacy helper for tests — uses fallback when parse/schema fails."""
    result = parse_planner_output(raw, fallback_task=fallback_task)
    if result.ok and result.tree.nodes:
        return result.tree
    if not result.tree.nodes:
        return fallback_task_tree(fallback_task)
    return result.tree


def extract_planning_trace(raw: str) -> str | None:
    """Return CoT block from Planner output, if present."""
    match = re.search(
        r"<planning_trace>\s*(.*?)\s*</planning_trace>",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip() or None


def _fallback_subtask(fallback_task: str) -> SubTaskNode:
    """Safe single-node plan when Planner output is unusable."""
    return SubTaskNode(
        id="st-1",
        description=fallback_task[:120],
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Search project and list relevant file paths with evidence",
        allowed_tools=default_allowed_tools(SubTaskKind.DIAGNOSE),
        context_files=[],
        needs_l1=False,
    )


def _repair_json_text(text: str) -> str:
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    repaired = re.sub(
        r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        repaired,
    )
    return repaired


def _extract_json_payload(raw: str) -> tuple[dict[str, Any], bool]:
    text = raw.strip()
    trace_end = re.search(r"</planning_trace>", text, re.IGNORECASE)
    if trace_end:
        text = text[trace_end.end() :].strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    last_exc: json.JSONDecodeError | None = None
    for candidate in (text, _repair_json_text(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_exc = exc
            continue
        if isinstance(data, dict):
            return data, True
        return {}, False
    log.debug("Planner JSON parse failed: %s", last_exc)
    return {"root_task": "", "nodes": []}, False


def _extract_json(raw: str) -> dict[str, Any]:
    payload, _ = _extract_json_payload(raw)
    return payload
