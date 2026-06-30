from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.agent.types import Message, system_message, user_message
from src.context.prompt_resources import load_internal_prompt
from src.executor.final_output import parse_executor_final
from src.harness.gates.types import TruncationPolicy
from src.harness.probe.token_budget import TokenBudget
from src.harness.subtask.artifacts import build_artifact_store
from src.harness.subtask.preload import load_context_file_contents
from src.planner.context_policy import effective_context_files
from src.planner.kinds import SubTaskKind
from src.planner.prior_context import (
    extract_line_refs_from_text,
    extract_paths_from_text,
    extract_symbol_hits_from_text,
    format_diagnose_handoff_block,
)
from src.planner.task_tree import SubTaskNode, TaskTree

log = logging.getLogger(__name__)

_EXECUTOR_FALLBACK = """You are MitKII Executor — complete ONE subtask using only allowed_tools."""

_EXECUTOR_SYSTEM_PROMPT: str | None = None
_TOKEN_COUNTER = TokenBudget()


def load_executor_system_prompt() -> str:
    global _EXECUTOR_SYSTEM_PROMPT
    if _EXECUTOR_SYSTEM_PROMPT is not None:
        return _EXECUTOR_SYSTEM_PROMPT
    _EXECUTOR_SYSTEM_PROMPT = load_internal_prompt(
        "executor_prompt.md",
        fallback=_EXECUTOR_FALLBACK,
    )
    return _EXECUTOR_SYSTEM_PROMPT


def _build_context_payload_block(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    project_root: Path,
    policy: TruncationPolicy,
    preload_mode: str,
    runtime_tools: frozenset[str] | None,
    context_pack: Any | None = None,
) -> str:
    context_files = effective_context_files(task_tree, subtask)
    runtime = sorted(runtime_tools if runtime_tools is not None else subtask.effective_allowed_tools())
    payload: dict[str, Any] = {
        "schema": "mitkii.executor_context_payload.v1",
        "subtask_id": subtask.id,
        "kind": subtask.kind.value,
        "allowed_tools": runtime,
        "quality_gate_l1": subtask.effective_needs_l1(),
        "context_files": list(context_files),
        "preload_mode": preload_mode,
        "policy_tier": policy.tier,
        "instructions": _context_payload_instructions(
            subtask=subtask,
            context_files=context_files,
            preload_mode=preload_mode,
            policy=policy,
        ),
    }
    if context_pack is not None:
        payload["context_pack"] = context_pack.to_agent_json(max_snippet_chars=4_000)
    parts = [
        "CONTEXT_PAYLOAD_JSON",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]

    if context_files and preload_mode != "paths_only" and policy.tier != "red":
        for rel, content in load_context_file_contents(
            project_root, context_files, policy=policy
        ):
            parts.append(f'\n<file path="{rel}">\n{content}\n</file>')
    return "\n".join(parts)


def _context_payload_instructions(
    *,
    subtask: SubTaskNode,
    context_files: list[str],
    preload_mode: str,
    policy: TruncationPolicy,
) -> list[str]:
    instructions: list[str] = []
    if context_files and preload_mode == "paths_only":
        instructions.append(
            "Paths are scoped but not fully preloaded. Use context_search with paths before editing."
        )
    elif context_files and policy.tier != "red":
        instructions.append("Full or sliced file content follows in <file> blocks.")
        if subtask.kind == SubTaskKind.EDIT:
            instructions.append(
                "For edit_file, copy a unique multi-line old_string from <file> content."
            )
            instructions.append("Never create /tmp or scratch files.")
        elif subtask.kind == SubTaskKind.DIAGNOSE:
            instructions.append("Use preloaded evidence and context_search only if exposed.")
    else:
        instructions.append(
            "No file content is preloaded. Use context_search for file:line/symbol/snippet evidence."
        )
    return instructions


def build_executor_messages(
    *,
    root_task: str,
    task_tree: TaskTree,
    subtask: SubTaskNode,
    project_root: Path,
    policy: TruncationPolicy | None = None,
    prior_summaries: dict[str, str] | None = None,
    preload_mode: str = "full",
    runtime_tools: frozenset[str] | None = None,
    context_pack: Any | None = None,
) -> list[Message]:
    """Layered Executor system messages (L0 prompt | L2 task tree | L3 subtask + preload)."""
    policy = policy or TruncationPolicy.green()

    context_files = effective_context_files(task_tree, subtask)
    handoff = build_executor_handoff_json(
        root_task=root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        runtime_tools=runtime_tools,
        context_files=context_files,
        prior_summaries=prior_summaries or {},
        context_pack=context_pack,
    )
    task_parts = [
        "EXECUTOR_HANDOFF_JSON",
        json.dumps(handoff, ensure_ascii=False, indent=2),
    ]
    if subtask.kind == SubTaskKind.EDIT and prior_summaries:
        intent = " ".join(
            part
            for part in (root_task, subtask.description, subtask.acceptance_criteria or "")
            if part
        )
        handoff = format_diagnose_handoff_block(
            prior_summaries,
            project_root,
            intent_text=intent,
        )
        if handoff:
            task_parts.extend(["", "DIAGNOSE_SLICE_HINT", handoff])

    scope = _build_context_payload_block(
        subtask=subtask,
        task_tree=task_tree,
        project_root=project_root,
        policy=policy,
        preload_mode=preload_mode,
        runtime_tools=runtime_tools,
        context_pack=context_pack,
    )

    return [
        system_message(load_executor_system_prompt(), cache_breakpoint=True),
        system_message("\n".join(task_parts), cache_breakpoint=True),
        system_message(scope, cache_breakpoint=True),
    ]


def build_executor_handoff_json(
    *,
    root_task: str,
    task_tree: TaskTree,
    subtask: SubTaskNode,
    project_root: Path,
    runtime_tools: frozenset[str] | None,
    context_files: list[str],
    prior_summaries: dict[str, str],
    context_pack: Any | None = None,
) -> dict[str, Any]:
    allowed_tools = sorted(runtime_tools if runtime_tools is not None else subtask.effective_allowed_tools())
    siblings = [
        {
            "id": node.id,
            "kind": node.kind.value,
            "status": node.status.value,
            "depends_on": list(node.depends_on),
        }
        for node in task_tree.nodes
    ]
    prior = _prior_handoff(
        prior_summaries,
        project_root,
    )
    artifact_store = build_artifact_store(
        prior_summaries,
        required=subtask.requires_artifacts,
    )
    payload: dict[str, Any] = {
        "schema": "mitkii.executor_handoff.v1",
        "root_task": root_task,
        "subtask": {
            "id": subtask.id,
            "kind": subtask.kind.value,
            "description": subtask.description,
            "acceptance_criteria": subtask.acceptance_criteria,
            "depends_on": list(subtask.depends_on),
            "requires_artifacts": list(subtask.requires_artifacts),
            "produces_artifacts": list(subtask.produces_artifacts),
            "write_scope": list(subtask.write_scope),
        },
        "allowed_tools": allowed_tools,
        "denied_tools": _denied_tools_for_allowed(allowed_tools),
        "context_scope": {
            "context_files": list(context_files),
            "write_scope": list(subtask.write_scope),
            "preload_mode": "paths_only" if context_files and runtime_tools and "context_search" in runtime_tools else "full_or_none",
        },
        "prior": {
            "facts": prior["facts"],
            "evidence": prior["evidence"],
            "known_negatives": prior["known_negatives"],
            "next_focus": prior["next_focus"],
        },
        "artifact_store": artifact_store,
        "plan_state": {
            "nodes": siblings,
        },
        "requirements": {
            "final_output": {
                "format": "json_object",
                "required_keys": [
                    "status",
                    "changed_files",
                    "validation",
                    "risks",
                    "handoff",
                ],
            },
            "tool_policy": (
                "Only tools in allowed_tools are available. Denied tools are not "
                "present in runtime schemas and must not be requested."
            ),
        },
    }
    if context_pack is not None:
        pack_json = context_pack.to_agent_json(max_snippet_chars=2_000)
        payload["context_pack_summary"] = {
            "confidence": pack_json.get("confidence"),
            "candidate_files": pack_json.get("candidate_files", []),
            "candidate_symbols": pack_json.get("candidate_symbols", []),
            "evidence": pack_json.get("evidence", []),
            "known_negatives": pack_json.get("known_negatives", []),
            "call_chain": pack_json.get("call_chain", []),
            "tool_policy": pack_json.get("tool_policy", {}),
            "budget": pack_json.get("budget", {}),
        }
    return payload


def _denied_tools_for_allowed(allowed_tools: list[str]) -> list[str]:
    catalog = {
        "context_search",
        "read_file",
        "read_files",
        "grep_search",
        "map_search",
        "glob_files",
        "list_dir",
        "git_status",
        "edit_file",
        "write_file",
        "delete_file",
        "replace_symbol",
        "shell_exec",
    }
    return sorted(catalog - set(allowed_tools))


def _prior_handoff(
    prior_summaries: dict[str, str],
    project_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    facts: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    next_focus: list[dict[str, Any]] = []
    for sid, text in prior_summaries.items():
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
                        negatives.append({"source_subtask": sid, **item})
                for item in parsed.handoff.get("next_focus", []) or []:
                    if isinstance(item, dict):
                        next_focus.append({"source_subtask": sid, **item})
                    elif str(item).strip():
                        next_focus.append({"source_subtask": sid, "focus": str(item)})
            if parsed.blocker or parsed.acceptance_met is False:
                negatives.append({
                    "source_subtask": sid,
                    "reason": parsed.blocker or parsed.result,
                })
        for rel, span in extract_line_refs_from_text(text, project_root).items():
            evidence.append({
                "source_subtask": sid,
                "path": rel,
                "line_range": f"{span[0]}-{span[1]}",
            })
        for rel in extract_paths_from_text(text, project_root):
            evidence.append({"source_subtask": sid, "path": rel})
        for rel, line, symbol in extract_symbol_hits_from_text(text, project_root):
            evidence.append({
                "source_subtask": sid,
                "path": rel,
                "line": str(line),
                "symbol": symbol,
            })
        lower = text.lower()
        if any(marker in lower for marker in ("no matches", "not found", "没有命中", "未找到")):
            negatives.append({"source_subtask": sid, "reason": text[:500]})
    return {
        "facts": _dedupe_dicts(facts, limit=20),
        "evidence": _dedupe_dicts(evidence, limit=20),
        "known_negatives": _dedupe_dicts(negatives, limit=10),
        "next_focus": _dedupe_dicts(next_focus, limit=20),
    }


def _dedupe_dicts(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def build_executor_scope_message(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    project_root: Path,
    policy: TruncationPolicy | None = None,
    preload_mode: str = "full",
    runtime_tools: frozenset[str] | None = None,
    context_pack: Any | None = None,
) -> Message:
    """L3 only — subtask scope plus optional file preload."""
    policy = policy or TruncationPolicy.green()
    scope = _build_context_payload_block(
        subtask=subtask,
        task_tree=task_tree,
        project_root=project_root,
        policy=policy,
        preload_mode=preload_mode,
        runtime_tools=runtime_tools,
        context_pack=context_pack,
    )
    return system_message(scope, cache_breakpoint=True)


def rebuild_executor_retry_messages(
    *,
    root_task: str,
    task_tree: TaskTree,
    subtask: SubTaskNode,
    project_root: Path,
    prior_summaries: dict[str, str] | None,
    error_trace: list[str],
    context_files: list[str],
    runtime_tools: frozenset[str] | None = None,
    exploration_digest: str | None = None,
    context_pack: Any | None = None,
) -> list[Message]:
    from src.executor.retry_strategy import classify_failure_pattern

    pattern = classify_failure_pattern(error_trace)
    messages = build_executor_messages(
        root_task=root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        policy=TruncationPolicy.green(),
        prior_summaries=prior_summaries,
        preload_mode="paths_only",
        runtime_tools=runtime_tools,
        context_pack=context_pack,
    )
    runtime_payload: dict[str, Any] = {
        "schema": "mitkii.executor_runtime.v1",
        "event": "retry",
        "subtask_id": subtask.id,
        "failure_pattern": pattern,
        "rules": [
            "Change the approach from the failed attempt.",
            "Do not repeat the same tool arguments or same edit/write result.",
            "Continue from session_digest if present.",
            "Use edit_file for partial changes; write_file only for new files or complete-file rewrites.",
            "Do not create /tmp or scratch files.",
        ],
        "allowed_paths": list(context_files),
        "recent_errors": [str(e) for e in error_trace[-6:]],
    }
    if exploration_digest and exploration_digest.strip():
        messages.append(_json_system_message(
            "SESSION_DIGEST_JSON",
            {
                "schema": "mitkii.session_digest.v1",
                "digest": exploration_digest.strip(),
            },
        ))
    messages.append(_json_user_message("EXECUTOR_RUNTIME_JSON", runtime_payload))
    return messages


def _json_system_message(label: str, payload: dict[str, Any]) -> Message:
    return system_message(
        label + "\n" + json.dumps(payload, ensure_ascii=False, indent=2),
        cache_breakpoint=True,
    )


def _json_user_message(label: str, payload: dict[str, Any]) -> Message:
    return user_message(label + "\n" + json.dumps(payload, ensure_ascii=False, indent=2))


def count_messages_tokens(messages: list[Message]) -> int:
    return _message_content_tokens(messages) + 3


def _message_content_tokens(messages: list[Message]) -> int:
    return sum(_TOKEN_COUNTER.count_message_tokens(m.to_dict()) for m in messages)


def estimate_messages_tokens(
    messages: list[Message],
    *,
    prefix_len: int = 0,
    prefix_tokens: int | None = None,
) -> int:
    """Count tokens; when prefix is stable, only tiktoken the ReAct tail."""
    if prefix_len <= 0 or prefix_tokens is None:
        return count_messages_tokens(messages)
    if len(messages) <= prefix_len:
        return prefix_tokens
    tail = _message_content_tokens(messages[prefix_len:])
    # prefix_tokens includes the single reply primer (+3) from seed_prefix
    return prefix_tokens + tail


def estimate_executor_prompt_tokens(
    counter: TokenBudget,
    *,
    root_task: str,
    task_tree: TaskTree,
    subtask: SubTaskNode,
    project_root: Path,
    policy: TruncationPolicy,
) -> int:
    messages = build_executor_messages(
        root_task=root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        policy=policy,
    )
    return counter.count_tokens([m.to_dict() for m in messages])
