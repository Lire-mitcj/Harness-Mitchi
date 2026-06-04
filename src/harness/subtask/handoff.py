from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.agent.types import Message, harness_nudge, system_message, user_message
from src.config.settings import MitKIISettings
from src.executor.edit_guard import edit_failure_hint
from src.executor.policy import effective_max_turns, resolve_executor_tools
from src.executor.retry_strategy import (
    ExecutorRetryStrategy,
    build_executor_retry_strategy,
    classify_failure_pattern,
)
from src.harness.edit.format import format_edit_targets_block
from src.harness.edit.resolve import resolve_edit_targets
from src.harness.edit.target import EditHandoff
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.context_pipeline import ExecutorContextConfig, ExecutorRuntimeState
from src.harness.subtask.preload import (
    detect_truncated_preloads,
    load_context_file_contents,
    norm_rel_path,
    preloaded_paths,
)
from src.harness.subtask.prompt_builder import (
    build_executor_messages,
    rebuild_executor_retry_messages,
)
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.planner.context_policy import effective_context_files
from src.planner.kinds import SubTaskKind
from src.planner.prior_context import extract_line_refs_from_summaries
from src.planner.task_tree import SubTaskNode, TaskTree

EDIT_TURN_RESERVE = 2
EDIT_TOOLS = frozenset({"edit_file", "write_file", "delete_file", "replace_symbol"})
DIAGNOSE_LOCATE_TOOLS = frozenset({"context_search"})


@dataclass
class SubtaskHandoffBundle:
    """Harness-prepared Executor input: prompt, runtime policy, context session inputs."""

    subtask: SubTaskNode
    task_tree: TaskTree
    root_task: str
    project_root: Path
    initial_messages: list[Message]
    startup_status: list[str] = field(default_factory=list)
    session_memory: ExploreSessionMemory = field(default_factory=ExploreSessionMemory.create)
    ctx_config: ExecutorContextConfig | None = None
    ctx_runtime: ExecutorRuntimeState | None = None
    retry_strategy: ExecutorRetryStrategy | None = None
    policy: TruncationPolicy = field(default_factory=TruncationPolicy.green)
    whitelist_files: list[str] = field(default_factory=list)
    diag_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    diag_handoff: bool = False
    edit_handoff: EditHandoff | None = None
    edit_splice_max_attempts: int = 2
    max_turns: int = 3
    turn_cap: int = 3
    qg_limit: int = 1
    context_search_available: bool = False


def prepare_executor_handoff(
    *,
    root_task: str,
    task_tree: TaskTree,
    subtask: SubTaskNode,
    project_root: Path,
    settings: MitKIISettings,
    truncation_policy: TruncationPolicy | None = None,
    prior_summaries: dict[str, str] | None = None,
    retry_feedback: list[str] | None = None,
    prior_exploration: str | None = None,
    subtask_attempt: int = 1,
    quality_gate_retry_limit: int | None = None,
    context_search_in_registry: bool = True,
    context_pack: Any | None = None,
) -> SubtaskHandoffBundle:
    """Build subtask prompt + runtime policy (Harness — not LLM reasoning)."""
    policy = truncation_policy or TruncationPolicy.green(
        settings.preflight_max_chars_per_file
    )
    diag_slices = extract_line_refs_from_summaries(
        prior_summaries or {},
        project_root,
        padding=settings.preflight_slice_padding,
    )
    if diag_slices:
        policy = replace(
            policy,
            line_slices={**(policy.line_slices or {}), **diag_slices},
        )

    retry_strategy = build_executor_retry_strategy(
        subtask,
        subtask_attempt=subtask_attempt,
        prior_errors=retry_feedback,
        prior_exploration=prior_exploration,
    )
    explore_restricted = retry_strategy.restrict_explore
    session_memory = ExploreSessionMemory.create(
        prior_exploration=prior_exploration,
        policy=policy,
        project_root=project_root,
    )
    max_turns = effective_max_turns(subtask.kind, settings)
    whitelist_files = effective_context_files(task_tree, subtask)
    if diag_slices:
        seen_whitelist = {norm_rel_path(p) for p in whitelist_files}
        for rel in diag_slices:
            if norm_rel_path(rel) not in seen_whitelist:
                whitelist_files.append(rel)
                subtask.context_files.append(rel)
                seen_whitelist.add(norm_rel_path(rel))
    whitelist_norm = frozenset(norm_rel_path(p) for p in whitelist_files)
    paths_only_mode = retry_strategy.paths_only
    loaded_contents: list[tuple[str, str]] = []
    if whitelist_files and policy.tier != "red":
        loaded_contents = load_context_file_contents(
            project_root, whitelist_files, policy=policy
        )
    files_in_prompt = preloaded_paths(
        project_root, whitelist_files, policy=policy, loaded=loaded_contents or None
    )
    diag_handoff = subtask.kind == SubTaskKind.EDIT and bool(diag_slices)
    coordinate_handoff = bool(diag_slices)
    diag_preloaded = diag_handoff and bool(files_in_prompt)
    prior_edit_fail = classify_failure_pattern(retry_feedback or []) in {
        "edit_not_found",
        "edit_identical",
        "edit_ambiguous",
    }
    incremental_edit = _prefers_incremental_edit(root_task, subtask)
    edit_read_fallback = prior_edit_fail and diag_handoff
    if coordinate_handoff and not prior_edit_fail:
        explore_restricted = True
    truncated_paths = detect_truncated_preloads(
        project_root, whitelist_files, policy=policy, loaded=loaded_contents or None
    )
    auto_scoped_edit = (
        subtask.kind == SubTaskKind.EDIT
        and not paths_only_mode
        and not explore_restricted
        and not diag_handoff
        and (bool(truncated_paths) or len(whitelist_files) >= 2)
    )
    use_paths_only = paths_only_mode or auto_scoped_edit
    preloaded_paths_set = files_in_prompt
    if (
        (use_paths_only or (subtask.kind == SubTaskKind.EDIT and truncated_paths))
        and not diag_handoff
    ):
        preloaded_paths_set = frozenset()
        if not truncated_paths:
            truncated_paths = whitelist_norm
    effective_preloaded = files_in_prompt if diag_handoff else preloaded_paths_set
    edit_handoff: EditHandoff | None = None
    if (
        subtask.kind == SubTaskKind.EDIT
        and settings.edit_splice_enabled
        and not incremental_edit
        and prior_summaries
    ):
        intent = " ".join(
            part
            for part in (root_task, subtask.description, subtask.acceptance_criteria or "")
            if part
        )
        targets = resolve_edit_targets(
            project_root=project_root,
            prior_summaries=prior_summaries,
            whitelist_files=whitelist_files,
            intent_text=intent,
        )
        if targets:
            edit_handoff = EditHandoff(targets=tuple(targets))
    splice_edit = bool(edit_handoff and edit_handoff.active)
    runtime_tools = resolve_executor_tools(
        subtask,
        preloaded_paths=effective_preloaded,
        truncated_paths=truncated_paths,
        explore_restricted=explore_restricted,
        splice_edit=splice_edit,
        edit_read_fallback=edit_read_fallback,
    )
    context_search_available = (
        context_search_in_registry
        and subtask.kind in {SubTaskKind.DIAGNOSE, SubTaskKind.EDIT}
        and not explore_restricted
    )
    if context_search_available:
        runtime_tools = runtime_tools | frozenset({"context_search"})
    runtime_tools = _apply_context_pack_tool_policy(runtime_tools, context_pack)

    compact_token_threshold = int(
        settings.max_context_tokens * settings.executor_compact_context_ratio
    )
    ctx_config = ExecutorContextConfig(
        root_task=root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        policy=policy,
        prior_summaries=prior_summaries,
        whitelist_files=whitelist_files,
        whitelist_norm=whitelist_norm,
        diag_handoff=diag_handoff,
        compact_token_threshold=compact_token_threshold,
        context_pack=context_pack,
    )
    ctx_runtime = ExecutorRuntimeState(
        paths_only_mode=paths_only_mode,
        use_paths_only=use_paths_only,
        preloaded_paths=preloaded_paths_set,
        truncated_paths=truncated_paths,
        active_runtime_tools=runtime_tools,
        explore_restricted=explore_restricted,
        edit_read_fallback=edit_read_fallback,
    )
    if ctx_runtime.edit_read_fallback:
        from src.executor.policy import enable_edit_read_fallback

        enable_edit_read_fallback(subtask=subtask, runtime=ctx_runtime)

    startup_status: list[str] = []
    if splice_edit and edit_handoff:
        syms = ", ".join(f"{t.path}:{t.symbol}" for t in edit_handoff.targets[:3])
        startup_status.append(
            f"Executor [{subtask.id}] · splice edit ({len(edit_handoff.targets)} target(s): {syms})"
        )
    elif incremental_edit and subtask.kind == SubTaskKind.EDIT:
        startup_status.append(
            f"Executor [{subtask.id}] · incremental edit "
            "(context_search first; splice disabled for query-style change)"
        )
    elif diag_handoff and diag_preloaded:
        startup_status.append(
            f"Executor [{subtask.id}] · diagnose handoff "
            "(slice preload — edit_file only)"
        )
    elif coordinate_handoff and files_in_prompt:
        startup_status.append(
            f"Executor [{subtask.id}] · coordinate handoff "
            "(slice preload — raw IO disabled)"
        )
    elif whitelist_files and not use_paths_only:
        startup_status.append(f"Preloading {len(whitelist_files)} context file(s)...")

    if paths_only_mode:
        messages = rebuild_executor_retry_messages(
            root_task=root_task,
            task_tree=task_tree,
            subtask=subtask,
            project_root=project_root,
            prior_summaries=prior_summaries,
            error_trace=list(retry_feedback or []),
            context_files=whitelist_files,
            runtime_tools=runtime_tools,
            exploration_digest=prior_exploration,
            context_pack=context_pack,
        )
        if explore_restricted:
            startup_status.append(
                f"Executor [{subtask.id}] · retry mode "
                "(digest — context_search/read fallback only if exposed, then edit_file)"
            )
        else:
            startup_status.append(
                f"Executor [{subtask.id}] · retry mode (paths only — "
                "context_search, then edit_file)"
            )
    elif auto_scoped_edit:
        messages = build_executor_messages(
            root_task=root_task,
            task_tree=task_tree,
            subtask=subtask,
            project_root=project_root,
            policy=policy,
            prior_summaries=prior_summaries,
            preload_mode="paths_only",
            runtime_tools=runtime_tools,
            context_pack=context_pack,
        )
        startup_status.append(
            f"Executor [{subtask.id}] · scoped mode "
            f"({len(whitelist_files)} files — context_search, then edit)"
        )
    else:
        messages = build_executor_messages(
            root_task=root_task,
            task_tree=task_tree,
            subtask=subtask,
            project_root=project_root,
            policy=policy,
            prior_summaries=prior_summaries,
            runtime_tools=runtime_tools,
            context_pack=context_pack,
        )

    if files_in_prompt and not use_paths_only:
        loaded = ", ".join(sorted(files_in_prompt)[:4])
        if len(files_in_prompt) > 4:
            loaded += ", …"
        startup_status.append(
            f"Preloaded {len(files_in_prompt)} file(s): {loaded}"
        )

    if edit_handoff and edit_handoff.active:
        messages.append(system_message(format_edit_targets_block(edit_handoff.targets)))

    messages.append(
        _bootstrap_runtime_message(
            subtask=subtask,
            root_task=root_task,
            project_root=project_root,
            retry_strategy=retry_strategy,
            paths_only_mode=paths_only_mode,
            auto_scoped_edit=auto_scoped_edit,
            explore_restricted=explore_restricted,
            diag_handoff=diag_handoff,
            splice_edit=splice_edit,
            edit_handoff=edit_handoff,
            edit_read_fallback=edit_read_fallback,
            diag_slices=diag_slices,
            whitelist_files=whitelist_files,
            files_in_prompt=files_in_prompt,
            truncated_paths=truncated_paths,
            runtime_tools=runtime_tools,
            prior_exploration=prior_exploration,
            retry_feedback=retry_feedback,
        )
    )

    turn_cap = max_turns
    return SubtaskHandoffBundle(
        subtask=subtask,
        task_tree=task_tree,
        root_task=root_task,
        project_root=project_root,
        initial_messages=messages,
        startup_status=startup_status,
        session_memory=session_memory,
        ctx_config=ctx_config,
        ctx_runtime=ctx_runtime,
        retry_strategy=retry_strategy,
        policy=policy,
        whitelist_files=whitelist_files,
        diag_slices=diag_slices,
        diag_handoff=diag_handoff,
        edit_handoff=edit_handoff,
        edit_splice_max_attempts=settings.edit_splice_max_attempts,
        max_turns=max_turns,
        turn_cap=turn_cap,
        qg_limit=quality_gate_retry_limit or settings.subtask_quality_gate_retries,
        context_search_available=context_search_available,
    )


def _apply_context_pack_tool_policy(
    runtime_tools: frozenset[str],
    context_pack: Any | None,
) -> frozenset[str]:
    if context_pack is None:
        return runtime_tools
    policy = getattr(context_pack, "tool_policy", {}) or {}
    denied = {str(tool) for tool in policy.get("denied_tools", [])}
    allowed = {str(tool) for tool in policy.get("allowed_tools", [])}
    tools = runtime_tools - denied
    if allowed:
        tools = tools & allowed
    return frozenset(tools)


def resolve_turn_tools(
    bundle: SubtaskHandoffBundle,
    *,
    turns_used: int,
    tool_rounds: int,
    file_changes: list[str],
    active_runtime_tools: frozenset[str],
) -> frozenset[str]:
    """Per-turn tool surface (Harness turn policy)."""
    subtask = bundle.subtask
    is_diagnose = subtask.kind == SubTaskKind.DIAGNOSE

    turn_tools = active_runtime_tools
    if is_diagnose:
        locate_tools = active_runtime_tools & DIAGNOSE_LOCATE_TOOLS
        if locate_tools:
            if tool_rounds >= 1:
                return frozenset()
            return locate_tools
    if (
        subtask.kind == SubTaskKind.EDIT
        and bundle.diag_handoff
        and bundle.ctx_runtime
        and bundle.ctx_runtime.edit_read_fallback
        and not file_changes
    ):
        write_tools = active_runtime_tools & EDIT_TOOLS
        if tool_rounds == 0 and "read_files" in active_runtime_tools:
            return write_tools | frozenset({"read_files"})
        if write_tools:
            return write_tools
    if (
        subtask.kind == SubTaskKind.EDIT
        and not file_changes
        and turns_used >= bundle.turn_cap - EDIT_TURN_RESERVE + 1
    ):
        write_only = active_runtime_tools & EDIT_TOOLS
        if write_only:
            turn_tools = write_only
    return turn_tools


def turn_control_nudges(
    bundle: SubtaskHandoffBundle,
    *,
    turns_used: int,
    tool_rounds: int,
    file_changes: list[str],
    error_trace: list[str] | None = None,
) -> list[Message]:
    """Harness turn-control messages injected before each LLM call."""
    out: list[Message] = []
    subtask = bundle.subtask

    if (
        subtask.kind == SubTaskKind.EDIT
        and not file_changes
        and error_trace
    ):
        edit_hint = edit_failure_hint(error_trace)
        if edit_hint:
            out.append(harness_nudge(edit_hint))

    if (
        subtask.kind == SubTaskKind.EDIT
        and not file_changes
        and turns_used == bundle.turn_cap
    ):
        out.append(harness_nudge(
            "Final turn: call edit_file if you know the target, use write_file only "
            "with complete file content, or summarize blocker with evidence from "
            "the session summary."
        ))

    if (
        subtask.kind == SubTaskKind.EDIT
        and bundle.edit_handoff
        and bundle.edit_handoff.active
        and not file_changes
        and turns_used >= 2
    ):
        t = bundle.edit_handoff.targets[0]
        out.append(harness_nudge(
            f"Call replace_symbol(path={t.path!r}, symbol={t.symbol!r}, new_body=...) "
            "with the full revised block — see <edit_target> original."
        ))
    elif (
        subtask.kind == SubTaskKind.EDIT
        and bundle.diag_handoff
        and not (bundle.ctx_runtime and bundle.ctx_runtime.edit_read_fallback)
        and not file_changes
        and turns_used >= 2
    ):
        targets = (
            ", ".join(bundle.whitelist_files[:3])
            if bundle.whitelist_files
            else "context_files"
        )
        out.append(harness_nudge(
            f"Diagnose slices are preloaded in <file> blocks. "
            f"Call edit_file on {targets} with a unique multi-line old_string — "
            "do not grep/read again."
        ))
    return out


def _prefers_incremental_edit(root_task: str, subtask: SubTaskNode) -> bool:
    if subtask.kind != SubTaskKind.EDIT:
        return False
    text = " ".join(
        part
        for part in (root_task, subtask.description, subtask.acceptance_criteria or "")
        if part
    ).lower()
    return any(
        marker in text
        for marker in (
            "sql",
            "query",
            "view",
            "schema",
            "接口",
            "端点",
            "查询",
            "订单",
            "登机牌",
            "视图",
        )
    )


def _bootstrap_runtime_message(
    *,
    subtask: SubTaskNode,
    root_task: str,
    project_root: Path,
    retry_strategy: ExecutorRetryStrategy,
    paths_only_mode: bool,
    auto_scoped_edit: bool,
    explore_restricted: bool,
    diag_handoff: bool,
    splice_edit: bool = False,
    edit_handoff: EditHandoff | None = None,
    edit_read_fallback: bool = False,
    diag_slices: dict[str, tuple[int, int]],
    whitelist_files: list[str],
    files_in_prompt: frozenset[str],
    truncated_paths: frozenset[str],
    runtime_tools: frozenset[str],
    prior_exploration: str | None,
    retry_feedback: list[str] | None,
) -> Message:
    rules: list[str] = ["Execute only this subtask."]
    if retry_strategy.user_hint:
        rules.append(retry_strategy.user_hint)
    if paths_only_mode:
        rules.append("Files are paths-only; use context_search with scoped paths before editing.")
    elif auto_scoped_edit:
        rules.append("Scoped edit: context_search on context_files, then edit_file.")
    if splice_edit and edit_handoff and edit_handoff.targets:
        rules.append("Use replace_symbol with the full revised block.")
    if edit_read_fallback and diag_handoff:
        rules.append("Read fallback is exposed only for exact old_string recovery.")
    elif explore_restricted and diag_handoff:
        rules.append("Diagnose line slices are preloaded; edit from those slices.")
    elif explore_restricted and diag_slices:
        rules.append("Prior milestone coordinates are preloaded; raw search is disabled.")
    elif explore_restricted:
        rules.append("Retry with prior digest; raw search is disabled.")
    if diag_slices and subtask.kind == SubTaskKind.EDIT:
        rules.append("Use diagnose line slices as the starting evidence.")
    if whitelist_files and truncated_paths and subtask.kind == SubTaskKind.EDIT:
        rules.append("Some context_files are truncated; use context_search on those paths first.")
    if whitelist_files and files_in_prompt and subtask.kind == SubTaskKind.EDIT:
        rules.append("Edit only preloaded context_files; never create scratch files.")
    if subtask.kind == SubTaskKind.DIAGNOSE:
        rules.append("Diagnose uses Harness context_search handoff; use at most one precise context_search if exposed.")
    if subtask.kind == SubTaskKind.VERIFY and runtime_tools == frozenset({"shell_exec"}):
        rules.append(f"Run shell_exec from project root ({project_root}); do not set working_dir to /workspace.")
    if prior_exploration and prior_exploration.strip():
        rules.append("Prior exploration is in session_digest; do not repeat the same context_search.")
    payload = {
        "schema": "mitkii.executor_runtime.v1",
        "event": "start",
        "subtask_id": subtask.id,
        "mode": {
            "paths_only": paths_only_mode,
            "auto_scoped_edit": auto_scoped_edit,
            "explore_restricted": explore_restricted,
            "diag_handoff": diag_handoff,
            "splice_edit": splice_edit,
            "edit_read_fallback": edit_read_fallback,
        },
        "allowed_tools": sorted(runtime_tools),
        "context_files": list(whitelist_files),
        "preloaded_files": sorted(files_in_prompt),
        "truncated_paths": sorted(truncated_paths),
        "diag_slices": {
            path: {"start": span[0], "end": span[1]}
            for path, span in sorted(diag_slices.items())
        },
        "rules": rules,
        "retry": {
            "attempt": retry_strategy.attempt,
            "feedback": [str(e) for e in (retry_feedback or [])[-8:]],
        },
    }
    return user_message("EXECUTOR_RUNTIME_JSON\n" + json.dumps(payload, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Orchestrator ↔ Executor cross-subtask handoff (prepare input / commit output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtaskCommitResult:
    """Outcome of committing a finished subtask back to session state."""

    summary_stored: bool = False
    context_files_updated: bool = False


def collect_prior_summaries(
    task_tree: TaskTree,
    node: SubTaskNode,
    subtask_summaries: dict[str, str],
) -> dict[str, str]:
    """Summaries from completed depends_on ancestors for the next subtask."""
    from src.planner.prior_context import prior_summaries_for_node

    return prior_summaries_for_node(task_tree, node, subtask_summaries)


def commit_subtask_success(
    *,
    task_tree: TaskTree,
    node: SubTaskNode,
    exec_result: dict,
    project_root: Path,
    subtask_summaries: dict[str, str],
    subtask_attempts: dict[str, int],
    subtask_exploration_digests: dict[str, str],
) -> SubtaskCommitResult:
    """Persist subtask output and propagate diagnose paths to dependents."""
    from src.planner.context_policy import enrich_task_tree_context_files
    from src.planner.prior_context import propagate_diagnose_paths

    subtask_attempts.pop(node.id, None)
    subtask_exploration_digests.pop(node.id, None)

    final_msg = exec_result.get("final_message")
    digest = exec_result.get("exploration_digest")
    summary_stored = False
    context_updated = False
    summary = _success_summary_with_digest(final_msg, digest)
    if summary:
        subtask_summaries[node.id] = summary
        summary_stored = True

    if node.kind == SubTaskKind.DIAGNOSE and summary:
        propagate_diagnose_paths(task_tree, node.id, summary, project_root)
        enrich_task_tree_context_files(task_tree)
        context_updated = True

    return SubtaskCommitResult(
        summary_stored=summary_stored,
        context_files_updated=context_updated,
    )


def _success_summary_with_digest(final_msg: object, digest: object) -> str:
    parts: list[str] = []
    if isinstance(final_msg, str) and final_msg.strip():
        parts.append(final_msg.strip())
    if isinstance(digest, str) and digest.strip():
        digest_text = digest.strip()
        if digest_text not in parts:
            parts.append("Executor evidence digest:\n" + digest_text)
    return "\n\n".join(parts)


def commit_subtask_failure(
    *,
    node: SubTaskNode,
    exec_result: dict | None,
    subtask_attempts: dict[str, int],
    subtask_exploration_digests: dict[str, str],
) -> int:
    """Record retry attempt and preserve exploration digest for the next run."""
    attempt = subtask_attempts.get(node.id, 0) + 1
    subtask_attempts[node.id] = attempt
    if exec_result:
        digest = exec_result.get("exploration_digest")
        if isinstance(digest, str) and digest.strip():
            subtask_exploration_digests[node.id] = digest.strip()
    return attempt
