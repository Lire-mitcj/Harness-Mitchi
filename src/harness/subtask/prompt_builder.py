from __future__ import annotations

import logging
from pathlib import Path

from src.agent.types import Message, system_message, user_message
from src.harness.gates.types import TruncationPolicy
from src.harness.probe.token_budget import TokenBudget
from src.harness.subtask.preload import load_context_file_contents
from src.planner.context_policy import effective_context_files
from src.planner.kinds import SubTaskKind
from src.planner.prior_context import format_diagnose_handoff_block, format_prior_summaries_block
from src.planner.task_tree import SubTaskNode, TaskTree

log = logging.getLogger(__name__)

_EXECUTOR_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "executor_prompt.md"
)

_EXECUTOR_FALLBACK = """You are MitKII Executor — complete ONE subtask using only allowed_tools."""

_EXECUTOR_SYSTEM_PROMPT: str | None = None
_TOKEN_COUNTER = TokenBudget()


def load_executor_system_prompt() -> str:
    global _EXECUTOR_SYSTEM_PROMPT
    if _EXECUTOR_SYSTEM_PROMPT is not None:
        return _EXECUTOR_SYSTEM_PROMPT
    if _EXECUTOR_PROMPT_PATH.is_file():
        _EXECUTOR_SYSTEM_PROMPT = _EXECUTOR_PROMPT_PATH.read_text(encoding="utf-8").strip()
    else:
        log.warning("executor_prompt.md missing, using embedded fallback")
        _EXECUTOR_SYSTEM_PROMPT = _EXECUTOR_FALLBACK
    return _EXECUTOR_SYSTEM_PROMPT


_EDIT_FILE_WORKFLOW = (
    "edit_file workflow (exact string replace):\n"
    "1. Pick the lines to change inside the <file> block above "
    "(skip any [repo_map slice…] header line).\n"
    "2. old_string = copy those lines verbatim (≥3 lines; exact spaces/quotes).\n"
    "3. new_string = same block with your fix (must differ from old_string).\n"
    "4. path = relative path (e.g. app.py).\n"
    "Never use only a function name or one-line signature — include the code/SQL you change."
)


def _build_subtask_scope_block(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    project_root: Path,
    policy: TruncationPolicy,
    preload_mode: str,
    runtime_tools: frozenset[str] | None,
) -> str:
    parts = [
        f"Your assigned subtask [{subtask.id}] (kind={subtask.kind.value}):",
        subtask.description,
        "",
    ]
    if subtask.acceptance_criteria:
        parts.append(f"Done when: {subtask.acceptance_criteria}")
        parts.append("")
    tools = ", ".join(
        sorted(runtime_tools if runtime_tools is not None else subtask.effective_allowed_tools())
    )
    l1_label = "enabled" if subtask.effective_needs_l1() else "skipped for this kind"
    context_files = effective_context_files(task_tree, subtask)
    parts.extend([
        f"Allowed tools (ONLY these): {tools}",
        f"Quality gate L1: {l1_label}.",
        "",
        "Whitelisted context files:",
    ])

    if context_files and preload_mode == "paths_only":
        parts.append(
            "Paths only (NOT fully preloaded). Prefer map_search(query) or "
            "read_files(paths=[...]) once for all "
            "paths below, or grep_search to locate code — then edit_file. "
            "Use write_file only for new files or complete-file rewrites. "
            "Never create /tmp or files outside this list:"
        )
        for rel in context_files:
            parts.append(f"  - {rel}")
    elif context_files and policy.tier != "red":
        parts.append(
            "Preloaded below (FULL file content — do NOT call read_file on these paths again):"
        )
        if subtask.kind == SubTaskKind.DIAGNOSE:
            parts.append(
                "Diagnose mode: use preloaded content + map_search or grep_search, "
                "then reply with findings (no more reads)."
            )
        elif subtask.kind == SubTaskKind.EDIT:
            parts.append(
                "Edit mode: whitelisted files are preloaded. "
                "Use edit_file on those paths for partial changes (no /tmp or scratch files). "
                "Use write_file only for new files or complete-file rewrites. "
                "If a file shows [truncated], use map_search, read_files, "
                "or grep_search on those paths first."
            )
            parts.append(_EDIT_FILE_WORKFLOW)
        for rel, content in load_context_file_contents(
            project_root, context_files, policy=policy
        ):
            parts.append(f'\n<file path="{rel}">\n{content}\n</file>')
    else:
        parts.append(
            "  (none preloaded — map_search/grep_search/read_file/list_dir work "
            "on any path under project root)"
        )
    return "\n".join(parts)


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
) -> list[Message]:
    """Layered Executor system messages (L0 prompt | L2 task tree | L3 subtask + preload)."""
    policy = policy or TruncationPolicy.green()

    task_parts = [
        f"Global task: {root_task}",
        "",
        "TaskTree outline:",
        task_tree.to_outline(),
    ]
    prior_block = format_prior_summaries_block(prior_summaries or {})
    if prior_block:
        task_parts.extend(["", prior_block])
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
            task_parts.extend(["", handoff])

    scope = _build_subtask_scope_block(
        subtask=subtask,
        task_tree=task_tree,
        project_root=project_root,
        policy=policy,
        preload_mode=preload_mode,
        runtime_tools=runtime_tools,
    )

    return [
        system_message(load_executor_system_prompt(), cache_breakpoint=True),
        system_message("\n".join(task_parts), cache_breakpoint=True),
        system_message(scope, cache_breakpoint=True),
    ]


def build_executor_scope_message(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    project_root: Path,
    policy: TruncationPolicy | None = None,
    preload_mode: str = "full",
    runtime_tools: frozenset[str] | None = None,
) -> Message:
    """L3 only — subtask scope plus optional file preload."""
    policy = policy or TruncationPolicy.green()
    scope = _build_subtask_scope_block(
        subtask=subtask,
        task_tree=task_tree,
        project_root=project_root,
        policy=policy,
        preload_mode=preload_mode,
        runtime_tools=runtime_tools,
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
) -> list[Message]:
    from src.executor.retry_strategy import classify_failure_pattern

    messages = build_executor_messages(
        root_task=root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        policy=TruncationPolicy.green(),
        prior_summaries=prior_summaries,
        preload_mode="paths_only",
        runtime_tools=runtime_tools,
    )
    lines = [
        f"Subtask [{subtask.id}] — retry after tool error(s).",
        f"Retry failure pattern: {classify_failure_pattern(error_trace)}.",
        "Change the approach from the failed attempt. Do NOT repeat the same "
        "tool arguments or produce the same edit/write result.",
        "Continue from the session summary if present; avoid re-reading the same line ranges.",
        "Use edit_file for partial changes. Use write_file only for new files "
        "or complete-file rewrites.",
        "Do NOT write /tmp or scratch files.",
    ]
    if context_files:
        lines.append("Allowed paths: " + ", ".join(context_files))
    if exploration_digest and exploration_digest.strip():
        from src.executor.exploration_digest import format_digest_system_block

        messages.append(format_digest_system_block(exploration_digest))
    if error_trace:
        lines.append("Recent errors:")
        lines.extend(f"- {e}" for e in error_trace[-6:])
    messages.append(user_message("\n".join(lines)))
    return messages


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
