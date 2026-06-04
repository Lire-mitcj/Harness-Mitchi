from __future__ import annotations

from src.config.settings import MitKIISettings
from src.planner.kinds import SubTaskKind
from src.planner.tool_policy import WRITE_TOOLS

if False:  # TYPE_CHECKING without import cycle
    from src.planner.task_tree import SubTaskNode

EXPLORE_TOOLS = frozenset({
    "context_search",
    "read_file",
    "read_files",
    "grep_search",
    "map_search",
    "glob_files",
    "list_dir",
    "git_status",
})

EDIT_SCOPED_EXPLORE = frozenset({
    "context_search",
})

# Harness always grants high-level context lookup on diagnose/edit.
HARNESS_CONTEXT_SEARCH = frozenset({"context_search"})


def effective_max_turns(kind: SubTaskKind, settings: MitKIISettings) -> int:
    """Per-kind Executor turn budget (one LLM call = one turn)."""
    if kind == SubTaskKind.DIAGNOSE:
        return settings.executor_max_turns_diagnose
    if kind == SubTaskKind.EDIT:
        return settings.executor_max_turns_edit
    if kind == SubTaskKind.VERIFY:
        return settings.executor_max_turns_verify
    if kind == SubTaskKind.SHELL:
        return settings.executor_max_turns_shell
    return settings.orchestrator_executor_max_turns


EDIT_READ_TOOLS = frozenset({"read_file", "read_files"})


REPLACE_SYMBOL_TOOL = "replace_symbol"
SPLICE_EDIT_TOOLS = frozenset({REPLACE_SYMBOL_TOOL, "write_file"})


def resolve_executor_tools(
    subtask: "SubTaskNode",
    *,
    preloaded_paths: frozenset[str],
    truncated_paths: frozenset[str] | None = None,
    explore_restricted: bool = False,
    edit_read_fallback: bool = False,
    splice_edit: bool = False,
) -> frozenset[str]:
    """Runtime tool surface for the Executor LLM.

    Planner ``allowed_tools`` defines write intent; Harness may grant scoped
    exploration (context_search) when full preload is absent or truncated.
    Raw read tools are only granted for explicit edit-read fallback.
    """
    from src.planner.task_tree import SubTaskNode  # noqa: F401

    allowed = frozenset(subtask.effective_allowed_tools())
    truncated = truncated_paths or frozenset()

    if subtask.kind == SubTaskKind.EDIT:
        if splice_edit:
            return SPLICE_EDIT_TOOLS
        write = allowed & WRITE_TOOLS
        tools = write if write else frozenset({"edit_file"})
        if edit_read_fallback:
            return tools | EDIT_READ_TOOLS
        scoped = not preloaded_paths or bool(truncated)
        if scoped:
            if explore_restricted:
                return tools
            return tools | EDIT_SCOPED_EXPLORE
        if explore_restricted:
            return tools
        return tools | HARNESS_CONTEXT_SEARCH

    if explore_restricted and subtask.kind == SubTaskKind.DIAGNOSE:
        return frozenset()

    if explore_restricted and subtask.kind != SubTaskKind.DIAGNOSE:
        return allowed - EXPLORE_TOOLS

    if not preloaded_paths:
        if subtask.kind == SubTaskKind.DIAGNOSE:
            return allowed | HARNESS_CONTEXT_SEARCH
        return allowed

    if subtask.kind == SubTaskKind.DIAGNOSE:
        return (allowed - frozenset({"read_file", "read_files"})) | HARNESS_CONTEXT_SEARCH

    if subtask.kind == SubTaskKind.VERIFY:
        return allowed - frozenset({
            "context_search",
            "read_file",
            "read_files",
            "grep_search",
            "map_search",
        })

    return allowed


def redundant_read_denial(
    rel_path: str,
    *,
    kind: SubTaskKind = SubTaskKind.DIAGNOSE,
    paths_only: bool = False,
) -> str:
    if kind == SubTaskKind.EDIT:
        if paths_only:
            return (
                f"read_file blocked: '{rel_path}' — use context_search(query, need) "
                "to locate code, then edit_file."
            )
        return (
            f"read_file blocked: '{rel_path}' is already preloaded in your system prompt. "
            "Use context_search if truncated, then edit_file to fix the code. "
            "Use write_file only with complete file content."
        )
    return (
        f"File '{rel_path}' is already preloaded in your system context "
        f'(<file path="{rel_path}">). Do not call read_file again — '
        "use the preloaded content and write your findings summary."
    )


def explore_tool_denial(tool_name: str) -> str:
    return (
        f"Tool '{tool_name}' is disabled for this subtask turn — "
        "check Allowed tools in your system prompt. "
        "In scoped edit mode use context_search(query, need), then edit_file."
    )


def enable_edit_read_fallback(
    *,
    subtask: "SubTaskNode",
    runtime: object,
) -> bool:
    """After edit_file match failure, grant read_file so the model can copy exact text."""
    if getattr(runtime, "edit_read_fallback", False):
        return False
    runtime.edit_read_fallback = True  # type: ignore[attr-defined]
    runtime.explore_restricted = False  # type: ignore[attr-defined]
    runtime.active_runtime_tools = resolve_executor_tools(  # type: ignore[attr-defined]
        subtask,
        preloaded_paths=runtime.preloaded_paths,  # type: ignore[attr-defined]
        truncated_paths=runtime.truncated_paths,  # type: ignore[attr-defined]
        explore_restricted=False,
        edit_read_fallback=True,
    )
    return True


def explore_first_turn_tools(runtime_tools: frozenset[str]) -> frozenset[str]:
    """Turn-1 exploration surface: prefer skill-owned context lookup."""
    batch = runtime_tools & frozenset({"context_search"})
    if batch:
        return batch
    batch = runtime_tools & frozenset({"grep_search", "map_search", "read_files"})
    if batch:
        return batch
    return runtime_tools & frozenset({"grep_search", "map_search", "read_file"})
