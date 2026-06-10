from __future__ import annotations

from typing import TYPE_CHECKING

from src.planner.kinds import SubTaskKind

if TYPE_CHECKING:
    from src.planner.task_tree import SubTaskNode

# Full executor tool catalog (Harness may expose a subset per subtask).
EXECUTOR_TOOLS: frozenset[str] = frozenset({
    "context_search",
    "read_file",
    "read_files",
    "grep_search",
    "map_search",
    "glob_files",
    "list_dir",
    "write_file",
    "edit_file",
    "delete_file",
    "shell_exec",
    "git_status",
})

WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "delete_file"})

KIND_DEFAULT_TOOLS: dict[SubTaskKind, frozenset[str]] = {
    SubTaskKind.DIAGNOSE: frozenset({
        "context_search",
        "git_status",
    }),
    SubTaskKind.DESIGN: frozenset({
        "context_search",
    }),
    SubTaskKind.EDIT: frozenset({
        "context_search",
        "write_file",
        "edit_file",
        "delete_file",
    }),
    SubTaskKind.VERIFY: frozenset({
        "context_search",
        "shell_exec",
    }),
    SubTaskKind.SHELL: frozenset({
        "shell_exec",
        "context_search",
    }),
}

KIND_FORBIDDEN_TOOLS: dict[SubTaskKind, frozenset[str]] = {
    SubTaskKind.DIAGNOSE: WRITE_TOOLS | frozenset({"shell_exec"}),
    SubTaskKind.DESIGN: WRITE_TOOLS | frozenset({"shell_exec", "git_status"}),
    SubTaskKind.EDIT: frozenset({"shell_exec"}),
    SubTaskKind.VERIFY: WRITE_TOOLS,
    SubTaskKind.SHELL: WRITE_TOOLS,
}

KIND_REQUIRED_TOOLS: dict[SubTaskKind, frozenset[str]] = {
    SubTaskKind.EDIT: frozenset({"edit_file", "write_file"}),  # at least one
    SubTaskKind.SHELL: frozenset({"shell_exec"}),
}


def default_allowed_tools(kind: SubTaskKind) -> list[str]:
    return sorted(KIND_DEFAULT_TOOLS[kind])


def normalize_allowed_tools(
    kind: SubTaskKind,
    explicit: list[str] | None,
) -> list[str]:
    """Resolve allowed_tools: use explicit subset or kind defaults."""
    catalog = KIND_DEFAULT_TOOLS[kind]
    if not explicit:
        return sorted(catalog)
    cleaned = [t for t in explicit if t in EXECUTOR_TOOLS and t in catalog]
    if cleaned:
        return sorted(set(cleaned))
    return sorted(catalog)


def effective_allowed_tools(node: SubTaskNode) -> frozenset[str]:
    return frozenset(normalize_allowed_tools(node.kind, node.allowed_tools or None))


def validate_node_tools(node: SubTaskNode) -> tuple[list[str], list[str]]:
    """Return (blocks, warns) for PlanGate."""
    blocks: list[str] = []
    warns: list[str] = []
    catalog = KIND_DEFAULT_TOOLS[node.kind]
    explicit = list(node.allowed_tools or [])
    declared = explicit if explicit else sorted(catalog)
    allowed = frozenset(declared)

    if not allowed:
        blocks.append(f"Subtask [{node.id}] has empty allowed_tools.")
        return blocks, warns

    unknown = [t for t in explicit if t not in EXECUTOR_TOOLS]
    if unknown:
        blocks.append(
            f"Subtask [{node.id}] unknown tools: {', '.join(unknown)}."
        )

    out_of_kind = sorted(t for t in declared if t not in catalog)
    if out_of_kind:
        blocks.append(
            f"Subtask [{node.id}] tools {out_of_kind} not permitted for kind={node.kind.value}."
        )

    forbidden = KIND_FORBIDDEN_TOOLS[node.kind] & allowed
    if forbidden:
        blocks.append(
            f"Subtask [{node.id}] forbids {sorted(forbidden)} for kind={node.kind.value}."
        )

    required = KIND_REQUIRED_TOOLS.get(node.kind)
    if required and not (required & allowed):
        if node.kind == SubTaskKind.EDIT:
            blocks.append(
                f"Subtask [{node.id}] kind=edit must allow edit_file or write_file."
            )
        else:
            blocks.append(
                f"Subtask [{node.id}] kind={node.kind.value} must include "
                f"{sorted(required)}."
            )

    if not node.acceptance_criteria.strip():
        warns.append(f"Subtask [{node.id}] missing acceptance_criteria.")

    if not explicit:
        warns.append(
            f"Subtask [{node.id}] omitted allowed_tools — Harness applied kind defaults."
        )

    return blocks, warns


def format_tool_denial(tool_name: str, node: SubTaskNode) -> str:
    allowed = ", ".join(sorted(effective_allowed_tools(node)))
    return (
        f"Tool '{tool_name}' is not allowed for subtask [{node.id}] "
        f"(kind={node.kind.value}). Allowed: {allowed}."
    )
