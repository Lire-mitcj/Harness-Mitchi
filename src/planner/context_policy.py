from __future__ import annotations

from src.planner.task_tree import SubTaskNode, TaskTree


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _merge_paths(base: list[str], extra: list[str]) -> list[str]:
    seen = {_norm_path(p) for p in base}
    merged = list(base)
    for path in extra:
        norm = _norm_path(path)
        if norm and norm not in seen:
            seen.add(norm)
            merged.append(path)
    return merged


def dependency_context_files(task_tree: TaskTree, node: SubTaskNode) -> list[str]:
    """Collect context_files from transitive depends_on ancestors."""
    collected: list[str] = []
    visited: set[str] = set()

    def walk(dep_id: str) -> None:
        if dep_id in visited:
            return
        visited.add(dep_id)
        dep = task_tree.get(dep_id)
        if dep is None:
            return
        collected.extend(dep.context_files)
        for parent_id in dep.depends_on:
            walk(parent_id)

    for dep_id in node.depends_on:
        walk(dep_id)
    return collected


def effective_context_files(task_tree: TaskTree, node: SubTaskNode) -> list[str]:
    """Union of this node's context_files and all dependency ancestors' files."""
    return _merge_paths(list(node.context_files), dependency_context_files(task_tree, node))


def enrich_task_tree_context_files(task_tree: TaskTree) -> None:
    """Merge dependency context_files into each node (in-place, before PlanGate)."""
    for node in task_tree.nodes:
        merged = effective_context_files(task_tree, node)
        node.context_files = merged
