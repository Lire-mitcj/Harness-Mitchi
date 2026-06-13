from __future__ import annotations

import logging
from pathlib import Path
from src.planner.task_tree import SubTaskNode, TaskTree

log = logging.getLogger(__name__)


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


def resolve_project_paths(project_root: Path, paths: list[str]) -> list[str]:
    resolved = []
    all_files = None

    for path in paths:
        path_str = path.strip().replace("\\", "/").lstrip("./")
        if not path_str:
            continue
        p = (project_root / path_str).resolve()
        try:
            if p.is_file():
                resolved.append(str(p.relative_to(project_root.resolve())).replace("\\", "/"))
                continue
        except OSError:
            pass

        # If not found directly, search all files in the project recursively
        if all_files is None:
            all_files = []
            for filepath in project_root.rglob("*"):
                if not any(part in {".git", ".venv", "venv", "node_modules", "__pycache__"} for part in filepath.parts):
                    try:
                        if filepath.is_file():
                            all_files.append(filepath.relative_to(project_root))
                    except Exception:
                        pass

        # Match by basename or suffix
        basename = path_str.split("/")[-1]
        matches = []
        for rel_path in all_files:
            rel_str = str(rel_path).replace("\\", "/")
            if rel_str.endswith("/" + path_str) or rel_str == path_str or rel_path.name == basename:
                matches.append(rel_str)

        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            log.warning("ambiguous_path: %s matches multiple files %s", path, matches)
            resolved.append(path_str)
        else:
            resolved.append(path_str)

    return list(dict.fromkeys(resolved))


def enrich_task_tree_context_files(task_tree: TaskTree, project_root: Path | None = None) -> None:
    """Merge dependency context_files into each node (in-place, before PlanGate)."""
    if project_root is None:
        project_root = Path.cwd()

    for node in task_tree.nodes:
        if node.context_files:
            node.context_files = resolve_project_paths(project_root, node.context_files)

    for node in task_tree.nodes:
        merged = effective_context_files(task_tree, node)
        node.context_files = merged
