from __future__ import annotations

from pathlib import Path

MUTATION_TOOLS = frozenset({"write_file", "edit_file", "delete_file", "replace_symbol"})


def should_apply_context_whitelist(tool_name: str) -> bool:
    """Only file mutations are restricted to subtask context_files."""
    return tool_name in MUTATION_TOOLS


def normalize_rel_path(project_root: Path, path: object) -> str | None:
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


def _canonical_rel(project_root: Path, raw: object) -> str | None:
    rel = normalize_rel_path(project_root, raw)
    if rel is None:
        return None
    return rel.replace("\\", "/").lstrip("./")


def is_path_in_project(project_root: Path, raw_path: object) -> bool:
    """True when raw_path resolves under project_root."""
    if not isinstance(raw_path, str) or not raw_path.strip():
        return False
    try:
        p = Path(raw_path)
        if not p.is_absolute():
            p = (project_root / p).resolve()
        else:
            p = p.resolve()
        p.relative_to(project_root.resolve())
        return True
    except (ValueError, OSError):
        return False


def is_path_allowed(
    project_root: Path,
    raw_path: object,
    whitelist: list[str],
) -> bool:
    """Return True when whitelist is empty (unrestricted) or path is whitelisted."""
    if not whitelist:
        return True
    rel = _canonical_rel(project_root, raw_path)
    if rel is None:
        return False
    allowed = {
        _canonical_rel(project_root, p) or str(p).replace("\\", "/").lstrip("./")
        for p in whitelist
    }
    if rel in allowed:
        return True
    return any(rel.startswith(f"{prefix}/") for prefix in allowed if prefix)


def collect_paths_from_tool(tool_name: str, arguments: dict[str, object]) -> list[str]:
    paths: list[str] = []
    if tool_name == "read_file":
        p = arguments.get("path")
        if isinstance(p, str):
            paths.append(p)
    elif tool_name == "read_files":
        raw = arguments.get("paths")
        if isinstance(raw, list):
            paths.extend(p for p in raw if isinstance(p, str))
    elif tool_name in {"write_file", "edit_file", "delete_file", "replace_symbol"}:
        p = arguments.get("path")
        if isinstance(p, str):
            paths.append(p)
    elif tool_name == "grep_search":
        p = arguments.get("path")
        if isinstance(p, str) and p not in {".", "./"}:
            paths.append(p)
    elif tool_name in {"list_dir", "glob_files"}:
        p = arguments.get("path")
        if isinstance(p, str) and p not in {".", "./"}:
            paths.append(p)
    return paths


def format_whitelist_denial(
    tool_name: str,
    blocked: list[str],
    whitelist: list[str],
    *,
    project_root: Path | None = None,
) -> str:
    allowed = ", ".join(whitelist) or "(none)"
    if project_root is not None:
        for raw in blocked:
            if not is_path_in_project(project_root, raw):
                return (
                    f"{tool_name} blocked: path '{raw}' is outside the project. "
                    "Scratch/note files (/tmp/*, etc.) are forbidden. "
                    f"Use grep_search on whitelisted files ({allowed}) to locate code, "
                    "then edit_file on one of those paths."
                )
    blocked_str = ", ".join(blocked)
    return (
        f"{tool_name} blocked: path(s) [{blocked_str}] not in context_files. "
        f"Allowed edit paths: {allowed}. "
        "Use grep_search to find the target code, then edit_file — "
        "do not create files outside this list."
    )
