from __future__ import annotations

from pathlib import Path

from src.harness.gates.types import TruncationPolicy

_TRUNCATION_MARKERS = (
    "[... truncated middle ...]",
    "\n[truncated]",
    "[truncated —",
    "[missing file:",
)

# Per-process cache: (project_root, files tuple, policy key) → loaded contents
_FILE_LOAD_CACHE: dict[tuple[str, tuple[str, ...], str], list[tuple[str, str]]] = {}


def _policy_cache_key(policy: TruncationPolicy) -> str:
    slices = tuple(sorted((policy.line_slices or {}).items()))
    return (
        f"{policy.tier}:{policy.max_chars_per_file}:{policy.head_lines}:{policy.tail_lines}:{slices}"
    )


def norm_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def preloaded_paths(
    project_root: Path,
    context_files: list[str],
    *,
    policy: TruncationPolicy,
    loaded: list[tuple[str, str]] | None = None,
) -> frozenset[str]:
    if not context_files or policy.tier == "red":
        return frozenset()
    contents = loaded if loaded is not None else load_context_file_contents(
        project_root, context_files, policy=policy
    )
    return frozenset(
        norm_rel_path(rel)
        for rel, content in contents
        if content
        and not content.startswith(("[missing file:", "[read error:", "[blocked:"))
    )


def detect_truncated_preloads(
    project_root: Path,
    context_files: list[str],
    *,
    policy: TruncationPolicy,
    loaded: list[tuple[str, str]] | None = None,
) -> frozenset[str]:
    """Paths whose preloaded content is incomplete (head/tail or char cap)."""
    if not context_files or policy.tier == "red":
        return frozenset()
    if policy.tier == "yellow":
        return frozenset(norm_rel_path(p) for p in context_files)
    truncated: set[str] = set()
    contents = loaded if loaded is not None else load_context_file_contents(
        project_root, context_files, policy=policy
    )
    for rel, content in contents:
        norm = norm_rel_path(rel)
        if content.startswith("["):
            continue
        if any(marker in content for marker in _TRUNCATION_MARKERS):
            truncated.add(norm)
    return frozenset(truncated)


def load_context_file_contents(
    project_root: Path,
    context_files: list[str],
    *,
    policy: TruncationPolicy | None = None,
) -> list[tuple[str, str]]:
    policy = policy or TruncationPolicy.green()
    if policy.tier == "red" or not context_files:
        return []

    root_key = str(project_root.resolve())
    files_key = tuple(norm_rel_path(f) for f in context_files)
    cache_key = (root_key, files_key, _policy_cache_key(policy))
    cached = _FILE_LOAD_CACHE.get(cache_key)
    if cached is not None:
        return cached

    loaded: list[tuple[str, str]] = []
    for rel in context_files:
        path = (project_root / rel).resolve()
        try:
            path.relative_to(project_root.resolve())
        except ValueError:
            loaded.append((rel, "[blocked: path outside project root]"))
            continue
        if not path.is_file():
            loaded.append((rel, f"[missing file: {rel}]"))
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            loaded.append((rel, f"[read error: {exc}]"))
            continue

        norm = norm_rel_path(rel)
        line_slice = (policy.line_slices or {}).get(norm)
        lines = text.splitlines(keepends=True)
        if line_slice is not None:
            start, end = line_slice
            start_idx = max(0, start - 1)
            end_idx = min(len(lines), end)
            sliced = lines[start_idx:end_idx]
            text = (
                f"[repo_map slice L{start}-{end} of {len(lines)} lines]\n"
                + "".join(sliced)
            )
            loaded.append((rel, text))
            continue

        if policy.head_lines is not None and policy.tail_lines is not None:
            head = lines[: policy.head_lines]
            tail = lines[-policy.tail_lines :] if len(lines) > policy.head_lines else []
            if head and tail and len(lines) > policy.head_lines + policy.tail_lines:
                text = "".join(head) + "\n[... truncated middle ...]\n" + "".join(tail)
            else:
                text = "".join(head or tail)
        elif len(text) > policy.max_chars_per_file:
            text = (
                text[: policy.max_chars_per_file]
                + "\n[truncated — use grep_search on this path for full content]"
            )

        loaded.append((rel, text))
    _FILE_LOAD_CACHE[cache_key] = loaded
    return loaded
