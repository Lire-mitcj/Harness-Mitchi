from __future__ import annotations

import re
from collections.abc import Iterable

from src.config.settings import MitKIISettings
from src.indexer.repo_map import RankedSymbol, RepoMap
from src.planner.task_tree import SubTaskNode, TaskTree

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]{2,}")


def resolve_repo_map_line_slices(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    context_files: list[str],
    repo_map: RepoMap,
    settings: MitKIISettings,
    target_files: Iterable[str] | None = None,
) -> dict[str, tuple[int, int]]:
    """Pick line ranges from repo_map symbols for preflight preload."""
    files = list(dict.fromkeys(target_files or context_files))
    if not files:
        return {}

    query_tokens = _query_tokens(subtask, task_tree)
    padding = settings.preflight_slice_padding
    max_symbols = settings.preflight_slice_max_symbols
    slices: dict[str, tuple[int, int]] = {}

    for rel in files:
        norm = rel.replace("\\", "/").lstrip("./")
        candidates = list(repo_map.symbols_by_file.get(norm, []))
        if not candidates:
            candidates = [s for s in repo_map.all_symbols if s.file_path == norm]

        ranked = _rank_symbols(candidates, query_tokens)
        if not ranked:
            continue

        ranges: list[tuple[int, int]] = []
        for sym in ranked[:max_symbols]:
            start = max(1, sym.start_line - padding)
            end = sym.end_line + padding
            ranges.append((start, end))
        merged = _merge_ranges(ranges)
        if merged:
            slices[norm] = merged

    return slices


def _query_tokens(subtask: SubTaskNode, task_tree: TaskTree) -> set[str]:
    text = " ".join(
        [
            task_tree.root_task,
            subtask.description,
            subtask.acceptance_criteria,
        ]
    ).lower()
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "into", "using",
        "file", "files", "read", "edit", "fix", "use", "run", "test",
    }
    return {t for t in _TOKEN_RE.findall(text) if t not in stop}


def _rank_symbols(
    symbols: list[RankedSymbol],
    query_tokens: set[str],
) -> list[RankedSymbol]:
    if not symbols:
        return []

    def score(sym: RankedSymbol) -> tuple[int, float]:
        name = sym.name.lower()
        sig = sym.signature.lower()
        hits = sum(1 for t in query_tokens if t in name or t in sig)
        return (hits, sym.score)

    return sorted(symbols, key=score, reverse=True)


def _merge_ranges(ranges: list[tuple[int, int]]) -> tuple[int, int] | None:
    if not ranges:
        return None
    ranges.sort()
    start, end = ranges[0]
    for s, e in ranges[1:]:
        if s <= end + 20:
            end = max(end, e)
        else:
            break
    return (start, end)
