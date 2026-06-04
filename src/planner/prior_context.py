from __future__ import annotations

import re
from pathlib import Path

from src.planner.context_policy import _merge_paths, _norm_path
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree

_PATH_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_./-])"
    r"((?:[\w.-]+/)*[\w.-]+\.(?:py|sql|md|html?|tsx?|jsx?|yml|yaml|json|toml|ini|cfg))"
    r"(?![A-Za-z0-9_.-])",
    re.IGNORECASE,
)

_LINE_REF_PATTERN = re.compile(
    r"(?P<path>[\w./-]+\.(?:py|sql))"
    r"(?:"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?"
    r"|\s+\|\s*(?P<line>\d+)"
    r"|\s+\(line\s+(?P<line_paren>\d+)\)"
    r"|\s+(?:at\s+)?(?P<line2>\d+)(?:-(?P<end2>\d+))?"
    r")",
    re.IGNORECASE,
)

_MAP_HIT_ROW = re.compile(
    r"\|\s*(?P<file>[\w./-]+\.\w+)\s*\|\s*(?P<line>\d+)\s*\|\s*(?P<symbol>[\w_]+)\s*\|",
    re.IGNORECASE,
)

_SYMBOL_AT_LINE = re.compile(
    r"(?P<symbol>[\w_]+)\s+(?:at|@)\s+(?P<path>[\w./-]+\.\w+):(?P<line>\d+)",
    re.IGNORECASE,
)

_EDIT_POSITIVE_HINTS = (
    "endpoint",
    "handler",
    "route",
    "query",
    "sql",
    "view",
    "schema",
    "接口",
    "端点",
    "处理函数",
    "查询",
    "订单",
    "登机牌",
    "视图",
    "目标",
)

_EVIDENCE_ONLY_HINTS = (
    "frontend",
    "client",
    "ui",
    "pdf",
    "test",
    "fixture",
    "前端",
    "调用",
    "生成",
    "证据",
)


def prior_summaries_for_node(
    task_tree: TaskTree,
    node: SubTaskNode,
    summaries: dict[str, str],
) -> dict[str, str]:
    """Summaries from transitive depends_on ancestors (execution order)."""
    ordered: dict[str, str] = {}
    visited: set[str] = set()

    def walk(dep_id: str) -> None:
        if dep_id in visited:
            return
        visited.add(dep_id)
        dep = task_tree.get(dep_id)
        if dep is None:
            return
        for parent_id in dep.depends_on:
            walk(parent_id)
        text = summaries.get(dep_id)
        if text:
            ordered[dep_id] = text

    for dep_id in node.depends_on:
        walk(dep_id)
    return ordered


def format_prior_summaries_block(summaries: dict[str, str]) -> str:
    if not summaries:
        return ""
    lines = [
        "Prior subtask results (from completed steps — use this evidence; "
        "do not repeat the same searches):",
    ]
    for sid, text in summaries.items():
        body = text.strip()
        if len(body) > 4000:
            body = body[:4000] + "\n...[truncated]"
        lines.append(f"\n### [{sid}]\n{body}")
    return "\n".join(lines)


def extract_paths_from_text(text: str, project_root: Path, *, limit: int = 8) -> list[str]:
    """Pull plausible project-relative file paths from a diagnose summary."""
    root = project_root.resolve()
    found: list[str] = []
    seen: set[str] = set()
    for match in _PATH_PATTERN.finditer(text):
        rel = _norm_path(match.group(1))
        if not rel or rel in seen:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        seen.add(rel)
        found.append(rel)
        if len(found) >= limit:
            break
    return found


def propagate_diagnose_paths(
    task_tree: TaskTree,
    diagnose_id: str,
    summary: str,
    project_root: Path,
) -> None:
    """After diagnose completes, attach discovered paths to dependent subtasks."""
    for node in task_tree.nodes:
        if diagnose_id not in node.depends_on:
            continue
        if node.kind == SubTaskKind.DIAGNOSE:
            continue
        intent = _node_intent_text(node)
        paths = extract_paths_from_text(summary, project_root)
        for rel in extract_line_refs_from_text(summary, project_root):
            if rel not in paths:
                paths.append(rel)
        paths = rank_edit_relevant_paths(summary, paths, intent_text=intent)
        if not paths:
            continue
        node.context_files = _merge_paths(list(node.context_files), paths)


def extract_line_refs_from_text(
    text: str,
    project_root: Path,
    *,
    padding: int = 15,
    limit: int = 5,
) -> dict[str, tuple[int, int]]:
    """Structured path:line refs from diagnose summaries or map_search text."""
    root = project_root.resolve()
    slices: dict[str, tuple[int, int]] = {}
    for match in _LINE_REF_PATTERN.finditer(text):
        rel = _norm_path(match.group("path"))
        if not rel:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        start_s = (
            match.group("start")
            or match.group("line")
            or match.group("line2")
            or match.group("line_paren")
        )
        if not start_s:
            continue
        start = int(start_s)
        end_s = match.group("end") or match.group("end2")
        end = int(end_s) if end_s else start + padding
        start = max(1, start - padding)
        end = end + padding
        prev = slices.get(rel)
        if prev is None or (end - start) > (prev[1] - prev[0]):
            slices[rel] = (start, end)
        if len(slices) >= limit:
            break
    return slices


def extract_line_refs_from_summaries(
    summaries: dict[str, str],
    project_root: Path,
    *,
    padding: int = 15,
) -> dict[str, tuple[int, int]]:
    merged: dict[str, tuple[int, int]] = {}
    for text in summaries.values():
        for rel, span in extract_line_refs_from_text(
            text, project_root, padding=padding
        ).items():
            prev = merged.get(rel)
            if prev is None:
                merged[rel] = span
            else:
                merged[rel] = (min(prev[0], span[0]), max(prev[1], span[1]))
    return merged


def extract_symbol_hits_from_text(
    text: str,
    project_root: Path,
    *,
    limit: int = 8,
) -> list[tuple[str, int, str]]:
    """Structured map_search / diagnose hits: (path, line, symbol)."""
    root = project_root.resolve()
    hits: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for match in _MAP_HIT_ROW.finditer(text):
        rel = _norm_path(match.group("file"))
        if not rel:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        hit = (rel, int(match.group("line")), match.group("symbol"))
        if hit in seen:
            continue
        seen.add(hit)
        hits.append(hit)
        if len(hits) >= limit:
            return hits
    for match in _SYMBOL_AT_LINE.finditer(text):
        rel = _norm_path(match.group("path"))
        if not rel:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if not path.is_file():
            continue
        hit = (rel, int(match.group("line")), match.group("symbol"))
        if hit in seen:
            continue
        seen.add(hit)
        hits.append(hit)
        if len(hits) >= limit:
            break
    return hits


def rank_edit_relevant_paths(
    text: str,
    paths: list[str],
    *,
    intent_text: str = "",
) -> list[str]:
    """Keep diagnose handoff focused on likely edit targets, not supporting evidence."""
    if not paths:
        return []
    if not intent_text.strip():
        return list(dict.fromkeys(paths))

    scored: list[tuple[int, int, str]] = []
    for idx, rel in enumerate(dict.fromkeys(paths)):
        score = _path_relevance_score(text, rel, intent_text)
        scored.append((score, -idx, rel))

    positives = [item for item in scored if item[0] > 0]
    if positives:
        max_score = max(score for score, _neg_idx, _rel in positives)
        focused = [item for item in positives if item[0] >= max_score - 2]
        return [rel for _score, _neg_idx, rel in sorted(focused, reverse=True)]

    neutral = [item for item in scored if item[0] == 0]
    if neutral:
        return [rel for _score, _neg_idx, rel in sorted(neutral, reverse=True)]

    return [rel for _score, _neg_idx, rel in scored]


def rank_edit_relevant_symbol_hits(
    text: str,
    hits: list[tuple[str, int, str]],
    *,
    intent_text: str = "",
) -> list[tuple[str, int, str]]:
    if not hits or not intent_text.strip():
        return hits
    ranked: list[tuple[int, int, tuple[str, int, str]]] = []
    for idx, hit in enumerate(hits):
        rel, _line, symbol = hit
        score = _path_relevance_score(text, rel, intent_text)
        if _keyword_overlap(_split_terms(symbol), _split_terms(intent_text)):
            score += 2
        ranked.append((score, -idx, hit))
    positives = [item for item in ranked if item[0] > 0]
    if positives:
        max_score = max(score for score, _neg_idx, _hit in positives)
        focused = [item for item in positives if item[0] >= max_score - 2]
        return [hit for _score, _neg_idx, hit in sorted(focused, reverse=True)]
    neutral = [item for item in ranked if item[0] == 0]
    if neutral:
        return [hit for _score, _neg_idx, hit in sorted(neutral, reverse=True)]
    return [hit for _score, _neg_idx, hit in ranked]


def format_diagnose_handoff_block(
    summaries: dict[str, str],
    project_root: Path,
    *,
    intent_text: str = "",
) -> str:
    """Compact structured hits for edit subtasks — reduces blind grep."""
    if not summaries:
        return ""
    hits: list[tuple[str, int, str]] = []
    slices: dict[str, tuple[int, int]] = {}
    for text in summaries.values():
        hits.extend(
            rank_edit_relevant_symbol_hits(
                text,
                extract_symbol_hits_from_text(text, project_root),
                intent_text=intent_text,
            )
        )
        for rel, span in extract_line_refs_from_text(text, project_root).items():
            prev = slices.get(rel)
            if prev is None:
                slices[rel] = span
            else:
                slices[rel] = (min(prev[0], span[0]), max(prev[1], span[1]))
    if intent_text:
        ranked_paths = rank_edit_relevant_paths(
            "\n".join(summaries.values()),
            list(slices),
            intent_text=intent_text,
        )
        slices = {rel: slices[rel] for rel in ranked_paths if rel in slices}
    if not hits and not slices:
        return ""
    lines = [
        "Structured locate results from prior diagnose "
        "(edit_file here — do NOT repeat map_search/grep for the same targets):",
    ]
    for rel, line, symbol in hits[:8]:
        lines.append(f"  - {rel}:{line}  {symbol}")
    for rel, (start, end) in list(slices.items())[:5]:
        lines.append(f"  - slice {rel}:{start}-{end}")
    return "\n".join(lines)


def _node_intent_text(node: SubTaskNode) -> str:
    return " ".join(
        part
        for part in (node.description, node.acceptance_criteria or "")
        if part
    )


def _path_relevance_score(text: str, rel: str, intent_text: str) -> int:
    score = 0
    intent_terms = _split_terms(intent_text)
    path_terms = _split_terms(rel)
    if _keyword_overlap(path_terms, intent_terms):
        score += 1
    if rel.endswith(".sql") and _keyword_overlap(
        {"sql", "view", "schema", "db", "视图"},
        intent_terms,
    ):
        score += 3

    for line in text.splitlines():
        if rel not in line:
            continue
        lower = line.lower()
        if any(hint in lower for hint in _EDIT_POSITIVE_HINTS):
            score += 2
        if any(hint in lower for hint in _EVIDENCE_ONLY_HINTS):
            score -= 2
        if _keyword_overlap(_split_terms(line), intent_terms):
            score += 1
    return score


def _split_terms(text: str) -> set[str]:
    lower = text.lower()
    terms = set(re.findall(r"[a-z0-9_]{3,}", lower))
    for hint in (*_EDIT_POSITIVE_HINTS, *_EVIDENCE_ONLY_HINTS):
        if hint in lower:
            terms.add(hint)
    return terms


def _keyword_overlap(a: set[str], b: set[str]) -> bool:
    return bool(a & b)
