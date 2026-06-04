from __future__ import annotations

import re

from src.agent.explore_guard import _normalize_grep_scope
from src.agent.types import Message, system_message

_SECTION_RE = re.compile(r"^===== (.+?) =====\s*$", re.MULTILINE)
_GREP_HIT_RE = re.compile(r"^(\S+:\d+:.+)$", re.MULTILINE)
_READ_RANGE_RE = re.compile(
    r"^(?:--- )?(.+?)(?:\s+lines?\s+(\d+)(?:-(\d+))?)?\s*---?\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_DEF_RE = re.compile(r"^(def |class |CREATE VIEW|CREATE OR REPLACE VIEW)", re.MULTILINE | re.IGNORECASE)
_SQL_HINT_RE = re.compile(
    r"\b(SELECT|CREATE VIEW|DROP VIEW|text\s*\(|boarding_pass|build_\w+_sql|v_\w+)\b",
    re.IGNORECASE,
)


def build_exploration_digest(
    messages: list[Message],
    *,
    max_chars: int = 12_000,
) -> str:
    """Compress tool outputs + read ranges into a session summary."""
    read_ranges: list[str] = []
    read_paths: list[str] = []
    grep_hits: list[str] = []
    grep_queries: list[str] = []
    map_queries: list[str] = []
    map_hits: list[str] = []
    code_snippets: list[str] = []
    errors: list[str] = []
    seen_ranges: set[str] = set()
    seen_paths: set[str] = set()

    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.name == "grep_search":
                    pattern = tc.arguments.get("pattern") or tc.arguments.get("query")
                    path = tc.arguments.get("path")
                    include = tc.arguments.get("include") or tc.arguments.get("glob")
                    scope_path, scope_glob = _grep_scope_from_args(path, include)
                    if isinstance(pattern, str):
                        q = f"{pattern!r} in {scope_path or scope_glob or '*'}"
                        if q not in grep_queries:
                            grep_queries.append(q)
                elif tc.name == "map_search":
                    query = tc.arguments.get("query")
                    if isinstance(query, str) and query not in map_queries:
                        map_queries.append(query)
                elif tc.name in {"read_file", "read_files"}:
                    _record_read_call(
                        tc.name,
                        tc.arguments,
                        read_ranges,
                        read_paths,
                        seen_ranges,
                        seen_paths,
                    )

        if msg.role != "tool":
            continue
        content = (msg.content or "").strip()
        if not content:
            continue
        if content.startswith("Error:"):
            errors.append(content[:240])
            continue

        if content.startswith("repo_map matches for"):
            for line in content.splitlines()[1:]:
                stripped = line.strip()
                if stripped.startswith("- ") and stripped not in map_hits:
                    map_hits.append(stripped[2:][:200])
                if len(map_hits) >= 20:
                    break

        for match in _SECTION_RE.finditer(content):
            path = match.group(1).split(" (")[0].strip()
            norm = _norm(path)
            if norm and norm not in seen_paths:
                seen_paths.add(norm)
                read_paths.append(norm)

        for hit in _GREP_HIT_RE.findall(content):
            if hit not in grep_hits:
                grep_hits.append(hit)
            if len(grep_hits) >= 50:
                break

        for line in content.splitlines():
            stripped = line.strip()
            if _DEF_RE.search(stripped) or (
                "SELECT" in stripped.upper() and len(stripped) > 20
            ):
                snip = stripped[:160]
                if snip not in code_snippets:
                    code_snippets.append(snip)
                if len(code_snippets) >= 25:
                    break

        for hint in _SQL_HINT_RE.findall(content):
            token = hint if isinstance(hint, str) else hint[0]
            if token not in code_snippets:
                code_snippets.append(token)

    lines: list[str] = [
        "Session exploration summary (paths, line ranges, map/grep hits, code seen so far):",
    ]
    if read_ranges:
        lines.append("Line ranges already read:")
        lines.extend(f"  - {r}" for r in read_ranges[:25])
    elif read_paths:
        lines.append("Files already read: " + ", ".join(read_paths[:20]))
    if map_queries:
        lines.append("Repo map searches already run:")
        lines.extend(f"  - {q}" for q in map_queries[:12])
    if map_hits:
        lines.append("Repo map hits (sample):")
        lines.extend(f"  - {h}" for h in map_hits[:15])
    if grep_queries:
        lines.append("Grep queries already run:")
        lines.extend(f"  - {q}" for q in grep_queries[:12])
    if grep_hits:
        lines.append("Grep hits (sample):")
        lines.extend(f"  - {h}" for h in grep_hits[:20])
    if code_snippets:
        lines.append("Code / SQL seen (snippets):")
        lines.extend(f"  - {s}" for s in code_snippets[:18])
    if errors:
        lines.append("Tool errors:")
        lines.extend(f"  - {e}" for e in errors[-5:])
    if len(lines) == 1:
        return ""

    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[: max_chars - 24] + "\n…[digest truncated]"
    return text


def _grep_scope_from_args(
    path: object,
    include: object,
) -> tuple[str | None, str | None]:
    p = path if isinstance(path, str) else None
    inc = include if isinstance(include, str) else None
    return _normalize_grep_scope(p, include=inc)


def _record_read_call(
    tool_name: str,
    arguments: dict,
    read_ranges: list[str],
    read_paths: list[str],
    seen_ranges: set[str],
    seen_paths: set[str],
) -> None:
    if tool_name == "read_file":
        paths = [arguments.get("path")] if isinstance(arguments.get("path"), str) else []
        start = arguments.get("start_line")
        end = arguments.get("end_line")
    else:
        paths = [p for p in (arguments.get("paths") or []) if isinstance(p, str)]
        start = arguments.get("start_line")
        end = arguments.get("end_line")

    for raw in paths:
        norm = _norm(str(raw))
        if not norm:
            continue
        if start and end:
            label = f"{norm}:{start}-{end}"
        elif start:
            label = f"{norm}:{start}+"
        else:
            label = norm
        if label not in seen_ranges:
            seen_ranges.add(label)
            read_ranges.append(label)
        if norm not in seen_paths:
            seen_paths.add(norm)
            read_paths.append(norm)


def _norm(path: str) -> str:
    p = path.replace("\\", "/").strip()
    for marker in ("/database-course-design/", "database-course-design/"):
        if marker in p:
            p = p.split(marker, 1)[-1]
    return p.lstrip("./")


def format_digest_system_block(digest: str) -> Message:
    return system_message(
        "Session exploration summary (preserved across context compression — "
        "use this instead of re-reading the same files):\n\n"
        + digest.strip()
    )
