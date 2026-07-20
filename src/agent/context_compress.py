"""Ordered 5-layer context compression for CoreLLM turns.

L1 Tool return cap     — after_tool truncate + history receipts (hooks/loop)
L2 Context size limit  — token-budget fold after L3/L4
L3 Age eviction        — demote non-hot verbatim by step age → locator
L4 Locator map         — file:symbol:span (+hash/step), no snippet
L5 LLM/heuristic summary — last resort; never for current hot focus

Hot (next edit / recent): full bodies in LOADED CODE ANCHORS.
Warm (aged / background): CODE LOCATORS only.
Cold (budget pressure): short summary or drop — never duplicate LOADED.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

# Keep verbatim if created within this many steps of the current turn,
# or if the file is in the hot set (active / next edit_plan target).
HOT_ANCHOR_AGE_STEPS = 1

# Assistant free-text beside tool_calls is almost unused by the executor.
ASSISTANT_NARRATION_MAX_CHARS = 200


def norm_file(path: str | None) -> str:
    return str(path or "").replace("\\", "/").lstrip("./")


def anchor_id(item: dict[str, Any]) -> str:
    span = item.get("span") or []
    if not item.get("file") or len(span) != 2:
        return ""
    return f"{item['file']}:{span[0]}-{span[1]}"


def anchor_to_locator(
    item: dict[str, Any],
    *,
    step: int | None = None,
    reason: str = "age",
) -> dict[str, Any]:
    """L4: durable pointer without code body."""
    span = item.get("span") or []
    content_hash = str(
        item.get("hash")
        or item.get("content_hash")
        or item.get("file_hash")
        or ""
    )
    locator: dict[str, Any] = {
        "id": anchor_id(item) or f"{item.get('file')}:?",
        "file": norm_file(str(item.get("file") or "")),
        "symbol": str(item.get("symbol") or ""),
        "span": [int(span[0]), int(span[1])] if len(span) == 2 else [],
        "hash": content_hash[:12] if content_hash else "",
        "reason": reason,
    }
    if step is not None:
        locator["step"] = int(step)
    return locator


def format_locator_line(item: dict[str, Any]) -> str:
    file_path = item.get("file") or "?"
    span = item.get("span") or []
    span_text = f":{span[0]}-{span[1]}" if len(span) == 2 else ""
    symbol = str(item.get("symbol") or "").strip()
    step = item.get("step")
    hash_text = str(item.get("hash") or "").strip()
    parts = [f"`{file_path}{span_text}`"]
    if symbol:
        parts.append(f"`{symbol}`")
    meta: list[str] = []
    if step is not None:
        meta.append(f"step={step}")
    if hash_text:
        meta.append(f"hash={hash_text}")
    reason = str(item.get("reason") or "").strip()
    if reason:
        meta.append(reason)
    line = "- " + " ".join(parts)
    if meta:
        line += f" ({', '.join(meta)})"
    return line


def format_code_locators_block(locators: list[dict[str, Any]] | tuple[dict[str, Any], ...]) -> str:
    if not locators:
        return ""
    seen: set[str] = set()
    lines: list[str] = []
    for item in locators:
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or format_locator_line(item))
        if key in seen:
            continue
        seen.add(key)
        lines.append(format_locator_line(item))
    if not lines:
        return ""
    return (
        "### CODE LOCATORS (no body — re-view only if patching this target) ###\n"
        + "\n".join(lines)
    )


def hot_files_for_turn(
    *,
    active_files: list[str] | tuple[str, ...] | None,
    edit_plan_current_file: str | None = None,
    priority_files: frozenset[str] | set[str] | None = None,
) -> set[str]:
    hot = {norm_file(path) for path in (active_files or ()) if path}
    if edit_plan_current_file:
        hot.add(norm_file(edit_plan_current_file))
    for path in priority_files or ():
        if path:
            hot.add(norm_file(path))
    return {path for path in hot if path}


def should_keep_verbatim(
    item: dict[str, Any],
    *,
    current_step: int,
    created_step: int,
    hot_files: set[str],
    hot_age: int = HOT_ANCHOR_AGE_STEPS,
) -> bool:
    file_path = norm_file(str(item.get("file") or ""))
    if file_path and file_path in hot_files:
        return True
    return (current_step - int(created_step)) <= max(0, hot_age)


def merge_locators(
    existing: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    incoming: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    by_id: dict[str, dict[str, Any]] = {}
    for item in list(existing) + list(incoming):
        if not isinstance(item, dict):
            continue
        key = str(item.get("id") or "")
        if not key:
            continue
        by_id[key] = item
    return tuple(by_id.values())


def loaded_anchor_ids(raw_evidence: list[Any] | None) -> set[str]:
    ids: set[str] = set()
    for item in raw_evidence or ():
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("verbatim_code") or "").strip()
        if not code:
            continue
        aid = anchor_id(item)
        if aid:
            ids.add(aid)
        # Also key by file for coarse dual-channel suppression.
        file_path = norm_file(str(item.get("file") or ""))
        if file_path:
            ids.add(f"file:{file_path}")
    return ids


def filter_summary_anchors_for_prompt(
    summaries: dict[str, str] | None,
    *,
    raw_evidence: list[Any] | None = None,
    locators: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
) -> dict[str, str]:
    """Drop summaries that duplicate LOADED ANCHORS or CODE LOCATORS (L4 wins)."""
    if not summaries:
        return {}
    blocked = loaded_anchor_ids(raw_evidence)
    for item in locators or ():
        if isinstance(item, dict) and item.get("id"):
            blocked.add(str(item["id"]))
            file_path = norm_file(str(item.get("file") or ""))
            if file_path:
                blocked.add(f"file:{file_path}")

    kept: dict[str, str] = {}
    for key, value in summaries.items():
        if key in blocked:
            continue
        # Keys may be file paths (budget fold) or file:span ids.
        if f"file:{norm_file(key)}" in blocked:
            continue
        if ":" in key:
            file_part = key.rsplit(":", 1)[0]
            # file:start-end → strip span
            match = re.match(r"^(.+):\d+-\d+$", key)
            if match:
                file_part = match.group(1)
            if f"file:{norm_file(file_part)}" in blocked:
                continue
        kept[key] = value
    return kept


_NARRATION_LEAD_RE = re.compile(
    r"(?is)^\s*(let me |i('ll| will) |i'm going to |looking |searching |"
    r"checking |reading |examining |我来|让我|先看|查看|检索).{0,80}"
)


def microcompact_assistant_content(content: str, *, has_tool_calls: bool) -> str:
    """Strip long Core narration that does not help execution (L1 history hygiene)."""
    text = (content or "").strip()
    if not text:
        return content
    if len(text) <= ASSISTANT_NARRATION_MAX_CHARS:
        return content
    if has_tool_calls:
        # Keep a short lead-in; tool_calls are authoritative.
        first_line = text.splitlines()[0].strip()
        if len(first_line) > ASSISTANT_NARRATION_MAX_CHARS:
            first_line = first_line[: ASSISTANT_NARRATION_MAX_CHARS - 1].rstrip() + "…"
        if _NARRATION_LEAD_RE.match(first_line) or len(text) > ASSISTANT_NARRATION_MAX_CHARS:
            return first_line if first_line else ""
        return text[: ASSISTANT_NARRATION_MAX_CHARS - 1].rstrip() + "…"
    # Final / mid answers: soft truncate only when extremely long.
    if len(text) > ASSISTANT_NARRATION_MAX_CHARS * 8:
        return text[: ASSISTANT_NARRATION_MAX_CHARS * 8 - 1].rstrip() + "…"
    return content


def microcompact_assistant_messages(messages: tuple[Any, ...]) -> tuple[Any, ...]:
    """Apply narration compact to assistant messages in history."""
    changed = False
    out: list[Any] = []
    for msg in messages:
        role = getattr(msg, "role", None)
        if role != "assistant":
            out.append(msg)
            continue
        content = getattr(msg, "content", "") or ""
        tool_calls = getattr(msg, "tool_calls", None)
        has_tools = bool(tool_calls)
        compacted = microcompact_assistant_content(content, has_tool_calls=has_tools)
        if compacted != content:
            changed = True
            out.append(replace(msg, content=compacted))
        else:
            out.append(msg)
    return tuple(out) if changed else messages
