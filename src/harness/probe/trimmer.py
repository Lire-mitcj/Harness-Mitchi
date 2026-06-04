from __future__ import annotations

import json
from typing import Any, Callable, Awaitable

SUMMARY_PROMPT = (
    "Summarize the following conversation turns into a single concise paragraph "
    "that preserves all key decisions, file paths mentioned, and tool outcomes. "
    "Do not lose any factual detail that would be needed to continue the task."
)


async def summarize_old_turns(
    messages: list[dict[str, Any]],
    llm_client: Callable[..., Awaitable[str]],
    *,
    keep_recent: int = 6,
) -> list[dict[str, Any]]:
    """Compress early conversation turns into a single summary message.

    Keeps the system message (index 0) and the last *keep_recent* messages
    intact; everything in between is replaced by a summary generated via
    *llm_client*.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    system = messages[0] if messages[0].get("role") == "system" else None
    start = 1 if system else 0
    boundary = len(messages) - keep_recent

    old_turns = messages[start:boundary]
    recent = messages[boundary:]

    old_text = "\n".join(
        f"[{m.get('role', '?')}]: {(m.get('content') or '')[:500]}"
        for m in old_turns
    )

    summary = await llm_client(
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": old_text},
        ],
    )

    summary_msg: dict[str, Any] = {
        "role": "system",
        "content": f"[Conversation summary]\n{summary}",
    }

    result: list[dict[str, Any]] = []
    if system:
        result.append(system)
    result.append(summary_msg)
    result.extend(recent)
    return result


def truncate_tool_outputs(
    messages: list[dict[str, Any]],
    max_per_tool: int = 2000,
) -> list[dict[str, Any]]:
    """Shorten tool-result messages that exceed *max_per_tool* characters."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "tool" and isinstance(msg.get("content"), str):
            content = msg["content"]
            if len(content) > max_per_tool:
                truncated = content[: max_per_tool - 40]
                suffix = f"\n\n... [truncated {len(content) - len(truncated)} chars]"
                msg = {**msg, "content": truncated + suffix}
        out.append(msg)
    return out


def evict_irrelevant_files(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove tool results that look like raw file contents from all but
    the most recent occurrence of each file path.

    Heuristic: if a tool result contains ``# File: <path>`` or starts with a
    block of source code (many lines without conversational text), it's
    treated as file content.
    """
    seen_files: dict[str, int] = {}
    evict_indices: set[int] = set()

    for idx, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        path = _extract_file_path(content)
        if path is None:
            continue
        if path in seen_files:
            evict_indices.add(seen_files[path])
        seen_files[path] = idx

    return [m for i, m in enumerate(messages) if i not in evict_indices]


def hard_truncate(messages: list[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Last-resort: drop the oldest non-system messages until the total
    character count is roughly under *budget* tokens (approximated as 4
    chars/token).
    """
    char_budget = budget * 4

    def _char_len(msg: dict[str, Any]) -> int:
        content = msg.get("content") or ""
        if isinstance(content, str):
            return len(content)
        return len(json.dumps(content))

    total_chars = sum(_char_len(m) for m in messages)
    if total_chars <= char_budget:
        return messages

    system = messages[0] if messages and messages[0].get("role") == "system" else None
    start = 1 if system else 0
    kept: list[dict[str, Any]] = list(reversed(messages[start:]))
    running = _char_len(system) if system else 0
    result: list[dict[str, Any]] = []

    for msg in kept:
        cost = _char_len(msg)
        if running + cost > char_budget:
            continue
        running += cost
        result.append(msg)

    result.reverse()
    if system:
        result.insert(0, system)
    return result


def _extract_file_path(content: str) -> str | None:
    """Try to pull a file path from a tool-result message."""
    for line in content.split("\n")[:5]:
        stripped = line.strip()
        if stripped.startswith("# File:"):
            return stripped.removeprefix("# File:").strip()
        if stripped.startswith("```") and "/" in stripped:
            return stripped.strip("`").strip()
    return None
