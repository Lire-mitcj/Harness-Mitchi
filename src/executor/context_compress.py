from __future__ import annotations

import json

from src.agent.types import Message, system_message, user_message
from src.executor.exploration_digest import build_exploration_digest, format_digest_system_block


def merge_exploration_digests(
    existing: str | None,
    messages: list[Message],
    *,
    max_chars: int = 10_000,
) -> str:
    """Merge prior digest with newly extracted tool/assistant evidence."""
    parts: list[str] = []
    if existing and existing.strip():
        parts.append(existing.strip())
    fresh = build_exploration_digest(messages, max_chars=max_chars // 2)
    if fresh:
        parts.append(fresh)
    notes = _assistant_notes(messages)
    if notes:
        parts.append("Assistant notes from prior turns:\n" + notes)
    if not parts:
        return ""
    merged = "\n\n".join(parts)
    if len(merged) > max_chars:
        return merged[: max_chars - 24] + "\n…[digest truncated]"
    return merged


def _assistant_notes(messages: list[Message], *, max_notes: int = 4, per_note: int = 500) -> str:
    picked: list[str] = []
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        text = (msg.content or "").strip()
        if not text or msg.tool_calls:
            continue
        picked.append(text[:per_note] + ("…" if len(text) > per_note else ""))
        if len(picked) >= max_notes:
            break
    if not picked:
        return ""
    return "\n".join(f"- {n}" for n in reversed(picked))


def rebuild_compacted_executor_messages(
    *,
    base_messages: list[Message],
    digest: str,
    error_trace: list[str] | None = None,
    compact_reason: str = "context size",
) -> list[Message]:
    """Replace fat tool history with digest + minimal retry context."""
    out = list(base_messages)
    if digest.strip():
        out.append(format_digest_system_block(digest))
    out.append(user_message(
        "EXECUTOR_RUNTIME_JSON\n"
        + json.dumps(
            {
                "schema": "mitkii.executor_runtime.v1",
                "event": "context_folded",
                "reason": compact_reason,
                "rules": [
                    "Continue from SESSION_DIGEST_JSON.",
                    "Avoid repeating the same line ranges or context_search queries.",
                ],
                "recent_errors": [str(e) for e in (error_trace or [])[-6:]],
            },
            ensure_ascii=False,
            indent=2,
        )
    ))
    return out


def fold_executor_messages(
    *,
    base_messages: list[Message],
    prior_messages: list[Message],
    running_digest: str,
    error_trace: list[str] | None = None,
    reason: str = "exploration fold",
) -> tuple[list[Message], str]:
    """Merge tool history into digest and return slim message list."""
    digest = merge_exploration_digests(running_digest, prior_messages)
    if not digest.strip():
        return prior_messages, running_digest
    folded = rebuild_compacted_executor_messages(
        base_messages=base_messages,
        digest=digest,
        error_trace=error_trace,
        compact_reason=reason,
    )
    return folded, digest
