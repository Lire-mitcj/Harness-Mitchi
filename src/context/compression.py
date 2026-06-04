from __future__ import annotations

from typing import Any, Protocol

from src.agent.types import Message, system_message
from src.context.window import count_tokens


class LLMClient(Protocol):
    """Minimal interface for an LLM that can produce summaries."""

    async def complete(self, messages: list[dict[str, Any]]) -> str: ...


SUMMARIZE_SYSTEM = (
    "You are a conversation compressor. Summarise the following conversation "
    "history into a concise, factual summary that preserves all key decisions, "
    "file changes, tool results, and open tasks. Use bullet points. "
    "Do NOT include pleasantries or meta-commentary."
)


async def summarize_messages(
    messages: list[Message],
    llm_client: LLMClient,
    *,
    keep_recent: int = 4,
) -> list[Message]:
    """Compress older messages into a single summary while keeping recent turns.

    The first message (system prompt) is always preserved.  The last
    *keep_recent* messages are kept verbatim.  Everything in between is
    summarised into a single system message.
    """
    if len(messages) <= keep_recent + 1:
        return messages  # nothing to compress

    system = messages[0] if messages[0].role == "system" else None
    start = 1 if system else 0
    to_compress = messages[start: -keep_recent]
    recent = messages[-keep_recent:]

    if not to_compress:
        return messages

    transcript = "\n".join(
        f"[{m.role}] {m.content or ''}" for m in to_compress
    )

    summary_text = await llm_client.complete([
        {"role": "system", "content": SUMMARIZE_SYSTEM},
        {"role": "user", "content": f"Conversation to summarise:\n\n{transcript}"},
    ])

    summary_msg = system_message(
        f"<compressed_history>\n{summary_text}\n</compressed_history>"
    )

    result: list[Message] = []
    if system:
        result.append(system)
    result.append(summary_msg)
    result.extend(recent)
    return result


def truncate_content(content: str, max_tokens: int) -> str:
    """Truncate *content* to fit within *max_tokens*, appending a notice."""
    tokens = count_tokens(content)
    if tokens <= max_tokens:
        return content

    # Binary search for the right character cutoff
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(content[:mid]) <= max_tokens - 20:  # reserve space for notice
            lo = mid
        else:
            hi = mid - 1

    return content[:lo] + "\n\n[... truncated ...]"
