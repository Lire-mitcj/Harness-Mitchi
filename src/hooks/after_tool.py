from __future__ import annotations

from typing import Any

from src.agent.types import ToolResult

DEFAULT_MAX_TOOL_OUTPUT_CHARS = 12_000
DEFAULT_KEEP_HEAD_CHARS = 8_000
DEFAULT_KEEP_TAIL_CHARS = 2_000


def trim_tool_output(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    keep_head_chars: int = DEFAULT_KEEP_HEAD_CHARS,
    keep_tail_chars: int = DEFAULT_KEEP_TAIL_CHARS,
) -> tuple[str, bool]:
    """Trim long tool-facing text while preserving useful head and tail context."""
    if len(text) <= max_chars:
        return text, False

    marker = (
        "\n\n"
        f"[tool_output_trimmed: omitted {len(text)} chars; "
        f"original_length={len(text)}; limit={max_chars}]"
        "\n\n"
    )
    if len(marker) >= max_chars:
        return marker[:max_chars], True

    content_budget = max_chars - len(marker)
    keep_head = max(0, min(keep_head_chars, content_budget))
    keep_tail = max(0, min(keep_tail_chars, content_budget - keep_head))
    omitted = len(text) - keep_head - keep_tail
    marker = marker.replace(
        f"omitted {len(text)} chars",
        f"omitted {omitted} chars",
    )
    return f"{text[:keep_head]}{marker}{text[-keep_tail:] if keep_tail else ''}", True


def apply_after_tool_output_limit(
    tool_name: str,
    result: ToolResult,
    *,
    max_chars: int = DEFAULT_MAX_TOOL_OUTPUT_CHARS,
) -> ToolResult:
    """After-tool hook that caps text exposed to the coordinator/UI.

    Durable artifacts in metadata, such as raw evidence or verbatim code, are kept
    intact. Only ToolResult.output and metadata.llm_observation are limited.
    """
    output, output_trimmed = trim_tool_output(result.output, max_chars=max_chars)
    metadata = dict(result.metadata or {})

    observation_trimmed = False
    observation = metadata.get("llm_observation")
    if isinstance(observation, str):
        metadata["llm_observation"], observation_trimmed = trim_tool_output(
            observation,
            max_chars=max_chars,
        )

    if not output_trimmed and not observation_trimmed:
        return result

    trim_info: dict[str, Any] = dict(metadata.get("tool_output_trim") or {})
    trim_info.update(
        {
            "tool": tool_name,
            "max_chars": max_chars,
            "output_trimmed": output_trimmed,
            "llm_observation_trimmed": observation_trimmed,
            "original_output_chars": len(result.output),
        }
    )
    metadata["tool_output_trim"] = trim_info

    return ToolResult(
        success=result.success,
        output=output,
        error=result.error,
        metadata=metadata,
    )
