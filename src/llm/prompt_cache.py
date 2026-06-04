from __future__ import annotations

import copy
import logging
from typing import Any, Literal

import litellm

log = logging.getLogger(__name__)

_CACHE_MARKER = "_mitkii_cache_breakpoint"
CacheTTL = Literal["5m", "1h"]


def mark_cache_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Tag a message dict for ephemeral prefix caching at the LLM boundary."""
    out = dict(message)
    out[_CACHE_MARKER] = True
    return out


def apply_prompt_cache(
    messages: list[dict[str, Any]],
    *,
    model: str,
    enabled: bool = True,
    min_tokens: int = 1024,
    ttl: CacheTTL = "5m",
) -> list[dict[str, Any]]:
    """Convert internal cache markers to provider ``cache_control`` blocks."""
    if not messages:
        return messages

    stripped = [_strip_marker(dict(m)) for m in messages]
    if not enabled:
        return stripped

    try:
        supported = litellm.supports_prompt_caching(model=model)
    except Exception:
        supported = False
    if not supported:
        log.debug("Prompt cache skipped — model %s does not support caching", model)
        return stripped

    cache_control: dict[str, str] = {"type": "ephemeral"}
    if ttl == "1h":
        cache_control["ttl"] = "1h"

    out: list[dict[str, Any]] = []
    for msg in messages:
        item = dict(msg)
        wants_cache = bool(item.pop(_CACHE_MARKER, False))
        if wants_cache and _message_meets_min_tokens(item, model=model, min_tokens=min_tokens):
            item["cache_control"] = dict(cache_control)
        out.append(item)
    return out


def extract_cache_usage(usage: Any) -> tuple[int, int]:
    """Return (cache_creation_input_tokens, cache_read_input_tokens)."""
    if usage is None:
        return 0, 0
    created = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    if created == 0 and read == 0:
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            created = int(getattr(details, "cached_tokens", 0) or 0)
    return created, read


def _strip_marker(msg: dict[str, Any]) -> dict[str, Any]:
    out = copy.copy(msg)
    out.pop(_CACHE_MARKER, None)
    return out


def _message_meets_min_tokens(
    msg: dict[str, Any],
    *,
    model: str,
    min_tokens: int,
) -> bool:
    content = msg.get("content")
    if not isinstance(content, str) or not content.strip():
        return False
    probe = {"role": msg.get("role", "user"), "content": content}
    try:
        tokens = litellm.token_counter(messages=[probe], model=model)
    except Exception:
        tokens = max(1, len(content) // 4)
    if tokens < min_tokens:
        log.debug(
            "Prompt cache breakpoint skipped — %d tokens < %d minimum",
            tokens,
            min_tokens,
        )
        return False
    return True
