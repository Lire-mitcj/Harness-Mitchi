from __future__ import annotations

from typing import Any

from src.tools.arg_normalize import normalize_grep_search_args


def prepare_grep_search_args(
    arguments: dict[str, Any],
    *,
    hint_text: str = "",
) -> dict[str, Any]:
    """Normalize grep_search args before preflight (assembled loop entry point)."""
    return normalize_grep_search_args(dict(arguments), hint_text=hint_text)
