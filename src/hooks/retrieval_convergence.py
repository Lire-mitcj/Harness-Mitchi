"""Helpers for retrieval saturation / duplicate-round convergence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

RETRIEVAL_TOOLS = frozenset({"grep_search", "view_symbol_code", "codebase_retrieve"})
PRIMARY_RETRIEVAL_TOOLS = frozenset({"grep_search", "view_symbol_code"})
HEAVY_RETRIEVAL_TOOLS = frozenset({"codebase_retrieve"})


def is_duplicate_retrieval_result(tool_name: str, result: Any) -> bool:
    """True when a retrieval call replayed already-grounded evidence."""
    if tool_name not in RETRIEVAL_TOOLS:
        return False

    metadata = getattr(result, "metadata", None) or {}
    if metadata.get("is_mock_success"):
        return True
    if metadata.get("duplicate_anchor_replay"):
        return True

    error_text = str(getattr(result, "error", "") or "")
    output_text = str(getattr(result, "output", "") or "")
    combined = f"{error_text} {output_text}".casefold()
    if any(
        phrase in combined
        for phrase in (
            "already present in current_context",
            "was already loaded previously",
            "do not re-fetch",
            "redundant search",
            "redundant search query",
            "symbol dominance detected",
        )
    ):
        return True
    if "duplicate" in combined and any(
        marker in combined for marker in ("blocked", "already", "do not re-fetch", "do not repeat")
    ):
        return True

    raw = metadata.get("raw_evidence_store")
    refresh = metadata.get("refresh_evidence_store")
    if isinstance(raw, list) and not raw and refresh:
        return True

    return False


def view_round_all_duplicate(pairs: Sequence[tuple[str, Any]]) -> bool:
    """True when every view_symbol_code call in the round replayed existing evidence."""
    view_calls = [
        (name, result) for name, result in pairs if name == "view_symbol_code"
    ]
    if not view_calls:
        return False
    return all(is_duplicate_retrieval_result(name, result) for name, result in view_calls)


def retrieval_round_all_duplicate(pairs: Sequence[tuple[str, Any]]) -> bool:
    """True when every retrieval tool in the round replayed existing evidence."""
    retrieval = [(name, result) for name, result in pairs if name in RETRIEVAL_TOOLS]
    if not retrieval:
        return False
    return all(is_duplicate_retrieval_result(name, result) for name, result in retrieval)


def retrieval_tool_signal_status(tool_name: str, result: Any) -> str:
    """Short status token for RUNTIME STATE last-tool summaries."""
    if tool_name in RETRIEVAL_TOOLS and is_duplicate_retrieval_result(tool_name, result):
        return "duplicate_replay"
    return "success" if getattr(result, "success", False) else "failed"


def format_duplicate_retrieval_receipt(
    tool_name: str,
    result: Any,
    *,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Explicit receipt when retrieval replayed already-grounded evidence."""
    metadata = getattr(result, "metadata", None) or {}
    observation = (
        str(metadata.get("duplicate_anchor_replay") or metadata.get("llm_observation") or "")
        or str(getattr(result, "output", "") or "")
    ).strip()
    error_text = str(getattr(result, "error", "") or "").strip()

    header = (
        "[RETRIEVAL DUPLICATE — NO NEW EVIDENCE]\n"
        "Status: already loaded (NOT an empty search, NOT a tool failure).\n"
        "Reuse STEP EVIDENCE `loaded` and LOADED CODE ANCHORS; do not retry this read."
    )
    if observation:
        return f"{header}\n\n{observation}"
    if error_text:
        return f"{header}\n\n{error_text}"

    args = arguments or {}
    target = str(args.get("target_file") or args.get("path") or "?")
    symbol = str(args.get("symbol") or args.get("pattern") or args.get("query") or "?")
    return (
        f"{header}\n\n"
        f"Requested: {tool_name} target=`{target}` symbol/query=`{symbol}` "
        "— already grounded in context."
    )
