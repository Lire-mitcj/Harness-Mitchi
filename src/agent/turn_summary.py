"""Compact turn summaries for system-prompt injection after history folding."""

from __future__ import annotations

import re
from collections.abc import Sequence

from src.agent.events import PARALLEL_RETRIEVAL_TOOLS
from src.agent.manifest import Sufficiency
from src.agent.run_state import RunState
from src.agent.types import Message, ToolCall


def summarize_tool_call(tc: ToolCall) -> str:
    args = tc.arguments or {}
    if tc.name == "view_symbol_code":
        target = args.get("target_file") or args.get("file") or "unknown"
        symbol = args.get("symbol") or args.get("name")
        if symbol:
            return f"view_symbol_code({target}::{symbol})"
        return f"view_symbol_code({target})"
    if tc.name == "grep_search":
        pattern = str(args.get("pattern") or "").strip()
        path = str(args.get("path") or ".").strip()
        include = str(args.get("include") or "").strip()
        scope = path if not include else f"{path}, include={include}"
        patterns = args.get("patterns") or []
        if isinstance(patterns, list) and patterns:
            preview = ", ".join(repr(str(item)) for item in patterns[:4])
            if len(patterns) > 4:
                preview += ", ..."
            return f"grep_search([{preview}] @ {scope})"
        return f"grep_search({pattern!r} @ {scope})"
    if tc.name == "codebase_retrieve":
        query = str(args.get("query") or "").strip()
        if len(query) > 140:
            query = query[:137] + "..."
        return f"codebase_retrieve({query!r})"
    if tc.name == "decision_edit":
        target = args.get("target_file") or "unknown"
        intent = " ".join(str(args.get("intent") or "").split())
        if len(intent) > 160:
            intent = intent[:157] + "..."
        return f"decision_edit({target}: {intent})" if intent else f"decision_edit({target})"
    return tc.name


def _is_process_only_intent(text: str) -> bool:
    lowered = text.casefold().strip()
    patterns = (
        r"^(?:let me|i need to|now let me)\s+(?:first\s+)?(?:examine|look|read|check|analy[sz]e|inspect|load)",
        r"\b(?:read|look|examine|check)\s+more\s+thoroughly\b",
        r"\bunderstand\s+the\s+(?:full|current)\s+(?:picture|state|system|app)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


def _first_error_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Error:"):
            return stripped[:240]
        if "BLOCK:" in stripped:
            return stripped[:240]
    compact = " ".join(text.split())
    return compact[:240] if compact else "failed"


def _extract_reuse_hint(text: str) -> str:
    context_match = re.search(r"CURRENT_CONTEXT\s*\(([^)]+)\)", text)
    if context_match:
        return context_match.group(1).strip()

    durable_match = re.search(
        r"already durable in the context:\s*\n-\s*`?([^`\n]+)`?\s*`?([^`\n(]+)?",
        text,
    )
    if durable_match:
        location = durable_match.group(1).strip()
        symbol = (durable_match.group(2) or "").strip()
        return f"{location} {symbol}".strip()

    span_match = re.search(r"`([^`:]+):(\d+-\d+)`\s*`?(\w+)?", text)
    if span_match:
        symbol = (span_match.group(3) or "").strip()
        base = f"{span_match.group(1)}:{span_match.group(2)}"
        return f"{base} {symbol}".strip() if symbol else base
    return ""


def _is_duplicate_or_block_outcome(content: str) -> bool:
    markers = (
        "RETRIEVAL DUPLICATE",
        "DUPLICATE ANCHOR",
        "BLOCK:",
        "BLOCK_SEARCH_FORCE_EDIT:",
    )
    return any(marker in content for marker in markers)


def _format_retrieval_outcome(tc: ToolCall, content: str) -> str | None:
    call = summarize_tool_call(tc)
    text = content or ""
    if _is_duplicate_or_block_outcome(text):
        reuse = _extract_reuse_hint(text)
        if reuse:
            return f"{call} → BLOCK (reuse {reuse})"
        return f"{call} → BLOCK ({_first_error_line(text)})"
    if tc.name in PARALLEL_RETRIEVAL_TOOLS and (
        "Error:" in text or "failed" in text.casefold()
    ):
        return f"{call} → {_first_error_line(text)}"
    return None


def _last_substantive_content(messages: Sequence[Message]) -> str:
    for message in reversed(messages):
        if message.role != "assistant" or not message.content.strip():
            continue
        if "[CONTEXT COLLAPSE" in message.content:
            continue
        normalized = " ".join(message.content.split())
        if _is_process_only_intent(normalized):
            continue
        if len(normalized) <= 400:
            return normalized
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[。！？.!?])\s+", normalized)
            if part.strip()
        ]
        concise = next((part for part in reversed(sentences) if len(part) <= 400), "")
        return concise or "执行已记录的工具调用与 checklist"
    return "沿用当前计划"


def _manifest_snapshot_line(
    run_state: RunState,
    tools_available: frozenset[str] | None,
) -> str:
    manifest = run_state.manifest
    edit_ready = manifest.sufficiency in {
        Sufficiency.SUFFICIENT_FOR_EDIT,
        Sufficiency.SUFFICIENT_FOR_VERIFY,
    }
    parts = [
        f"edit_ready={'yes' if edit_ready else 'no'}",
        f"no_gain={run_state.retrieval_no_gain_rounds}",
        f"sufficiency={manifest.sufficiency}",
    ]
    if tools_available:
        parts.append(f"tools={','.join(sorted(tools_available))}")
    if run_state.grep_suggested_views:
        views = [
            f"{item.get('file')}::{item.get('symbol')}"
            for item in run_state.grep_suggested_views[:3]
            if item.get("file") or item.get("symbol")
        ]
        if views:
            parts.append(f"suggested_views={'; '.join(views)}")
    failures = [
        item.need for item in manifest.failure_items[:1]
    ] + list(run_state.validation.issues[:1])
    if failures:
        parts.append(f"fix={failures[0][:120]}")
    return "- 状态快照（折叠时）：" + " | ".join(parts)


def build_turn_summary(
    folded_messages: Sequence[Message],
    *,
    run_state: RunState,
    tools_available: frozenset[str] | None = None,
) -> str:
    """Build a role-aware summary; tool_calls are authoritative over assistant narration."""
    reads: list[str] = []
    edits: list[str] = []
    errors: list[str] = []
    retrieval_outcomes: list[str] = []
    last_tool_calls: list[ToolCall] = []
    pending_calls: list[ToolCall] = []

    for message in folded_messages:
        if message.tool_calls:
            last_tool_calls = list(message.tool_calls)
            pending_calls = list(message.tool_calls)
            for tc in message.tool_calls:
                if tc.name in PARALLEL_RETRIEVAL_TOOLS:
                    reads.append(summarize_tool_call(tc))
                elif tc.name == "decision_edit":
                    edits.append(summarize_tool_call(tc))
        elif message.role == "tool":
            if pending_calls:
                tc = pending_calls.pop(0)
                outcome = _format_retrieval_outcome(tc, message.content)
                if outcome:
                    retrieval_outcomes.append(outcome)
            reads.extend(
                re.findall(r"(?:file|anchor):\s*`?([^`\s]+)", message.content)
            )
            if "Error:" in message.content or "failed" in message.content.casefold():
                errors.append(" ".join(message.content.split())[:300])

    if last_tool_calls:
        decision = ", ".join(summarize_tool_call(tc) for tc in last_tool_calls)
    else:
        decision = _last_substantive_content(folded_messages)

    lines = [
        "### TURN SUMMARY",
        f"- 决策：{decision}",
        f"- 已读取：{', '.join(dict.fromkeys(reads)) if reads else '无'}",
        f"- 编辑：{', '.join(dict.fromkeys(edits)) if edits else '无'}",
    ]
    if retrieval_outcomes:
        lines.append(
            f"- 检索结果：{', '.join(dict.fromkeys(retrieval_outcomes))}"
        )
    lines.extend([
        f"- 验证/错误：{errors[-1] if errors else '无'}",
        _manifest_snapshot_line(run_state, tools_available),
        "- 下一步：严格依据 RUNTIME STATE 与 STEP EVIDENCE 行动",
    ])
    return "\n".join(lines)
