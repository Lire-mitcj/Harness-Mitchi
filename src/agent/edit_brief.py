"""EditBrief contract: Core → harness gate → EditLLM.

P0/P1/P2 helpers shared by preflight, decision_edit, and STEP EVIDENCE.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.agent.edit_materialize import sanitize_focus_symbols

MAX_FOCUS_SYMBOLS_PER_STEP = 3
MAX_INTENT_CHARS = 420
MAX_CONTEXT_SPAN_LINES = 220
MAX_AVAILABLE_SYMBOLS_LISTED = 24
# Core may pass a 1–2 line insert anchor; expand so EditLLM sees the symbol body.
MIN_FOCUS_CONTEXT_LINES = 24

_MULTI_CLAUSE_RE = re.compile(
    r"(?:"
    r"\band\b|\balso\b|\bthen\b|\bas\s+well\s+as\b|"
    r"以及|同时|并且|然后|并\s*且|除此之外|"
    r"[;；]"
    r")",
    re.IGNORECASE,
)
_EDIT_VERB_RE = re.compile(
    r"(?:"
    r"\b(?:add|wrap|implement|apply|insert|modify|fix|define|create|update|rename|remove|delete)\b|"
    r"添加|修改|实现|包装|装饰|定义|新增|更新|删除|插入|重命名"
    r")",
    re.IGNORECASE,
)
_FOLLOWON_SPLIT_RE = re.compile(
    r"(?:"
    r",\s*and\s+(?=add\b|create\b|implement\b|define\b|insert\b)|"
    r"\s+and\s+(?=add\b|create\b|implement\b|define\b|insert\b)|"
    r"，(?:并|同时)?(?:添加|新增|实现|定义)|"
    r"以及(?:添加|新增|实现)|"
    r"同时(?:添加|新增|实现)"
    r")",
    re.IGNORECASE,
)


def intent_needs_split(intent: str) -> bool:
    """True when intent packs more than one coherent EditLLM change.

    Same-file ``Update X and add nearby helper`` (2 verbs) is allowed — Edit may
    emit ≤3 SITE blocks. Three-or-more edit verbs with conjunctions should be
    expanded into multiple plan steps (harness auto-split), not left as one call.
    """
    text = str(intent or "").strip()
    if not text:
        return False
    if len(text) > MAX_INTENT_CHARS:
        return True
    verbs = list(_EDIT_VERB_RE.finditer(text))
    if len(verbs) >= 4:
        return True
    if len(verbs) >= 3 and _MULTI_CLAUSE_RE.search(text) and len(text) > 120:
        return True
    return False


def _window_span_lines(item: Mapping[str, Any]) -> int | None:
    span = item.get("span")
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        return None
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return None
    if end < start:
        return None
    return end - start + 1


def tighten_decision_edit_context(
    arguments: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Replace oversized context_window spans with per-symbol disk spans.

    Core often passes ``[1, EOF]``; after auto-split each step still needs a
    tight window or EditBrief E4 fires. Prefer ``locate_symbol_span`` for each
    focus symbol when any window exceeds ``MAX_CONTEXT_SPAN_LINES``.
    """
    args = dict(arguments)
    target = str(args.get("target_file") or args.get("file") or "").strip()
    focus_raw = args.get("focus_symbols") or []
    if isinstance(focus_raw, str):
        focus = [focus_raw.strip()] if focus_raw.strip() else []
    elif isinstance(focus_raw, Sequence):
        focus = [str(s).strip() for s in focus_raw if str(s).strip()]
    else:
        focus = []
    windows_raw = args.get("context_window") or []
    windows: list[dict[str, Any]] = (
        [dict(item) for item in windows_raw if isinstance(item, Mapping)]
        if isinstance(windows_raw, Sequence) and not isinstance(windows_raw, (str, bytes))
        else []
    )
    if not target or not windows:
        return args
    oversized = any(
        (n := _window_span_lines(item)) is not None and n > MAX_CONTEXT_SPAN_LINES
        for item in windows
    )
    if not oversized:
        return args

    if project_root is not None and focus:
        from src.agent.edit_materialize import locate_symbol_span

        tightened: list[dict[str, Any]] = []
        for symbol in focus:
            span = locate_symbol_span(Path(project_root), target, symbol)
            if span is None:
                continue
            tightened.append({
                "file": target,
                "span": [span[0], span[1]],
                "reason": f"{symbol} (tightened from oversized context_window)",
            })
        if tightened:
            args["context_window"] = tightened
            return args

    # Fallback: clip each oversized span to MAX lines from the start.
    clipped: list[dict[str, Any]] = []
    for item in windows:
        n = _window_span_lines(item)
        if n is None or n <= MAX_CONTEXT_SPAN_LINES:
            clipped.append(item)
            continue
        span = item.get("span")
        start = int(span[0])
        clipped.append({
            **item,
            "span": [start, start + MAX_CONTEXT_SPAN_LINES - 1],
            "reason": str(item.get("reason") or "") + " (clipped)",
        })
    args["context_window"] = clipped
    return args


def _window_for_symbol(
    windows: Sequence[Mapping[str, Any]],
    symbol: str,
    index: int,
) -> list[dict[str, Any]]:
    """Pick the context_window entry that best matches a focus symbol."""
    sym = symbol.casefold()
    ranked: list[tuple[int, dict[str, Any]]] = []
    for item in windows:
        if not isinstance(item, Mapping):
            continue
        reason = str(item.get("reason") or "").casefold()
        score = 0
        if sym and sym in reason:
            score += 3
        if sym and reason.startswith(sym):
            score += 2
        if score:
            ranked.append((score, dict(item)))
    if ranked:
        ranked.sort(key=lambda pair: -pair[0])
        return [ranked[0][1]]
    if 0 <= index < len(windows) and isinstance(windows[index], Mapping):
        return [dict(windows[index])]
    if windows and isinstance(windows[0], Mapping):
        return [dict(windows[0])]
    return []


def expand_decision_edit_to_steps(
    arguments: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Expand one Core decision_edit into 1..N equal-sized plan steps.

    Prefer one step per ``focus_symbols`` entry (matched to context_window by
    reason/order). Fall back to intent follow-on splitting. Returns a single
    unchanged step when the brief is already unit-sized.
    """
    target = str(arguments.get("target_file") or arguments.get("file") or "").strip()
    intent = str(arguments.get("intent") or arguments.get("goal") or "").strip()
    focus_raw = arguments.get("focus_symbols") or []
    if isinstance(focus_raw, str):
        focus = [focus_raw.strip()] if focus_raw.strip() else []
    elif isinstance(focus_raw, Sequence):
        focus = [str(s).strip() for s in focus_raw if str(s).strip()]
    else:
        focus = []
    windows_raw = arguments.get("context_window") or []
    windows: list[Mapping[str, Any]] = (
        [item for item in windows_raw if isinstance(item, Mapping)]
        if isinstance(windows_raw, Sequence) and not isinstance(windows_raw, (str, bytes))
        else []
    )

    needs = intent_needs_split(intent) or len(focus) >= 3
    if not needs:
        return [tighten_decision_edit_context(arguments, project_root=project_root)]

    steps: list[dict[str, Any]] = []
    if len(focus) >= 2:
        for index, symbol in enumerate(focus[:8]):
            win = _window_for_symbol(windows, symbol, index)
            reason = str((win[0].get("reason") if win else "") or "").strip()
            if reason and "tightened" not in reason.casefold():
                step_intent = reason
            else:
                step_intent = f"For `{symbol}`: {_short_intent_clause(intent)}"
            step: dict[str, Any] = {
                "target_file": target,
                "intent": step_intent,
                "focus_symbols": [symbol],
                "context_window": win or [dict(w) for w in windows[:1]],
                "step_id": f"auto-split-{index}",
            }
            if arguments.get("major_step_index") is not None:
                step["major_step_index"] = arguments.get("major_step_index")
            steps.append(
                tighten_decision_edit_context(step, project_root=project_root)
            )
        if steps:
            return steps

    primary, followons = split_intent_followons(intent)
    if followons:
        primary_focus = focus[:1] or focus
        primary_win = (
            _window_for_symbol(windows, primary_focus[0], 0)
            if primary_focus
            else [dict(w) for w in windows[:1]]
        )
        steps.append(
            tighten_decision_edit_context(
                {
                    "target_file": target,
                    "intent": primary,
                    "focus_symbols": list(primary_focus) if primary_focus else list(focus),
                    "context_window": primary_win or [dict(w) for w in windows],
                    "step_id": "auto-split-0",
                },
                project_root=project_root,
            )
        )
        for index, follow in enumerate(followons[:3]):
            sym = focus[min(index + 1, len(focus) - 1)] if focus else ""
            win = _window_for_symbol(windows, sym, index + 1) if sym else (
                [dict(w) for w in windows[index + 1 : index + 2]]
                or [dict(w) for w in windows[:1]]
            )
            steps.append(
                tighten_decision_edit_context(
                    {
                        "target_file": target,
                        "intent": follow,
                        "focus_symbols": [sym] if sym else list(focus[:1]),
                        "context_window": win,
                        "step_id": f"auto-split-{index + 1}",
                    },
                    project_root=project_root,
                )
            )
        return steps

    return [tighten_decision_edit_context(arguments, project_root=project_root)]


def _short_intent_clause(intent: str, *, limit: int = 160) -> str:
    text = " ".join(str(intent or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def split_intent_followons(intent: str) -> tuple[str, list[str]]:
    """Split compound intent into (primary, follow-on intents) for auto-queue."""
    text = str(intent or "").strip()
    if not text:
        return "", []
    parts = [
        p.strip(" ,，")
        for p in _FOLLOWON_SPLIT_RE.split(text)
        if p and p.strip(" ,，")
    ]
    if len(parts) < 2:
        return text, []
    primary = parts[0]
    followons: list[str] = []
    for part in parts[1:]:
        if not _EDIT_VERB_RE.search(part):
            lowered = part.casefold()
            if lowered.startswith(("a new ", "function ", "helper ")):
                part = f"Add {part}"
            elif not lowered.startswith("add"):
                part = f"Add {part}"
        followons.append(part)
    return primary, followons


def derive_edit_queue_followons(
    decision_edit_args: Sequence[Mapping[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Residual plan steps when Core packs a compound decision_edit in one call.

    Expands each overloaded call into atomic steps and returns everything after
    the first (the in-turn call should be rewritten to the first step).
    """
    residual: list[dict[str, Any]] = []
    for args in decision_edit_args:
        if not isinstance(args, Mapping):
            continue
        steps = expand_decision_edit_to_steps(args, project_root=project_root)
        if len(steps) <= 1:
            continue
        residual.extend(steps[1:])
    return residual
def focus_count_needs_split(focus_symbols: Sequence[str] | None) -> bool:
    names = [str(s).strip() for s in (focus_symbols or []) if str(s).strip()]
    return len(names) > MAX_FOCUS_SYMBOLS_PER_STEP


def context_span_too_large(
    context_window: Sequence[Mapping[str, Any]] | None,
    *,
    project_root: Path | None = None,
) -> bool:
    root = Path(project_root) if project_root is not None else None
    for item in context_window or []:
        if not isinstance(item, Mapping):
            continue
        span = item.get("span")
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        try:
            start, end = int(span[0]), int(span[1])
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        file_path = str(item.get("file") or "").strip()
        if root is not None and file_path:
            path = (root / file_path).resolve()
            try:
                if path.is_file():
                    line_count = len(
                        path.read_text(encoding="utf-8", errors="replace").splitlines()
                    )
                    if line_count > 0:
                        end = min(end, line_count)
            except OSError:
                pass
        if (end - start + 1) > MAX_CONTEXT_SPAN_LINES:
            return True
    return False


def suggest_split_hint(
    *,
    intent: str,
    focus_symbols: Sequence[str],
    target_file: str,
) -> str:
    """Concrete edit_plan example when Core must split an overloaded step."""
    symbols = [str(s).strip() for s in focus_symbols if str(s).strip()]
    path = str(target_file or "path.py").strip() or "path.py"
    if len(symbols) >= 2:
        steps = []
        for index, symbol in enumerate(symbols[:4]):
            steps.append(
                "{"
                f'"target_file":"{path}",'
                f'"intent":"<one change for {symbol}>",'
                f'"focus_symbols":["{symbol}"],'
                f'"context_window":[{{"file":"{path}","span":[<start>,<end>]}}]'
                "}"
            )
        body = ",\n  ".join(steps)
        return (
            "Split into more edit_plan steps (one coherent change each), e.g.\n"
            f"```edit_plan\n[\n  {body}\n]\n```"
        )
    snippet = (intent or "").strip()
    if len(snippet) > 80:
        snippet = snippet[:77] + "..."
    return (
        "Split into more edit_plan steps (one coherent change per step; "
        f"intent currently packs too much: {snippet!r})."
    )


def inspect_edit_brief_spec(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> str | None:
    """Core preflight: reject malformed / overloaded plan steps before EditLLM.

    All Core-facing fields are required (no optional fields). Returns an
    ErrorClass=E4_SPEC message (→ Core replan), or None when the step may proceed.
    """
    if tool_name != "decision_edit":
        return None

    target_file = str(arguments.get("target_file") or "").strip()
    intent = str(arguments.get("intent") or "").strip()
    focus_raw = arguments.get("focus_symbols") or []
    if not isinstance(focus_raw, list):
        focus_raw = []
    focus_symbols = [str(s).strip() for s in focus_raw if str(s).strip()]
    context_window = arguments.get("context_window")

    # Required-field checks (no optional fields).
    if not focus_symbols:
        return (
            "ErrorClass=E4_SPEC: plan step missing focus_symbols — list 1–"
            f"{MAX_FOCUS_SYMBOLS_PER_STEP} on-disk symbols from STEP EVIDENCE "
            "available_symbols."
        )
    if not isinstance(context_window, list) or not context_window:
        return (
            "ErrorClass=E4_SPEC: plan step missing context_window — pass ≥1 span "
            "covering focus_symbols (from LOADED CODE ANCHORS)."
        )

    if intent_needs_split(intent) and len(focus_symbols) < 2:
        # Single-focus but still overloaded prose — harness cannot auto-split by
        # symbol; ask Core to emit an edit_plan. Multi-focus overloads are expanded
        # in expand_decision_edit_to_steps before execute (not E4).
        if len(intent) > MAX_INTENT_CHARS:
            hint = suggest_split_hint(
                intent=intent,
                focus_symbols=focus_symbols,
                target_file=target_file,
            )
            return (
                "ErrorClass=E4_SPEC: intent is too long for one plan step "
                f"(>{MAX_INTENT_CHARS} chars) and has no focus_symbols to auto-split. "
                f"{hint}"
            )

    if len(focus_symbols) > 8:
        hint = suggest_split_hint(
            intent=intent,
            focus_symbols=focus_symbols,
            target_file=target_file,
        )
        return (
            "ErrorClass=E4_SPEC: focus_symbols has "
            f"{len(focus_symbols)} names (max 8 auto-split). {hint}"
        )

    if context_span_too_large(context_window, project_root=project_root):
        return (
            "ErrorClass=E4_SPEC: context_window span exceeds "
            f"{MAX_CONTEXT_SPAN_LINES} lines — pass a tight span covering "
            "focus_symbols only (from LOADED CODE ANCHORS), not 1..EOF."
        )

    if project_root is None or not target_file:
        return None

    from src.agent.edit_materialize import list_file_symbol_names

    disk_names = list_file_symbol_names(Path(project_root), target_file)
    # Non-Python / unparsed files: cannot validate focus against def/class inventory.
    if not disk_names:
        return None

    kept, dropped, remapped = sanitize_focus_symbols(
        Path(project_root),
        target_file,
        focus_symbols,
    )
    if not kept and dropped:
        disk_hint = (
            " On-disk symbols include: ["
            + ", ".join(disk_names[:MAX_AVAILABLE_SYMBOLS_LISTED])
            + "]. Pick from these (or omit focus and SITE an existing name)."
        )
        return (
            "ErrorClass=E4_SPEC: none of focus_symbols exist on disk "
            f"(dropped={dropped}). Do not invent names for helpers you plan to "
            f"add — insert them via MODE+ANCHOR under an existing symbol.{disk_hint}"
        )

    # Majority hallucination with only weak remaps still risks Edit SITE errors.
    if len(dropped) >= 2 and len(kept) <= 1 and not remapped:
        return (
            "ErrorClass=E4_SPEC: too many unknown focus_symbols "
            f"(dropped={dropped}, kept={kept}). Use only on-disk names from "
            "STEP EVIDENCE available_symbols / loaded anchors; new symbols go "
            "in REPLACE via insert_after, not in focus_symbols."
        )

    return None


def format_applied_diff_summary(
    original: str,
    attempted: str,
    *,
    target_file: str = "",
    max_hunk_headers: int = 6,
) -> str:
    """Compact Core-facing summary (+N -M + hunk headers); not a full diff."""
    orig_lines = (original or "").splitlines()
    new_lines = (attempted or "").splitlines()
    if orig_lines == new_lines:
        return "applied_diff_summary: no textual change"
    diff = list(
        difflib.unified_diff(
            orig_lines,
            new_lines,
            fromfile=f"a/{target_file}" if target_file else "a/file",
            tofile=f"b/{target_file}" if target_file else "b/file",
            lineterm="",
            n=0,
        )
    )
    added = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    headers = [line for line in diff if line.startswith("@@")][:max_hunk_headers]
    parts = [f"applied_diff_summary: [{target_file or 'file'}] +{added} -{removed}"]
    if headers:
        parts.append("hunks: " + "; ".join(headers))
    return "\n".join(parts)


def available_symbols_lines(
    project_root: Path | None,
    file_paths: Sequence[str],
    *,
    max_per_file: int = MAX_AVAILABLE_SYMBOLS_LISTED,
) -> list[str]:
    """STEP EVIDENCE lines listing on-disk symbols Core may put in focus_symbols."""
    if project_root is None or not file_paths:
        return []
    from src.agent.edit_materialize import list_file_symbol_names

    lines: list[str] = [
        "available_symbols (on-disk only — use these in focus_symbols; "
        "do not invent names for symbols you plan to add):"
    ]
    root = Path(project_root)
    any_listed = False
    for path in file_paths[:3]:
        names = list_file_symbol_names(root, path)
        if not names:
            continue
        any_listed = True
        shown = names[:max_per_file]
        suffix = "" if len(names) <= max_per_file else f" …(+{len(names) - max_per_file})"
        lines.append(f"- {path}: {', '.join(shown)}{suffix}")
    if not any_listed:
        return []
    return lines
