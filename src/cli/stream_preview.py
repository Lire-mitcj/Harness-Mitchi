from __future__ import annotations

import re

_JSON_NOISE = re.compile(r"^[\s{\"`\[\]:,0-9a-z_-]+$", re.I)
_ROOT_TASK = re.compile(r'"root_task"\s*:\s*"([^"]{1,80})"', re.I)
_KIND = re.compile(r'"kind"\s*:\s*"(diagnose|edit|verify|shell)"', re.I)


def planner_status_hint(partial: str) -> str | None:
    """Human-readable Planner progress from partial JSON (never show raw JSON)."""
    if not partial or len(partial) < 8:
        return None
    kinds = _KIND.findall(partial)
    if kinds:
        uniq = []
        for k in kinds:
            if k not in uniq:
                uniq.append(k)
        return f"{' → '.join(uniq)} ({len(uniq)} step{'s' if len(uniq) != 1 else ''})"
    match = _ROOT_TASK.search(partial)
    if match:
        task = match.group(1).replace("\\n", " ").strip()
        if task:
            return f"task: {task[:60]}"
    if partial.count("{") and len(partial) > 12:
        return "writing TaskTree JSON…"
    return None


def executor_reasoning_preview(text: str, *, max_len: int = 120) -> str:
    """One-line executor intent before tool calls — skip markdown / JSON noise."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or len(line) < 4:
            continue
        if line.startswith(("#", "|", "```", "{", "[")):
            continue
        if _JSON_NOISE.match(line.replace(" ", "")):
            continue
        if any(tok in line.lower() for tok in ('"kind"', '"nodes"', "root_task")):
            continue
        one = " ".join(line.split())
        if len(one) > max_len:
            return one[: max_len - 1] + "…"
        return one
    return ""


def executor_stream_hint(partial: str, *, max_len: int = 120) -> str:
    """One-line gray status while a long executor answer is generating."""
    stripped = (partial or "").strip()
    if not stripped:
        return ""
    for line in stripped.splitlines():
        s = line.strip()
        if s.startswith("#"):
            title = re.sub(r"^#+\s*", "", s).strip()
            if title:
                one = " ".join(title.split())
                if len(one) > max_len:
                    return one[: max_len - 1] + "…"
                return one
    preview = executor_reasoning_preview(stripped)
    if preview:
        return preview
    return f"writing response ({len(stripped)} chars)…"


def should_stream_executor_answer(text: str, *, has_tool_calls: bool) -> bool:
    """Final-answer turns should not stream to CLI (shown once at subtask done)."""
    if has_tool_calls:
        return True
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.startswith("#") or "## " in stripped[:80]:
        return False
    if "| " in stripped and "---" in stripped:
        return False
    return len(stripped) < 160
