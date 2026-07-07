from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


_READ_ONLY_INTENT_RE = re.compile(
    r"(?:"
    r"查看|确认|理解|inspect|read\b|check\s+how|how\s+.+\s+mount|"
    r"如何|挂载.*确认|wiring|了解|弄清"
    r")",
    re.IGNORECASE,
)
_READ_LED_INTENT_RE = re.compile(
    r"^\s*(?:查看|确认|理解|inspect|check|read\b|如何|了解|弄清)",
    re.IGNORECASE,
)
_STRONG_EDIT_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:add|wrap|implement|apply|insert|modify|fix|define|create)\b|"
    r"添加|修改|实现|包装|装饰|定义|新增"
    r")",
    re.IGNORECASE,
)


def _edit_target_file(manifest: Any) -> str | None:
    if manifest is None:
        return None
    try:
        from src.agent.manifest import primary_edit_target_files

        targets = primary_edit_target_files(manifest, "")
        if targets:
            return targets[0]
    except Exception:
        return None
    return None


def _is_read_only_edit_intent(intent: str) -> bool:
    if _READ_LED_INTENT_RE.search(intent):
        return True
    if not _READ_ONLY_INTENT_RE.search(intent):
        return False
    return not bool(_STRONG_EDIT_INTENT_RE.search(intent))


def inspect_decision_edit_intent(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    manifest: Any = None,
) -> str | None:
    """Block read-only or wrong-file decision_edit misuse."""
    if tool_name != "decision_edit":
        return None

    intent = str(arguments.get("intent") or "").strip()
    if not intent or not _is_read_only_edit_intent(intent):
        return None

    target_file = _norm_path(str(arguments.get("target_file") or ""))
    edit_target = _edit_target_file(manifest)
    norm_edit = _norm_path(edit_target) if edit_target else ""

    if norm_edit and target_file and target_file != norm_edit:
        return (
            "BLOCK: decision_edit is for applying patches, not reading code. "
            f"Use view_symbol_code or LOADED CODE ANCHORS for inspection; "
            f"apply edits with decision_edit("
            f"{edit_target!r}, intent=<patch step>, focus_symbols=[...])."
        )

    # Read-led probe on the primary edit target is left to DecisionLLM (e.g. schema checks).
    if norm_edit and target_file and target_file == norm_edit:
        return None

    inspect_target = target_file or norm_edit or "the file"
    return (
        "BLOCK: decision_edit is for applying patches, not reading code. "
        f"Use view_symbol_code or LOADED CODE ANCHORS to inspect {inspect_target!r}; "
        "call decision_edit only with a concrete patch intent "
        "(focus_symbols + context_window)."
    )
