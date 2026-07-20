from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


def _norm_path(path: str) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./")


_READ_ONLY_INTENT_RE = re.compile(
    r"(?:"
    r"查看|确认|理解|inspect|read\b|check\s+how|how\s+.+\s+mount|"
    r"如何|挂载.*确认|wiring|了解|弄清|完整内容|完整文件"
    r")",
    re.IGNORECASE,
)
_READ_LED_INTENT_RE = re.compile(
    r"^\s*(?:查看|确认|理解|inspect|check|read\b|如何|了解|弄清)",
    re.IGNORECASE,
)
_STRONG_EDIT_INTENT_RE = re.compile(
    r"(?:"
    r"\b(?:add|wrap|implement|apply|insert|modify|fix|define|create|update)\b|"
    r"添加|修改|实现|包装|装饰|定义|新增|更新"
    r")",
    re.IGNORECASE,
)
# Whole-file browsing masquerading as an edit (the actual abuse pattern): the
# model asks decision_edit to dump the entire file instead of using view.
_WHOLE_FILE_BROWSE_RE = re.compile(
    r"完整内容|完整文件|完整代码|整个文件|full\s+file|entire\s+file|whole\s+file",
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
    # A concrete mutation verb anywhere (add / wrap / 修改 / ...) means this is a
    # real edit, even when the intent opens with a read-led verb such as
    # "Inspect the handler and add validation". Strong-edit wins over read-led.
    if _STRONG_EDIT_INTENT_RE.search(intent):
        return False
    if _READ_LED_INTENT_RE.search(intent):
        return True
    if not _READ_ONLY_INTENT_RE.search(intent):
        return False
    return True


def inspect_decision_edit_intent(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    manifest: Any = None,
) -> str | None:
    """Block read-only or wrong-file decision_edit misuse.

    When Scheme B leaves only ``decision_edit`` open, Core must not smuggle a
    ``view`` through a read-led intent (even on the primary edit target).
    Decision LLM always generates a patch — it is not a file browser.
    """
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
            f"Use LOADED CODE ANCHORS (or view_symbol_code when allowed) for "
            f"inspection; apply edits with decision_edit("
            f"{edit_target!r}, intent=<concrete patch step>, focus_symbols=[...])."
        )

    focus_symbols = [
        str(symbol).strip()
        for symbol in (arguments.get("focus_symbols") or [])
        if str(symbol).strip()
    ]
    is_whole_file_browse = bool(_WHOLE_FILE_BROWSE_RE.search(intent))

    # On the primary edit target a concrete pre-edit probe (real focus_symbols,
    # e.g. "does table X exist / where to insert") is left to the DecisionLLM.
    # Only block the whole-file browse pattern, or a read-led intent with no
    # concrete focus to anchor an edit.
    if norm_edit and target_file and target_file == norm_edit:
        if not is_whole_file_browse and focus_symbols:
            return None

    inspect_target = target_file or norm_edit or "the file"
    return (
        "BLOCK: decision_edit is for applying patches, not reading code. "
        f"Evidence is already sufficient — use LOADED CODE ANCHORS for "
        f"{inspect_target!r} and call decision_edit only with a concrete "
        "patch intent (what to change), plus focus_symbols + tight context_window. "
        "Do not open a whole-file span just to '查看'."
    )
