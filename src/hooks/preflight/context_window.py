from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def inspect_context_window_disk(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    project_root: Path | None,
    manifest: Any = None,
) -> str | None:
    """Mechanical disk checks for decision_edit context_window spans only.

    Insertion anchors, new symbols, and patch placement are left to DecisionLLM.
    """
    _ = manifest
    if tool_name != "decision_edit" or project_root is None:
        return None

    context_window = arguments.get("context_window")
    if not context_window:
        return None

    for idx, item in enumerate(context_window):
        if not isinstance(item, dict):
            continue
        ref_file = _norm_path(str(item.get("file") or ""))
        ref_span = item.get("span")
        if not ref_file or not isinstance(ref_span, list) or len(ref_span) < 2:
            continue

        try:
            start_line = int(ref_span[0])
            end_line = int(ref_span[1])
        except (TypeError, ValueError):
            continue

        abs_path = (project_root / ref_file).resolve()
        if not abs_path.is_file():
            return (
                f"Invalid decision_edit: context_window index {idx} references "
                f"missing file {ref_file!r}."
            )

        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            return (
                f"Invalid decision_edit: unable to read context_window file "
                f"{ref_file!r}: {exc}"
            )

        line_count = len(lines)
        if start_line < 1 or end_line < 1 or start_line > end_line:
            return (
                f"Invalid decision_edit: context_window index {idx} has invalid "
                f"span [{start_line}, {end_line}]."
            )
        if start_line > line_count:
            return (
                f"Invalid decision_edit: context_window index {idx} span "
                f"[{start_line}, {end_line}] starts after end of {ref_file} "
                f"(file has {line_count} lines)."
            )

        effective_end = min(end_line, line_count)
        snippet = "\n".join(lines[start_line - 1 : effective_end])
        if not snippet.strip():
            return (
                f"Invalid decision_edit: context_window index {idx} span "
                f"[{start_line}, {end_line}] is empty in {ref_file}."
            )

        # end_line past EOF is allowed — decision_edit clips to file length when reading.

    return None
