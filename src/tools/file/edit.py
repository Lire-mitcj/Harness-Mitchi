from __future__ import annotations

from pathlib import Path

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing an exact string match. "
        "old_string must appear exactly once in the file; include enough "
        "surrounding lines to make the match unique. old_string and new_string "
        "must differ."
    )
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit"},
            "old_string": {
                "type": "string",
                "description": (
                    "Exact multi-line snippet to find (copy from read_file/preload; "
                    "include surrounding lines so the match is unique)"
                ),
            },
            "new_string": {
                "type": "string",
                "description": "Replacement snippet (must differ from old_string)",
            },
        },
        "required": ["path", "old_string", "new_string"],
    }

    async def execute(self, **params: object) -> ToolResult:
        validated = self.validate_params(params)
        file_path = Path(validated["path"]).expanduser().resolve()
        old_string: str = validated["old_string"]
        new_string: str = validated["new_string"]

        if old_string == new_string:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "old_string and new_string are identical — provide the actual "
                    "code change (copy old lines from the file, then supply the "
                    "modified version as new_string)."
                ),
            )

        if not file_path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")
        if not file_path.is_file():
            return ToolResult(success=False, output="", error=f"Not a file: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"IO error reading {file_path}: {exc}")

        occurrences = content.count(old_string)
        if occurrences == 0:
            hint = _not_found_hint(content, old_string)
            return ToolResult(
                success=False,
                output="",
                error=f"old_string not found in {file_path}. {hint}",
            )
        if occurrences > 1:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"old_string appears {occurrences} times in {file_path}. "
                    "Provide a more unique string (include surrounding context) "
                    "so the match is unambiguous."
                ),
            )

        new_content = content.replace(old_string, new_string, 1)

        try:
            file_path.write_text(new_content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"IO error writing {file_path}: {exc}")

        old_lines = old_string.count("\n") + 1
        new_lines = new_string.count("\n") + 1
        delta = new_lines - old_lines
        delta_str = f"+{delta}" if delta > 0 else str(delta)

        return ToolResult(
            success=True,
            output=(
                f"Edited {file_path}: replaced {old_lines} lines with {new_lines} lines "
                f"({delta_str} net). File now has {new_content.count(chr(10)) + 1} total lines."
            ),
            metadata={"path": str(file_path)},
        )


def _not_found_hint(content: str, old_string: str) -> str:
    """Actionable hint when the model sends a too-short or mismatched old_string."""
    if old_string.count("\n") >= 2:
        return (
            "Double-check whitespace and exact content — copy from read_file or the "
            "preloaded <file> block."
        )
    needle = old_string.strip()
    if not needle:
        return "old_string is empty — provide a multi-line snippet from the file."
    for i, line in enumerate(content.splitlines(), start=1):
        if needle in line or line.strip() == needle:
            return (
                f"Your old_string looks too short. Near line {i}: {line.strip()[:100]!r} — "
                "copy ≥3 surrounding lines verbatim into old_string, then apply your change "
                "in new_string."
            )
    return (
        "Use read_file or the preloaded <file> slice — copy a multi-line block verbatim, "
        "not just a symbol name."
    )
