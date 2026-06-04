from __future__ import annotations

from pathlib import Path

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool

_SUSPICIOUS_OVERWRITE_MIN_EXISTING_LINES = 10
_SUSPICIOUS_OVERWRITE_RATIO = 0.4


class WriteFileTool(Tool):
    name = "write_file"
    description = "Create or overwrite a file with the given content."
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to create/overwrite"},
            "content": {"type": "string", "description": "Full content to write to the file"},
        },
        "required": ["path", "content"],
    }

    async def execute(self, **params: object) -> ToolResult:
        validated = self.validate_params(params)
        file_path = Path(validated["path"]).expanduser().resolve()
        content: str = validated["content"]

        existed = file_path.exists()

        try:
            if existed:
                existing = file_path.read_text(encoding="utf-8")
                denial = _suspicious_partial_overwrite_reason(
                    existing=existing,
                    replacement=content,
                    path=file_path,
                )
                if denial:
                    return ToolResult(success=False, output="", error=denial)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            action = "Overwrote" if existed else "Created"

            return ToolResult(
                success=True,
                output=f"{action} {file_path} ({line_count} lines, {len(content)} bytes)",
                metadata={"path": str(file_path), "lines": line_count, "created": not existed},
            )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {file_path}")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"IO error writing {file_path}: {exc}")


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


def _suspicious_partial_overwrite_reason(
    *,
    existing: str,
    replacement: str,
    path: Path,
) -> str | None:
    """Reject common LLM failure mode: write_file(existing, small edited snippet)."""
    existing_lines = _line_count(existing)
    replacement_lines = _line_count(replacement)
    if existing_lines < _SUSPICIOUS_OVERWRITE_MIN_EXISTING_LINES:
        return None
    if replacement_lines >= max(3, int(existing_lines * _SUSPICIOUS_OVERWRITE_RATIO)):
        return None
    if not replacement.strip():
        return None
    if replacement.strip() in existing:
        reason = "content is a snippet from the existing file"
    else:
        reason = (
            f"replacement is only {replacement_lines} line(s) for an existing "
            f"{existing_lines}-line file"
        )
    return (
        f"Refusing suspicious write_file overwrite for {path}: {reason}. "
        "Use edit_file for partial changes, or call write_file with the complete "
        "file content if a full rewrite is intended."
    )
