from __future__ import annotations

from pathlib import Path

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class DeleteFileTool(Tool):
    name = "delete_file"
    description = "Delete a file at the specified path."
    risk_level = RiskLevel.DANGEROUS
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to delete"},
        },
        "required": ["path"],
    }

    async def execute(self, **params: object) -> ToolResult:
        validated = self.validate_params(params)
        file_path = Path(validated["path"]).expanduser().resolve()

        if not file_path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")
        if not file_path.is_file():
            return ToolResult(
                success=False,
                output="",
                error=f"Not a regular file (refusing to delete): {file_path}",
            )

        try:
            size = file_path.stat().st_size
            file_path.unlink()
            return ToolResult(
                success=True,
                output=f"Deleted {file_path} ({size:,} bytes)",
                metadata={"path": str(file_path), "size": size},
            )
        except PermissionError:
            return ToolResult(success=False, output="", error=f"Permission denied: {file_path}")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"Failed to delete {file_path}: {exc}")
