from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class GlobFilesTool(Tool):
    name = "glob_files"
    description = "Find files matching a glob pattern."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.ts')",
            },
            "path": {
                "type": "string",
                "description": "Root directory to search from (default: current dir)",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        pattern: str = validated["pattern"]
        root = Path(validated.get("path", ".")).expanduser().resolve()

        if not root.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {root}")
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {root}")

        try:
            matches = sorted(root.glob(pattern))
        except ValueError as exc:
            return ToolResult(success=False, output="", error=f"Invalid glob pattern: {exc}")

        MAX_RESULTS = 200
        truncated = len(matches) > MAX_RESULTS
        display = matches[:MAX_RESULTS]

        lines: list[str] = []
        for p in display:
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            suffix = "/" if p.is_dir() else ""
            lines.append(f"{rel}{suffix}")

        output = "\n".join(lines) if lines else "No files matched the pattern."
        if truncated:
            output += f"\n\n[Showing first {MAX_RESULTS} of {len(matches)} matches.]"

        return ToolResult(
            success=True,
            output=output,
            metadata={"total_matches": len(matches), "truncated": truncated},
        )
