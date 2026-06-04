from __future__ import annotations

import asyncio
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class GrepSearchTool(Tool):
    name = "grep_search"
    description = "Search for a pattern in files using ripgrep. Supports regex."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current dir)",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py', '*.ts')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return",
                "default": 50,
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        pattern: str = validated["pattern"]
        path: str = validated.get("path", ".")
        include: str | None = validated.get("include")
        max_results: int = validated.get("max_results", 50)

        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        cmd += ["--max-count", str(max_results)]

        if include:
            cmd += ["--glob", include]

        cmd += ["--", pattern, path]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
        except FileNotFoundError:
            return ToolResult(
                success=False,
                output="",
                error="ripgrep (rg) not found. Install it: https://github.com/BurntSushi/ripgrep",
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(success=False, output="", error="Search timed out after 30s")

        output = stdout.decode(errors="replace").strip()
        err = stderr.decode(errors="replace").strip()

        if proc.returncode == 1:
            return ToolResult(success=True, output="No matches found.")
        if proc.returncode and proc.returncode > 1:
            return ToolResult(success=False, output="", error=f"rg error (exit {proc.returncode}): {err}")

        lines = output.splitlines()
        truncated = len(lines) >= max_results
        result_text = "\n".join(lines[:max_results])
        if truncated:
            result_text += f"\n\n[Results capped at {max_results} lines. Narrow your query or increase max_results.]"

        return ToolResult(
            success=True,
            output=result_text,
            metadata={"match_count": len(lines), "truncated": truncated},
        )
