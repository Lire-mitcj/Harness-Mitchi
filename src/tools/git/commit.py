from __future__ import annotations

import asyncio
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


async def _run_git(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace").strip(),
        stderr.decode(errors="replace").strip(),
    )


class GitCommitTool(Tool):
    name = "git_commit"
    description = "Stage files and create a git commit."
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message"},
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Files to stage before committing. If empty, commits whatever is already staged.",
            },
        },
        "required": ["message"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        message: str = validated["message"]
        files: list[str] = validated.get("files", [])

        if files:
            rc, out, err = await _run_git("add", "--", *files)
            if rc != 0:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"git add failed: {err or out}",
                )

        rc, out, err = await _run_git("diff", "--cached", "--quiet")
        if rc == 0:
            return ToolResult(
                success=False,
                output="",
                error="Nothing staged to commit. Stage files first or pass `files` parameter.",
            )

        rc, out, err = await _run_git("commit", "-m", message)
        if rc != 0:
            return ToolResult(success=False, output="", error=f"git commit failed: {err or out}")

        return ToolResult(success=True, output=out)
