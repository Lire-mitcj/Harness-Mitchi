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


class GitStashTool(Tool):
    name = "git_stash"
    description = "Manage git stash: save, pop, or list stashed changes."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["save", "pop", "list"],
                "description": "Stash operation to perform",
            },
            "message": {
                "type": "string",
                "description": "Optional message when saving a stash",
            },
        },
        "required": ["operation"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        operation: str = validated["operation"]
        message: str | None = validated.get("message")

        match operation:
            case "save":
                args = ["stash", "push"]
                if message:
                    args += ["-m", message]
                rc, out, err = await _run_git(*args)
            case "pop":
                rc, out, err = await _run_git("stash", "pop")
            case "list":
                rc, out, err = await _run_git("stash", "list")
            case _:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Unknown operation '{operation}'. Use: save, pop, list",
                )

        if rc != 0:
            return ToolResult(success=False, output="", error=f"git stash {operation} failed: {err or out}")

        return ToolResult(success=True, output=out or f"git stash {operation} completed (no output)")
