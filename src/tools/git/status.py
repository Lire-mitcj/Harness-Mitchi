from __future__ import annotations

import asyncio
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


async def _run_git(*args: str, cwd: str | None = None) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace").strip(),
        stderr.decode(errors="replace").strip(),
    )


class GitStatusTool(Tool):
    name = "git_status"
    description = "Show git status, diff, and recent log."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "detailed": {
                "type": "boolean",
                "description": "Include git diff and recent log (default: false)",
                "default": False,
            },
        },
        "required": [],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        detailed: bool = validated.get("detailed", False)

        rc, status_out, status_err = await _run_git("status", "--short", "--branch")
        if rc != 0:
            return ToolResult(
                success=False,
                output="",
                error=f"git status failed: {status_err or status_out}",
            )

        sections = [f"=== git status ===\n{status_out}"]

        if detailed:
            _, diff_out, _ = await _run_git("diff", "--stat")
            if diff_out:
                sections.append(f"=== git diff --stat ===\n{diff_out}")

            _, log_out, _ = await _run_git(
                "log", "--oneline", "--no-decorate", "-10",
            )
            if log_out:
                sections.append(f"=== recent commits ===\n{log_out}")

        return ToolResult(success=True, output="\n\n".join(sections))
