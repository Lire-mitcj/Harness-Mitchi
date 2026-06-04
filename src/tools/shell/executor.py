from __future__ import annotations

import asyncio
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool

BLOCKED_PATTERNS = frozenset({
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=/dev/zero",
    ":(){:|:&};:",
})


class ShellExecTool(Tool):
    name = "shell_exec"
    description = "Execute a shell command and return its output."
    risk_level = RiskLevel.DANGEROUS
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 30)",
                "default": 30,
            },
            "working_dir": {
                "type": "string",
                "description": "Working directory for the command",
            },
        },
        "required": ["command"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        command: str = validated["command"]
        timeout: int = validated.get("timeout", 30)
        working_dir: str | None = validated.get("working_dir")

        for pattern in BLOCKED_PATTERNS:
            if pattern in command:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"Command blocked for safety: contains '{pattern}'",
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=working_dir,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {timeout}s. Consider increasing the timeout.",
            )
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"Failed to execute command: {exc}")

        stdout_text = stdout.decode(errors="replace").strip()
        stderr_text = stderr.decode(errors="replace").strip()

        MAX_OUTPUT = 50_000
        if len(stdout_text) > MAX_OUTPUT:
            stdout_text = stdout_text[:MAX_OUTPUT] + f"\n\n[Output truncated at {MAX_OUTPUT} chars]"

        output_parts: list[str] = []
        if stdout_text:
            output_parts.append(stdout_text)
        if stderr_text:
            output_parts.append(f"[stderr]\n{stderr_text}")

        combined = "\n\n".join(output_parts) if output_parts else "(no output)"

        no_match_search = _is_no_match_search_result(command, proc.returncode, stderr_text)
        if no_match_search and combined == "(no output)":
            combined = "(no matches)"

        return ToolResult(
            success=proc.returncode == 0 or no_match_search,
            output=combined,
            error=f"Exit code {proc.returncode}" if proc.returncode and not no_match_search else None,
            metadata={
                "exit_code": proc.returncode,
                "command": command,
                "no_match_search": no_match_search,
            },
        )


def _is_search_command(command: str) -> bool:
    """grep/rg return 1 for no matches; that is useful evidence, not tool failure."""
    normalized = command.strip()
    return (
        normalized.startswith(("grep ", "rg "))
        or " grep " in normalized
        or " rg " in normalized
    )


def _is_no_match_search_result(command: str, exit_code: int | None, stderr: str) -> bool:
    return exit_code == 1 and not stderr and _is_search_command(command)
