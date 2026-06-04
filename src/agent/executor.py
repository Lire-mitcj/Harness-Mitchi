from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.agent.types import ToolResult
from src.tools.base import Tool
from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 120.0


class ToolExecutor:
    """Wraps tool execution with error handling, timeout, and logging.

    Sits between the agent loop and the raw :class:`ToolRegistry`, adding
    consistent timing, structured error messages, and per-call metadata.
    """

    def __init__(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    async def execute(
        self,
        tool_name: str,
        params: dict[str, Any],
        registry: ToolRegistry,
        sandbox: Any | None = None,
    ) -> ToolResult:
        """Look up and execute a tool with timeout and error wrapping.

        If *sandbox* is provided and supports ``run_in_sandbox``, the tool
        execution is delegated to it (for dangerous operations).
        """
        tool = registry.get(tool_name)
        if tool is None:
            available = ", ".join(t[0] for t in registry.list_tools())
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{tool_name}'. Available: {available}",
            )

        try:
            validated = tool.validate_params(params)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        start = time.monotonic()
        try:
            if sandbox is not None and hasattr(sandbox, "run_in_sandbox"):
                result = await asyncio.wait_for(
                    sandbox.run_in_sandbox(tool, validated),
                    timeout=self.timeout,
                )
            else:
                result = await asyncio.wait_for(
                    tool.execute(**validated),
                    timeout=self.timeout,
                )
        except asyncio.TimeoutError:
            elapsed = time.monotonic() - start
            log.warning("Tool '%s' timed out after %.1fs", tool_name, elapsed)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' timed out after {self.timeout:.0f}s",
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            log.exception("Tool '%s' raised after %.2fs", tool_name, elapsed)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{tool_name}' failed: {type(exc).__name__}: {exc}",
            )

        elapsed = time.monotonic() - start
        log.debug(
            "Tool '%s' completed in %.2fs (success=%s)",
            tool_name,
            elapsed,
            result.success,
        )
        if result.metadata is None:
            result = ToolResult(
                success=result.success,
                output=result.output,
                error=result.error,
                metadata={"elapsed_s": round(elapsed, 3)},
            )
        else:
            result.metadata["elapsed_s"] = round(elapsed, 3)

        return result
