from __future__ import annotations

import logging
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool

log = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry that maps tool names to their implementations."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            log.warning("Overwriting already-registered tool '%s'", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self, *, include: frozenset[str] | None = None) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function schemas for registered tools."""
        tools = self._tools.values()
        if include is not None:
            tools = [t for t in tools if t.name in include]
        return [tool.to_schema() for tool in tools]

    async def call(self, name: str, params: dict[str, Any]) -> ToolResult:
        """Look up a tool by name, validate params, and execute it."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output="",
                error=f"Unknown tool '{name}'. Available: {', '.join(sorted(self._tools))}",
            )

        try:
            validated = tool.validate_params(params)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        try:
            return await tool.execute(**validated)
        except Exception as exc:
            log.exception("Tool '%s' raised an unexpected error", name)
            return ToolResult(
                success=False,
                output="",
                error=f"Tool '{name}' failed: {type(exc).__name__}: {exc}",
            )

    def list_tools(self) -> list[tuple[str, str, RiskLevel]]:
        """Return (name, description, risk_level) for every registered tool."""
        return [
            (t.name, t.description, t.risk_level)
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


def create_default_registry(
    *,
    repo_map_service: object | None = None,
) -> ToolRegistry:
    """Build a registry pre-loaded with all built-in tools."""
    from src.tools.file.delete import DeleteFileTool
    from src.tools.file.replace_symbol import ReplaceSymbolTool
    from src.tools.file.edit import EditFileTool
    from src.tools.file.read import ReadFileTool, ReadFilesTool
    from src.tools.file.write import WriteFileTool
    from src.tools.git.commit import GitCommitTool
    from src.tools.git.stash import GitStashTool
    from src.tools.git.status import GitStatusTool
    from src.tools.search.glob import GlobFilesTool
    from src.tools.search.grep import GrepSearchTool
    from src.tools.search.list_dir import ListDirTool
    from src.tools.search.map_search import MapSearchTool
    from src.tools.shell.executor import ShellExecTool

    registry = ToolRegistry()
    for tool_cls in (
        ReadFileTool,
        ReadFilesTool,
        WriteFileTool,
        EditFileTool,
        ReplaceSymbolTool,
        DeleteFileTool,
        GrepSearchTool,
        GlobFilesTool,
        ListDirTool,
        ShellExecTool,
        GitStatusTool,
        GitCommitTool,
        GitStashTool,
    ):
        registry.register(tool_cls())

    if repo_map_service is not None:
        registry.register(MapSearchTool(repo_map_service))

    log.debug("Default registry created with %d tools", len(registry))
    return registry
