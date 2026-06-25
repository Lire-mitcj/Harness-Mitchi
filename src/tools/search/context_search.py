from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.types import RiskLevel, ToolResult
from src.skills import CodeSearchSkill, SkillContext
from src.tools.base import Tool

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


class ContextSearchTool(Tool):
    name = "context_search"
    description = (
        "Ask Harness to find relevant code context. Provide what you need; Harness "
        "uses repo_map, grep, and bounded snippets. Prefer this over read_file, "
        "read_files, grep_search, or map_search."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to locate or verify in the codebase.",
            },
            "need": {
                "type": "string",
                "description": (
                    "The evidence needed, e.g. file:line, symbol, SQL snippet, "
                    "caller, or exact edit target."
                ),
            },
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional file or directory hints to constrain search.",
            },
            "max_results": {
                "type": "integer",
                "description": "Soft result cap for the search backend.",
                "default": 80,
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, project_root: Path, tools: ToolRegistry) -> None:
        self.project_root = project_root.resolve()
        self._search = CodeSearchSkill(project_root=self.project_root, tools=tools)

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        query = str(validated.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output="", error="query must not be empty")
        need = str(validated.get("need") or "").strip()
        paths = [
            str(path).strip().replace("\\", "/").lstrip("./")
            for path in (validated.get("paths") or [])
            if str(path).strip()
        ]
        scope = f"\nScope paths: {', '.join(paths)}" if paths else ""
        extra_query = f"{query}\nNeed: {need or 'relevant file:line, symbol, and snippet'}{scope}"
        result = await self._search.run(
            SkillContext(user_request=extra_query),
            extra_query=extra_query,
            search_query=query,
            search_paths=tuple(paths),
        )
        output = result.metadata.get("search_summary", result.metadata.get("search_output", result.summary))
        if not result.success:
            return ToolResult(success=False, output=output, error=result.summary)
        return ToolResult(success=True, output=output)
