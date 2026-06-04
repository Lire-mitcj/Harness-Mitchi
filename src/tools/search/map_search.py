from __future__ import annotations

from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.indexer.repo_map_service import RepoMapService
from src.tools.base import Tool


class MapSearchTool(Tool):
    name = "map_search"
    description = (
        "Search the project repo map (ctags/parser index + PageRank). "
        "Returns symbol names, file paths, line ranges, and signatures — "
        "faster than grep when locating classes, functions, SQL views, or procedures."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Symbol name, file path fragment, or keyword",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default 20)",
                "default": 20,
            },
        },
        "required": ["query"],
    }

    def __init__(self, repo_map_service: RepoMapService) -> None:
        self._service = repo_map_service

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        query: str = validated["query"]
        limit: int = validated.get("limit", 20)

        if not self._service.enabled:
            return ToolResult(
                success=False,
                output="",
                error="Repo map is disabled (MITKII_REPO_MAP_ENABLED=false).",
            )

        hits = self._service.search(query, limit=limit)
        if not hits:
            return ToolResult(
                success=True,
                output=f'No repo_map matches for "{query}". Try grep_search or glob_files.',
            )

        lines = [f'repo_map matches for "{query}" ({len(hits)}):']
        for sym in hits:
            sig = sym.signature.replace("\n", " ")[:100]
            sig_part = f"  {sig}" if sig else ""
            lines.append(
                f"- {sym.location}  {sym.kind} {sym.name}{sig_part}  "
                f"score={sym.score:.5f}"
            )
        return ToolResult(
            success=True,
            output="\n".join(lines),
            metadata={"match_count": len(hits), "query": query},
        )
