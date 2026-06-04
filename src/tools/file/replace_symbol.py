from __future__ import annotations

from pathlib import Path

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class ReplaceSymbolTool(Tool):
    """Schema-only; execution is handled by Harness splice apply in tool_pipeline."""

    name = "replace_symbol"
    description = (
        "Replace a diagnosed symbol span with a full new function/block body. "
        "Harness resolves line boundaries and verifies anchor hash before splicing."
    )
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path from EditTarget (e.g. app.py)",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol name from EditTarget (must match handoff)",
            },
            "new_body": {
                "type": "string",
                "description": (
                    "Complete revised function/block including def/header lines; "
                    "must differ from the original in <edit_target>."
                ),
            },
        },
        "required": ["path", "symbol", "new_body"],
    }

    async def execute(self, **params: object) -> ToolResult:
        _ = params
        return ToolResult(
            success=False,
            output="",
            error="replace_symbol must be executed via Harness tool_pipeline.",
        )
