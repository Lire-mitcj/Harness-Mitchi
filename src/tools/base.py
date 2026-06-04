from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from src.agent.types import RiskLevel, ToolCall, ToolResult


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses must set the class-level ``name``, ``description``, and
    ``parameters`` attributes, and implement ``execute``.
    """

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema describing accepted params
    risk_level: RiskLevel = RiskLevel.SAFE

    @abstractmethod
    async def execute(self, **params: Any) -> ToolResult:
        ...

    def to_schema(self) -> dict[str, Any]:
        """Return the OpenAI-compatible function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Basic validation against the JSON Schema ``parameters`` spec.

        Checks required fields are present and coerces simple types.  This is
        intentionally lightweight — the LLM is expected to produce well-formed
        arguments most of the time.
        """
        schema_props: dict[str, Any] = self.parameters.get("properties", {})
        required: list[str] = self.parameters.get("required", [])

        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter '{key}' for tool '{self.name}'")

        validated: dict[str, Any] = {}
        for key, value in params.items():
            if key not in schema_props:
                continue  # silently drop unknown params
            prop = schema_props[key]
            expected_type = prop.get("type")
            validated[key] = _coerce(value, expected_type)

        return validated


def _coerce(value: Any, expected_type: str | None) -> Any:
    """Best-effort type coercion for JSON-decoded values."""
    if expected_type is None or value is None:
        return value
    try:
        match expected_type:
            case "integer":
                return int(value)
            case "number":
                return float(value)
            case "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes")
                return bool(value)
            case "string":
                return str(value)
            case "array":
                if isinstance(value, str):
                    return json.loads(value)
                return value
            case _:
                return value
    except (ValueError, TypeError, json.JSONDecodeError):
        return value
