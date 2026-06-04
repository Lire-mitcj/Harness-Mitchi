from __future__ import annotations

import json
import re

from src.agent.types import ToolCall

_DSML_BLOCK = re.compile(
    r"<｜DSML｜tool_calls>.*?</｜DSML｜tool_calls>",
    re.DOTALL,
)
_DSML_INVOKE = re.compile(
    r"<｜DSML｜invoke\s+name=\"(?P<name>[^\"]+)\".*?</｜DSML｜invoke>",
    re.DOTALL,
)
_DSML_PARAM = re.compile(
    r"<｜DSML｜parameter\s+name=\"(?P<name>[^\"]+)\"[^>]*>"
    r"(?P<body>.*?)"
    r"</｜DSML｜parameter>",
    re.DOTALL,
)
_DSML_ANY_TAG = re.compile(r"</?｜DSML｜[^>]*>", re.DOTALL)


def split_dsml_tool_calls(text: str | None) -> tuple[str | None, list[ToolCall]]:
    """Extract text-protocol tool calls emitted as content by some providers."""
    if not text or "DSML" not in text:
        return text, []

    calls: list[ToolCall] = []
    for block in _DSML_BLOCK.findall(text):
        for match in _DSML_INVOKE.finditer(block):
            name = match.group("name").strip()
            args: dict = {}
            body = match.group(0)
            for param in _DSML_PARAM.finditer(body):
                param_name = param.group("name")
                raw = param.group("body").strip()
                parsed = _parse_jsonish(raw)
                if param_name == "arguments" and isinstance(parsed, dict):
                    args.update(parsed)
                else:
                    args[param_name] = parsed
            if name:
                calls.append(ToolCall(
                    id=f"dsml_{len(calls) + 1}",
                    name=name,
                    arguments=args,
                ))

    cleaned = _DSML_BLOCK.sub("", text)
    cleaned = re.sub(r"<｜DSML｜invoke\s+name=\"[^\"]+\">\s*", "", cleaned)
    cleaned = cleaned.replace("</｜DSML｜invoke>", "")
    cleaned = _DSML_ANY_TAG.sub("", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None, calls


def strip_dsml_text(text: str | None) -> str:
    cleaned, _ = split_dsml_tool_calls(text)
    return cleaned or ""


def _parse_jsonish(raw: str) -> object:
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
