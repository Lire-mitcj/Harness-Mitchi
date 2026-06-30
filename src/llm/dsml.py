from __future__ import annotations

import json
import re
from collections.abc import Set
from html import unescape
from typing import Any

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
_XML_TOOL_BLOCK = re.compile(
    r"<(?P<tag>tool_calls|tool_results)>.*?</(?P=tag)>",
    re.DOTALL | re.IGNORECASE,
)
_XML_INVOKE = re.compile(
    r"<invoke\s+name=[\"'](?P<name>[^\"']+)[\"'][^>]*>(?P<body>.*?)</invoke>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAMETER = re.compile(
    r"<parameter\s+name=[\"'](?P<name>[^\"']+)[\"'][^>]*>"
    r"(?P<body>.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)
_XML_TOOL_CALL = re.compile(
    r"<tool_call>(?P<body>.*?)</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
_XML_FUNCTION = re.compile(
    r"<function>(?P<body>.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
_XML_TOOL_NAME = re.compile(
    r"<tool_name>(?P<name>.*?)</tool_name>",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAMETERS = re.compile(
    r"<parameters>(?P<body>.*?)</parameters>",
    re.DOTALL | re.IGNORECASE,
)
_XML_ARGUMENT = re.compile(
    r"<(?P<name>[A-Za-z_][A-Za-z0-9_]*)>(?P<body>.*?)</(?P=name)>",
    re.DOTALL,
)
# Bare XML: <tool_name><param>val</param>...</tool_name> (DeepSeek style)
_BARE_XML_TOOL = re.compile(
    r"<(?P<tool>[A-Za-z_][A-Za-z0-9_]*)>(?P<body>.*?)</(?P=tool)>",
    re.DOTALL,
)


def split_dsml_tool_calls(
    text: str | None,
    *,
    known_tool_names: Set[str] | None = None,
) -> tuple[str | None, list[ToolCall]]:
    """Extract text-protocol tool calls emitted as content by some providers.

    When *known_tool_names* is supplied the parser also recognises "bare XML"
    tool calls emitted by models such as DeepSeek, e.g.
    ``<grep_search><pattern>foo</pattern></grep_search>``.
    """
    if not text:
        return text, []
    # This function is also called for every streaming delta. Preserve ordinary
    # chunk whitespace byte-for-byte or token boundaries such as ``" the"``
    # collapse into the previous word when the UI concatenates deltas.
    lowered = text.casefold()
    has_standard_markers = "dsml" in lowered or any(
        marker in lowered
        for marker in (
            "<tool_calls",
            "<tool_results",
            "<invoke name=",
            "<tool_call>",
            "<function>",
        )
    )
    # Detect bare XML tool tags: <tool_name> where tool_name is a known tool
    has_bare_xml = False
    if not has_standard_markers and known_tool_names:
        for name in known_tool_names:
            if f"<{name.casefold()}>" in lowered:
                has_bare_xml = True
                break
    if not has_standard_markers and not has_bare_xml:
        return text, []

    calls: list[ToolCall] = []
    for block in _DSML_BLOCK.findall(text):
        for match in _DSML_INVOKE.finditer(block):
            name = match.group("name").strip()
            dsml_args: dict[str, Any] = {}
            body = match.group(0)
            for param in _DSML_PARAM.finditer(body):
                param_name = param.group("name")
                raw = param.group("body").strip()
                parsed = _parse_jsonish(raw)
                if param_name == "arguments" and isinstance(parsed, dict):
                    dsml_args.update(parsed)
                else:
                    dsml_args[param_name] = parsed
            if name:
                calls.append(ToolCall(
                    id=f"dsml_{len(calls) + 1}",
                    name=name,
                    arguments=dsml_args,
                ))

    for block_match in _XML_TOOL_BLOCK.finditer(text):
        block = block_match.group(0)
        for match in _XML_INVOKE.finditer(block):
            invoke_args = {
                param.group("name").strip(): _parse_jsonish(
                    unescape(param.group("body").strip())
                )
                for param in _XML_PARAMETER.finditer(match.group("body"))
            }
            name = unescape(match.group("name").strip())
            if name:
                calls.append(
                    ToolCall(
                        id=f"xml_{len(calls) + 1}",
                        name=name,
                        arguments=invoke_args,
                    )
                )
        for match in _XML_TOOL_CALL.finditer(block):
            body = match.group("body")
            name_match = _XML_TOOL_NAME.search(body)
            params_match = _XML_PARAMETERS.search(body)
            if not name_match:
                continue
            call_args: dict[str, Any] = {}
            if params_match:
                for arg in _XML_ARGUMENT.finditer(params_match.group("body")):
                    call_args[arg.group("name")] = _parse_jsonish(
                        unescape(arg.group("body").strip())
                    )
            calls.append(
                ToolCall(
                    id=f"xml_{len(calls) + 1}",
                    name=unescape(name_match.group("name").strip()),
                    arguments=call_args,
                )
            )

    # Some OpenAI-compatible providers omit the plural <tool_calls> wrapper.
    # Only use this fallback when the wrapped parser found nothing, avoiding
    # duplicate calls for well-formed payloads.
    if not calls:
        for match in _XML_INVOKE.finditer(text):
            invoke_args = {
                param.group("name").strip(): _parse_jsonish(
                    unescape(param.group("body").strip())
                )
                for param in _XML_PARAMETER.finditer(match.group("body"))
            }
            name = unescape(match.group("name").strip())
            if name:
                calls.append(ToolCall(
                    id=f"xml_{len(calls) + 1}",
                    name=name,
                    arguments=invoke_args,
                ))

    if not calls:
        for match in _XML_FUNCTION.finditer(text):
            body = match.group("body")
            name_match = _XML_TOOL_NAME.search(body)
            params_match = _XML_PARAMETERS.search(body)
            if not name_match:
                continue
            call_args = {
                arg.group("name"): _parse_jsonish(unescape(arg.group("body").strip()))
                for arg in _XML_ARGUMENT.finditer(
                    params_match.group("body") if params_match else ""
                )
            }
            calls.append(ToolCall(
                id=f"xml_{len(calls) + 1}",
                name=unescape(name_match.group("name").strip()),
                arguments=call_args,
            ))

    # --- Bare XML tool calls (DeepSeek style) ---
    if not calls and known_tool_names:
        known_tool_map = {name.lower(): name for name in known_tool_names}
        for match in _BARE_XML_TOOL.finditer(text):
            tool_name = match.group("tool").lower()
            if tool_name not in known_tool_map:
                continue
            orig_name = known_tool_map[tool_name]
            body = match.group("body")
            bare_args: dict[str, Any] = {}
            for arg in _XML_ARGUMENT.finditer(body):
                bare_args[arg.group("name")] = _parse_jsonish(
                    unescape(arg.group("body").strip())
                )
            calls.append(
                ToolCall(
                    id=f"barexml_{len(calls) + 1}",
                    name=orig_name,
                    arguments=bare_args,
                )
            )

    cleaned = _DSML_BLOCK.sub("", text)
    cleaned = re.sub(r"<｜DSML｜invoke\s+name=\"[^\"]+\">\s*", "", cleaned)
    cleaned = cleaned.replace("</｜DSML｜invoke>", "")
    cleaned = _DSML_ANY_TAG.sub("", cleaned)
    if calls:
        cleaned = _XML_TOOL_BLOCK.sub("", cleaned)
        cleaned = _XML_TOOL_CALL.sub("", cleaned)
        cleaned = _XML_FUNCTION.sub("", cleaned)
        cleaned = _XML_INVOKE.sub("", cleaned)
        has_barexml = any(tc.id.startswith("barexml_") for tc in calls)
        if has_barexml:
            for tc in calls:
                if tc.id.startswith("barexml_"):
                    cleaned = re.sub(
                        rf"<{re.escape(tc.name)}>.*?</{re.escape(tc.name)}>",
                        "",
                        cleaned,
                        flags=re.DOTALL,
                    )
            cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL)
    cleaned = cleaned.strip()
    return cleaned or None, calls


def contains_tool_call_markup(
    text: str | None,
    *,
    known_tool_names: Set[str] | None = None,
) -> bool:
    if not text:
        return False
    lowered = text.casefold()
    if any(
        marker in lowered
        for marker in (
            "<tool_calls",
            "<tool_call>",
            "<tool_name>",
            "<invoke name=",
            "<function>",
            "<｜dsml｜tool_calls>",
        )
    ):
        return True
    # Detect bare XML tool tags from models like DeepSeek
    if known_tool_names:
        for name in known_tool_names:
            if f"<{name.casefold()}>" in lowered:
                return True
    return False


def strip_dsml_text(
    text: str | None,
    *,
    known_tool_names: Set[str] | None = None,
) -> str:
    cleaned, _ = split_dsml_tool_calls(text, known_tool_names=known_tool_names)
    return cleaned or ""


def _parse_jsonish(raw: str) -> object:
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
