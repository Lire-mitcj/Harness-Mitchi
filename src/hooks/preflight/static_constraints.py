from __future__ import annotations

import logging
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_GREP_PATTERN_REQUIRED = (
    "grep_search requires a non-empty 'pattern' or 'patterns' list. "
    "Do not call with only include/mode — derive 4-8 concrete patterns from the "
    "user task and STEP EVIDENCE discovery_hints. "
    "Example: grep_search(patterns=['CREATE TABLE', 'ticket_order', "
    "'@router\\.', 'build_router'], include='*.sql')."
)


def inspect_static_constraints(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
    manifest: Any = None,
) -> str | None:
    """Layer 1: Static constraints and fast parameter pruning. Runs in < 1ms."""
    _ = manifest
    # 1. Allowed tools boundary check
    if tool_name not in allowed_tools:
        return f"Tool {tool_name!r} is not in the reducer-provided allow list."

    # In-place pruning / information pruning if mutable dictionary
    is_mutable = isinstance(arguments, dict)

    # 2. Grep search validation & pruning
    if tool_name == "grep_search":
        pattern = arguments.get("pattern")
        patterns = arguments.get("patterns")
        pattern_str = str(pattern or "").strip()
        pattern_items: list[str] = []
        if isinstance(patterns, list):
            for item in patterns[:8]:
                text = str(item or "").strip()
                if text and text not in pattern_items:
                    pattern_items.append(text)
        if pattern_str and pattern_str not in pattern_items:
            pattern_items.insert(0, pattern_str)
        if not pattern_items:
            return _GREP_PATTERN_REQUIRED

        mode = str(arguments.get("mode") or "default").strip()
        if mode == "structure" and not pattern_items:
            return (
                "grep_search mode=structure requires an explicit pattern; "
                "for bootstrap use patterns=[...] with mode=default."
            )

        if is_mutable:
            if pattern_str:
                if (pattern_str.startswith("`") and pattern_str.endswith("`")) or (
                    pattern_str.startswith('"') and pattern_str.endswith('"')
                ):
                    pattern_str = pattern_str[1:-1].strip()
                arguments["pattern"] = pattern_str
            if isinstance(patterns, list):
                arguments["patterns"] = pattern_items[:8]

            # Prune and normalize path
            if "path" in arguments:
                arguments["path"] = str(arguments["path"]).strip().rstrip("/")
            
            # Prune and cap excessive limits (max_results) to prevent token blowout
            if "max_results" in arguments:
                try:
                    val = int(arguments["max_results"])
                    if val > 100:
                        arguments["max_results"] = 50
                except (ValueError, TypeError):
                    pass

    # 3. View symbol validation & pruning
    elif tool_name == "view_symbol_code":
        target_file = arguments.get("target_file")
        symbol = arguments.get("symbol")
        
        if not target_file or not str(target_file).strip():
            return "view_symbol_code requires target_file."
        if not symbol or not str(symbol).strip():
            return "view_symbol_code requires symbol."

        if is_mutable:
            arguments["target_file"] = str(target_file).strip()
            symbol_text = str(symbol).strip()
            file_stem = Path(str(target_file).strip()).stem
            file_name = Path(str(target_file).strip()).name
            if symbol_text in {file_stem, file_name}:
                return (
                    "view_symbol_code symbol must be a code identifier (function/class/DDL name), "
                    "not the filename. Use grep_search first to locate a concrete target."
                )
            arguments["symbol"] = symbol_text

    # 4. Decision edit validation & pruning
    elif tool_name == "decision_edit":
        target_file = arguments.get("target_file")
        intent = arguments.get("intent")
        if not target_file or not str(target_file).strip():
            return "Invalid decision_edit: missing required 'target_file' field."
        if not intent or not str(intent).strip():
            return "Invalid decision_edit: missing required 'intent' description."
        intent_text = str(intent).strip()
        focus_symbols = arguments.get("focus_symbols")
        context_window = arguments.get("context_window")

        # Verify focus_symbols format
        if focus_symbols is not None:
            if not isinstance(focus_symbols, list):
                return "Invalid decision_edit: 'focus_symbols' must be a JSON array of strings."
            for idx, item in enumerate(focus_symbols):
                if not isinstance(item, str):
                    return f"Invalid decision_edit: 'focus_symbols' index {idx} must be a string."

        # Verify context_window format
        if context_window is not None:
            if not isinstance(context_window, list):
                return "Invalid decision_edit: 'context_window' must be a JSON array of objects."
            for idx, item in enumerate(context_window):
                if not isinstance(item, dict):
                    return f"Invalid decision_edit: 'context_window' index {idx} must be a JSON object."
                
                ref_file = item.get("file")
                ref_span = item.get("span")
                if not ref_file or not isinstance(ref_file, str) or not ref_file.strip():
                    return f"Invalid decision_edit: 'context_window' index {idx} is missing a valid 'file' string."
                if ref_span is None:
                    return f"Invalid decision_edit: 'context_window' index {idx} is missing a 'span' range."
                if not isinstance(ref_span, list) or len(ref_span) < 2:
                    return f"Invalid decision_edit: 'context_window' index {idx} 'span' must be a list containing [start_line, end_line]."
                try:
                    start_line = int(ref_span[0])
                    end_line = int(ref_span[1])
                    if start_line > end_line:
                        return f"Invalid decision_edit: 'context_window' index {idx} 'span' start_line {start_line} cannot be greater than end_line {end_line}."
                except (ValueError, TypeError):
                    return f"Invalid decision_edit: 'context_window' index {idx} 'span' elements must be valid integers."

        # Verify constraints format
        constraints = arguments.get("constraints")
        if constraints is not None:
            if not isinstance(constraints, dict):
                return "Invalid decision_edit: 'constraints' must be a JSON object."

        if is_mutable:
            arguments["target_file"] = str(target_file).strip()
            arguments["intent"] = intent_text

    return None
