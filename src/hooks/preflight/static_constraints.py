from __future__ import annotations

import logging
from collections.abc import Mapping, Set
from typing import Any

log = logging.getLogger(__name__)


def inspect_static_constraints(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
) -> str | None:
    """Layer 1: Static constraints and fast parameter pruning. Runs in < 1ms."""
    # 1. Allowed tools boundary check
    if tool_name not in allowed_tools:
        return f"Tool {tool_name!r} is not in the reducer-provided allow list."

    # In-place pruning / information pruning if mutable dictionary
    is_mutable = isinstance(arguments, dict)

    # 2. Grep search validation & pruning
    if tool_name == "grep_search":
        pattern = arguments.get("pattern")
        if pattern is None:
            return "grep_search requires a 'pattern' parameter."
        
        pattern_str = str(pattern).strip()
        if not pattern_str:
            return "grep_search requires a non-empty pattern."
            
        if is_mutable:
            # Pruning redundant quotes, backticks, or trailing newlines in search patterns
            if (pattern_str.startswith("`") and pattern_str.endswith("`")) or (pattern_str.startswith('"') and pattern_str.endswith('"')):
                pattern_str = pattern_str[1:-1].strip()
            arguments["pattern"] = pattern_str

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
            arguments["symbol"] = str(symbol).strip()

    # 4. Decision edit validation & pruning
    elif tool_name == "decision_edit":
        target_file = arguments.get("target_file")
        intent = arguments.get("intent")
        if not target_file or not str(target_file).strip():
            return "SIGNAL: Invalid Task Packet. Missing required 'target_file' field."
        if not intent or not str(intent).strip():
            return "SIGNAL: Invalid Task Packet. Missing required 'intent' description."

        # Verify focus_symbols format
        focus_symbols = arguments.get("focus_symbols")
        if focus_symbols is not None:
            if not isinstance(focus_symbols, list):
                return "SIGNAL: Invalid Task Packet. 'focus_symbols' must be a JSON array of strings."
            for idx, item in enumerate(focus_symbols):
                if not isinstance(item, str):
                    return f"SIGNAL: Invalid Task Packet. 'focus_symbols' index {idx} must be a string."

        # Verify context_window format
        context_window = arguments.get("context_window")
        if context_window is not None:
            if not isinstance(context_window, list):
                return "SIGNAL: Invalid Task Packet. 'context_window' must be a JSON array of objects."
            for idx, item in enumerate(context_window):
                if not isinstance(item, dict):
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} must be a JSON object."
                
                ref_file = item.get("file")
                ref_span = item.get("span")
                if not ref_file or not isinstance(ref_file, str) or not ref_file.strip():
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} is missing a valid 'file' string."
                if ref_span is None:
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} is missing a 'span' range."
                if not isinstance(ref_span, list) or len(ref_span) < 2:
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' must be a list containing [start_line, end_line]."
                try:
                    start_line = int(ref_span[0])
                    end_line = int(ref_span[1])
                    if start_line > end_line:
                        return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' start_line {start_line} cannot be greater than end_line {end_line}."
                except (ValueError, TypeError):
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' elements must be valid integers."

        # Verify constraints format
        constraints = arguments.get("constraints")
        if constraints is not None:
            if not isinstance(constraints, dict):
                return "SIGNAL: Invalid Task Packet. 'constraints' must be a JSON object."

        if is_mutable:
            arguments["target_file"] = str(target_file).strip()
            arguments["intent"] = str(intent).strip()

    return None
