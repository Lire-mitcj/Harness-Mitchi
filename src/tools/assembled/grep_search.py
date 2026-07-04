from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool
from src.tools.grep_match_symbols import (
    extract_symbol_from_match_line,
    suggested_views_from_matches,
)

MAX_BATCH_PATTERNS = 8


class GrepSearchTool(Tool):
    name = "grep_search"
    description = (
        "Primary discovery tool: locate code with ripgrep (exact text/regex). "
        "REQUIRED: provide `pattern` or `patterns` (4-8 concrete terms from the task). "
        "Never call with only include/mode. Derive patterns from the user task and "
        "STEP EVIDENCE discovery_hints (table names, route paths, handler symbols, "
        "CREATE TABLE, @router) — not single vague words like 'order'. Prefer one call "
        "with `patterns` (batch) or regex alternation (`foo|bar|baz`). Supports modes: "
        "default (literal/regex), symbol (definitions only), import, structure "
        "(file summary — requires pattern; avoid for bootstrap). A hit only locates the "
        "next read; follow suggested_views with view_symbol_code."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Single regex/literal pattern (use when only one term is needed).",
            },
            "patterns": {
                "type": "array",
                "items": {"type": "string"},
                "maxItems": MAX_BATCH_PATTERNS,
                "description": (
                    "Batch discovery: 4-8 concrete patterns run in parallel with the same "
                    "path/include/mode. Example for order timeline task: "
                    "['CREATE TABLE.*order', 'order_timeline', '@router.*order', "
                    "'ticket_order', 'include_router']."
                ),
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current dir)",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py', '*.sql')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return (total across batch)",
                "default": 50,
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Ignore case distinctions in patterns",
                "default": False,
            },
            "mode": {
                "type": "string",
                "enum": ["default", "symbol", "import", "structure"],
                "description": (
                    "Search mode. Prefer default/symbol for bootstrap. "
                    "structure requires an explicit pattern."
                ),
                "default": "default",
            },
        },
        "anyOf": [
            {"required": ["pattern"]},
            {"required": ["patterns"]},
        ],
    }

    @staticmethod
    def _normalize_pattern_list(validated: dict[str, Any]) -> list[str]:
        patterns: list[str] = []
        raw_patterns = validated.get("patterns")
        if isinstance(raw_patterns, list):
            for item in raw_patterns[:MAX_BATCH_PATTERNS]:
                text = str(item or "").strip()
                if text and text not in patterns:
                    patterns.append(text)
        single = str(validated.get("pattern") or "").strip()
        if single and single not in patterns:
            patterns.insert(0, single)
        return patterns

    @staticmethod
    def _build_cmd_base(
        *,
        case_insensitive: bool,
        include: str | None,
    ) -> list[str]:
        cmd_base = ["rg", "--line-number", "--no-heading", "--color=never"]
        if case_insensitive:
            cmd_base += ["-i"]
        excludes = [
            "!*.lock",
            "!package-lock.json",
            "!*.map",
            "!*.svg",
            "!.git/",
            "!.venv/",
            "!node_modules/",
            "!dist/",
            "!build/",
        ]
        for exc in excludes:
            cmd_base += ["--glob", exc]
        if include:
            cmd_base += ["--glob", include]
        return cmd_base

    async def _run_rg(
        self,
        cmd_base: list[str],
        search_pat: str,
        path: str,
    ) -> tuple[int, str, str]:
        cmd = list(cmd_base) + ["--", search_pat, path]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        except FileNotFoundError as exc:
            raise RuntimeError("ripgrep (rg) not found.") from exc
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            except Exception:
                pass
            raise TimeoutError("Search timed out after 30s") from exc

    async def _collect_stdout_for_pattern(
        self,
        *,
        pattern: str,
        cmd_base: list[str],
        path: str,
        mode: str,
        case_insensitive: bool,
    ) -> tuple[int, str, str]:
        if mode == "symbol":
            has_regex = any(c in pattern for c in "*+?()[]{}|\\^$")
            pat_escaped = pattern if has_regex else re.escape(pattern)
            search_pat = rf"\b(def|class|async\s+def|function|fn)\b.*{pat_escaped}"
            return await self._run_rg(cmd_base, search_pat, path)
        if mode == "import":
            has_regex = any(c in pattern for c in "*+?()[]{}|\\^$")
            pat_escaped = pattern if has_regex else re.escape(pattern)
            search_pat = rf"\b(import|from)\b.*{pat_escaped}"
            return await self._run_rg(cmd_base, search_pat, path)

        words = pattern.split()
        is_multi_word = len(words) > 1 and not any(c in pattern for c in "*+?()[]{}|\\^$")
        is_identifier = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern) is not None

        if mode == "default" and is_identifier:
            def_pattern = rf"\b(def|class|async\s+def|function|fn)\s+{pattern}\b"
            code, out, err = await self._run_rg(cmd_base, def_pattern, path)
            if code == 0 and out.strip():
                return code, out, err
            return await self._run_rg(cmd_base, rf"\b{pattern}\b", path)
        if is_multi_word:
            first_word = words[0]
            code, out, err = await self._run_rg(cmd_base, first_word, path)
            if code != 0:
                return code, out, err
            filtered_lines = []
            for line in out.splitlines():
                if all(
                    word.lower() in line.lower() if case_insensitive else word in line
                    for word in words[1:]
                ):
                    filtered_lines.append(line)
            stdout_data = "\n".join(filtered_lines)
            return (0 if filtered_lines else 1), stdout_data, err
        return await self._run_rg(cmd_base, pattern, path)

    @staticmethod
    def _matches_from_raw_lines(raw_lines: list[str], *, limit: int) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for line in raw_lines[:limit]:
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
            file_path = parts[0]
            try:
                line_num = int(parts[1])
            except ValueError:
                line_num = 1
            content = parts[2]
            symbol_name = extract_symbol_from_match_line(content)
            matches.append(
                {
                    "file": file_path,
                    "symbol": symbol_name,
                    "span": [line_num, line_num],
                    "match_line": content.strip(),
                }
            )
        return matches

    def _empty_payload(self, *, mode: str) -> dict[str, Any]:
        if mode == "structure":
            return {"file_level_summary": [], "suggested_views": []}
        return {
            "matches": [],
            "returned_matches": 0,
            "total_matches": 0,
            "truncated": False,
            "next_action": None,
            "suggested_views": [],
        }

    def _result_from_raw_lines(
        self,
        raw_lines: list[str],
        *,
        mode: str,
        max_results: int,
        searched_patterns: list[str],
    ) -> ToolResult:
        if not raw_lines:
            payload = self._empty_payload(mode=mode)
            return ToolResult(
                success=True,
                output=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={
                    "match_count": 0,
                    "returned_matches": 0,
                    "truncated": False,
                    "searched_patterns": searched_patterns,
                },
            )

        if mode == "structure":
            file_counts = Counter()
            for line in raw_lines:
                parts = line.split(":", 2)
                if parts:
                    file_counts[parts[0]] += 1
            summary = [
                {"file": file_path, "exists": True, "match_count": count}
                for file_path, count in file_counts.items()
            ]
            payload = {"file_level_summary": summary}
            return ToolResult(
                success=True,
                output=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={
                    "match_count": len(raw_lines),
                    "returned_matches": len(summary),
                    "truncated": False,
                    "searched_patterns": searched_patterns,
                    "raw_evidence_store": [
                        {"file": item["file"], "match_count": item["match_count"]}
                        for item in summary
                    ],
                },
            )

        truncated = len(raw_lines) > max_results
        matches = self._matches_from_raw_lines(raw_lines, limit=max_results)
        suggested_views = suggested_views_from_matches(matches)
        symbols = [item["symbol"] for item in suggested_views]
        next_action = (
            {
                "tool": "view_symbol_code",
                "symbols": symbols,
                "suggested_views": suggested_views,
            }
            if suggested_views
            else None
        )
        payload = {
            "matches": matches,
            "returned_matches": len(matches),
            "total_matches": len(raw_lines),
            "truncated": truncated,
            "next_action": next_action,
            "searched_patterns": searched_patterns,
            "suggested_views": suggested_views,
        }
        return ToolResult(
            success=True,
            output=json.dumps(payload, ensure_ascii=False, indent=2),
            metadata={
                "match_count": len(raw_lines),
                "returned_matches": len(matches),
                "truncated": truncated,
                "next_action": next_action,
                "searched_patterns": searched_patterns,
                "suggested_views": suggested_views,
                "raw_evidence_store": matches,
            },
        )

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        path: str = validated.get("path", ".")
        include: str | None = validated.get("include")
        max_results: int = validated.get("max_results", 50)
        case_insensitive: bool = validated.get("case_insensitive", False)
        mode: str = validated.get("mode", "default")
        pattern_list = self._normalize_pattern_list(validated)
        if not pattern_list:
            return ToolResult(
                success=False,
                output="",
                error=(
                    "grep_search requires a non-empty 'pattern' or 'patterns' list. "
                    "Do not call with only include/mode. Example: "
                    "grep_search(patterns=['CREATE TABLE', 'ticket_order', "
                    "'@router\\.', 'build_router'], include='*.sql') "
                    "and include='*.py' in a second call or combined path."
                ),
            )
        if mode == "structure" and len(pattern_list) != 1:
            return ToolResult(
                success=False,
                output="",
                error="grep_search mode=structure supports one explicit pattern only.",
            )

        cmd_base = self._build_cmd_base(case_insensitive=case_insensitive, include=include)

        try:
            results = await asyncio.gather(
                *[
                    self._collect_stdout_for_pattern(
                        pattern=pattern,
                        cmd_base=cmd_base,
                        path=path,
                        mode=mode,
                        case_insensitive=case_insensitive,
                    )
                    for pattern in pattern_list
                ]
            )
        except Exception as exc:
            return ToolResult(success=False, output="", error=str(exc))

        merged_lines: list[str] = []
        seen_lines: set[str] = set()
        last_error = ""
        for exit_code, stdout_data, stderr_data in results:
            if exit_code > 1:
                last_error = stderr_data.strip() or f"rg exit {exit_code}"
                continue
            for line in stdout_data.strip().splitlines():
                if line not in seen_lines:
                    seen_lines.add(line)
                    merged_lines.append(line)

        if last_error and not merged_lines:
            return ToolResult(success=False, output="", error=f"rg error: {last_error}")

        return self._result_from_raw_lines(
            merged_lines,
            mode=mode,
            max_results=max_results,
            searched_patterns=pattern_list,
        )
