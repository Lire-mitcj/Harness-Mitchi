from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool
from src.tools.grep_match_symbols import (
    enrich_grep_matches,
    grep_search_fingerprint,
    normalize_match_path,
    parse_rg_hit_line,
)

MAX_BATCH_PATTERNS = 8
_CONTEXT_AFTER_PATTERN = re.compile(
    r"exception_handler|add_exception_handler|@(?:app|router)\.|"
    r"include_router|FastAPI\s*\(|create_app|wire_routes",
)


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
        "(per-file hit summary plus suggested_views for top files). A hit only locates the "
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
            {        "required": ["pattern"]},
            {"required": ["patterns"]},
        ],
    }

    def __init__(self, *, project_root: Path | None = None) -> None:
        self.project_root = project_root.resolve() if project_root is not None else None

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
        *,
        context_after: int = 0,
    ) -> tuple[int, str, str]:
        cmd = list(cmd_base)
        if context_after > 0:
            cmd.extend(["-A", str(context_after)])
        cmd.extend(["--", search_pat, path])
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
            search_pat = rf"\b(?:async\s+def|def|class)\s+{pat_escaped}\b"
            return await self._run_rg(cmd_base, search_pat, path)
        if mode == "import":
            has_regex = any(c in pattern for c in "*+?()[]{}|\\^$")
            pat_escaped = pattern if has_regex else re.escape(pattern)
            search_pat = rf"\b(import|from)\b.*{pat_escaped}"
            return await self._run_rg(cmd_base, search_pat, path)

        context_after = 2 if _CONTEXT_AFTER_PATTERN.search(pattern) else 0

        words = pattern.split()
        is_multi_word = len(words) > 1 and not any(c in pattern for c in "*+?()[]{}|\\^$")
        is_identifier = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern) is not None

        if mode == "default" and is_identifier:
            def_pattern = rf"\b(?:async\s+def|def|class)\s+{pattern}\b"
            code, out, err = await self._run_rg(cmd_base, def_pattern, path)
            def_lines = out.strip().splitlines() if code == 0 and out.strip() else []
            fallback_code, fallback_out, fallback_err = await self._run_rg(
                cmd_base,
                rf"\b{pattern}\b",
                path,
                context_after=context_after,
            )
            fallback_lines = (
                fallback_out.strip().splitlines()
                if fallback_code == 0 and fallback_out.strip()
                else []
            )
            merged = list(dict.fromkeys([*def_lines, *fallback_lines]))
            if merged:
                return 0, "\n".join(merged), err or fallback_err
            return fallback_code, fallback_out, fallback_err
        if is_multi_word:
            first_word = words[0]
            code, out, err = await self._run_rg(
                cmd_base,
                first_word,
                path,
                context_after=context_after,
            )
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
        return await self._run_rg(
            cmd_base,
            pattern,
            path,
            context_after=context_after,
        )

    @staticmethod
    def _matches_from_raw_lines(
        raw_lines: list[tuple[str, str]],
        *,
        limit: int,
        project_root: Path | None = None,
    ) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        for line, matched_pattern in raw_lines[:limit]:
            parsed = parse_rg_hit_line(
                line,
                matched_pattern,
                project_root=project_root,
            )
            if parsed is not None:
                matches.append(parsed)
        return matches

    @staticmethod
    def _structure_seed_lines(
        raw_lines: list[tuple[str, str]],
        *,
        max_files: int = 12,
    ) -> list[tuple[str, str]]:
        per_file: dict[str, tuple[str, str]] = {}
        for line, pattern in raw_lines:
            parts = line.split(":", 2)
            if not parts:
                continue
            file_key = parts[0]
            if file_key not in per_file:
                per_file[file_key] = (line, pattern)
            if len(per_file) >= max_files:
                break
        return list(per_file.values())

    @staticmethod
    def _build_next_action(suggested_views: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not suggested_views:
            return None
        symbols = [str(item.get("symbol") or "") for item in suggested_views if item.get("symbol")]
        return {
            "tool": "view_symbol_code",
            "symbols": symbols,
            "suggested_views": suggested_views,
        }

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
        raw_lines: list[tuple[str, str]],
        *,
        mode: str,
        max_results: int,
        searched_patterns: list[str],
        project_root: Path | None = None,
        repo_map: Any = None,
        search_fingerprint: str = "",
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
                    "search_fingerprint": search_fingerprint,
                    "empty_result": True,
                },
            )

        if mode == "structure":
            file_counts: Counter[str] = Counter()
            for line, _pattern in raw_lines:
                parts = line.split(":", 2)
                if parts:
                    file_counts[normalize_match_path(parts[0], project_root)] += 1
            summary = [
                {"file": file_path, "exists": True, "match_count": count}
                for file_path, count in file_counts.items()
            ]
            seed_lines = self._structure_seed_lines(raw_lines)
            matches, suggested_views = enrich_grep_matches(
                self._matches_from_raw_lines(
                    seed_lines,
                    limit=len(seed_lines),
                    project_root=project_root,
                ),
                searched_patterns=searched_patterns,
                project_root=project_root,
                repo_map=repo_map,
            )
            next_action = self._build_next_action(suggested_views)
            payload = {
                "file_level_summary": summary,
                "matches": matches,
                "returned_matches": len(matches),
                "total_matches": len(raw_lines),
                "truncated": False,
                "next_action": next_action,
                "searched_patterns": searched_patterns,
                "suggested_views": suggested_views,
            }
            return ToolResult(
                success=True,
                output=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={
                    "match_count": len(raw_lines),
                    "returned_matches": len(summary),
                    "truncated": False,
                    "searched_patterns": searched_patterns,
                    "suggested_views": suggested_views,
                    "search_fingerprint": search_fingerprint,
                    "raw_evidence_store": matches or [
                        {"file": item["file"], "match_count": item["match_count"]}
                        for item in summary
                    ],
                },
            )

        truncated = len(raw_lines) > max_results
        matches, suggested_views = enrich_grep_matches(
            self._matches_from_raw_lines(
                raw_lines,
                limit=max_results,
                project_root=project_root,
            ),
            searched_patterns=searched_patterns,
            project_root=project_root,
            repo_map=repo_map,
        )
        top_kinds = [str(item.get("match_kind") or "") for item in matches[:5]]
        next_action = self._build_next_action(suggested_views)
        payload = {
            "matches": matches,
            "returned_matches": len(matches),
            "total_matches": len(raw_lines),
            "truncated": truncated,
            "next_action": next_action,
            "searched_patterns": searched_patterns,
            "suggested_views": suggested_views,
            "match_kinds_top": top_kinds,
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
                "search_fingerprint": search_fingerprint,
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

        merged_lines: list[tuple[str, str]] = []
        seen_lines: set[str] = set()
        last_error = ""
        for pattern, (exit_code, stdout_data, stderr_data) in zip(pattern_list, results):
            if exit_code > 1:
                last_error = stderr_data.strip() or f"rg exit {exit_code}"
                continue
            for line in stdout_data.strip().splitlines():
                if line not in seen_lines:
                    seen_lines.add(line)
                    merged_lines.append((line, pattern))

        if last_error and not merged_lines:
            return ToolResult(success=False, output="", error=f"rg error: {last_error}")

        project_root = params.get("_project_root")
        if project_root is not None and not isinstance(project_root, Path):
            project_root = Path(str(project_root))
        if project_root is None and self.project_root is not None:
            project_root = self.project_root
        repo_map = params.get("_repo_map")
        fingerprint = grep_search_fingerprint(
            pattern_list,
            path=path,
            include=include,
            mode=mode,
        )

        return self._result_from_raw_lines(
            merged_lines,
            mode=mode,
            max_results=max_results,
            searched_patterns=pattern_list,
            project_root=project_root,
            repo_map=repo_map,
            search_fingerprint=fingerprint,
        )
