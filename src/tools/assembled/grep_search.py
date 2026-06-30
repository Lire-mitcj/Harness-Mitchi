from __future__ import annotations

import asyncio
import re
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


class GrepSearchTool(Tool):
    name = "grep_search"
    description = "Search for a pattern in files using ripgrep. Supports regex."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "path": {
                "type": "string",
                "description": "Directory or file to search in (default: current dir)",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g. '*.py', '*.ts')",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of matching lines to return",
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
                "description": "Search mode to focus/summarize results. 'default' for standard regex/AND; 'symbol' to only return definitions (def/class/async def); 'import' to only return import statements; 'structure' to return a summary of which files contain matches instead of line details.",
                "default": "default",
            },
        },
        "required": ["pattern"],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        pattern: str = validated["pattern"]
        path: str = validated.get("path", ".")
        include: str | None = validated.get("include")
        max_results: int = validated.get("max_results", 50)
        case_insensitive: bool = validated.get("case_insensitive", False)
        mode: str = validated.get("mode", "default")

        # Base cmd
        cmd_base = ["rg", "--line-number", "--no-heading", "--color=never"]
        if case_insensitive:
            cmd_base += ["-i"]

        # Default excludes to suppress noise
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

        stdout_data = ""
        stderr_data = ""
        exit_code = 0

        # Sub-executor helper
        async def run_rg(search_pat: str) -> tuple[int, str, str]:
            cmd = list(cmd_base) + ["--", search_pat, path]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
                return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")
            except FileNotFoundError:
                raise RuntimeError("ripgrep (rg) not found.")
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                raise TimeoutError("Search timed out after 30s")

        try:
            if mode == "symbol":
                has_regex = any(c in pattern for c in "*+?()[]{}|\\^$")
                pat_escaped = pattern if has_regex else re.escape(pattern)
                search_pat = rf"\b(def|class|async\s+def|function|fn)\b.*{pat_escaped}"
                exit_code, stdout_data, stderr_data = await run_rg(search_pat)
            elif mode == "import":
                has_regex = any(c in pattern for c in "*+?()[]{}|\\^$")
                pat_escaped = pattern if has_regex else re.escape(pattern)
                search_pat = rf"\b(import|from)\b.*{pat_escaped}"
                exit_code, stdout_data, stderr_data = await run_rg(search_pat)
            else:
                # default or structure mode
                words = pattern.split()
                is_multi_word = len(words) > 1 and not any(c in pattern for c in "*+?()[]{}|\\^$")
                is_identifier = re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", pattern) is not None

                if mode == "default" and is_identifier:
                    # Promotion step: search for declaration first
                    def_pattern = rf"\b(def|class|async\s+def|function|fn)\s+{pattern}\b"
                    code, out, err = await run_rg(def_pattern)
                    if code == 0 and out.strip():
                        exit_code, stdout_data, stderr_data = code, out, err
                    else:
                        exit_code, stdout_data, stderr_data = await run_rg(rf"\b{pattern}\b")
                elif is_multi_word:
                    # Multi-word AND search: Search for first word, filter by others in Python
                    first_word = words[0]
                    code, out, err = await run_rg(first_word)
                    if code == 0:
                        lines = out.splitlines()
                        filtered_lines = []
                        for line in lines:
                            match_all = True
                            for word in words[1:]:
                                if case_insensitive:
                                    if word.lower() not in line.lower():
                                        match_all = False
                                        break
                                else:
                                    if word not in line:
                                        match_all = False
                                        break
                            if match_all:
                                filtered_lines.append(line)
                        stdout_data = "\n".join(filtered_lines)
                        exit_code = 0 if filtered_lines else 1
                        stderr_data = err
                    else:
                        exit_code, stdout_data, stderr_data = code, out, err
                else:
                    exit_code, stdout_data, stderr_data = await run_rg(pattern)
        except Exception as e:
            return ToolResult(success=False, output="", error=str(e))

        import json
        if exit_code == 1 or not stdout_data.strip():
            if mode == "structure":
                payload = {
                    "file_level_summary": []
                }
            else:
                payload = {
                    "matches": [],
                    "returned_matches": 0,
                    "total_matches": 0,
                    "truncated": False,
                    "next_action": None,
                }
            return ToolResult(
                success=True,
                output=json.dumps(payload, ensure_ascii=False, indent=2),
                metadata={"match_count": 0, "returned_matches": 0, "truncated": False},
            )
        if exit_code > 1:
            return ToolResult(success=False, output="", error=f"rg error (exit {exit_code}): {stderr_data.strip()}")

        raw_lines = stdout_data.strip().splitlines()

        if mode == "structure":
            from collections import Counter
            file_counts = Counter()
            for line in raw_lines:
                parts = line.split(":", 2)
                if parts:
                    file_counts[parts[0]] += 1
            
            summary = []
            for file_path, count in file_counts.items():
                summary.append({
                    "file": file_path,
                    "exists": True,
                    "match_count": count
                })
            
            payload = {
                "file_level_summary": summary
            }
            output_json = json.dumps(payload, ensure_ascii=False, indent=2)
            return ToolResult(
                success=True,
                output=output_json,
                metadata={
                    "match_count": len(raw_lines),
                    "returned_matches": len(summary),
                    "truncated": False,
                    "raw_evidence_store": [{"file": item["file"], "match_count": item["match_count"]} for item in summary]
                }
            )

        truncated = len(raw_lines) > max_results

        matches = []
        for line in raw_lines[:max_results]:
            parts = line.split(":", 2)
            if len(parts) >= 3:
                file_path = parts[0]
                try:
                    line_num = int(parts[1])
                except ValueError:
                    line_num = 1
                content = parts[2]

                sym_match = re.search(r'\b(def|class)\s+([A-Za-z0-9_]+)\b', content)
                symbol_name = sym_match.group(2) if sym_match else ""

                matches.append({
                    "file": file_path,
                    "symbol": symbol_name,
                    "span": [line_num, line_num],
                    "match_line": content.strip()
                })

        symbols = list(dict.fromkeys(item["symbol"] for item in matches if item["symbol"]))
        next_action = (
            {"tool": "view_symbol_code", "symbols": symbols}
            if symbols
            else None
        )
        payload = {
            "matches": matches,
            "returned_matches": len(matches),
            "total_matches": len(raw_lines),
            "truncated": truncated,
            "next_action": next_action,
        }
        output_json = json.dumps(payload, ensure_ascii=False, indent=2)

        return ToolResult(
            success=True,
            output=output_json,
            metadata={
                "match_count": len(raw_lines),
                "returned_matches": len(matches),
                "truncated": truncated,
                "next_action": next_action,
                "raw_evidence_store": matches,
            },
        )
