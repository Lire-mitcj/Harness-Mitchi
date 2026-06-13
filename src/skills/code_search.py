from __future__ import annotations

import json
import re
import ast
from dataclasses import dataclass, field
from pathlib import Path

from src.context.retriever import build_context_queries
from src.skills.base import SkillContext, SkillResult
from src.skills.sql_ast import (
    ProjectSqlAstCache,
    extract_sql_literals_from_python,
    parse_query,
    parse_python_sql_queries,
)
from src.tools.registry import ToolRegistry

_HYDRATION_VERSION = "edit_context_hydration_v4_view_dependency_columns"


class CodeSearchSkill:
    name = "code_search"

    def __init__(self, *, project_root: Path, tools: ToolRegistry) -> None:
        self.project_root = project_root.resolve()
        self.tools = tools

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        query = str(kwargs.get("extra_query") or context.user_request).strip()
        search_query = str(kwargs.get("search_query") or query).strip()
        if not query:
            return SkillResult(
                success=False,
                summary="code_search requires extra_query or user_request.",
                missing_info=("extra_query",),
            )

        search_paths = tuple(
            str(path).strip().replace("\\", "/").lstrip("./")
            for path in (kwargs.get("search_paths") or ())
            if str(path).strip()
        )
        calls = _search_calls(
            context,
            search_query,
            project_root=self.project_root,
            search_paths=search_paths,
        )
        if not calls:
            return SkillResult(
                success=False,
                summary="code_search could not build a search plan.",
                missing_info=("search_plan",),
            )

        outputs: list[str] = []
        errors: list[str] = []
        for name, args in calls:
            result = await self.tools.call(name, args)
            header = _format_call(name, args)
            if result.success:
                outputs.append(f"{header}\n{result.output}")
            else:
                errors.append(f"{header}\n{result.error or result.output}")

        search_output = "\n\n".join(outputs + errors)
        hydrated = _hydrate_snippets(
            self.project_root,
            search_output,
            intended_change=query,
            context=context,
            **kwargs,
        )
        if hydrated.snippets:
            search_output = (
                f"{search_output}\n\n<context_snippets>\n"
                + "\n\n".join(hydrated.snippets)
                + "\n</context_snippets>"
            )
        if hydrated.edit_context is not None:
            edit_context_json = json.dumps(
                hydrated.edit_context,
                ensure_ascii=False,
                indent=2,
            )
            search_output = (
                f"{search_output}\n\nEDIT_CONTEXT_JSON\n"
                + edit_context_json
            )
        else:
            edit_context_json = ""
        if hydrated.target_error:
            missing_info_list = ["target_not_hydrated"]
            for err in hydrated.target_error.split("; "):
                missing_info_list.append(err)
            return SkillResult(
                success=False,
                summary=f"code_search target resolution failed: {hydrated.target_error}",
                missing_info=tuple(missing_info_list),
                warnings=tuple(hydrated.warnings),
                metadata={
                    "search_output": search_output,
                    "edit_context_targets": str(_edit_context_target_count(hydrated.edit_context)),
                    "hydration_root": str(self.project_root),
                    "hydration_hits": str(hydrated.hit_count),
                    "hydration_failures": "; ".join(hydrated.failures[:4]),
                    "hydration_version": _HYDRATION_VERSION,
                    "edit_context_json": edit_context_json,
                    "hydration_hit_paths": "; ".join(hydrated.hit_paths[:8]),
                    "warnings": json.dumps(hydrated.warnings, ensure_ascii=False),
                },
            )
        if errors and not outputs:
            return SkillResult(
                success=False,
                summary="code_search failed: " + "; ".join(errors),
                missing_info=tuple(errors),
                warnings=tuple(hydrated.warnings),
                metadata={
                    "search_output": search_output,
                    "edit_context_targets": str(_edit_context_target_count(hydrated.edit_context)),
                    "hydration_root": str(self.project_root),
                    "hydration_hits": str(hydrated.hit_count),
                    "hydration_failures": "; ".join(hydrated.failures[:4]),
                    "hydration_version": _HYDRATION_VERSION,
                    "edit_context_json": edit_context_json,
                    "hydration_hit_paths": "; ".join(hydrated.hit_paths[:8]),
                    "warnings": json.dumps(hydrated.warnings, ensure_ascii=False),
                },
            )
        summary = f"code_search completed {len(calls)} batched call(s)."
        if hydrated.warnings:
            summary += f" Warnings: {'; '.join(hydrated.warnings)}"
        return SkillResult(
            success=True,
            summary=summary,
            warnings=tuple(hydrated.warnings),
            metadata={
                "search_output": search_output,
                "edit_context_targets": str(_edit_context_target_count(hydrated.edit_context)),
                "hydration_root": str(self.project_root),
                "hydration_hits": str(hydrated.hit_count),
                "hydration_failures": "; ".join(hydrated.failures[:4]),
                "hydration_version": _HYDRATION_VERSION,
                "edit_context_json": edit_context_json,
                "hydration_hit_paths": "; ".join(hydrated.hit_paths[:8]),
                "warnings": json.dumps(hydrated.warnings, ensure_ascii=False),
            },
        )


def _search_calls(
    context: SkillContext,
    query: str,
    *,
    project_root: Path,
    search_paths: tuple[str, ...] = (),
) -> list[tuple[str, dict[str, object]]]:
    queries = build_context_queries(query, limit=10)
    if not queries:
        queries = [query]
    py_pattern = _search_pattern(query, queries, root_query=context.user_request, file_type="py")
    sql_pattern = _search_pattern(query, queries, root_query=context.user_request, file_type="sql")
    calls: list[tuple[str, dict[str, object]]] = []

    if context.context_pack is not None and context.context_pack.search_plan:
        for plan in context.context_pack.search_plan[:4]:
            plan_terms = [
                str(term)
                for term in plan.patterns[:12]
                if str(term).strip()
            ]
            globs = plan.globs or ("*",)
            path = plan.files[0] if len(plan.files) == 1 else "."
            for glob in globs[:2]:
                is_sql = glob.endswith(".sql") or "sql" in glob.lower()
                p_type = "sql" if is_sql else "py"
                plan_pattern = _search_pattern(query, plan_terms, root_query=context.user_request, file_type=p_type) or (sql_pattern if is_sql else py_pattern)
                calls.append((
                    "grep_search",
                    {
                        "pattern": plan_pattern,
                        "path": path,
                        "include": glob,
                        "max_results": 80,
                    },
                ))
        _append_fallback_search_calls(
            calls,
            queries=queries,
            py_pattern=py_pattern,
            sql_pattern=sql_pattern,
            project_root=project_root,
            search_paths=search_paths,
        )
        return calls

    _append_fallback_search_calls(
        calls,
        queries=queries,
        py_pattern=py_pattern,
        sql_pattern=sql_pattern,
        project_root=project_root,
        search_paths=search_paths,
    )
    return calls


def _append_fallback_search_calls(
    calls: list[tuple[str, dict[str, object]]],
    *,
    queries: list[str],
    py_pattern: str,
    sql_pattern: str,
    project_root: Path,
    search_paths: tuple[str, ...] = (),
) -> None:
    seen = {
        (
            name,
            str(args.get("query") or args.get("pattern") or ""),
            str(args.get("path") or ""),
            str(args.get("include") or ""),
        )
        for name, args in calls
    }

    def add(name: str, args: dict[str, object]) -> None:
        key = (
            name,
            str(args.get("query") or args.get("pattern") or ""),
            str(args.get("path") or ""),
            str(args.get("include") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        calls.append((name, args))

    for map_query in queries[:3]:
        add("map_search", {"query": map_query, "limit": 20})
    grep_paths = search_paths or (".",)
    for path in grep_paths[:4]:
        grep_path = str((project_root / path).resolve()) if not Path(path).is_absolute() else path
        add(
            "grep_search",
            {
                "pattern": py_pattern,
                "path": grep_path,
                "include": "*.py",
                "max_results": 80,
            },
        )
        add(
            "grep_search",
            {
                "pattern": sql_pattern,
                "path": grep_path,
                "include": "*.sql",
                "max_results": 80,
            },
        )


def _search_pattern(query: str, queries: list[str], root_query: str = "", file_type: str = "py") -> str:
    text = query.lower()
    root_text = root_query.lower()

    is_view_change = (
        "视图" in query
        or "view" in text
        or "视图" in root_query
        or "view" in root_text
        or "查询" in query
        or "query" in text
        or "sql" in text
        or "sql" in root_text
    )

    if file_type == "sql":
        if is_view_change:
            sql_patterns = [r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW"]
            views_in_query = _extract_views_from_text(query) + _extract_views_from_text(root_query)
            for v in views_in_query:
                sql_patterns.append(rf"\b{re.escape(v)}\b")
            return "|".join(dict.fromkeys(sql_patterns))

    # 1. symbol 精确匹配
    symbols = _extract_symbols(query)
    if symbols:
        return "|".join(rf"\b{re.escape(sym)}\b" for sym in symbols)
        
    # 2. 用户明确文件内搜索
    has_explicit_file = False
    for token in re.split(r"\s+", query):
        if "." in token and token.split(".")[-1] in {"py", "sql", "js", "ts", "tsx", "jsx", "json", "toml", "yaml", "yml", "md"}:
            has_explicit_file = True
            break
            
    if has_explicit_file:
        terms = []
        for term in queries:
            if "." not in term and term.lower() not in {"file", "line", "symbol", "snippet", "code"}:
                terms.append(term)
        if terms:
            return "|".join(re.escape(term) for term in terms)

    # 3. domain terms 辅助搜索
    terms = [
        term
        for term in queries[:12]
        if term.strip()
        and term.lower()
        not in {
            "file",
            "line",
            "symbol",
            "snippet",
            "code",
            "evidence",
            "行号",
            "代码片段",
            "证据",
            "api",
            "query",
            "sql",
            "db",
            "查询",
            "接口",
            "方法",
            "视图",
            "view",
        }
    ]
    if not terms:
        terms = [
            t for t in queries[:12]
            if t.lower() not in {"file", "line", "symbol", "snippet", "code"}
        ]
        
    if is_view_change:
        if "定义" in query or "definition" in text or "create" in text or "定义" in root_query or "definition" in root_text or "create" in root_text:
            if file_type != "py":
                terms.append(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW")
            terms.extend([
                r"--\s*视图",
                r"视图[:：]",
            ])
        else:
            if file_type != "py":
                terms.append(r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW")
            terms.extend([
                r"\bview\b",
                r"视图",
            ])

    seen = set()
    unique_terms = []
    for t in terms:
        if t.lower() not in seen:
            seen.add(t.lower())
            if any(char in t for char in "\\(|?+-*") or t in {"视图", "视图[:：]", r"--\s*视图"}:
                unique_terms.append(t)
            else:
                unique_terms.append(re.escape(t))

    return "|".join(unique_terms)


def _format_call(name: str, args: dict[str, object]) -> str:
    if name == "grep_search":
        return (
            f"[grep_search pattern={args.get('pattern')!r} "
            f"path={args.get('path')!r} include={args.get('include')!r}]"
        )
    if name == "map_search":
        return f"[map_search query={args.get('query')!r}]"
    return f"[{name} {args!r}]"


_GREP_HIT = re.compile(
    r"^\s*(?:-\s+)?(?P<path>[^:\n]+):(?P<line>\d+)(?::|\s*\|)",
    re.MULTILINE,
)
_MAP_HIT = re.compile(
    r"^\s*-\s+(?P<path>[^:\s]+):(?P<start>\d+)(?:-(?P<end>\d+))?",
    re.MULTILINE,
)
_RANGE_HIT = re.compile(
    r"^\s*(?:-\s+)?(?P<path>[\w./-]+\.(?:py|sql|tsx?|jsx?)):"
    r"(?P<start>\d+)-(?P<end>\d+)\b",
    re.MULTILINE,
)


class _HydratedSearch:
    def __init__(
        self,
        *,
        snippets: list[str],
        edit_context: dict[str, object] | None,
        hit_count: int,
        failures: list[str],
        hit_paths: list[str],
        target_error: str = "",
        warnings: list[str] | None = None,
    ) -> None:
        self.snippets = snippets
        self.edit_context = edit_context
        self.hit_count = hit_count
        self.failures = failures
        self.hit_paths = hit_paths
        self.target_error = target_error
        self.warnings = warnings or []


def _merge_ranges(ranges: list[tuple[int, int]], gap: int = 20, max_lines: int = 140) -> list[tuple[int, int]]:
    if not ranges:
        return []

    # First split any range that is too large
    split_inputs = []
    for r_start, r_end in ranges:
        if r_end - r_start + 1 > max_lines:
            curr = r_start
            while curr <= r_end:
                seg_end = min(r_end, curr + max_lines - 1)
                split_inputs.append((curr, seg_end))
                curr = seg_end + 1
        else:
            split_inputs.append((r_start, r_end))
            
    sorted_ranges = sorted(split_inputs, key=lambda r: r[0])
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end + gap:
            potential_end = max(prev_end, curr_end)
            if potential_end - prev_start + 1 <= max_lines:
                merged[-1] = (prev_start, potential_end)
                continue
        merged.append(current)
    return merged


def _hydrate_snippets(
    project_root: Path,
    search_output: str,
    *,
    intended_change: str = "",
    max_files: int = 6,
    padding: int = 3,
    max_chars: int = 10_000,
    context: SkillContext | None = None,
    **kwargs: object,
) -> _HydratedSearch:
    """Read bounded snippets around repo_map/grep hits; never expose raw full-file IO."""
    if context is not None and context.context_pack is not None:
        budget_chars = context.context_pack.budget.get("snippet_chars")
        if budget_chars:
            max_chars = max(max_chars, budget_chars)
            
    file_hits: dict[str, tuple[Path, list[tuple[int, int]]]] = {}
    root = project_root.resolve()
    failures: list[str] = []

    def add(path: str, start: int, end: int | None = None) -> None:
        raw = path.strip().replace("\\", "/")
        if not raw or raw.startswith("["):
            return
        start = max(1, start)
        end = max(start, end or start)
        resolved, display = _resolve_hit_path(root, raw, required_line=end)
        if resolved is None:
            failures.append(f"{raw}: not found under hydration root {root}")
            return
        current = file_hits.get(display)
        if current is None:
            file_hits[display] = (resolved, [(start, end)])
            return
        current[1].append((start, end))

    for match in _GREP_HIT.finditer(search_output):
        add(match.group("path"), int(match.group("line")))
    for match in _MAP_HIT.finditer(search_output):
        add(
            match.group("path"),
            int(match.group("start")),
            int(match.group("end") or match.group("start")),
        )
    for match in _RANGE_HIT.finditer(search_output):
        add(match.group("path"), int(match.group("start")), int(match.group("end")))

    flat_hits: list[tuple[Path, str, int, int]] = []
    for display, (resolved, ranges) in file_hits.items():
        merged = _merge_ranges(ranges, gap=20, max_lines=1000)
        for start, end in merged:
            flat_hits.append((resolved, display, start, end))

    snippets: list[str] = []
    evidence_targets: list[dict[str, object]] = []
    used_chars = 0
    target_keys: set[tuple[str, int, int]] = set()

    ordered_hits = sorted(
        flat_hits,
        key=lambda item: (
            0 if item[0].suffix in {".py", ".ts", ".tsx", ".js", ".jsx"} else 1,
            -(item[3] - item[2]),
            item[1],
        ),
    )
    for path, display, start, end in ordered_hits[:max_files]:
        try:
            if not path.is_file():
                failures.append(f"{display}: file not found after resolve")
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            failures.append(f"{display}: read failed")
            continue
        if not lines:
            failures.append(f"{display}: file is empty")
            continue
        if start > len(lines):
            failures.append(
                f"{display}:{start}-{end}: line range outside file length {len(lines)}"
            )
            continue
        target_start = max(1, start)
        target_end = min(len(lines), end)
        if path.suffix == ".py":
            ast_range = _view_ast_range_for_hit(project_root, display, target_start, target_end)
            if ast_range is not None:
                target_start, target_end = ast_range
            else:
                target_start, target_end = _expand_python_symbol_range(
                    lines,
                    target_start,
                    target_end,
                    max_lines=1000,
                )
        elif path.suffix == ".sql":
            ast_range = _view_ast_range_for_hit(project_root, display, target_start, target_end)
            if ast_range is not None:
                target_start, target_end = ast_range
            else:
                target_start, target_end = _expand_sql_symbol_range(
                    lines,
                    target_start,
                    target_end,
                    max_lines=1000,
                )
        if target_end < target_start:
            failures.append(
                f"{display}:{start}-{end}: invalid target range "
                f"{target_start}-{target_end}"
            )
            continue
        target_code = "\n".join(lines[target_start - 1 : target_end])
        if not target_code.strip():
            failures.append(f"{display}:{target_start}-{target_end}: empty target code")
            continue
        if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx", ".sql"}:
            if _is_editable_target(target_code, display, intended_change, context):
                editable_targets.append({
                    "file": display,
                    "symbol": symbol_lock.name if symbol_lock else "",
                    "start_line": target_start,
                    "end_line": target_end,
                    "current_code": target_code,
                    "display_code": edit_context_code,
                    "full_range_line_count": target_end - target_start + 1,
                    "hydration_simplified": edit_context_code != target_code,
                    "intended_change": intended_change,
                    "acceptance_criteria": ["Target behavior is changed as requested."],
                })
            elif _is_sql_view_change(intended_change):
                failures.append(
                    f"{display}:{target_start}-{target_end}: not editable for "
                    "SQL/view query change; no SELECT/FROM/JOIN table reference"
                )

        s = max(1, target_start - padding)
        e = min(len(lines), target_end + padding)
        if e < s:
            failures.append(f"{display}:{start}-{end}: invalid hydrated range {s}-{e}")
            continue
        body = "\n".join(
            f"{line_no}: {lines[line_no - 1]}"
            for line_no in range(s, e + 1)
        )
        chunk = f'<snippet path="{display}" lines="{s}-{e}">\n{body}\n</snippet>'
        if used_chars + len(chunk) <= max_chars:
            snippets.append(chunk)
            used_chars += len(chunk)
        else:
            core_body = "\n".join(
                f"{line_no}: {lines[line_no - 1]}"
                for line_no in range(target_start, target_end + 1)
            )
            core_body = _optimize_snippet_body(core_body, display)
            core_chunk = (
                f'<snippet path="{display}" lines="{target_start}-{target_end}" status="truncated_to_core">\n'
                f"# Padding lines omitted due to token budget limits. Use read_file/view_file to view context.\n"
                f"{core_body}\n"
                f"</snippet>"
            )
            if used_chars + len(core_chunk) <= max_chars:
                snippets.append(core_chunk)
                used_chars += len(core_chunk)
                failures.append(f"{display}:{s}-{e}: snippet padded lines omitted due to budget")
            else:
                ref_chunk = (
                    f'<snippet path="{display}" lines="{target_start}-{target_end}" status="omitted_due_to_budget">\n'
                    f"# Full snippet omitted due to token budget limits. Use read_file/view_file to view lines {target_start}-{target_end}.\n"
                    f"</snippet>"
                )
                if used_chars + len(ref_chunk) <= max_chars:
                    snippets.append(ref_chunk)
                    used_chars += len(ref_chunk)
                    failures.append(f"{display}:{s}-{e}: snippet omitted due to budget")
                else:
                    failures.append(f"{display}:{s}-{e}: snippet skipped by display token budget")

    resolution = resolve_targets(
        context.user_request if context is not None else intended_change,
        intended_change=intended_change,
        context=context,
        fallback_targets=evidence_targets,
    )
    primary_targets, target_failures = hydrate_targets(
        root,
        resolution,
        intended_change=intended_change,
    )

    # Gather warnings and check hard failures
    warnings_list = []
    primary_symbol_not_found_errors = []
    primary_not_hydrated_errors = []
    no_sql_errors = []

    hydrated_symbols = {str(t.get("symbol") or "") for t in primary_targets}

    for failure in target_failures:
        if ":" in failure:
            reason, sym = failure.split(":", 1)
            exists_in_repo = sym in hydrated_symbols
            if reason != "primary_target_required":
                exists_in_repo = True

            if _is_primary_symbol(sym, intended_change, exists_in_repo):
                if reason == "primary_target_required":
                    primary_symbol_not_found_errors.append(failure)
                elif reason == "current_code_required":
                    primary_not_hydrated_errors.append(failure)
                elif reason == "no_rewriteable_sql_found":
                    if requires_editable_target:
                        no_sql_errors.append(failure)
                    else:
                        warnings_list.append(failure)
                else:
                    warnings_list.append(failure)
            else:
                warnings_list.append(failure)

    # Check if explicitly specified files exist
    search_paths = tuple(
        str(path).strip().replace("\\", "/").lstrip("./")
        for path in (kwargs.get("search_paths") or ())
        if str(path).strip()
    )
    explicit_files = _extract_explicit_files_from_query(intended_change)
    file_not_found_errors = []
    for path_str in list(search_paths) + explicit_files:
        if "*" in path_str or not path_str.endswith((".py", ".sql", ".js", ".ts", ".tsx", ".jsx", ".json", ".toml", ".yaml", ".yml", ".md")):
            continue
        p = (root / path_str.lstrip("./")).resolve()
        try:
            if not p.is_file():
                file_not_found_errors.append(f"file_not_found:{path_str}")
        except OSError:
            file_not_found_errors.append(f"file_not_found:{path_str}")

    editable_targets = _merge_primary_and_evidence_targets(primary_targets, evidence_targets)
    fully_hydrated_targets = []
    for target in editable_targets:
        if "inner_targets" not in target or not target.get("hydrated"):
            hydrated_target = _resolve_inner_targets(target, resolution, intended_change)
            fully_hydrated_targets.append(hydrated_target.to_edit_target(intended_change))
        else:
            fully_hydrated_targets.append(target)
    editable_targets = fully_hydrated_targets

    edit_context = build_edit_context(
        editable_targets,
        intended_change=intended_change,
        project_root=root,
        context=context,
        resolution=resolution,
        **kwargs,
    )
    edit_context_targets = _edit_context_target_count(edit_context)

    hard_errors = list(file_not_found_errors) + list(primary_symbol_not_found_errors) + list(primary_not_hydrated_errors) + list(no_sql_errors)
    if requires_editable_target and edit_context is None:
        hard_errors.append("target_symbol_required")
    if requires_editable_target and not hard_errors:
        primary_symbols_in_query = [sym for sym in resolution.symbols if _is_primary_symbol(sym, intended_change, True)]
        if primary_symbols_in_query:
            matched_targets = [t for t in primary_targets if str(t.get("symbol")) in primary_symbols_in_query]
            writable_matched = [t for t in matched_targets if not t.get("read_only") and not str(t.get("file", "")).endswith(".sql")]
            if not matched_targets:
                hard_errors.append("primary_target_required")
            elif writable_matched and not any(str(t.get("current_code", "")).strip() for t in writable_matched):
                hard_errors.append("current_code_required")
            elif _is_sql_view_change(intended_change):
                if writable_matched and not any(_code_contains_sql(str(t.get("current_code", ""))) or _has_dynamic_sql_clues(str(t.get("current_code", ""))) for t in writable_matched):
                    hard_errors.append("no_rewriteable_sql_found")
        else:
            writable_edit_context_targets = len([t for t in editable_targets if not t.get("read_only") and not str(t.get("file", "")).endswith(".sql")])
            if writable_edit_context_targets == 0:
                hard_errors.append("current_code_required")

    target_error = ""
    if hard_errors:
        unique_hard_errors = list(dict.fromkeys(hard_errors))
        target_error = "; ".join(unique_hard_errors)

    primary_snippets, used_chars = _snippets_for_targets(
        root,
        primary_targets,
        padding=padding,
        max_chars=max_chars,
        used_chars=0,
    )
    if primary_snippets:
        snippets = primary_snippets + snippets
        used_chars = sum(len(chunk) for chunk in snippets)

    return _HydratedSearch(
        snippets=snippets,
        edit_context=edit_context,
        hit_count=len(file_hits),
        failures=failures + target_failures,
        hit_paths=[f"{display}:{start}-{end}" for _path, display, start, end in ordered_hits],
        target_error=target_error,
        warnings=warnings_list,
    )


def _format_snippet(display: str, lines: list[str], start: int, end: int) -> str:
    body = "\n".join(
        f"{line_no}: {lines[line_no - 1]}"
        for line_no in range(start, end + 1)
    )
    return f'<snippet path="{display}" lines="{start}-{end}">\n{body}\n</snippet>'


def _merge_primary_and_evidence_targets(
    primary_targets: list[dict[str, object]],
    evidence_targets: list[dict[str, object]],
) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()
    for target in [*primary_targets, *evidence_targets]:
        key = (
            str(target.get("file") or ""),
            int(target.get("start_line") or 0),
            int(target.get("end_line") or 0),
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(target)
    return merged


def _snippets_for_targets(
    project_root: Path,
    targets: list[dict[str, object]],
    *,
    padding: int,
    max_chars: int,
    used_chars: int,
) -> tuple[list[str], int]:
    snippets: list[str] = []
    for target in targets:
        display = str(target.get("file") or "")
        if not display:
            continue
        path = project_root / display
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        start = int(target.get("start_line") or 1)
        end = int(target.get("end_line") or start)
        s = max(1, start - padding)
        e = min(len(lines), end + padding)
        if e < s:
            continue
        chunk = _format_snippet(display, lines, s, e)
        if used_chars + len(chunk) > max_chars:
            continue
        snippets.append(chunk)
        used_chars += len(chunk)
    return snippets, used_chars


def _symbol_text_for_targeting(
    context: SkillContext | None,
    intended_change: str,
) -> str:
    parts = []
    if context is not None:
        parts.append(context.user_request)
    parts.append(intended_change)
    if context is not None:
        parts.append(str(context.metadata.get("target_symbol", "") or ""))
        if context.context_pack is not None:
            parts.append(str(context.context_pack.metadata.get("target_symbol", "") or ""))
    return " ".join(part for part in parts if part)


def _target_view_from_context_or_text(
    context: SkillContext | None,
    text: str,
) -> str:
    if context is not None:
        if context.context_pack is not None:
            value = str(context.context_pack.metadata.get("target_view") or "").strip()
            if value:
                return value
        value = str(context.metadata.get("target_view") or "").strip()
        if value:
            return value
    views = _extract_views_from_text(text)
    return views[0] if views else ""


def _context_pack_symbols(context: SkillContext | None) -> list[str]:
    if context is None or context.context_pack is None:
        return []
    symbols: list[str] = []
    for info in context.context_pack.candidate_symbols:
        name = str(info.get("name") or "").strip()
        if name and not _looks_like_sql_variable(name):
            symbols.append(name)
    value = str(context.context_pack.metadata.get("target_symbol") or "").strip()
    if value and not _looks_like_sql_variable(value):
        symbols.insert(0, value)
    return list(dict.fromkeys(symbols))


def _extract_python_symbol_candidates(text: str) -> list[str]:
    return list(dict.fromkeys(
        symbol
        for symbol in _extract_symbols(text)
        if not _looks_like_sql_variable(symbol)
        and not _looks_like_view_name(symbol)
    ))


def _looks_like_sql_variable(name: str) -> bool:
    lowered = name.lower()
    return lowered.endswith("sql") or lowered.endswith("_sql") or lowered in {"sql", "count_sql"}


def _looks_like_view_name(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered.startswith("v_")
        or lowered.startswith("view_")
        or lowered.endswith("_view")
        or "_view_" in lowered
    )


def _resolve_inner_targets(
    target: dict[str, object],
    resolution: TargetResolution,
    intended_change: str,
) -> HydratedTarget:
    current_code = str(target.get("current_code") or "")
    sql_presence = _code_contains_sql(current_code)
    inner: dict[str, object] = {}
    is_readonly = target.get("read_only") or str(target.get("file", "")).endswith(".sql")
    hydrated = bool(current_code.strip()) or is_readonly
    sql_dynamic = False
    has_clues = _has_dynamic_sql_clues(current_code)
    if resolution.target_sql_kind == "count":
        inner["sql_kind"] = "count"
        literal = _select_count_literal(current_code, resolution.target_sql_variable)
        if literal is None:
            if has_clues:
                sql_dynamic = True
        else:
            model = parse_query(literal.sql)
            if model is None or not model.from_table:
                sql_dynamic = True
            else:
                where_clean = model.where
                if where_clean.upper().startswith("WHERE "):
                    where_clean = where_clean[6:].strip()
                inner.update({
                    "sql_variable": literal.variable,
                    "count_where_clause": where_clean,
                    "count_clauses": {
                        "where": where_clean,
                        "joins": [join.to_dict() for join in model.joins],
                        "from": model.from_table,
                    }
                })
    else:
        if not parse_python_sql_queries(current_code) and has_clues:
            sql_dynamic = True
    inner["sql_dynamic"] = sql_dynamic

    ret_sql_kind = resolution.target_sql_kind
    if ret_sql_kind == "count" and sql_dynamic:
        ret_sql_kind = "dynamic_count_query"

    return HydratedTarget(
        file=str(target.get("file") or ""),
        symbol=str(target.get("symbol") or target.get("name") or ""),
        start_line=int(target.get("start_line") or 0),
        end_line=int(target.get("end_line") or 0),
        current_code=current_code,
        sql_presence=sql_presence,
        target_sql_kind=ret_sql_kind,
        target_sql_variable=resolution.target_sql_variable,
        hydrated=hydrated,
        inner_targets=inner,
        source=str(target.get("targeting") or resolution.source),
    )


def _select_count_literal(current_code: str, target_sql_variable: str):
    literals = extract_sql_literals_from_python(current_code)
    if target_sql_variable:
        for literal in literals:
            if literal.variable == target_sql_variable and _is_count_sql(literal.sql):
                return literal
    for literal in literals:
        if _is_count_sql(literal.sql):
            return literal
    return None


def _find_explicit_python_symbol_targets(
    project_root: Path,
    intended_change: str,
    *,
    symbol_text: str = "",
) -> list[dict[str, object]]:
    symbols = _extract_symbols(symbol_text or intended_change)
    if not symbols:
        return []
    symbol_order = {symbol: idx for idx, symbol in enumerate(symbols)}
    symbol_set = set(symbols)
    targets: list[dict[str, object]] = []
    for path in sorted(project_root.rglob("*.py")):
        if _is_ignored_path(path):
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(code)
        except (OSError, SyntaxError):
            continue
        lines = code.splitlines()
        display = str(path.resolve().relative_to(project_root)).replace("\\", "/")
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                continue
            if node.name not in symbol_set:
                continue
            start = int(getattr(node, "lineno", 1) or 1)
            end = int(getattr(node, "end_lineno", start) or start)
            current_code = "\n".join(lines[start - 1 : end])
            if not current_code.strip():
                continue
            sql_presence = _code_contains_sql(current_code)
            targets.append({
                "file": display,
                "symbol": node.name,
                "start_line": start,
                "end_line": end,
                "current_code": current_code,
                "intended_change": intended_change,
                "acceptance_criteria": ["Target behavior is changed as requested."],
                "targeting": "explicit_python_symbol_ast",
                "sql_presence": sql_presence,
            })
    targets.sort(
        key=lambda target: (
            symbol_order.get(str(target.get("symbol") or ""), 10_000),
            str(target.get("file") or ""),
            int(target.get("start_line") or 0),
        )
    )
    return targets


def _is_ignored_path(path: Path) -> bool:
    ignored = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
        ".mypy_cache",
        ".pytest_cache",
    }
    return any(part in ignored for part in path.parts)


def _code_contains_sql(code: str) -> bool:
    return bool(re.search(r"(?is)\bSELECT\b.{0,500}\bFROM\b", code))


def _has_dynamic_sql_clues(code: str) -> bool:
    lowered = code.lower()
    clues = ("sql", "query", "select", "count", "join", "from", "where")
    return any(w in lowered for w in clues)


def _find_all_views(project_root: Path) -> list[str]:
    return ProjectSqlAstCache(project_root).view_names()


def _find_all_view_details(project_root: Path) -> dict[str, dict[str, object]]:
    details: dict[str, dict[str, object]] = {}
    for key, model in ProjectSqlAstCache(project_root).views().items():
        details[key] = model.to_dependency_dict()
    return details


def _view_ast_range_for_hit(
    project_root: Path,
    display_file: str,
    start: int,
    end: int,
) -> tuple[int, int] | None:
    for view in ProjectSqlAstCache(project_root).views().values():
        sql_range = view.sql_range
        if sql_range is None or sql_range.file != display_file:
            continue
        if start <= sql_range.end_line and end >= sql_range.start_line:
            return sql_range.start_line, sql_range.end_line
    return None


def _extract_views_from_text(text: str) -> list[str]:
    pattern1 = re.compile(r"\b(v_[a-zA-Z0-9_]+|[a-zA-Z0-9_]+_view[s]?)\b", re.IGNORECASE)
    views = set(pattern1.findall(text))
    pattern2 = re.compile(
        r"\b(?:view|视图|视图名)[:：\s`']*\b([a-zA-Z0-9_.]+)\b",
        re.IGNORECASE
    )
    for name in pattern2.findall(text):
        if name.lower() not in {"name", "definition", "schema", "the", "a", "to", "and", "or", "of", "in"}:
            views.add(name)
    return sorted(list(views))


def _infer_target_view(
    intended_change: str,
    views: list[str],
    editable_targets: list[dict[str, object]]
) -> str:
    if not views:
        return ""
    
    def tokenize(s: str) -> list[str]:
        return re.findall(r"\b[a-zA-Z0-9]+\b", s.replace("_", " ").lower())
    
    terms = set(tokenize(intended_change))
    for target in editable_targets:
        file = str(target.get("file") or "")
        terms.update(tokenize(file))
        current_code = str(target.get("current_code") or "")
        for m in re.finditer(r"\bdef\s+([a-zA-Z0-9_]+)\b", current_code, re.IGNORECASE):
            terms.update(tokenize(m.group(1)))
        for m in re.finditer(r"\b(?:FROM|JOIN)\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?", current_code, re.IGNORECASE):
            terms.update(tokenize(m.group(1)))

    stopwords = {
        "def", "class", "async", "await", "return", "import", "from", "as",
        "if", "else", "elif", "for", "while", "in", "not", "and", "or", "try", "except",
        "view", "query", "sql", "db", "api", "file", "line", "symbol", "snippet",
        "code", "evidence", "test", "v", "视图", "查询", "接口", "方法",
        "change", "replace", "use", "with", "the", "this", "method", "function",
        "using", "into", "to", "for", "of", "a", "an", "is"
    }
    filtered_terms = {t for t in terms if t not in stopwords}
    
    best_view = ""
    best_score = -1
    intended_lower = intended_change.lower()
    for view in views:
        view_tokens = set(tokenize(view))
        score = 0
        for token in view_tokens:
            if token in stopwords:
                continue
            if token in filtered_terms:
                score += 10
            for term in filtered_terms:
                if term in token or token in term:
                    score += 2
        
        view_lower = view.lower()
        if view_lower in intended_lower:
            score += 100
            
        # Enhanced matching with Chinese/pinyin/English combinations
        groups = [
            ({"登机牌", "boarding", "dengjipai", "dengji", "pass"}, {"boarding", "pass", "ticket", "dengji", "dengjipai"}),
            ({"订单", "order", "dingdan"}, {"order", "report", "detail", "dingdan"}),
            ({"机票", "ticket", "jipiao"}, {"ticket", "report", "detail", "jipiao", "passenger"}),
            ({"监控", "monitor", "flight", "jiankong"}, {"monitor", "flight", "jiankong", "load"}),
            ({"详情", "detail", "xiangqing"}, {"detail", "xiangqing"}),
            ({"报告", "report", "baogao"}, {"report", "baogao"}),
        ]
        for query_set, view_set in groups:
            if any(q_term in intended_lower for q_term in query_set):
                if any(v_term in view_lower for v_term in view_set):
                    score += 20

        if score > best_score:
            best_score = score
            best_view = view
            
    if best_score <= 0 or not best_view:
        if views:
            return views[0]
        return ""
    return best_view


def _build_edit_plan_context(
    editable_targets: list[dict[str, object]],
    *,
    intended_change: str,
    project_root: Path,
    context: SkillContext | None = None,
    **kwargs: object,
) -> dict[str, object] | None:
    if not editable_targets:
        return None
    scope = [
        str(target["file"])
        for target in editable_targets
        if str(target.get("file") or "")
    ]
    views = _find_all_views(project_root)
    view_details = _find_all_view_details(project_root)
    if not views:
        views = _extract_views_from_text(intended_change)
    
    task_analysis = kwargs.get("task_analysis")
    analysis_strategy = ""
    if isinstance(task_analysis, dict):
        analysis_strategy = str(
            task_analysis.get("edit_strategy") or task_analysis.get("intent") or ""
        ).strip()

    target_view = ""
    if context is not None:
        if context.context_pack is not None:
            target_view = context.context_pack.metadata.get("target_view") or ""
        if not target_view:
            target_view = context.metadata.get("target_view") or ""
    if not target_view:
        target_view = str(kwargs.get("target_view") or "").strip()
    if not target_view:
        target_view = _infer_target_view(intended_change, views, editable_targets)
        
    target_symbols = _extract_symbols(intended_change)
    target_symbol = target_symbols[0] if target_symbols else "目标代码"
    
    is_view = "视图" in intended_change or "view" in intended_change.lower() or "sql" in intended_change.lower()
    operation = "replace_dependency" if is_view else "general_edit"
    edit_strategy = "deterministic_rewrite"
    if operation == "replace_dependency":
        if target_sql_kind == "count":
            if is_dynamic_count:
                operation = "dynamic_count_query_rewrite"
                edit_strategy = "dynamic_sql_rewrite"
            else:
                operation = "count_query_view_rewrite"
                edit_strategy = "deterministic_rewrite"
        else:
            if has_dynamic_sql:
                operation = "dynamic_sql_rewrite"
                edit_strategy = "dynamic_sql_rewrite"
            else:
                edit_strategy = "deterministic_rewrite"
    
    replaces = []
    for target in editable_targets:
        code = str(target.get("current_code") or "")
        for query in parse_python_sql_queries(code):
            replaces.extend(query.source_tables)
    replaces = list(dict.fromkeys(replaces))
    
    resolved_deps = []
    if is_view and target_view:
        detail = view_details.get(target_view.lower()) or view_details.get(target_view.split(".")[-1].lower())
        dep_replaces = replaces
        columns: list[str] = []
        evidence: list[str] = []
        if isinstance(detail, dict):
            columns = [str(col) for col in detail.get("columns") or [] if str(col).strip()]
            column_sources = [
                item
                for item in detail.get("column_sources") or []
                if isinstance(item, dict)
            ]
            source_to_view_column = {
                str(key): str(value)
                for key, value in (detail.get("source_to_view_column") or {}).items()
            }
            view_column_to_source = {
                str(key): str(value)
                for key, value in (detail.get("view_column_to_source") or {}).items()
            }
            column_defaults = {
                str(key): str(value)
                for key, value in (detail.get("column_defaults") or {}).items()
            }
            evidence = [str(item) for item in detail.get("evidence") or [] if str(item).strip()]
            dep_replaces = list(dict.fromkeys(
                dep_replaces
                + [str(obj) for obj in detail.get("replaces_objects") or [] if str(obj).strip()]
            ))
        else:
            column_sources = []
            source_to_view_column = {}
            view_column_to_source = {}
            column_defaults = {}
        resolved_deps.append({
            "role": "replacement_source",
            "kind": "database_view",
            "name": target_view,
            "columns": columns,
            "column_sources": column_sources,
            "source_to_view_column": source_to_view_column,
            "view_column_to_source": view_column_to_source,
            "column_defaults": column_defaults,
            "replaces_objects": dep_replaces,
            "evidence": evidence,
            "confidence": 0.95
        })
        
    edit_targets = []
    for target in editable_targets[:4]:
        sql_literals = extract_sql_literals_from_python(str(target.get("current_code") or ""))
        query_models = [
            query.to_dict()
            for query in parse_python_sql_queries(str(target.get("current_code") or ""))
        ]
        edit_targets.append({
            "file": str(target.get("file") or ""),
            "symbol": str(target.get("symbol") or target.get("name") or target_symbol),
            "current_code": str(target.get("current_code") or ""),
            "display_code": str(target.get("display_code") or target.get("current_code") or ""),
            "hydration_simplified": bool(target.get("hydration_simplified")),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line"),
            "sql_queries": query_models,
            "sql_literals": _sql_literal_metadata(sql_literals),
            "sql_presence": bool(target.get("sql_presence", _code_contains_sql(str(target.get("current_code") or "")))),
            "inner_targets": dict(target.get("inner_targets") or {}),
            "symbol_lock": target.get("symbol_lock"),
        })
        
    return {
        "schema": "mitkii.edit_context.v2",
        "builder": "EditPlanBuilder",
        "code_edit_ready": True,
        "edit_strategy": edit_strategy,
        "dependencies_resolved": bool(resolved_deps) if is_view else True,
        "dependencies_resolution_source": "search_hydration_advisory",
        "task_intent": {
            "operation": operation,
            "edit_strategy": edit_strategy,
            "target_symbol": target_symbol,
            "target_type": "sql_variable" if target_sql_variable else "symbol",
            "parent_symbol": target_symbol,
            "local_target": target_sql_variable,
            "target_sql_kind": target_sql_kind,
            "target_sql_variable": target_sql_variable,
            "goal": "use existing implementation/object instead of current query builder" if is_view else "edit code as requested"
        },
        "edit_targets": edit_targets,
        "resolved_dependencies": resolved_deps,
        "constraints": [
            "Do not invent dependencies",
            "Prefer resolved_dependencies when present; fallback may use target_view and available columns",
            "Preserve public function signature",
            "Preserve returned field names used by callers",
        ] + (
            [
                "Keep original clauses/where_clause for list_sql unchanged.",
                "Add count_clauses and count_where_clause for count_sql only.",
                "Do not modify list_sql.",
                "Do not call or edit build_order_detail_sql.",
                "Replace o./p./f. aliases only inside count_clauses.",
            ]
            if operation == "dynamic_count_query_rewrite"
            else (
                [
                    "For COUNT queries, rewrite the entire SQL statement. Do not just replace FROM/JOIN; you must also rewrite all column references and table aliases in the WHERE clause (and other clauses) to use the new view alias/columns."
                ]
                if target_sql_kind == "count"
                else [
                    "For SQL replacement, rebuild SELECT from replacement_source.columns and remove only replaces_objects JOINs"
                ]
            )
        ),
        "acceptance": [
            {
                "type": "diff_must_touch_symbol",
                "symbol": target_symbol
            },
            {
                "type": "must_reference_dependency",
                "role": "replacement_source"
            },
            {
                "type": "must_not_reference_unresolved_dependency"
            },
            {
                "type": "compile_or_syntax_check"
            }
        ],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": list(dict.fromkeys(scope)),
        },
        "snippets": editable_targets[:4],
        "editable_targets": editable_targets[:4],
        "intended_change": intended_change,
        "available_views": views,
        "target_view": target_view,
        "target_sql_kind": target_sql_kind,
        "target_sql_variable": target_sql_variable,
        "acceptance_criteria": ["Target behavior is changed as requested."],
    }


def _infer_target_sql_kind(text: str) -> str:
    lowered = text.lower()
    if "count" in lowered or "计数" in text or "数量" in text or "总数" in text:
        return "count"
    return ""


def _infer_target_sql_variable(text: str, target_sql_kind: str) -> str:
    explicit = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*sql[A-Za-z0-9_]*)\b", text)
    for name in explicit:
        if target_sql_kind == "count" and "count" in name.lower():
            return name
    if target_sql_kind == "count":
        return "count_sql"
    return explicit[0] if explicit else ""


def _target_symbol_from_context(context: SkillContext | None) -> str:
    if context is None:
        return ""
    for source in (
        context.metadata,
        context.context_pack.metadata if context.context_pack is not None else {},
    ):
        value = str(source.get("target_symbol", "") or "").strip()
        if value:
            return value
    return ""


def _is_count_sql(sql: str) -> bool:
    model = parse_query(sql)
    if model is None:
        return bool(re.search(r"(?is)\bSELECT\s+COUNT\s*\(", sql))
    return any(
        item.kind == "aggregate" and re.search(r"(?i)\bCOUNT\s*\(", item.expression)
        for item in model.selects
    )


def _sql_literal_metadata(sql_literals: list[object]) -> list[dict[str, object]]:
    metadata: list[dict[str, object]] = []
    for literal in sql_literals:
        sql = str(getattr(literal, "sql", "") or "")
        model = parse_query(sql)
        metadata.append({
            "variable": str(getattr(literal, "variable", "") or ""),
            "kind": "count" if _is_count_sql(sql) else "query",
            "source_tables": model.source_tables if model is not None else [],
        })
    return metadata


def _extract_symbols(query: str) -> list[str]:
    raw_tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)
    stopwords = {
        "def", "class", "async", "await", "return", "import", "from", "as",
        "if", "else", "elif", "for", "while", "in", "not", "and", "or", "try", "except",
        "view", "query", "sql", "db", "api", "file", "line", "symbol", "snippet",
        "code", "evidence", "test", "orders", "order", "boarding", "ticket",
        "count", "select", "where", "join", "from",
        "need", "list", "app", "py", "init", "subtask", "main", "st", "replan",
        "视图", "查询", "接口", "订单", "登机牌", "机票", "航班", "报表", "方法",
        "change", "replace", "use", "with", "the", "this", "method", "function",
        "using", "into", "from", "to", "for", "in", "of", "and", "a", "an", "is",
        "用", "替换", "这个", "改成", "修改", "使用"
    }
    symbols = []
    for token in raw_tokens:
        if token.lower() not in stopwords:
            symbols.append(token)
    return symbols


def _is_code_style(token: str, query: str) -> bool:
    if "_" in token and not token.startswith("_") and not token.endswith("_"):
        return True
    if re.match(r"^[A-Z][a-z0-9]+[A-Z][a-zA-Z0-9]*$", token):
        return True
    escaped = re.escape(token)
    if re.search(rf"[`'\"]{escaped}[`'\"]|\b{escaped}\s*\(", query):
        return True
    return False


def _is_primary_symbol(symbol: str, query: str, symbol_exists_in_repo: bool) -> bool:
    stopwords = {
        "def", "class", "async", "await", "return", "import", "from", "as",
        "if", "else", "elif", "for", "while", "in", "not", "and", "or", "try", "except",
        "view", "query", "sql", "db", "api", "file", "line", "symbol", "snippet",
        "code", "evidence", "test", "orders", "order", "boarding", "ticket",
        "count", "select", "where", "join", "from",
        "need", "list", "app", "py", "init", "subtask", "main", "st", "replan",
        "视图", "查询", "接口", "订单", "登机牌", "机票", "航班", "报表", "方法",
        "change", "replace", "use", "with", "the", "this", "method", "function",
        "using", "into", "from", "to", "for", "in", "of", "and", "a", "an", "is",
        "用", "替换", "这个", "改成", "修改", "使用"
    }
    lowered = symbol.lower()
    if lowered in stopwords:
        return False
    soft_or_noise = {
        "count", "list", "app", "init", "view", "sql", "join", "query", "main", "py",
        "subtask", "st", "replan", "file"
    }
    if lowered in soft_or_noise:
        return False
    return symbol_exists_in_repo or _is_code_style(symbol, query)


def _extract_explicit_files_from_query(query: str) -> list[str]:
    files = []
    for token in re.split(r"[\s`']+", query):
        token = token.strip(".,;:?!()")
        if "." in token:
            ext = token.split(".")[-1].lower()
            if ext in {"py", "sql", "js", "ts", "tsx", "jsx", "json", "toml", "yaml", "yml", "md"}:
                files.append(token)
    return files


def _find_enclosing_python_symbol_range(lines: list[str], line_no: int) -> tuple[int, int]:
    """Find the start and end line of the python class/function enclosing line_no."""
    if not lines:
        return line_no, line_no
    
    target_idx = line_no - 1
    def_idx = -1
    for idx in range(target_idx, -1, -1):
        line = lines[idx]
        if re.match(r"^\s*(?:async\s+def|def|class)\s+\w+", line):
            def_idx = idx
            break
            
    if def_idx == -1:
        return line_no, line_no
        
    def_line = lines[def_idx]
    base_indent = len(def_line) - len(def_line.lstrip(" "))
    end_idx = def_idx
    for idx in range(def_idx + 1, len(lines)):
        candidate = lines[idx]
        if not candidate.strip():
            end_idx = idx
            continue
        indent = len(candidate) - len(candidate.lstrip(" "))
        if indent <= base_indent:
            break
        end_idx = idx
        
    return def_idx + 1, end_idx + 1


def _expand_python_symbol_range(
    lines: list[str],
    start: int,
    end: int,
    max_lines: int = 140,
) -> tuple[int, int]:
    sym_start, _ = _find_enclosing_python_symbol_range(lines, start)
    _, sym_end = _find_enclosing_python_symbol_range(lines, end)
    
    if sym_end - sym_start + 1 <= max_lines:
        return sym_start, max(sym_end, sym_start)
        
    mid = (start + end) // 2
    target_start = max(sym_start, mid - max_lines // 2)
    target_end = min(sym_end, target_start + max_lines - 1)
    return target_start, target_end


def _python_symbol_lock(
    code: str,
    *,
    file: str,
    absolute_start: int,
) -> TargetSymbol | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    if len(nodes) != 1:
        return None
    node = nodes[0]
    relative_start = int(getattr(node, "lineno", 1) or 1)
    relative_end = int(getattr(node, "end_lineno", relative_start) or relative_start)
    return TargetSymbol(
        name=node.name,
        file=file,
        range=(
            absolute_start + relative_start - 1,
            absolute_start + relative_end - 1,
        ),
    )


def _find_enclosing_sql_range(lines: list[str], line_no: int) -> tuple[int, int]:
    if not lines:
        return line_no, line_no
    
    target_idx = line_no - 1
    start_idx = target_idx
    for idx in range(target_idx, -1, -1):
        line = lines[idx]
        if ";" in line and idx < target_idx:
            start_idx = idx + 1
            break
        line_upper = line.strip().upper()
        if re.search(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\b", line_upper):
            start_idx = idx
            break
        if line_upper.startswith("SELECT"):
            start_idx = idx
            # Do not break, keep scanning upward to find CREATE VIEW if it exists
            
    end_idx = target_idx
    for idx in range(target_idx, len(lines)):
        line = lines[idx]
        if ";" in line:
            end_idx = idx
            break
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip().upper()
            if re.search(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\b", next_line) or next_line.startswith("SELECT"):
                end_idx = idx
                break
        end_idx = idx
        
    return start_idx + 1, end_idx + 1


def _expand_sql_symbol_range(
    lines: list[str],
    start: int,
    end: int,
    max_lines: int = 140,
) -> tuple[int, int]:
    sql_start, _ = _find_enclosing_sql_range(lines, start)
    _, sql_end = _find_enclosing_sql_range(lines, end)
    
    if sql_end - sql_start + 1 <= max_lines:
        return sql_start, max(sql_end, sql_start)
        
    mid = (start + end) // 2
    target_start = max(sql_start, mid - max_lines // 2)
    target_end = min(sql_end, target_start + max_lines - 1)
    return target_start, target_end


def _extract_all_search_terms(query: str) -> list[str]:
    terms = _extract_symbols(query)
    terms = [t.lower() for t in terms]
    
    domain_map = {
        "登机牌": ["boarding", "pass", "ticket"],
        "订单": ["order"],
        "机票": ["ticket", "passenger"],
        "航班": ["flight"],
    }
    for zh, eng_list in domain_map.items():
        if zh in query:
            terms.extend(eng_list)
    return list(dict.fromkeys(terms))


def _extract_patch_intent_json(context: SkillContext | None, **kwargs: object) -> dict[str, Any] | None:
    # 1. Check kwargs evidence / search_output
    for key in ("evidence", "search_output"):
        val = kwargs.get(key)
        if isinstance(val, str) and "PATCH_INTENT_JSON" in val:
            parsed = _parse_intent_from_str(val)
            if parsed:
                return parsed
        elif isinstance(val, dict) and "edit_targets" in val:
            return val

    # 2. Check context properties
    if context is not None:
        if context.context_pack is not None:
            for item in context.context_pack.evidence or []:
                if isinstance(item, dict):
                    p_json = item.get("patch_intent_json")
                    if isinstance(p_json, str):
                        try:
                            return json.loads(p_json)
                        except Exception:
                            pass
                    for k, v in item.items():
                        if isinstance(v, str) and "PATCH_INTENT_JSON" in v:
                            parsed = _parse_intent_from_str(v)
                            if parsed:
                                return parsed
                                
        metadata = getattr(context, "metadata", None)
        if isinstance(metadata, dict):
            p_json = metadata.get("patch_intent_json")
            if isinstance(p_json, str):
                try:
                    return json.loads(p_json)
                except Exception:
                    pass
            for k, v in metadata.items():
                if isinstance(v, str) and "PATCH_INTENT_JSON" in v:
                    parsed = _parse_intent_from_str(v)
                    if parsed:
                        return parsed
                    
    return None


def _parse_intent_from_str(text: str) -> dict[str, Any] | None:
    idx = text.find("PATCH_INTENT_JSON")
    if idx == -1:
        return None
    sub = text[idx + len("PATCH_INTENT_JSON") :].strip()
    start = sub.find("{")
    end = sub.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(sub[start : end + 1])
        except Exception:
            pass
    return None


def _is_editable_target(
    current_code: str,
    display_file: str,
    intended_change: str,
    context: SkillContext | None,
    *,
    edit_strategy: str = "",
    task_analysis: dict[str, object] | None = None,
) -> bool:
    target_symbols = set()
    patch_intent = _extract_patch_intent_json(context)
    if isinstance(patch_intent, dict):
        edit_targets = patch_intent.get("edit_targets") or []
        for t in edit_targets:
            if isinstance(t, dict) and t.get("symbol"):
                target_symbols.add(str(t["symbol"]))

    if isinstance(task_analysis, dict):
        for t in task_analysis.get("editable_targets") or []:
            if isinstance(t, dict) and t.get("symbol"):
                target_symbols.add(str(t["symbol"]))
    if context is not None:
        if context.patch_plan is not None:
            for sym in context.patch_plan.target_symbols:
                target_symbols.add(sym)
            for edit in context.patch_plan.edits:
                if edit.symbol:
                    target_symbols.add(edit.symbol)
        if context.context_pack is not None:
            for s_info in context.context_pack.candidate_symbols:
                s_name = str(s_info.get("name") or "")
                if s_name:
                    target_symbols.add(s_name)

    if not target_symbols:
        for sym in _extract_all_search_terms(intended_change):
            target_symbols.add(sym)

    symbol_matched = False
    for sym in target_symbols:
        if re.search(rf"(?:\b|_){re.escape(sym)}(?:\b|_)", current_code, re.IGNORECASE):
            symbol_matched = True
            break

    if not symbol_matched:
        try:
            root_path = Path.cwd()
            if context and getattr(context, "project_root", None):
                root_path = Path(context.project_root)
            file_path = root_path / display_file
            if file_path.is_file():
                file_content = file_path.read_text(encoding="utf-8", errors="replace")
                for sym in target_symbols:
                    if re.search(rf"(?:\b|_){re.escape(sym)}(?:\b|_)", file_content, re.IGNORECASE):
                        symbol_matched = True
                        break
        except Exception:
            pass

    if not symbol_matched:
        return False

    target_files = set()
    if isinstance(patch_intent, dict):
        edit_targets = patch_intent.get("edit_targets") or []
        for t in edit_targets:
            if isinstance(t, dict) and t.get("file"):
                target_files.add(str(t["file"]).replace("\\", "/"))

    if isinstance(task_analysis, dict):
        for t in task_analysis.get("editable_targets") or []:
            if isinstance(t, dict) and t.get("file"):
                target_files.add(str(t["file"]).replace("\\", "/"))
    if context is not None:
        if context.patch_plan is not None:
            for plan_file in context.patch_plan.files_to_edit:
                target_files.add(plan_file.replace("\\", "/"))
        if context.context_pack is not None:
            for f_info in context.context_pack.candidate_files:
                f_path = str(f_info.get("file") or "")
                if display_file == f_path or display_file.endswith("/" + f_path) or f_path.endswith("/" + display_file):
                    handoff_matched = True
                    break
            if not handoff_matched:
                for s_info in context.context_pack.candidate_symbols:
                    s_name = str(s_info.get("name") or "")
                    if re.search(rf"(?:\b|_){re.escape(s_name)}(?:\b|_)", current_code, re.IGNORECASE):
                        handoff_matched = True
                        break
            if not handoff_matched:
                for r_path in context.context_pack.relevant_files:
                    if display_file == r_path or display_file.endswith("/" + r_path) or r_path.endswith("/" + display_file):
                        handoff_matched = True
                        break
                        
    has_sql = bool(re.search(r"(?is)\bSELECT\b.{0,200}\bFROM\b|\b(?:FROM|JOIN)\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*", current_code))
    is_sql_change = _is_sql_view_change(intended_change)
    
    if display_file.endswith(".sql"):
        return has_sql and (symbol_matched or handoff_matched or is_sql_change)
        
    if is_sql_change:
        return (symbol_matched or handoff_matched) and has_sql
        
    return symbol_matched or handoff_matched


def _is_sql_view_change(text: str) -> bool:
    lowered = text.lower()
    has_view = "视图" in text or "view" in lowered
    has_replace = any(
        marker in lowered
        for marker in ("replace", "use", "switch", "using")
    ) or any(marker in text for marker in ("替换", "使用", "改成", "改为", "用已有视图", "用现有视图"))
    return has_view and has_replace


def _resolve_hit_path(
    root: Path,
    raw: str,
    *,
    required_line: int = 1,
) -> tuple[Path | None, str]:
    try:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
            try:
                display = str(resolved.relative_to(root)).replace("\\", "/")
            except ValueError:
                return None, raw
            if not resolved.is_file() or not _file_has_line(resolved, required_line):
                return None, raw
            return resolved, display

        rel = raw.lstrip("./")
        resolved = (root / rel).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            return None, raw
        if not resolved.is_file() or not _file_has_line(resolved, required_line):
            return None, rel
        return resolved, str(resolved.relative_to(root)).replace("\\", "/")
    except OSError:
        return None, raw


def _file_has_line(path: Path, line_no: int) -> bool:
    if line_no <= 1:
        return path.is_file()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, _line in enumerate(fh, start=1):
                if idx >= line_no:
                    return True
    except OSError:
        return False
    return False


def _edit_context_target_count(edit_context: dict[str, object] | None) -> int:
    if not edit_context:
        return 0
    targets = edit_context.get("editable_targets")
    return len(targets) if isinstance(targets, list) else 0
    if line_no <= 1:
        return path.is_file()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for idx, _line in enumerate(fh, start=1):
                if idx >= line_no:
                    return True
    except OSError:
        return False
    return False


def _edit_context_target_count(edit_context: dict[str, object] | None) -> int:
    if not edit_context:
        return 0
    targets = edit_context.get("editable_targets")
    return len(targets) if isinstance(targets, list) else 0
