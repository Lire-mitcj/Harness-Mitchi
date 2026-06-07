from __future__ import annotations

import json
import re
from pathlib import Path

from src.context.retriever import build_context_queries
from src.skills.base import SkillContext, SkillResult
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
        if errors and not outputs:
            return SkillResult(
                success=False,
                summary="code_search failed: " + "; ".join(errors),
                missing_info=tuple(errors),
                metadata={
                    "search_output": search_output,
                    "edit_context_targets": str(_edit_context_target_count(hydrated.edit_context)),
                    "hydration_root": str(self.project_root),
                    "hydration_hits": str(hydrated.hit_count),
                    "hydration_failures": "; ".join(hydrated.failures[:4]),
                    "hydration_version": _HYDRATION_VERSION,
                    "edit_context_json": edit_context_json,
                    "hydration_hit_paths": "; ".join(hydrated.hit_paths[:8]),
                },
            )
        return SkillResult(
            success=True,
            summary=f"code_search completed {len(calls)} batched call(s).",
            metadata={
                "search_output": search_output,
                "edit_context_targets": str(_edit_context_target_count(hydrated.edit_context)),
                "hydration_root": str(self.project_root),
                "hydration_hits": str(hydrated.hit_count),
                "hydration_failures": "; ".join(hydrated.failures[:4]),
                "hydration_version": _HYDRATION_VERSION,
                "edit_context_json": edit_context_json,
                "hydration_hit_paths": "; ".join(hydrated.hit_paths[:8]),
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
    ) -> None:
        self.snippets = snippets
        self.edit_context = edit_context
        self.hit_count = hit_count
        self.failures = failures
        self.hit_paths = hit_paths


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
    editable_targets: list[dict[str, object]] = []
    used_chars = 0
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
            target_start, target_end = _expand_python_symbol_range(
                lines,
                target_start,
                target_end,
                max_lines=1000,
            )
        elif path.suffix == ".sql":
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
                    "start_line": target_start,
                    "end_line": target_end,
                    "current_code": target_code,
                    "intended_change": intended_change,
                    "acceptance_criteria": ["Target behavior is changed as requested."],
                })
            elif _is_sql_view_change(intended_change):
                failures.append(
                    f"{display}:{target_start}-{target_end}: not editable for "
                    "SQL/view query change; no SELECT/FROM/JOIN table reference"
                )

        s = max(1, start - padding)
        e = min(len(lines), end + padding)
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
                for line_no in range(start, end + 1)
            )
            core_chunk = (
                f'<snippet path="{display}" lines="{start}-{end}" status="truncated_to_core">\n'
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
                    f'<snippet path="{display}" lines="{start}-{end}" status="omitted_due_to_budget">\n'
                    f"# Full snippet omitted due to token budget limits. Use read_file/view_file to view lines {start}-{end}.\n"
                    f"</snippet>"
                )
                if used_chars + len(ref_chunk) <= max_chars:
                    snippets.append(ref_chunk)
                    used_chars += len(ref_chunk)
                    failures.append(f"{display}:{s}-{e}: snippet omitted due to budget")
                else:
                    failures.append(f"{display}:{s}-{e}: snippet skipped by display token budget")

    return _HydratedSearch(
        snippets=snippets,
        edit_context=_build_edit_plan_context(
            editable_targets,
            intended_change=intended_change,
            project_root=project_root,
            context=context,
            **kwargs,
        ),
        hit_count=len(file_hits),
        failures=failures,
        hit_paths=[f"{display}:{start}-{end}" for _path, display, start, end in ordered_hits],
    )


def _find_all_views(project_root: Path) -> list[str]:
    views = set()
    pattern = re.compile(
        r"\bCREATE\s+[^;]*?\bVIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\.\"'\`\[\]]+)",
        re.IGNORECASE
    )
    for path in project_root.rglob("*"):
        try:
            if path.suffix.lower() in {".sql", ".py"} and path.is_file():
                content = path.read_text(encoding="utf-8", errors="ignore")
                for match in pattern.finditer(content):
                    raw_name = match.group(1)
                    cleaned = re.sub(r'[\"\'\`\[\]]', '', raw_name)
                    views.add(cleaned.lower())
                    if '.' in cleaned:
                        views.add(cleaned.split('.')[-1].lower())
        except Exception:
            pass
    return sorted(list(views))


def _find_all_view_details(project_root: Path) -> dict[str, dict[str, object]]:
    details: dict[str, dict[str, object]] = {}
    pattern = re.compile(
        r"\bCREATE\s+[^;]*?\bVIEW\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?P<name>[a-zA-Z0-9_\.\"'\`\[\]]+)\s+AS\s+"
        r"(?P<select>SELECT\b[\s\S]*?)(?:;|$)",
        re.IGNORECASE,
    )
    for path in project_root.rglob("*"):
        try:
            if path.suffix.lower() not in {".sql", ".py"} or not path.is_file():
                continue
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for match in pattern.finditer(content):
            raw_name = match.group("name")
            cleaned = re.sub(r'[\"\'\`\[\]]', '', raw_name).strip()
            if not cleaned:
                continue
            columns = _extract_select_output_columns(match.group("select"))
            replaces = _extract_all_tables_from_sql(match.group("select"))
            detail = {
                "role": "replacement_source",
                "kind": "database_view",
                "name": cleaned,
                "columns": columns,
                "replaces_objects": replaces,
                "evidence": [str(path.relative_to(project_root)).replace("\\", "/")],
                "confidence": 0.95,
            }
            for key in {cleaned.lower(), cleaned.split(".")[-1].lower()}:
                details[key] = detail
    return details


def _extract_select_output_columns(sql: str) -> list[str]:
    try:
        import sqlparse

        parsed = sqlparse.parse(sql)
        if not parsed:
            return []
        stmt = parsed[0]
        tokens = list(stmt.tokens)
        select_idx = -1
        from_idx = -1
        for idx, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword.DML and token.value.upper() == "SELECT":
                select_idx = idx
            elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                from_idx = idx
                break
        if select_idx < 0 or from_idx <= select_idx:
            return []
        columns: list[str] = []
        for token in tokens[select_idx + 1 : from_idx]:
            if token.is_whitespace:
                continue
            if isinstance(token, sqlparse.sql.IdentifierList):
                items = list(token.get_identifiers())
            elif isinstance(token, sqlparse.sql.Identifier):
                items = [token]
            else:
                items = [token] if token.value.strip() and token.value.strip() != "," else []
            for item in items:
                if "*" in item.value:
                    continue
                alias = item.get_alias() if hasattr(item, "get_alias") else None
                real_name = item.get_real_name() if hasattr(item, "get_real_name") else None
                name = alias or real_name or item.value
                clean = _clean_sql_identifier(name)
                if clean and clean.lower() not in {"as", "case", "when", "then", "else", "end"}:
                    columns.append(clean)
        return list(dict.fromkeys(columns))
    except Exception:
        return []


def _extract_all_tables_from_sql(sql: str) -> list[str]:
    tables: list[str] = []
    for match in re.finditer(
        r"\b(?:FROM|JOIN)\s+[`\"\[]?([a-zA-Z0-9_.]+)[`\"\]]?",
        sql,
        re.IGNORECASE,
    ):
        tables.append(match.group(1).split(".")[-1].lower())
    return list(dict.fromkeys(tables))


def _clean_sql_identifier(raw: str) -> str:
    text = raw.strip().split(".")[-1]
    text = re.sub(r'[\"\'\`\[\]]', "", text)
    text = re.sub(r"\W+$", "", text)
    return text.strip()




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
        if "登机牌" in intended_change or "boarding" in intended_change.lower():
            if "boarding" in view_lower or "ticket" in view_lower:
                score += 20
        if "订单" in intended_change or "order" in intended_change.lower():
            if "order" in view_lower:
                score += 20
        if "机票" in intended_change or "ticket" in intended_change.lower():
            if "ticket" in view_lower or "passenger" in view_lower:
                score += 20

        if score > best_score:
            best_score = score
            best_view = view
            
    if best_score <= 0:
        if len(views) == 1:
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
    
    replaces = []
    for target in editable_targets:
        code = str(target.get("current_code") or "")
        replaces.extend(_extract_all_tables_from_python_code(code))
    replaces = list(dict.fromkeys(replaces))
    
    resolved_deps = []
    if target_view:
        detail = view_details.get(target_view.lower()) or view_details.get(target_view.split(".")[-1].lower())
        dep_replaces = replaces
        columns: list[str] = []
        evidence: list[str] = []
        if isinstance(detail, dict):
            columns = [str(col) for col in detail.get("columns") or [] if str(col).strip()]
            evidence = [str(item) for item in detail.get("evidence") or [] if str(item).strip()]
            dep_replaces = list(dict.fromkeys(
                dep_replaces
                + [str(obj) for obj in detail.get("replaces_objects") or [] if str(obj).strip()]
            ))
        resolved_deps.append({
            "role": "replacement_source",
            "kind": "database_view",
            "name": target_view,
            "columns": columns,
            "replaces_objects": dep_replaces,
            "evidence": evidence,
            "confidence": 0.95
        })
        
    edit_targets = []
    for target in editable_targets[:4]:
        edit_targets.append({
            "file": str(target.get("file") or ""),
            "symbol": str(target.get("symbol") or target.get("name") or target_symbol),
            "current_code": str(target.get("current_code") or ""),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line")
        })
        
    return {
        "schema": "mitkii.edit_context.v2",
        "builder": "EditPlanBuilder",
        "code_edit_ready": True,
        "task_intent": {
            "operation": operation,
            "target_symbol": target_symbol,
            "goal": "use existing implementation/object instead of current query builder" if is_view else "edit code as requested"
        },
        "edit_targets": edit_targets,
        "resolved_dependencies": resolved_deps,
        "constraints": [
            "Do not invent dependencies",
            "Only use resolved_dependencies",
            "Preserve public function signature",
            "Preserve returned field names used by callers",
            "For SQL replacement, rebuild SELECT from replacement_source.columns and remove only replaces_objects JOINs"
        ],
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
        "acceptance_criteria": ["Target behavior is changed as requested."],
    }


def _extract_symbols(query: str) -> list[str]:
    raw_tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", query)
    stopwords = {
        "def", "class", "async", "await", "return", "import", "from", "as",
        "if", "else", "elif", "for", "while", "in", "not", "and", "or", "try", "except",
        "view", "query", "sql", "db", "api", "file", "line", "symbol", "snippet",
        "code", "evidence", "test", "orders", "order", "boarding", "ticket",
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


def _is_editable_target(
    current_code: str,
    display_file: str,
    intended_change: str,
    context: SkillContext | None,
) -> bool:
    user_symbols = _extract_all_search_terms(intended_change)
    symbol_matched = False
    for sym in user_symbols:
        if re.search(rf"(?:\b|_){re.escape(sym)}(?:\b|_)", current_code, re.IGNORECASE):
            symbol_matched = True
            break
            
    handoff_matched = False
    if context is not None:
        if context.patch_plan is not None:
            plan = context.patch_plan
            for plan_file in plan.files_to_edit:
                if display_file == plan_file or display_file.endswith("/" + plan_file) or plan_file.endswith("/" + display_file):
                    handoff_matched = True
                    break
            if not handoff_matched:
                for plan_sym in plan.target_symbols:
                    if re.search(rf"(?:\b|_){re.escape(plan_sym)}(?:\b|_)", current_code, re.IGNORECASE):
                        handoff_matched = True
                        break
            if not handoff_matched:
                for edit in plan.edits:
                    if edit.path and (display_file == edit.path or display_file.endswith("/" + edit.path) or edit.path.endswith("/" + display_file)):
                        handoff_matched = True
                        break
                    if edit.symbol and re.search(rf"(?:\b|_){re.escape(edit.symbol)}(?:\b|_)", current_code, re.IGNORECASE):
                        handoff_matched = True
                        break
                        
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
    return (
        "视图" in text
        or "查询" in text
        or "sql" in lowered
        or "query" in lowered
        or "view" in lowered
    )


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


def _extract_all_tables_from_python_code(code: str) -> list[str]:
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
        re.DOTALL
    )
    tables = set()
    for match in string_pattern.finditer(code):
        quote_content = match.group("quote")
        inner = quote_content[3:-3] if quote_content.startswith(('"""', "'''")) else quote_content[1:-1]
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            for m in re.finditer(r"\b(?:FROM|JOIN)\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?", inner, re.IGNORECASE):
                tables.add(m.group(1).split(".")[-1].lower())
    return sorted(list(tables))
