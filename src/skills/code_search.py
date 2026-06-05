from __future__ import annotations

import json
import re
from pathlib import Path

from src.context.retriever import build_context_queries
from src.skills.base import SkillContext, SkillResult
from src.tools.registry import ToolRegistry

_HYDRATION_VERSION = "edit_context_hydration_v3_pipe_hits"


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
    pattern = _search_pattern(query, queries, root_query=context.user_request)
    calls: list[tuple[str, dict[str, object]]] = []

    if context.context_pack is not None and context.context_pack.search_plan:
        for plan in context.context_pack.search_plan[:4]:
            plan_terms = [
                str(term)
                for term in plan.patterns[:12]
                if str(term).strip()
            ]
            plan_pattern = _search_pattern(query, plan_terms, root_query=context.user_request) or pattern
            globs = plan.globs or ("*",)
            path = plan.files[0] if len(plan.files) == 1 else "."
            for glob in globs[:2]:
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
            pattern=pattern,
            project_root=project_root,
            search_paths=search_paths,
        )
        return calls

    _append_fallback_search_calls(
        calls,
        queries=queries,
        pattern=pattern,
        project_root=project_root,
        search_paths=search_paths,
    )
    return calls


def _append_fallback_search_calls(
    calls: list[tuple[str, dict[str, object]]],
    *,
    queries: list[str],
    pattern: str,
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
        for include in ("*.py", "*.sql"):
            add(
                "grep_search",
                {
                    "pattern": pattern,
                    "path": grep_path,
                    "include": include,
                    "max_results": 80,
                },
            )


def _search_pattern(query: str, queries: list[str], root_query: str = "") -> str:
    text = query.lower()
    root_text = root_query.lower()
    
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
        
    if "视图" in query or "view" in text or "视图" in root_query or "view" in root_text:
        if "定义" in query or "definition" in text or "create" in text or "定义" in root_query or "definition" in root_text or "create" in root_text:
            terms.extend([
                r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW",
                r"--\s*视图",
                r"视图[:：]",
            ])
        else:
            terms.extend([
                r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW",
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


def _merge_ranges(ranges: list[tuple[int, int]], gap: int = 20) -> list[tuple[int, int]]:
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
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
        merged = _merge_ranges(ranges, gap=20)
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
        if path.suffix in {".py", ".ts", ".tsx", ".js", ".jsx"}:
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
        if used_chars + len(chunk) > max_chars:
            failures.append(
                f"{display}:{s}-{e}: snippet skipped by display token budget"
            )
            continue
        snippets.append(chunk)
        used_chars += len(chunk)
    return _HydratedSearch(
        snippets=snippets,
        edit_context=_build_edit_plan_context(
            editable_targets,
            intended_change=intended_change,
        ),
        hit_count=len(file_hits),
        failures=failures,
        hit_paths=[f"{display}:{start}-{end}" for _path, display, start, end in ordered_hits],
    )


def _build_edit_plan_context(
    editable_targets: list[dict[str, object]],
    *,
    intended_change: str,
) -> dict[str, object] | None:
    if not editable_targets:
        return None
    scope = [
        str(target["file"])
        for target in editable_targets
        if str(target.get("file") or "")
    ]
    return {
        "schema": "mitkii.edit_context.v1",
        "builder": "EditPlanBuilder",
        "code_edit_ready": True,
        "snippets": editable_targets[:4],
        "editable_targets": editable_targets[:4],
        "intended_change": intended_change,
        "acceptance_criteria": ["Target behavior is changed as requested."],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": list(dict.fromkeys(scope)),
        },
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
) -> tuple[int, int]:
    sym_start, _ = _find_enclosing_python_symbol_range(lines, start)
    _, sym_end = _find_enclosing_python_symbol_range(lines, end)
    return sym_start, max(sym_end, sym_start)


def _is_editable_target(
    current_code: str,
    display_file: str,
    intended_change: str,
    context: SkillContext | None,
) -> bool:
    # 1. 命中用户明确 symbol
    user_symbols = _extract_symbols(intended_change)
    for sym in user_symbols:
        if re.search(rf"\b{re.escape(sym)}\b", current_code):
            return True
            
    # 2. 命中 plan 指定 file/symbol
    if context is not None and context.patch_plan is not None:
        plan = context.patch_plan
        for plan_file in plan.files_to_edit:
            if display_file == plan_file or display_file.endswith("/" + plan_file) or plan_file.endswith("/" + display_file):
                return True
        for plan_sym in plan.target_symbols:
            if re.search(rf"\b{re.escape(plan_sym)}\b", current_code):
                return True
        for edit in plan.edits:
            if edit.path and (display_file == edit.path or display_file.endswith("/" + edit.path) or edit.path.endswith("/" + display_file)):
                return True
            if edit.symbol and re.search(rf"\b{re.escape(edit.symbol)}\b", current_code):
                return True

    # 3. 命中 SQL 结构
    if bool(re.search(r"(?is)\bSELECT\b.{0,200}\bFROM\b|\b(?:FROM|JOIN)\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*", current_code)):
        return True

    return False


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
