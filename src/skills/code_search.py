from __future__ import annotations

import re
from pathlib import Path

from src.context.retriever import build_context_queries
from src.skills.base import SkillContext, SkillResult
from src.tools.registry import ToolRegistry


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
        snippets = _hydrate_snippets(self.project_root, search_output)
        if snippets:
            search_output = (
                f"{search_output}\n\n<context_snippets>\n"
                + "\n\n".join(snippets)
                + "\n</context_snippets>"
            )
        if errors and not outputs:
            return SkillResult(
                success=False,
                summary="code_search failed: " + "; ".join(errors),
                missing_info=tuple(errors),
                metadata={"search_output": search_output},
            )
        return SkillResult(
            success=True,
            summary=f"code_search completed {len(calls)} batched call(s).",
            metadata={"search_output": search_output},
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
    pattern = _search_pattern(query, queries)
    calls: list[tuple[str, dict[str, object]]] = []

    if context.context_pack is not None and context.context_pack.search_plan:
        for plan in context.context_pack.search_plan[:4]:
            plan_terms = [
                str(term)
                for term in plan.patterns[:12]
                if str(term).strip()
            ]
            plan_pattern = _search_pattern(query, plan_terms) or pattern
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


def _search_pattern(query: str, queries: list[str]) -> str:
    text = query.lower()
    if "登机牌" in query or "boarding" in text or "boarding_pass" in text:
        terms = [r"登机牌", r"boarding_pass", r"boarding"]
        if "视图" in query or "view" in text:
            terms.extend([
                r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW",
                r"--\s*视图",
                r"视图[:：]",
            ])
        return "|".join(terms)
    if ("视图" in query or "view" in text) and (
        "定义" in query or "definition" in text or "create" in text
    ):
        return r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW|--\s*视图|视图[:：]"
    if "视图" in query or "view" in text:
        return r"CREATE\s+(?:OR\s+REPLACE\s+)?VIEW|\bview\b|视图"
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
        }
    ]
    if not terms:
        terms = queries[:12]
    return "|".join(re.escape(term) for term in terms)


def _format_call(name: str, args: dict[str, object]) -> str:
    if name == "grep_search":
        return (
            f"[grep_search pattern={args.get('pattern')!r} "
            f"path={args.get('path')!r} include={args.get('include')!r}]"
        )
    if name == "map_search":
        return f"[map_search query={args.get('query')!r}]"
    return f"[{name} {args!r}]"


_GREP_HIT = re.compile(r"^(?P<path>[^:\n]+):(?P<line>\d+):", re.MULTILINE)
_MAP_HIT = re.compile(
    r"^-\s+(?P<path>[^:\s]+):(?P<start>\d+)(?:-(?P<end>\d+))?",
    re.MULTILINE,
)


def _hydrate_snippets(
    project_root: Path,
    search_output: str,
    *,
    max_files: int = 6,
    padding: int = 3,
    max_chars: int = 10_000,
) -> list[str]:
    """Read bounded snippets around repo_map/grep hits; never expose raw full-file IO."""
    hits: dict[str, tuple[int, int]] = {}

    def add(path: str, start: int, end: int | None = None) -> None:
        raw = path.strip().replace("\\", "/")
        if not raw or raw.startswith("["):
            return
        try:
            resolved = Path(raw)
            if not resolved.is_absolute():
                resolved = (project_root / raw.lstrip("./")).resolve()
            rel = str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
        except (OSError, ValueError):
            rel = raw.lstrip("./")
        start = max(1, start)
        end = max(start, end or start)
        current = hits.get(rel)
        if current is None:
            hits[rel] = (start, end)
            return
        hits[rel] = (min(current[0], start), max(current[1], end))

    for match in _GREP_HIT.finditer(search_output):
        add(match.group("path"), int(match.group("line")))
    for match in _MAP_HIT.finditer(search_output):
        add(
            match.group("path"),
            int(match.group("start")),
            int(match.group("end") or match.group("start")),
        )

    snippets: list[str] = []
    used_chars = 0
    root = project_root.resolve()
    for rel, (start, end) in list(hits.items())[:max_files]:
        try:
            path = (root / rel).resolve()
            path.relative_to(root)
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except (OSError, ValueError):
            continue
        if not lines:
            continue
        s = max(1, start - padding)
        e = min(len(lines), end + padding)
        body = "\n".join(
            f"{line_no}: {lines[line_no - 1]}"
            for line_no in range(s, e + 1)
        )
        chunk = f'<snippet path="{rel}" lines="{s}-{e}">\n{body}\n</snippet>'
        if used_chars + len(chunk) > max_chars:
            break
        snippets.append(chunk)
        used_chars += len(chunk)
    return snippets
