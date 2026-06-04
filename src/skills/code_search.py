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
        if not query:
            return SkillResult(
                success=False,
                summary="code_search requires extra_query or user_request.",
                missing_info=("extra_query",),
            )

        calls = _search_calls(context, query)
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
) -> list[tuple[str, dict[str, object]]]:
    queries = build_context_queries(query, limit=10)
    if not queries:
        queries = [query]
    pattern = "|".join(re.escape(term) for term in queries[:12])
    calls: list[tuple[str, dict[str, object]]] = []

    if context.context_pack is not None and context.context_pack.search_plan:
        for plan in context.context_pack.search_plan[:4]:
            plan_pattern = "|".join(re.escape(term) for term in plan.patterns[:12]) or pattern
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
        return calls

    for map_query in queries[:3]:
        calls.append(("map_search", {"query": map_query, "limit": 20}))
    for include in ("*.py", "*.sql"):
        calls.append((
            "grep_search",
            {
                "pattern": pattern,
                "path": ".",
                "include": include,
                "max_results": 80,
            },
        ))
    return calls


def _format_call(name: str, args: dict[str, object]) -> str:
    if name == "grep_search":
        return (
            f"[grep_search pattern={args.get('pattern')!r} "
            f"path={args.get('path')!r} include={args.get('include')!r}]"
        )
    if name == "map_search":
        return f"[map_search query={args.get('query')!r}]"
    return f"[{name} {args!r}]"
