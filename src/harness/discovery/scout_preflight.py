from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.tools.registry import ToolRegistry


def derive_grep_patterns(user_request: str) -> list[str]:
    """Build ripgrep patterns from the user request (deterministic, no LLM)."""
    req = user_request.lower()
    patterns: list[str] = []

    if any(k in user_request for k in ("注册", "登录", "接口")) or "register" in req:
        patterns.append(r"register|signup|sign_up")
    if any(k in user_request for k in ("事务", "事务异常")) or "transaction" in req:
        patterns.append(r"transaction|commit|rollback|session\.commit|db\.session|事务")
    if any(k in user_request for k in ("数据库", "mysql", "sql")) or "database" in req:
        patterns.append(r"OperationalError|IntegrityError|commit|rollback|BEGIN|COMMIT")
    if "500" in req or "异常" in user_request or "error" in req:
        patterns.append(r"except|Exception|HTTPException|raise")

    if not patterns:
        # Last resort: pull latin identifiers from the request
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", user_request)
        if tokens:
            patterns.append("|".join(dict.fromkeys(tokens[:4])))
        else:
            patterns.append(r"def |class ")

    return patterns[:4]


async def run_scout_preflight(
    tools: ToolRegistry,
    user_request: str,
) -> str:
    """Harness-driven grep before Scout LLM — guarantees evidence even if 7B skips tools."""
    patterns = derive_grep_patterns(user_request)

    async def _grep(pattern: str, include: str) -> str | None:
        result = await tools.call(
            "grep_search",
            {
                "pattern": pattern,
                "path": ".",
                "include": include,
                "max_results": 40 if include == "*.py" else 30,
            },
        )
        if not result.success or not result.output:
            return None
        if result.output.strip() == "No matches found.":
            return None
        label = f"## grep /{pattern}/"
        if include != "*.py":
            label += " (sql+cfg)"
        return f"{label}\n{result.output.strip()}"

    py_chunks = [
        c for c in await asyncio.gather(*[_grep(p, "*.py") for p in patterns]) if c
    ]
    if py_chunks:
        return "\n\n".join(py_chunks)

    wide_chunks = [
        c
        for c in await asyncio.gather(
            *[_grep(p, "*.{sql,py,env,yml,yaml}") for p in patterns]
        )
        if c
    ]
    return "\n\n".join(wide_chunks)
