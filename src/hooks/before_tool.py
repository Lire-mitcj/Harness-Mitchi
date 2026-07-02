from __future__ import annotations

import logging
from collections.abc import Mapping, Set
from typing import Any

from src.hooks.preflight.static_constraints import inspect_static_constraints
from src.hooks.preflight.fact_locking import inspect_fact_locking_async

log = logging.getLogger(__name__)


def inspect_tool_request(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
) -> str | None:
    """Delegates to Layer 1 static constraints validation hook."""
    return inspect_static_constraints(
        tool_name=tool_name,
        arguments=arguments,
        allowed_tools=allowed_tools,
    )


async def inspect_tool_request_async(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
    has_compile_error: bool = False,
    search_history: list[dict[str, Any]] | None = None,
    repo_map: Any = None,
    embedder: Any = None,
    embeddings_cache: dict[str, list[float]] | None = None,
    gravity_controller: Any = None,
    checklist: list[str] | None = None,
    context_anchors_code: list[dict[str, Any]] | None = None,
    raw_evidence_store: list[dict[str, Any]] | None = None,
    git_diff: str | None = None,
    modified_files: list[str] | None = None,
) -> str | None:
    """Delegates sequentially to Layer 1 (Static Constraints) and Layer 2 (Fact Locking) preflight hooks."""
    # 1. Layer 1: Static Constraints precheck
    err = inspect_static_constraints(
        tool_name=tool_name,
        arguments=arguments,
        allowed_tools=allowed_tools,
    )
    if err:
        return err

    # 2. Layer 2: Dynamic Fact Locking precheck
    return await inspect_fact_locking_async(
        tool_name=tool_name,
        arguments=arguments,
        has_compile_error=has_compile_error,
        search_history=search_history,
        repo_map=repo_map,
        embedder=embedder,
        embeddings_cache=embeddings_cache,
        gravity_controller=gravity_controller,
        checklist=checklist,
        context_anchors_code=context_anchors_code,
        raw_evidence_store=raw_evidence_store,
        git_diff=git_diff,
        modified_files=modified_files,
    )
