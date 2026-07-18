from __future__ import annotations

import logging
from collections.abc import Mapping, Set
from typing import Any

from pathlib import Path

from src.hooks.preflight.context_window import inspect_context_window_disk
from src.hooks.preflight.decision_edit_intent import inspect_decision_edit_intent
from src.hooks.preflight.fact_locking import inspect_fact_locking_async
from src.hooks.preflight.static_constraints import inspect_static_constraints

log = logging.getLogger(__name__)


def inspect_tool_request(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
    manifest: Any = None,
) -> str | None:
    """Delegates to Layer 1 static constraints validation hook."""
    return inspect_static_constraints(
        tool_name=tool_name,
        arguments=arguments,
        allowed_tools=allowed_tools,
        manifest=manifest,
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
    manifest: Any = None,
    project_root: Path | None = None,
    edit_recovery: bool = False,
    rounds_since_last_edit: int = 0,
) -> str | None:
    """Run static constraints, then fact-locking preflight hooks."""
    # 1. Layer 1: Static Constraints precheck
    err = inspect_static_constraints(
        tool_name=tool_name,
        arguments=arguments,
        allowed_tools=allowed_tools,
        manifest=manifest,
    )
    if err:
        return err

    # 1b. Disk-backed context_window validation for decision_edit
    err = inspect_context_window_disk(
        tool_name,
        arguments,
        project_root=project_root,
        manifest=manifest,
    )
    if err:
        return err

    err = inspect_decision_edit_intent(
        tool_name,
        arguments,
        manifest=manifest,
    )
    if err:
        return err

    # 2. Dynamic Fact Locking: block reads of SATISFIED evidence, allow STALE.
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
        manifest=manifest,
        edit_recovery=edit_recovery,
        rounds_since_last_edit=rounds_since_last_edit,
    )
