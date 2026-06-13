from __future__ import annotations

import ast
import asyncio
import difflib
import inspect
import json
import re
import textwrap
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.skills.base import SkillContext, SkillResult
from src.skills.sql_ast import (
    extract_sql_literals_from_python,
    parse_query,
    rewrite_query_with_view,
)

EditComplete = Callable[[list[dict[str, object]]], Awaitable[str]]


class CodeEditSkill:
    name = "code_edit"

    def __init__(
        self,
        *,
        project_root: Path,
        llm_complete: EditComplete | None = None,
        edit_plan_timeout: float = 12.0,
        patch_window_timeout: float = 25.0,
    ) -> None:
        self.project_root = project_root.resolve()
        self.llm_complete = llm_complete
        self.edit_plan_timeout = edit_plan_timeout
        self.patch_window_timeout = patch_window_timeout

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        plan = context.patch_plan
        if plan is None:
            return await self._run_focused_edit(context, **kwargs)
        if not plan.is_executable():
            return SkillResult(
                success=False,
                summary="patch_plan is not executable.",
                missing_info=plan.missing_info or ("non_executable_patch_plan",),
                requires_fallback=True,
            )

        import hashlib
        changed: list[str] = []
        errors: list[str] = []
        original_files: dict[str, str] = {}
        for edit in plan.edits:
            path = _resolve_under_root(self.project_root, edit.path)
            if path is None:
                errors.append(f"{edit.path}: outside project root")
                continue
            if not path.is_file():
                errors.append(f"{edit.path}: file not found")
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"{edit.path}: read failed: {exc}")
                continue
            original_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            occurrences = content.count(edit.old_string)
            if occurrences != 1:
                errors.append(
                    f"{edit.path}: old_string occurrence count is {occurrences}, expected 1"
                )
                continue
            new_content = content.replace(edit.old_string, edit.new_string, 1)
            new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            if original_hash == new_hash:
                errors.append(f"{edit.path}: file hash did not change (no modification made)")
                continue
            rel = _rel_display(self.project_root, path)
            original_files.setdefault(rel, content)
            try:
                path.write_text(new_content, encoding="utf-8")
                # Verify immediately from disk
                written_content = path.read_text(encoding="utf-8")
                written_hash = hashlib.sha256(written_content.encode("utf-8")).hexdigest()
                if written_hash == original_hash:
                    errors.append(f"{edit.path}: written content hash equals original content hash")
                    continue
            except OSError as exc:
                errors.append(f"{edit.path}: write/read verification failed: {exc}")
                continue
            changed.append(rel)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(changed),
                missing_info=tuple(errors),
                requires_fallback=True,
            )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} patch edit(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={"original_files_json": json.dumps(original_files)},
        )

    async def _run_focused_edit(
        self,
        context: SkillContext,
        **kwargs: object,
    ) -> SkillResult:
        if self.llm_complete is None:
            return SkillResult(
                success=False,
                summary="code_edit requires llm_complete when patch_plan is absent.",
                missing_info=("llm_complete",),
            )
        instruction = str(kwargs.get("instruction") or context.user_request).strip()
        evidence = str(kwargs.get("search_output") or "").strip()
        progress_events: list[dict[str, object]] = []
        progress_callback = kwargs.get("progress_callback")
        await _emit_progress(
            progress_events,
            progress_callback,
            "code_edit.started",
            instruction=instruction,
        )
        if not evidence:
            return SkillResult(
                success=False,
                summary="code_edit requires search_output evidence.",
                missing_info=("search_output",),
            )
        contract = kwargs.get("handoff_contract")
        edit_context = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
        task_analysis = kwargs.get("task_analysis")
        if isinstance(edit_context, dict) and isinstance(task_analysis, dict):
            strategy = str(task_analysis.get("edit_strategy") or "").strip()
            if strategy:
                edit_context["edit_strategy"] = strategy
            if task_analysis.get("edit_ready") is not None:
                edit_context["harness_edit_ready"] = bool(task_analysis.get("edit_ready"))
            if task_analysis.get("acceptance_contract"):
                edit_context["acceptance_contract"] = task_analysis.get("acceptance_contract")
        strategy_for_view = _edit_strategy(edit_context)
        if (
            isinstance(edit_context, dict)
            and strategy_for_view == "sql_view_rewrite"
            and not edit_context.get("target_view")
        ):
            replacement_source_name = ""
            resolved_deps = edit_context.get("resolved_dependencies")
            if isinstance(resolved_deps, list):
                for dep in resolved_deps:
                    if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                        replacement_source_name = str(dep.get("name") or "").strip()
                        break
            if replacement_source_name:
                edit_context["target_view"] = replacement_source_name
            else:
                t_view = _extract_target_view_from_contract(contract if isinstance(contract, dict) else None)
                if not t_view:
                    t_view = _select_target_view(
                        evidence=evidence,
                        instruction=instruction,
                        handoff_contract=contract if isinstance(contract, dict) else None,
                    )
                if t_view:
                    edit_context["target_view"] = t_view

        _lock_edit_context_symbols(edit_context)

        ready_error = _validate_edit_context_ready(self.project_root, edit_context)
        if ready_error:
            return SkillResult(
                success=False,
                summary=f"code_edit not ready: {ready_error}",
                missing_info=(ready_error,),
                metadata={"raw_preview": _preview(evidence)},
            )
        target_symbol = _target_symbol_from_edit_context(edit_context)
        await _emit_progress(
            progress_events,
            progress_callback,
            "target.selected",
            symbol=target_symbol,
        )
        plan_messages = _edit_plan_messages(
            instruction=instruction,
            evidence=evidence,
            handoff_contract=contract if isinstance(contract, dict) else None,
        )
        try:
            plan_raw = await asyncio.wait_for(
                self.llm_complete(plan_messages),
                timeout=self.edit_plan_timeout,
            )
        except TimeoutError:
            return SkillResult(
                success=False,
                summary="EditPlanBuilder timed out.",
                missing_info=("edit_plan_timeout",),
                metadata={"progress_events": json.dumps(progress_events)},
            )
        plan_payload = _extract_json_payload(plan_raw)
        edit_targets = _editable_targets(edit_context)
        if plan_payload is None:
            payload = None
            raw_preview = plan_raw
        else:
            payload, raw_preview = await self._build_payload_from_plan(
                plan_payload,
                edit_targets=edit_targets,
                instruction=instruction,
                evidence=evidence,
                progress_events=progress_events,
                progress_callback=progress_callback,
            )
        if payload is None:
            return SkillResult(
                success=False,
                summary=(
                    "code_edit could not produce a valid structured edit plan, "
                    "and no deterministic SQL view edit could be derived."
                ),
                missing_info=("edit_plan_json",),
                metadata={
                    "raw_preview": _preview(raw_preview),
                    "progress_events": json.dumps(progress_events),
                },
            )

        edits = payload.get("edits")
        confidence = payload.get("confidence", 0.0)
        missing = payload.get("missing_info") or ()
        if not isinstance(confidence, int | float) or float(confidence) < 0.75:
            return SkillResult(
                success=False,
                summary=(
                    "code_edit confidence too low "
                    f"(confidence={confidence}, missing_info={missing})"
                ),
                missing_info=tuple(str(item) for item in missing if isinstance(item, str)),
                metadata={
                    "raw_preview": _preview(raw_preview),
                    "progress_events": json.dumps(progress_events),
                },
            )
        if not isinstance(edits, list) or not edits:
            return SkillResult(
                success=False,
                summary="code_edit model returned no edits.",
                missing_info=("edits",),
                metadata={
                    "raw_preview": _preview(raw_preview),
                    "progress_events": json.dumps(progress_events),
                },
            )

        changed: list[str] = []
        errors: list[str] = []
        original_files: dict[str, str] = {}
        for idx, item in enumerate(edits):
            if not isinstance(item, dict):
                errors.append(f"edits[{idx}] must be an object")
                continue
            path, old_string = _resolve_edit_item_target(item, edit_targets)
            new_string = str(item.get("new_string") or "")
            if not path or not old_string or not new_string:
                errors.append(
                    f"edits[{idx}] requires path+old_string+new_string"
                )
                continue
            if old_string == new_string:
                errors.append(f"edits[{idx}] old_string and new_string must differ")
                continue
            if len(old_string) > 50_000 or len(new_string) > 50_000:
                errors.append(f"edits[{idx}] is too large; use a smaller exact snippet")
                continue
            target = _find_target_for_edit(path, old_string, edit_targets)
            if target is None:
                errors.append(f"edits[{idx}] is not contained in an editable target")
                continue
            current_code = str(target.get("current_code") or "")
            symbol = str(target.get("symbol") or target.get("name") or target_symbol)
            if current_code.count(old_string) != 1:
                errors.append(
                    f"edits[{idx}] old_string occurrence count in target symbol is "
                    f"{current_code.count(old_string)}, expected 1"
                )
                continue
            syntax_error = _validate_python_patch(
                path=path,
                current_code=current_code,
                old_string=old_string,
                new_string=new_string,
            )
            if syntax_error:
                errors.append(f"edits[{idx}] {syntax_error}")
                continue
            result = _apply_exact_edit(
                self.project_root,
                path=path,
                old_string=old_string,
                new_string=new_string,
                original_files=original_files,
                target_item=target,
            )
            if result.startswith("ok:"):
                changed.append(result[3:])
                await _emit_progress(
                    progress_events,
                    progress_callback,
                    "patch.applied",
                    file=path,
                    symbol=symbol,
                )
            else:
                errors.append(result)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(dict.fromkeys(changed)),
                missing_info=tuple(errors),
                metadata={
                    "raw_preview": _preview(raw_preview),
                    "progress_events": json.dumps(progress_events),
                },
            )
        await _emit_progress(
            progress_events,
            progress_callback,
            "validator.started",
            files=list(dict.fromkeys(changed)),
        )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} edit_file action(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={
                "raw_preview": _preview(raw_preview),
                "confidence": str(confidence),
                "original_files_json": json.dumps(original_files),
                "progress_events": json.dumps(progress_events),
            },
        )

    async def _build_payload_from_plan(
        self,
        plan_payload: dict[str, object],
        *,
        edit_targets: list[dict[str, object]],
        instruction: str,
        evidence: str,
        progress_events: list[dict[str, object]] | None = None,
        progress_callback: object = None,
    ) -> tuple[dict[str, object] | None, str]:
        edits = plan_payload.get("edits")
        confidence = plan_payload.get("confidence", 0.0)
        missing = plan_payload.get("missing_info") or []
        if not isinstance(edits, list) or not edits:
            return plan_payload, json.dumps(plan_payload, ensure_ascii=False)

        edit_context = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
        fallback_view = ""
        if isinstance(edit_context, dict):
            resolved_deps = edit_context.get("resolved_dependencies")
            if isinstance(resolved_deps, list):
                for dep in resolved_deps:
                    if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                        fallback_view = str(dep.get("name") or "").strip()
                        break
            if not fallback_view:
                fallback_view = str(edit_context.get("target_view") or "").strip()
        target_symbol = _target_symbol_from_edit_context(edit_context)
            
        built_edits: list[dict[str, object]] = []
        raw_parts = [json.dumps(plan_payload, ensure_ascii=False)]
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return None, raw_parts[0]
            target_index = edit.get("target_index")
            if not isinstance(target_index, int) or not (0 <= target_index < len(edit_targets)):
                return None, raw_parts[0]
            if target_symbol and not _target_matches_symbol(edit_targets[target_index], target_symbol):
                return {
                    "edits": [],
                    "confidence": 0.0,
                    "missing_info": ["target_symbol"],
                }, (
                    f"selected target_index {target_index} does not match "
                    f"required target_symbol {target_symbol}"
                )
            current_code = str(edit_targets[target_index].get("current_code") or "")
            display_code = str(edit_targets[target_index].get("display_code") or current_code)
            file = str(edit_targets[target_index].get("file") or "")
            
            operation = edit.get("operation")
            is_dynamic_op = operation in ("dynamic_sql_rewrite", "dynamic_count_query_rewrite")
            if _is_count_target(edit_context, edit) and not is_dynamic_op and not _target_has_sql_query(current_code):
                return {
                    "edits": [],
                    "confidence": 0.0,
                    "missing_info": ["target_not_hydrated"],
                }, "count target current_code is not hydrated with SELECT/FROM SQL"
            target_view = str(edit.get("target_view") or fallback_view).strip()
            if operation == "dynamic_count_query_rewrite":
                patch_window = _extract_patch_window(
                    current_code,
                    edit_context or {},
                    edit,
                    target=edit_targets[target_index],
                    target_view=target_view,
                )
                if patch_window is None:
                    return {
                        "edits": [],
                        "confidence": 0.0,
                        "missing_info": ["patch_window"],
                    }, "failed to resolve a writable patch window"
                await _emit_progress(
                    progress_events,
                    progress_callback,
                    "patch_window.resolved",
                    file=file,
                    symbol=patch_window["symbol"],
                    absolute_start_line=patch_window["absolute_start_line"],
                    absolute_end_line=patch_window["absolute_end_line"],
                    target_type=patch_window.get("target_type") or "",
                    parent_symbol=patch_window.get("parent_symbol") or "",
                    local_target=patch_window.get("local_target") or "",
                    canonical_old_string=patch_window.get("canonical_old_string") or "",
                )
                await _emit_progress(
                    progress_events,
                    progress_callback,
                    "patch_builder.started",
                    file=file,
                    symbol=patch_window["symbol"],
                )
                try:
                    patch_raw = await asyncio.wait_for(
                        self.llm_complete(
                            _patch_window_messages(
                                instruction=instruction,
                                target_view=target_view,
                                target_sql_kind="count",
                                target_symbol=str(patch_window["symbol"]),
                                patch_window=patch_window,
                                constraints=_patch_constraints(edit_context),
                            )
                        ),
                        timeout=self.patch_window_timeout,
                    )
                except TimeoutError:
                    return {
                        "edits": [],
                        "confidence": 0.0,
                        "missing_info": ["patch_window_timeout"],
                    }, "PatchWindowBuilder timed out"
                raw_parts.append(patch_raw)
                await _emit_progress(
                    progress_events,
                    progress_callback,
                    "patch_builder.finished",
                    file=file,
                    symbol=patch_window["symbol"],
                )
                patch_payload = _parse_patch_window_payload(patch_raw)
                if patch_payload is None:
                    return {
                        "edits": [],
                        "confidence": 0.0,
                        "missing_info": ["patch_window_json"],
                    }, "\n\n".join(raw_parts)
                if "replacement" in patch_payload:
                    old_str = str(patch_window.get("canonical_old_string") or "")
                    new_str = str(patch_payload["replacement"].get("new_string") or "")
                    if old_str and new_str:
                        if old_str.endswith("\n") and not new_str.endswith("\n"):
                            new_str += "\n"
                        elif old_str.endswith("\r\n") and not new_str.endswith("\r\n"):
                            new_str += "\r\n"
                        old_indent = len(old_str) - len(old_str.lstrip(" \t"))
                        new_indent = len(new_str) - len(new_str.lstrip(" \t"))
                        if old_indent > 0 and new_indent == 0:
                            indent_prefix = old_str[:old_indent]
                            new_str_lines = new_str.splitlines(keepends=True)
                            new_str = "".join(indent_prefix + line for line in new_str_lines)
                    patch_payload["patches"] = [{
                        "old_string": old_str,
                        "new_string": new_str
                    }]
                patch_errors = _validate_patch_window_payload(
                    patch_payload,
                    current_code=current_code,
                    patch_window=patch_window,
                    target_view=target_view,
                    target_sql_kind="count",
                    target_sql_variable=_target_sql_variable(edit_context, edit),
                    constraints=_patch_constraints(edit_context),
                    path=file,
                )
                if patch_errors:
                    return {
                        "edits": [],
                        "confidence": 0.0,
                        "missing_info": patch_errors,
                    }, "\n\n".join(raw_parts)
                for patch in patch_payload["patches"]:
                    built_edits.append({
                        "path": file,
                        "old_string": patch["old_string"],
                        "new_string": patch["new_string"],
                    })
                confidence = min(
                    float(confidence) if isinstance(confidence, int | float) else 0.0,
                    float(patch_payload.get("confidence", 0.0)),
                )
                continue

            new_string = None
            if operation in ("replace_sql_source", "replace_dependency", "use_existing") and target_view:
                tmp_ctx = dict(edit_context or {})
                tmp_ctx["target_view"] = target_view
                if operation == "count_query_view_rewrite":
                    tmp_ctx["target_sql_kind"] = "count"
                if "resolved_dependencies" in tmp_ctx:
                    resolved_deps = []
                    for dep in tmp_ctx["resolved_dependencies"]:
                        if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                            new_dep = dict(dep)
                            new_dep["name"] = target_view
                            resolved_deps.append(new_dep)
                        else:
                            resolved_deps.append(dep)
                    tmp_ctx["resolved_dependencies"] = resolved_deps
                try:
                    new_string = generate_sql_patch(current_code, tmp_ctx)
                except ProjectionMappingError:
                    return {
                        "edits": [],
                        "confidence": 0.0,
                        "missing_info": ["column_mapping"],
                    }, "failed to rewrite projection: missing column mapping"
            
            if new_string is not None:
                built_edits.append({
                    "target_index": target_index,
                    "new_string": _clean_llm_code(new_string),
                })
                continue

            replacement_messages = _replacement_messages(
                instruction=instruction,
                file=file,
                target_index=target_index,
                current_code=current_code,
                edit_plan=edit,
                evidence=evidence,
                target_view=target_view,
            )
            replacement_raw = await self.llm_complete(replacement_messages)
            raw_parts.append(replacement_raw)
            replacement_payload = _extract_json_payload(replacement_raw)
            if replacement_payload is None:
                return None, "\n\n".join(raw_parts)
            new_string = str(replacement_payload.get("new_string") or "")
            if not new_string:
                return None, "\n\n".join(raw_parts)
            built_edits.append({
                "target_index": target_index,
                "new_string": _clean_llm_code(new_string),
            })
        return {
            "edits": built_edits,
            "confidence": confidence,
            "missing_info": missing,
        }, "\n\n".join(raw_parts)


class ProjectionMappingError(Exception):
    pass


def rewrite_query_source(
    current_sql: str,
    target_source: str,
    replaces_objects: list[str],
    target_columns: list[str] | None = None,
    column_defaults: dict[str, str] | None = None,
    column_sources: list[dict[str, object]] | None = None,
    source_to_view_column: dict[str, object] | None = None,
) -> str:
    model = parse_query(current_sql)
    target_alias = "v"
    if not replaces_objects:
        if model is not None and model.from_table:
            replaces_objects = [model.from_table]
            if not target_columns and model.from_alias:
                target_alias = model.from_alias

    ast_rewrite = rewrite_query_with_view(
        current_sql,
        target_source=target_source,
        replaces_objects=replaces_objects,
        target_columns=target_columns,
        column_defaults=column_defaults,
        column_sources=column_sources,
        source_to_view_column=source_to_view_column,
        target_alias=target_alias,
    )
    if ast_rewrite is not None:
        return ast_rewrite

    if target_columns:
        raise ProjectionMappingError("Projection rewrite failed due to missing column mapping")
    return current_sql


def replace_table_with_view_in_sql(current_sql: str, target_view: str) -> str:
    return rewrite_query_source(current_sql, target_view, [])


def generate_sql_patch(current_code: str, edit_context: dict) -> str | None:
    dep_name = ""
    replaces = []
    columns = []
    column_defaults = {}
    column_sources: list[dict[str, object]] = []
    source_to_view_column: dict[str, object] = {}
    resolved_deps = edit_context.get("resolved_dependencies") or []
    for dep in resolved_deps:
        if isinstance(dep, dict) and dep.get("role") == "replacement_source":
            dep_name = str(dep.get("name") or "").strip()
            replaces = list(dep.get("replaces_objects") or [])
            columns = list(dep.get("columns") or [])
            column_sources = [
                item
                for item in dep.get("column_sources") or []
                if isinstance(item, dict)
            ]
            raw_source_mapping = dep.get("source_to_view_column") or {}
            if isinstance(raw_source_mapping, dict):
                source_to_view_column = {
                    str(key): str(value)
                    for key, value in raw_source_mapping.items()
                }
            raw_defaults = dep.get("column_defaults") or {}
            if isinstance(raw_defaults, dict):
                column_defaults = {
                    str(key): str(value)
                    for key, value in raw_defaults.items()
                }
            break
            
    if not dep_name:
        dep_name = str(edit_context.get("target_view") or "").strip()
        
    if not dep_name:
        return None

    literals = extract_sql_literals_from_python(current_code)
    literal = _select_sql_literal_for_patch(literals, edit_context)
    if literal is not None:
        target_sql_kind = str(edit_context.get("target_sql_kind") or "").strip().lower()
        effective_columns = [] if target_sql_kind == "count" else columns
        modified_inner = rewrite_query_source(
            literal.sql,
            dep_name,
            replaces,
            effective_columns,
            column_defaults,
            column_sources,
            source_to_view_column,
        )
        new_quote_content = f"{literal.quote}{modified_inner}{literal.quote}"
        return (
            current_code[: literal.start_offset]
            + new_quote_content
            + current_code[literal.end_offset :]
        )
            
    return None


def deterministic_replace_sql_with_view(current_code: str, view_name: str) -> str | None:
    return generate_sql_patch(current_code, {"target_view": view_name})


def _select_sql_literal_for_patch(
    literals: list[object],
    edit_context: dict,
):
    if not literals:
        return None

    target_variable = str(edit_context.get("target_sql_variable") or "").strip()
    if not target_variable:
        task_intent = edit_context.get("task_intent")
        if isinstance(task_intent, dict):
            target_variable = str(task_intent.get("target_sql_variable") or "").strip()
    if target_variable:
        for literal in literals:
            if str(getattr(literal, "variable", "") or "") == target_variable:
                return literal

    target_kind = str(edit_context.get("target_sql_kind") or "").strip().lower()
    if not target_kind:
        task_intent = edit_context.get("task_intent")
        if isinstance(task_intent, dict):
            target_kind = str(task_intent.get("target_sql_kind") or "").strip().lower()
    if target_kind == "count":
        for literal in literals:
            if _is_count_query(str(getattr(literal, "sql", "") or "")):
                return literal

    return literals[0]


def _is_count_query(sql: str) -> bool:
    model = parse_query(sql)
    if model is None:
        return bool(re.search(r"(?is)\bSELECT\s+COUNT\s*\(", sql))
    return any(
        item.kind == "aggregate" and re.search(r"(?i)\bCOUNT\s*\(", item.expression)
        for item in model.selects
    )




def _clean_llm_code(code: str) -> str:
    res = code
    res = res.replace('\\"""', '"""').replace('\\\'\'\'', '\'\'\'')
    res = res.replace('\"\"\"', '"""').replace('\'\'\'', '\'\'\'')
    return res


def _edit_strategy(ctx: dict[str, object] | None) -> str:
    if not isinstance(ctx, dict):
        return ""
    return str(ctx.get("edit_strategy") or "").strip() or "general_edit"



def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        root = project_root.resolve()
        raw = rel.replace("\\", "/").strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (root / raw.lstrip("./")).resolve()
        path.relative_to(root)
        return path
    except (OSError, ValueError):
        return None


def _rel_display(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        text = str(path).replace("\\", "/")
        return text if Path(text).is_absolute() else text.lstrip("./")


def _apply_exact_edit(
    project_root: Path,
    *,
    path: str,
    old_string: str,
    new_string: str,
    original_files: dict[str, str] | None = None,
    target_item: dict[str, object] | None = None,
) -> str:
    target_path = _resolve_under_root(project_root, path)
    if target_path is None:
        return f"{path}: outside project root"
    if not target_path.is_file():
        return f"{path}: file not found"
    try:
        content = target_path.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{path}: read failed: {exc}"
    
    import hashlib
    original_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    range_restricted = False
    if target_item is not None:
        start_line = target_item.get("start_line")
        end_line = target_item.get("end_line")
        if isinstance(start_line, int) and isinstance(end_line, int) and start_line > 0 and end_line >= start_line:
            range_restricted = True

    if range_restricted:
        content_lines = content.splitlines(keepends=True)
        if start_line <= len(content_lines) and end_line <= len(content_lines):
            func_body = "".join(content_lines[start_line - 1 : end_line])
            occurrences = func_body.count(old_string)
            if occurrences != 1:
                return f"{path}: old_string occurrence count in target symbol is {occurrences}, expected 1"
            new_func_body = func_body.replace(old_string, new_string, 1)
            new_content = "".join(content_lines[:start_line - 1]) + new_func_body + "".join(content_lines[end_line:])
        else:
            occurrences = content.count(old_string)
            if occurrences != 1:
                return f"{path}: old_string occurrence count is {occurrences}, expected 1"
            new_content = content.replace(old_string, new_string, 1)
    else:
        occurrences = content.count(old_string)
        if occurrences != 1:
            return f"{path}: old_string occurrence count is {occurrences}, expected 1"
        new_content = content.replace(old_string, new_string, 1)

    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    if original_hash == new_hash:
        return f"{path}: file hash did not change (no modification made)"
        
    rel = _rel_display(project_root, target_path)
    if original_files is not None:
        original_files.setdefault(rel, content)
    try:
        target_path.write_text(new_content, encoding="utf-8")
        # Verify immediately from disk
        written_content = target_path.read_text(encoding="utf-8")
        written_hash = hashlib.sha256(written_content.encode("utf-8")).hexdigest()
        if written_hash == original_hash:
            return f"{path}: written content hash equals original content hash"
    except OSError as exc:
        return f"{path}: write/read verification failed: {exc}"
    return f"ok:{rel}"


def _editable_targets(ctx: dict[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(ctx, dict):
        return []
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    edit_targets = ctx.get("edit_targets")
    merged: list[dict[str, object]] = []
    for idx, target in enumerate(targets or []):
        if not isinstance(target, dict):
            continue
        item = dict(target)
        if isinstance(edit_targets, list) and idx < len(edit_targets):
            rich = edit_targets[idx]
            if isinstance(rich, dict):
                for key in ("symbol", "name", "sql_queries", "sql_literals", "sql_presence"):
                    if key not in item and key in rich:
                        item[key] = rich[key]
        merged.append(item)
    return merged


def _target_symbol_from_edit_context(ctx: dict[str, object] | None) -> str:
    if not isinstance(ctx, dict):
        return ""
    task_intent = ctx.get("task_intent")
    if isinstance(task_intent, dict):
        value = str(task_intent.get("target_symbol") or "").strip()
        if value and value != "目标代码":
            return value
    value = str(ctx.get("target_symbol") or "").strip()
    return "" if value == "目标代码" else value


def _target_matches_symbol(target: dict[str, object], target_symbol: str) -> bool:
    symbol = str(target.get("symbol") or target.get("name") or "").strip()
    if symbol == target_symbol:
        return True
    current_code = str(target.get("current_code") or "")
    return bool(
        re.search(
            rf"(?m)^\s*(?:async\s+def|def|class)\s+{re.escape(target_symbol)}\b",
            current_code,
        )
    )


_EDIT_OPERATIONS = {
    "general_edit",
    "replace_dependency",
    "use_existing",
    "replace_sql_source",
    "count_query_view_rewrite",
    "dynamic_count_query_rewrite",
    "dynamic_sql_rewrite",
}


def _declared_target_symbol(current_code: str) -> str:
    try:
        tree = ast.parse(current_code)
    except SyntaxError:
        return ""
    declarations = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    if len(declarations) == 1:
        return declarations[0]
    if not declarations and tree.body:
        return "<module>"
    return ""


def _lock_edit_context_symbols(ctx: dict[str, object] | None) -> None:
    if not isinstance(ctx, dict):
        return
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    if not isinstance(targets, list) or not targets:
        return
    locked_symbols: list[str] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        symbol = str(target.get("symbol") or target.get("name") or "").strip()
        if not symbol:
            symbol = _declared_target_symbol(str(target.get("current_code") or ""))
            if symbol:
                target["symbol"] = symbol
        if not symbol:
            continue
        locked_symbols.append(symbol)
        if "symbol_lock" not in target:
            target["symbol_lock"] = {
                "name": symbol,
                "file": str(target.get("file") or ""),
                "range": [target.get("start_line"), target.get("end_line")],
                "immutable": True,
            }
    task_intent = ctx.get("task_intent")
    if not isinstance(task_intent, dict):
        task_intent = {"operation": "general_edit"}
        ctx["task_intent"] = task_intent
    if not str(task_intent.get("target_symbol") or "").strip():
        unique_symbols = list(dict.fromkeys(locked_symbols))
        if len(unique_symbols) == 1:
            task_intent["target_symbol"] = unique_symbols[0]


def _validate_symbol_lock(target: dict[str, object]) -> str:
    symbol = str(target.get("symbol") or target.get("name") or "").strip()
    if not symbol:
        return "missing symbol"
    if symbol in _EDIT_OPERATIONS or symbol.lower() == "modify":
        return f"operation value '{symbol}' cannot be used as target_symbol"
    declared = _declared_target_symbol(str(target.get("current_code") or ""))
    if declared and declared != symbol:
        return f"AST symbol '{declared}' does not match target symbol '{symbol}'"
    lock = target.get("symbol_lock")
    if lock is None:
        return ""
    if not isinstance(lock, dict):
        return "symbol_lock is not an object"
    lock_name = str(lock.get("name") or "").strip()
    lock_file = str(lock.get("file") or "").replace("\\", "/")
    target_file = str(target.get("file") or "").replace("\\", "/")
    lock_range = lock.get("range")
    expected_range = [target.get("start_line"), target.get("end_line")]
    if lock.get("immutable") is not True:
        return "symbol_lock is not immutable"
    if lock_name != symbol or lock_file != target_file or lock_range != expected_range:
        return "symbol_lock does not match editable target"
    return ""


def _is_count_target(
    ctx: dict[str, object] | None,
    item: dict[str, object] | None = None,
) -> bool:
    values: list[str] = []
    if isinstance(item, dict):
        values.append(str(item.get("target_sql_kind") or ""))
        values.append(str(item.get("operation") or ""))
    if isinstance(ctx, dict):
        values.append(str(ctx.get("target_sql_kind") or ""))
        task_intent = ctx.get("task_intent")
        if isinstance(task_intent, dict):
            values.append(str(task_intent.get("target_sql_kind") or ""))
            values.append(str(task_intent.get("operation") or ""))
    return any(value.strip().lower() in {"count", "count_query_view_rewrite", "dynamic_count_query_rewrite"} for value in values)


def _filter_authoritative_targets(
    hydrated_targets: list[dict[str, object]],
    edit_context: dict[str, object],
    project_root: Path,
) -> list[dict[str, object]] | None:
    specs = _authoritative_target_specs(edit_context)
    if not specs:
        return None
    filtered = [
        target
        for target in hydrated_targets
        if any(_target_matches_spec(target, spec) for spec in specs)
    ]
    if filtered:
        return filtered
    return _materialize_authoritative_targets(specs, edit_context, project_root)


def _authoritative_target_specs(edit_context: dict[str, object]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    patch_intent = edit_context.get("patch_intent")
    if isinstance(patch_intent, dict):
        raw_targets = patch_intent.get("edit_targets")
        if isinstance(raw_targets, list):
            specs.extend(t for t in raw_targets if isinstance(t, dict))
    raw_targets = edit_context.get("edit_targets")
    if isinstance(raw_targets, list):
        specs.extend(t for t in raw_targets if isinstance(t, dict))
    task_intent = edit_context.get("task_intent")
    if isinstance(task_intent, dict):
        symbol = str(task_intent.get("target_symbol") or "").strip()
        if symbol:
            specs.append({"symbol": symbol})
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for spec in specs:
        file = str(spec.get("file") or "").replace("\\", "/").strip()
        symbol = str(spec.get("symbol") or spec.get("symbol_or_api") or "").strip()
        start = spec.get("start_line") or spec.get("line_start") or spec.get("line")
        end = spec.get("end_line") or spec.get("line_end") or spec.get("line")
        try:
            start_i = int(start) if start else 0
        except (TypeError, ValueError):
            start_i = 0
        try:
            end_i = int(end) if end else 0
        except (TypeError, ValueError):
            end_i = 0
        if not file and not symbol:
            continue
        key = (file, symbol, start_i, end_i)
        if key in seen:
            continue
        seen.add(key)
        normalized.append({
            "file": file,
            "symbol": symbol,
            "start_line": start_i,
            "end_line": end_i,
        })
    return normalized


def _target_matches_spec(target: dict[str, object], spec: dict[str, object]) -> bool:
    target_file = str(target.get("file") or "").replace("\\", "/")
    spec_file = str(spec.get("file") or "").replace("\\", "/")
    if spec_file and not (
        target_file == spec_file
        or target_file.endswith("/" + spec_file)
        or spec_file.endswith("/" + target_file)
    ):
        return False

    spec_symbol = str(spec.get("symbol") or "").strip()
    target_symbol = str(target.get("symbol") or "").strip()
    current_code = str(target.get("current_code") or "")
    if spec_symbol:
        if target_symbol and target_symbol == spec_symbol:
            return True
        if re.search(rf"\b(?:async\s+def|def|class)\s+{re.escape(spec_symbol)}\b", current_code):
            return True
        if re.search(rf"(?:\b|_){re.escape(spec_symbol)}(?:\b|_)", current_code):
            return True
        return False

    spec_start = spec.get("start_line")
    spec_end = spec.get("end_line")
    target_start = target.get("start_line")
    target_end = target.get("end_line")
    if (
        isinstance(spec_start, int)
        and isinstance(spec_end, int)
        and spec_start > 0
        and spec_end >= spec_start
        and isinstance(target_start, int)
        and isinstance(target_end, int)
    ):
        return spec_start <= target_end and target_start <= spec_end
    return bool(spec_file)


def _materialize_authoritative_targets(
    specs: list[dict[str, object]],
    edit_context: dict[str, object],
    project_root: Path,
) -> list[dict[str, object]]:
    materialized: list[dict[str, object]] = []
    intended_change = str(edit_context.get("intended_change") or "").strip()
    acceptance = (
        edit_context.get("acceptance_criteria")
        or edit_context.get("acceptance")
        or "Target behavior is changed as requested."
    )
    for spec in specs:
        file = str(spec.get("file") or "").replace("\\", "/").strip()
        if not file:
            continue
        path = _resolve_under_root(project_root, file)
        if path is None or not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        start = spec.get("start_line")
        end = spec.get("end_line")
        symbol = str(spec.get("symbol") or "").strip()
        if not (
            isinstance(start, int)
            and isinstance(end, int)
            and start > 0
            and end >= start
        ):
            found = _find_symbol_range(lines, symbol) if symbol else None
            if found is None:
                continue
            start, end = found
        start = max(1, int(start))
        end = min(len(lines), int(end))
        if end < start:
            continue
        current_code = "\n".join(lines[start - 1 : end])
        if not current_code.strip():
            continue
        materialized.append({
            "file": file,
            "symbol": symbol,
            "start_line": start,
            "end_line": end,
            "current_code": current_code,
            "display_code": current_code,
            "hydration_simplified": False,
            "intended_change": intended_change,
            "acceptance_criteria": acceptance,
            "materialized_from_patch_intent": True,
        })
    return materialized


def _find_symbol_range(lines: list[str], symbol: str) -> tuple[int, int] | None:
    if not symbol:
        return None
    for idx, line in enumerate(lines):
        if not re.search(rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b", line):
            continue
        start = idx + 1
        indent = len(line) - len(line.lstrip(" "))
        end = idx + 1
        for j in range(idx + 1, len(lines)):
            candidate = lines[j]
            if not candidate.strip():
                end = j + 1
                continue
            next_indent = len(candidate) - len(candidate.lstrip(" "))
            if next_indent <= indent:
                break
            end = j + 1
        return start, end
    return None


def _replace_marker_json(text: str, marker: str, payload: dict[str, object]) -> str:
    if marker not in text:
        return text.rstrip() + f"\n\n{marker}\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    prefix, rest = text.split(marker, 1)
    block = _extract_marker_json(text, marker)
    if not isinstance(block, dict):
        return prefix.rstrip() + f"\n\n{marker}\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    raw = json.dumps(block, ensure_ascii=False, indent=2)
    replacement = json.dumps(payload, ensure_ascii=False, indent=2)
    if raw in rest:
        return prefix + marker + rest.replace(raw, replacement, 1)
    return prefix.rstrip() + f"\n\n{marker}\n" + replacement


def _resolve_edit_item_target(
    item: dict[str, object],
    targets: list[dict[str, object]],
) -> tuple[str, str]:
    path = str(item.get("path") or "")
    old_string = str(item.get("old_string") or "")
    if path and old_string:
        return path, old_string
    return path, old_string


def _edit_plan_messages(
    *,
    instruction: str,
    evidence: str,
    handoff_contract: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    contract_text = (
        json.dumps(handoff_contract, ensure_ascii=False, indent=2)
        if handoff_contract
        else "{}"
    )
    prepared_evidence = _prepare_edit_plan_evidence(evidence)
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII EditPlanBuilder. Output ONE raw JSON object only. "
                "No markdown, no prose, no Python code. You must not generate "
                "replacement code or specify any view names. "
                "Select editable_targets by index and specify the operation. "
                "If task_intent.target_symbol is set, you must select only an "
                "editable_target whose symbol matches it. "
                "Use operation=count_query_view_rewrite or dynamic_count_query_rewrite when the requested SQL target "
                "is a COUNT/count_sql query. "
                "Required schema: "
                '{"edits":[{"target_index":0,"operation":"replace_sql_source"}],'
                '"confidence":1.0,"missing_info":[]}. '
                "If no provided editable_target is clearly correct, return "
                '{"edits":[],"confidence":0.0,"missing_info":["target"]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                "HARD_HANDOFF_CONTRACT_JSON\n"
                f"{contract_text[:6000]}\n\n"
                "<code_search_results>\n"
                f"{prepared_evidence}\n"
                "</code_search_results>"
            ),
        },
    ]


def _edit_messages(
    *,
    instruction: str,
    evidence: str,
    handoff_contract: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Backward-compatible alias for tests and older callers."""
    return _edit_plan_messages(
        instruction=instruction,
        evidence=evidence,
        handoff_contract=handoff_contract,
    )


def _patch_window_messages(
    *,
    instruction: str,
    target_view: str,
    target_sql_kind: str,
    target_symbol: str,
    patch_window: dict[str, object],
    constraints: list[str],
) -> list[dict[str, object]]:
    t_view = target_view or str(edit_plan.get("target_view") or "").strip()
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII ReplacementBuilder. Output ONE raw JSON object only. "
                "No markdown, no prose. Required schema: "
                '{"new_string":"complete replacement for current_code"}. '
                "Return only the replacement for the single provided current_code. "
                "Preserve indentation, imports, surrounding behavior, and style. "
                "Do not edit files or symbols outside this current_code. "
                "CRITICAL: If replacing a query or dependency, you MUST only "
                f"use the replacement source '{t_view}'. "
                "Do NOT use any other name. The name MUST exist in the "
                "repository's resolved dependencies or available views."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                f"Target view: {target_view}\n"
                f"Target SQL kind: {target_sql_kind}\n"
                f"Target symbol: {target_symbol}\n\n"
                "PATCH_WINDOW_JSON\n"
                f"{json.dumps(patch_window, ensure_ascii=False, indent=2)}\n\n"
                "CONSTRAINTS_JSON\n"
                f"{json.dumps(constraints, ensure_ascii=False)}"
            ),
        },
    ]


def _find_query_construction_range_text(content: str) -> tuple[int, int, str] | None:
    """Find the enclosed range of statements constructing query clauses and SQL variables.
    Returns (start_line_1_indexed, end_line_1_indexed, text_content).
    """
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    def is_query_construction_stmt(node: ast.AST) -> bool:
        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id in {"clauses", "where_clauses", "where_clause", "count_sql", "list_sql", "sql"}:
                        return True
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Attribute) and isinstance(child.func.value, ast.Name):
                    if child.func.value.id in {"clauses", "where_clauses"} and child.func.attr == "append":
                        return True
        return False

    body_stmts = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_stmts = node.body
            break
    if not body_stmts:
        body_stmts = tree.body

    selected = []
    for stmt in body_stmts:
        if is_query_construction_stmt(stmt):
            selected.append(stmt)

    if not selected:
        return None

    start_line = min(getattr(stmt, "lineno", 1) for stmt in selected)
    end_line = max(getattr(stmt, "end_lineno", getattr(stmt, "lineno", 1)) for stmt in selected)

    lines = content.splitlines(keepends=True)
    if start_line <= len(lines) and end_line <= len(lines):
        text = "".join(lines[start_line - 1 : end_line])
        return start_line, end_line, text
    return None


def _find_variable_assignment_range_text(
    content: str,
    var_name: str,
) -> tuple[int, int, str] | None:
    """Find the exact assignment statement text for var_name in content."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return None

    ranges = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    start = getattr(node, "lineno", 1)
                    end = getattr(node, "end_lineno", start)
                    ranges.append((start, end))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == var_name:
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                ranges.append((start, end))

    if len(ranges) != 1:
        return None

    start, end = ranges[0]
    lines = content.splitlines(keepends=True)
    if start <= len(lines) and end <= len(lines):
        return start, end, "".join(lines[start - 1 : end])
    return None


def _extract_patch_window(
    current_code: str,
    edit_context: dict[str, object],
    edit_plan: dict[str, object],
    *,
    target: dict[str, object] | None = None,
    target_view: str,
) -> dict[str, object] | None:
    lines = current_code.splitlines(keepends=True)
    if not lines:
        return None
    task_intent = edit_context.get("task_intent")
    if not isinstance(task_intent, dict):
        task_intent = {}
    operation = str(
        edit_plan.get("operation")
        or edit_context.get("operation")
        or task_intent.get("operation")
        or ""
    ).strip()
    target_data = target or {}
    symbol = str(target_data.get("symbol") or target_data.get("name") or "").strip()
    parent_symbol = str(
        task_intent.get("parent_symbol")
        or edit_context.get("parent_symbol")
        or ""
    ).strip()
    local_target = str(
        task_intent.get("local_target")
        or edit_context.get("local_target")
        or _target_sql_variable(edit_context, edit_plan)
        or ""
    ).strip()
    target_type = str(
        task_intent.get("target_type")
        or edit_context.get("target_type")
        or ("sql_variable" if local_target else "symbol")
    ).strip()
    requires_local_window = target_type == "sql_variable" or operation in {
        "dynamic_sql_rewrite",
        "dynamic_count_query_rewrite",
    }
    if requires_local_window:
        if not symbol or not parent_symbol or parent_symbol != symbol or not local_target:
            return None
        assignment = _find_variable_assignment_range_text(current_code, local_target)
        if assignment is None:
            return None
        assignment_start, assignment_end, assignment_text = assignment
        function_start = target_data.get("start_line")
        if not isinstance(function_start, int):
            return None
        target_sql_kind = str(
            edit_plan.get("target_sql_kind")
            or edit_context.get("target_sql_kind")
            or task_intent.get("target_sql_kind")
            or ""
        ).strip().lower()
        return {
            "file": str(target_data.get("file") or ""),
            "symbol": symbol,
            "absolute_start_line": function_start + assignment_start - 1,
            "absolute_end_line": function_start + assignment_end - 1,
            "window_code": assignment_text,
            "canonical_old_string": assignment_text,
            "target_view": target_view,
            "target_sql_kind": target_sql_kind,
            "target_type": "sql_variable",
            "parent_symbol": parent_symbol,
            "local_target": local_target,
            "operation": operation,
            "target_id": f"{parent_symbol}.{local_target}",
        }
    target_sql_kind = str(
        edit_plan.get("target_sql_kind")
        or edit_context.get("target_sql_kind")
        or (
            edit_context.get("task_intent", {}).get("target_sql_kind")
            if isinstance(edit_context.get("task_intent"), dict)
            else ""
        )
        or ""
    ).strip().lower()
    target_sql_variable = _target_sql_variable(edit_context, edit_plan)
    priorities = [
        target_sql_variable,
        "count_sql",
        "COUNT(",
        "count_query",
        "total_sql",
        "from_clause",
        "JOIN",
        "where_clause",
    ]
    hit = -1
    for needle in priorities:
        if not needle:
            continue
        lowered = needle.lower()
        hit = next(
            (idx for idx, line in enumerate(lines) if lowered in line.lower()),
            -1,
        )
        if hit >= 0:
            break
    if hit < 0:
        return None

    operation = str(
        edit_plan.get("operation")
        or edit_context.get("operation")
        or (
            edit_context.get("task_intent", {}).get("operation")
            if isinstance(edit_context.get("task_intent"), dict)
            else ""
        )
        or ""
    ).strip()

    is_dynamic_count = (operation == "dynamic_count_query_rewrite")
    canonical_old_string = None
    ast_applied = False

    if is_dynamic_count:
        ast_res = _find_query_construction_range_text(current_code)
        if ast_res:
            ast_start, ast_end, ast_text = ast_res
            start = ast_start - 1
            end = ast_end
            canonical_old_string = ast_text
            ast_applied = True
        else:
            start = max(0, hit - 80)
            end = min(len(lines), hit + 80)
            canonical_old_string = "".join(lines[start:end])
            ast_applied = True

    if not ast_applied:
        start = max(0, hit - 30)
        end = min(len(lines), hit + 81)
        if end - start > 120:
            start = max(0, hit - 25)
            end = min(len(lines), start + 120)
            if hit >= end:
                end = hit + 1
                start = max(0, end - 120)

    target_data = target or {}
    function_start = target_data.get("start_line")
    if not isinstance(function_start, int):
        function_start = 1
    symbol = str(
        target_data.get("symbol")
        or target_data.get("name")
        or _target_symbol_from_edit_context(edit_context)
    ).strip()

    if not ast_applied and target_sql_variable:
        assignment = _find_variable_assignment_range_text(current_code, target_sql_variable)
        canonical_old_string = assignment[2] if assignment else None

    task_intent = edit_context.get("task_intent") if isinstance(edit_context, dict) else None
    target_type = ""
    local_target = ""
    parent_symbol = ""
    if isinstance(task_intent, dict):
        target_type = str(task_intent.get("target_type") or "").strip()
        local_target = str(task_intent.get("local_target") or "").strip()
        parent_symbol = str(task_intent.get("parent_symbol") or "").strip()
    if not target_type and isinstance(edit_context, dict):
        target_type = str(edit_context.get("target_type") or "").strip()
    if not local_target and isinstance(edit_context, dict):
        local_target = str(edit_context.get("local_target") or "").strip()
    if not parent_symbol and isinstance(edit_context, dict):
        parent_symbol = str(edit_context.get("parent_symbol") or "").strip()

    # Fallbacks
    if not target_type:
        if (
            (task_intent and (task_intent.get("target_sql_variable") or task_intent.get("local_target")))
            or (isinstance(edit_context, dict) and (edit_context.get("target_sql_variable") or edit_context.get("local_target")))
            or target_sql_variable
        ):
            target_type = "sql_variable"
    if not local_target:
        local_target = target_sql_variable
    if not parent_symbol:
        local_symbol = target_data.get("symbol") or target_data.get("name")
        parent_symbol = str(local_symbol or "")

    res = {
        "file": str(target_data.get("file") or ""),
        "symbol": symbol,
        "absolute_start_line": function_start + start,
        "absolute_end_line": function_start + end - 1,
        "window_code": "".join(lines[start:end]),
        "target_view": target_view,
        "target_sql_kind": target_sql_kind,
        "target_type": target_type,
        "parent_symbol": parent_symbol,
        "local_target": local_target,
        "operation": operation,
    }
    if canonical_old_string:
        res["canonical_old_string"] = canonical_old_string
        if operation == "dynamic_count_query_rewrite":
            res["target_id"] = f"{symbol}.query_construction_block"
        else:
            res["target_id"] = f"{symbol}.{target_sql_variable}"
    return res


def _parse_patch_window_payload(raw: str) -> dict[str, object] | None:
    payload = _extract_json_payload(raw)
    if not isinstance(payload, dict):
        return None
    confidence = payload.get("confidence")
    if not isinstance(confidence, int | float):
        return None

    # Handle replacement schema
    if "replacement" in payload and isinstance(payload["replacement"], dict):
        rep = payload["replacement"]
        new_string = rep.get("new_string")
        if isinstance(new_string, str):
            return {
                "replacement": {
                    "target_id": str(rep.get("target_id") or ""),
                    "new_string": new_string,
                },
                "confidence": float(confidence),
                "notes": str(payload.get("notes") or ""),
            }

    # Fallback to patches schema
    patches = payload.get("patches")
    if not isinstance(patches, list) or not patches:
        return None
    normalized: list[dict[str, str]] = []
    for patch in patches:
        if not isinstance(patch, dict):
            return None
        old_string = patch.get("old_string")
        new_string = patch.get("new_string")
        if not isinstance(old_string, str) or not isinstance(new_string, str):
            return None
        normalized.append({
            "old_string": old_string,
            "new_string": new_string,
        })
    return {
        "patches": normalized,
        "confidence": float(confidence),
        "notes": str(payload.get("notes") or ""),
    }


def _validate_patch_window_payload(
    payload: dict[str, object],
    *,
    current_code: str,
    patch_window: dict[str, object],
    target_view: str,
    target_sql_kind: str,
    target_sql_variable: str,
    constraints: list[str],
    path: str,
) -> list[str]:
    errors: list[str] = []
    window_code = str(patch_window.get("window_code") or "")
    patches = payload.get("patches")
    if not isinstance(patches, list):
        return ["patches"]
    updated_code = current_code
    allow_joins = any(
        "join" in item.lower() and any(word in item.lower() for word in ("allow", "keep", "preserve", "允许", "保留"))
        for item in constraints
    )
    for idx, patch in enumerate(patches):
        if not isinstance(patch, dict):
            errors.append(f"patches[{idx}]")
            continue
        old_string = str(patch.get("old_string") or "")
        new_string = str(patch.get("new_string") or "")
        if not old_string:
            errors.append(f"patches[{idx}].old_string")
            continue
        if old_string == new_string:
            errors.append(f"patches[{idx}].no_change")
        if current_code.count(old_string) != 1:
            errors.append(f"patches[{idx}].old_string_not_unique_in_symbol")
        if old_string not in window_code:
            errors.append(f"patches[{idx}].outside_patch_window")
        if target_sql_kind in ("count", "dynamic_count_query"):
            if target_view and target_view not in new_string:
                errors.append(f"patches[{idx}].missing_target_view")
            if not _patch_targets_count(old_string, target_sql_variable, target_sql_kind, window_code, current_code):
                errors.append(f"patches[{idx}].not_count_query")
            if target_sql_kind == "dynamic_count_query":
                if "build_order_detail_sql" in old_string:
                    errors.append(f"patches[{idx}].cannot_modify_helper_call")
            if not allow_joins:
                old_joins = _explicit_join_sources(old_string)
                retained = old_joins & _explicit_join_sources(new_string)
                if retained:
                    errors.append(f"patches[{idx}].retains_old_join:{sorted(retained)[0]}")
        if old_string in updated_code:
            updated_code = updated_code.replace(old_string, new_string, 1)
    syntax_error = _validate_python_code(path, updated_code)
    if syntax_error:
        errors.append(syntax_error)
    return errors


def _patch_targets_count(
    old_string: str,
    target_sql_variable: str,
    target_sql_kind: str,
    window_code: str,
    current_code: str,
) -> bool:
    if "count" in old_string.lower():
        return True
    if target_sql_variable and target_sql_variable.lower() in old_string.lower():
        return True
    
    # Check lines before old_string in current_code
    if old_string:
        idx = current_code.find(old_string)
        if idx >= 0:
            before_code = current_code[:idx]
            lines = before_code.splitlines()
            # Also check the current line containing the patch
            lines.append(current_code[idx:].splitlines()[0] if current_code[idx:].splitlines() else "")
            # Check last 25 lines
            for line in lines[-25:]:
                line_lower = line.lower()
                if "count" in line_lower:
                    return True
                if target_sql_variable and target_sql_variable.lower() in line_lower:
                    return True
    return False


def _explicit_join_sources(code: str) -> set[str]:
    return {
        match.group(1).lower()
        for match in re.finditer(
            r"(?i)\bJOIN\s+[`\"]?([A-Za-z_][A-Za-z0-9_.]*)",
            code,
        )
    }


def _target_sql_variable(
    edit_context: dict[str, object] | None,
    edit_plan: dict[str, object] | None,
) -> str:
    for source in (edit_plan, edit_context):
        if isinstance(source, dict):
            value = str(source.get("target_sql_variable") or "").strip()
            if value:
                return value
    if isinstance(edit_context, dict):
        task_intent = edit_context.get("task_intent")
        if isinstance(task_intent, dict):
            return str(task_intent.get("target_sql_variable") or "").strip()
    return ""


def _patch_constraints(edit_context: dict[str, object] | None) -> list[str]:
    if not isinstance(edit_context, dict):
        return []
    values = edit_context.get("constraints") or []
    return [str(item) for item in values if str(item).strip()]


def _minimal_exact_patch(old_code: str, new_code: str) -> tuple[str, str] | None:
    if old_code == new_code:
        return None
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    changed = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    if not changed:
        return None
    old_start = min(op[1] for op in changed)
    old_end = max(op[2] for op in changed)
    new_start = min(op[3] for op in changed)
    new_end = max(op[4] for op in changed)
    while old_start >= 0 and old_end <= len(old_lines):
        old_string = "".join(old_lines[old_start:old_end])
        new_string = "".join(new_lines[new_start:new_end])
        if old_string and old_code.count(old_string) == 1:
            return old_string, new_string
        if old_start == 0 and old_end == len(old_lines):
            break
        if old_start > 0:
            old_start -= 1
            new_start = max(0, new_start - 1)
        if old_end < len(old_lines):
            old_end += 1
            new_end = min(len(new_lines), new_end + 1)
    return None


def _find_target_for_edit(
    path: str,
    old_string: str,
    targets: list[dict[str, object]],
) -> dict[str, object] | None:
    normalized = path.replace("\\", "/").lstrip("./")
    matches = [
        target
        for target in targets
        if str(target.get("file") or "").replace("\\", "/").lstrip("./") == normalized
        and old_string in str(target.get("current_code") or "")
    ]
    return matches[0] if len(matches) == 1 else None


def _validate_python_patch(
    *,
    path: str,
    current_code: str,
    old_string: str,
    new_string: str,
) -> str:
    if old_string not in current_code:
        return "old_string is outside target symbol"
    return _validate_python_code(
        path,
        current_code.replace(old_string, new_string, 1),
    )


def _validate_python_code(path: str, code: str) -> str:
    if not path.lower().endswith(".py"):
        return ""
    try:
        ast.parse(textwrap.dedent(code))
    except SyntaxError as exc:
        return f"python_syntax:{exc.msg}:line_{exc.lineno}"
    return ""


async def _emit_progress(
    events: list[dict[str, object]] | None,
    callback: object,
    event: str,
    **data: object,
) -> None:
    item = {"event": event, **data}
    if events is not None:
        events.append(item)
    if not callable(callback):
        return
    result = callback(item)
    if inspect.isawaitable(result):
        await result


def _prepare_edit_plan_evidence(evidence: str) -> str:
    ctx = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
    if not isinstance(ctx, dict):
        return evidence[:12_000]
    compact = dict(ctx)
    compact_targets: list[dict[str, object]] = []
    for idx, target in enumerate(_editable_targets(ctx)):
        current_code = str(target.get("current_code") or "")
        display_code = str(target.get("display_code") or current_code)
        compact_targets.append({
            "index": idx,
            "file": target.get("file"),
            "symbol": target.get("symbol") or target.get("name"),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line"),
            "code_preview": _preview(display_code, limit=1200),
        })
    compact["editable_targets"] = compact_targets
    compact["snippets"] = compact_targets
    before = evidence.split("EDIT_CONTEXT_JSON", 1)[0][:6_000]
    return (
        before.rstrip()
        + "\n\nEDIT_CONTEXT_JSON_COMPACT\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def _validate_edit_context_ready(
    project_root: Path,
    ctx: dict[str, object] | None,
) -> str:
    if ctx is None:
        return "missing EDIT_CONTEXT_JSON with editable_targets/current_code"
    if ctx.get("code_edit_ready") is not True:
        return "EDIT_CONTEXT_JSON code_edit_ready is not true"
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    if not isinstance(targets, list) or not targets:
        return "no editable_targets/snippets"
    task_intent = ctx.get("task_intent")
    if not isinstance(task_intent, dict):
        return "missing task_intent"
    target_symbol = str(task_intent.get("target_symbol") or "").strip()
    if not target_symbol or target_symbol == "目标代码":
        return "missing target_symbol"
    if target_symbol in _EDIT_OPERATIONS or target_symbol.lower() == "modify":
        return f"operation value '{target_symbol}' cannot be used as target_symbol"
    intended_change = str(ctx.get("intended_change") or "").strip()
    if not intended_change and isinstance(task_intent, dict):
        intended_change = str(task_intent.get("goal") or "").strip()
    if not intended_change:
        return "missing intended_change"

    strategy = _edit_strategy(ctx)
    is_replacement_op = False
    if isinstance(task_intent, dict):
        op = task_intent.get("operation")
        if op in (
            "replace_dependency",
            "use_existing",
            "replace_sql_source",
            "count_query_view_rewrite",
            "dynamic_count_query_rewrite",
            "dynamic_sql_rewrite",
        ):
            is_replacement_op = True
    if strategy == "sql_view_rewrite":
        is_replacement_op = True
    if is_replacement_op and strategy != "sql_view_rewrite":
        error_json = json.dumps({
            "expected_strategy": "sql_view_rewrite",
            "actual_strategy": strategy,
            "reason": f"Operation '{op if isinstance(task_intent, dict) else 'replacement'}' requires sql_view_rewrite strategy but got '{strategy}'."
        })
        return f"diagnose_strategy_mismatch {error_json}"

    if is_replacement_op:
        replacement_source_name = ""
        resolved_deps = ctx.get("resolved_dependencies")
        if isinstance(resolved_deps, list):
            for dep in resolved_deps:
                if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                    replacement_source_name = str(dep.get("name") or "").strip()
                    break
        if not replacement_source_name:
            replacement_source_name = str(ctx.get("target_view") or "").strip()
        if not replacement_source_name:
            return "missing replacement_source/target_view in EDIT_CONTEXT_JSON for replacement operation"

    acceptance = ctx.get("acceptance_criteria") or ctx.get("acceptance")
    if not _has_acceptance(acceptance):
        return "missing acceptance_criteria"
    tool_policy = ctx.get("tool_policy")
    if not isinstance(tool_policy, dict):
        return "missing tool_policy"
    allowed = tool_policy.get("allowed_tools")
    if not isinstance(allowed, list) or "edit_file" not in allowed:
        return "tool_policy does not allow edit_file"
    scope = {
        _scope_key(str(item), project_root)
        for item in (tool_policy.get("scope") or [])
        if str(item).strip()
    }
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            return f"editable_targets[{idx}] is not an object"
        symbol_error = _validate_symbol_lock(target)
        if symbol_error:
            return f"editable_targets[{idx}] {symbol_error}"
        file_raw = str(target.get("file") or "")
        file = file_raw.replace("\\", "/")
        start = target.get("start_line")
        end = target.get("end_line")
        current_code = str(target.get("current_code") or "")
        target_change = str(target.get("intended_change") or intended_change).strip()
        target_acceptance = target.get("acceptance_criteria") or target.get("acceptance") or acceptance
        if not file or not isinstance(start, int) or not isinstance(end, int):
            return f"editable_targets[{idx}] missing file + line range"
        if end < start:
            return f"editable_targets[{idx}] invalid line range"
        if not current_code.strip():
            return f"editable_targets[{idx}] missing current_code"

        is_db_view_change = _edit_strategy(ctx) == "sql_view_rewrite"
        resolved_deps = ctx.get("resolved_dependencies")
        if isinstance(resolved_deps, list):
            for dep in resolved_deps:
                if isinstance(dep, dict) and dep.get("role") == "replacement_source" and dep.get("kind") == "database_view":
                    is_db_view_change = True
                    break

        if is_db_view_change:
            op = ""
            if isinstance(task_intent, dict):
                op = str(task_intent.get("operation") or "")
            is_dynamic_op = op in ("dynamic_sql_rewrite", "dynamic_count_query_rewrite")
            has_static = _target_has_sql_query(current_code)
            has_dynamic = _has_dynamic_sql_clues(current_code)
            if not has_static and not has_dynamic:
                return "no_rewriteable_sql_found: not a SQL/query target for a view-query edit"

            if not is_dynamic_op and not has_static:
                if _is_count_target(ctx, target):
                    return "target_not_hydrated"
                return (
                    f"editable_targets[{idx}] is not a SQL/query target for "
                    "a view-query edit"
                )
        if not target_change:
            return f"editable_targets[{idx}] missing intended_change"
        if not _has_acceptance(target_acceptance):
            return f"editable_targets[{idx}] missing acceptance_criteria"
        if scope and _scope_key(file, project_root) not in scope:
            return f"tool_policy scope does not cover {file}"
        path = _resolve_under_root(project_root, file)
        if path is None or not path.is_file():
            return f"patch_validator cannot locate {file}"
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"patch_validator cannot read {file}: {exc}"
        if current_code not in content:
            return f"patch_validator cannot locate current_code in {file}"
    if not any(
        isinstance(target, dict)
        and str(target.get("symbol") or target.get("name") or "").strip() == target_symbol
        for target in targets
    ):
        return f"target_symbol '{target_symbol}' does not match any editable target"
    return ""


def _has_acceptance(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(
            (isinstance(item, dict) and bool(item)) or str(item).strip()
            for item in value
        )
    if isinstance(value, dict):
        return bool(value)
    return False


def _target_has_sql_query(current_code: str) -> bool:
    lowered = current_code.lower()
    if re.search(r"(?is)\bSELECT\b.*\bFROM\b", current_code):
        return True
    if re.search(r"(?is)\b(?:FROM|JOIN)\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*", current_code):
        return True
    if any(kw in lowered for kw in ("select", "from", "join", "view")):
        return True
    if any(term in lowered for term in ("sql", "query", "database", "table")):
        return True
    return False


def _has_dynamic_sql_clues(code: str) -> bool:
    lowered = code.lower()
    clues = ("sql", "query", "select", "count", "join", "from", "where")
    return any(w in lowered for w in clues)


def _scope_key(raw: str, project_root: Path) -> str:
    text = raw.replace("\\", "/").strip()
    try:
        p = Path(text)
        resolved = p.resolve() if p.is_absolute() else (project_root / text.lstrip("./")).resolve()
        try:
            return str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(resolved).replace("\\", "/")
    except OSError:
        return text.lstrip("./")


def _extract_marker_json(text: str, marker: str) -> dict[str, object] | None:
    if marker not in text:
        return None
    payload = text.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for idx, ch in enumerate(payload[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None
    try:
        data = json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_json_payload(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    
    # Try direct parse
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
        
    # Try fixing unescaped newlines and parse again
    try:
        fixed_text = re.sub(r'(?<!\\)\n', '\\\\n', text)
        payload = json.loads(fixed_text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    # Fallback for EditPlanBuilder JSON corrupted by unescaped quotes/newlines
    try:
        ti_match = re.search(r'"target_index"\s*:\s*(\d+)', text)
        if ti_match:
            edit_item = {
                "target_index": int(ti_match.group(1)),
                "operation": "replace_sql_source",
            }
            tv_match = re.search(r'"target_view"\s*:\s*["\']([a-zA-Z0-9_.]+)["\']', text)
            if tv_match:
                edit_item["target_view"] = tv_match.group(1)
            return {
                "edits": [edit_item],
                "confidence": 1.0,
                "missing_info": []
            }
    except Exception:
        pass

    return None



def _extract_target_view_from_contract(contract: dict[str, object] | None) -> str:
    if not contract:
        return ""
    must_modify = contract.get("must_modify")
    if isinstance(must_modify, list):
        for item in must_modify:
            if isinstance(item, dict):
                should_change = str(item.get("should_change_to") or "")
                match = re.search(r"use view\s+([a-zA-Z0-9_.]+)", should_change, re.IGNORECASE)
                if match:
                    return match.group(1)
    return ""


def _preview(raw: str, *, limit: int = 500) -> str:
    text = " ".join((raw or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _extract_sql_aliases_from_python(code: str) -> list[str]:
    # Find SQL strings in code
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')'
    )
    aliases = []
    for match in string_pattern.finditer(code):
        inner = match.group("quote")
        if inner.startswith(('"""', "'''")):
            inner = inner[3:-3]
        else:
            inner = inner[1:-1]
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            # Extract columns
            sel_match = re.search(r"(?is)\bSELECT\b([\s\S]+?)\bFROM\b", inner)
            if sel_match:
                columns_part = sel_match.group(1)
                # Split by comma (ignoring commas inside parentheses)
                parts = []
                depth = 0
                start = 0
                quote = ""
                for idx, ch in enumerate(columns_part):
                    if quote:
                        if ch == quote:
                            quote = ""
                        continue
                    if ch in {"'", '"', "`"}:
                        quote = ch
                        continue
                    if ch == "(":
                        depth += 1
                    elif ch == ")" and depth > 0:
                        depth -= 1
                    elif ch == "," and depth == 0:
                        parts.append(columns_part[start:idx])
                        start = idx + 1
                parts.append(columns_part[start:])
                
                for col in parts:
                    col = col.strip()
                    if not col:
                        continue
                    as_match = re.search(r"\bAS\s+([a-zA-Z0-9_\"'`\[\]]+)", col, re.I)
                    if as_match:
                        aliases.append(as_match.group(1).replace('`','').replace('"','').replace("'", "").strip())
                        continue
                    # Find last identifier
                    ids = re.findall(r"\b[a-zA-Z0-9_.]+\b", col)
                    if ids:
                        aliases.append(ids[-1].split('.')[-1].strip())
    return [a for a in dict.fromkeys(aliases) if a and not a.startswith("*")]


def _contains_legacy_sql_references(code: str, replaces: list[str], old_aliases: list[str]) -> list[str]:
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')'
    )
    found = []
    replaces_lower = {r.lower() for r in replaces if r}
    aliases_lower = {a.lower() for a in old_aliases if a}
    
    for match in string_pattern.finditer(code):
        inner = match.group("quote")
        if inner.startswith(('"""', "'''")):
            inner = inner[3:-3]
        else:
            inner = inner[1:-1]
            
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            # Clean SQL comments
            inner_clean = re.sub(r"/\*[\s\S]*?\*/", "", inner)
            inner_clean = re.sub(r"--.*", "", inner_clean)
            
            # Check for table references
            for r in replaces_lower:
                if re.search(rf"\b{re.escape(r)}\b", inner_clean, re.IGNORECASE):
                    found.append(r)
            # Check for alias.column pattern
            for a in aliases_lower:
                if re.search(rf"\b{re.escape(a)}\.", inner_clean, re.IGNORECASE):
                    found.append(f"{a}.")
    return list(dict.fromkeys(found))


def _verify_signature_unchanged(old_code: str, new_code: str, target_symbol: str) -> bool:
    if not target_symbol or target_symbol == "目标代码":
        return True
    # Find line matching def symbol_name(...)
    pattern = re.compile(rf"def\s+{re.escape(target_symbol)}\s*\(.*?\)\s*:", re.DOTALL)
    old_match = pattern.search(old_code)
    if not old_match:
        # Target symbol might not be a function definition in old_code
        return True
    new_match = pattern.search(new_code)
    if not new_match:
        # The def signature was deleted or altered
        return False
    # Clean whitespace and compare signatures
    old_sig = " ".join(old_match.group(0).split())
    new_sig = " ".join(new_match.group(0).split())
    return old_sig == new_sig


def _verify_body_changed_and_contains_view(old_code: str, new_code: str, target_symbol: str, target_view: str) -> tuple[bool, bool]:
    if not target_symbol or target_symbol == "目标代码":
        body_changed = old_code.strip() != new_code.strip()
        contains_view = target_view.lower() in new_code.lower()
        return body_changed, contains_view
        
    pattern = re.compile(rf"def\s+{re.escape(target_symbol)}\s*\(.*?\)\s*:", re.DOTALL)
    old_match = pattern.search(old_code)
    new_match = pattern.search(new_code)
    if old_match and new_match:
        old_body = old_code[old_match.end():].strip()
        new_body = new_code[new_match.end():].strip()
        body_changed = old_body != new_body
        contains_view = target_view.lower() in new_body.lower()
        return body_changed, contains_view
    
    return old_code.strip() != new_code.strip(), target_view.lower() in new_code.lower()


def _sql_bounded_replacement_messages(
    *,
    instruction: str,
    file: str,
    target_index: int,
    current_code: str,
    target_view: str,
    available_columns: list[str],
    old_aliases: list[str],
    constraints: list[str],
) -> list[dict[str, object]]:
    constraints_text = "\n".join(f"- {c}" for c in constraints)
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII SQL ReplacementBuilder. Output ONE raw JSON object only. \n"
                "No markdown, no prose. Required schema:\n"
                '{"new_string":"complete replacement for current_code"}\n\n'
                "Return only the replacement for the single provided current_code.\n"
                "Preserve indentation, surrounding behavior, and python structure.\n"
                "You must perform a bounded SQL replacement, modifying ONLY the SQL statement/string inside the code.\n"
                "CRITICAL CONSTRAINTS:\n"
                f"- You must only use the target view '{target_view}'. Do NOT invent other views or tables.\n"
                f"- Available columns in target view '{target_view}': {available_columns}\n"
                f"- Old SELECT output aliases to map: {old_aliases}\n"
                "- You must preserve all original output field/column names (aliases).\n"
                "- You must NOT delete or modify the WHERE clause (e.g. keep any placeholders or variables like WHERE passenger_id = :passenger_id).\n"
                "- You must NOT use SELECT *; explicitly name all columns.\n"
                "- You must not change the function signature or delete any python logic.\n"
                f"{constraints_text}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                f"File: {file}\nTarget index: {target_index}\n\n"
                "CURRENT_CODE:\n"
                f"{current_code}\n"
            )
        }
    ]


def _extract_table_aliases_from_sql(code: str) -> list[str]:
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')'
    )
    aliases = []
    
    def walk(tokens, in_from_clause=False):
        local_aliases = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            val = token.value.upper().strip()
            
            if token.ttype is sqlparse.tokens.Keyword or token.ttype is sqlparse.tokens.Keyword.DML:
                if val in ("FROM", "JOIN") or "JOIN" in val:
                    in_from_clause = True
                elif val in ("ON", "USING", "WHERE", "GROUP", "ORDER", "LIMIT", "UNION", "HAVING", ";", "SELECT", "INSERT", "UPDATE", "DELETE"):
                    in_from_clause = False
                elif any(val.startswith(kw) for kw in ("GROUP", "ORDER", "LIMIT", "UNION", "HAVING")):
                    in_from_clause = False
            elif isinstance(token, sqlparse.sql.Where) or val.startswith("WHERE"):
                in_from_clause = False
                
            if in_from_clause:
                if isinstance(token, sqlparse.sql.IdentifierList):
                    for subtok in token.get_identifiers():
                        if isinstance(subtok, sqlparse.sql.Identifier):
                            alias = subtok.get_alias()
                            if alias:
                                local_aliases.append(alias.strip().replace('"', '').replace("'", "").replace("`", ""))
                elif isinstance(token, sqlparse.sql.Identifier):
                    alias = token.get_alias()
                    if alias:
                        local_aliases.append(alias.strip().replace('"', '').replace("'", "").replace("`", ""))
            
            if hasattr(token, "tokens") and token.tokens:
                is_paren = isinstance(token, sqlparse.sql.Parenthesis)
                local_aliases.extend(walk(token.tokens, in_from_clause=False if is_paren else in_from_clause))
                
            i += 1
        return local_aliases

    for match in string_pattern.finditer(code):
        inner = match.group("quote")
        if inner.startswith(('"""', "'''")):
            inner = inner[3:-3]
        else:
            inner = inner[1:-1]
        
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            try:
                parsed = sqlparse.parse(inner)
                for stmt in parsed:
                    aliases.extend(walk(stmt.tokens))
            except Exception:
                pass
                
    return [a for a in dict.fromkeys(aliases) if a]
