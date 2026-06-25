from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import sqlparse


from src.skills.base import SkillContext, SkillResult

EditComplete = Callable[[list[dict[str, object]]], Awaitable[str]]


class CodeEditSkill:
    name = "code_edit"

    def __init__(
        self,
        *,
        project_root: Path,
        llm_complete: EditComplete | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.llm_complete = llm_complete

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
        if not evidence:
            # Fallback to in-memory search cache passed in kwargs
            search_cache = kwargs.get("_search_cache") or {}
            evidence = search_cache.get("search_output") or ""
        if not evidence:
            return SkillResult(
                success=False,
                summary="code_edit requires search_output evidence.",
                missing_info=("search_output",),
            )
        contract = kwargs.get("handoff_contract")
        edit_context = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
        if isinstance(edit_context, dict) and not edit_context.get("target_view"):
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

        ready_error = _validate_edit_context_ready(self.project_root, edit_context)
        if ready_error:
            return SkillResult(
                success=False,
                summary=f"code_edit not ready: {ready_error}",
                missing_info=(ready_error,),
                metadata={"raw_preview": _preview(evidence)},
            )
        plan_messages = _edit_plan_messages(
            instruction=instruction,
            evidence=evidence,
            handoff_contract=contract if isinstance(contract, dict) else None,
        )
        plan_raw = await self.llm_complete(plan_messages)
        plan_payload = _extract_json_payload(plan_raw)
        edit_targets = _editable_targets(edit_context)
        if plan_payload is None:
            payload = _fallback_edit_payload(
                edit_context,
                evidence=evidence,
                instruction=instruction,
                handoff_contract=contract if isinstance(contract, dict) else None,
            )
            raw_preview = plan_raw
        else:
            payload, raw_preview = await self._build_payload_from_plan(
                plan_payload,
                edit_targets=edit_targets,
                instruction=instruction,
                evidence=evidence,
            )
        if payload is None:
            return SkillResult(
                success=False,
                summary=(
                    "code_edit could not produce a valid structured edit plan, "
                    "and no deterministic SQL view edit could be derived."
                ),
                missing_info=("edit_plan_json", "deterministic_edit"),
                metadata={"raw_preview": _preview(raw_preview)},
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
                metadata={"raw_preview": _preview(raw_preview)},
            )
        if not isinstance(edits, list) or not edits:
            return SkillResult(
                success=False,
                summary="code_edit model returned no edits.",
                missing_info=("edits",),
                metadata={"raw_preview": _preview(raw_preview)},
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
                    f"edits[{idx}] requires target_index+new_string or "
                    "path+old_string+new_string"
                )
                continue
            if old_string == new_string:
                errors.append(f"edits[{idx}] old_string and new_string must differ")
                continue
            if len(old_string) > 50_000 or len(new_string) > 50_000:
                errors.append(f"edits[{idx}] is too large; use a smaller exact snippet")
                continue
            result = _apply_exact_edit(
                self.project_root,
                path=path,
                old_string=old_string,
                new_string=new_string,
                original_files=original_files,
            )
            if result.startswith("ok:"):
                changed.append(result[3:])
            else:
                errors.append(result)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(dict.fromkeys(changed)),
                missing_info=tuple(errors),
                metadata={"raw_preview": _preview(raw_preview)},
            )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} edit_file action(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={
                "raw_preview": _preview(raw_preview),
                "confidence": str(confidence),
                "original_files_json": json.dumps(original_files),
            },
        )

    async def _build_payload_from_plan(
        self,
        plan_payload: dict[str, object],
        *,
        edit_targets: list[dict[str, object]],
        instruction: str,
        evidence: str,
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
            
        built_edits: list[dict[str, object]] = []
        raw_parts = [json.dumps(plan_payload, ensure_ascii=False)]
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return None, raw_parts[0]
            target_index = edit.get("target_index")
            if not isinstance(target_index, int) or not (0 <= target_index < len(edit_targets)):
                return None, raw_parts[0]
            current_code = str(edit_targets[target_index].get("current_code") or "")
            file = str(edit_targets[target_index].get("file") or "")
            
            operation = edit.get("operation")
            target_view = str(edit.get("target_view") or fallback_view).strip()
            new_string = None
            if operation in ("replace_sql_source", "replace_dependency", "use_existing") and target_view:
                tmp_ctx = dict(edit_context or {})
                tmp_ctx["target_view"] = target_view
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
) -> str:
    parsed = sqlparse.parse(current_sql)[0]
    tokens = list(parsed.tokens)
    
    from_idx = -1
    for idx, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
            from_idx = idx
            break
            
    if from_idx == -1:
        return current_sql
        
    main_table_idx = -1
    for j in range(from_idx + 1, len(tokens)):
        t = tokens[j]
        if t.is_whitespace:
            continue
        main_table_idx = j
        break
        
    if main_table_idx == -1:
        return current_sql
        
    main_table_token = tokens[main_table_idx]
    alias = main_table_token.get_alias() if hasattr(main_table_token, 'get_alias') else None
    old_main_alias = alias
    old_main_table_name = ""
    if isinstance(main_table_token, sqlparse.sql.Identifier):
        old_main_table_name = main_table_token.get_real_name() or main_table_token.value
    elif main_table_token.ttype is None and main_table_token.value.strip():
        old_main_table_name = main_table_token.value.strip()
    
    replaces_lower = {obj.lower() for obj in replaces_objects}
    old_main_clean = old_main_table_name.strip().split('.')[-1].split()[0].replace('"', '').replace("'", "").replace("`", "").lower()
    strict_projection = bool(target_columns)
    if strict_projection and (old_main_clean in replaces_lower or replaces_lower):
        alias = _fresh_view_alias({old_main_alias or "", *replaces_lower})

    new_val = f"{target_source} {alias}" if alias else target_source
    new_token = sqlparse.sql.Token(sqlparse.tokens.Name, new_val)
    
    end_idx = len(tokens)
    for j in range(main_table_idx + 1, len(tokens)):
        t = tokens[j]
        val = t.value.upper().strip()
        if isinstance(t, sqlparse.sql.Where) or val.startswith("WHERE"):
            end_idx = j
            break
        if t.ttype is sqlparse.tokens.Keyword or t.ttype is sqlparse.tokens.Keyword.DML:
            if val in ("GROUP", "ORDER", "LIMIT", "UNION", "HAVING", ";", "SELECT", "INSERT", "UPDATE", "DELETE"):
                end_idx = j
                break
            if any(val.startswith(kw) for kw in ("GROUP", "ORDER", "LIMIT", "UNION", "HAVING")):
                end_idx = j
                break
                
    j = main_table_idx + 1
    middle_tokens = []
    removed_aliases = set()
    
    while j < end_idx:
        t = tokens[j]
        val = t.value.upper().strip()
        if "JOIN" in val and (t.ttype is sqlparse.tokens.Keyword or t.ttype is None):
            join_end = end_idx
            for k in range(j + 1, end_idx):
                tk = tokens[k]
                tk_val = tk.value.upper().strip()
                if "JOIN" in tk_val and (tk.ttype is sqlparse.tokens.Keyword or tk.ttype is None):
                    join_end = k
                    break
            
            joined_table_name = ""
            joined_table_alias = ""
            for k in range(j + 1, join_end):
                tk = tokens[k]
                if isinstance(tk, sqlparse.sql.Identifier):
                    real_name = tk.get_real_name()
                    joined_table_name = real_name or tk.value
                    joined_table_alias = tk.get_alias()
                    break
                elif tk.ttype is None and tk.value.strip() and not tk.is_whitespace:
                    joined_table_name = tk.value
                    break
            
            cleaned_joined_table = joined_table_name.strip().split('.')[-1].split()[0].replace('"', '').replace("'", "").replace("`", "").lower()
            
            if cleaned_joined_table in replaces_lower:
                if joined_table_alias:
                    removed_aliases.add(joined_table_alias)
                j = join_end
            else:
                middle_tokens.extend(tokens[j:join_end])
                j = join_end
        else:
            middle_tokens.append(t)
            j += 1
            
    target_alias = alias or target_source
    
    legacy_lower = {a.lower() for a in removed_aliases if a}
    for obj in replaces_objects:
        legacy_lower.add(obj.lower())
    if old_main_table_name:
        legacy_lower.add(old_main_table_name.lower())
    if old_main_alias:
        legacy_lower.add(old_main_alias.lower())
        
    def references_aliases(token, replaced_aliases):
        if isinstance(token, sqlparse.sql.Identifier):
            sub = token.tokens
            if len(sub) >= 2 and sub[1].value == '.':
                prefix = sub[0].value.strip().replace('`','').replace('"','').replace("'", "").lower()
                if prefix in replaced_aliases:
                    return True
        if hasattr(token, 'tokens') and token.tokens:
            return any(references_aliases(sub, replaced_aliases) for sub in token.tokens)
        return False

    def rewrite_identifiers(token, legacy_aliases, target_alias):
        if not hasattr(token, 'tokens') or not token.tokens:
            return
        
        if isinstance(token, sqlparse.sql.Identifier):
            sub = token.tokens
            if len(sub) >= 3:
                first_t = sub[0]
                second_t = sub[1]
                if (first_t.ttype in (sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder) or first_t.ttype is None) and second_t.value == '.':
                    prefix = first_t.value.strip().replace('`','').replace('"','').replace("'", "").lower()
                    if prefix in legacy_aliases:
                        if target_alias:
                            first_t.value = target_alias
                        else:
                            token.tokens = sub[2:]
                            
        for sub_token in token.tokens:
            rewrite_identifiers(sub_token, legacy_aliases, target_alias)

    if target_columns:
        select_idx = -1
        for idx, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword.DML and token.value.upper() == "SELECT":
                select_idx = idx
                break
                
        if select_idx != -1:
            def get_select_items(tokens_list, s_idx, f_idx):
                res_items = []
                for idx in range(s_idx + 1, f_idx):
                    token = tokens_list[idx]
                    if token.is_whitespace:
                        continue
                    if isinstance(token, sqlparse.sql.IdentifierList):
                        for ident in token.get_identifiers():
                            res_items.append(ident)
                    elif isinstance(token, sqlparse.sql.Identifier):
                        res_items.append(token)
                    elif token.ttype is None and token.value.strip() and token.value.strip() != ',':
                        res_items.append(token)
                return res_items
                
            def get_output_name(item):
                if hasattr(item, 'get_alias') and item.get_alias():
                    return item.get_alias().replace('"', '').replace("'", "").replace("`", "").strip()
                if hasattr(item, 'get_real_name') and item.get_real_name():
                    return item.get_real_name().replace('"', '').replace("'", "").replace("`", "").strip()
                val = item.value.strip()
                parts = val.split('.')
                return parts[-1].replace('"', '').replace("'", "").replace("`", "").strip()
                
            select_items = get_select_items(tokens, select_idx, from_idx)
            target_col_lookup = {
                str(col).strip().lower(): str(col).strip()
                for col in target_columns
                if str(col).strip()
            }
            defaults = {
                str(key).strip().lower(): str(value).strip()
                for key, value in (column_defaults or {}).items()
                if str(key).strip() and str(value).strip()
            }
            items_to_rewrite = []
            
            for item in select_items:
                out_name = get_output_name(item)
                out_key = out_name.lower()
                if "*" in item.value:
                    raise ProjectionMappingError("Projection rewrite failed: SELECT * has no explicit column mapping")
                if out_key in target_col_lookup:
                    source_col = target_col_lookup[out_key]
                    new_val = f"{target_alias}.{source_col}" if target_alias else source_col
                    items_to_rewrite.append((item, new_val))
                elif out_key in defaults:
                    items_to_rewrite.append((item, f"{defaults[out_key]} AS {out_name}"))
                else:
                    raise ProjectionMappingError(
                        f"Projection rewrite failed due to missing column mapping: {out_name}"
                    )
                
            for item, new_val in items_to_rewrite:
                parsed_new = sqlparse.parse(new_val)[0]
                new_item_token = parsed_new.tokens[0]
                
                parent = item.parent
                if parent and hasattr(parent, 'tokens'):
                    idx_in_parent = parent.tokens.index(item)
                    parent.tokens[idx_in_parent] = new_item_token
                    new_item_token.parent = parent
                    
    rewrite_identifiers(parsed, legacy_lower, target_alias)
    
    new_tokens = tokens[:from_idx + 1]
    if from_idx + 1 < len(tokens) and tokens[from_idx + 1].is_whitespace:
        new_tokens.append(tokens[from_idx + 1])
    else:
        new_tokens.append(sqlparse.sql.Token(sqlparse.tokens.Whitespace, " "))
        
    new_tokens.append(new_token)
    new_tokens.extend(middle_tokens)
    
    if end_idx > 0 and end_idx < len(tokens):
        pre_end_token = tokens[end_idx - 1]
        if pre_end_token.is_whitespace:
            new_tokens.append(pre_end_token)
            
    if end_idx < len(tokens):
        new_tokens.extend(tokens[end_idx:])
        
    parsed.tokens = new_tokens
    return str(parsed)


def _fresh_view_alias(blocked: set[str]) -> str:
    blocked_lower = {item.lower() for item in blocked if item}
    for candidate in ("v", "rv", "src"):
        if candidate not in blocked_lower:
            return candidate
    return "replacement_view"


def replace_table_with_view_in_sql(current_sql: str, target_view: str) -> str:
    return rewrite_query_source(current_sql, target_view, [])


def generate_sql_patch(current_code: str, edit_context: dict) -> str | None:
    dep_name = ""
    replaces = []
    columns = []
    column_defaults = {}
    resolved_deps = edit_context.get("resolved_dependencies") or []
    for dep in resolved_deps:
        if isinstance(dep, dict) and dep.get("role") == "replacement_source":
            dep_name = str(dep.get("name") or "").strip()
            replaces = list(dep.get("replaces_objects") or [])
            columns = list(dep.get("columns") or [])
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

    string_pattern = re.compile(
        r'(?P<prefix>[frFR]*)(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
    )
    for match in string_pattern.finditer(current_code):
        quote_content = match.group("quote")
        if quote_content.startswith(('"""', "'''")):
            inner = quote_content[3:-3]
            quote_type = quote_content[:3]
        else:
            inner = quote_content[1:-1]
            quote_type = quote_content[0]
            
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            modified_inner = rewrite_query_source(
                inner,
                dep_name,
                replaces,
                columns,
                column_defaults,
            )
            new_quote_content = f"{quote_type}{modified_inner}{quote_type}"
            start, end = match.span("quote")
            return current_code[:start] + new_quote_content + current_code[end:]
            
    return None


def deterministic_replace_sql_with_view(current_code: str, view_name: str) -> str | None:
    return generate_sql_patch(current_code, {"target_view": view_name})




def _clean_llm_code(code: str) -> str:
    res = code
    res = res.replace('\\"""', '"""').replace('\\\'\'\'', '\'\'\'')
    res = res.replace('\"\"\"', '"""').replace('\'\'\'', '\'\'\'')
    return res



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
) -> str:
    target = _resolve_under_root(project_root, path)
    if target is None:
        return f"{path}: outside project root"
    if not target.is_file():
        return f"{path}: file not found"
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{path}: read failed: {exc}"
    
    import hashlib
    original_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    occurrences = content.count(old_string)
    if occurrences != 1:
        return f"{path}: old_string occurrence count is {occurrences}, expected 1"
    
    new_content = content.replace(old_string, new_string, 1)
    new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    if original_hash == new_hash:
        return f"{path}: file hash did not change (no modification made)"
        
    rel = _rel_display(project_root, target)
    if original_files is not None:
        original_files.setdefault(rel, content)
    try:
        target.write_text(new_content, encoding="utf-8")
        # Verify immediately from disk
        written_content = target.read_text(encoding="utf-8")
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
    return [target for target in targets or [] if isinstance(target, dict)]


def _resolve_edit_item_target(
    item: dict[str, object],
    targets: list[dict[str, object]],
) -> tuple[str, str]:
    path = str(item.get("path") or "")
    old_string = str(item.get("old_string") or "")
    if path and old_string:
        return path, old_string

    target_index = item.get("target_index")
    if isinstance(target_index, int) and 0 <= target_index < len(targets):
        target = targets[target_index]
        return (
            str(target.get("file") or ""),
            str(target.get("current_code") or ""),
        )

    if path and not old_string:
        matching = [
            target for target in targets
            if str(target.get("file") or "").replace("\\", "/") == path.replace("\\", "/")
        ]
        if len(matching) == 1:
            return path, str(matching[0].get("current_code") or "")
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


def _replacement_messages(
    *,
    instruction: str,
    file: str,
    target_index: int,
    current_code: str,
    edit_plan: dict[str, object],
    evidence: str,
    target_view: str = "",
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
                f"File: {file}\nTarget index: {target_index}\n\n"
                "EDIT_PLAN_JSON\n"
                f"{json.dumps(edit_plan, ensure_ascii=False, indent=2)}\n\n"
                "Relevant evidence summary:\n"
                f"{_prepare_replacement_evidence(evidence)}\n\n"
                "CURRENT_CODE\n"
                f"{current_code[:50_000]}"
            ),
        },
    ]


def _prepare_replacement_evidence(evidence: str) -> str:
    marker = "EDIT_CONTEXT_JSON"
    before = evidence.split(marker, 1)[0] if marker in evidence else evidence
    return before[:6_000].strip()


def _prepare_edit_evidence(evidence: str) -> str:
    marker = "EDIT_CONTEXT_JSON"
    if marker not in evidence:
        return evidence[:12_000]
    before, after = evidence.split(marker, 1)
    return (
        before[:6_000].rstrip()
        + "\n\n"
        + marker
        + "\n"
        + after.strip()[:50_000]
    )


def _prepare_edit_plan_evidence(evidence: str) -> str:
    ctx = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
    if not isinstance(ctx, dict):
        return evidence[:12_000]
    compact = dict(ctx)
    compact_targets: list[dict[str, object]] = []
    for idx, target in enumerate(_editable_targets(ctx)):
        current_code = str(target.get("current_code") or "")
        compact_targets.append({
            "index": idx,
            "file": target.get("file"),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line"),
            "code_preview": _preview(current_code, limit=1200),
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
    intended_change = str(ctx.get("intended_change") or "").strip()
    if not intended_change and isinstance(task_intent, dict):
        intended_change = str(task_intent.get("goal") or "").strip()
    if not intended_change:
        return "missing intended_change"

    is_replacement_op = False
    if isinstance(task_intent, dict):
        op = task_intent.get("operation")
        if op in ("replace_dependency", "use_existing", "replace_sql_source"):
            is_replacement_op = True
    if not is_replacement_op and _is_sql_view_change(intended_change):
        is_replacement_op = True

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

        available_views = ctx.get("available_views")
        if available_views:
            normalized_views = []
            for v in available_views:
                if isinstance(v, dict):
                    normalized_views.append(str(v.get("name") or "").lower())
                elif isinstance(v, str):
                    normalized_views.append(v.lower())
            if replacement_source_name.lower() not in normalized_views:
                return f"target_view '{replacement_source_name}' not found in available_views: {normalized_views}"

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

        is_db_view_change = False
        resolved_deps = ctx.get("resolved_dependencies")
        if isinstance(resolved_deps, list):
            for dep in resolved_deps:
                if isinstance(dep, dict) and dep.get("role") == "replacement_source" and dep.get("kind") == "database_view":
                    is_db_view_change = True
                    break
        if not is_db_view_change and (_is_sql_view_change(target_change) or _is_sql_view_change(intended_change)):
            is_db_view_change = True

        if is_db_view_change and not _target_has_sql_query(current_code):
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


def _is_sql_view_change(text: str) -> bool:
    lowered = text.lower()
    has_view = "视图" in text or "view" in lowered
    has_replace = any(word in lowered for word in ["replace", "use", "change", "替换", "用", "改", "使用"])
    return has_view and has_replace



def _target_has_sql_query(current_code: str) -> bool:
    return bool(
        re.search(
            r"(?is)\bSELECT\b.{0,200}\bFROM\b|\b(?:FROM|JOIN)\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*",
            current_code,
        )
    )


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


def _fallback_edit_payload(
    ctx: dict[str, object] | None,
    *,
    evidence: str,
    instruction: str,
    handoff_contract: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(ctx, dict):
        return {
            "edits": [],
            "confidence": 0.0,
            "missing_info": ["missing_context"],
            "source": "deterministic_sql_view_fallback",
        }
    view_name = str(ctx.get("target_view") or "").strip()
    if not view_name:
        view_name = _select_target_view(
            evidence=evidence,
            instruction=instruction,
            handoff_contract=handoff_contract,
        )
    if not view_name:
        return {
            "edits": [],
            "confidence": 0.0,
            "missing_info": ["missing_target_view"],
            "source": "deterministic_sql_view_fallback",
        }
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    if not isinstance(targets, list) or not targets:
        return {
            "edits": [],
            "confidence": 0.0,
            "missing_info": ["missing_targets"],
            "source": "deterministic_sql_view_fallback",
        }

    edits: list[dict[str, str]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        path = str(target.get("file") or "").strip()
        current_code = str(target.get("current_code") or "")
        if not path or not current_code:
            continue
        tmp_ctx = dict(ctx)
        tmp_ctx["target_view"] = view_name
        if "resolved_dependencies" in tmp_ctx:
            resolved_deps = []
            for dep in tmp_ctx["resolved_dependencies"]:
                if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                    new_dep = dict(dep)
                    new_dep["name"] = view_name
                    resolved_deps.append(new_dep)
                else:
                    resolved_deps.append(dep)
            tmp_ctx["resolved_dependencies"] = resolved_deps
        new_code = generate_sql_patch(current_code, tmp_ctx)
        if new_code is None:
            new_code = _replace_boarding_pass_table(current_code, view_name)
        if new_code == current_code:
            continue
        edits.append({
            "path": path,
            "old_string": current_code,
            "new_string": new_code,
        })
        break

    if not edits:
        return {
            "edits": [],
            "confidence": 0.0,
            "missing_info": ["no_matching_edits"],
            "source": "deterministic_sql_view_fallback",
        }
    return {
        "edits": edits,
        "confidence": 0.82,
        "missing_info": [],
        "source": "deterministic_sql_view_fallback",
    }


def _select_target_view(
    *,
    evidence: str,
    instruction: str,
    handoff_contract: dict[str, object] | None,
) -> str:
    contract_text = (
        json.dumps(handoff_contract, ensure_ascii=False)
        if handoff_contract
        else ""
    )
    text = "\n".join([instruction, evidence, contract_text])
    candidates = list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(
            r"(?i)\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([A-Za-z_][A-Za-z0-9_]*)",
            text,
        )
    ))
    candidates.extend(
        candidate
        for candidate in re.findall(r"\b(?:v|view)_[A-Za-z0-9_]*\b", text)
        if candidate not in candidates
    )
    if not candidates:
        return ""

    def score(name: str) -> tuple[int, str]:
        lowered = name.lower()
        value = 0
        if "boarding" in lowered:
            value += 8
        if "ticket" in lowered:
            value += 6
        if "report" in lowered or "detail" in lowered:
            value += 3
        if "flight" in lowered or "monitor" in lowered:
            value -= 5
        return value, name

    best_score, best = max(score(candidate) for candidate in candidates)
    return best if best_score > 0 else ""


def _replace_boarding_pass_table(current_code: str, view_name: str) -> str:
    pattern = re.compile(
        r"(?i)\b(?P<keyword>FROM|JOIN)\s+(?P<quote>[`\"]?)boarding_pass(?P=quote)\b"
    )

    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{match.group('keyword')} {quote}{view_name}{quote}"

    return pattern.sub(repl, current_code)


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

    # Fallback for ReplacementBuilder JSON corrupted by unescaped quotes/newlines in new_string
    try:
        pattern = re.compile(r'["\']new_string["\']\s*:\s*["\']', re.IGNORECASE)
        match = pattern.search(text)
        if match:
            start_idx = match.end()
            r_brace = text.rfind('}')
            r_quote = -1
            for i in range(r_brace - 1, start_idx - 1, -1):
                if text[i] in ('"', "'"):
                    r_quote = i
                    break
            if r_quote > start_idx:
                code_part = text[start_idx:r_quote]
                res = code_part.replace('\\n', '\n').replace('\\t', '\t')
                res = res.replace('\\"', '"').replace("\\'", "'")
                res = res.replace('\\\\', '\\')
                return {"new_string": res}
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
