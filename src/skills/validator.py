from __future__ import annotations

import json
import ast
import py_compile
import re
import subprocess
import textwrap
from pathlib import Path

import sqlparse


from src.skills.base import SkillContext, SkillResult


def get_git_head_content(project_root: Path, rel_path: str) -> str | None:
    try:
        clean_rel = rel_path.replace("\\", "/")
        if clean_rel.startswith("/"):
            try:
                clean_rel = Path(clean_rel).relative_to(project_root).as_posix()
            except ValueError:
                pass
        if not clean_rel.startswith("./"):
            clean_rel = f"./{clean_rel}"
        res = subprocess.run(
            ["git", "show", f"HEAD:{clean_rel}"],
            cwd=str(project_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout
    except Exception:
        return None


def _extract_removed_aliases(sql: str, forbidden_tables: set[str]) -> set[str]:
    removed_aliases = set()
    try:
        parsed = sqlparse.parse(sql)[0]
        tokens = list(parsed.tokens)
        
        from_idx = -1
        for idx, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                from_idx = idx
                break
        if from_idx == -1:
            return removed_aliases
            
        for idx in range(from_idx + 1, len(tokens)):
            t = tokens[idx]
            val = t.value.upper().strip()
            if "JOIN" in val and (t.ttype is sqlparse.tokens.Keyword or t.ttype is None):
                join_end = len(tokens)
                for k in range(idx + 1, len(tokens)):
                    tk = tokens[k]
                    tk_val = tk.value.upper().strip()
                    if "JOIN" in tk_val and (tk.ttype is sqlparse.tokens.Keyword or tk.ttype is None):
                        join_end = k
                        break
                        
                joined_table_name = ""
                joined_table_alias = ""
                for k in range(idx + 1, join_end):
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
                if cleaned_joined_table in forbidden_tables:
                    if joined_table_alias:
                        removed_aliases.add(joined_table_alias)
    except Exception:
        pass
    return removed_aliases


def validate_sql_references(project_root: Path, changed_files: list[str] | tuple[str, ...]) -> list[str]:
    repo_db_objects = set()
    pattern = re.compile(
        r"\bCREATE\s+[^;]*?\b(?:VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\.\"'\`\[\]]+)",
        re.IGNORECASE
    )
    
    for path in project_root.rglob("*"):
        try:
            if path.suffix.lower() in {".sql", ".py"} and path.is_file():
                content = path.read_text(encoding="utf-8", errors="ignore")
                for match in pattern.finditer(content):
                    raw_name = match.group(1)
                    cleaned = re.sub(r'[\"\'\`\[\]]', '', raw_name)
                    repo_db_objects.add(cleaned.lower())
                    if '.' in cleaned:
                        repo_db_objects.add(cleaned.split('.')[-1].lower())
        except Exception:
            pass


    errors: list[str] = []
    
    db_ref_patterns = [
        re.compile(r"\b(?:FROM|JOIN)\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?", re.IGNORECASE),
        re.compile(r"\bUPDATE\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?\s+SET\b", re.IGNORECASE),
        re.compile(r"\bINSERT\s+INTO\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?", re.IGNORECASE),
        re.compile(r"\bDELETE\s+FROM\s+[`\"]?([a-zA-Z0-9_.]+)[`\"]?", re.IGNORECASE),
    ]
    cte_pattern = re.compile(r"\b(?:WITH|,)\s+([a-zA-Z0-9_.]+)\s+AS\b", re.IGNORECASE)
    py_string_pattern = re.compile(
        r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'',
        re.IGNORECASE
    )

    def is_likely_sql(literal: str) -> bool:
        lowered = literal.lower()
        sql_keywords = {
            "select", "insert", "update", "delete", "create", "drop", "alter",
            "where", "join", "from", "into", "values", "set", "on", "group", "by",
            "order", "limit", "offset", "having", "union", "all", "index", "table",
            "view", "database", "schema", "key", "constraint"
        }
        words = re.findall(r"\b[a-z]+\b", lowered)
        kw_count = sum(1 for w in words if w in sql_keywords)
        if any(k in lowered for k in ["select", "insert", "update", "delete", "create"]):
            return True
        return kw_count >= 2

    def extract_objects(text: str, is_py: bool = False) -> set[str]:
        objects = set()
        if is_py:
            for literal in py_string_pattern.findall(text):
                if is_likely_sql(literal):
                    for pattern in db_ref_patterns:
                        for match in pattern.finditer(literal):
                            name = match.group(1)
                            start_idx = match.start(1)
                            end_idx = match.end(1)
                            
                            line_start = literal.rfind('\n', 0, match.start()) + 1
                            line_end = literal.find('\n', match.end())
                            if line_end < 0:
                                line_end = len(literal)
                            line_content = literal[line_start:line_end].strip()
                            
                            if "import " in line_content and re.match(r"^\s*from\s+[a-zA-Z0-9_\.]+\s+import\b", line_content, re.IGNORECASE):
                                continue
                                
                            if start_idx > 0 and literal[start_idx - 1] == '{':
                                continue
                            if end_idx < len(literal) and literal[end_idx] in ('(', '}'):
                                continue
                            objects.add(name.split(".")[-1].lower())
        else:
            for pattern in db_ref_patterns:
                for match in pattern.finditer(text):
                    name = match.group(1)
                    start_idx = match.start(1)
                    end_idx = match.end(1)
                    if start_idx > 0 and text[start_idx - 1] == '{':
                        continue
                    if end_idx < len(text) and text[end_idx] in ('(', '}'):
                        continue
                    objects.add(name.split(".")[-1].lower())
        return objects


    for rel in changed_files:
        path = (project_root / rel.replace("\\", "/").lstrip("./")).resolve()
        if not path.is_file():
            continue
        
        try:
            current_content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        
        is_py = (path.suffix.lower() == ".py")
        new_objects = extract_objects(current_content, is_py=is_py)
        ctes = {cte.split(".")[-1].lower() for cte in cte_pattern.findall(current_content)}
        
        original_content = get_git_head_content(project_root, rel)
        if original_content is not None:
            old_objects = extract_objects(original_content, is_py=is_py)
        else:
            old_objects = set()
            
        introduced_objects = new_objects - old_objects - ctes
        
        for obj in introduced_objects:
            if obj not in repo_db_objects:
                errors.append(
                    f"{rel} references database object '{obj}' which does not exist in the repository's .sql definitions."
                )
                
    return errors





class ValidatorSkill:
    name = "validator"

    def __init__(self, *, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        changed_files = tuple(str(path) for path in kwargs.get("changed_files", ()) or ())
        if not changed_files:
            return SkillResult(
                success=True,
                summary="No changed files to validate.",
                validation_result="skipped",
            )

        errors: list[str] = []

        original_files = {}
        for source in (kwargs, context.metadata if context else None):
            if not isinstance(source, dict):
                continue
            for k in ("original_files_json", "original_files"):
                val = source.get(k)
                if isinstance(val, dict):
                    original_files.update(val)
                elif isinstance(val, str) and val.strip():
                    try:
                        d = json.loads(val)
                        if isinstance(d, dict):
                            original_files.update(d)
                    except Exception:
                        pass
        
        # Extract allowed snippet ranges from context_pack
        snippet_ranges = []
        if context and context.context_pack:
            source_snippets = context.context_pack.focused_snippets or context.context_pack.snippets
            for snippet in source_snippets:
                snippet_ranges.append({
                    "file": snippet.file_path,
                    "start_line": snippet.start_line,
                    "end_line": snippet.end_line
                })

        # 1. Base SQL references check
        db_errors = validate_sql_references(self.project_root, changed_files)
        errors.extend(db_errors)

        # Build repository database objects set for schema validation
        repo_db_objects = set()
        pattern = re.compile(
            r"\bCREATE\s+[^;]*?\b(?:VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\.\"'\`\[\]]+)",
            re.IGNORECASE
        )
        for path in self.project_root.rglob("*"):
            try:
                if path.suffix.lower() in {".sql", ".py"} and path.is_file():
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    for match in pattern.finditer(content):
                        raw_name = match.group(1)
                        cleaned = re.sub(r'[\"\'\`\[\]]', '', raw_name)
                        repo_db_objects.add(cleaned.lower())
                        if '.' in cleaned:
                            repo_db_objects.add(cleaned.split('.')[-1].lower())
            except Exception:
                pass

        # Extract target symbols and target view from EDIT_CONTEXT_JSON if present, else fallback
        evidence = str(kwargs.get("search_output") or kwargs.get("evidence") or "").strip()
        if not evidence and context:
            evidence = getattr(context, "search_output", "") or ""
            if not evidence and hasattr(context, "context_pack") and context.context_pack:
                evidence = getattr(context.context_pack, "evidence", "") or ""

        edit_context = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
        target_view = ""
        target_symbols = []
        is_view_change = False

        if isinstance(edit_context, dict):
            # 1. 提取 target_view / replacement_source
            resolved_deps = edit_context.get("resolved_dependencies")
            if isinstance(resolved_deps, list):
                for dep in resolved_deps:
                    if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                        target_view = str(dep.get("name") or "").strip()
                        break
            if not target_view:
                target_view = str(edit_context.get("target_view") or "").strip()

            # 2. 提取 target_symbols
            task_intent = edit_context.get("task_intent")
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
                    is_view_change = True
                tsym = task_intent.get("target_symbol")
                if tsym:
                    target_symbols.append(str(tsym))
            
            edit_targets = edit_context.get("edit_targets")
            if not primary_target_symbol and isinstance(edit_targets, list):
                for t in edit_targets:
                    if isinstance(t, dict) and t.get("symbol"):
                        target_symbols.append(str(t.get("symbol")))
                        strict_symbols_check = True
            
            editable_targets = edit_context.get("editable_targets") or edit_context.get("snippets")
            if not primary_target_symbol and isinstance(editable_targets, list):
                for t in editable_targets:
                    if isinstance(t, dict) and t.get("symbol"):
                        target_symbols.append(str(t.get("symbol")))
                        strict_symbols_check = True
            patch_intent = edit_context.get("patch_intent")
            if isinstance(patch_intent, dict):
                patch_targets = patch_intent.get("edit_targets")
                if isinstance(patch_targets, list):
                    for t in patch_targets:
                        if isinstance(t, dict) and t.get("symbol"):
                            target_symbols.append(str(t.get("symbol")))
                            strict_symbols_check = True

        # Extract/override from kwargs if provided
        if not target_type and kwargs.get("target_type"):
            target_type = str(kwargs.get("target_type")).strip()
        if not local_target and kwargs.get("local_target"):
            local_target = str(kwargs.get("local_target")).strip()
        if not parent_symbol and kwargs.get("parent_symbol"):
            parent_symbol = str(kwargs.get("parent_symbol")).strip()

        # Fallback to contract extraction
        handoff_contract = kwargs.get("handoff_contract")
        if isinstance(handoff_contract, dict):
            must_modify = handoff_contract.get("must_modify") or []
            for item in must_modify:
                if isinstance(item, dict):
                    should_change = str(item.get("should_change_to") or "")
                    if not target_view:
                        match = re.search(r"use view\s+([a-zA-Z0-9_.]+)", should_change, re.IGNORECASE)
                        if match:
                            target_view = match.group(1)
                    symbol_or_api = str(item.get("symbol_or_api") or "")
                    if symbol_or_api and symbol_or_api != "目标代码" and symbol_or_api not in target_symbols:
                        target_symbols.append(symbol_or_api)
                        strict_symbols_check = True

        if not target_symbols:
            target_symbols = _extract_symbols_from_text(context.user_request)

        # Final fallback for primary_target_symbol and parent_symbol
        if not parent_symbol and target_symbols:
            parent_symbol = target_symbols[0]
        if not primary_target_symbol and parent_symbol:
            primary_target_symbol = parent_symbol


        checked: list[str] = []
        validation_details_list = []

        for rel in changed_files:
            path = _resolve_under_root(self.project_root, rel)
            if path is None:
                errors.append(f"{rel}: outside project root")
                continue
            if not path.is_file():
                errors.append(f"{rel}: file not found")
                continue
            checked.append(rel)

            # Python syntax check
            if path.suffix == ".py":
                try:
                    py_compile.compile(str(path), doraise=True)
                except py_compile.PyCompileError as exc:
                    errors.append(f"{rel}: py_compile failed: {exc.msg}")
                    continue

                try:
                    res = subprocess.run(
                        ["ruff", "check", "--select", "F821,F822,F823", str(path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        cwd=str(self.project_root),
                    )
                    if res.returncode != 0 and res.stdout.strip():
                        for line in res.stdout.splitlines():
                            if line.strip() and ("Undefined name" in line or "F82" in line):
                                errors.append(f"{rel}: ruff check failed: {line.strip()}")
                except FileNotFoundError:
                    pass

            # Diff-based rich validation (PatchIntentValidator)
            # Retrieve original content from original_files dict (pre-edit snapshot) or fall back to git
            old_content = None
            old_source = "git_head"
            if rel in original_files:
                old_content = original_files[rel]
                old_source = "original_files_json"
            else:
                old_content = get_git_head_content(self.project_root, rel)
            if old_content is None:
                continue

            try:
                new_content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            import difflib
            diff_lines = list(difflib.unified_diff(
                old_content.splitlines(),
                new_content.splitlines(),
                fromfile="old",
                tofile="new",
                lineterm=""
            ))

            if analysis_strategy == "function_refactor":
                errors.extend(_validate_function_refactor_contract(old_content, new_content))

            if not diff_lines:
                continue

            modified_line_nos = _parse_modified_old_lines(diff_lines)
            removed_text, added_text, diff_summary = _extract_diff_hunks(diff_lines)

            added_identifiers = list(_extract_identifiers(added_text) - _extract_identifiers(removed_text))
            removed_identifiers = list(_extract_identifiers(removed_text) - _extract_identifiers(added_text))

            # Find changed symbols (classes/functions)
            old_symbol_ranges = _find_symbol_ranges(old_content)
            
            # Print validator debugging logs
            parent_range = old_symbol_ranges.get("admin_list_orders")
            local_ranges = _find_variable_ranges(old_content, "count_sql")
            print(f"[DEBUG] validator old_source = {old_source}")
            print(f"[DEBUG] validator parent_range = {parent_range}")
            print(f"[DEBUG] validator local_ranges = {local_ranges}")
            
            changed_symbols = []
            
            is_sql_variable_attribution = False
            if path.suffix == ".py" and target_type == "sql_variable" and local_target and parent_symbol:
                if parent_symbol in old_symbol_ranges:
                    parent_start, parent_end = old_symbol_ranges[parent_symbol]
                    local_ranges = _find_variable_ranges(old_content, local_target)
                    matching_ranges = [
                        (start, end) for start, end in local_ranges
                        if parent_start <= start and end <= parent_end
                    ]
                    
                    if not matching_ranges:
                        errors.append(
                            f"Intent validation failed: local target '{local_target}' range not found "
                            f"within parent symbol '{parent_symbol}' in {rel}."
                        )
                    else:
                        out_of_range_lines = []
                        for line_no in modified_line_nos:
                            in_any_range = any(
                                start <= line_no <= end for start, end in matching_ranges
                            )
                            if not in_any_range:
                                out_of_range_lines.append(line_no)
                        
                        if out_of_range_lines:
                            errors.append(
                                f"Intent validation failed: changed lines {out_of_range_lines} in {rel} "
                                f"fall outside the local target '{local_target}' range inside '{parent_symbol}'."
                            )
                        else:
                            changed_symbols = [parent_symbol]
                            is_sql_variable_attribution = True

            if not is_sql_variable_attribution:
                for sym, (start, end) in old_symbol_ranges.items():
                    if any(start <= line_no <= end for line_no in modified_line_nos):
                        changed_symbols.append(sym)
                        
            if primary_target_symbol:
                outside_symbols = [
                    sym for sym in changed_symbols
                    if sym != primary_target_symbol
                ]
                if outside_symbols:
                    errors.append(
                        "Intent validation failed: modified symbols outside target "
                        f"'{primary_target_symbol}': {outside_symbols}"
                    )

            # 1. Extract target view if not set yet (only for view change tasks)
            if is_view_change and not target_view:
                matching_views_added = [
                    ident for ident in added_identifiers
                    if ident.lower() in repo_db_objects
                ]
                if matching_views_added:
                    target_view = matching_views_added[0]

            # 2. Check snippet range boundaries if snippets are available (prevent out-of-bounds edit)
            if snippet_ranges:
                abs_rel = _resolve_under_root(self.project_root, rel)
                if abs_rel:
                    file_ranges = []
                    for r in snippet_ranges:
                        r_abs = _resolve_under_root(self.project_root, r["file"])
                        if r_abs and r_abs == abs_rel:
                            file_ranges.append(r)
                    
                    if file_ranges:
                        for line_no in modified_line_nos:
                            in_range = any(
                                r["start_line"] <= line_no <= r["end_line"]
                                for r in file_ranges
                            )
                            if not in_range:
                                errors.append(
                                    f"Intent validation failed: modified line {line_no} in {rel} "
                                    f"is outside any allowed snippet ranges: {[(r['start_line'], r['end_line']) for r in file_ranges]}"
                                )

            # 3. Verify view reference, existence, and old table elimination
            if target_view and any(sym in changed_symbols for sym in target_symbols):
                view_referenced = (
                    target_view in added_identifiers 
                    or target_view.lower() in [ident.lower() for ident in added_identifiers]
                )
                if not view_referenced:
                    errors.append(
                        f"Intent validation failed: target view '{target_view}' is not referenced in the diff of {rel} (added identifiers: {added_identifiers})."
                    )

                if target_view.lower() not in repo_db_objects:
                    errors.append(
                        f"Intent validation failed: target view '{target_view}' does not exist in the repository's database schema definitions."
                    )

                new_symbol_ranges = _find_symbol_ranges(new_content)
                for sym in target_symbols:
                    if sym in changed_symbols and sym in new_symbol_ranges:
                        sym_start, sym_end = new_symbol_ranges[sym]
                        new_lines = new_content.splitlines()[sym_start - 1 : sym_end]
                        new_function_code = "\n".join(new_lines)
                        
                        if sym in old_symbol_ranges:
                            old_start, old_end = old_symbol_ranges[sym]
                            old_lines = old_content.splitlines()[old_start - 1 : old_end]
                            old_function_code = "\n".join(old_lines)
                            
                            # Tables that should no longer be queried (only the primary FROM tables in the original SQL queries)
                            old_sqls = []
                            string_pattern = re.compile(
                                r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
                                re.DOTALL
                            )
                            for match in string_pattern.finditer(old_function_code):
                                quote_content = match.group("quote")
                                inner = quote_content[3:-3] if quote_content.startswith(('"""', "'''")) else quote_content[1:-1]
                                if "SELECT" in inner.upper() and "FROM" in inner.upper():
                                    old_sqls.append(inner)
                                    
                            replaces_objects = []
                            if isinstance(edit_context, dict):
                                resolved_deps = edit_context.get("resolved_dependencies") or []
                                for dep in resolved_deps:
                                    if isinstance(dep, dict) and dep.get("role") == "replacement_source":
                                        replaces_objects = list(dep.get("replaces_objects") or [])
                                        break
                                        
                            forbidden_tables = set()
                            if replaces_objects:
                                for tbl in replaces_objects:
                                    forbidden_tables.add(tbl.lower())
                            else:
                                for sql in old_sqls:
                                    forbidden_tables.update(_extract_primary_tables(sql))
                            
                            forbidden_tables = forbidden_tables - {target_view.lower()}
                            
                            still_present = [
                                tbl for tbl in forbidden_tables
                                if re.search(rf"\b{re.escape(tbl)}\b", new_function_code, re.IGNORECASE)
                            ]
                            if still_present:
                                errors.append(
                                    f"Intent validation failed: old table/view '{still_present[0]}' is still "
                                    f"referenced in the modified function '{sym}'."
                                )
                                
                            removed_aliases = set()
                            for sql in old_sqls:
                                removed_aliases.update(_extract_removed_aliases(sql, forbidden_tables))
                            still_present_aliases = [
                                alias for alias in removed_aliases
                                if re.search(rf"\b{re.escape(alias)}\b\.", new_function_code, re.IGNORECASE)
                            ]
                            if still_present_aliases:
                                errors.append(
                                    f"Intent validation failed: legacy alias '{still_present_aliases[0]}' is still "
                                    f"referenced in the modified function '{sym}'."
                                )
                            
                            semantic_errors = _validate_sql_semantic_changes(old_function_code, new_function_code)
                            errors.extend(semantic_errors)
                            contract_errors = _validate_sql_replacement_contract(
                                old_function_code,
                                new_function_code,
                                target_view=target_view,
                                replaces_objects=replaces_objects,
                                dependency_columns=_replacement_dependency_columns(edit_context),
                                target_sql_kind=target_sql_kind,
                            )
                            errors.extend(contract_errors)
                            if target_sql_kind == "count":
                                list_errors = _validate_non_count_queries_unchanged(
                                    old_function_code,
                                    new_function_code,
                                    target_sql_variable=target_sql_variable,
                                )
                                errors.extend(list_errors)

            import hashlib
            file_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
            details = {
                "changed_file": rel,
                "changed_symbols": list(dict.fromkeys(changed_symbols)),
                "diff_summary": diff_summary[:1000] + "..." if len(diff_summary) > 1000 else diff_summary,
                "added_identifiers": added_identifiers,
                "removed_identifiers": removed_identifiers,
                "file_hash": file_hash,
            }
            validation_details_list.append(details)

        # Check that target symbol was modified somewhere in the changes
        if target_symbols and strict_symbols_check:
            all_modified_symbols = []
            for details in validation_details_list:
                all_modified_symbols.extend(details["changed_symbols"])
            if not any(sym in all_modified_symbols for sym in target_symbols):
                errors.append(
                    f"Intent validation failed: none of the target symbols {target_symbols} "
                    f"were modified in the changes. All modified symbols: {all_modified_symbols}"
                )

        # Check that target view dependency was used/referenced in the diff
        if target_view:
            view_referenced = False
            for details in validation_details_list:
                added_idents = [i.lower() for i in details.get("added_identifiers", [])]
                if target_view.lower() in added_idents:
                    view_referenced = True
                    break
            if not view_referenced:
                errors.append(
                    f"Intent validation failed: target dependency '{target_view}' is not referenced/used in the new code diff."
                )

        if errors:
            requires_fallback = _is_replannable_validation_error(errors)
            return SkillResult(
                success=False,
                summary="Validation failed: " + "; ".join(errors),
                validation_result="failed",
                missing_info=tuple(errors),
                requires_fallback=requires_fallback,
                metadata={
                    "failure_code": (
                        "validator_context_mismatch"
                        if requires_fallback
                        else "validator_acceptance_failed"
                    ),
                    "structured_errors": json.dumps(
                        _structured_validation_errors(errors),
                        ensure_ascii=False,
                    ),
                },
            )

        rich_summary = f"Validated {len(checked)} changed file(s)."
        if validation_details_list:
            first = validation_details_list[0]
            rich_summary = (
                f"Validated {first['changed_file']} (hash={first['file_hash'][:8]}). "
                f"Modified symbols: {first['changed_symbols']}. "
                f"Added identifiers: {first['added_identifiers']}."
            )

        return SkillResult(
            success=True,
            summary=rich_summary,
            validation_result="passed",
            metadata={
                "validation_details": json.dumps(validation_details_list, ensure_ascii=False)
            }
        )


def _is_replannable_validation_error(errors: list[str]) -> bool:
    patterns = (
        "outside any allowed snippet ranges",
        "none of the target symbols",
        "patch_validator cannot locate current_code",
        "symbol mismatch",
        "range mismatch",
        "target_context_missing",
        "simplified_context_requires_deterministic_patch",
    )
    return any(any(pattern in error for pattern in patterns) for error in errors)


def _structured_validation_errors(errors: list[str]) -> list[dict[str, str]]:
    structured: list[dict[str, str]] = []
    for error in errors:
        code = "validator_acceptance_failed"
        if "outside any allowed snippet ranges" in error:
            code = "snippet_range_mismatch"
        elif "none of the target symbols" in error:
            code = "target_symbol_mismatch"
        elif "patch_validator cannot locate current_code" in error:
            code = "current_code_mismatch"
        elif "target_context_missing" in error:
            code = "target_context_missing"
        elif "simplified_context_requires_deterministic_patch" in error:
            code = "simplified_context_requires_deterministic_patch"
        structured.append({"code": code, "message": error})
    return structured


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        p = Path(rel).resolve()
        if p.is_absolute():
            p.relative_to(project_root)
            return p
    except (OSError, ValueError):
        pass
    try:
        path = (project_root / rel.replace("\\", "/").lstrip("./")).resolve()
        path.relative_to(project_root)
        return path
    except (OSError, ValueError):
        return None


def _find_symbol_ranges(content: str) -> dict[str, tuple[int, int]]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return _find_symbol_ranges_by_indent(content)

    ranges = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = getattr(node, "lineno", 1)
            end = getattr(node, "end_lineno", start)
            ranges[node.name] = (start, end)
    return ranges


def _find_symbol_ranges_by_indent(content: str) -> dict[str, tuple[int, int]]:
    """Find the start and end line (1-indexed) of each class/function definition."""
    lines = content.splitlines()
    ranges = {}
    for idx, line in enumerate(lines):
        match = re.match(r"^\s*(?:async\s+def|def|class)\s+(?P<name>\w+)", line)
        if match:
            name = match.group("name")
            start = idx
            indent = len(line) - len(line.lstrip(" "))
            end = idx
            for j in range(idx + 1, len(lines)):
                if not lines[j].strip():
                    continue
                line_indent = len(lines[j]) - len(lines[j].lstrip(" "))
                if line_indent <= indent:
                    break
                end = j
            ranges[name] = (start + 1, end + 1)
    return ranges


def _find_variable_ranges(content: str, var_name: str) -> list[tuple[int, int]]:
    """Find the start and end lines (1-indexed, inclusive) of assignments to var_name."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        # Fallback to regex if syntax error
        ranges = []
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            if re.search(rf"\b{re.escape(var_name)}\s*=", line):
                ranges.append((idx + 1, idx + 1))
        return ranges

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
    return ranges



def _validate_non_count_queries_unchanged(
    old_code: str,
    new_code: str,
    *,
    target_sql_variable: str,
) -> list[str]:
    old_assignments = _query_assignments(old_code)
    new_assignments = _query_assignments(new_code)
    errors: list[str] = []
    for name, old_value in old_assignments.items():
        lowered = name.lower()
        if name == target_sql_variable or "count" in lowered or "total" in lowered:
            continue
        if name not in new_assignments:
            errors.append(
                f"Intent validation failed: non-count query variable '{name}' was removed."
            )
        elif new_assignments[name] != old_value:
            errors.append(
                f"Intent validation failed: non-count query variable '{name}' was modified."
            )
    return errors


def _query_assignments(code: str) -> dict[str, str]:
    try:
        tree = ast.parse(textwrap.dedent(code))
    except SyntaxError:
        return {}
    assignments: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            lowered = name.lower()
            if "sql" not in lowered and "query" not in lowered:
                continue
            assignments[name] = ast.dump(value, include_attributes=False)
    return assignments


def _parse_modified_old_lines(diff_lines: list[str]) -> list[int]:
    """Parse unified diff lines to extract line numbers in the old file."""
    modified_lines = []
    old_line_no = 0
    for line in diff_lines:
        if line.startswith("@@"):
            match = re.match(r"^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@", line)
            if match:
                old_line_no = int(match.group(1))
        elif line.startswith("-") and not line.startswith("---"):
            modified_lines.append(old_line_no)
            old_line_no += 1
        elif line.startswith("+") and not line.startswith("+++"):
            pass
        elif line.startswith(" "):
            old_line_no += 1
    return modified_lines


def _extract_diff_hunks(diff_lines: list[str]) -> tuple[str, str, str]:
    """Extract removed lines, added lines, and a clean diff summary."""
    removed_parts = []
    added_parts = []
    summary_lines = []
    for line in diff_lines:
        if line.startswith("@@"):
            summary_lines.append(line)
        elif line.startswith("-") and not line.startswith("---"):
            removed_parts.append(line[1:])
            summary_lines.append(line)
        elif line.startswith("+") and not line.startswith("+++"):
            added_parts.append(line[1:])
            summary_lines.append(line)
    return "\n".join(removed_parts), "\n".join(added_parts), "\n".join(summary_lines)


def _extract_identifiers(text: str) -> set[str]:
    """Extract identifiers, ignoring common Python keywords and builtins."""
    keywords = {
        "def", "class", "async", "await", "return", "import", "from", "as",
        "if", "else", "elif", "for", "while", "in", "not", "and", "or", "try", "except",
        "pass", "True", "False", "None", "global", "nonlocal", "lambda", "yield",
        "with", "assert", "break", "continue", "del", "raise", "finally", "text",
        "str", "int", "dict", "list", "tuple", "set"
    }
    tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
    return {t for t in tokens if t not in keywords}


def _extract_symbols_from_text(text: str) -> list[str]:
    """Extract candidate symbols from task instruction text."""
    raw_tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text)
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


def _extract_select_fields(sql_str: str) -> set[str]:
    import sqlparse
    parsed = sqlparse.parse(sql_str)
    if not parsed:
        return set()
    statement = parsed[0]
    
    tokens = statement.tokens
    select_idx = -1
    from_idx = -1
    for idx, token in enumerate(tokens):
        if token.ttype is sqlparse.tokens.Keyword.DML and token.value.upper() == "SELECT":
            select_idx = idx
        elif token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
            from_idx = idx
            break
            
    if select_idx == -1 or from_idx == -1 or from_idx <= select_idx:
        return set()
        
    select_tokens = tokens[select_idx + 1:from_idx]
    fields = set()
    for t in select_tokens:
        if t.is_whitespace:
            continue
        if t.ttype is sqlparse.tokens.Wildcard or t.value == "*":
            fields.add("*")
        elif isinstance(t, sqlparse.sql.IdentifierList):
            for ident in t.get_identifiers():
                if ident.ttype is sqlparse.tokens.Wildcard or ident.value == "*":
                    fields.add("*")
                else:
                    alias = ident.get_alias()
                    real_name = ident.get_real_name()
                    name = alias or real_name or ident.value
                    fields.add(name.strip().split('.')[-1].lower())
        elif isinstance(t, sqlparse.sql.Identifier):
            alias = t.get_alias()
            real_name = t.get_real_name()
            name = alias or real_name or t.value
            fields.add(name.strip().split('.')[-1].lower())
        else:
            if '*' in t.value:
                fields.add('*')
            words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", t.value)
            for w in words:
                fields.add(w.lower())
                
    keywords = {"cast", "char", "as", "coalesce", "case", "when", "then", "else", "end"}
    return {f for f in fields if f not in keywords}



def _validate_sql_semantic_changes(old_code: str, new_code: str) -> list[str]:
    errors = []
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
        re.DOTALL
    )
    
    old_sqls = []
    for match in string_pattern.finditer(old_code):
        quote_content = match.group("quote")
        inner = quote_content[3:-3] if quote_content.startswith(('"""', "'''")) else quote_content[1:-1]
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            old_sqls.append(inner)
            
    new_sqls = []
    for match in string_pattern.finditer(new_code):
        quote_content = match.group("quote")
        inner = quote_content[3:-3] if quote_content.startswith(('"""', "'''")) else quote_content[1:-1]
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            new_sqls.append(inner)
            
    for old_sql, new_sql in zip(old_sqls, new_sqls):
        old_fields = _extract_select_fields(old_sql)
        new_fields = _extract_select_fields(new_sql)
        
        if "*" in new_fields and "*" not in old_fields:
            errors.append("SQL semantic check failed: Avoid using SELECT *, please preserve the original SELECT field list.")
            
        missing = old_fields - new_fields
        if missing and "*" not in new_fields:
            errors.append(f"SQL semantic check failed: SELECT fields are lost: {list(missing)}")
            
    return errors


def _replacement_dependency_columns(edit_context: dict[str, object] | None) -> list[str]:
    if not isinstance(edit_context, dict):
        return []
    resolved_deps = edit_context.get("resolved_dependencies") or []
    if not isinstance(resolved_deps, list):
        return []
    for dep in resolved_deps:
        if isinstance(dep, dict) and dep.get("role") == "replacement_source":
            return [str(col) for col in dep.get("columns") or [] if str(col).strip()]
    return []


def _validate_sql_replacement_contract(
    old_code: str,
    new_code: str,
    *,
    target_view: str,
    replaces_objects: list[str],
    dependency_columns: list[str],
    target_sql_kind: str = "",
) -> list[str]:
    errors: list[str] = []
    if not target_view or not replaces_objects:
        return errors

    old_sqls = _extract_sql_literals(old_code)
    new_sqls = _extract_sql_literals(new_code)
    columns_lower = {col.lower() for col in dependency_columns if col.strip()}
    forbidden_tables = {obj.split(".")[-1].lower() for obj in replaces_objects if str(obj).strip()}
    forbidden_tables.discard(target_view.split(".")[-1].lower())

    for idx, new_sql in enumerate(new_sqls):
        if target_view.lower() not in new_sql.lower():
            continue
        compile_error = _sql_compile_error(new_sql)
        if compile_error:
            errors.append(f"SQL replacement contract failed: SQL does not compile: {compile_error}")

        if dependency_columns:
            selected = _extract_select_fields(new_sql)
            is_count = (target_sql_kind == "count") or ("count(" in new_sql.lower()) or ("count (" in new_sql.lower())
            
            if not is_count:
                if "*" in selected:
                    errors.append("SQL replacement contract failed: SELECT * is not allowed with a resolved dependency column list.")
                
                ignored_count_fields = {"total", "count", "total_orders", "total_rows", "cnt"}
                missing = []
                for field in selected:
                    if field == "*":
                        continue
                    if field in ignored_count_fields:
                        continue
                    if field not in columns_lower:
                        missing.append(field)
                
                missing = sorted(missing)
                if missing:
                    errors.append(
                        "SQL replacement contract failed: SELECT fields are not in "
                        f"resolved_dependency.columns: {missing}"
                    )

        remaining_tables = _extract_all_tables(new_sql)
        still_joined = sorted(tbl for tbl in forbidden_tables if tbl in remaining_tables)
        if still_joined:
            errors.append(
                "SQL replacement contract failed: replaces_objects still referenced "
                f"in FROM/JOIN: {still_joined}"
            )

        if dependency_columns:
            old_sql = old_sqls[idx] if idx < len(old_sqls) else "\n".join(old_sqls)
            aliases = _extract_aliases_for_tables(old_sql, forbidden_tables)
            select_where = _select_and_where_text(new_sql)
            stale_aliases = sorted(
                alias for alias in aliases
                if re.search(rf"\b{re.escape(alias)}\s*\.", select_where, re.IGNORECASE)
            )
            if stale_aliases:
                errors.append(
                    "SQL replacement contract failed: old aliases still appear in "
                    f"SELECT/WHERE: {stale_aliases}"
                )
    return errors


def _validate_function_refactor_contract(old_code: str, new_code: str) -> list[str]:
    errors: list[str] = []
    old_manual_refs = len(re.findall(r"\bpassenger_id_no\b", old_code))
    new_manual_refs = len(re.findall(r"\bpassenger_id_no\b", new_code))
    uses_unified_helper = bool(
        re.search(r"\b(?:_fetch_order_detail|normalize_order_record)\b", new_code)
    )
    if old_manual_refs and not uses_unified_helper:
        errors.append(
            "Function refactor validation failed: expected unified helper "
            "(_fetch_order_detail or normalize_order_record) to be used."
        )
    if old_manual_refs > 1 and new_manual_refs >= old_manual_refs and uses_unified_helper:
        errors.append(
            "Function refactor validation failed: manual passenger_id_no logic was not reduced."
        )
    return errors


def _extract_sql_literals(code: str) -> list[str]:
    string_pattern = re.compile(
        r'(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
        re.DOTALL
    )
    sqls: list[str] = []
    for match in string_pattern.finditer(code):
        quote_content = match.group("quote")
        inner = quote_content[3:-3] if quote_content.startswith(('"""', "'''")) else quote_content[1:-1]
        if "SELECT" in inner.upper() and "FROM" in inner.upper():
            sqls.append(inner)
    return sqls


def _sql_compile_error(sql: str) -> str:
    try:
        parsed = sqlparse.parse(sql)
    except Exception as exc:
        return str(exc)
    if not parsed:
        return "empty parse"
    text = str(parsed[0]).upper()
    if "SELECT" not in text or "FROM" not in text:
        return "missing SELECT/FROM"
    return ""


def _extract_aliases_for_tables(sql: str, table_names: set[str]) -> set[str]:
    aliases: set[str] = set()
    pattern = re.compile(
        r"\b(?:FROM|JOIN)\s+[`\"\[]?(?P<table>[a-zA-Z0-9_.]+)[`\"\]]?"
        r"(?:\s+(?:AS\s+)?(?P<alias>[a-zA-Z_][a-zA-Z0-9_]*))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sql):
        table = match.group("table").split(".")[-1].lower()
        if table not in table_names:
            continue
        alias = match.group("alias")
        aliases.add(table)
        if alias and alias.upper() not in {
            "ON", "WHERE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "FULL",
            "GROUP", "ORDER", "LIMIT", "HAVING",
        }:
            aliases.add(alias)
    return aliases


def _select_and_where_text(sql: str) -> str:
    upper = sql.upper()
    select_start = upper.find("SELECT")
    from_start = upper.find("FROM", select_start + 6)
    where_start = upper.find("WHERE", from_start + 4)
    parts: list[str] = []
    if select_start >= 0 and from_start > select_start:
        parts.append(sql[select_start:from_start])
    if where_start >= 0:
        end = len(sql)
        for keyword in (" GROUP ", " ORDER ", " LIMIT ", " HAVING ", " UNION "):
            pos = upper.find(keyword, where_start + 5)
            if pos >= 0:
                end = min(end, pos)
        parts.append(sql[where_start:end])
    return "\n".join(parts)


def _extract_primary_tables(sql_str: str) -> set[str]:
    import sqlparse
    tables = set()
    parsed = sqlparse.parse(sql_str)
    for stmt in parsed:
        tokens = stmt.tokens
        for idx, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() == "FROM":
                for j in range(idx + 1, len(tokens)):
                    t = tokens[j]
                    if t.is_whitespace:
                        continue
                    if isinstance(t, sqlparse.sql.Identifier):
                        real_name = t.get_real_name()
                        name = real_name or t.value
                        tables.add(name.strip().split('.')[-1].lower())
                    else:
                        tables.add(t.value.strip().split('.')[-1].lower())
                    break
    return tables


def _extract_all_tables(sql_str: str) -> set[str]:
    import sqlparse
    tables = set()
    parsed = sqlparse.parse(sql_str)
    for stmt in parsed:
        tokens = stmt.tokens
        for idx, token in enumerate(tokens):
            if token.ttype is sqlparse.tokens.Keyword and token.value.upper() in ("FROM", "JOIN"):
                for j in range(idx + 1, len(tokens)):
                    t = tokens[j]
                    if t.is_whitespace:
                        continue
                    if isinstance(t, sqlparse.sql.Identifier):
                        real_name = t.get_real_name()
                        name = real_name or t.value
                        tables.add(name.strip().split('.')[-1].lower())
                    elif isinstance(t, sqlparse.sql.IdentifierList):
                        for ident in t.get_identifiers():
                            real_name = ident.get_real_name()
                            name = real_name or ident.value
                            tables.add(name.strip().split('.')[-1].lower())
                    else:
                        tables.add(t.value.strip().split('.')[-1].lower())
                    break
    return tables


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
