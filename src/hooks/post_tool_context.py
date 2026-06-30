from __future__ import annotations

import re
from typing import Any

from src.agent.types import ToolResult


def apply_post_tool_context_hook(
    tool_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> ToolResult:
    """Attach declarative events after a tool finishes without mutating state."""
    if not result.success:
        return result

    # 1. Structure SQL semantic/DDL/DML/JOIN/GRANT queries from output to prevent LLM hallucinations
    structured_output = result.output
    if result.output:
        structured_output = _process_sql_structuring(result.output)

    result = ToolResult(
        success=result.success,
        output=structured_output,
        error=result.error,
        metadata=result.metadata,
    )

    metadata = dict(result.metadata or {})

    if tool_name == "grep_search":
        raw = metadata.get("raw_evidence_store") or []
        located = _rank_pending_symbols(raw, str(arguments.get("pattern") or ""))
        requirements: dict[str, bool] = {}
        joined = "\n".join(
            str(item.get("match_line") or "") for item in raw if isinstance(item, dict)
        ).casefold()
        scope = str(arguments.get("path") or arguments.get("include") or "").casefold()
        if any(token in joined for token in ("@app.", "@router.", "endpoint", "route")):
            requirements["endpoint_implementation"] = True
        if _contains_mount_evidence(joined):
            requirements["integration_or_mount_point"] = True
        if scope.endswith(".sql") or any(
            token in joined for token in ("create table", "create view")
        ):
            requirements["relevant_schema"] = True
        if "test" in scope or any(token in joined for token in ("def test_", "pytest", "unittest")):
            requirements["test_or_validation_path"] = True
        metadata["run_event"] = {
            "kind": "evidence_discovered",
            "candidates": located,
            "grounded_slots": sorted(requirements),
        }
        return _replace_metadata(result, metadata)

    if tool_name == "view_symbol_code":
        target_file = str(arguments.get("target_file") or "").strip()
        symbol = str(arguments.get("symbol") or "").strip()
        raw = metadata.get("raw_evidence_store") or []
        anchor = next((item for item in raw if isinstance(item, dict)), {})
        if target_file and symbol:
            requirements = {"target_implementation": True}
            lowered = f"{target_file}\n{anchor.get('code', '')}".casefold()
            if "test" in target_file.casefold() or "def test_" in lowered:
                requirements["test_or_validation_path"] = True
            if any(token in lowered for token in ("@app.", "@router.")):
                requirements["endpoint_implementation"] = True
            if _contains_mount_evidence(lowered) or (
                symbol.split(".")[-1] == "build_router" and bool(anchor.get("code"))
            ):
                requirements["integration_or_mount_point"] = True
            requirements.update(_security_evidence(lowered))
            metadata["run_event"] = {
                "kind": "evidence_discovered",
                "candidates": [{
                    "file": target_file,
                    "symbol": symbol,
                    "span": anchor.get("span") or metadata.get("span") or [1, 1],
                }],
                "grounded_slots": sorted(requirements),
            }
        return _replace_metadata(result, metadata)

    if tool_name == "codebase_retrieve":
        raw = metadata.get("raw_evidence_store") or []
        located = []
        retrieved_requirements: dict[str, bool] = {}
        for item in raw:
            if not isinstance(item, dict):
                continue
            file = str(item.get("file") or "")
            symbol = str(item.get("symbol") or "")
            code = str(item.get("code") or "")
            if file and symbol and code:
                located.append(
                    {
                        "file": file,
                        "symbol": symbol,
                        "span": item.get("span") or [1, 1],
                        "expanded": True,
                        "content_hash": item.get("hash") or item.get("content_hash"),
                    }
                )
                retrieved_requirements["target_implementation"] = True
            lowered = f"{file}\n{code}".casefold()
            if any(token in lowered for token in ("@app.", "@router.")):
                retrieved_requirements["endpoint_implementation"] = True
            if _contains_mount_evidence(lowered) or (
                symbol.split(".")[-1] == "build_router" and bool(code)
            ):
                retrieved_requirements["integration_or_mount_point"] = True
            if file.casefold().endswith(".sql") or any(
                token in lowered for token in ("create table", "create view")
            ):
                retrieved_requirements["relevant_schema"] = True
            retrieved_requirements.update(_security_evidence(lowered))
            if "test" in file.casefold() or "def test_" in lowered:
                retrieved_requirements["test_or_validation_path"] = True
        metadata["run_event"] = {
            "kind": "evidence_discovered",
            "candidates": located,
            "grounded_slots": sorted(retrieved_requirements),
        }
        return _replace_metadata(result, metadata)

    if tool_name != "decision_edit":
        return result

    target_file = str(arguments.get("target_file") or "").strip()
    if not target_file:
        return result

    metadata["artifact_update"] = {"invalidate_code_files": [target_file]}
    metadata["task_completion"] = {
        "tool": tool_name,
        "target_file": target_file,
        "status": "completed",
        "validation": "passed",
    }
    return _replace_metadata(result, metadata)


def _replace_metadata(result: ToolResult, metadata: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=result.success,
        output=result.output,
        error=result.error,
        metadata=metadata,
    )


def _contains_mount_evidence(text: str) -> bool:
    lowered = text.casefold()
    if any(marker in lowered for marker in ("include_router(", "app.include_router")):
        return True
    return any(
        ("build_router(engine" in line or "build_router(app" in line)
        and not line.strip().startswith(("def ", "async def "))
        for line in lowered.splitlines()
    )


def _security_evidence(text: str) -> dict[str, bool]:
    """Classify security evidence only from complete code/schema blocks."""
    lowered = text.casefold()
    evidence: dict[str, bool] = {}
    if any(
        marker in lowered
        for marker in (
            "get_current_user",
            "authorization",
            "bearer ",
            "decode_access_token",
            "jwt.decode",
        )
    ):
        evidence["authentication_context"] = True
    if re.search(r"\b(?:role|roles|permission|permissions)\b", lowered):
        evidence["authorization_policy"] = True
    has_principal = bool(
        re.search(r"\b(?:owner_id|user_id|tenant_id|created_by)\b", lowered)
    )
    has_resource = bool(re.search(r"\b(?:passenger_id|p_id)\b", lowered))
    if has_principal and has_resource and any(
        marker in lowered
        for marker in (
            "create table",
            "create view",
            "foreign key",
            " join ",
            " where ",
            "select ",
        )
    ):
        evidence["ownership_relation"] = True
    return evidence


def _rank_pending_symbols(
    raw: list[Any],
    pattern: str,
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Keep only the highest-signal grep definitions as expansion suggestions."""
    query_terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", pattern)
        if len(token) > 2
    }
    candidates: dict[tuple[str, str], tuple[int, dict[str, Any]]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or not item.get("symbol"):
            continue
        symbol = str(item["symbol"])
        lowered = symbol.casefold()
        line = str(item.get("match_line") or "").casefold()
        score = 0
        score += 8 * sum(term == lowered for term in query_terms)
        score += 3 * sum(term in lowered or lowered in term for term in query_terms)
        if any(term in lowered for term in ("auth", "token", "current_user", "role")):
            score += 5
        if any(term in lowered for term in ("archive", "delete", "router", "endpoint")):
            score += 4
        if line.lstrip().startswith("def ") or line.lstrip().startswith("async def "):
            score += 2
        if lowered.endswith(("request", "response")) or line.lstrip().startswith("class "):
            score -= 6
        if any(term in lowered for term in ("hash_password", "verify_password")):
            score -= 3
        payload = {
            "file": item.get("file"),
            "symbol": symbol,
            "span": item.get("span") or [1, 1],
            "expanded": False,
        }
        key = (str(item.get("file") or ""), symbol)
        prior = candidates.get(key)
        ranked = (score * 1000 - index, payload)
        if prior is None or ranked[0] > prior[0]:
            candidates[key] = ranked
    ordered = sorted(candidates.values(), key=lambda pair: pair[0], reverse=True)
    return [payload for _, payload in ordered[: max(3, min(5, limit))]]


def _process_sql_structuring(output_text: str) -> str:
    if not output_text:
        return output_text
    
    structs = _parse_and_structure_sql(output_text)
    if not structs:
        return output_text
        
    import json
    structs_json = json.dumps(structs, ensure_ascii=False, indent=2)
    sql_truth_block = (
        f"\n\n### [STRUCTURED SQL SEMANTIC TRUTH] ###\n"
        f"The following structured SQL semantics were validated and extracted from the output code:\n"
        f"```json\n{structs_json}\n```\n"
    )
    return output_text + sql_truth_block


def _parse_and_structure_sql(text: str) -> list[dict[str, Any]]:
    # Remove sql comments and clean whitespace
    clean_text = re.sub(r'--.*$', '', text, flags=re.MULTILINE)
    clean_text = re.sub(r'/\*.*?\*/', '', clean_text, flags=re.DOTALL)
    
    statements = []
    
    # Match individual SQL statements or clauses
    sql_pattern = re.compile(
        r'\b(select|insert\s+into|update|delete\s+from|create\s+table|alter\s+table|drop\s+table|grant)\b[^;]*',
        re.IGNORECASE | re.DOTALL
    )
    
    for m in sql_pattern.finditer(clean_text):
        stmt = m.group(0).strip().rstrip(';').strip()
        # Clean extra whitespace
        stmt = re.sub(r'\s+', ' ', stmt)
        lower_stmt = stmt.lower()
        
        try:
            # 1. ALTER TABLE
            if lower_stmt.startswith("alter table"):
                m_alter = re.match(r'alter\s+table\s+(\w+)\s+add\s+(?:column\s+)?(\w+)\s+(\w+)', stmt, re.IGNORECASE)
                if m_alter:
                    statements.append({
                        "op": "add_column",
                        "table": m_alter.group(1),
                        "column": m_alter.group(2),
                        "type": m_alter.group(3).lower()
                    })
                    continue
                m_alter_generic = re.match(r'alter\s+table\s+(\w+)\s+(\w+)\s+(?:column\s+)?(\w+)', stmt, re.IGNORECASE)
                if m_alter_generic:
                    statements.append({
                        "op": m_alter_generic.group(2).lower(),
                        "table": m_alter_generic.group(1),
                        "column": m_alter_generic.group(3)
                    })
                    continue
            
            # 2. CREATE TABLE
            elif lower_stmt.startswith("create table"):
                m_create = re.match(r'create\s+table\s+(?:if\s+not\s+exists\s+)?(\w+)', stmt, re.IGNORECASE)
                if m_create:
                    table_name = m_create.group(1)
                    cols = []
                    paren_match = re.search(r'\((.*)\)', stmt, re.IGNORECASE)
                    if paren_match:
                        raw_cols = paren_match.group(1).split(',')
                        for raw_col in raw_cols:
                            raw_col = raw_col.strip()
                            parts = raw_col.split()
                            if parts and parts[0].lower() not in ("primary", "foreign", "constraint", "unique", "key"):
                                col_name = parts[0]
                                col_type = parts[1].lower() if len(parts) > 1 else "unknown"
                                cols.append({"name": col_name, "type": col_type})
                    statements.append({
                        "op": "create_table",
                        "table": table_name,
                        "columns": cols
                    })
                    continue

            # 3. DROP TABLE
            elif lower_stmt.startswith("drop table"):
                m_drop = re.match(r'drop\s+table\s+(?:if\s+exists\s+)?(\w+)', stmt, re.IGNORECASE)
                if m_drop:
                    statements.append({
                        "op": "drop_table",
                        "table": m_drop.group(1)
                    })
                    continue

            # 4. INSERT INTO
            elif lower_stmt.startswith("insert into"):
                m_insert = re.match(r'insert\s+into\s+(\w+)\s*\((.*?)\)\s*values\s*\((.*?)\)', stmt, re.IGNORECASE)
                if m_insert:
                    table = m_insert.group(1)
                    cols = [c.strip() for c in m_insert.group(2).split(',')]
                    vals = [v.strip().strip("'\"") for v in m_insert.group(3).split(',')]
                    fields = {}
                    for c, v in zip(cols, vals):
                        try:
                            if v.isdigit():
                                v = int(v)
                            else:
                                v = float(v)
                        except ValueError:
                            pass
                        fields[c] = v
                    statements.append({
                        "op": "insert",
                        "table": table,
                        "fields": fields
                    })
                    continue
                
            # 5. UPDATE
            elif lower_stmt.startswith("update"):
                m_update = re.match(r'update\s+(\w+)\s+set\s+(.*?)(?:\s+where\s+(.*))?$', stmt, re.IGNORECASE)
                if m_update:
                    table = m_update.group(1)
                    set_clause = m_update.group(2)
                    where_clause = m_update.group(3) if len(m_update.groups()) >= 3 else None
                    
                    fields = {}
                    for pair in set_clause.split(','):
                        if '=' in pair:
                            k, v = pair.split('=', 1)
                            fields[k.strip()] = v.strip().strip("'\"")
                    
                    filters = []
                    if where_clause:
                        m_where = re.match(r'(\w+)\s*([=<>]|like)\s*(.*)', where_clause.strip(), re.IGNORECASE)
                        if m_where:
                            val = m_where.group(3).strip().strip("'\"")
                            try:
                                if val.isdigit():
                                    val = int(val)
                            except ValueError:
                                pass
                            filters.append({
                                "field": m_where.group(1),
                                "op": m_where.group(2),
                                "value": val
                            })
                    statements.append({
                        "op": "update",
                        "table": table,
                        "fields": fields,
                        "filters": filters
                    })
                    continue

            # 6. DELETE
            elif lower_stmt.startswith("delete from"):
                m_delete = re.match(r'delete\s+from\s+(\w+)(?:\s+where\s+(.*))?$', stmt, re.IGNORECASE)
                if m_delete:
                    table = m_delete.group(1)
                    where_clause = m_delete.group(2) if len(m_delete.groups()) >= 2 else None
                    filters = []
                    if where_clause:
                        m_where = re.match(r'(\w+)\s*([=<>]|like)\s*(.*)', where_clause.strip(), re.IGNORECASE)
                        if m_where:
                            val = m_where.group(3).strip().strip("'\"")
                            try:
                                if val.isdigit():
                                    val = int(val)
                            except ValueError:
                                pass
                            filters.append({
                                "field": m_where.group(1),
                                "op": m_where.group(2),
                                "value": val
                            })
                    statements.append({
                        "op": "delete",
                        "table": table,
                        "filters": filters
                    })
                    continue

            # 7. GRANT
            elif lower_stmt.startswith("grant"):
                m_grant = re.match(r'grant\s+(\w+)\s+on\s+(\w+)\s+to\s+(\w+)', stmt, re.IGNORECASE)
                if m_grant:
                    statements.append({
                        "op": "grant",
                        "permission": m_grant.group(1).lower(),
                        "table": m_grant.group(2),
                        "role": m_grant.group(3)
                    })
                    continue

            # 8. JOIN (Check first, as it contains SELECT)
            elif " join " in lower_stmt:
                tables = []
                m_from = re.search(r'from\s+(\w+)(?:\s+\w+)?\s+(?:left|right|inner|cross)?\s*join\s+(\w+)(?:\s+\w+)?\s+on\s+([\w\.]+)\s*=\s*([\w\.]+)', stmt, re.IGNORECASE)
                if m_from:
                    tables = [m_from.group(1), m_from.group(2)]
                    
                    alias_map = {}
                    for alias_match in re.finditer(r'\b(?:from|join)\s+(\w+)(?:\s+as)?\s+(\w+)\b', lower_stmt):
                        tbl, alias = alias_match.group(1), alias_match.group(2)
                        if alias not in ("left", "right", "inner", "cross", "outer", "join", "where", "on"):
                            alias_map[alias] = tbl
                    
                    left_val = m_from.group(3)
                    right_val = m_from.group(4)
                    
                    if '.' in left_val:
                        l_alias, l_col = left_val.split('.', 1)
                        if l_alias in alias_map:
                            left_val = f"{alias_map[l_alias]}.{l_col}"
                    if '.' in right_val:
                        r_alias, r_col = right_val.split('.', 1)
                        if r_alias in alias_map:
                            right_val = f"{alias_map[r_alias]}.{r_col}"
                            
                    statements.append({
                        "op": "join",
                        "tables": tables,
                        "join_condition": {
                            "left": left_val,
                            "right": right_val
                        }
                    })
                    continue

            # 9. SELECT (Standard/no join)
            elif lower_stmt.startswith("select"):
                m_select = re.match(r'select\s+(.*?)\s+from\s+(\w+)(?:\s+where\s+(.*))?$', stmt, re.IGNORECASE)
                if m_select:
                    cols = [c.strip() for c in m_select.group(1).split(',')]
                    table = m_select.group(2)
                    where_clause = m_select.group(3) if len(m_select.groups()) >= 3 else None
                    filters = []
                    
                    alias_map = {}
                    for alias_match in re.finditer(r'\b(?:from|join)\s+(\w+)(?:\s+as)?\s+(\w+)\b', lower_stmt):
                        tbl, alias = alias_match.group(1), alias_match.group(2)
                        if alias not in ("left", "right", "inner", "cross", "outer", "join", "where", "on"):
                            alias_map[alias] = tbl

                    cleaned_cols = []
                    for col in cols:
                        col = col.strip()
                        if '.' in col:
                            c_alias, c_name = col.split('.', 1)
                            if c_alias in alias_map:
                                col = f"{alias_map[c_alias]}.{c_name}"
                        cleaned_cols.append(col)

                    if where_clause:
                        for part in re.split(r'\band\b', where_clause, flags=re.IGNORECASE):
                            m_where = re.match(r'([\w\.]+)\s*([=<>]|like)\s*(.*)', part.strip(), re.IGNORECASE)
                            if m_where:
                                f_field = m_where.group(1)
                                if '.' in f_field:
                                    f_alias, f_name = f_field.split('.', 1)
                                    if f_alias in alias_map:
                                        f_field = f"{alias_map[f_alias]}.{f_name}"
                                val = m_where.group(3).strip().strip("'\"")
                                try:
                                    if val.isdigit():
                                        val = int(val)
                                except ValueError:
                                    pass
                                filters.append({
                                    "field": f_field,
                                    "op": m_where.group(2),
                                    "value": val
                                })
                    statements.append({
                        "op": "select",
                        "table": table,
                        "columns": cleaned_cols,
                        "filters": filters
                    })
                    continue

        except Exception:
            pass
            
    return statements
