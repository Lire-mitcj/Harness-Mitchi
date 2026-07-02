from __future__ import annotations

import ast
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from src.agent.contracts import ValidationDecision, ValidationResult
from src.agent.inter_llm import _strip_json_fence
from src.agent.sql_parser import UniversalSqlParser

_VIEW_RE = re.compile(r"(?:\bview\b|view_|_view|视图)", re.IGNORECASE)
_SQL_RE = re.compile(r"\b(select|from|join|create\s+view|update|insert)\b", re.IGNORECASE)
_PATCH_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\n(?P<search>.*?)\n=======\n(?P<replace>.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)
_TRIPLE_STRING_RE = re.compile(
    r"(?:[rubfRUBF]{0,4})"
    r"(?P<quote>'''|\"\"\")(?P<body>.*?)(?P=quote)",
    re.DOTALL,
)


class CursorValidator:
    """V2 validator: deterministic AST/schema gates plus advisory LLM judge."""

    def __init__(
        self,
        project_root: Path,
        *,
        command: tuple[str, ...] = ("pytest",),
        timeout: float = 120.0,
        max_error_chars: int = 2000,
        semantic_llm: Any | None = None,
        semantic_timeout: float = 30.0,
    ) -> None:
        self.project_root = project_root.resolve()
        self.command = command
        self.timeout = timeout
        self.max_error_chars = max_error_chars
        self.semantic_llm = semantic_llm
        self.semantic_timeout = semantic_timeout
        self.sql_parser = UniversalSqlParser()


    async def validate(
        self,
        *,
        target_file: str = "",
        patch: str = "",
        original_content: str = "",
        patched_content: str = "",
        user_intent: str = "",
    ) -> ValidationResult:
        ast_result = self.validate_ast(target_file, original_content, patched_content, patch)
        semantic_result = await self.validate_semantic(
            target_file=target_file,
            patch=patch,
            original_content=original_content,
            patched_content=patched_content,
            user_intent=user_intent,
        )
        execution_result = await self.validate_execution(target_file)
        decision, score = _fuse_validation(ast_result, semantic_result, execution_result)
        error = _validation_error(ast_result, semantic_result, execution_result, decision)
        return ValidationResult(
            success=decision == "commit",
            error=error,
            ast=ast_result,
            semantic=semantic_result,
            execution=execution_result,
            decision=decision,
            score=score,
        )

    def validate_ast(
        self,
        target_file: str,
        original_content: str,
        patched_content: str,
        patch: str = "",
    ) -> dict[str, object]:
        issues: list[str] = []
        trace: dict[str, object] | None = None
        suffix = Path(target_file).suffix.casefold()
        patch_blocks = _patch_blocks(patch)

        if suffix == ".py":
            try:
                patched_tree = ast.parse(patched_content)
            except SyntaxError as exc:
                patched_tree = None
                trace = {
                    "exception_type": type(exc).__name__,
                    "line_number": exc.lineno,
                    "offset": exc.offset,
                    "error_msg": exc.msg,
                }
                issues.append("python_syntax_error")
                issues.append(
                    "python_syntax_detail:"
                    f"{type(exc).__name__}:line={exc.lineno}:offset={exc.offset}:{exc.msg}"
                )
            else:
                missing = _missing_patch_scope_symbols(patch_blocks, patched_tree)
                if missing:
                    issues.append("symbol_missing:" + ",".join(missing[:5]))

        if suffix == ".sql":
            try:
                self.sql_parser.parse_file(patched_content, target_file or "<patch>")
            except Exception as exc:  # pragma: no cover - parser should be defensive
                issues.append(f"sql_syntax_error:{exc}")
        elif _SQL_RE.search(patched_content):
            sql_fragments = _patch_sql_fragments(patch_blocks)
            sql_scope = "\n".join(sql_fragments) if sql_fragments else patched_content
            if _has_unbalanced_sql_quotes(sql_scope):
                issues.append("sql_syntax_error:unbalanced_quote")

        result: dict[str, object] = {"pass": not issues, "issues": issues}
        if trace is not None:
            result["trace"] = trace
        return result

    async def validate_semantic(
        self,
        *,
        target_file: str,
        patch: str,
        original_content: str,
        patched_content: str,
        user_intent: str,
    ) -> dict[str, object]:
        schema_result = self.validate_schema(
            target_file=target_file,
            patch=patch,
            original_content=original_content,
            patched_content=patched_content,
        )
        llm_result = await self._llm_semantic_check(
            target_file=target_file,
            patch=patch,
            original_content=original_content,
            patched_content=patched_content,
            user_intent=user_intent,
        )
        details: dict[str, object] = {"schema": schema_result}
        if llm_result:
            details["llm_judge"] = llm_result
        passed = bool(schema_result.get("pass"))
        return {
            "score": 1.0 if passed else 0.0,
            "pass": passed,
            "details": details,
        }

    def validate_schema(
        self,
        *,
        target_file: str,
        patch: str,
        original_content: str,
        patched_content: str,
    ) -> dict[str, object]:
        checks: dict[str, object] = {}
        issues: list[str] = []
        missing_fields: list[str] = []

        patch_blocks = _patch_blocks(patch)
        suffix = Path(target_file).suffix.casefold()
        if suffix == ".sql":
            sql_fragments = tuple(replace for _search, replace in patch_blocks)
        else:
            sql_fragments = _patch_sql_fragments(patch_blocks)
        sql_scope = "\n".join(sql_fragments) if sql_fragments else patched_content

        if _SQL_RE.search(sql_scope):
            alias_result = _sql_alias_safety(sql_scope)
            checks["alias_safety"] = alias_result
            if alias_result["checked"] and not alias_result["pass"]:
                issues.append("DEAD_SQL_ALIAS")

            field_result = self._validate_view_semantic_binding(
                target_file=target_file,
                patch_sql=sql_scope,
                original_content=original_content,
                patched_content=patched_content,
            )
            checks["view_semantic_binding"] = field_result
            raw_missing = field_result.get("missing_fields", [])
            if isinstance(raw_missing, list):
                missing_fields = [str(item) for item in raw_missing]
            if field_result["checked"] and not field_result["pass"]:
                issues.append("SELECT_FIELD_NOT_IN_VIEW")

        return {
            "pass": not issues,
            "issues": issues,
            "checks": checks,
            "missing_fields": missing_fields,
        }

    def _validate_view_semantic_binding(
        self,
        *,
        target_file: str,
        patch_sql: str,
        original_content: str,
        patched_content: str,
    ) -> dict[str, object]:
        view_schemas = _collect_view_schemas(
            self.project_root,
            target_file=target_file,
            original_content=original_content,
            patched_content=patched_content,
        )
        original_tables = _select_tables(original_content)
        checks: list[dict[str, object]] = []
        missing: set[str] = set()
        warnings: list[str] = []
        checked = False
        for statement in _select_statements(patch_sql):
            source = _view_source(statement, view_schemas)
            if source is None:
                continue
            view_name, alias = source
            schema = view_schemas.get(view_name.casefold(), {})
            fields = _string_set(schema.get("fields", set()))
            view_tables = _string_set(schema.get("tables", set()))
            relation_equivalent = bool(
                original_tables and view_tables and original_tables <= view_tables
            )
            if not fields and not view_tables:
                checks.append({
                    "view": view_name,
                    "alias": alias,
                    "checked": False,
                    "reason": "view_schema_unavailable",
                })
                continue
            checked = True
            select_fields = _select_source_fields(statement, alias)
            statement_missing = sorted(select_fields - fields)
            field_pass = not statement_missing
            if statement_missing and relation_equivalent:
                warnings.append("FIELD_MISMATCH_ALLOWED_BY_RELATION_EQUIVALENCE")
            elif statement_missing:
                missing.update(statement_missing)
            checks.append({
                "view": view_name,
                "alias": alias,
                "checked": True,
                "select_fields": sorted(select_fields),
                "view_fields": sorted(fields),
                "original_tables": sorted(original_tables),
                "view_tables": sorted(view_tables),
                "relation_equivalent": relation_equivalent,
                "dependency_equivalent": relation_equivalent,
                "field_containment": field_pass,
                "missing_fields": statement_missing,
            })
        return {
            "checked": checked,
            "pass": not missing,
            "missing_fields": sorted(missing),
            "warnings": sorted(set(warnings)),
            "original_tables": sorted(original_tables),
            "view_schemas": {
                view: {
                    "fields": sorted(_string_set(schema.get("fields", set()))),
                    "tables": sorted(_string_set(schema.get("tables", set()))),
                }
                for view, schema in sorted(view_schemas.items())
            },
            "checks": checks,
        }

    async def _llm_semantic_check(
        self,
        *,
        target_file: str,
        patch: str,
        original_content: str,
        patched_content: str,
        user_intent: str,
    ) -> dict[str, object]:
        if self.semantic_llm is None or not user_intent.strip():
            return {}
        prompt = (
            "Explain semantic risk for this code patch. You are advisory only. "
            "Do not decide pass/fail, rollback, retry, or commit. "
            "Return strict JSON only: "
            '{"semantic_analysis":"short","risk":"low|medium|high","risks":["..."]}.'
        )
        payload = {
            "user_intent": user_intent,
            "target_file": target_file,
            "patch": patch[:6000],
            "original": original_content[:6000],
            "patched": patched_content[:6000],
        }
        try:
            response = await asyncio.wait_for(
                self.semantic_llm.chat(
                    [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": json.dumps(payload, ensure_ascii=False),
                        },
                    ],
                    tools=None,
                    stream=False,
                ),
                timeout=self.semantic_timeout,
            )
            content = getattr(response, "content", "") or ""
            data = json.loads(_strip_json_fence(content))
        except Exception as exc:
            return {
                "semantic_analysis": f"llm_unavailable:{exc}",
                "risk": "medium",
                "risks": [],
            }
        risk = str(data.get("risk", "medium")).casefold()
        if risk not in {"low", "medium", "high"}:
            risk = "medium"
        return {
            "semantic_analysis": str(data.get("semantic_analysis", ""))[:1000],
            "risk": risk,
            "risks": [str(item)[:200] for item in data.get("risks", [])[:5]]
            if isinstance(data.get("risks", []), list)
            else [],
        }

    async def validate_execution(self, target_file: str = "") -> dict[str, object]:
        command = list(self.command)
        if command and command[0] == "pytest" and target_file:
            test_file = _find_target_test_file(self.project_root, target_file)
            if test_file:
                try:
                    rel_path = test_file.relative_to(self.project_root).as_posix()
                    command.append(rel_path)
                except Exception:
                    pass
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.project_root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return {
                "pass": True,
                "status": "INFRA_ERROR",
                "warning": "execution validator command is unavailable",
                "error": f"validator command not found: {self.command[0]}",
            }
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            return {
                "pass": True,
                "status": "INFRA_ERROR",
                "warning": "execution validator timed out",
                "error": f"validator timed out after {self.timeout:g}s",
            }
        if process.returncode == 0:
            return {"pass": True, "status": "PASS", "error": ""}
        output = (stdout + b"\n" + stderr).decode(errors="replace").strip()
        if not output:
            output = f"validator exited with code {process.returncode}"
        status = _execution_status(output, process.returncode or 1)
        tail = self._tail(output)
        if status == "NO_TESTS":
            return {
                "pass": True,
                "status": "NO_TESTS",
                "warning": "pytest has no matching tests",
                "error": tail,
            }
        if status == "INFRA_ERROR":
            return {
                "pass": True,
                "status": "INFRA_ERROR",
                "warning": "execution validator infrastructure error",
                "error": tail,
            }
        return {"pass": False, "status": "FAIL", "error": tail}

    def _tail(self, output: str) -> str:
        if len(output) <= self.max_error_chars:
            return output
        return "...[truncated]\n" + output[-self.max_error_chars :]


def _parse_python(content: str) -> ast.AST | None:
    if not content.strip():
        return None
    try:
        return ast.parse(content)
    except SyntaxError:
        return None


def _python_symbols(tree: ast.AST | None) -> set[str]:
    if tree is None:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def _patch_blocks(patch: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (match.group("search"), match.group("replace"))
        for match in _PATCH_BLOCK_RE.finditer(patch)
    )


def _missing_patch_scope_symbols(
    patch_blocks: tuple[tuple[str, str], ...],
    patched_tree: ast.AST,
) -> list[str]:
    if not patch_blocks:
        return []
    patched_symbols = _python_symbols(patched_tree)
    required: set[str] = set()
    for search, replace in patch_blocks:
        required.update(_python_symbols(_parse_python(search)))
        replace_symbols = _python_symbols(_parse_python(replace))
        if replace_symbols:
            required &= replace_symbols if required else replace_symbols
    return sorted(required - patched_symbols)


def _patch_sql_fragments(patch_blocks: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    import io
    import textwrap
    import tokenize

    fragments: list[str] = []
    for _search, replace in patch_blocks:
        dedented = textwrap.dedent(replace)
        try:
            tree = ast.parse(dedented)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if _SQL_RE.search(node.value):
                        fragments.append(node.value)
        except SyntaxError:
            pass

        if not fragments:
            try:
                tokens = []
                gen = tokenize.generate_tokens(io.StringIO(dedented).readline)
                while True:
                    try:
                        tok = next(gen)
                        tokens.append(tok)
                    except StopIteration:
                        break
                    except tokenize.TokenError:
                        break
                for tok in tokens:
                    if tok.type in (tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                        val = tok.string
                        if tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
                            if _SQL_RE.search(val):
                                fragments.append(val)
                        else:
                            try:
                                val_eval = ast.literal_eval(val)
                                if isinstance(val_eval, str):
                                    if _SQL_RE.search(val_eval):
                                        fragments.append(val_eval)
                            except Exception:
                                match = re.search(r"['\"]", val)
                                if match:
                                    start_idx = match.start()
                                    quote_char = val[start_idx]
                                    if val[start_idx:].startswith(quote_char * 3):
                                        body = val[start_idx + 3 : -3]
                                    else:
                                        body = val[start_idx + 1 : -1]
                                    if _SQL_RE.search(body):
                                        fragments.append(body)
            except Exception:
                pass

        if not fragments:
            for match in _TRIPLE_STRING_RE.finditer(replace):
                body = match.group("body")
                if _SQL_RE.search(body):
                    fragments.append(body)
            if not fragments:
                string_re = re.compile(r'(?P<quote>[\'"])(?P<body>.*?)(?P=quote)', re.DOTALL)
                for match in string_re.finditer(replace):
                    body = match.group("body")
                    if _SQL_RE.search(body):
                        fragments.append(body)
            if not fragments and _SQL_RE.search(replace):
                fragments.append(replace)
    return tuple(fragments)


def _has_unbalanced_sql_quotes(content: str) -> bool:
    single = double = 0
    escaped = False
    for char in content:
        if char == "\\" and not escaped:
            escaped = True
            continue
        if char == "'" and not escaped:
            single += 1
        elif char == '"' and not escaped:
            double += 1
        escaped = False
    return bool(single % 2 or double % 2)


_VIEW_SCHEMA_CACHE: dict[
    str, tuple[tuple[int, int, int], dict[str, dict[str, set[str]]]]
] = {}


def _collect_view_schemas(
    project_root: Path,
    *,
    target_file: str,
    original_content: str,
    patched_content: str,
) -> dict[str, dict[str, set[str]]]:
    global _VIEW_SCHEMA_CACHE
    view_schemas: dict[str, dict[str, set[str]]] = {}
    for content in (original_content, patched_content):
        view_schemas.update(_view_schemas_from_content(content))

    suffixes = {".sql", ".py"}
    for path in _iter_schema_files(project_root):
        if path.suffix.casefold() not in suffixes:
            continue
        rel = path.relative_to(project_root).as_posix()
        if rel == target_file:
            continue
        path_str = str(path.resolve())
        try:
            stat = path.stat()
            # st_mtime is commonly rounded by filesystems and test fixtures.  A
            # size component prevents a rapid schema edit from reusing the old
            # parsed view definition when the timestamp has not advanced.
            signature = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
        except OSError:
            continue
        cached_entry = _VIEW_SCHEMA_CACHE.get(path_str)
        if cached_entry and cached_entry[0] == signature:
            view_schemas.update(cached_entry[1])
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            file_schemas = _view_schemas_from_content(content)
            _VIEW_SCHEMA_CACHE[path_str] = (signature, file_schemas)
            view_schemas.update(file_schemas)
        except OSError:
            continue
    return view_schemas


def _find_target_test_file(project_root: Path, target_file: str) -> Path | None:
    if not target_file:
        return None
    target_path = Path(target_file)
    name = target_path.name
    if not name.endswith(".py"):
        return None
    test_name = f"test_{name}"
    tests_dir = project_root / "tests"
    if tests_dir.is_dir():
        for p in tests_dir.rglob(test_name):
            if p.is_file():
                return p
    return None


def _iter_schema_files(project_root: Path) -> tuple[Path, ...]:
    ignored = {".git", ".venv", "venv", "__pycache__", "node_modules"}
    paths: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored for part in path.parts):
            continue
        if path.suffix.casefold() in {".sql", ".py"}:
            paths.append(path)
    return tuple(paths[:500])


def _view_schemas_from_content(content: str) -> dict[str, dict[str, set[str]]]:
    views: dict[str, dict[str, set[str]]] = {}
    pattern = re.compile(
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?[`\"\[]?(?P<name>[\w$.]+)[`\"\]]?"
        r".*?\bAS\b(?P<body>.*?)(?:;|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(content):
        name = match.group("name").split(".")[-1].strip("`\"[]").casefold()
        body = match.group("body")
        fields = _select_fields(body)
        tables = _select_tables(body)
        if fields or tables:
            views[name] = {"fields": fields, "tables": tables}
    return views


def _select_statements(content: str) -> tuple[str, ...]:
    statements: list[str] = []
    pattern = re.compile(
        r"\bSELECT\b.*?\bFROM\b.*?(?=(?:;|\n\s*(?:SELECT|CREATE|UPDATE|INSERT)\b|\Z))",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(content):
        statement = match.group(0).strip()
        if statement:
            statements.append(statement)
    return tuple(statements)


def _view_source(
    statement: str,
    view_schemas: dict[str, dict[str, set[str]]],
) -> tuple[str, str] | None:
    match = re.search(
        r"\bFROM\s+[`\"\[]?(?P<name>[\w$.]+)[`\"\]]?"
        r"(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*))?",
        statement,
        re.IGNORECASE,
    )
    if match is None:
        return None
    view_name = match.group("name").split(".")[-1].strip("`\"[]")
    alias = match.group("alias") or view_name
    if view_name.casefold() not in view_schemas and not _VIEW_RE.search(view_name):
        return None
    return view_name, alias


def _select_tables(content: str) -> set[str]:
    tables = {
        raw.split(".")[-1].strip("`\"[]").casefold()
        for raw in re.findall(
            r"\b(?:from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)",
            content,
            re.IGNORECASE,
        )
    }
    return {table for table in tables if table and table not in _SQL_RESERVED_ALIASES}


def _string_set(value: object) -> set[str]:
    if not isinstance(value, set | frozenset | tuple | list):
        return set()
    return {str(item).casefold() for item in value if str(item)}


def _select_source_fields(statement: str, alias: str) -> set[str]:
    select_match = re.search(r"\bselect\b(?P<fields>.*?)\bfrom\b", statement, re.I | re.S)
    if select_match is None:
        return set()
    alias_low = alias.casefold()
    fields: set[str] = set()
    for raw in _split_sql_csv(select_match.group("fields")):
        expr = raw.strip()
        if not expr:
            continue
        for owner, field in re.findall(
            r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
            expr,
        ):
            if owner.casefold() == alias_low:
                fields.add(field.casefold())
        if "." in expr or "(" in expr or "*" in expr:
            continue
        source = re.split(r"\s+\bAS\b\s+", expr, flags=re.IGNORECASE)[0]
        source = source.strip().split()[0] if source.strip() else ""
        source = re.sub(r"[^A-Za-z0-9_]", "", source)
        if source and not source.isdigit():
            fields.add(source.casefold())
    return fields


def _split_sql_csv(value: str) -> list[str]:
    items: list[str] = []
    current: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    for char in value:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            elif char == "," and depth == 0:
                items.append("".join(current))
                current = []
                continue
        current.append(char)
    if current:
        items.append("".join(current))
    return items


def _select_fields(content: str) -> set[str]:
    match = re.search(r"\bselect\b(?P<fields>.*?)\bfrom\b", content, re.IGNORECASE | re.DOTALL)
    if match is None:
        return set()
    fields: set[str] = set()
    for raw in _split_sql_csv(match.group("fields")):
        expression = raw.strip()
        if not expression or expression == "*":
            continue
        # A view exposes the alias when one is present; otherwise it exposes
        # the source column.  Splitting on SQL commas is nesting-aware, unlike
        # str.split(','), so function arguments do not become phantom fields.
        alias_match = re.search(
            r"\s+(?:AS\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*$",
            expression,
            re.IGNORECASE,
        )
        if alias_match:
            token = alias_match.group(1)
        else:
            source = expression.rsplit(".", 1)[-1]
            token = re.sub(r"[^a-zA-Z0-9_]", "", source)
        if token:
            fields.add(token.casefold())
    return fields


def _sql_alias_safety(content: str) -> dict[str, object]:
    aliases = {
        alias.casefold()
        for alias in re.findall(
            r"\b(?:from|join)\s+[`\"\[]?[\w.]+[`\"\]]?(?:\s+(?:as\s+)?)"
            r"([A-Za-z_][A-Za-z0-9_]*)\b",
            content,
            re.IGNORECASE,
        )
        if alias.casefold() not in _SQL_RESERVED_ALIASES
    }
    table_names = {
        name.split(".")[-1].strip("`\"[]").casefold()
        for name in re.findall(
            r"\b(?:from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)",
            content,
            re.IGNORECASE,
        )
    }
    used_aliases = {
        alias.casefold()
        for alias in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", content)
    }
    used_aliases -= table_names
    used_aliases -= _SQL_RESERVED_ALIASES
    if not used_aliases:
        return {"checked": False, "pass": True, "aliases": sorted(aliases), "dead_aliases": []}
    dead_aliases = sorted(used_aliases - aliases)
    return {
        "checked": True,
        "pass": not dead_aliases,
        "aliases": sorted(aliases),
        "used_aliases": sorted(used_aliases),
        "dead_aliases": dead_aliases,
    }


_SQL_RESERVED_ALIASES = {
    "select",
    "from",
    "join",
    "where",
    "and",
    "or",
    "on",
    "as",
    "order",
    "group",
    "by",
    "limit",
    "offset",
}


def _fuse_validation(
    ast_result: dict[str, object],
    semantic_result: dict[str, object],
    execution_result: dict[str, object],
) -> tuple[ValidationDecision, float]:
    ast_pass = bool(ast_result.get("pass"))
    schema_pass = bool(semantic_result.get("pass"))
    execution_pass = bool(execution_result.get("pass"))
    score = round(
        (0.4 if ast_pass else 0.0)
        + (0.4 if schema_pass else 0.0)
        + (0.2 if execution_pass else 0.0),
        4,
    )
    if not ast_pass:
        return "rollback", score
    if not schema_pass:
        return "rollback", score
    return "commit", score


def _validation_error(
    ast_result: dict[str, object],
    semantic_result: dict[str, object],
    execution_result: dict[str, object],
    decision: ValidationDecision,
) -> str:
    if decision == "commit":
        return ""
    if not ast_result.get("pass"):
        raw_issues = ast_result.get("issues", [])
        issues = raw_issues if isinstance(raw_issues, list) else []
        return "AST validation failed: " + ", ".join(
            str(item) for item in issues
        )
    if not semantic_result.get("pass"):
        details = semantic_result.get("details", {})
        schema = details.get("schema", {})
        schema_issues = schema.get("issues", [])
        issues = schema_issues if isinstance(schema_issues, list) else []
        if issues:
            message = "Schema validation failed: " + ", ".join(
                str(item) for item in issues
            )
            missing_fields = schema.get("missing_fields", [])
            if isinstance(missing_fields, list) and missing_fields:
                message += "; missing_fields=" + ",".join(
                    str(item) for item in missing_fields
                )
            return message
        return "Schema validation failed: " + json.dumps(
            details,
            ensure_ascii=False,
            sort_keys=True,
        )
    if not execution_result.get("pass"):
        err = execution_result.get("error", "test suite failed")
        return f"Execution validation failed: {err}"
    return "Validation failed"


def _execution_status(output: str, returncode: int) -> str:
    lowered = output.casefold()
    if "no tests collected" in lowered or "collected 0 items" in lowered:
        return "NO_TESTS"
    if "no tests ran" in lowered or returncode == 5:
        return "NO_TESTS"
    if "modulenotfounderror" in lowered or "importerror" in lowered:
        return "INFRA_ERROR"
    if "permission denied" in lowered or "command not found" in lowered:
        return "INFRA_ERROR"
    return "FAIL"
