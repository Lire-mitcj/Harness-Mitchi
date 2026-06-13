from __future__ import annotations

import re
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp


@dataclass(frozen=True)
class SqlRange:
    file: str
    start_line: int
    end_line: int

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


@dataclass(frozen=True)
class PythonSqlLiteral:
    sql: str
    start_offset: int
    end_offset: int
    quote: str
    variable: str = ""


@dataclass(frozen=True)
class ColumnSource:
    name: str
    source_table: str = ""
    source_alias: str = ""
    source_column: str = ""
    expression: str = ""
    kind: str = "column"

    @property
    def source_key(self) -> str:
        if self.source_table and self.source_column:
            return f"{self.source_table.lower()}.{self.source_column.lower()}"
        if self.source_column:
            return self.source_column.lower()
        return ""

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "source_table": self.source_table,
            "source_alias": self.source_alias,
            "source_column": self.source_column,
            "expression": self.expression,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class SelectItem:
    output_name: str
    expression: str
    source_table: str = ""
    source_alias: str = ""
    source_column: str = ""
    kind: str = "column"

    def to_dict(self) -> dict[str, str]:
        return {
            "output_name": self.output_name,
            "expression": self.expression,
            "source_table": self.source_table,
            "source_alias": self.source_alias,
            "source_column": self.source_column,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class JoinModel:
    table: str
    alias: str = ""
    kind: str = ""
    on: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "table": self.table,
            "alias": self.alias,
            "kind": self.kind,
            "on": self.on,
        }


@dataclass(frozen=True)
class QueryModel:
    sql: str
    selects: list[SelectItem] = field(default_factory=list)
    from_table: str = ""
    from_alias: str = ""
    joins: list[JoinModel] = field(default_factory=list)
    where: str = ""
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    aliases: dict[str, str] = field(default_factory=dict)

    @property
    def source_tables(self) -> list[str]:
        tables = [self.from_table, *(join.table for join in self.joins)]
        return list(dict.fromkeys(t for t in tables if t))

    def to_dict(self) -> dict[str, object]:
        return {
            "selects": [item.to_dict() for item in self.selects],
            "from": {"table": self.from_table, "alias": self.from_alias},
            "joins": [join.to_dict() for join in self.joins],
            "where": self.where,
            "group_by": list(self.group_by),
            "order_by": list(self.order_by),
            "source_tables": self.source_tables,
            "aliases": dict(self.aliases),
        }


@dataclass(frozen=True)
class TableModel:
    name: str
    columns: list[str]
    sql_range: SqlRange | None = None
    sql: str = ""

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "columns": list(self.columns),
        }
        if self.sql_range is not None:
            data["sql_range"] = self.sql_range.to_dict()
        return data


@dataclass(frozen=True)
class ViewModel:
    name: str
    columns: list[str]
    column_sources: list[ColumnSource]
    source_tables: list[str]
    aliases: dict[str, str]
    sql_range: SqlRange | None = None
    sql: str = ""

    @property
    def source_to_view_column(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for source in self.column_sources:
            if source.source_key:
                mapping[source.source_key] = source.name
            if source.source_alias and source.source_column:
                mapping[f"{source.source_alias.lower()}.{source.source_column.lower()}"] = source.name
            if source.source_column:
                mapping.setdefault(source.source_column.lower(), source.name)
        return mapping

    @property
    def view_column_to_source(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for source in self.column_sources:
            if source.source_key:
                mapping[source.name.lower()] = source.source_key
        return mapping

    @property
    def column_defaults(self) -> dict[str, str]:
        defaults: dict[str, str] = {}
        for source in self.column_sources:
            if source.kind != "literal" or not source.name:
                continue
            default_sql = _alias_inner_sql(source.expression)
            if default_sql:
                defaults[source.name] = default_sql
        return defaults

    def to_dependency_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "role": "replacement_source",
            "kind": "database_view",
            "name": self.name,
            "columns": list(self.columns),
            "column_sources": [source.to_dict() for source in self.column_sources],
            "source_to_view_column": self.source_to_view_column,
            "view_column_to_source": self.view_column_to_source,
            "column_defaults": self.column_defaults,
            "replaces_objects": list(self.source_tables),
            "aliases": dict(self.aliases),
            "confidence": 0.95,
        }
        if self.sql_range is not None:
            data["sql_range"] = self.sql_range.to_dict()
            data["evidence"] = [self.sql_range.file]
        return data


class ProjectSqlAstCache:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._views: dict[str, ViewModel] | None = None
        self._tables: dict[str, TableModel] | None = None

    def views(self) -> dict[str, ViewModel]:
        if self._views is None:
            self._views = scan_project_views(self.project_root)
        return self._views

    def view_names(self) -> list[str]:
        names: set[str] = set()
        for view in self.views().values():
            full = view.name.lower()
            if full:
                names.add(full)
                names.add(full.split(".")[-1])
        return sorted(names)

    def get_view(self, name: str) -> ViewModel | None:
        key = _clean_name(name)
        if not key:
            return None
        views = self.views()
        return views.get(key) or views.get(key.split(".")[-1])

    def tables(self) -> dict[str, TableModel]:
        if self._tables is None:
            self._tables = scan_project_tables(self.project_root)
        return self._tables

    def get_table(self, name: str) -> TableModel | None:
        key = _clean_name(name)
        if not key:
            return None
        tables = self.tables()
        return tables.get(key) or tables.get(key.split(".")[-1])


def scan_project_views(project_root: Path) -> dict[str, ViewModel]:
    views: dict[str, ViewModel] = {}
    for path in project_root.rglob("*"):
        if path.suffix.lower() not in {".sql", ".py"} or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(project_root)).replace("\\", "/")
        for name, sql, start_line, end_line in extract_view_sql_blocks(content):
            model = parse_view_sql(
                name,
                sql,
                sql_range=SqlRange(rel, start_line, end_line),
            )
            if model is None:
                continue
            for key in {_clean_name(model.name), _clean_name(model.name).split(".")[-1]}:
                if key:
                    views[key] = model
    return views


def scan_project_tables(project_root: Path) -> dict[str, TableModel]:
    tables: dict[str, TableModel] = {}
    for path in project_root.rglob("*"):
        if path.suffix.lower() not in {".sql", ".py"} or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = str(path.relative_to(project_root)).replace("\\", "/")
        for name, sql, start_line, end_line in extract_table_sql_blocks(content):
            model = parse_table_sql(
                name,
                sql,
                sql_range=SqlRange(rel, start_line, end_line),
            )
            if model is None:
                continue
            for key in {_clean_name(model.name), _clean_name(model.name).split(".")[-1]}:
                if key:
                    tables[key] = model
    return tables


def extract_view_sql_blocks(content: str) -> list[tuple[str, str, int, int]]:
    pattern = re.compile(
        r"\bCREATE\s+[^;]*?\bVIEW\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[^\s]+)\s+AS\s+"
        r"(?P<select>SELECT\b[\s\S]*?)(?:;|$)",
        re.IGNORECASE,
    )
    blocks: list[tuple[str, str, int, int]] = []
    for match in pattern.finditer(content):
        name = re.sub(r"[\"'`\[\]]", "", match.group("name")).strip()
        sql = match.group("select").strip()
        start_line = content.count("\n", 0, match.start()) + 1
        end_line = content.count("\n", 0, match.end()) + 1
        if name and sql:
            blocks.append((name, sql, start_line, end_line))
    return blocks


def extract_table_sql_blocks(content: str) -> list[tuple[str, str, int, int]]:
    pattern = re.compile(
        r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(?P<name>[^\s(]+)\s*"
        r"(?P<body>\([\s\S]*?\))\s*(?:;|$)",
        re.IGNORECASE,
    )
    blocks: list[tuple[str, str, int, int]] = []
    for match in pattern.finditer(content):
        name = re.sub(r"[\"'`\[\]]", "", match.group("name")).strip()
        sql = f"CREATE TABLE {name} {match.group('body')}"
        start_line = content.count("\n", 0, match.start()) + 1
        end_line = content.count("\n", 0, match.end()) + 1
        if name and sql:
            blocks.append((name, sql, start_line, end_line))
    return blocks


def parse_view_sql(
    name: str,
    sql: str,
    *,
    sql_range: SqlRange | None = None,
) -> ViewModel | None:
    query = parse_query(sql)
    if query is None:
        return None
    columns = [item.output_name for item in query.selects if item.output_name]
    column_sources = [
        ColumnSource(
            name=item.output_name,
            source_table=item.source_table,
            source_alias=item.source_alias,
            source_column=item.source_column,
            expression=item.expression,
            kind=item.kind,
        )
        for item in query.selects
        if item.output_name
    ]
    return ViewModel(
        name=name,
        columns=list(dict.fromkeys(columns)),
        column_sources=column_sources,
        source_tables=query.source_tables,
        aliases=query.aliases,
        sql_range=sql_range,
        sql=sql,
    )


def parse_table_sql(
    name: str,
    sql: str,
    *,
    sql_range: SqlRange | None = None,
) -> TableModel | None:
    try:
        expression = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        try:
            expression = sqlglot.parse_one(sql)
        except Exception:
            return None
    columns = [
        column.name
        for column in expression.find_all(exp.ColumnDef)
        if column.name
    ]
    return TableModel(
        name=name,
        columns=list(dict.fromkeys(columns)),
        sql_range=sql_range,
        sql=sql,
    )


def parse_query(sql: str) -> QueryModel | None:
    try:
        expression = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        try:
            expression = sqlglot.parse_one(sql)
        except Exception:
            return None
    if not isinstance(expression, exp.Select):
        select = expression.find(exp.Select)
        if select is None:
            return None
        expression = select

    aliases = _table_aliases(expression)
    from_table, from_alias = _from_table(expression)
    joins = _joins(expression)
    selects = [
        _select_item(item, aliases)
        for item in expression.expressions
    ]
    where_expr = expression.args.get("where")
    group_expr = expression.args.get("group")
    order_expr = expression.args.get("order")
    return QueryModel(
        sql=sql,
        selects=selects,
        from_table=from_table,
        from_alias=from_alias,
        joins=joins,
        where=where_expr.sql(dialect="mysql") if where_expr is not None else "",
        group_by=[
            item.sql(dialect="mysql")
            for item in (group_expr.expressions if group_expr is not None else [])
        ],
        order_by=[
            item.sql(dialect="mysql")
            for item in (order_expr.expressions if order_expr is not None else [])
        ],
        aliases=aliases,
    )


def extract_sql_strings_from_python(code: str) -> list[str]:
    return [literal.sql for literal in extract_sql_literals_from_python(code)]


def extract_sql_literals_from_python(code: str) -> list[PythonSqlLiteral]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _extract_sql_literals_with_regex_fallback(code)

    line_offsets = _line_offsets(code)
    parent_map = _parent_map(tree)
    literals: list[PythonSqlLiteral] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if parse_query(node.value) is None:
            continue
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        segment = code[start:end]
        quote = _detect_python_quote(segment)
        if quote:
            literals.append(
                PythonSqlLiteral(
                    sql=node.value,
                    start_offset=start,
                    end_offset=end,
                    quote=quote,
                    variable=_assignment_name_for_node(node, parent_map),
                )
            )
    return literals


def parse_python_sql_queries(code: str) -> list[QueryModel]:
    models: list[QueryModel] = []
    for sql in extract_sql_strings_from_python(code):
        model = parse_query(sql)
        if model is not None:
            models.append(model)
    return models


def _extract_sql_literals_with_regex_fallback(code: str) -> list[PythonSqlLiteral]:
    pattern = re.compile(
        r'(?P<prefix>[frFR]*)(?P<quote>"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\')',
    )
    literals: list[PythonSqlLiteral] = []
    for match in pattern.finditer(code):
        raw_quote = match.group("quote")
        if raw_quote.startswith(('"""', "'''")):
            inner = raw_quote[3:-3]
            quote = raw_quote[:3]
        else:
            inner = raw_quote[1:-1]
            quote = raw_quote[0]
        if parse_query(inner) is None:
            continue
        literals.append(
            PythonSqlLiteral(
                sql=inner,
                start_offset=match.start("quote"),
                end_offset=match.end("quote"),
                quote=quote,
            )
        )
    return literals


def _line_offsets(code: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", code):
        offsets.append(match.end())
    return offsets


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _assignment_name_for_node(
    node: ast.AST,
    parent_map: dict[ast.AST, ast.AST],
) -> str:
    current: ast.AST | None = node
    while current is not None:
        parent = parent_map.get(current)
        if isinstance(parent, ast.Assign):
            return _target_name(parent.targets[0]) if parent.targets else ""
        if isinstance(parent, ast.AnnAssign):
            return _target_name(parent.target)
        current = parent
    return ""


def _target_name(target: ast.AST) -> str:
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return ""


def _detect_python_quote(segment: str) -> str:
    text = segment.lstrip("fFrRuUbB")
    for quote in ('"""', "'''", '"', "'"):
        if text.startswith(quote):
            return quote
    return ""


def source_to_view_column_lookup(
    column_sources: list[dict[str, object]],
    explicit_mapping: dict[str, object] | None = None,
) -> dict[str, str]:
    lookup: dict[str, str] = {
        str(key).lower(): str(value)
        for key, value in (explicit_mapping or {}).items()
        if str(key).strip() and str(value).strip()
    }
    for item in column_sources:
        view_column = str(item.get("name") or "").strip()
        source_column = str(item.get("source_column") or "").strip()
        source_table = str(item.get("source_table") or "").strip()
        source_alias = str(item.get("source_alias") or "").strip()
        if not view_column or not source_column:
            continue
        for table in (source_table, source_alias):
            table_key = _clean_name(table)
            if table_key:
                lookup[f"{table_key}.{source_column.lower()}"] = view_column
        lookup.setdefault(source_column.lower(), view_column)
    return lookup


def source_lookup_key(
    alias_or_table: str,
    column: str,
    alias_table_map: dict[str, str],
) -> str:
    alias_key = _clean_name(alias_or_table)
    column_key = _clean_name(column)
    table_key = alias_table_map.get(alias_key, alias_key)
    return f"{table_key}.{column_key}"


def rewrite_query_with_view(
    sql: str,
    *,
    target_source: str,
    replaces_objects: list[str],
    target_columns: list[str] | None = None,
    column_defaults: dict[str, str] | None = None,
    column_sources: list[dict[str, object]] | None = None,
    source_to_view_column: dict[str, object] | None = None,
    target_alias: str = "v",
) -> str | None:
    try:
        expression = sqlglot.parse_one(sql, read="mysql")
    except Exception:
        return None
    if not isinstance(expression, exp.Select):
        select = expression.find(exp.Select)
        if select is None:
            return None
        expression = select

    with_expr = expression.args.get("with")
    if with_expr is None:
        with_expr = expression.args.get("with_")
    if with_expr is not None:
        changed = False
        for cte in with_expr.expressions:
            inner = cte.this
            if not isinstance(inner, exp.Select):
                continue
            rewritten_inner = rewrite_query_with_view(
                inner.sql(dialect="mysql"),
                target_source=target_source,
                replaces_objects=replaces_objects,
                target_columns=target_columns,
                column_defaults=column_defaults,
                column_sources=column_sources,
                source_to_view_column=source_to_view_column,
                target_alias=target_alias,
            )
            if rewritten_inner is None:
                continue
            try:
                cte.set("this", sqlglot.parse_one(rewritten_inner, read="mysql"))
            except Exception:
                continue
            changed = True
        if changed:
            return expression.sql(dialect="mysql")

    aliases = _table_aliases(expression)
    replaces_lower = {_clean_name(obj) for obj in replaces_objects if str(obj).strip()}
    if not replaces_lower:
        return None
    legacy_aliases = {
        alias
        for alias, table in aliases.items()
        if table in replaces_lower or alias in replaces_lower
    }
    if not legacy_aliases:
        return None

    source_lookup = source_to_view_column_lookup(
        column_sources or [],
        source_to_view_column,
    )
    defaults = {
        str(key).lower(): str(value)
        for key, value in (column_defaults or {}).items()
        if str(key).strip() and str(value).strip()
    }
    target_col_lookup = {
        str(col).strip().lower(): str(col).strip()
        for col in (target_columns or [])
        if str(col).strip()
    }

    select_items: list[exp.Expression] = []
    for item in expression.expressions:
        output_name = item.alias_or_name
        output_key = output_name.lower() if output_name else ""
        if _is_aggregate_expression(item):
            select_items.append(_rewrite_columns(item.copy(), aliases, legacy_aliases, source_lookup, target_alias))
            continue
        if output_key in target_col_lookup:
            replacement = exp.column(target_col_lookup[output_key], table=target_alias)
            select_items.append(replacement)
            continue
        if output_key in defaults:
            try:
                default_expr = sqlglot.parse_one(defaults[output_key], read="mysql")
            except Exception:
                default_expr = exp.Literal.string(defaults[output_key].strip("'\""))
            select_items.append(exp.alias_(default_expr, output_name, quoted=False))
            continue
        if target_columns:
            return None
        select_items.append(_rewrite_columns(item.copy(), aliases, legacy_aliases, source_lookup, target_alias))

    expression.set("expressions", select_items)
    expression.set("from_", exp.From(this=exp.to_table(target_source).as_(target_alias)))
    kept_joins = []
    for join in expression.args.get("joins") or []:
        table = join.find(exp.Table)
        table_name = _clean_name(table.name) if table is not None else ""
        table_alias = _clean_name(table.alias_or_name) if table is not None else ""
        if table_name in replaces_lower or table_alias in legacy_aliases:
            continue
        kept_joins.append(_rewrite_columns(join.copy(), aliases, legacy_aliases, source_lookup, target_alias))
    expression.set("joins", kept_joins)

    for key in ("where", "group", "order", "having"):
        clause = expression.args.get(key)
        if clause is not None:
            expression.set(
                key,
                _rewrite_columns(clause.copy(), aliases, legacy_aliases, source_lookup, target_alias),
            )
    return expression.sql(dialect="mysql")


def _rewrite_columns(
    expression: exp.Expression,
    aliases: dict[str, str],
    legacy_aliases: set[str],
    source_lookup: dict[str, str],
    target_alias: str,
) -> exp.Expression:
    def transform(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Column):
            return node
        table_alias = _clean_name(node.table)
        if table_alias not in legacy_aliases:
            return node
        table = aliases.get(table_alias, table_alias)
        key = f"{table}.{_clean_name(node.name)}"
        mapped = source_lookup.get(key) or source_lookup.get(_clean_name(node.name)) or node.name
        return exp.column(mapped, table=target_alias)

    return expression.transform(transform)


def _is_aggregate_expression(expression: exp.Expression) -> bool:
    target = expression.this if isinstance(expression, exp.Alias) else expression
    return isinstance(target, exp.AggFunc) or target.find(exp.AggFunc) is not None


def _select_item(item: exp.Expression, aliases: dict[str, str]) -> SelectItem:
    output_name = item.alias_or_name
    expression = item.sql(dialect="mysql")
    source = _single_source_column(item, aliases)
    kind = _expression_kind(item)
    if not output_name:
        output_name = source[2] if source[2] else _clean_name(expression)
    return SelectItem(
        output_name=output_name,
        expression=expression,
        source_table=source[0],
        source_alias=source[1],
        source_column=source[2],
        kind=kind,
    )


def _expression_kind(item: exp.Expression) -> str:
    target = item.this if isinstance(item, exp.Alias) else item
    if isinstance(target, exp.Literal):
        return "literal"
    if isinstance(target, exp.AggFunc) or target.find(exp.AggFunc) is not None:
        return "aggregate"
    columns = list(target.find_all(exp.Column))
    if len(columns) == 1 and isinstance(target, exp.Column):
        return "column"
    if len(columns) == 1 and isinstance(item, exp.Alias):
        return "column"
    if columns:
        return "computed"
    return "literal" if target.find(exp.Literal) is not None else "computed"


def _single_source_column(
    item: exp.Expression,
    aliases: dict[str, str],
) -> tuple[str, str, str]:
    target = item.this if isinstance(item, exp.Alias) else item
    columns = list(target.find_all(exp.Column))
    if len(columns) != 1:
        return "", "", ""
    column = columns[0]
    source_alias = str(column.table or "")
    source_table = aliases.get(source_alias.lower(), source_alias.lower())
    return source_table, source_alias, str(column.name or "")


def _table_aliases(expression: exp.Select) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for table in expression.find_all(exp.Table):
        table_name = _clean_name(table.name)
        alias = _clean_name(table.alias_or_name)
        if table_name:
            aliases[table_name] = table_name
        if alias:
            aliases[alias] = table_name
    return aliases


def _from_table(expression: exp.Select) -> tuple[str, str]:
    from_expr = expression.args.get("from") or expression.args.get("from_")
    if from_expr is None:
        return "", ""
    table = from_expr.find(exp.Table)
    if table is None:
        return "", ""
    return _clean_name(table.name), _clean_name(table.alias_or_name)


def _joins(expression: exp.Select) -> list[JoinModel]:
    joins: list[JoinModel] = []
    for join in expression.args.get("joins") or []:
        table = join.find(exp.Table)
        if table is None:
            continue
        on_expr = join.args.get("on")
        joins.append(
            JoinModel(
                table=_clean_name(table.name),
                alias=_clean_name(table.alias_or_name),
                kind=str(join.args.get("kind") or ""),
                on=on_expr.sql(dialect="mysql") if on_expr is not None else "",
            )
        )
    return joins


def _clean_name(raw: Any) -> str:
    text = str(raw or "").strip()
    text = re.sub(r"[\"'`\[\]]", "", text)
    return text.split(".")[-1].lower()


def _alias_inner_sql(expression: str) -> str:
    parts = re.split(r"\s+AS\s+", expression, maxsplit=1, flags=re.IGNORECASE)
    return parts[0].strip() if parts else expression.strip()
