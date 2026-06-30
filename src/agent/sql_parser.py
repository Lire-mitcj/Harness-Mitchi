from __future__ import annotations

import re

from src.tools.assembled.ast_structure import SqlStatementSymbol

_DDL_RE = re.compile(
    r"\bCREATE\s+(?:OR\s+REPLACE\s+)?"
    r"(?P<kind>VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?P<name>[`\"\[]?[\w$.]+[`\"\]]?)",
    re.IGNORECASE,
)
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)
_INSERT_RE = re.compile(
    r"\bINSERT\s+INTO\s+(?P<name>[`\"\[]?[\w$.]+[`\"\]]?)",
    re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"\bUPDATE\s+(?P<name>[`\"\[]?[\w$.]+[`\"\]]?)",
    re.IGNORECASE,
)
_TABLE_REF_RE = re.compile(
    r"\b(?:FROM|JOIN|INTO|UPDATE)\s+([`\"\[]?[\w$.]+[`\"\]]?)",
    re.IGNORECASE,
)
_COMMENT_RE = re.compile(r"--.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)


class UniversalSqlParser:
    """Dependency-free SQL statement scanner for DDL/DQL/DML affinity symbols."""

    def parse_text_block(
        self,
        text: str,
        file_path: str,
        base_line: int = 1,
    ) -> tuple[SqlStatementSymbol, ...]:
        symbols: list[SqlStatementSymbol] = []
        for statement, start_line, end_line in _statements(text, base_line):
            cleaned = _strip_comments(statement).strip()
            if not cleaned:
                continue
            symbol = self._parse_statement(cleaned, file_path, start_line, end_line)
            if symbol is not None:
                symbols.append(symbol)
        return tuple(symbols)

    def parse_file(
        self,
        text: str,
        file_path: str,
    ) -> tuple[SqlStatementSymbol, ...]:
        return self.parse_text_block(text, file_path, 1)

    def _parse_statement(
        self,
        statement: str,
        file_path: str,
        start_line: int,
        end_line: int,
    ) -> SqlStatementSymbol | None:
        ddl_match = _DDL_RE.search(statement)
        tables = _tables_referenced(statement)
        signature = " ".join(statement.split())[:240]
        if ddl_match:
            ddl_kind = ddl_match.group("kind").casefold()
            name = _clean_identifier(ddl_match.group("name"))
            kind = "ddl_view" if ddl_kind == "view" else "ddl_table"
            if kind == "ddl_view":
                columns = _view_columns(statement)
                if columns:
                    signature = (
                        f"CREATE VIEW `{name}` AVAILABLE FIELDS: "
                        f"[{', '.join(dict.fromkeys(columns))}]"
                    )
            return SqlStatementSymbol(
                file_path=file_path,
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                tables_referenced=tables,
            )

        insert_match = _INSERT_RE.search(statement)
        if insert_match:
            target = _clean_identifier(insert_match.group("name"))
            return SqlStatementSymbol(
                file_path=file_path,
                name=f"INSERT:{target}",
                kind="dml_insert",
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                tables_referenced=tables or (target,),
            )

        update_match = _UPDATE_RE.search(statement)
        if update_match:
            target = _clean_identifier(update_match.group("name"))
            return SqlStatementSymbol(
                file_path=file_path,
                name=f"UPDATE:{target}",
                kind="dml_update",
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                tables_referenced=tables or (target,),
            )

        if _SELECT_RE.search(statement) and tables:
            primary = tables[0]
            return SqlStatementSymbol(
                file_path=file_path,
                name=f"SELECT:{primary}",
                kind="dml_select",
                start_line=start_line,
                end_line=end_line,
                signature=signature,
                tables_referenced=tables,
            )
        return None


def _statements(text: str, base_line: int) -> tuple[tuple[str, int, int], ...]:
    statements: list[tuple[str, int, int]] = []
    current: list[str] = []
    start_line = base_line
    in_single = False
    in_double = False
    line_no = base_line
    for raw_line in text.splitlines() or [""]:
        if not current and not _strip_comments(raw_line).strip():
            line_no += 1
            continue
        if not current and raw_line.strip():
            start_line = line_no
        current.append(raw_line)
        for index, char in enumerate(raw_line):
            prev = raw_line[index - 1] if index > 0 else ""
            if char == "'" and prev != "\\" and not in_double:
                in_single = not in_single
            elif char == '"' and prev != "\\" and not in_single:
                in_double = not in_double
            elif char == ";" and not in_single and not in_double:
                statement = "\n".join(current)
                statements.append((statement, start_line, line_no))
                current = []
                break
        line_no += 1
    if current and "\n".join(current).strip():
        statements.append(("\n".join(current), start_line, line_no - 1))
    return tuple(statements)


def _tables_referenced(statement: str) -> tuple[str, ...]:
    seen: set[str] = set()
    tables: list[str] = []
    for match in _TABLE_REF_RE.finditer(statement):
        name = _clean_identifier(match.group(1))
        if not name or name.casefold() in seen:
            continue
        seen.add(name.casefold())
        tables.append(name)
    return tuple(tables)


def _view_columns(statement: str) -> tuple[str, ...]:
    select_match = re.search(r"\bSELECT\b", statement, re.IGNORECASE)
    from_match = re.search(r"\bFROM\b", statement, re.IGNORECASE)
    if select_match is None or from_match is None or from_match.start() <= select_match.end():
        return ()
    select_part = statement[select_match.end():from_match.start()]
    columns: list[str] = []
    for chunk in select_part.split(","):
        chunk_str = chunk.strip()
        if not chunk_str:
            continue
        alias_match = re.search(r"\bAS\s+([A-Za-z0-9_]+)\b", chunk_str, re.IGNORECASE)
        if alias_match:
            raw_col = alias_match.group(1).strip()
        else:
            raw_col = chunk_str.split(".")[-1].strip()
        clean_col = re.sub(r"[^a-zA-Z0-9_]", "", raw_col)
        if clean_col and not clean_col.isdigit():
            columns.append(clean_col)
    return tuple(columns)


def _clean_identifier(value: str) -> str:
    return value.strip().strip("`\"[]").split(".")[-1]


def _strip_comments(statement: str) -> str:
    return _COMMENT_RE.sub(" ", statement)
