from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.indexer.language_profiles import LanguageProfile

from src.indexer.language_profiles import profile_for_extension


@dataclass
class Symbol:
    name: str
    kind: str  # "function", "class", "method", "import"
    start_line: int
    end_line: int
    signature: str = ""


@dataclass
class ParseResult:
    path: str
    functions: list[Symbol] = field(default_factory=list)
    classes: list[Symbol] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    references: list[tuple[str, str]] = field(default_factory=list)

    @property
    def all_symbols(self) -> list[Symbol]:
        return self.functions + self.classes


class CodeParser:
    """Extracts symbols from source files.

    Currently uses regex-based parsing as a fallback.
    Tree-sitter integration will replace this for supported languages.
    """

    def parse_file(self, path: Path) -> ParseResult:
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ParseResult(path=str(path))

        suffix = path.suffix.lower()
        if suffix == ".py":
            return self._parse_python(str(path), content)
        if suffix == ".sql":
            return self._parse_sql(str(path), content)
        profile = profile_for_extension(suffix)
        if profile is not None:
            return self._parse_with_profile(str(path), content, profile)
        return self._parse_generic(str(path), content)

    def _parse_with_profile(
        self,
        path: str,
        content: str,
        profile: LanguageProfile,
    ) -> ParseResult:
        result = ParseResult(path=path)
        lines = content.splitlines()
        seen: set[tuple[str, int]] = set()
        for index, line in enumerate(lines, 1):
            for pattern in profile.symbol_extractors:
                match = pattern.search(line)
                if not match:
                    continue
                name = next((g for g in match.groups() if g), None)
                if not name:
                    continue
                key = (name, index)
                if key in seen:
                    break
                seen.add(key)
                kind = "class" if profile.id in {"java", "proto"} and "class" in pattern.pattern else "function"
                if profile.id == "go" and "type" in pattern.pattern:
                    kind = "class"
                result.functions.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        start_line=index,
                        end_line=index,
                        signature=line.strip()[:160],
                    )
                )
                break
            if profile.import_line_re.search(line):
                result.imports.append(line.strip())
        return result

    def _parse_python(self, path: str, content: str) -> ParseResult:
        result = ParseResult(path=path)
        lines = content.splitlines()

        func_re = re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)")
        class_re = re.compile(r"^class\s+(\w+)\s*(?:\(([^)]*)\))?:")
        import_re = re.compile(r"^(?:from\s+\S+\s+)?import\s+(.+)")

        for i, line in enumerate(lines, 1):
            m = import_re.match(line)
            if m:
                result.imports.append(line.strip())
                continue

            m = class_re.match(line)
            if m:
                end = self._find_block_end(lines, i - 1)
                result.classes.append(Symbol(
                    name=m.group(1),
                    kind="class",
                    start_line=i,
                    end_line=end,
                    signature=line.strip(),
                ))
                continue

            m = func_re.match(line)
            if m:
                indent = len(m.group(1))
                kind = "method" if indent > 0 else "function"
                end = self._find_block_end(lines, i - 1)
                result.functions.append(Symbol(
                    name=m.group(2),
                    kind=kind,
                    start_line=i,
                    end_line=end,
                    signature=line.strip(),
                ))

        result.references.extend(_python_call_references(path, lines, result.functions))
        return result

    _SQL_OBJECT_RE = re.compile(
        r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?"
        r"(VIEW|PROCEDURE|FUNCTION|TRIGGER|TABLE)\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?"
        r"([`\"\[]?[\w$]+[`\"\]]?)",
        re.IGNORECASE,
    )
    _SQL_ALTER_RE = re.compile(
        r"^\s*ALTER\s+(VIEW|PROCEDURE|FUNCTION|TRIGGER|TABLE)\s+"
        r"([`\"\[]?[\w$]+[`\"\]]?)",
        re.IGNORECASE,
    )

    def _parse_sql(self, path: str, content: str) -> ParseResult:
        result = ParseResult(path=path)
        seen: set[tuple[str, int]] = set()

        for i, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue

            for pattern, default_kind in ((self._SQL_OBJECT_RE, None), (self._SQL_ALTER_RE, None)):
                m = pattern.match(line)
                if not m:
                    continue
                kind_raw = m.group(1).lower()
                name = m.group(2).strip("`\"[]")
                key = (name, i)
                if key in seen:
                    break
                seen.add(key)
                kind = {
                    "view": "view",
                    "procedure": "procedure",
                    "function": "function",
                    "trigger": "trigger",
                    "table": "table",
                }.get(kind_raw, "sql")
                result.functions.append(
                    Symbol(
                        name=name,
                        kind=kind,
                        start_line=i,
                        end_line=i,
                        signature=stripped[:160],
                    )
                )
                break

        return result

    def _parse_generic(self, path: str, content: str) -> ParseResult:
        result = ParseResult(path=path)
        func_re = re.compile(
            r"(?:(?:export\s+)?(?:async\s+)?function\s+(\w+))"
            r"|(?:(?:pub\s+)?fn\s+(\w+))"
            r"|(?:func\s+(\w+))"
        )
        for i, line in enumerate(content.splitlines(), 1):
            m = func_re.search(line)
            if m:
                name = m.group(1) or m.group(2) or m.group(3)
                if name:
                    result.functions.append(Symbol(
                        name=name, kind="function",
                        start_line=i, end_line=i,
                        signature=line.strip()[:120],
                    ))
        return result

    @staticmethod
    def _find_block_end(lines: list[str], start_idx: int) -> int:
        if start_idx >= len(lines):
            return start_idx + 1
        first_line = lines[start_idx]
        base_indent = len(first_line) - len(first_line.lstrip())
        for i in range(start_idx + 1, min(start_idx + 500, len(lines))):
            line = lines[i]
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            if indent <= base_indent:
                return i
        return len(lines)


_PY_CALL_RE = re.compile(r"(?:^|[^\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_PY_CALL_EXCLUDE = frozenset({
    "assert",
    "bool",
    "dict",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "print",
    "range",
    "set",
    "str",
    "super",
    "tuple",
})


def _python_call_references(
    path: str,
    lines: list[str],
    functions: list[Symbol],
) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for sym in functions:
        body = lines[sym.start_line : sym.end_line]
        src = f"{path}:{sym.name}"
        for line in body:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for match in _PY_CALL_RE.finditer(line):
                name = match.group(1)
                if name in _PY_CALL_EXCLUDE or name == sym.name:
                    continue
                key = (src, name)
                if key in seen:
                    continue
                seen.add(key)
                refs.append(key)
    return refs
