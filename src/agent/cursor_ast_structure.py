from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent.cursor_repo_map_lookup import CandidateSymbol


@dataclass(frozen=True, slots=True)
class AstNode:
    symbol: str
    file: str
    signature: str
    calls: tuple[str, ...]
    lines: tuple[int, int]
    code_slice: str
    source_symbol: Any


@dataclass(frozen=True, slots=True)
class SqlStatementSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    tables_referenced: tuple[str, ...] = ()
    parent_symbol: str = ""
    parent_symbol_id: str = ""

    @property
    def symbol_id(self) -> str:
        return f"{self.file_path}:{self.name}:{self.start_line}"


class CursorAstStructureLayer:
    """Ground candidate symbols to function-level code slices and call edges."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def ground(
        self,
        candidates: tuple[CandidateSymbol, ...],
        *,
        limit: int,
    ) -> tuple[AstNode, ...]:
        nodes: list[AstNode] = []
        seen: set[str] = set()
        for candidate in candidates:
            symbol = candidate.symbol
            symbol_id = _symbol_id(symbol)
            if symbol_id in seen:
                continue
            seen.add(symbol_id)
            node = self._node(symbol)
            if node is not None:
                nodes.append(node)
            if len(nodes) >= limit:
                break
        return tuple(nodes)

    def _node(self, symbol: Any) -> AstNode | None:
        rel = str(symbol.file_path).replace("\\", "/").lstrip("./")
        path = self.project_root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return None
        start = max(1, int(symbol.start_line))
        end = min(len(lines), max(start, int(symbol.end_line)))
        code = "\n".join(lines[start - 1:end])
        if str(getattr(symbol, "kind", "")).startswith(("ddl_", "dml_")):
            calls = tuple(getattr(symbol, "tables_referenced", ()))
        else:
            calls = tuple(dict.fromkeys(
                name
                for name in re.findall(r"(?:^|[^\w])([A-Za-z_][A-Za-z0-9_]*)\s*\(", code)
                if name not in {"if", "for", "while", "return", "len", "str", "int", "print"}
                and name != symbol.name
            ))
        return AstNode(
            symbol=str(symbol.name),
            file=rel,
            signature=str(getattr(symbol, "signature", "")),
            calls=calls,
            lines=(start, end),
            code_slice=code,
            source_symbol=symbol,
        )


def _symbol_id(symbol: Any) -> str:
    return str(
        getattr(
            symbol,
            "symbol_id",
            f"{symbol.file_path}:{symbol.name}:{symbol.start_line}",
        )
    )
