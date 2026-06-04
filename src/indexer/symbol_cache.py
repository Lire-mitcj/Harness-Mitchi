from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path

from src.indexer.ctags import CtagsIndexResult, CtagsSymbol
from src.indexer.ctags import parse_import_modules
from src.indexer.parser import CodeParser
from src.indexer.scanner import EXTENSION_LANGUAGE_MAP

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    path TEXT PRIMARY KEY,
    hash TEXT NOT NULL,
    language TEXT,
    last_indexed REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    signature TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS symbol_refs (
    src TEXT NOT NULL,
    dst TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
CREATE INDEX IF NOT EXISTS idx_refs_src ON symbol_refs(src);
"""


class SymbolCache:
    """Sync SQLite symbol cache — single persistence layer for RepoMap."""

    def __init__(self, db_path: Path, project_root: Path) -> None:
        self.db_path = db_path
        self.project_root = project_root.resolve()
        self._parser = CodeParser()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def file_hash(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        except OSError:
            return ""

    def is_empty(self) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()
            return int(row[0]) == 0

    def replace_all(self, indexed: CtagsIndexResult, *, source: str = "parser") -> None:
        """Replace cache from a full-project index."""
        now = time.time()
        with self._connect() as conn:
            conn.execute("DELETE FROM symbol_refs")
            conn.execute("DELETE FROM symbols")
            conn.execute("DELETE FROM files")
            files_seen: set[str] = set()
            for sym in indexed.symbols:
                if sym.file_path not in files_seen:
                    files_seen.add(sym.file_path)
                    abs_path = self.project_root / sym.file_path
                    lang = EXTENSION_LANGUAGE_MAP.get(abs_path.suffix.lower())
                    conn.execute(
                        "INSERT INTO files (path, hash, language, last_indexed) VALUES (?, ?, ?, ?)",
                        (
                            sym.file_path,
                            self.file_hash(abs_path) if abs_path.is_file() else "",
                            lang,
                            now,
                        ),
                    )
                conn.execute(
                    "INSERT INTO symbols "
                    "(file_path, name, kind, start_line, end_line, signature) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        sym.file_path,
                        sym.name,
                        sym.kind,
                        sym.start_line,
                        sym.end_line,
                        sym.signature,
                    ),
                )
            for src, dst in indexed.references:
                conn.execute(
                    "INSERT INTO symbol_refs (src, dst) VALUES (?, ?)",
                    (src, dst),
                )
            conn.commit()

    def reindex_paths(self, rel_paths: set[str]) -> None:
        """Re-parse changed files and update cache entries."""
        if not rel_paths:
            return
        now = time.time()
        with self._connect() as conn:
            for rel in sorted(rel_paths):
                norm = rel.replace("\\", "/").lstrip("./")
                abs_path = (self.project_root / norm).resolve()
                try:
                    abs_path.relative_to(self.project_root)
                except ValueError:
                    continue
                if not abs_path.is_file():
                    conn.execute("DELETE FROM symbols WHERE file_path = ?", (norm,))
                    conn.execute("DELETE FROM files WHERE path = ?", (norm,))
                    conn.execute("DELETE FROM symbol_refs WHERE src = ? OR src LIKE ?", (norm, f"{norm}:%"))
                    continue

                result = self._parser.parse_file(abs_path)
                fh = self.file_hash(abs_path)
                lang = EXTENSION_LANGUAGE_MAP.get(abs_path.suffix.lower())
                conn.execute("DELETE FROM symbols WHERE file_path = ?", (norm,))
                conn.execute("DELETE FROM symbol_refs WHERE src = ? OR src LIKE ?", (norm, f"{norm}:%"))
                conn.execute(
                    "INSERT OR REPLACE INTO files (path, hash, language, last_indexed) VALUES (?, ?, ?, ?)",
                    (norm, fh, lang, now),
                )
                for sym in result.all_symbols:
                    conn.execute(
                        "INSERT INTO symbols "
                        "(file_path, name, kind, start_line, end_line, signature) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (norm, sym.name, sym.kind, sym.start_line, sym.end_line, sym.signature),
                    )
                for imp in result.imports:
                    for mod in parse_import_modules(imp):
                        conn.execute(
                            "INSERT INTO symbol_refs (src, dst) VALUES (?, ?)",
                            (norm, mod),
                        )
                for cls in result.classes:
                    for base in _class_bases(cls.signature):
                        conn.execute(
                            "INSERT INTO symbol_refs (src, dst) VALUES (?, ?)",
                            (f"{norm}:{cls.name}", base),
                        )
            conn.commit()

    def load_index(self) -> CtagsIndexResult | None:
        with self._connect() as conn:
            sym_rows = conn.execute(
                "SELECT file_path, name, kind, start_line, end_line, signature FROM symbols"
            ).fetchall()
            if not sym_rows:
                return None
            symbols = [
                CtagsSymbol(
                    file_path=r["file_path"],
                    name=r["name"],
                    kind=r["kind"],
                    start_line=int(r["start_line"]),
                    end_line=int(r["end_line"]),
                    signature=str(r["signature"] or ""),
                )
                for r in sym_rows
            ]
            ref_rows = conn.execute("SELECT src, dst FROM symbol_refs").fetchall()
            references = [(str(r["src"]), str(r["dst"])) for r in ref_rows]
            return CtagsIndexResult(symbols=symbols, references=references, source="cache")

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            files = int(conn.execute("SELECT COUNT(*) FROM files").fetchone()[0])
            symbols = int(conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0])
            return {"files": files, "symbols": symbols}


def _class_bases(signature: str) -> list[str]:
    if "(" not in signature:
        return []
    inner = signature.split("(", 1)[1].split(")", 1)[0]
    return [b.strip() for b in inner.split(",") if b.strip() and b.strip() != "object"]
