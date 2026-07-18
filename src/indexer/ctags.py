from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from src.indexer.path_globs import ctags_exclude_args, repo_map_path_allowed
from src.indexer.parser import CodeParser, Symbol
from src.indexer.project_stack import detect_project_stack, indexable_extensions_for_stack
from src.indexer.scanner import DEFAULT_IGNORE_DIRS, EXTENSION_LANGUAGE_MAP, ProjectScanner

log = logging.getLogger(__name__)

_INDEXABLE_SUFFIXES = frozenset(EXTENSION_LANGUAGE_MAP.keys())
_CTAGS_EXTRA_EXCLUDES = (
    "*.json",
    "*.md",
    "*.lock",
    "*.svg",
    "*.map",
    "docs",
    "testdata",
)


@dataclass
class CtagsSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str = ""


@dataclass
class CtagsIndexResult:
    symbols: list[CtagsSymbol] = field(default_factory=list)
    references: list[tuple[str, str]] = field(default_factory=list)
    source: str = "parser"


def index_project(
    project_root: Path,
    *,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
) -> CtagsIndexResult:
    """Index symbols via universal-ctags JSON, else regex parser fallback."""
    root = project_root.resolve()
    stack = detect_project_stack(root)
    ctags_bin = shutil.which("ctags") or shutil.which("universal-ctags")
    if ctags_bin:
        try:
            result = _run_ctags_json(
                ctags_bin,
                root,
                exclude_globs=exclude_globs,
            )
            if result.symbols:
                result.source = "ctags"
                result = _filter_index_for_stack(
                    result,
                    stack,
                    include_globs=include_globs,
                    exclude_globs=exclude_globs,
                )
                return result
        except Exception as exc:
            log.debug("ctags indexing failed, falling back to parser: %s", exc)
    result = _index_with_parser(
        root,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )
    return _filter_index_for_stack(
        result,
        stack,
        include_globs=include_globs,
        exclude_globs=exclude_globs,
    )


def _filter_index_for_stack(
    result: CtagsIndexResult,
    stack,
    *,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
) -> CtagsIndexResult:
    allowed = indexable_extensions_for_stack(stack)
    if not allowed and not include_globs and not exclude_globs:
        return result
    kept_files = {
        sym.file_path
        for sym in result.symbols
        if (not allowed or Path(sym.file_path).suffix.lower() in allowed)
        and repo_map_path_allowed(
            sym.file_path,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        )
    }
    symbols = [sym for sym in result.symbols if sym.file_path in kept_files]
    references: list[tuple[str, str]] = []
    for src, dst in result.references:
        src_file = src.split(":", 1)[0]
        if src_file in kept_files or dst in kept_files or "." not in Path(dst).name:
            references.append((src, dst))
    return CtagsIndexResult(
        symbols=symbols,
        references=references,
        source=result.source,
    )


def _run_ctags_json(
    ctags_bin: str,
    project_root: Path,
    *,
    exclude_globs: tuple[str, ...] = (),
) -> CtagsIndexResult:
    excludes = [f"--exclude={name}" for name in sorted(DEFAULT_IGNORE_DIRS)]
    for pattern in _CTAGS_EXTRA_EXCLUDES:
        excludes.append(f"--exclude={pattern}")
    excludes.extend(ctags_exclude_args(exclude_globs))
    cmd = [
        ctags_bin,
        "--recurse",
        "--fields=+neS",
        "--extras=+q",
        "--output-format=json",
        *excludes,
        str(project_root),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode not in {0, 1}:
        raise RuntimeError(proc.stderr.strip() or f"ctags exit {proc.returncode}")

    symbols: list[CtagsSymbol] = []
    references: list[tuple[str, str]] = []
    seen: set[tuple[str, str, int]] = set()

    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("_type") != "tag":
            continue
        raw_path = obj.get("path")
        name = obj.get("name")
        if not isinstance(raw_path, str) or not isinstance(name, str):
            continue
        try:
            rel = Path(raw_path).resolve().relative_to(project_root).as_posix()
        except ValueError:
            continue
        line_no = int(obj.get("line") or 1)
        end_line = int(obj.get("end") or line_no)
        kind = str(obj.get("kind") or "symbol")
        signature = str(obj.get("signature") or obj.get("pattern") or "")[:200]
        key = (rel, name, line_no)
        if key in seen:
            continue
        seen.add(key)
        symbols.append(
            CtagsSymbol(
                file_path=rel,
                name=name,
                kind=kind,
                start_line=line_no,
                end_line=end_line,
                signature=signature,
            )
        )
        roles = obj.get("roles")
        if isinstance(roles, str) and "ref" in roles:
            references.append((f"{rel}:{name}", name))

    return CtagsIndexResult(symbols=symbols, references=references)


def _index_with_parser(
    project_root: Path,
    *,
    include_globs: tuple[str, ...] = (),
    exclude_globs: tuple[str, ...] = (),
) -> CtagsIndexResult:
    stack = detect_project_stack(project_root)
    allowed = indexable_extensions_for_stack(stack)
    scanner = ProjectScanner(project_root)
    structure = scanner.scan(max_files=5000)
    parser = CodeParser()
    symbols: list[CtagsSymbol] = []
    references: list[tuple[str, str]] = []

    for path in structure.files:
        if path.suffix.lower() not in _INDEXABLE_SUFFIXES:
            continue
        if allowed and path.suffix.lower() not in allowed:
            continue
        try:
            rel = path.relative_to(project_root).as_posix()
        except ValueError:
            continue
        if not repo_map_path_allowed(
            rel,
            include_globs=include_globs,
            exclude_globs=exclude_globs,
        ):
            continue
        result = parser.parse_file(path)
        for sym in result.all_symbols:
            symbols.append(
                CtagsSymbol(
                    file_path=rel,
                    name=sym.name,
                    kind=sym.kind,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    signature=sym.signature,
                )
            )
        for imp in result.imports:
            for mod in parse_import_modules(imp):
                references.append((rel, mod))
        for cls in result.classes:
            bases = _class_bases(cls.signature)
            for base in bases:
                references.append((f"{rel}:{cls.name}", base))
        symbol_starts = {
            sym.name: sym.start_line
            for sym in result.all_symbols
        }
        for src, dst in result.references:
            src_rel = _relative_reference_src(src, path, rel, symbol_starts)
            references.append((src_rel, dst))

    return CtagsIndexResult(symbols=symbols, references=references, source="parser")


def parse_import_modules(line: str) -> list[str]:
    """Extract imported module paths from a Python import line."""
    text = line.strip()
    modules: list[str] = []
    if text.startswith("from "):
        parts = text.split()
        if len(parts) >= 2:
            mod = parts[1].split(" as ")[0].strip()
            if mod:
                modules.append(mod)
    elif text.startswith("import "):
        chunk = text[len("import ") :].strip()
        for part in chunk.split(","):
            mod = part.strip().split(" as ")[0].strip()
            if mod:
                modules.append(mod)
    return modules


def _import_target(line: str) -> str | None:
    mods = parse_import_modules(line)
    return mods[0] if mods else None


def _class_bases(signature: str) -> list[str]:
    if "(" not in signature:
        return []
    inner = signature.split("(", 1)[1].split(")", 1)[0]
    return [b.strip() for b in inner.split(",") if b.strip() and b.strip() != "object"]


def _relative_reference_src(
    src: str,
    path: Path,
    rel: str,
    symbol_starts: dict[str, int],
) -> str:
    prefix = str(path)
    if src.startswith(prefix + ":"):
        suffix = src[len(prefix) + 1:]
        name = suffix.split(":", 1)[0]
        start = symbol_starts.get(name)
        if start is not None:
            return f"{rel}:{name}:{start}"
        return f"{rel}:{suffix}"
    return src
