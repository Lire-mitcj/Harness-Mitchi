from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from src.indexer.ctags import CtagsIndexResult, CtagsSymbol

MAX_IMPORT_TARGETS = 3
MAX_NAME_TARGETS = 2


def build_module_index(file_nodes: dict[str, str]) -> dict[str, list[str]]:
    """Map dotted module paths and stems to file node ids."""
    index: dict[str, set[str]] = defaultdict(set)
    for rel, fid in file_nodes.items():
        norm = rel.replace("\\", "/")
        if norm.endswith("/__init__.py"):
            pkg_parts = norm[: -len("/__init__.py")].split("/")
            if pkg_parts and pkg_parts != [""]:
                index[".".join(pkg_parts)].add(fid)
        if "." in norm:
            stem_path = norm.rsplit(".", 1)[0]
        else:
            stem_path = norm
        parts = [p for p in stem_path.split("/") if p]
        if not parts:
            continue
        index[parts[-1]].add(fid)
        for i in range(len(parts)):
            index[".".join(parts[i:])].add(fid)
            index[".".join(parts[: i + 1])].add(fid)
    return {k: sorted(v) for k, v in index.items()}


def resolve_import_targets(
    module: str,
    *,
    module_index: dict[str, list[str]],
    file_nodes: dict[str, str],
    max_targets: int = MAX_IMPORT_TARGETS,
) -> list[str]:
    mod = module.strip()
    if not mod:
        return []
    if mod in module_index:
        return module_index[mod][:max_targets]

    py_rel = mod.replace(".", "/") + ".py"
    if py_rel in file_nodes:
        return [file_nodes[py_rel]]

    init_rel = mod.replace(".", "/") + "/__init__.py"
    if init_rel in file_nodes:
        return [file_nodes[init_rel]]

    stem = mod.split(".")[-1]
    if stem in module_index:
        return module_index[stem][:max_targets]
    return []


def symbol_targets_by_name(
    name: str,
    *,
    name_to_ids: dict[str, list[str]],
    src_file: str | None = None,
    prefer_files: set[str] | None = None,
    max_targets: int = MAX_NAME_TARGETS,
) -> list[str]:
    candidates = name_to_ids.get(name, [])
    if not candidates:
        return []
    if src_file:
        same = [c for c in candidates if c.startswith(f"{src_file}:")]
        if same:
            return same[:1]
    if prefer_files:
        preferred = [
            c for c in candidates if c.split(":", 1)[0] in prefer_files
        ]
        if preferred:
            return preferred[:max_targets]
    return candidates[:max_targets]


def build_reference_edges(
    indexed: CtagsIndexResult,
    *,
    file_nodes: dict[str, str],
    symbol_nodes: dict[tuple[str, str, int], str],
    name_to_ids: dict[str, list[str]],
) -> list[tuple[str, str]]:
    module_index = build_module_index(file_nodes)
    file_imports: dict[str, set[str]] = defaultdict(set)
    edges: list[tuple[str, str]] = []

    for src, dst in indexed.references:
        if src in file_nodes and dst in file_nodes:
            edges.append((file_nodes[src], file_nodes[dst]))
            continue
        if src in file_nodes:
            targets = resolve_import_targets(
                dst, module_index=module_index, file_nodes=file_nodes
            )
            for tgt in targets:
                if tgt != file_nodes[src]:
                    edges.append((file_nodes[src], tgt))
                    file_imports[src].add(tgt.replace("file:", "", 1))
            continue
        if ":" in src and _looks_like_symbol_id(src):
            src_file = src.split(":", 1)[0]
            prefer = file_imports.get(src_file, set())
            for target in symbol_targets_by_name(
                dst,
                name_to_ids=name_to_ids,
                src_file=src_file,
                prefer_files=prefer or None,
            ):
                if target != src:
                    edges.append((src, target))

    for sym in indexed.symbols:
        sid = symbol_nodes[(sym.file_path, sym.name, sym.start_line)]
        for base in _bases_from_signature(sym):
            for target in symbol_targets_by_name(
                base,
                name_to_ids=name_to_ids,
                src_file=sym.file_path,
            ):
                if target != sid:
                    edges.append((sid, target))

    return edges


def _looks_like_symbol_id(value: str) -> bool:
    parts = value.split(":")
    return len(parts) >= 3 and parts[-1].isdigit()


def _bases_from_signature(sym: CtagsSymbol) -> list[str]:
    if sym.kind not in {"class", "c"}:
        return []
    sig = sym.signature
    if "(" not in sig:
        return []
    inner = sig.split("(", 1)[1].split(")", 1)[0]
    return [b.strip() for b in inner.split(",") if b.strip()]
