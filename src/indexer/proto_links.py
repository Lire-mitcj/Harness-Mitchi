"""Link ``.proto`` definitions to generated Go/Java/protobuf stubs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.indexer.ctags import CtagsSymbol

_PROTO_SERVICE_RE = re.compile(r"^\s*service\s+(\w+)", re.MULTILINE)
_PROTO_MESSAGE_RE = re.compile(r"^\s*message\s+(\w+)", re.MULTILINE)
_PROTO_ENUM_RE = re.compile(r"^\s*enum\s+(\w+)", re.MULTILINE)
_PROTO_GO_PACKAGE_RE = re.compile(
    r'option\s+go_package\s*=\s*"(?P<path>[^";]+)(?:;(?P<name>[^"]*))?"'
)
_PROTO_JAVA_PACKAGE_RE = re.compile(
    r'option\s+java_package\s*=\s*"(?P<path>[^"]+)"'
)
_MAX_PROTO_FILES = 200
_MAX_GENERATED_PER_PROTO = 6


@dataclass(frozen=True, slots=True)
class ProtoDefinition:
    file_path: str
    name: str
    kind: str
    start_line: int
    go_package: str = ""
    java_package: str = ""


def _rel_posix(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _skip_dir(part: str) -> bool:
    return part in {".git", "node_modules", "venv", ".venv", "target", "build", "vendor"}


def scan_proto_definitions(project_root: Path) -> list[ProtoDefinition]:
    """Scan ``.proto`` files for service/message/enum definitions."""
    root = project_root.resolve()
    out: list[ProtoDefinition] = []
    count = 0
    for path in root.rglob("*.proto"):
        if count >= _MAX_PROTO_FILES:
            break
        if any(_skip_dir(part) for part in path.parts):
            continue
        if not path.is_file():
            continue
        count += 1
        rel = _rel_posix(path, root)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        go_pkg = ""
        java_pkg = ""
        go_match = _PROTO_GO_PACKAGE_RE.search(text)
        if go_match:
            go_pkg = go_match.group("path").strip()
        java_match = _PROTO_JAVA_PACKAGE_RE.search(text)
        if java_match:
            java_pkg = java_match.group("path").strip()
        for kind, pattern in (
            ("service", _PROTO_SERVICE_RE),
            ("message", _PROTO_MESSAGE_RE),
            ("enum", _PROTO_ENUM_RE),
        ):
            for match in pattern.finditer(text):
                line = text[: match.start()].count("\n") + 1
                out.append(
                    ProtoDefinition(
                        file_path=rel,
                        name=match.group(1),
                        kind=kind,
                        start_line=line,
                        go_package=go_pkg,
                        java_package=java_pkg,
                    )
                )
    return out


def proto_ctags_symbols(project_root: Path) -> list[CtagsSymbol]:
    """Synthetic ctags symbols for proto definitions missing from the index."""
    return [
        CtagsSymbol(
            file_path=item.file_path,
            name=item.name,
            kind=f"proto_{item.kind}",
            start_line=item.start_line,
            end_line=item.start_line,
            signature=f"{item.kind} {item.name}",
        )
        for item in scan_proto_definitions(project_root)
    ]


def generated_go_names(proto_name: str, kind: str) -> list[str]:
    """Common protoc-gen-go / grpc-go symbol names for a proto definition."""
    if kind == "service":
        return [
            proto_name,
            f"{proto_name}Client",
            f"{proto_name}Server",
            f"Unimplemented{proto_name}Server",
            f"New{proto_name}Client",
            f"Register{proto_name}Server",
        ]
    return [proto_name]


def _generated_go_files(
    proto_file: str,
    *,
    project_root: Path,
    go_package: str,
) -> list[str]:
    stem = Path(proto_file).stem
    parent = Path(proto_file).parent
    candidates: list[str] = []
    seen: set[str] = set()

    def add(rel: str) -> None:
        norm = rel.replace("\\", "/")
        if norm in seen:
            return
        if (project_root / norm).is_file():
            seen.add(norm)
            candidates.append(norm)

    for suffix in (f"{stem}.pb.go", f"{stem}_grpc.pb.go"):
        add(str(parent / suffix).replace("\\", "/"))

    if go_package:
        tail = go_package.rstrip("/").split("/")[-2:]
        for path in project_root.rglob("*.pb.go"):
            if any(_skip_dir(part) for part in path.parts):
                continue
            rel = _rel_posix(path, project_root)
            if stem in path.name and all(part in rel for part in tail if part):
                add(rel)
            elif path.stem.startswith(stem):
                add(rel)

    for path in project_root.rglob(f"{stem}*.pb.go"):
        if any(_skip_dir(part) for part in path.parts):
            continue
        add(_rel_posix(path, project_root))

    return candidates[:_MAX_GENERATED_PER_PROTO]


def _symbol_ids_in_files(
    names: list[str],
    files: set[str],
    *,
    name_to_ids: dict[str, list[str]],
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        for sid in name_to_ids.get(name, []):
            file_path = sid.split(":", 1)[0]
            if file_path not in files:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
    return out


def build_proto_reference_edges(
    project_root: Path,
    *,
    symbol_nodes: dict[tuple[str, str, int], str],
    name_to_ids: dict[str, list[str]],
    file_nodes: dict[str, str],
) -> list[tuple[str, str]]:
    """Add proto definition ↔ generated stub symbol edges to the reference graph."""
    root = project_root.resolve()
    edges: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def link(src: str, dst: str) -> None:
        if src == dst:
            return
        key = (src, dst)
        if key in seen:
            return
        seen.add(key)
        edges.append(key)

    for definition in scan_proto_definitions(root):
        proto_key = (definition.file_path, definition.name, definition.start_line)
        proto_sid = symbol_nodes.get(proto_key)
        if proto_sid is None:
            continue
        gen_files = _generated_go_files(
            definition.file_path,
            project_root=root,
            go_package=definition.go_package,
        )
        if not gen_files:
            continue
        gen_file_set = set(gen_files)
        target_ids = _symbol_ids_in_files(
            generated_go_names(definition.name, definition.kind),
            gen_file_set,
            name_to_ids=name_to_ids,
        )
        proto_file_id = file_nodes.get(definition.file_path)
        for target_id in target_ids:
            link(proto_sid, target_id)
            link(target_id, proto_sid)
            if proto_file_id is not None:
                link(proto_file_id, target_id)
    return edges
