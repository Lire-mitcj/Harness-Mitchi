"""Lightweight Maven ``pom.xml`` parsing for multi-module layout hints."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class MavenModule:
    """A Maven reactor module relative to the parent POM directory."""

    path: str
    artifact_id: str
    packaging: str = "jar"


@dataclass(frozen=True, slots=True)
class MavenProject:
    """Parsed root (or sub) POM metadata."""

    pom_path: Path
    artifact_id: str
    packaging: str
    modules: tuple[MavenModule, ...]


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(root: ET.Element, name: str) -> str | None:
    for child in root:
        if _local_tag(child.tag) == name and child.text:
            return child.text.strip()
    return None


def _module_paths(root: ET.Element) -> list[str]:
    paths: list[str] = []
    for child in root:
        if _local_tag(child.tag) != "modules":
            continue
        for mod in child:
            if _local_tag(mod.tag) == "module" and mod.text:
                text = mod.text.strip()
                if text:
                    paths.append(text.replace("\\", "/"))
    return paths


def parse_pom(pom_path: Path) -> MavenProject | None:
    """Parse artifactId, packaging, and reactor modules from a POM file."""
    if not pom_path.is_file():
        return None
    try:
        root = ET.parse(pom_path).getroot()
    except (ET.ParseError, OSError):
        return None

    artifact_id = _child_text(root, "artifactId") or pom_path.parent.name
    packaging = _child_text(root, "packaging") or "jar"
    parent_dir = pom_path.parent
    modules: list[MavenModule] = []
    for rel in _module_paths(root):
        module_pom = parent_dir / rel / "pom.xml"
        sub_artifact = rel.rsplit("/", 1)[-1]
        sub_packaging = "jar"
        if module_pom.is_file():
            sub = parse_pom(module_pom)
            if sub is not None:
                sub_artifact = sub.artifact_id
                sub_packaging = sub.packaging
        modules.append(
            MavenModule(path=rel, artifact_id=sub_artifact, packaging=sub_packaging)
        )
    return MavenProject(
        pom_path=pom_path.resolve(),
        artifact_id=artifact_id,
        packaging=packaging,
        modules=tuple(modules),
    )


def find_maven_project(project_root: Path) -> MavenProject | None:
    """Return parsed Maven metadata when ``pom.xml`` exists at the project root."""
    pom = project_root.resolve() / "pom.xml"
    return parse_pom(pom)


def module_paths_from_pom(pom_path: Path) -> tuple[str, ...]:
    """Return reactor module relative paths declared in a POM."""
    project = parse_pom(pom_path)
    if project is None:
        return ()
    return tuple(mod.path for mod in project.modules)
