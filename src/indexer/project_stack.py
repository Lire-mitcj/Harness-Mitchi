"""Detect primary languages and default tool hints from repository markers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config.settings import MitKIISettings
from src.indexer.language_profiles import (
    GO,
    JAVA,
    PROTO,
    PYTHON,
    SQL,
    LanguageProfile,
    profiles_for_ids,
    searchable_extensions,
)

_DEFAULT_VALIDATOR: tuple[str, ...] = ("pytest",)


@dataclass(frozen=True, slots=True)
class ProjectStack:
    """Repository language mix and derived defaults for retrieval/validation."""

    primary: str  # python | go | java | mixed
    languages: tuple[str, ...]
    default_include_globs: tuple[str, ...]
    validator_command: tuple[str, ...]
    discovery_patterns: tuple[str, ...]
    maven_modules: tuple[str, ...] = ()

    def profiles(self) -> tuple[LanguageProfile, ...]:
        return profiles_for_ids(self.languages)


def _count_by_extension(root: Path, extensions: frozenset[str], *, limit: int = 4000) -> int:
    count = 0
    for path in root.rglob("*"):
        if count >= limit:
            break
        if not path.is_file():
            continue
        if any(part in {".git", "node_modules", "venv", ".venv", "target", "build"} for part in path.parts):
            continue
        if path.suffix.lower() in extensions:
            count += 1
    return count


def detect_project_stack(project_root: Path | None) -> ProjectStack:
    """Infer language stack from build markers and a light file census."""
    if project_root is None or not project_root.is_dir():
        return _python_default()

    root = project_root.resolve()
    markers: dict[str, bool] = {
        "python": any((root / name).is_file() for name in ("pyproject.toml", "requirements.txt", "setup.py")),
        "go": (root / "go.mod").is_file(),
        "java": any((root / name).is_file() for name in ("pom.xml", "build.gradle", "build.gradle.kts")),
        "proto": bool(list(root.rglob("*.proto"))[:1]),
    }

    if not any(markers.values()):
        py_count = _count_by_extension(root, frozenset({".py"}))
        go_count = _count_by_extension(root, frozenset({".go"}))
        java_count = _count_by_extension(root, frozenset({".java"}))
        if go_count > py_count and go_count >= java_count and go_count > 0:
            markers["go"] = True
        elif java_count > py_count and java_count > 0:
            markers["java"] = True
        else:
            markers["python"] = True

    active = [key for key, present in markers.items() if present]
    if not active:
        return _python_default()

    if len(active) == 1:
        primary = active[0]
    else:
        primary = "mixed"

    globs: list[str] = []
    patterns: list[str] = []

    if "python" in active:
        globs.extend(PYTHON.default_globs)
        patterns.extend(PYTHON.discovery_patterns)
    if "go" in active:
        globs.extend(GO.default_globs)
        patterns.extend(GO.discovery_patterns)
    maven_modules: tuple[str, ...] = ()
    if "java" in active:
        globs.extend(JAVA.default_globs)
        patterns.extend(JAVA.discovery_patterns)
        pom = root / "pom.xml"
        if pom.is_file():
            from src.indexer.maven_project import module_paths_from_pom

            maven_modules = module_paths_from_pom(pom)
    if "proto" in active:
        globs.extend(PROTO.default_globs)
        patterns.extend(PROTO.discovery_patterns)
    if (root / "db").is_dir() or _count_by_extension(root, frozenset({".sql"})) > 0:
        globs.extend(SQL.default_globs)
        patterns.extend(SQL.discovery_patterns)

    validator = _validator_for_stack(active, root, primary=primary)

    # Deduplicate while preserving order
    globs = list(dict.fromkeys(globs))
    patterns = list(dict.fromkeys(patterns))

    return ProjectStack(
        primary=primary,
        languages=tuple(active),
        default_include_globs=tuple(globs) if globs else ("*.*",),
        validator_command=validator,
        discovery_patterns=tuple(patterns[:12]),
        maven_modules=maven_modules,
    )


def _python_default() -> ProjectStack:
    return ProjectStack(
        primary="python",
        languages=("python",),
        default_include_globs=PYTHON.default_globs,
        validator_command=("pytest",),
        discovery_patterns=PYTHON.discovery_patterns,
    )


def _validator_for_stack(
    active: list[str],
    root: Path,
    *,
    primary: str,
) -> tuple[str, ...]:
    """Pick one execution validator; go/java win over pytest when markers exist."""
    if "go" in active and (root / "go.mod").is_file():
        return ("go", "test", "./...")
    if "java" in active and any(
        (root / name).is_file() for name in ("pom.xml", "build.gradle", "build.gradle.kts")
    ):
        return ("mvn", "-q", "test")
    if "python" in active:
        return ("pytest",)
    if primary == "proto":
        return ("pytest",)
    return _DEFAULT_VALIDATOR


def validator_command_for_target(
    *,
    stack: ProjectStack,
    target_file: str,
    project_root: Path,
) -> tuple[str, ...]:
    """Pick an execution validator for a specific edited file in mixed monorepos."""
    if not target_file.strip():
        return stack.validator_command

    suffix = Path(target_file).suffix.casefold()
    root = project_root.resolve()

    if suffix == ".go" and (root / "go.mod").is_file():
        return ("go", "test", "./...")
    if suffix == ".py":
        return ("pytest",)
    if suffix in {".java", ".kt", ".scala"} and any(
        (root / name).is_file() for name in ("pom.xml", "build.gradle", "build.gradle.kts")
    ):
        return ("mvn", "-q", "test")
    if suffix in {".sql", ".proto"}:
        return ()

    return stack.validator_command


def maven_module_for_target(
    target_file: str,
    maven_modules: tuple[str, ...],
) -> str | None:
    """Return the longest matching Maven reactor module for a source path."""
    if not target_file or not maven_modules:
        return None
    normalized = target_file.replace("\\", "/")
    path_parts = Path(normalized).parts
    best: str | None = None
    best_len = -1
    for module in maven_modules:
        mod = module.replace("\\", "/").strip("/")
        if not mod:
            continue
        mod_parts = Path(mod).parts
        if len(path_parts) >= len(mod_parts) and path_parts[: len(mod_parts)] == mod_parts:
            if len(mod_parts) > best_len:
                best = mod
                best_len = len(mod_parts)
    return best


def indexable_extensions_for_stack(stack: ProjectStack) -> frozenset[str]:
    """File suffixes that should contribute symbols to repo_map for this stack."""
    exts = set(searchable_extensions(stack.languages))
    if "java" in stack.languages:
        exts.add(".xml")
    if "sql" in stack.languages:
        exts.add(".sql")
    return frozenset(exts)


def apply_project_stack_to_settings(
    settings: MitKIISettings,
    project_root: Path,
) -> MitKIISettings:
    """Override default pytest validator when the repo stack implies another command."""
    if not settings.cursor_validator_auto:
        return settings
    if tuple(settings.cursor_validator_command) != _DEFAULT_VALIDATOR:
        return settings
    stack = detect_project_stack(project_root)
    if stack.validator_command == _DEFAULT_VALIDATOR:
        return settings
    return settings.model_copy(
        update={"cursor_validator_command": list(stack.validator_command)}
    )


def default_include_for_stack(stack: ProjectStack) -> str | None:
    """Return a single include glob or brace pattern for grep when stack is narrow."""
    if stack.primary == "go":
        return "*.go"
    if stack.primary == "java":
        return "*.{java,xml}"
    if stack.primary == "python":
        return "*.py"
    if len(stack.default_include_globs) == 1:
        return stack.default_include_globs[0]
    return None
