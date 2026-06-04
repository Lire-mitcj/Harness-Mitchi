from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectStructure:
    root: Path
    files: list[Path] = field(default_factory=list)
    directories: list[Path] = field(default_factory=list)
    total_files: int = 0
    languages: dict[str, int] = field(default_factory=dict)


EXTENSION_LANGUAGE_MAP: dict[str, str] = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".jsx": "JavaScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".c": "C", ".cpp": "C++",
    ".h": "C", ".hpp": "C++", ".rb": "Ruby", ".php": "PHP",
    ".swift": "Swift", ".kt": "Kotlin", ".scala": "Scala",
    ".cs": "C#", ".vue": "Vue", ".svelte": "Svelte",
    ".html": "HTML", ".css": "CSS", ".scss": "SCSS",
    ".yaml": "YAML", ".yml": "YAML", ".toml": "TOML",
    ".json": "JSON", ".md": "Markdown", ".sql": "SQL",
    ".sh": "Shell", ".bash": "Shell",
}

DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    "dist", "build", ".eggs", ".egg-info", "target",
    ".next", ".nuxt", "coverage", "htmlcov", ".mitkii",
}

DEFAULT_IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo", "*.so",
    "*.dylib", "*.dll", "*.exe", "*.o", "*.a",
}


class ProjectScanner:
    """Walks a project directory tree respecting .gitignore patterns."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self._gitignore_patterns = self._load_gitignore()

    def scan(self, max_files: int = 10_000) -> ProjectStructure:
        structure = ProjectStructure(root=self.root)
        languages: dict[str, int] = {}

        for path in self._walk(self.root):
            if path.is_dir():
                structure.directories.append(path)
            elif path.is_file():
                if len(structure.files) >= max_files:
                    break
                structure.files.append(path)
                lang = EXTENSION_LANGUAGE_MAP.get(path.suffix.lower())
                if lang:
                    languages[lang] = languages.get(lang, 0) + 1

        structure.total_files = len(structure.files)
        structure.languages = dict(sorted(languages.items(), key=lambda x: -x[1]))
        return structure

    def get_directory_tree(self, max_depth: int = 3) -> str:
        lines: list[str] = [f"{self.root.name}/"]
        self._tree_walk(self.root, "", 0, max_depth, lines)
        return "\n".join(lines[:300])

    def _tree_walk(
        self, directory: Path, prefix: str, depth: int, max_depth: int, lines: list[str],
    ) -> None:
        if depth >= max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
        except PermissionError:
            return

        visible = [e for e in entries if not self._should_ignore(e)]
        for i, entry in enumerate(visible):
            is_last = i == len(visible) - 1
            connector = "└── " if is_last else "├── "
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{connector}{entry.name}{suffix}")
            if entry.is_dir():
                extension = "    " if is_last else "│   "
                self._tree_walk(entry, prefix + extension, depth + 1, max_depth, lines)

    def _walk(self, directory: Path) -> list[Path]:
        result: list[Path] = []
        try:
            entries = sorted(directory.iterdir(), key=lambda p: p.name)
        except PermissionError:
            return result

        for entry in entries:
            if self._should_ignore(entry):
                continue
            result.append(entry)
            if entry.is_dir():
                result.extend(self._walk(entry))
        return result

    def _should_ignore(self, path: Path) -> bool:
        name = path.name
        if path.is_dir() and name in DEFAULT_IGNORE_DIRS:
            return True
        if path.is_file():
            for pattern in DEFAULT_IGNORE_FILES:
                if fnmatch.fnmatch(name, pattern):
                    return True
        rel = str(path.relative_to(self.root))
        for pattern in self._gitignore_patterns:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _load_gitignore(self) -> list[str]:
        gitignore = self.root / ".gitignore"
        if not gitignore.exists():
            return []
        patterns: list[str] = []
        for line in gitignore.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line.rstrip("/"))
        return patterns
