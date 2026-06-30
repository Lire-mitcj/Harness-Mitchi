from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class ContextLoader:
    """Memoized state loader for system variables, git status, and CLAUDE.md / .mitkii rules."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._system_context_cache: dict[str, Any] = {}
        self._rules_cache: dict[tuple[str, ...], str] = {}

    def get_system_context(self, git_diff: str, validation_error: str | None) -> str:
        """Cache-friendly formatting of the git status and compile errors."""
        cache_key = f"{hash(git_diff)}:{hash(validation_error)}"
        if cache_key in self._system_context_cache:
            return self._system_context_cache[cache_key]

        parts = []
        if git_diff:
            parts.append(f"Git Diff:\n```diff\n{git_diff}\n```")
        else:
            parts.append("Git Status: Clean (No unsaved changes).")
        if validation_error:
            parts.append(f"Build/Validation Errors:\n```\n{validation_error}\n```")

        context_str = "\n\n".join(parts)
        self._system_context_cache[cache_key] = context_str
        return context_str

    def get_hierarchical_rules(self, active_files: list[str]) -> str:
        """Resolve, merge, and cache hierarchical rules files (CLAUDE.md and .mitkii/rules.md).

        Walks up the directory tree from each active file up to the project root
        to find folder-specific instructions.
        """
        cache_key = tuple(sorted(active_files))
        if cache_key in self._rules_cache:
            return self._rules_cache[cache_key]

        rules_found = []
        seen_paths = set()

        # 1. Project level rules first
        project_rules_path = self.project_root / ".mitkii" / "rules.md"
        if project_rules_path.exists() and project_rules_path not in seen_paths:
            seen_paths.add(project_rules_path)
            try:
                rules_found.append(
                    f"### Project Rules ({project_rules_path.name}) ###\n"
                    + project_rules_path.read_text(encoding="utf-8").strip()
                )
            except OSError as exc:
                log.warning("Failed to read project rules from %s: %s", project_rules_path, exc)

        # 2. Directory levels for each active file
        for file in active_files:
            abs_file = (self.project_root / file).resolve()
            if not abs_file.exists():
                continue

            curr = abs_file.parent
            file_rules = []
            while curr != self.project_root and curr.is_relative_to(self.project_root):
                r_md = curr / ".mitkii" / "rules.md"
                if r_md.exists() and r_md not in seen_paths:
                    seen_paths.add(r_md)
                    try:
                        rel_path = r_md.relative_to(self.project_root)
                        file_rules.insert(
                            0,
                            f"### Directory Rules ({rel_path}) ###\n"
                            + r_md.read_text(encoding="utf-8").strip(),
                        )
                    except OSError as exc:
                        log.warning("Failed to read directory rules from %s: %s", r_md, exc)
                curr = curr.parent
            rules_found.extend(file_rules)

        merged_rules = "\n\n".join(rules_found)
        self._rules_cache[cache_key] = merged_rules
        return merged_rules
