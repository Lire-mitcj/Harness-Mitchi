from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.types import Message, system_message, user_message
from src.context.file_tracker import FileTracker
from src.planner.intent_hints import cap_structure_text

if TYPE_CHECKING:
    from src.indexer.repo_map_service import RepoMapService


SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "system_prompt.md"


@dataclass(frozen=True, slots=True)
class ProjectContextSnapshot:
    text: str
    fingerprint: str


class ContextBuilder:
    """Assembles the full message list for each LLM call.

    Composes: system prompt + project context + conversation history + enriched user message.
    """

    def __init__(
        self,
        project_root: Path,
        file_tracker: FileTracker,
        project_rules: str | None = None,
        repo_map_service: RepoMapService | None = None,
        *,
        repo_map_max_chars: int = 12_000,
        structure_max_chars: int = 3500,
    ) -> None:
        self.project_root = project_root
        self.file_tracker = file_tracker
        self.project_rules = project_rules
        self.repo_map_service = repo_map_service
        self.repo_map_max_chars = repo_map_max_chars
        self.structure_max_chars = structure_max_chars
        self._project_snapshot: ProjectContextSnapshot | None = None

    def invalidate_project_context(self) -> None:
        """Drop cached L1 text after repo map refresh."""
        self._project_snapshot = None

    async def get_project_context_snapshot(self) -> ProjectContextSnapshot | None:
        live_fp = self._live_project_fingerprint()
        if (
            self._project_snapshot is not None
            and self._project_snapshot.fingerprint == live_fp
        ):
            return self._project_snapshot
        text = await self._build_project_context_fresh()
        if not text:
            self._project_snapshot = None
            return None
        self._project_snapshot = ProjectContextSnapshot(text=text, fingerprint=live_fp)
        return self._project_snapshot

    def _live_project_fingerprint(self) -> str:
        parts: list[str] = [f"tree_cap:{self.structure_max_chars}"]
        if self.repo_map_service is not None and self.repo_map_service.map is not None:
            repo_map = self.repo_map_service.map
            parts.append(
                f"map:{repo_map.symbol_count}:{repo_map.build_ms}:{repo_map.source}"
            )
        parts.append(f"chars:{self.repo_map_max_chars}")
        return "|".join(parts)

    async def build(
        self,
        user_msg: str,
        conversation_history: list[Message] | None = None,
    ) -> list[Message]:
        messages: list[Message] = []

        sys_prompt = self._build_system_prompt()
        messages.append(system_message(sys_prompt, cache_breakpoint=True))

        project_ctx = await self._build_project_context()
        if project_ctx:
            messages.append(system_message(project_ctx, cache_breakpoint=True))

        if conversation_history:
            messages.extend(conversation_history)

        relevant_files = await self._find_relevant_files(user_msg)
        enriched = self._enrich_with_files(user_msg, relevant_files)
        messages.append(user_message(enriched))

        return messages

    # ── Private helpers ──────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        template = self._load_prompt_template()
        project_info = self._get_project_info()
        rules = self.project_rules or "(no project rules defined)"
        return template.format(project_info=project_info, project_rules=rules)

    def _load_prompt_template(self) -> str:
        if SYSTEM_PROMPT_PATH.exists():
            return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        return (
            "You are MitKII, an AI coding agent. Help the user with their coding tasks "
            "using the available tools.\n\n"
            "<project_info>\n{project_info}\n</project_info>\n\n"
            "<project_rules>\n{project_rules}\n</project_rules>"
        )

    def _get_project_info(self) -> str:
        parts: list[str] = []
        parts.append(f"Root: {self.project_root}")

        git_branch = self._get_git_branch()
        if git_branch:
            parts.append(f"Git branch: {git_branch}")

        recent = self.file_tracker.get_recent(limit=5)
        if recent:
            parts.append("Recently touched files:\n" + "\n".join(f"  - {f}" for f in recent))

        return "\n".join(parts)

    def _get_git_branch(self) -> str | None:
        head = self.project_root / ".git" / "HEAD"
        if not head.exists():
            return None
        try:
            content = head.read_text(encoding="utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                return content[len("ref: refs/heads/"):]
            return content[:12]
        except OSError:
            return None

    async def _build_project_context(self) -> str | None:
        snap = await self.get_project_context_snapshot()
        return snap.text if snap else None

    async def build_planner_project_context(self) -> str | None:
        return await self._build_project_context()

    async def _build_project_context_fresh(self) -> str | None:
        parts: list[str] = []
        if self.repo_map_service is not None:
            block = self.repo_map_service.to_planner_context(
                max_chars=self.repo_map_max_chars
            )
            if block:
                parts.append(block)
        tree = self._get_directory_tree(max_depth=2)
        if tree:
            block = f"<project_structure>\n{tree}\n</project_structure>"
            parts.append(cap_structure_text(block, max_chars=self.structure_max_chars))
        if not parts:
            return None
        return "\n\n".join(parts)

    def _get_directory_tree(self, max_depth: int = 2) -> str:
        lines: list[str] = []
        root = self.project_root

        def _walk(directory: Path, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name))
            except PermissionError:
                return

            dirs = [e for e in entries if e.is_dir() and not self._should_ignore(e)]
            files = [e for e in entries if e.is_file() and not self._should_ignore(e)]

            for i, d in enumerate(dirs):
                connector = "├── " if (i < len(dirs) - 1 or files) else "└── "
                lines.append(f"{prefix}{connector}{d.name}/")
                extension = "│   " if (i < len(dirs) - 1 or files) else "    "
                _walk(d, prefix + extension, depth + 1)

            for i, f in enumerate(files):
                connector = "├── " if i < len(files) - 1 else "└── "
                lines.append(f"{prefix}{connector}{f.name}")

        lines.append(f"{root.name}/")
        _walk(root, "", 0)
        return "\n".join(lines[:200])

    def _should_ignore(self, path: Path) -> bool:
        name = path.name
        ignore = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".mypy_cache", ".pytest_cache", ".ruff_cache", ".eggs",
            "dist", "build", ".egg-info", ".tox", ".mitkii",
        }
        return name in ignore or name.startswith(".")

    async def _find_relevant_files(self, message: str) -> list[tuple[str, str]]:
        """Extract file paths mentioned in the message and load their content."""
        files: list[tuple[str, str]] = []

        for token in message.split():
            cleaned = token.strip("\"'`,:;()[]{}").replace("\\", "/")
            candidate = self.project_root / cleaned
            if candidate.is_file() and candidate.stat().st_size < 100_000:
                try:
                    content = candidate.read_text(encoding="utf-8", errors="replace")
                    rel = os.path.relpath(candidate, self.project_root)
                    files.append((rel, content))
                    self.file_tracker.record_read(rel)
                except OSError:
                    pass

        return files

    def _enrich_with_files(
        self, message: str, files: list[tuple[str, str]]
    ) -> str:
        if not files:
            return message

        parts = [message, "\n\n<attached_files>"]
        for path, content in files:
            parts.append(f"\n<file path=\"{path}\">\n{content}\n</file>")
        parts.append("\n</attached_files>")
        return "\n".join(parts)
