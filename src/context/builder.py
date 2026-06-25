from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.agent.types import Message, system_message, user_message
from src.context.file_tracker import FileTracker

def cap_structure_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... (truncated)"

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

        # --- (1) System Layer ---
        sys_prompt = self._build_system_prompt()
        
        # Build Environment Info
        env_parts: list[str] = [
            f"Root: {self.project_root}",
            "OS: Linux",
            "Python: 3.11"
        ]
        git_branch = self._get_git_branch()
        if git_branch:
            env_parts.append(f"Git branch: {git_branch}")
        env_info = "\n".join(env_parts)
        
        # Output style rules
        output_styles = (
            "## Output Styles & Policies:\n"
            "- Implement highly aesthetic, dynamic, and modern layouts with Harmonious HSL colors, Outfit/Google fonts, micro-animations, glassmorphism, and responsive design.\n"
            "- Strictly enforce unique, descriptive element IDs for frontend testing.\n"
            "- Never try to modify multiple files in a single edit tool call. All multi-file modifications must be split into step-by-step sequential tool calls.\n"
            "- Complete page-level SEO details (title tags, meta description, and single h1 element)."
        )
        
        # Skill descriptions
        skills = (
            "## Available Skills:\n"
            "- **Code Search**: Locates matched paths and caches hydrated results in `.mitkii/search_cache.json`.\n"
            "- **Code Edit**: Performs exact block replacement on target files.\n"
            "- **Validator**: Programmatically syntax checks, compile tests, and semantic verifies changes."
        )
        
        system_layer_info = f"{sys_prompt}\n\n## Environment Info:\n{env_info}\n\n{output_styles}\n\n{skills}"

        # --- (2) Project Config Layer ---
        claude_hierarchy = self._load_claude_hierarchy()
        path_scoped_rules = self._load_path_scoped_rules()
        project_ctx = await self._build_project_context() or "(no project structure context)"

        # --- (3) Memory Layer ---
        recent_files = self.file_tracker.get_recent(limit=5)
        working_memory = "\n".join(f"  - {f}" for f in recent_files) if recent_files else "(no recently touched files)"
        project_memory = await self._load_project_memory_facts()

        # Combine System Layer, Project Config, and Memory Layer into one system message for caching
        initial_system = (
            "# ⚙️ CONTEXT SYSTEM ARCHITECTURE\n\n"
            "<system_layer>\n"
            f"{system_layer_info}\n"
            "</system_layer>\n\n"
            "<project_config>\n"
            "## Project Structure (Directory Tree & Repo Map):\n"
            f"{project_ctx}\n\n"
            "## Hierarchical Configuration (CLAUDE.md Hierarchy):\n"
            f"{claude_hierarchy}\n\n"
            "## Path-scoped Rules (.claude/rules/*):\n"
            f"{path_scoped_rules}\n"
            "</project_config>\n\n"
            "<memory>\n"
            "## Working Memory (Recently Touched Files):\n"
            f"{working_memory}\n\n"
            "## Project Memory (SQLite Conventions & Facts):\n"
            f"{project_memory}\n"
            "</memory>"
        )
        
        messages.append(system_message(initial_system, cache_breakpoint=True))

        # --- (4) Conversation Layer ---
        if conversation_history:
            messages.extend(conversation_history)

        # --- (5) Runtime Layer ---
        relevant_files = await self._find_relevant_files(user_msg)
        enriched = self._enrich_with_files(user_msg, relevant_files)
        messages.append(user_message(enriched))

        return messages

    # ── Private helpers ──────────────────────────────────────────

    def _load_claude_hierarchy(self) -> str:
        """Find and read CLAUDE.md files hierarchically, looking up to 5 levels from project_root."""
        parts: list[str] = []
        curr = self.project_root.resolve()
        for i in range(5):
            claude_md = curr / "CLAUDE.md"
            if claude_md.is_file():
                try:
                    content = claude_md.read_text(encoding="utf-8")
                    parts.append(f"### CLAUDE.md (Level {i+1}: {curr.name})\n{content}")
                except Exception:
                    pass
            # Move to parent
            parent = curr.parent
            if parent == curr:
                break
            curr = parent
        return "\n\n".join(reversed(parts)) if parts else "(no CLAUDE.md files found in the hierarchy)"

    def _load_path_scoped_rules(self) -> str:
        """Load any rules from .claude/rules/* and .mitkii/rules/* directories."""
        rules_content: list[str] = []
        for rules_dir_name in (".claude/rules", ".mitkii/rules", "rules"):
            rules_dir = self.project_root / rules_dir_name
            if rules_dir.is_dir():
                try:
                    for rule_file in sorted(rules_dir.glob("*.md")):
                        if rule_file.is_file():
                            content = rule_file.read_text(encoding="utf-8")
                            rules_content.append(f"### Rule: {rule_file.name}\n{content}")
                except Exception:
                    pass
        return "\n\n".join(rules_content) if rules_content else "(no path-scoped rules found)"

    async def _load_project_memory_facts(self) -> str:
        """Connect to project database and retrieve memory facts."""
        db_path = self.project_root / ".mitkii" / "project_memory.db"
        if not db_path.is_file():
            return "(no active project memory DB found)"
        
        try:
            import aiosqlite
            async with aiosqlite.connect(str(db_path)) as db:
                async with db.execute("SELECT key, value FROM project_facts LIMIT 20") as cursor:
                    rows = await cursor.fetchall()
                    if not rows:
                        return "(no project memory facts recorded)"
                    parts = []
                    for key, val in rows:
                        parts.append(f"- **{key}**: {val}")
                    return "\n".join(parts)
        except Exception as e:
            return f"(error reading project memory: {e})"

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

        parts = [message, "\n\n<runtime_layer>", "<attached_files>"]
        for path, content in files:
            parts.append(f"\n<file path=\"{path}\">\n{content}\n</file>")
        parts.append("\n</attached_files>")
        parts.append("</runtime_layer>")
        return "\n".join(parts)
