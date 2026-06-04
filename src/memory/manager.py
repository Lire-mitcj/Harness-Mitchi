from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.types import Message
from src.memory.layers.long_term import LongTermMemory
from src.memory.layers.project import ProjectMemory
from src.memory.layers.working import WorkingMemory


class MemoryManager:
    """Coordinates the three memory layers.

    * **Working** — ephemeral, in-memory, current session only.
    * **Project** — SQLite-backed, per-project conventions and structure.
    * **Long-term** — SQLite-backed, cross-project knowledge at ``~/.mitkii/``.
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or (Path.home() / ".mitkii")
        self.working = WorkingMemory()
        self.project: ProjectMemory | None = None
        self.long_term = LongTermMemory(self._data_dir)
        self._project_path: Path | None = None

    async def init(self, project_path: Path) -> None:
        """Initialise all memory layers for *project_path*."""
        self._project_path = project_path.resolve()
        self.working = WorkingMemory()

        self.project = ProjectMemory(self._project_path)
        await self.project.init()

        await self.long_term.init()

    # ------------------------------------------------------------------
    # Unified recall
    # ------------------------------------------------------------------

    async def recall(self, query: str) -> list[dict[str, Any]]:
        """Search across all memory layers and return merged results."""
        results: list[dict[str, Any]] = []

        # Working memory scratch
        for key, val in self.working.scratch.items():
            if query.lower() in key.lower() or query.lower() in str(val).lower():
                results.append({
                    "source": "working",
                    "key": key,
                    "content": str(val),
                })

        # Project conventions
        if self.project is not None:
            conventions = await self.project.get_conventions()
            for c in conventions:
                if query.lower() in c.get("content", "").lower():
                    results.append({
                        "source": "project",
                        "category": c.get("category", ""),
                        "content": c["content"],
                    })

        # Long-term search
        lt_results = await self.long_term.search(query, limit=5)
        for r in lt_results:
            results.append({
                "source": "long_term",
                "content": r.get("content", ""),
                "project": r.get("project"),
            })

        return results

    # ------------------------------------------------------------------
    # Unified remember
    # ------------------------------------------------------------------

    async def remember(self, key: str, value: str, layer: str = "working") -> None:
        """Store a piece of information in the specified layer."""
        match layer:
            case "working":
                self.working.remember(key, value)
            case "project":
                if self.project is not None:
                    await self.project.add_convention(
                        category=key, content=value, source="agent"
                    )
            case "long_term":
                await self.long_term.add_knowledge(
                    topic=key,
                    content=value,
                    project=str(self._project_path) if self._project_path else None,
                )
            case _:
                self.working.remember(key, value)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def end_session(self, messages: list[Message]) -> None:
        """Compress the current session into long-term memory.

        Produces a summary of the conversation and saves it alongside
        the list of modified files.
        """
        if not messages:
            return

        summary_parts: list[str] = []
        for msg in messages:
            if msg.role in ("user", "assistant") and msg.content:
                prefix = "User" if msg.role == "user" else "Agent"
                content = msg.content[:200]
                summary_parts.append(f"[{prefix}] {content}")

        summary = "\n".join(summary_parts[-20:])  # last 20 turns

        project_name = str(self._project_path) if self._project_path else "unknown"
        changed = self.working.get_changed_files()

        await self.long_term.save_session_summary(
            summary=summary,
            project=project_name,
            files=changed or None,
        )

    # ------------------------------------------------------------------
    # Context string for LLM
    # ------------------------------------------------------------------

    async def get_memory_context(self) -> str:
        """Build a memory summary string suitable for injection into the system prompt."""
        sections: list[str] = []

        # Working memory plan
        if self.working.current_plan:
            sections.append(f"Current plan:\n{self.working.current_plan}")

        # Working memory file changes
        changed = self.working.get_changed_files()
        if changed:
            sections.append("Files modified this session:\n" + "\n".join(f"  - {f}" for f in changed))

        # Project summary
        if self.project is not None:
            proj_summary = await self.project.get_project_summary()
            if proj_summary and proj_summary != "No project memory recorded yet.":
                sections.append(proj_summary)

        # User preferences
        prefs = await self.long_term.get_user_preferences()
        if prefs:
            pref_lines = [f"  {k}: {v}" for k, v in prefs.items()]
            sections.append("User preferences:\n" + "\n".join(pref_lines))

        return "\n\n".join(sections) if sections else ""

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def close(self) -> None:
        if self.project is not None:
            await self.project.close()
        await self.long_term.close()
