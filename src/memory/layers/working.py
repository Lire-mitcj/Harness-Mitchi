from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    """L1: In-memory store for the current session.

    Holds conversation context, current plan, and file change tracking.
    Cleared when the session ends; summarized into long-term memory.
    """

    notes: list[str] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    current_plan: str | None = None
    plan_progress: dict[str, str] = field(default_factory=dict)
    key_decisions: list[str] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    def record_file_change(self, path: str) -> None:
        if path not in self.file_changes:
            self.file_changes.append(path)

    def set_plan(self, plan: str) -> None:
        self.current_plan = plan

    def update_progress(self, step_id: str, status: str) -> None:
        self.plan_progress[step_id] = status

    def add_decision(self, decision: str) -> None:
        self.key_decisions.append(decision)

    def search(self, query: str) -> list[str]:
        query_lower = query.lower()
        results: list[str] = []
        for note in self.notes:
            if query_lower in note.lower():
                results.append(note)
        for decision in self.key_decisions:
            if query_lower in decision.lower():
                results.append(decision)
        return results

    def generate_summary(self) -> str | None:
        parts: list[str] = []
        if self.file_changes:
            parts.append(f"Modified files: {', '.join(self.file_changes)}")
        if self.key_decisions:
            parts.append("Key decisions:\n" + "\n".join(f"- {d}" for d in self.key_decisions))
        if self.current_plan:
            parts.append(f"Plan: {self.current_plan}")
        return "\n\n".join(parts) if parts else None

    def export(self) -> dict:
        return {
            "notes": self.notes,
            "file_changes": self.file_changes,
            "current_plan": self.current_plan,
            "plan_progress": self.plan_progress,
            "key_decisions": self.key_decisions,
        }

    def import_(self, data: dict) -> None:
        self.notes = data.get("notes", [])
        self.file_changes = data.get("file_changes", [])
        self.current_plan = data.get("current_plan")
        self.plan_progress = data.get("plan_progress", {})
        self.key_decisions = data.get("key_decisions", [])

    def clear(self) -> None:
        self.notes.clear()
        self.file_changes.clear()
        self.current_plan = None
        self.plan_progress.clear()
        self.key_decisions.clear()
