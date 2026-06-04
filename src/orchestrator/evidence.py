from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidencePack:
    """Failure bundle handed to Planner.re_plan()."""

    subtask_id: str
    subtask_description: str
    error_trace: list[str] = field(default_factory=list)
    file_diffs: dict[str, str] = field(default_factory=dict)
    executor_turns_used: int = 0
    last_assistant_message: str | None = None
    changed_files: list[str] = field(default_factory=list)
    context_files: list[str] = field(default_factory=list)
    subtask_attempts: int = 0

    def to_prompt_block(self) -> str:
        lines = [
            f"Failed subtask: [{self.subtask_id}] {self.subtask_description}",
            f"Executor turns used: {self.executor_turns_used}",
            f"Subtask attempts before re-plan: {self.subtask_attempts}",
        ]
        if self.context_files:
            lines.append(
                "Subtask edit scope (context_files): "
                + ", ".join(self.context_files)
            )
        if self.last_assistant_message:
            lines.append(f"Last assistant message:\n{self.last_assistant_message}")
        if self.error_trace:
            lines.append("Error trace:")
            lines.extend(f"  - {err}" for err in self.error_trace)
        if self.changed_files:
            lines.append("Changed files: " + ", ".join(self.changed_files))
        if self.file_diffs:
            lines.append("Diff evidence:")
            for path, diff in self.file_diffs.items():
                lines.append(f"--- {path} ---")
                lines.append(diff)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "subtask_description": self.subtask_description,
            "error_trace": list(self.error_trace),
            "file_diffs": dict(self.file_diffs),
            "executor_turns_used": self.executor_turns_used,
            "last_assistant_message": self.last_assistant_message,
            "changed_files": list(self.changed_files),
            "context_files": list(self.context_files),
            "subtask_attempts": self.subtask_attempts,
        }
