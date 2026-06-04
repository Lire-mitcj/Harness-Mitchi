from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PatchEdit:
    path: str
    old_string: str
    new_string: str
    symbol: str = ""

    def is_valid(self) -> bool:
        return (
            bool(self.path.strip())
            and bool(self.old_string)
            and bool(self.new_string)
            and self.old_string != self.new_string
        )


@dataclass(frozen=True)
class PatchPlan:
    files_to_edit: tuple[str, ...] = ()
    target_symbols: tuple[str, ...] = ()
    intended_changes: tuple[str, ...] = ()
    edits: tuple[PatchEdit, ...] = ()
    validation_plan: tuple[str, ...] = ()
    requires_confirmation: bool = False
    confidence: float = 0.0
    missing_info: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def is_executable(self, *, threshold: float = 0.75) -> bool:
        return (
            self.confidence >= threshold
            and bool(self.files_to_edit)
            and bool(self.intended_changes)
            and bool(self.edits)
            and all(edit.is_valid() for edit in self.edits)
            and not self.missing_info
            and not self.requires_confirmation
        )
