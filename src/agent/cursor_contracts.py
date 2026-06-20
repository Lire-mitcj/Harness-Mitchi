from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DecisionAction = Literal["edit", "answer", "ask_clarify"]
ValidationDecision = Literal["commit", "rollback", "retry"]


@dataclass(frozen=True, slots=True)
class RetrievalSymbol:
    file: str
    name: str
    start_line: int
    end_line: int
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    calls: tuple[str, ...] = ()
    kind: str = ""
    tables_referenced: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    files: tuple[str, ...] = ()
    symbols: tuple[RetrievalSymbol, ...] = ()

    @property
    def candidate_files(self) -> tuple[str, ...]:
        ordered: list[str] = []
        for symbol in self.symbols:
            if symbol.file not in ordered:
                ordered.append(symbol.file)
        for file in self.files:
            if file not in ordered:
                ordered.append(file)
        return tuple(ordered)


@dataclass(frozen=True, slots=True)
class ContextWindow:
    file: str
    start_line: int
    end_line: int
    content: str
    symbols: tuple[str, ...] = ()
    semantic_tags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextPack:
    windows: tuple[ContextWindow, ...] = ()

    @property
    def candidate_files(self) -> tuple[str, ...]:
        return tuple(window.file for window in self.windows)


@dataclass(frozen=True, slots=True)
class Decision:
    action: DecisionAction
    answer: str = ""
    clarification: str = ""
    target_file: str = ""
    patch: str = ""
    suggested_completion: float = 0.0


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    success: bool
    file: str
    error: str = ""
    original_content: str = ""
    attempted_content: str = ""
    rolled_back: bool = False


@dataclass(frozen=True, slots=True)
class ValidationResult:
    success: bool
    error: str = ""
    ast: dict[str, object] | None = None
    semantic: dict[str, object] | None = None
    execution: dict[str, object] | None = None
    decision: ValidationDecision = "rollback"
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class InterHint:
    intent: str
    domains: tuple[str, ...]
    concepts: tuple[str, ...]
    ambiguity: bool
    confidence: float


@dataclass(frozen=True, slots=True)
class SemanticAnnotations:
    tags_by_file: dict[str, tuple[str, ...]]
