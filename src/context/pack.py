from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContextSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str = ""
    score: float = 0.0

    @property
    def location(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class ContextSnippet:
    file_path: str
    start_line: int
    end_line: int
    text: str
    source: str = "repo_map"

    @property
    def location(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class SearchPlan:
    module: str
    files: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    globs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ContextPack:
    user_request: str
    relevant_files: tuple[str, ...] = ()
    symbols: tuple[ContextSymbol, ...] = ()
    snippets: tuple[ContextSnippet, ...] = ()
    constraints: tuple[str, ...] = ()
    confidence: float = 0.0
    missing_info: tuple[str, ...] = ()
    search_plan: tuple[SearchPlan, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def is_high_confidence(self, *, threshold: float = 0.75) -> bool:
        return self.confidence >= threshold and not self.missing_info

    def to_planner_block(self, *, max_snippet_chars: int = 4000) -> str:
        lines = [
            '<context_pack source="ContextRetriever">',
            f"confidence: {self.confidence:.2f}",
        ]
        if self.relevant_files:
            lines.append("relevant_files:")
            lines.extend(f"- {path}" for path in self.relevant_files)
        if self.symbols:
            lines.append("symbols:")
            lines.extend(
                f"- {symbol.location} {symbol.kind} {symbol.name} {symbol.signature}".rstrip()
                for symbol in self.symbols
            )
        if self.search_plan:
            lines.append("search_plan:")
            for plan in self.search_plan:
                lines.append(
                    f"- module={plan.module} files={list(plan.files)!r} "
                    f"patterns={'|'.join(plan.patterns)!r} globs={list(plan.globs)!r}"
                )
        if self.snippets:
            lines.append("snippets:")
            remaining = max_snippet_chars
            for snippet in self.snippets:
                if remaining <= 0:
                    lines.append("...[snippets truncated]")
                    break
                body = snippet.text[:remaining]
                remaining -= len(body)
                lines.append(f"--- {snippet.location} ({snippet.source})")
                lines.append(body)
        if self.constraints:
            lines.append("constraints:")
            lines.extend(f"- {constraint}" for constraint in self.constraints)
        if self.missing_info:
            lines.append("missing_info:")
            lines.extend(f"- {item}" for item in self.missing_info)
        lines.append("</context_pack>")
        return "\n".join(lines)
