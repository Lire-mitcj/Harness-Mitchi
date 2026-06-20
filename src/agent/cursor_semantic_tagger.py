from __future__ import annotations

import re

from src.agent.cursor_contracts import ContextPack, SemanticAnnotations

_TAG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("validation", re.compile(r"\b(validat|validator|check|lint|pytest)\w*\b", re.I)),
    ("sql_check", re.compile(r"(?:^|\W|_)(sql|select|table|view|cte)(?:$|\W|_)", re.I)),
    ("error_handling", re.compile(r"\b(error|exception|fail|raise|except)\w*\b", re.I)),
    ("retrieval", re.compile(r"\b(retriev|search|grep|symbol|ast)\w*\b", re.I)),
    ("configuration", re.compile(r"\b(config|setting|environment)\w*\b", re.I)),
)


class CursorSemanticTagger:
    """Deterministic display-only annotations for already-built context."""

    def annotate(self, context_pack: ContextPack) -> SemanticAnnotations:
        tags_by_file: dict[str, tuple[str, ...]] = {}
        for window in context_pack.windows:
            haystack = f"{window.file}\n{' '.join(window.symbols)}\n{window.content}"
            tags = tuple(tag for tag, pattern in _TAG_PATTERNS if pattern.search(haystack))
            tags_by_file[window.file] = tags
        return SemanticAnnotations(tags_by_file=tags_by_file)
