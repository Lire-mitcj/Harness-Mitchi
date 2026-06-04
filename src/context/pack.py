from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


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
    task: dict[str, Any] = field(default_factory=dict)
    candidate_files: tuple[dict[str, Any], ...] = ()
    candidate_symbols: tuple[dict[str, Any], ...] = ()
    relevant_files: tuple[str, ...] = ()
    symbols: tuple[ContextSymbol, ...] = ()
    repo_map: tuple[dict[str, Any], ...] = ()
    snippets: tuple[ContextSnippet, ...] = ()
    focused_snippets: tuple[ContextSnippet, ...] = ()
    evidence: tuple[dict[str, Any], ...] = ()
    known_negatives: tuple[dict[str, Any], ...] = ()
    call_chain: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    tool_policy: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    missing_info: tuple[str, ...] = ()
    search_plan: tuple[SearchPlan, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)

    def is_high_confidence(self, *, threshold: float = 0.75) -> bool:
        return self.confidence >= threshold and not self.missing_info

    def to_agent_json(self, *, max_snippet_chars: int = 12_000) -> dict[str, Any]:
        snippets: list[dict[str, Any]] = []
        remaining = max_snippet_chars
        source_snippets = self.focused_snippets or self.snippets
        for snippet in source_snippets:
            if remaining <= 0:
                break
            body = snippet.text[:remaining]
            remaining -= len(body)
            snippets.append({
                "file": snippet.file_path,
                "start_line": snippet.start_line,
                "end_line": snippet.end_line,
                "source": snippet.source,
                "content": body,
            })
        return {
            "schema": "mitkii.context_pack.v1",
            "task": self.task or {
                "objective": self.user_request,
                "mode": "unknown",
                "risk": "medium",
            },
            "candidate_files": list(self.candidate_files),
            "candidate_symbols": list(self.candidate_symbols),
            "repo_map": list(self.repo_map),
            "focused_snippets": snippets,
            "evidence": list(self.evidence),
            "known_negatives": list(self.known_negatives),
            "call_chain": list(self.call_chain),
            "constraints": list(self.constraints),
            "tool_policy": self.tool_policy,
            "budget": self.budget,
            "confidence": self.confidence,
            "missing_info": list(self.missing_info),
            "search_plan": [
                {
                    "module": plan.module,
                    "files": list(plan.files),
                    "patterns": list(plan.patterns),
                    "globs": list(plan.globs),
                }
                for plan in self.search_plan
            ],
        }

    def to_agent_block(self, *, max_snippet_chars: int = 12_000) -> str:
        return (
            "CONTEXT_PACK_JSON\n"
            + json.dumps(
                self.to_agent_json(max_snippet_chars=max_snippet_chars),
                ensure_ascii=False,
                indent=2,
            )
        )

    def to_planner_block(self, *, max_snippet_chars: int = 4000) -> str:
        lines = [
            '<context_pack source="ContextRetriever">',
            f"confidence: {self.confidence:.2f}",
        ]
        if self.candidate_files:
            lines.append("candidate_files:")
            lines.extend(
                f"- {item.get('file')} score={float(item.get('score', 0.0)):.2f} "
                f"reasons={','.join(str(r) for r in item.get('reasons', []))}"
                for item in self.candidate_files[:10]
            )
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
        if self.evidence:
            lines.append("evidence:")
            for item in self.evidence[:12]:
                lines.append("- " + json.dumps(item, ensure_ascii=False, sort_keys=True))
        if self.known_negatives:
            lines.append("known_negatives:")
            for item in self.known_negatives[:8]:
                lines.append("- " + json.dumps(item, ensure_ascii=False, sort_keys=True))
        if self.call_chain:
            lines.append("call_chain:")
            lines.extend(f"- {edge}" for edge in self.call_chain[:12])
        if self.constraints:
            lines.append("constraints:")
            lines.extend(f"- {constraint}" for constraint in self.constraints)
        if self.missing_info:
            lines.append("missing_info:")
            lines.extend(f"- {item}" for item in self.missing_info)
        lines.append("</context_pack>")
        return "\n".join(lines)
