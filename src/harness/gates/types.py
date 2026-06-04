from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal


class GateVerdict(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


@dataclass
class GateResult:
    passed: bool
    verdict: GateVerdict
    gate: str
    messages: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls, gate: str, *, messages: list[str] | None = None, **meta: Any) -> GateResult:
        return cls(
            passed=True,
            verdict=GateVerdict.PASS,
            gate=gate,
            messages=list(messages or []),
            metadata=dict(meta),
        )

    @classmethod
    def warn(
        cls,
        gate: str,
        messages: list[str],
        *,
        actions: list[str] | None = None,
        **meta: Any,
    ) -> GateResult:
        return cls(
            passed=True,
            verdict=GateVerdict.WARN,
            gate=gate,
            messages=messages,
            actions=list(actions or []),
            metadata=dict(meta),
        )

    @classmethod
    def block(
        cls,
        gate: str,
        messages: list[str],
        *,
        actions: list[str] | None = None,
        **meta: Any,
    ) -> GateResult:
        return cls(
            passed=False,
            verdict=GateVerdict.BLOCK,
            gate=gate,
            messages=messages,
            actions=list(actions or ["re_plan"]),
            metadata=dict(meta),
        )


@dataclass
class TruncationPolicy:
    """How PreflightProbe loads whitelisted context files for Executor."""

    tier: Literal["green", "yellow", "red"] = "green"
    max_chars_per_file: int = 12_000
    head_lines: int | None = None
    tail_lines: int | None = None
    line_slices: dict[str, tuple[int, int]] = field(default_factory=dict)

    @classmethod
    def green(cls, max_chars: int = 12_000) -> TruncationPolicy:
        return cls(tier="green", max_chars_per_file=max_chars)

    @classmethod
    def yellow(
        cls,
        *,
        head: int = 150,
        tail: int = 50,
        max_chars: int = 6_000,
        line_slices: dict[str, tuple[int, int]] | None = None,
    ) -> TruncationPolicy:
        return cls(
            tier="yellow",
            max_chars_per_file=max_chars,
            head_lines=head,
            tail_lines=tail,
            line_slices=dict(line_slices or {}),
        )

    @classmethod
    def red_fallback(cls) -> TruncationPolicy:
        """Last-resort: no preloaded files — Executor discovers via tools."""
        return cls(tier="red", max_chars_per_file=0, head_lines=None, tail_lines=None)


@dataclass
class PreflightResult:
    passed: bool
    verdict: GateVerdict
    policy: TruncationPolicy
    estimated_tokens: int
    budget_tokens: int
    messages: list[str] = field(default_factory=list)
    skip_preload: bool = False
