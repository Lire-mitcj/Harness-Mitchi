from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExecutorFinal:
    result: str
    acceptance_met: bool | None = None
    evidence: list[dict[str, str]] = field(default_factory=list)
    blocker: str = ""
    raw: dict[str, Any] | None = None


def parse_executor_final(text: str | None) -> ExecutorFinal | None:
    """Parse the preferred Executor final JSON object.

    Returns None for legacy/plain-text finals so callers can keep compatibility.
    """
    if not text:
        return None
    parsed = _loads_json_object(text.strip())
    if parsed is None:
        return None

    result = str(parsed.get("result") or parsed.get("summary") or "").strip()
    blocker = str(parsed.get("blocker") or "").strip()
    acceptance_raw = parsed.get("acceptance_met")
    acceptance_met = acceptance_raw if isinstance(acceptance_raw, bool) else None
    evidence = _normalize_evidence(parsed.get("evidence"))
    return ExecutorFinal(
        result=result,
        acceptance_met=acceptance_met,
        evidence=evidence,
        blocker=blocker,
        raw=parsed,
    )


def normalize_executor_final_text(text: str | None) -> tuple[str, dict[str, Any] | None]:
    structured = parse_executor_final(text)
    if structured is None:
        return (text or ""), None
    return format_executor_final(structured), structured.raw


def format_executor_final(final: ExecutorFinal) -> str:
    lines = [f"Result: {final.result or '(no result provided)'}"]
    if final.evidence:
        lines.append("Evidence:")
        for item in final.evidence:
            parts: list[str] = []
            location = item.get("location") or _location(item)
            if location:
                parts.append(location)
            symbol = item.get("symbol", "")
            if symbol:
                parts.append(symbol)
            snippet = item.get("snippet", "")
            if snippet:
                parts.append(snippet)
            reason = item.get("reason", "")
            if reason:
                parts.append(reason)
            lines.append("- " + " | ".join(parts))
    elif final.blocker:
        lines.append("Evidence: (none)")
    conclusion = (
        "acceptance met"
        if final.acceptance_met is True
        else "acceptance not met"
        if final.acceptance_met is False
        else "acceptance unknown"
    )
    if final.blocker:
        conclusion += f"; blocker: {final.blocker}"
    lines.append(f"Conclusion: {conclusion}.")
    return "\n".join(lines)


def _loads_json_object(text: str) -> dict[str, Any] | None:
    candidate = text
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        candidate = "\n".join(lines).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_evidence(raw: object) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append({"snippet": text})
            continue
        if not isinstance(item, dict):
            continue
        normalized = {
            str(key): str(value).strip()
            for key, value in item.items()
            if value is not None and str(value).strip()
        }
        if normalized:
            out.append(normalized)
    return out


def _location(item: dict[str, str]) -> str:
    path = item.get("path", "")
    line = item.get("line") or item.get("line_range") or item.get("lines") or ""
    if path and line:
        return f"{path}:{line}"
    return path
