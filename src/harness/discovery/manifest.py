from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class VictimFile:
    path: str
    lines: list[int] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "lines": list(self.lines), "note": self.note}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VictimFile:
        raw_lines = data.get("lines") or []
        lines = [int(x) for x in raw_lines if isinstance(x, (int, float, str)) and str(x).isdigit()]
        return cls(
            path=str(data.get("path") or ""),
            lines=lines,
            note=str(data.get("note") or ""),
        )


@dataclass
class FileSnippet:
    path: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "content": self.content}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileSnippet:
        return cls(
            path=str(data.get("path") or ""),
            content=str(data.get("content") or ""),
        )


@dataclass
class DiagnosticsManifest:
    """Structured discovery output from the Scout phase."""

    user_request: str
    root_cause: str | None = None
    error_evidence: list[str] = field(default_factory=list)
    victim_files: list[VictimFile] = field(default_factory=list)
    repro_commands: list[str] = field(default_factory=list)
    file_snippets: list[FileSnippet] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    scout_turns_used: int = 0
    skipped: bool = False

    @classmethod
    def skipped_manifest(cls, user_request: str, project_structure: str) -> DiagnosticsManifest:
        return cls(
            user_request=user_request,
            root_cause=None,
            uncertainties=["Discovery skipped (/plan direct mode or scout disabled)."],
            file_snippets=[
                FileSnippet(path="__project_structure__", content=project_structure[:8000])
            ],
            skipped=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_request": self.user_request,
            "root_cause": self.root_cause,
            "error_evidence": list(self.error_evidence),
            "victim_files": [v.to_dict() for v in self.victim_files],
            "repro_commands": list(self.repro_commands),
            "file_snippets": [s.to_dict() for s in self.file_snippets],
            "uncertainties": list(self.uncertainties),
            "scout_turns_used": self.scout_turns_used,
            "skipped": self.skipped,
        }

    def to_planner_block(self, *, compact: bool = True) -> str:
        lines = ["<discovery_manifest>"]
        if self.skipped:
            lines.append("Discovery: SKIPPED (direct /plan mode)")
        else:
            lines.append(f"Discovery: {self.scout_turns_used} scout turn(s)")
        if self.root_cause:
            lines.append(f"Root cause: {self.root_cause[:200]}")
        if self.error_evidence:
            lines.append("Error evidence:")
            limit = 3 if compact else 5
            lines.extend(f"  - {e[:160]}" for e in self.error_evidence[:limit])
        if self.victim_files:
            lines.append("Victim files:")
            for v in self.victim_files[:3]:
                loc = f" lines {v.lines}" if v.lines else ""
                note = f" — {v.note[:60]}" if v.note else ""
                lines.append(f"  - {v.path}{loc}{note}")
        if not compact and self.repro_commands:
            lines.append("Repro commands:")
            lines.extend(f"  $ {c}" for c in self.repro_commands[:3])
        if not compact and self.file_snippets:
            lines.append("File snippets:")
            for snip in self.file_snippets[:2]:
                if snip.path == "__project_structure__":
                    continue
                body = snip.content[:800 if compact else 4000]
                lines.append(f'\n<file path="{snip.path}">\n{body}\n</file>')
        if self.uncertainties:
            lines.append("Uncertainties:")
            lines.extend(f"  - {u[:120]}" for u in self.uncertainties[:2])
        lines.append("</discovery_manifest>")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiagnosticsManifest:
        return cls(
            user_request=str(data.get("user_request") or ""),
            root_cause=data.get("root_cause"),
            error_evidence=[str(x) for x in data.get("error_evidence") or []],
            victim_files=[
                VictimFile.from_dict(v)
                for v in data.get("victim_files") or []
                if isinstance(v, dict)
            ],
            repro_commands=[str(x) for x in data.get("repro_commands") or []],
            file_snippets=[
                FileSnippet.from_dict(s)
                for s in data.get("file_snippets") or []
                if isinstance(s, dict)
            ],
            uncertainties=[str(x) for x in data.get("uncertainties") or []],
            scout_turns_used=int(data.get("scout_turns_used") or 0),
            skipped=bool(data.get("skipped")),
        )


def extract_discovery_trace(raw: str) -> str | None:
    """Return CoT block from Scout output, if present."""
    match = re.search(
        r"<discovery_trace>\s*(.*?)\s*</discovery_trace>",
        raw,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip() or None


def _strip_discovery_trace(text: str) -> str:
    """Remove CoT preamble; tolerate malformed Scout tags."""
    stripped = text.strip()
    stripped = re.sub(
        r"<discovery_trace>\s*.*?\s*</discovery_trace>",
        "",
        stripped,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if re.search(r"<discovery_trace>", stripped, re.IGNORECASE):
        stripped = re.sub(
            r"<discovery_trace>.*?(?=\{)",
            "",
            stripped,
            count=1,
            flags=re.DOTALL | re.IGNORECASE,
        )
    stripped = re.sub(r"</?discovery_trace[^>]*>", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _is_displayable_root_cause(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.lower()
    if any(token in lowered for token in ("discovery_trace", "planning_trace", "```")):
        return False
    if "<" in value or value.count("\n") > 1:
        return False
    return len(value.strip()) <= 200


def discovery_display_summary(manifest: DiagnosticsManifest) -> str:
    """One-line CLI-safe discovery summary (never dumps raw CoT)."""
    turns = manifest.scout_turns_used
    if turns == 0 and manifest.victim_files:
        head = "Discovery complete (preflight grep)"
    else:
        head = f"Discovery complete ({turns} turn{'s' if turns != 1 else ''})"
    if _is_displayable_root_cause(manifest.root_cause):
        return f"{head} — {manifest.root_cause}"
    if manifest.victim_files:
        paths = ", ".join(v.path for v in manifest.victim_files[:3] if v.path)
        if paths:
            return f"{head} — files: {paths}"
    if manifest.error_evidence:
        snippet = summarize_manifest_line(manifest.error_evidence[0])
        return f"{head} — {snippet}"
    if manifest.uncertainties:
        return f"{head} — {summarize_manifest_line(manifest.uncertainties[0])}"
    return head


def summarize_manifest_line(text: str, *, max_len: int = 100) -> str:
    one_line = " ".join(text.split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1] + "…"


def _repair_json_blob(text: str) -> str:
    """Best-effort fixes for common LLM JSON mistakes."""
    fixed = text.strip()
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    fixed = fixed.replace("\u201c", '"').replace("\u201d", '"')
    return fixed


def _load_manifest_dict(text: str) -> dict[str, Any] | None:
    for candidate in (text, _repair_json_blob(text)):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def parse_manifest_json(raw: str, *, user_request: str) -> DiagnosticsManifest:
    text = _strip_discovery_trace(raw.strip())
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    data = _load_manifest_dict(text)
    if data is None:
        return DiagnosticsManifest(
            user_request=user_request,
            root_cause=None,
            uncertainties=["Manifest JSON parse failed; Planner will rely on project context."],
        )
    if not isinstance(data, dict):
        return DiagnosticsManifest(
            user_request=user_request,
            uncertainties=["Invalid manifest payload."],
        )
    data.setdefault("user_request", user_request)
    manifest = DiagnosticsManifest.from_dict(data)
    if manifest.root_cause and not _is_displayable_root_cause(manifest.root_cause):
        manifest.uncertainties = list(manifest.uncertainties) + [
            "root_cause from Scout looked malformed; ignored for display.",
        ]
        manifest.root_cause = None
    return _trim_manifest(manifest)


def _trim_manifest(manifest: DiagnosticsManifest) -> DiagnosticsManifest:
    """Enforce Scout output size limits before Planner consumes manifest."""
    if manifest.root_cause and len(manifest.root_cause) > 200:
        manifest.root_cause = manifest.root_cause[:200] + "…"
    manifest.error_evidence = manifest.error_evidence[:5]
    manifest.victim_files = manifest.victim_files[:3]
    manifest.repro_commands = manifest.repro_commands[:3]
    manifest.uncertainties = manifest.uncertainties[:2]
    trimmed_snippets: list[FileSnippet] = []
    for snip in manifest.file_snippets[:2]:
        content = snip.content
        if len(content) > 1500:
            content = content[:1500] + "\n[truncated by harness]"
        trimmed_snippets.append(FileSnippet(path=snip.path, content=content))
    manifest.file_snippets = trimmed_snippets
    return manifest


def manifest_actionable(manifest: DiagnosticsManifest) -> bool:
    """Enough Scout signal to skip Executor diagnose subtasks."""
    if manifest.skipped:
        return False
    if _is_displayable_root_cause(manifest.root_cause):
        return True
    if manifest.victim_files:
        return True
    if manifest.error_evidence and len(manifest.error_evidence) >= 2:
        return True
    if manifest.file_snippets and manifest.victim_files:
        return True
    return False
