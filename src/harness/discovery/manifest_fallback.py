from __future__ import annotations

import json
import re

from src.harness.discovery.manifest import (
    DiagnosticsManifest,
    VictimFile,
    extract_discovery_trace,
    manifest_actionable,
)

_GREP_PATH = re.compile(
    r"^\.?/?([\w./-]+\.(?:py|sql|js|ts|go|java|rb|yml|yaml|xml|proto))(?::\d+)?:",
    re.M,
)
_READ_FILES_HEADER = re.compile(
    r"^===== ([\w./-]+) \(\d+ lines\) =====",
    re.M,
)
_JSON_PATH_LINES = re.compile(
    r'"path"\s*:\s*"([^"]+)"\s*,\s*"lines"\s*:\s*\[([^\]]*)\]',
    re.I,
)
_VICTIM_TEXT = re.compile(
    r"([\w./-]+\.(?:py|sql|js|ts))\s*(?:lines?|:)\s*(\d+)",
    re.I,
)
_QUOTED_EVIDENCE = re.compile(r'["\']([^"\']{8,120})["\']')
_CJK_EVIDENCE = re.compile(r"[\u4e00-\u9fff]{4,40}")


def manifest_reject_reason(manifest: DiagnosticsManifest) -> str:
    if any("parse failed" in u.lower() for u in manifest.uncertainties):
        return "json_parse_failed"
    if manifest.root_cause and not manifest.victim_files and not manifest.error_evidence:
        if "discovery_trace" in (manifest.root_cause or "").lower():
            return "root_cause_is_trace_not_json"
    if not manifest.root_cause and not manifest.victim_files and not manifest.error_evidence:
        return "empty_manifest"
    return "not_actionable"


def _collect_victims_from_text(text: str) -> dict[str, list[int]]:
    victims: dict[str, list[int]] = {}
    for match in _GREP_PATH.finditer(text):
        path = match.group(1).replace("\\", "/").lstrip("./")
        line_m = re.search(r":(\d+):", match.group(0))
        line = int(line_m.group(1)) if line_m else 0
        victims.setdefault(path, [])
        if line and line not in victims[path]:
            victims[path].append(line)
    for match in _READ_FILES_HEADER.finditer(text):
        victims.setdefault(match.group(1).replace("\\", "/").lstrip("./"), [])
    for match in _JSON_PATH_LINES.finditer(text):
        path = match.group(1).replace("\\", "/").lstrip("./")
        lines_raw = match.group(2)
        lines = [int(x) for x in re.findall(r"\d+", lines_raw)]
        victims.setdefault(path, [])
        for line in lines:
            if line not in victims[path]:
                victims[path].append(line)
    for match in _VICTIM_TEXT.finditer(text):
        path = match.group(1).replace("\\", "/").lstrip("./")
        line = int(match.group(2))
        victims.setdefault(path, [])
        if line not in victims[path]:
            victims[path].append(line)
    return victims


def _collect_evidence_from_text(text: str, user_request: str) -> list[str]:
    evidence: list[str] = []
    for match in _QUOTED_EVIDENCE.finditer(text):
        snippet = match.group(1).strip()
        if any(k in snippet for k in ("事务", "register", "transaction", "error", "Error", "异常")):
            evidence.append(snippet[:160])
    for match in _CJK_EVIDENCE.finditer(text):
        snippet = match.group(0)
        if any(k in snippet for k in ("事务", "注册", "异常", "数据库", "接口")):
            evidence.append(snippet)
    if "事务" in user_request and not evidence:
        evidence.append(user_request[:120])
    return list(dict.fromkeys(evidence))[:5]


def extract_manifest_from_raw(
    raw: str,
    *,
    user_request: str,
    preflight_grep: str = "",
) -> DiagnosticsManifest | None:
    """Parse victim paths and evidence from malformed Scout text or preflight grep."""
    trace = extract_discovery_trace(raw) or ""
    combined = "\n".join(part for part in (preflight_grep, trace, raw) if part)
    victims_map = _collect_victims_from_text(combined)
    evidence = _collect_evidence_from_text(combined, user_request)

    if not victims_map and not evidence and not preflight_grep:
        return None

    victims = [
        VictimFile(path=p, lines=sorted(lines)[:5], note="extracted from scout text")
        for p, lines in list(victims_map.items())[:3]
    ]
    root = None
    for e in evidence:
        if any(k in e.lower() for k in ("transaction", "事务", "error", "exception", "异常")):
            root = e[:120]
            break
    if not root and victims:
        root = f"Issue likely in {victims[0].path}"
    elif not root and "注册" in user_request:
        root = "Registration endpoint issue (see victim_files)"

    manifest = DiagnosticsManifest(
        user_request=user_request,
        root_cause=root,
        error_evidence=evidence,
        victim_files=victims,
        uncertainties=["Manifest JSON invalid — harness extracted fields from text/grep."],
    )
    if not manifest_actionable(manifest):
        return None
    return manifest


def build_manifest_from_tool_history(
    messages: list,
    *,
    user_request: str,
    preflight_grep: str = "",
) -> DiagnosticsManifest | None:
    """Synthesize a minimal manifest from Scout tool outputs when JSON fails."""
    victim_paths: dict[str, list[int]] = {}
    evidence: list[str] = []

    if preflight_grep:
        victim_paths.update(_collect_victims_from_text(preflight_grep))
        evidence.extend(_collect_evidence_from_text(preflight_grep, user_request))

    for msg in messages:
        role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        if role == "system" and isinstance(content, str) and "<scout_preflight_grep>" in content:
            victim_paths.update(_collect_victims_from_text(content))
            evidence.extend(_collect_evidence_from_text(content, user_request))
            continue
        if role != "tool":
            if role == "assistant" and isinstance(content, str):
                victim_paths.update(_collect_victims_from_text(content))
                evidence.extend(_collect_evidence_from_text(content, user_request))
            continue
        if not isinstance(content, str) or not content.strip():
            continue
        if content.startswith("Error:"):
            evidence.append(content[:200])
            continue
        victim_paths.update(_collect_victims_from_text(content))
        evidence.extend(_collect_evidence_from_text(content, user_request))

    if not victim_paths and not evidence:
        return None

    victims = [
        VictimFile(path=p, lines=sorted(lines)[:5], note="from scout tools")
        for p, lines in list(victim_paths.items())[:3]
    ]
    root = None
    for e in evidence:
        if any(k in e.lower() for k in ("transaction", "事务", "error", "exception", "异常")):
            root = e[:120]
            break
    if not root and victims:
        root = f"Issue likely in {victims[0].path}"

    manifest = DiagnosticsManifest(
        user_request=user_request,
        root_cause=root,
        error_evidence=list(dict.fromkeys(evidence))[:5],
        victim_files=victims,
        uncertainties=["Manifest JSON failed — harness synthesized this from tool results."],
    )
    if not manifest_actionable(manifest):
        return None
    return manifest


def try_parse_manifest_with_fallback(
    raw: str,
    *,
    user_request: str,
    messages: list | None = None,
    preflight_grep: str = "",
) -> DiagnosticsManifest:
    from src.harness.discovery.manifest import parse_manifest_json

    manifest = parse_manifest_json(raw, user_request=user_request)
    if manifest_actionable(manifest):
        return manifest

    extracted = extract_manifest_from_raw(
        raw, user_request=user_request, preflight_grep=preflight_grep
    )
    if extracted is not None:
        return extracted

    if messages:
        fallback = build_manifest_from_tool_history(
            messages, user_request=user_request, preflight_grep=preflight_grep
        )
        if fallback is not None:
            return fallback

    if preflight_grep:
        fallback = build_manifest_from_tool_history(
            [], user_request=user_request, preflight_grep=preflight_grep
        )
        if fallback is not None:
            return fallback

    return manifest
