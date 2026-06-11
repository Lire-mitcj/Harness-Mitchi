from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.executor.final_output import parse_executor_final

ARTIFACT_STORE_SCHEMA = "mitkii.artifact_store.v1"
_SUPPORTED_KINDS = frozenset({"code_target", "database_view", "patch_intent"})
_MARKER_RE = re.compile(r"(?P<marker>EDIT_CONTEXT_JSON|PATCH_INTENT_JSON)\s*(?P<body>\{.*?\})(?=\n[A-Z_]+_JSON|\Z)", re.S)


@dataclass
class ArtifactRecord:
    """A shared evidence artifact. Confidence changes prompt priority, never gating."""

    kind: str
    canonical_id: str
    name: str = ""
    aliases: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    columns: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    producer: str = ""
    version: int = 1
    payload: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "canonical_id": self.canonical_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "evidence": list(self.evidence),
            "confidence": self.confidence,
            "producer": self.producer,
            "version": self.version,
        }
        if self.columns:
            out["columns"] = list(self.columns)
        if self.payload:
            out["payload"] = dict(self.payload)
        if self.conflicts:
            out["conflicts"] = list(self.conflicts)
        return out


def build_artifact_store(
    prior_summaries: dict[str, str],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    """Build a non-authoritative artifact slice from prior Executor outputs."""
    required_set = {item.strip() for item in (required or []) if item.strip()}
    records: dict[tuple[str, str], ArtifactRecord] = {}
    warnings: list[dict[str, Any]] = []

    for source_subtask, text in prior_summaries.items():
        for artifact in _extract_artifacts(source_subtask, text):
            if required_set and artifact.kind not in required_set and artifact.canonical_id not in required_set:
                continue
            key = (artifact.kind, artifact.canonical_id)
            existing = records.get(key)
            if existing is None:
                records[key] = artifact
                continue
            _merge_artifact(existing, artifact)

    for artifact in records.values():
        for conflict in artifact.conflicts:
            warnings.append({
                "kind": artifact.kind,
                "canonical_id": artifact.canonical_id,
                "conflict": conflict,
                "hint": "Verify this field locally before editing; artifacts are evidence hints only.",
            })

    return {
        "schema": ARTIFACT_STORE_SCHEMA,
        "policy": {
            "artifacts_are_hints": True,
            "may_not_block_edit": True,
        },
        "artifacts": [artifact.to_dict() for artifact in records.values()],
        "warnings": _dedupe_dicts(warnings, limit=20),
    }


def _extract_artifacts(source_subtask: str, text: str) -> list[ArtifactRecord]:
    out: list[ArtifactRecord] = []
    parsed = parse_executor_final(text)
    if parsed is not None:
        handoff = parsed.handoff or {}
        raw_artifacts = handoff.get("artifacts") or handoff.get("artifact_updates") or []
        if isinstance(raw_artifacts, list):
            for raw in raw_artifacts:
                artifact = _artifact_from_mapping(raw, source_subtask)
                if artifact is not None:
                    out.append(artifact)
        out.extend(_artifacts_from_evidence(parsed.evidence, source_subtask))
        out.extend(_artifacts_from_mapping_hints(handoff, source_subtask))

    for marker, payload in _extract_marker_json(text):
        out.extend(_artifacts_from_marker(marker, payload, source_subtask))
    return out


def _artifact_from_mapping(raw: object, source_subtask: str) -> ArtifactRecord | None:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or raw.get("artifact_type") or "").strip()
    if kind not in _SUPPORTED_KINDS:
        return None
    name = str(raw.get("name") or raw.get("canonical_id") or "").strip()
    canonical_id = str(raw.get("canonical_id") or _default_canonical_id(kind, name)).strip()
    if not canonical_id:
        return None
    confidence = _float_between(raw.get("confidence"), default=0.5)
    return ArtifactRecord(
        kind=kind,
        canonical_id=canonical_id,
        name=name,
        aliases=_string_list(raw.get("aliases")),
        evidence=_dict_list(raw.get("evidence")),
        columns=_dict_list(raw.get("columns")),
        confidence=confidence,
        producer=str(raw.get("producer") or source_subtask),
        version=int(raw.get("version")) if isinstance(raw.get("version"), int) else 1,
        payload={
            str(k): v
            for k, v in raw.items()
            if k
            not in {
                "kind",
                "artifact_type",
                "canonical_id",
                "name",
                "aliases",
                "evidence",
                "columns",
                "confidence",
                "producer",
                "version",
            }
        },
    )


def _artifacts_from_evidence(
    evidence: list[dict[str, Any]],
    source_subtask: str,
) -> list[ArtifactRecord]:
    out: list[ArtifactRecord] = []
    for item in evidence:
        path = str(item.get("path") or item.get("file") or "").strip()
        if not path:
            continue
        symbol = str(item.get("symbol") or "").strip()
        line = item.get("line") or item.get("line_range") or item.get("lines")
        canonical = f"code_target:{path}:{symbol or line or 'candidate'}"
        out.append(
            ArtifactRecord(
                kind="code_target",
                canonical_id=canonical,
                name=symbol or path,
                evidence=[{"source_subtask": source_subtask, **item}],
                confidence=0.65,
                producer=source_subtask,
            )
        )
    return out


def _artifacts_from_mapping_hints(
    data: dict[str, Any],
    source_subtask: str,
) -> list[ArtifactRecord]:
    out: list[ArtifactRecord] = []
    target_view = str(data.get("target_view") or "").strip()
    if target_view:
        out.append(_database_view_artifact(target_view, source_subtask, data))
    for view in _dict_list(data.get("available_views")):
        name = str(view.get("name") or view.get("view") or "").strip()
        if name:
            out.append(_database_view_artifact(name, source_subtask, view))
    if data.get("patch_intent") or data.get("editable_targets") or data.get("snippets"):
        out.append(
            ArtifactRecord(
                kind="patch_intent",
                canonical_id=f"patch_intent:{source_subtask}",
                name=f"patch intent from {source_subtask}",
                evidence=_dict_list(data.get("snippets")),
                confidence=_float_between(data.get("confidence"), default=0.6),
                producer=source_subtask,
                payload={
                    key: data[key]
                    for key in ("patch_intent", "editable_targets", "target_view")
                    if key in data
                },
            )
        )
    return out


def _artifacts_from_marker(
    marker: str,
    payload: dict[str, Any],
    source_subtask: str,
) -> list[ArtifactRecord]:
    if marker == "PATCH_INTENT_JSON":
        return [
            ArtifactRecord(
                kind="patch_intent",
                canonical_id=f"patch_intent:{source_subtask}",
                name=f"patch intent from {source_subtask}",
                confidence=_float_between(payload.get("confidence"), default=0.6),
                producer=source_subtask,
                payload=payload,
            )
        ]
    return _artifacts_from_mapping_hints(payload, source_subtask)


def _database_view_artifact(
    name: str,
    source_subtask: str,
    data: dict[str, Any],
) -> ArtifactRecord:
    return ArtifactRecord(
        kind="database_view",
        canonical_id=_default_canonical_id("database_view", name),
        name=name,
        aliases=_string_list(data.get("aliases")),
        evidence=_dict_list(data.get("evidence")),
        columns=_dict_list(data.get("columns") or data.get("fields")),
        confidence=_float_between(data.get("confidence"), default=0.65),
        producer=source_subtask,
        payload={
            key: data[key]
            for key in ("source", "schema", "target_view")
            if key in data
        },
    )


def _merge_artifact(existing: ArtifactRecord, incoming: ArtifactRecord) -> None:
    existing.aliases = _dedupe_strings([*existing.aliases, *incoming.aliases])
    existing.evidence = _dedupe_dicts([*existing.evidence, *incoming.evidence], limit=20)
    existing.columns = _merge_columns(existing, incoming.columns)
    existing.confidence = max(existing.confidence, incoming.confidence)
    existing.version = max(existing.version, incoming.version)
    existing.payload = {**incoming.payload, **existing.payload}


def _merge_columns(
    artifact: ArtifactRecord,
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {_column_key(col): dict(col) for col in artifact.columns if _column_key(col)}
    for col in incoming:
        key = _column_key(col)
        if not key:
            continue
        current = by_key.get(key)
        if current is None:
            by_key[key] = dict(col)
            continue
        current_type = str(current.get("type") or "unknown")
        incoming_type = str(col.get("type") or "unknown")
        if (
            current_type != incoming_type
            and current_type != "unknown"
            and incoming_type != "unknown"
        ):
            artifact.conflicts.append({
                "column": key,
                "field": "type",
                "values": [current_type, incoming_type],
            })
        observed = _dedupe_strings(
            _string_list(current.get("observed_names")) + _string_list(col.get("observed_names"))
        )
        if observed:
            current["observed_names"] = observed
        evidence = _dedupe_strings(
            _string_list(current.get("evidence_refs")) + _string_list(col.get("evidence_refs"))
        )
        if evidence:
            current["evidence_refs"] = evidence
    return list(by_key.values())


def _extract_marker_json(text: str) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    for match in _MARKER_RE.finditer(text.strip()):
        try:
            parsed = json.loads(match.group("body"))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            out.append((match.group("marker"), parsed))
    return out


def _default_canonical_id(kind: str, name: str) -> str:
    cleaned = name.strip().lower()
    return f"{kind}:{cleaned}" if cleaned else ""


def _column_key(column: dict[str, Any]) -> str:
    return str(column.get("canonical_name") or column.get("name") or "").strip().lower()


def _string_list(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _dict_list(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _dedupe_strings(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _dedupe_dicts(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _float_between(raw: object, *, default: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, value))
