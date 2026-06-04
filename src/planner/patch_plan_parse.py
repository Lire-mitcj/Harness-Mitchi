from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.planner.patch_plan import PatchEdit, PatchPlan


@dataclass(frozen=True)
class PatchPlanParseResult:
    raw: str
    payload: dict[str, Any]
    patch_plan: PatchPlan | None = None
    json_ok: bool = False
    schema_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.json_ok and self.patch_plan is not None and not self.schema_errors

    @property
    def all_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.json_ok:
            errors.append("PatchPlan output must be one raw JSON object.")
        errors.extend(self.schema_errors)
        return errors


def parse_patch_plan_output(raw: str) -> PatchPlanParseResult:
    payload, json_ok = _extract_json_payload(raw)
    if not json_ok:
        return PatchPlanParseResult(raw=raw, payload=payload, json_ok=False)
    errors = validate_patch_plan_payload(payload)
    patch_plan = build_patch_plan_from_payload(payload) if not errors else None
    return PatchPlanParseResult(
        raw=raw,
        payload=payload,
        patch_plan=patch_plan,
        json_ok=True,
        schema_errors=errors,
    )


def validate_patch_plan_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    plan = payload.get("patch_plan")
    if not isinstance(plan, dict):
        if isinstance(payload.get("nodes"), list):
            return ["received TaskTree/nodes JSON; expected patch_plan object."]
        return ["patch_plan must be an object."]

    for field_name in ("files_to_edit", "target_symbols", "intended_changes"):
        value = plan.get(field_name)
        if value is not None and not _is_string_list(value):
            errors.append(f"patch_plan.{field_name} must be a string array.")

    edits = plan.get("edits")
    if not isinstance(edits, list) or not edits:
        errors.append("patch_plan.edits must contain at least one edit object.")
    elif isinstance(edits, list):
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                errors.append(f"patch_plan.edits[{idx}] must be an object.")
                continue
            for key in ("path", "old_string", "new_string"):
                if not isinstance(edit.get(key), str) or not edit.get(key):
                    errors.append(f"patch_plan.edits[{idx}].{key} must be a non-empty string.")
            if edit.get("old_string") == edit.get("new_string"):
                errors.append(f"patch_plan.edits[{idx}] old_string and new_string must differ.")

    confidence = plan.get("confidence", 0.0)
    if not isinstance(confidence, int | float):
        errors.append("patch_plan.confidence must be a number.")
    elif not 0 <= float(confidence) <= 1:
        errors.append("patch_plan.confidence must be between 0 and 1.")

    if plan.get("requires_confirmation") is not None and not isinstance(
        plan.get("requires_confirmation"),
        bool,
    ):
        errors.append("patch_plan.requires_confirmation must be boolean.")
    if plan.get("missing_info") is not None and not _is_string_list(plan.get("missing_info")):
        errors.append("patch_plan.missing_info must be a string array.")
    if plan.get("validation_plan") is not None and not _is_string_list(
        plan.get("validation_plan")
    ):
        errors.append("patch_plan.validation_plan must be a string array.")
    return errors


def build_patch_plan_from_payload(payload: dict[str, Any]) -> PatchPlan:
    plan = payload.get("patch_plan") if isinstance(payload.get("patch_plan"), dict) else {}
    edits = tuple(
        PatchEdit(
            path=str(edit.get("path") or ""),
            old_string=str(edit.get("old_string") or ""),
            new_string=str(edit.get("new_string") or ""),
            symbol=str(edit.get("symbol") or ""),
        )
        for edit in plan.get("edits") or []
        if isinstance(edit, dict)
    )
    files = _string_tuple(plan.get("files_to_edit"))
    if not files:
        files = tuple(dict.fromkeys(edit.path for edit in edits if edit.path))
    return PatchPlan(
        files_to_edit=files,
        target_symbols=_string_tuple(plan.get("target_symbols")),
        intended_changes=_string_tuple(plan.get("intended_changes")),
        edits=edits,
        validation_plan=_string_tuple(plan.get("validation_plan")),
        requires_confirmation=bool(plan.get("requires_confirmation", False)),
        confidence=float(plan.get("confidence", 0.0)),
        missing_info=_string_tuple(plan.get("missing_info")),
        metadata={"source": "planner_patch_plan"},
    )


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


def _is_string_list(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _extract_json_payload(raw: str) -> tuple[dict[str, Any], bool]:
    text = raw.strip()
    fence_start = text.find("```")
    if fence_start >= 0:
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}, False
    return (payload, True) if isinstance(payload, dict) else ({}, False)
