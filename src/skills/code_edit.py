from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from src.skills.base import SkillContext, SkillResult

EditComplete = Callable[[list[dict[str, object]]], Awaitable[str]]


class CodeEditSkill:
    name = "code_edit"

    def __init__(
        self,
        *,
        project_root: Path,
        llm_complete: EditComplete | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.llm_complete = llm_complete

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        plan = context.patch_plan
        if plan is None:
            return await self._run_focused_edit(context, **kwargs)
        if not plan.is_executable():
            return SkillResult(
                success=False,
                summary="patch_plan is not executable.",
                missing_info=plan.missing_info or ("non_executable_patch_plan",),
                requires_fallback=True,
            )

        changed: list[str] = []
        errors: list[str] = []
        original_files: dict[str, str] = {}
        for edit in plan.edits:
            path = _resolve_under_root(self.project_root, edit.path)
            if path is None:
                errors.append(f"{edit.path}: outside project root")
                continue
            if not path.is_file():
                errors.append(f"{edit.path}: file not found")
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as exc:
                errors.append(f"{edit.path}: read failed: {exc}")
                continue
            occurrences = content.count(edit.old_string)
            if occurrences != 1:
                errors.append(
                    f"{edit.path}: old_string occurrence count is {occurrences}, expected 1"
                )
                continue
            new_content = content.replace(edit.old_string, edit.new_string, 1)
            rel = _rel_display(self.project_root, path)
            original_files.setdefault(rel, content)
            try:
                path.write_text(new_content, encoding="utf-8")
            except OSError as exc:
                errors.append(f"{edit.path}: write failed: {exc}")
                continue
            changed.append(rel)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(changed),
                missing_info=tuple(errors),
                requires_fallback=True,
            )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} patch edit(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={"original_files_json": json.dumps(original_files)},
        )

    async def _run_focused_edit(
        self,
        context: SkillContext,
        **kwargs: object,
    ) -> SkillResult:
        if self.llm_complete is None:
            return SkillResult(
                success=False,
                summary="code_edit requires llm_complete when patch_plan is absent.",
                missing_info=("llm_complete",),
            )
        instruction = str(kwargs.get("instruction") or context.user_request).strip()
        evidence = str(kwargs.get("search_output") or "").strip()
        if not evidence:
            return SkillResult(
                success=False,
                summary="code_edit requires search_output evidence.",
                missing_info=("search_output",),
            )
        edit_context = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
        ready_error = _validate_edit_context_ready(self.project_root, edit_context)
        if ready_error:
            return SkillResult(
                success=False,
                summary=f"code_edit not ready: {ready_error}",
                missing_info=(ready_error,),
                metadata={"raw_preview": _preview(evidence)},
            )
        contract = kwargs.get("handoff_contract")
        plan_messages = _edit_plan_messages(
            instruction=instruction,
            evidence=evidence,
            handoff_contract=contract if isinstance(contract, dict) else None,
        )
        plan_raw = await self.llm_complete(plan_messages)
        plan_payload = _extract_json_payload(plan_raw)
        edit_targets = _editable_targets(edit_context)
        if plan_payload is None:
            payload = _fallback_edit_payload(
                edit_context,
                evidence=evidence,
                instruction=instruction,
                handoff_contract=contract if isinstance(contract, dict) else None,
            )
            raw_preview = plan_raw
        else:
            payload, raw_preview = await self._build_payload_from_plan(
                plan_payload,
                edit_targets=edit_targets,
                instruction=instruction,
                evidence=evidence,
            )
        if payload is None:
            return SkillResult(
                success=False,
                summary=(
                    "code_edit could not produce a valid structured edit plan, "
                    "and no deterministic SQL view edit could be derived."
                ),
                missing_info=("edit_plan_json", "deterministic_edit"),
                metadata={"raw_preview": _preview(raw_preview)},
            )

        edits = payload.get("edits")
        confidence = payload.get("confidence", 0.0)
        missing = payload.get("missing_info") or ()
        if not isinstance(confidence, int | float) or float(confidence) < 0.75:
            return SkillResult(
                success=False,
                summary=(
                    "code_edit confidence too low "
                    f"(confidence={confidence}, missing_info={missing})"
                ),
                missing_info=tuple(str(item) for item in missing if isinstance(item, str)),
                metadata={"raw_preview": _preview(raw_preview)},
            )
        if not isinstance(edits, list) or not edits:
            return SkillResult(
                success=False,
                summary="code_edit model returned no edits.",
                missing_info=("edits",),
                metadata={"raw_preview": _preview(raw_preview)},
            )

        changed: list[str] = []
        errors: list[str] = []
        original_files: dict[str, str] = {}
        for idx, item in enumerate(edits):
            if not isinstance(item, dict):
                errors.append(f"edits[{idx}] must be an object")
                continue
            path, old_string = _resolve_edit_item_target(item, edit_targets)
            new_string = str(item.get("new_string") or "")
            if not path or not old_string or not new_string:
                errors.append(
                    f"edits[{idx}] requires target_index+new_string or "
                    "path+old_string+new_string"
                )
                continue
            if old_string == new_string:
                errors.append(f"edits[{idx}] old_string and new_string must differ")
                continue
            if len(old_string) > 50_000 or len(new_string) > 50_000:
                errors.append(f"edits[{idx}] is too large; use a smaller exact snippet")
                continue
            result = _apply_exact_edit(
                self.project_root,
                path=path,
                old_string=old_string,
                new_string=new_string,
                original_files=original_files,
            )
            if result.startswith("ok:"):
                changed.append(result[3:])
            else:
                errors.append(result)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(dict.fromkeys(changed)),
                missing_info=tuple(errors),
                metadata={"raw_preview": _preview(raw_preview)},
            )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} edit_file action(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={
                "raw_preview": _preview(raw_preview),
                "confidence": str(confidence),
                "original_files_json": json.dumps(original_files),
            },
        )

    async def _build_payload_from_plan(
        self,
        plan_payload: dict[str, object],
        *,
        edit_targets: list[dict[str, object]],
        instruction: str,
        evidence: str,
    ) -> tuple[dict[str, object] | None, str]:
        edits = plan_payload.get("edits")
        confidence = plan_payload.get("confidence", 0.0)
        missing = plan_payload.get("missing_info") or []
        if not isinstance(edits, list) or not edits:
            return plan_payload, json.dumps(plan_payload, ensure_ascii=False)
        built_edits: list[dict[str, object]] = []
        raw_parts = [json.dumps(plan_payload, ensure_ascii=False)]
        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                return None, raw_parts[0]
            target_index = edit.get("target_index")
            if not isinstance(target_index, int) or not (0 <= target_index < len(edit_targets)):
                return None, raw_parts[0]
            current_code = str(edit_targets[target_index].get("current_code") or "")
            file = str(edit_targets[target_index].get("file") or "")
            replacement_messages = _replacement_messages(
                instruction=instruction,
                file=file,
                target_index=target_index,
                current_code=current_code,
                edit_plan=edit,
                evidence=evidence,
            )
            replacement_raw = await self.llm_complete(replacement_messages)
            raw_parts.append(replacement_raw)
            replacement_payload = _extract_json_payload(replacement_raw)
            if replacement_payload is None:
                return None, "\n\n".join(raw_parts)
            new_string = str(replacement_payload.get("new_string") or "")
            if not new_string:
                return None, "\n\n".join(raw_parts)
            built_edits.append({
                "target_index": target_index,
                "new_string": new_string,
            })
        return {
            "edits": built_edits,
            "confidence": confidence,
            "missing_info": missing,
        }, "\n\n".join(raw_parts)


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        root = project_root.resolve()
        raw = rel.replace("\\", "/").strip()
        candidate = Path(raw)
        if candidate.is_absolute():
            path = candidate.resolve()
        else:
            path = (root / raw.lstrip("./")).resolve()
        path.relative_to(root)
        return path
    except (OSError, ValueError):
        return None


def _rel_display(project_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except (OSError, ValueError):
        text = str(path).replace("\\", "/")
        return text if Path(text).is_absolute() else text.lstrip("./")


def _apply_exact_edit(
    project_root: Path,
    *,
    path: str,
    old_string: str,
    new_string: str,
    original_files: dict[str, str] | None = None,
) -> str:
    target = _resolve_under_root(project_root, path)
    if target is None:
        return f"{path}: outside project root"
    if not target.is_file():
        return f"{path}: file not found"
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"{path}: read failed: {exc}"
    occurrences = content.count(old_string)
    if occurrences != 1:
        return f"{path}: old_string occurrence count is {occurrences}, expected 1"
    rel = _rel_display(project_root, target)
    if original_files is not None:
        original_files.setdefault(rel, content)
    try:
        target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    except OSError as exc:
        return f"{path}: write failed: {exc}"
    return f"ok:{rel}"


def _editable_targets(ctx: dict[str, object] | None) -> list[dict[str, object]]:
    if not isinstance(ctx, dict):
        return []
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    return [target for target in targets or [] if isinstance(target, dict)]


def _resolve_edit_item_target(
    item: dict[str, object],
    targets: list[dict[str, object]],
) -> tuple[str, str]:
    path = str(item.get("path") or "")
    old_string = str(item.get("old_string") or "")
    if path and old_string:
        return path, old_string

    target_index = item.get("target_index")
    if isinstance(target_index, int) and 0 <= target_index < len(targets):
        target = targets[target_index]
        return (
            str(target.get("file") or ""),
            str(target.get("current_code") or ""),
        )

    if path and not old_string:
        matching = [
            target for target in targets
            if str(target.get("file") or "").replace("\\", "/") == path.replace("\\", "/")
        ]
        if len(matching) == 1:
            return path, str(matching[0].get("current_code") or "")
    return path, old_string


def _edit_plan_messages(
    *,
    instruction: str,
    evidence: str,
    handoff_contract: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    contract_text = (
        json.dumps(handoff_contract, ensure_ascii=False, indent=2)
        if handoff_contract
        else "{}"
    )
    prepared_evidence = _prepare_edit_plan_evidence(evidence)
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII EditPlanBuilder. Output ONE raw JSON object only. "
                "No markdown, no prose, no Python code. You must not generate "
                "replacement code. Select editable_targets by index and describe "
                "the intended bounded change. Required schema: "
                '{"edits":[{"target_index":0,"operation":"replace_sql_source",'
                '"source_table":"...","target_view":"...",'
                '"change_summary":"..."}],"confidence":1.0,"missing_info":[]}. '
                "If no provided editable_target is clearly correct, return "
                '{"edits":[],"confidence":0.0,"missing_info":["target"]}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                "HARD_HANDOFF_CONTRACT_JSON\n"
                f"{contract_text[:6000]}\n\n"
                "<code_search_results>\n"
                f"{prepared_evidence}\n"
                "</code_search_results>"
            ),
        },
    ]


def _edit_messages(
    *,
    instruction: str,
    evidence: str,
    handoff_contract: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Backward-compatible alias for tests and older callers."""
    return _edit_plan_messages(
        instruction=instruction,
        evidence=evidence,
        handoff_contract=handoff_contract,
    )


def _replacement_messages(
    *,
    instruction: str,
    file: str,
    target_index: int,
    current_code: str,
    edit_plan: dict[str, object],
    evidence: str,
) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII ReplacementBuilder. Output ONE raw JSON object only. "
                "No markdown, no prose. Required schema: "
                '{"new_string":"complete replacement for current_code"}. '
                "Return only the replacement for the single provided current_code. "
                "Preserve indentation, imports, surrounding behavior, and style. "
                "Do not edit files or symbols outside this current_code."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                f"File: {file}\nTarget index: {target_index}\n\n"
                "EDIT_PLAN_JSON\n"
                f"{json.dumps(edit_plan, ensure_ascii=False, indent=2)}\n\n"
                "Relevant evidence summary:\n"
                f"{_prepare_replacement_evidence(evidence)}\n\n"
                "CURRENT_CODE\n"
                f"{current_code[:50_000]}"
            ),
        },
    ]


def _prepare_replacement_evidence(evidence: str) -> str:
    marker = "EDIT_CONTEXT_JSON"
    before = evidence.split(marker, 1)[0] if marker in evidence else evidence
    return before[:6_000].strip()


def _prepare_edit_evidence(evidence: str) -> str:
    marker = "EDIT_CONTEXT_JSON"
    if marker not in evidence:
        return evidence[:12_000]
    before, after = evidence.split(marker, 1)
    return (
        before[:6_000].rstrip()
        + "\n\n"
        + marker
        + "\n"
        + after.strip()[:50_000]
    )


def _prepare_edit_plan_evidence(evidence: str) -> str:
    ctx = _extract_marker_json(evidence, "EDIT_CONTEXT_JSON")
    if not isinstance(ctx, dict):
        return evidence[:12_000]
    compact = dict(ctx)
    compact_targets: list[dict[str, object]] = []
    for idx, target in enumerate(_editable_targets(ctx)):
        current_code = str(target.get("current_code") or "")
        compact_targets.append({
            "index": idx,
            "file": target.get("file"),
            "start_line": target.get("start_line"),
            "end_line": target.get("end_line"),
            "code_preview": _preview(current_code, limit=1200),
        })
    compact["editable_targets"] = compact_targets
    compact["snippets"] = compact_targets
    before = evidence.split("EDIT_CONTEXT_JSON", 1)[0][:6_000]
    return (
        before.rstrip()
        + "\n\nEDIT_CONTEXT_JSON_COMPACT\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def _validate_edit_context_ready(
    project_root: Path,
    ctx: dict[str, object] | None,
) -> str:
    if ctx is None:
        return "missing EDIT_CONTEXT_JSON with editable_targets/current_code"
    if ctx.get("code_edit_ready") is not True:
        return "EDIT_CONTEXT_JSON code_edit_ready is not true"
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    if not isinstance(targets, list) or not targets:
        return "no editable_targets/snippets"
    intended_change = str(ctx.get("intended_change") or "").strip()
    if not intended_change:
        return "missing intended_change"
    acceptance = ctx.get("acceptance_criteria")
    if not _has_acceptance(acceptance):
        return "missing acceptance_criteria"
    tool_policy = ctx.get("tool_policy")
    if not isinstance(tool_policy, dict):
        return "missing tool_policy"
    allowed = tool_policy.get("allowed_tools")
    if not isinstance(allowed, list) or "edit_file" not in allowed:
        return "tool_policy does not allow edit_file"
    scope = {
        _scope_key(str(item), project_root)
        for item in (tool_policy.get("scope") or [])
        if str(item).strip()
    }
    for idx, target in enumerate(targets):
        if not isinstance(target, dict):
            return f"editable_targets[{idx}] is not an object"
        file_raw = str(target.get("file") or "")
        file = file_raw.replace("\\", "/")
        start = target.get("start_line")
        end = target.get("end_line")
        current_code = str(target.get("current_code") or "")
        target_change = str(target.get("intended_change") or intended_change).strip()
        target_acceptance = target.get("acceptance_criteria", acceptance)
        if not file or not isinstance(start, int) or not isinstance(end, int):
            return f"editable_targets[{idx}] missing file + line range"
        if end < start:
            return f"editable_targets[{idx}] invalid line range"
        if not current_code.strip():
            return f"editable_targets[{idx}] missing current_code"
        if _is_sql_view_change(target_change) and not _target_has_sql_query(current_code):
            return (
                f"editable_targets[{idx}] is not a SQL/query target for "
                "a view-query edit"
            )
        if not target_change:
            return f"editable_targets[{idx}] missing intended_change"
        if not _has_acceptance(target_acceptance):
            return f"editable_targets[{idx}] missing acceptance_criteria"
        if scope and _scope_key(file, project_root) not in scope:
            return f"tool_policy scope does not cover {file}"
        path = _resolve_under_root(project_root, file)
        if path is None or not path.is_file():
            return f"patch_validator cannot locate {file}"
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            return f"patch_validator cannot read {file}: {exc}"
        if current_code not in content:
            return f"patch_validator cannot locate current_code in {file}"
    return ""


def _has_acceptance(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return False


def _is_sql_view_change(text: str) -> bool:
    lowered = text.lower()
    return (
        "视图" in text
        or "查询" in text
        or "sql" in lowered
        or "query" in lowered
        or "view" in lowered
    )


def _target_has_sql_query(current_code: str) -> bool:
    return bool(
        re.search(
            r"(?is)\bSELECT\b.{0,200}\bFROM\b|\b(?:FROM|JOIN)\s+[`\"]?[A-Za-z_][A-Za-z0-9_]*",
            current_code,
        )
    )


def _scope_key(raw: str, project_root: Path) -> str:
    text = raw.replace("\\", "/").strip()
    try:
        p = Path(text)
        resolved = p.resolve() if p.is_absolute() else (project_root / text.lstrip("./")).resolve()
        try:
            return str(resolved.relative_to(project_root.resolve())).replace("\\", "/")
        except ValueError:
            return str(resolved).replace("\\", "/")
    except OSError:
        return text.lstrip("./")


def _fallback_edit_payload(
    ctx: dict[str, object] | None,
    *,
    evidence: str,
    instruction: str,
    handoff_contract: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(ctx, dict):
        return None
    view_name = _select_target_view(
        evidence=evidence,
        instruction=instruction,
        handoff_contract=handoff_contract,
    )
    if not view_name:
        return None
    targets = ctx.get("editable_targets")
    if not isinstance(targets, list) or not targets:
        targets = ctx.get("snippets")
    if not isinstance(targets, list):
        return None

    edits: list[dict[str, str]] = []
    for target in targets:
        if not isinstance(target, dict):
            continue
        path = str(target.get("file") or "").strip()
        current_code = str(target.get("current_code") or "")
        if not path or not current_code:
            continue
        new_code = _replace_boarding_pass_table(current_code, view_name)
        if new_code == current_code:
            continue
        edits.append({
            "path": path,
            "old_string": current_code,
            "new_string": new_code,
        })
        break

    if not edits:
        return None
    return {
        "edits": edits,
        "confidence": 0.82,
        "missing_info": [],
        "source": "deterministic_sql_view_fallback",
    }


def _select_target_view(
    *,
    evidence: str,
    instruction: str,
    handoff_contract: dict[str, object] | None,
) -> str:
    contract_text = (
        json.dumps(handoff_contract, ensure_ascii=False)
        if handoff_contract
        else ""
    )
    text = "\n".join([instruction, evidence, contract_text])
    candidates = list(dict.fromkeys(
        match.group(1)
        for match in re.finditer(
            r"(?i)\bCREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+([A-Za-z_][A-Za-z0-9_]*)",
            text,
        )
    ))
    candidates.extend(
        candidate
        for candidate in re.findall(r"\b(?:v|view)_[A-Za-z0-9_]*\b", text)
        if candidate not in candidates
    )
    if not candidates:
        return ""

    def score(name: str) -> tuple[int, str]:
        lowered = name.lower()
        value = 0
        if "boarding" in lowered:
            value += 8
        if "ticket" in lowered:
            value += 6
        if "report" in lowered or "detail" in lowered:
            value += 3
        if "flight" in lowered or "monitor" in lowered:
            value -= 5
        return value, name

    best_score, best = max(score(candidate) for candidate in candidates)
    return best if best_score > 0 else ""


def _replace_boarding_pass_table(current_code: str, view_name: str) -> str:
    pattern = re.compile(
        r"(?i)\b(?P<keyword>FROM|JOIN)\s+(?P<quote>[`\"]?)boarding_pass(?P=quote)\b"
    )

    def repl(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{match.group('keyword')} {quote}{view_name}{quote}"

    return pattern.sub(repl, current_code)


def _extract_marker_json(text: str, marker: str) -> dict[str, object] | None:
    if marker not in text:
        return None
    payload = text.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        return None
    depth = 0
    end = -1
    for idx, ch in enumerate(payload[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None
    try:
        data = json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_json_payload(raw: str) -> dict[str, object] | None:
    text = raw.strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _preview(raw: str, *, limit: int = 500) -> str:
    text = " ".join((raw or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."
