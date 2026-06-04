from __future__ import annotations

import json
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
            try:
                path.write_text(new_content, encoding="utf-8")
            except OSError as exc:
                errors.append(f"{edit.path}: write failed: {exc}")
                continue
            changed.append(edit.path.replace("\\", "/").lstrip("./"))

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

        messages = _edit_messages(instruction=instruction, evidence=evidence)
        raw = await self.llm_complete(messages)
        payload = _extract_json_payload(raw)
        if payload is None:
            return SkillResult(
                success=False,
                summary="code_edit model output was not a JSON object.",
                missing_info=("edit_json",),
                metadata={"raw_preview": _preview(raw)},
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
                metadata={"raw_preview": _preview(raw)},
            )
        if not isinstance(edits, list) or not edits:
            return SkillResult(
                success=False,
                summary="code_edit model returned no edits.",
                missing_info=("edits",),
                metadata={"raw_preview": _preview(raw)},
            )

        changed: list[str] = []
        errors: list[str] = []
        for idx, item in enumerate(edits):
            if not isinstance(item, dict):
                errors.append(f"edits[{idx}] must be an object")
                continue
            path = str(item.get("path") or "")
            old_string = str(item.get("old_string") or "")
            new_string = str(item.get("new_string") or "")
            if not path or not old_string or not new_string:
                errors.append(f"edits[{idx}] requires path, old_string, and new_string")
                continue
            if old_string == new_string:
                errors.append(f"edits[{idx}] old_string and new_string must differ")
                continue
            if len(old_string) > 3000 or len(new_string) > 3000:
                errors.append(f"edits[{idx}] is too large; use a smaller exact snippet")
                continue
            result = _apply_exact_edit(
                self.project_root,
                path=path,
                old_string=old_string,
                new_string=new_string,
            )
            if result.startswith("ok:"):
                changed.append(path.replace("\\", "/").lstrip("./"))
            else:
                errors.append(result)

        if errors:
            return SkillResult(
                success=False,
                summary="code_edit failed: " + "; ".join(errors),
                changed_files=tuple(dict.fromkeys(changed)),
                missing_info=tuple(errors),
                metadata={"raw_preview": _preview(raw)},
            )
        return SkillResult(
            success=True,
            summary=f"Applied {len(changed)} edit_file action(s).",
            changed_files=tuple(dict.fromkeys(changed)),
            metadata={"raw_preview": _preview(raw), "confidence": str(confidence)},
        )


def _resolve_under_root(project_root: Path, rel: str) -> Path | None:
    try:
        path = (project_root / rel.replace("\\", "/").lstrip("./")).resolve()
        path.relative_to(project_root)
        return path
    except (OSError, ValueError):
        return None


def _apply_exact_edit(
    project_root: Path,
    *,
    path: str,
    old_string: str,
    new_string: str,
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
    try:
        target.write_text(content.replace(old_string, new_string, 1), encoding="utf-8")
    except OSError as exc:
        return f"{path}: write failed: {exc}"
    return f"ok:{path}"


def _edit_messages(*, instruction: str, evidence: str) -> list[dict[str, object]]:
    return [
        {
            "role": "system",
            "content": (
                "You are MitKII code_edit skill. Output ONE raw JSON object only. "
                "No markdown, no prose. You are not an agent and cannot call tools. "
                "Use only the provided code_search evidence. Output edit_file actions "
                'as {"edits":[{"path":"...","old_string":"exact snippet",'
                '"new_string":"replacement snippet"}],"confidence":0.0,'
                '"missing_info":[]}. Keep old_string/new_string small and exact. '
                "If the exact edit is not clear, return edits=[] with confidence<0.75."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Instruction:\n{instruction}\n\n"
                "<code_search_results>\n"
                f"{evidence[:12000]}\n"
                "</code_search_results>"
            ),
        },
    ]


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
