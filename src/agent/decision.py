from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from src.agent.contracts import ContextPack, Decision, InterHint
from src.agent.inter_llm import _strip_json_fence
from src.context.prompt_resources import load_internal_prompt


class DecisionError(ValueError):
    pass


@lru_cache(maxsize=1)
def _decision_system_prompt(schema: str) -> str:
    prompt = load_internal_prompt(
        "decision_prompt.md",
        fallback="You are a bounded code-edit decision model. Return {{SCHEMA}}.",
    )
    return prompt.replace("{{SCHEMA}}", schema)


class CursorDecisionLLM:
    """Bounded LLM decision point: answer, clarify, or one-file patch."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def decide(
        self,
        *,
        state_text: str,
        context_pack: ContextPack,
        hint: InterHint | None = None,
    ) -> tuple[Decision, Any]:
        messages = self.build_messages(
            state_text=state_text,
            context_pack=context_pack,
            hint=hint,
        )
        response = await self.llm.chat(messages, tools=None, stream=False)
        content = getattr(response, "content", "") or ""
        return self.parse(content, context_pack.candidate_files), response

    def build_messages(
        self,
        *,
        state_text: str,
        context_pack: ContextPack,
        hint: InterHint | None,
        evidence_flag: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        hint_text = "unavailable"
        if hint is not None:
            hint_text = json.dumps({
                "intent": hint.intent,
                "domains": list(hint.domains),
                "concepts": list(hint.concepts),
                "ambiguity": hint.ambiguity,
                "confidence": hint.confidence,
            }, ensure_ascii=False)
        context = []
        for window in context_pack.windows:
            tags = ", ".join(window.semantic_tags) or "none"
            role = getattr(window, "role", "reference")
            mode = getattr(window, "mode", "snippet")
            context.append(
                f'<file path="{window.file}" role="{role}" mode="{mode}" lines="{window.start_line}-'
                f'{window.end_line}" tags="{tags}">\n'
                f"{window.content}\n</file>"
            )
        schema = (
            '{"action":"edit|answer|ask_clarify","answer":"",'
            '"clarification":"","target_file":"","patch":"","suggested_completion":0}'
        )
        if evidence_flag is None:
            evidence_flag = {
                "retrieval_results": list(context_pack.candidate_files),
                "can_answer": len(context_pack.windows) > 0,
            }
        evidence_flag_text = json.dumps(evidence_flag, ensure_ascii=False)
        return [{
            "role": "system",
            "content": _decision_system_prompt(schema),
        }, {
            "role": "user",
            "content": (
                f"EVIDENCE_FLAG\n{evidence_flag_text}\n\n"
                f"CURRENT_STATE\n{state_text}\n\n"
                f"OPTIONAL_INTER_HINT\n{hint_text}\n\n"
                "CURRENT_CONTEXT\n" + "\n\n".join(context)
            ),
        }]

    @staticmethod
    def parse(content: str, candidate_files: tuple[str, ...]) -> Decision:
        try:
            data = json.loads(_strip_json_fence(content))
        except (TypeError, json.JSONDecodeError) as exc:
            raise DecisionError(f"decision_schema: invalid JSON: {exc}") from exc

        required_keys = {"action", "answer", "clarification", "target_file", "patch"}
        provided_keys = set(data)
        if not required_keys.issubset(provided_keys):
            raise DecisionError("decision_schema: missing required fields")
        extra_keys = provided_keys - required_keys - {"suggested_completion"}
        if extra_keys:
            raise DecisionError("decision_schema: unexpected fields")
        if not all(isinstance(data[key], str) for key in required_keys):
            raise DecisionError(
                "decision_schema: every field except suggested_completion must be a string"
            )

        raw_completion = data.get("suggested_completion", 0)
        if isinstance(raw_completion, str):
            raw_completion = raw_completion.replace("%", "").strip()
        try:
            val = float(raw_completion)
            stage_completion = min(1.0, max(0.0, val / (100.0 if val > 1.0 else 1.0)))
        except (ValueError, TypeError):
            stage_completion = 0.0

        action = data["action"]
        if action not in {"edit", "answer", "ask_clarify"}:
            raise DecisionError(f"decision_schema: unsupported action {action!r}")
        decision = Decision(
            action=action,
            answer=data["answer"],
            clarification=data["clarification"],
            target_file=_normalize_path(data["target_file"]),
            patch=data["patch"],
            suggested_completion=stage_completion,
        )
        if action == "edit":
            if not decision.target_file or not decision.patch:
                raise DecisionError("decision_schema: edit requires target_file and patch")
            normalized_candidates = {_normalize_path(path) for path in candidate_files}
            if decision.target_file not in normalized_candidates:
                raise DecisionError("decision_scope: target_file is not a retrieved candidate")
            if decision.answer or decision.clarification:
                raise DecisionError("decision_schema: edit cannot include answer text")
        elif decision.target_file or decision.patch:
            raise DecisionError("decision_schema: non-edit action cannot include edit fields")
        elif action == "answer" and not decision.answer:
            raise DecisionError("decision_schema: answer is required")
        elif action == "ask_clarify" and not decision.clarification:
            raise DecisionError("decision_schema: clarification is required")
        return decision


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")
