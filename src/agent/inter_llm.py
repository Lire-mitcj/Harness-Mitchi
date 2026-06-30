from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.agent.contracts import InterHint

log = logging.getLogger(__name__)

MAX_INTER_ITEMS = 8
MAX_INTER_TEXT = 80


class CursorInterLLM:
    """Open-vocabulary semantic hint generator."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def build_messages(self, user_input: str) -> list[dict[str, str]]:
        return [{
            "role": "system",
            "content": (
                "Understand the user's request without forcing it into a predefined "
                "domain taxonomy. "
                "Infer the task intent in your own concise words, all plausible technical domains, "
                "the central concepts, and whether the wording is ambiguous. Domain and "
                "concept values are open vocabulary: do not choose from a fixed enum and "
                "do not discard plausible meanings. "
                "Return exactly one JSON object with this schema: "
                '{"intent":"concise intent","domains":["open vocabulary"],'
                '"concepts":["open vocabulary"],"ambiguity":false,"confidence":0.0}. '
                "Use JSON arrays even for zero or one item. Confidence must be between "
                "0.0 and 1.0. "
                "Return no prose, Markdown, file guesses, retrieval plans, or tool instructions."
            ),
        }, {"role": "user", "content": user_input}]

    async def generate(self, user_input: str) -> InterHint | None:
        messages = self.build_messages(user_input)
        try:
            response = await self.llm.chat(messages, tools=None, stream=False)
            hint = self.parse(getattr(response, "content", "") or "")
            if hint is not None:
                return hint
        except Exception as exc:
            log.warning("CursorInterLLM generate LLM call failed: %s", exc)
        return self.fallback(user_input)

    @staticmethod
    def parse(content: str) -> InterHint | None:
        log.warning("CursorInterLLM raw LLM output: %s", content)
        try:
            cleaned = _strip_json_fence(content)
            data = None
            try:
                data = json.loads(cleaned)
            except Exception:
                # Try to replace mismatched single/double quotes around strings
                # e.g., "项目管理'] -> "项目管理"
                repaired = re.sub(r'["\']([^"\']*)["\']', r'"\1"', cleaned)
                # Quote unquoted keys (e.g. {key: "value"} -> {"key": "value"})
                repaired = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', repaired)
                # Remove trailing commas
                repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
                try:
                    data = json.loads(repaired)
                except Exception:
                    log.warning("CursorInterLLM JSON parse failed, falling back to regex extraction.")
                    data = _regex_extract_inter_llm(content)

            log.warning("CursorInterLLM parsed JSON: %s", json.dumps(data, ensure_ascii=False))
            required = {"intent", "domains", "concepts", "ambiguity", "confidence"}
            if not isinstance(data, dict) or not all(k in data for k in required):
                log.warning("CursorInterLLM JSON schema mismatch or missing keys: %s", data)
                return None
            intent = _bounded_text(data["intent"])
            domains = _bounded_items(data["domains"])
            concepts = _bounded_items(data["concepts"])
            ambiguity = data["ambiguity"]
            confidence = float(data["confidence"])
            if not intent:
                log.warning("CursorInterLLM invalid intent: %s", intent)
                return None
            if not isinstance(data["domains"], (list, tuple)) or not isinstance(data["concepts"], (list, tuple)):
                log.warning("CursorInterLLM domains/concepts must be arrays")
                return None
            if not isinstance(ambiguity, bool):
                log.warning("CursorInterLLM ambiguity must be boolean")
                return None
            if not 0.0 <= confidence <= 1.0:
                log.warning("CursorInterLLM invalid confidence value: %s", confidence)
                return None
            return InterHint(
                intent=intent,
                domains=domains,
                concepts=concepts,
                ambiguity=ambiguity,
                confidence=confidence,
            )
        except Exception as e:
            log.warning("CursorInterLLM failed to parse JSON: %s", e)
            return None

    @staticmethod
    def fallback(user_input: str) -> InterHint:
        hint = InterHint(
            intent="understand request",
            domains=(),
            concepts=(),
            ambiguity=True,
            confidence=0.0,
        )
        log.warning("CursorInterLLM falling back to local heuristic: %s", hint)
        return hint


def _bounded_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()[:MAX_INTER_TEXT]


def _bounded_items(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = _bounded_text(raw)
        key = item.casefold()
        if not item or key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= MAX_INTER_ITEMS:
            break
    return tuple(items)


def _strip_json_fence(text: str) -> str:
    # 1. Strip markdown code fences if any
    text = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # 2. Extract the first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end+1]

    # 3. Remove single-line and multi-line comments
    text = re.sub(r"//.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # 4. Remove trailing commas from lists and objects
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text.strip()


def _regex_extract_inter_llm(text: str) -> dict[str, Any]:
    # Extract intent
    intent_match = re.search(r'"intent"\s*:\s*["\']([^"\']*)["\']', text)
    if not intent_match:
        intent_match = re.search(r'intent\s*:\s*["\']([^"\']*)["\']', text)
    intent = intent_match.group(1) if intent_match else "understand request"

    # Extract ambiguity
    ambiguity_match = re.search(r'"ambiguity"\s*:\s*(true|false)', text, re.IGNORECASE)
    if not ambiguity_match:
        ambiguity_match = re.search(r'ambiguity\s*:\s*(true|false)', text, re.IGNORECASE)
    ambiguity = True if ambiguity_match and ambiguity_match.group(1).lower() == "true" else False

    # Extract confidence
    confidence_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', text)
    if not confidence_match:
        confidence_match = re.search(r'confidence\s*:\s*([0-9.]+)', text)
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0

    def extract_array(key: str) -> list[str]:
        pattern = rf'"{key}"\s*:\s*\[([^\]]*)\]'
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            pattern = rf"'{key}'\s*:\s*\[([^\]]*)\]"
            match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            pattern = rf"{key}\s*:\s*\[([^\]]*)\]"
            match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return []
        array_content = match.group(1)
        words = re.findall(r'["\']([^"\']*)["\']', array_content)
        if not words:
            words = [w.strip() for w in re.split(r'\s*,\s*', array_content) if w.strip()]
        return [w.strip('"\' ') for w in words if w.strip('"\' ')]

    return {
        "intent": intent,
        "domains": extract_array("domains"),
        "concepts": extract_array("concepts"),
        "ambiguity": ambiguity,
        "confidence": confidence
    }
