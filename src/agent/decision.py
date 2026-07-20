from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from src.agent.contracts import ContextPack, Decision, InterHint
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
        edit_only: bool = False,
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
            "ACTION: edit\n"
            "TARGET_FILE: <one file listed in CURRENT_CONTEXT>\n"
            "COMPLETION: <integer 0-100>\n"
            "SITE: symbol=<name from focus_symbols / CURRENT_CONTEXT>\n"
            "MODE: insert_after\n"
            "ANCHOR: <exact on-disk line>\n"
            "<<<<<<< REPLACE\n"
            "<only new/changed lines — do NOT rewrite the whole symbol>\n"
            ">>>>>>> REPLACE"
            if edit_only
            else (
                "ACTION: edit|answer|ask_clarify\n"
                "COMPLETION: <integer 0-100>\n"
                "# then, depending on ACTION:\n"
                "#  edit        -> TARGET_FILE: <file>, then one or more:\n"
                "#                SITE: symbol=<name>   (or span=<start>-<end>)\n"
                "#                MODE: insert_after|insert_before|replace_anchor|replace\n"
                "#                ANCHOR: <exact on-disk line>   (required for delta modes)\n"
                "#                <<<<<<< REPLACE\n"
                "#                <only the new/changed snippet>\n"
                "#                >>>>>>> REPLACE\n"
                "#                Prefer delta MODE; avoid full-symbol rewrite.\n"
                "#                Do NOT emit SEARCH — harness merges at ANCHOR.\n"
                "#  answer      -> ANSWER: <text>\n"
                "#  ask_clarify -> CLARIFICATION: <text>"
            )
        )
        if evidence_flag is None:
            evidence_flag = {
                "retrieval_results": list(context_pack.candidate_files),
                "can_answer": len(context_pack.windows) > 0,
            }
        evidence_flag_text = json.dumps(evidence_flag, ensure_ascii=False)
        system_prompt = _decision_system_prompt(schema)
        if edit_only:
            system_prompt += (
                "\n\nEDIT_ONLY_MODE: The caller already invoked `decision_edit`. "
                "Return `ACTION: edit` only. Never return `ask_clarify` or `answer`. "
                "Prefer MODE insert_after/insert_before/replace_anchor with ANCHOR + "
                "a short REPLACE snippet — do NOT rewrite the whole symbol. "
                "Do not emit SEARCH. Never echo on-disk text unchanged. "
                "At most 3 SITE blocks; prefer one."
            )
        user_parts = [
            f"EVIDENCE_FLAG\n{evidence_flag_text}",
            f"CURRENT_STATE\n{state_text}",
        ]
        # Skip empty InterHint noise in edit_only (Core already fixed intent).
        if hint is not None or not edit_only:
            user_parts.append(f"OPTIONAL_INTER_HINT\n{hint_text}")
        user_parts.append("CURRENT_CONTEXT\n" + "\n\n".join(context))
        return [{
            "role": "system",
            "content": system_prompt,
        }, {
            "role": "user",
            "content": "\n\n".join(user_parts),
        }]

    @staticmethod
    def parse(
        content: str,
        candidate_files: tuple[str, ...],
        *,
        edit_only: bool = False,
    ) -> Decision:
        # Preferred format: a plain header block (`ACTION:`/`TARGET_FILE:` …)
        # followed by the raw SEARCH/REPLACE patch. The patch is NOT a JSON
        # string, so no escaping of quotes/newlines/backslashes is needed — this
        # removes the entire class of "invalid JSON" edit failures at the source.
        data = _parse_block_format(content)
        if data is None:
            # Backward/robust fallback: some responses are still JSON. Use a light
            # extractor (fences + brace slice only); the shared _strip_json_fence
            # also runs comment/trailing-comma regexes over the whole blob, which
            # silently corrupts patch code — e.g. \s{2,} -> \s{2}, a // b, http://.
            cleaned = _extract_json_object(content)
            try:
                data = json.loads(cleaned)
            except (TypeError, json.JSONDecodeError) as exc:
                # Legacy JSON salvage chain: fix invalid \escapes and raw control
                # chars, then bound the patch by SEARCH/REPLACE sentinels.
                try:
                    data = json.loads(_repair_invalid_json_escapes(cleaned))
                except (TypeError, json.JSONDecodeError):
                    data = _salvage_edit_by_markers(cleaned)
                    if data is None:
                        raise DecisionError(
                            f"decision_schema: invalid JSON: {exc}"
                        ) from exc

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
        if edit_only and action != "edit":
            raise DecisionError(
                f"decision_schema: edit-only mode forbids action {action!r}; "
                "return SITE + REPLACE (no SEARCH) instead of clarifying."
            )
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


_VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
_CONTROL_CHAR_ESCAPES = {
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
    "\b": "\\b",
    "\f": "\\f",
}


def _repair_invalid_json_escapes(text: str) -> str:
    """Repair the two transport artifacts the Edit model produces in patch JSON.

    Walks the text tracking JSON string boundaries. Inside a string it fixes:

    1. Invalid backslash escapes: a backslash followed by a valid escape char
       (`" \\ / b f n r t u`) is copied verbatim as a pair — so already-correct
       escapes (including `\\\\` and `\\"`) are preserved and never toggle the
       string state — while any other backslash is doubled (e.g. regex ``\\W``).
    2. Raw control characters: the model routinely writes a multi-line code patch
       with literal newlines/tabs instead of ``\\n``/``\\t``, which JSON forbids
       inside strings ("Invalid control character"). These are escaped in place.

    Backslashes and control characters outside strings are left untouched (the
    latter are legal JSON whitespace between tokens).
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\":
            nxt = text[i + 1] if i + 1 < n else ""
            if in_string and nxt in _VALID_JSON_ESCAPES:
                # Valid escape pair: copy both, so an escaped quote never toggles
                # in_string and an already-doubled backslash stays doubled.
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            if in_string:
                out.append("\\\\")
                i += 1
                continue
            out.append(ch)
            i += 1
            continue
        if in_string and ord(ch) < 0x20:
            # Unescaped control char inside a string is illegal JSON; escape it.
            out.append(_CONTROL_CHAR_ESCAPES.get(ch, f"\\u{ord(ch):04x}"))
            i += 1
            continue
        if not in_string and ch == ",":
            # Drop a structural trailing comma (",}" / ",]") some models emit.
            # Done here (string-aware) instead of via a blob regex so code like
            # a regex quantifier \s{2,} inside a string is never touched.
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "]}":
                i += 1
                continue
        if ch == '"':
            in_string = not in_string
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_json_object(text: str) -> str:
    """Extract the JSON object from an LLM response without mangling its contents.

    Strips markdown fences and slices from the first ``{`` to the last ``}``.
    Deliberately does NOT remove comments or trailing commas via blob regexes —
    those are string-unaware and corrupt patch code. Structural trailing-comma
    tolerance is handled string-aware inside ``_repair_invalid_json_escapes``.
    """
    stripped = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    stripped = re.sub(r"\n```$", "", stripped, flags=re.MULTILINE)
    stripped = stripped.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    return stripped.strip()


_SEARCH_MARKER = "<<<<<<< SEARCH"
_REPLACE_OPEN = "<<<<<<< REPLACE"
_REPLACE_MARKER = ">>>>>>> REPLACE"


def _strip_outer_fence(text: str) -> str:
    """Remove a single markdown code fence wrapping the whole response, if any.

    Only touches the very start/end, so fences that appear *inside* a patch body
    (e.g. a regex like ``r"(```|...)"``) are left untouched.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_]*[ \t]*\n", "", stripped)
        stripped = re.sub(r"\n```[ \t]*$", "", stripped)
    return stripped.strip()


def _parse_block_format(content: str) -> dict[str, Any] | None:
    """Parse the header-block edit format into a schema-shaped dict, else None.

    Preferred edit format (no SEARCH — harness fills it from disk):

        ACTION: edit
        TARGET_FILE: path/to/file.py
        COMPLETION: 50
        SITE: symbol=Foo
        <<<<<<< REPLACE
        ...complete new text for that symbol/span...
        >>>>>>> REPLACE

    Legacy SEARCH/REPLACE blocks are still accepted for backward compatibility.
    ``answer`` / ``ask_clarify`` use an ``ANSWER:`` / ``CLARIFICATION:`` body
    instead of a patch. Returns None when there is no ``ACTION:`` header (so the
    caller falls back to JSON parsing).
    """
    text = _strip_outer_fence(content)
    if not text or text.lstrip().startswith("{"):
        return None
    action_match = re.search(r"(?im)^[ \t]*ACTION[ \t]*:[ \t]*(\w+)[ \t]*$", text)
    if not action_match:
        return None
    action = action_match.group(1).lower().strip()

    def _field(name: str) -> str:
        match = re.search(rf"(?im)^[ \t]*{name}[ \t]*:[ \t]*(.*)$", text)
        return match.group(1).strip() if match else ""

    target_file = _field("TARGET_FILE")
    completion = _field("COMPLETION") or "0"
    answer = ""
    clarification = ""
    patch = ""

    if action == "edit":
        search_at = text.find(_SEARCH_MARKER)
        replace_open = text.find(_REPLACE_OPEN)
        replace_close = text.rfind(_REPLACE_MARKER)
        if search_at != -1 and replace_close > search_at:
            # Legacy: LLM-emitted SEARCH/REPLACE (still accepted).
            patch = text[search_at : replace_close + len(_REPLACE_MARKER)].strip("\n")
        elif replace_open != -1 and replace_close > replace_open:
            # New contract: SITE + REPLACE-only (or bare REPLACE).
            site_match = re.search(r"(?im)^[ \t]*SITE[ \t]*:", text)
            start = (
                site_match.start()
                if site_match is not None and site_match.start() < replace_open
                else replace_open
            )
            patch = text[start : replace_close + len(_REPLACE_MARKER)].strip("\n")
        else:
            return None
    elif action in {"answer", "ask_clarify"}:
        marker = "ANSWER" if action == "answer" else "CLARIFICATION"
        body_match = re.search(rf"(?im)^[ \t]*{marker}[ \t]*:[ \t]*(.*)$", text)
        body = text[body_match.start(1):].strip() if body_match else ""
        if action == "answer":
            answer = body
        else:
            clarification = body
    else:
        return None

    return {
        "action": action,
        "answer": answer,
        "clarification": clarification,
        "target_file": target_file,
        "patch": patch,
        "suggested_completion": completion,
    }


def _escape_json_string_body(segment: str) -> str:
    """Escape a raw code segment into a valid JSON string body.

    Preserves the model's existing backslash escapes (``\\\\``, ``\\uXXXX``,
    ``\\"`` …) by copying every ``\\X`` pair verbatim, and only escapes the
    characters the model tends to leave raw: unescaped double quotes and control
    characters. The result can be wrapped in quotes and handed to ``json.loads``,
    which then decodes the backslash/unicode escapes correctly.
    """
    out: list[str] = []
    i = 0
    n = len(segment)
    while i < n:
        ch = segment[i]
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(segment[i + 1])
                i += 2
                continue
            i += 1
            continue
        if ch == '"':
            out.append('\\"')
            i += 1
            continue
        if ord(ch) < 0x20:
            out.append(_CONTROL_CHAR_ESCAPES.get(ch, f"\\u{ord(ch):04x}"))
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _salvage_edit_by_markers(text: str) -> dict[str, Any] | None:
    """Rebuild an edit decision from SEARCH/REPLACE sentinels when JSON is broken.

    Some Edit-model outputs escape the patch inconsistently (raw quotes and raw
    newlines mixed with correctly-doubled backslashes), which defeats quote-aware
    JSON repair. The ``<<<<<<< SEARCH`` / ``>>>>>>> REPLACE`` markers are
    unambiguous, so we bound the patch value with them and re-escape only that
    span, then read the small metadata fields with tolerant regexes. Returns a
    schema-shaped dict, or ``None`` when the markers/target are absent.
    """
    patch_open = re.search(r'"patch"\s*:\s*"', text)
    if not patch_open:
        return None
    patch_start = patch_open.end()
    replace_at = text.rfind(_REPLACE_MARKER)
    if replace_at < patch_start:
        return None
    patch_raw = text[patch_start : replace_at + len(_REPLACE_MARKER)]
    if _SEARCH_MARKER not in patch_raw:
        return None
    try:
        patch_value = json.loads('"' + _escape_json_string_body(patch_raw) + '"')
    except (TypeError, json.JSONDecodeError):
        return None

    target_match = re.search(
        r'"target_file"\s*:\s*"(.*?)"\s*,\s*"patch"\s*:', text, re.DOTALL
    )
    if not target_match:
        return None
    target_file = target_match.group(1).strip()

    completion_match = re.search(
        r'"suggested_completion"\s*:\s*"?(\d+(?:\.\d+)?)"?', text
    )
    suggested_completion = completion_match.group(1) if completion_match else "0"

    return {
        "action": "edit",
        "answer": "",
        "clarification": "",
        "target_file": target_file,
        "patch": patch_value,
        "suggested_completion": suggested_completion,
    }


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")
