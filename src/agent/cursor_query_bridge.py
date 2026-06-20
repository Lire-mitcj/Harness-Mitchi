from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)

MAX_EXPANDED_TERMS = 8
MAX_KEYWORDS = 8
MAX_FILE_HINTS = 6
MAX_TERM_LENGTH = 80

_QUERY_ALIASES: dict[str, tuple[str, ...]] = {
    "接口": ("api", "endpoint", "route"),
    "配置": ("config", "configuration", "settings"),
    "日志": ("log", "logger", "logging"),
    "测试": ("test", "tests", "pytest"),
    "校验": ("validate", "validation", "validator", "check"),
    "验证": ("validate", "validation", "validator", "verify"),
    "错误": ("error", "exception", "failure"),
    "异常": ("error", "exception", "raise", "except"),
    "处理": ("handle", "handler", "handling"),
    "检索": ("retrieve", "retrieval", "search"),
    "搜索": ("search", "grep", "find"),
    "状态": ("state", "status"),
    "执行": ("execute", "executor", "execution"),
    "模型": ("model", "llm"),
    "上下文": ("context", "prompt"),
    "压缩": ("compress", "compression", "compact"),
    "用户": ("user",),
    "文件": ("file", "path"),
    "函数": ("function", "def"),
    "类": ("class",),
    "数据库": ("database", "db", "sql"),
    "查询": ("query", "lookup", "select", "search"),
    "视图": (
        "view",
        "views",
        "ui",
        "component",
        "page",
        "CREATE VIEW",
        "schema",
        "sql",
        "sql_view",
        "database_view",
    ),
    "登机牌": ("boarding", "boarding_pass", "pass", "ticket", "ticket_lookup"),
    "订单": ("orders", "order"),
}


@dataclass(frozen=True, slots=True)
class QueryBridgeResult:
    intent: Literal["list", "query", "modify", "debug", "explain"]
    expanded_terms: list[str]
    keywords: list[str]
    symbols: list[str]
    file_hints: list[str]
    domain: str = ""
    concepts: list[str] | None = None
    constraints: dict[str, list[str]] | None = None

    def search_terms(self, *, limit: int = 12) -> tuple[str, ...]:
        merged: list[str] = []
        seen: set[str] = set()
        concepts = self.concepts or []
        layer_hints = (self.constraints or {}).get("layer_hint", [])
        for term in (
            self.keywords
            + self.expanded_terms
            + concepts
            + ([self.domain] if self.domain else [])
            + layer_hints
        ):
            if not term.strip():
                continue
            normalized = term.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            merged.append(term)
            if len(merged) >= limit:
                break
        return tuple(merged)


class CursorQueryBridge:
    """Stage 2 Query Bridge: Semantic Expansion using LLM and prompt file."""

    def __init__(self, llm: Any, *, timeout: float = 15.0) -> None:
        self.llm = llm
        self.timeout = timeout

    def build_messages(self, user_query: str) -> list[dict[str, str]]:
        try:
            prompt_path = (
                Path(__file__).resolve().parent.parent.parent
                / "prompts"
                / "query_bridge_prompt.md"
            )
            system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            log.warning("CursorQueryBridge failed to load prompt from file: %s", exc)
            system_prompt = (
                "You are part of a multi-layer code intelligence system composed of 4 stages:\n"
                "User Query -> Query Bridge -> Retriever -> Fusion Engine.\n"
                "Convert the user query into retrieval signals. Return strict JSON:\n"
                '{"intent":"modify|query|debug|explain","domain":"",'
                '"concepts":[],"expanded_terms":[],"constraints":{"layer_hint":[],"exclude":[]}}\n'
                "Return at most 8 expanded_terms, 8 keywords, 8 symbols, and 6 file_hints."
            )

        user_content = json.dumps({"user_query": user_query}, ensure_ascii=False)

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

    async def generate_raw(self, user_query: str) -> str | None:
        if self.llm is None:
            log.warning("CursorQueryBridge has no LLM client")
            return None
        try:
            log.warning(
                "CursorQueryBridge: requesting retrieval signals with %.2fs timeout",
                self.timeout,
            )
            messages = self.build_messages(user_query)
            async with asyncio.timeout(self.timeout):
                response = await self.llm.chat(messages, tools=None, stream=False)
            return getattr(response, "content", "") or ""
        except TimeoutError:
            log.warning(
                "CursorQueryBridge: LLM timed out after %.2fs; using heuristic",
                self.timeout,
            )
            return None
        except Exception as exc:
            log.warning(
                "CursorQueryBridge generate LLM call failed: %s; using heuristic",
                exc,
            )
            return None

    async def generate(self, user_query: str) -> QueryBridgeResult:
        """Compatibility entry point; guarded runtimes should use generate_raw."""
        content = await self.generate_raw(user_query)
        if not content:
            return self.fallback(user_query)
        return self.parse(content)

    def parse(self, content: str) -> QueryBridgeResult:
        log.warning("CursorQueryBridge raw LLM output: %s", content)
        data = repair_json(content)

        # Read old dual-output responses for compatibility, but never request them.
        if isinstance(data, dict) and "structured_json" in data:
            sub = data["structured_json"]
        else:
            sub = data

        if not isinstance(sub, dict):
            log.warning("CursorQueryBridge sub-schema is not a dict. Falling back to regex.")
            sub = _regex_extract_query_bridge(content)

        log.warning(
            "CursorQueryBridge final parsed structure: %s",
            json.dumps(sub, ensure_ascii=False),
        )

        intent = sub.get("intent", "explain")
        domain = str(sub.get("domain") or "").strip()[:MAX_TERM_LENGTH]
        concepts = _bounded_terms(sub.get("concepts"), MAX_KEYWORDS)
        expanded_terms = _bounded_terms(sub.get("expanded_terms"), MAX_EXPANDED_TERMS)
        keywords = _bounded_terms(sub.get("keywords"), MAX_KEYWORDS)
        # QueryBridge must not return repository symbols. Keep the field empty
        # for compatibility with older guardrail/fusion code paths.
        symbols: list[str] = []
        file_hints = _bounded_terms(sub.get("file_hints"), MAX_FILE_HINTS)
        constraints = _constraints(sub.get("constraints"))

        if intent == "list":
            intent = "query"
        if intent not in {"query", "modify", "debug", "explain"}:
            intent = "explain"

        if not concepts:
            concepts = keywords[:]
        if not keywords:
            keywords = concepts[:MAX_KEYWORDS]
        if not file_hints:
            file_hints = constraints.get("layer_hint", [])[:MAX_FILE_HINTS]
        return QueryBridgeResult(
            intent=intent,
            expanded_terms=expanded_terms,
            keywords=keywords,
            symbols=symbols,
            file_hints=file_hints,
            domain=domain,
            concepts=concepts,
            constraints=constraints,
        )

    def fallback(self, user_query: str) -> QueryBridgeResult:
        tokens = _rewrite_tokens(user_query)
        keywords = [t for t in tokens if t in user_query]
        expanded = [t for t in tokens if t not in keywords]

        file_hints: list[str] = []
        expanded = _prioritize_expanded(expanded, user_query)
        expanded = expanded[:MAX_EXPANDED_TERMS]

        if not any((expanded, keywords, file_hints)):
            expanded = ["code", "module", "class", "function"]

        # Simple keyword-based intent classification for fallback
        lowered_query = user_query.lower()
        modify_words = (
            "修改", "改成", "换成", "更新", "编辑", "修复",
            "change", "modify", "edit", "update", "replace", "fix", "rewrite",
        )
        debug_words = (
            "测试", "调试", "报错", "异常", "错误",
            "bug", "debug", "error", "fail", "exception",
        )
        query_words = ("列出", "列表", "查找", "展示", "list", "show", "find")
        if any(w in lowered_query for w in modify_words):
            intent: Literal["list", "query", "modify", "debug", "explain"] = "modify"
        elif any(w in lowered_query for w in debug_words):
            intent = "debug"
        elif any(w in lowered_query for w in query_words):
            intent = "query"
        else:
            intent = "explain"

        domain = _infer_domain(user_query, tokens)
        layer_hint = _infer_layers(tokens)
        excludes = _infer_excludes(tokens)

        return QueryBridgeResult(
            intent=intent,
            expanded_terms=expanded,
            keywords=keywords,
            symbols=[],
            file_hints=file_hints or layer_hint,
            domain=domain,
            concepts=_infer_concepts(user_query, tokens),
            constraints={"layer_hint": layer_hint, "exclude": excludes},
        )


def _rewrite_tokens(query: str) -> tuple[str, ...]:
    tokens: set[str] = {
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", query)
        if len(token) >= 2
    }
    chinese_phrases = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    for phrase in chinese_phrases:
        if len(phrase) <= 32:
            tokens.add(phrase)
        try:
            import jieba  # type: ignore[import-untyped]

            tokens.update(word for word in jieba.lcut(phrase) if len(word) >= 2)
        except ImportError:
            tokens.update(
                phrase[index:index + 2]
                for index in range(max(0, len(phrase) - 1))
            )

    for chinese, aliases in _QUERY_ALIASES.items():
        if chinese in query or chinese in tokens:
            tokens.update(aliases)

    return tuple(sorted(tokens, key=lambda token: (len(token), token))[:32])


def _bounded_terms(value: Any, limit: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str):
            continue
        term = raw.strip()
        normalized = term.casefold()
        if (
            len(term) < 2
            or len(term) > MAX_TERM_LENGTH
            or normalized in seen
        ):
            continue
        seen.add(normalized)
        terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def _constraints(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {"layer_hint": [], "exclude": []}
    return {
        "layer_hint": _bounded_terms(value.get("layer_hint"), MAX_FILE_HINTS),
        "exclude": _bounded_terms(value.get("exclude"), MAX_FILE_HINTS),
    }


def _infer_domain(query: str, tokens: tuple[str, ...]) -> str:
    if "登机牌" in query or {"boarding_pass", "ticket"} & set(tokens):
        return "ticketing"
    if "订单" in query or {"order", "orders"} & set(tokens):
        return "orders"
    if any(word in query.lower() for word in ("auth", "login", "用户", "权限")):
        return "auth"
    if any(word in query.lower() for word in ("payment", "pay", "支付")):
        return "payment"
    if "数据库" in query or "sql" in tokens:
        return "database"
    return "codebase"


def _infer_concepts(query: str, tokens: tuple[str, ...]) -> list[str]:
    token_set = {token.casefold() for token in tokens}
    concepts: list[str] = []

    def add(value: str) -> None:
        if value not in concepts and len(concepts) < MAX_KEYWORDS:
            concepts.append(value)

    if "登机牌" in query or {"boarding_pass", "boarding", "ticket"} & token_set:
        add("boarding pass")
    if "接口" in query or {"api", "endpoint", "route"} & token_set:
        add("query api" if "查询" in query or "query" in token_set else "api endpoint")
    if "视图" in query or {"view", "views", "sql"} & token_set:
        if {"sql", "database", "db", "select"} & token_set or "查询" in query:
            add("view based query")
        else:
            add("view")
    if "订单" in query or {"orders", "order"} & token_set:
        add("order")
    if not concepts:
        for token in tokens:
            if re.match(r"^[A-Za-z][A-Za-z0-9_ -]*$", token):
                add(token.replace("_", " "))
    return concepts[:MAX_KEYWORDS]


def _prioritize_expanded(expanded: list[str], query: str) -> list[str]:
    priority: list[str] = []
    if "登机牌" in query:
        priority.extend(["boarding_pass", "ticket_lookup", "boarding", "ticket"])
    if "接口" in query:
        priority.extend(["api", "endpoint", "route"])
    if "视图" in query:
        priority.extend([
            "view",
            "sql_view",
            "database_view",
            "CREATE VIEW",
            "component",
            "page",
            "schema",
            "sql",
        ])
    ordered = priority + sorted(expanded, key=len, reverse=True)
    out: list[str] = []
    seen: set[str] = set()
    for term in ordered:
        key = term.casefold()
        if key in seen or term not in expanded:
            continue
        seen.add(key)
        out.append(term)
    return out


def _infer_layers(tokens: tuple[str, ...]) -> list[str]:
    token_set = {token.casefold() for token in tokens}
    layers: list[str] = []
    if token_set & {"api", "endpoint", "route", "handler"}:
        layers.append("api")
    if token_set & {"service", "handler", "business"}:
        layers.append("service")
    if token_set & {"database", "db", "sql", "view", "schema", "select"}:
        layers.append("dao")
    return layers[:MAX_FILE_HINTS]


def _infer_excludes(tokens: tuple[str, ...]) -> list[str]:
    token_set = {token.casefold() for token in tokens}
    excludes: list[str] = []
    if {"boarding_pass", "ticket", "orders", "order"} & token_set:
        excludes.append("auth unrelated")
    if {"sql", "view", "database", "db"} & token_set:
        excludes.append("ui unrelated")
    return excludes[:MAX_FILE_HINTS]


def repair_json(text: str) -> dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Attempt 1: Quote unquoted keys (e.g., {key: "value"} -> {"key": "value"})
    cleaned_v2 = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', cleaned)
    # Replace single quotes around strings with double quotes
    cleaned_v2 = re.sub(r"'\s*([^']*?)\s*'", r'"\1"', cleaned_v2)
    # Remove trailing commas
    cleaned_v2 = re.sub(r",\s*([\]}])", r"\1", cleaned_v2)

    try:
        parsed = json.loads(cleaned_v2)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Attempt 2: Fallback to regex-based extraction
    log.warning("JSON parsing/repair failed on text, falling back to regex extraction.")
    return _regex_extract_query_bridge(text)


def _regex_extract_query_bridge(text: str) -> dict[str, Any]:
    # Extract intent
    intent_match = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    if not intent_match:
        intent_match = re.search(r"'intent'\s*:\s*'([^']+)'", text)
    if not intent_match:
        intent_match = re.search(r'intent\s*:\s*"?([a-zA-Z]+)"?', text)
    intent = intent_match.group(1) if intent_match else "explain"
    if intent not in {"list", "query", "modify", "debug", "explain"}:
        intent = "explain"

    domain_match = re.search(r'"domain"\s*:\s*"([^"]+)"', text)
    if not domain_match:
        domain_match = re.search(r"'domain'\s*:\s*'([^']+)'", text)
    domain = domain_match.group(1) if domain_match else ""

    # Extract arrays.
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
        # Extract all strings/words inside the array content
        words = re.findall(r'"([^"]*)"', array_content)
        if not words:
            words = re.findall(r"'([^']*)'", array_content)
        if not words:
            # Fallback to listing alphanumeric words separated by comma
            words = [w.strip() for w in re.split(r'\s*,\s*', array_content) if w.strip()]
        return [w.strip('"\' ') for w in words if w.strip('"\' ')]

    return {
        "intent": intent,
        "domain": domain,
        "concepts": extract_array("concepts"),
        "expanded_terms": extract_array("expanded_terms"),
        "keywords": extract_array("keywords"),
        "symbols": extract_array("symbols"),
        "file_hints": extract_array("file_hints"),
        "constraints": {
            "layer_hint": extract_array("layer_hint"),
            "exclude": extract_array("exclude"),
        },
    }


def _strip_json_fence(text: str) -> str:
    import re
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
