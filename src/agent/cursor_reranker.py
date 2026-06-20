from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)


class SiliconFlowReranker:
    """SiliconFlow rerank API adapter for cross-encoder context ranking."""

    def __init__(
        self,
        *,
        model: str = "Qwen/Qwen3-Reranker-8B",
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.timeout = timeout

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> dict[int, float]:
        if not documents:
            return {}
        return await asyncio.to_thread(
            self._rerank_sync,
            query,
            documents,
            min(top_n, len(documents)),
        )

    def _rerank_sync(
        self,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> dict[int, float]:
        api_key = self.api_key or os.environ.get("SILICONFLOW_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        if not api_key:
            log.debug("SiliconFlow reranker disabled: missing API key")
            return {}

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        request = Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            log.debug("SiliconFlow reranker request failed: %s", exc)
            return {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            log.debug("SiliconFlow reranker returned non-JSON response")
            return {}
        return _parse_rerank_scores(data)

    def _endpoint(self) -> str:
        base = (
            self.api_base
            or os.environ.get("SILICONFLOW_API_BASE")
            or os.environ.get("OPENAI_API_BASE")
            or "https://api.siliconflow.cn/v1"
        ).rstrip("/")
        if base.endswith("/rerank"):
            return base
        if base.endswith("/v1"):
            return f"{base}/rerank"
        return f"{base}/v1/rerank"


def _parse_rerank_scores(data: Any) -> dict[int, float]:
    results = data.get("results") if isinstance(data, dict) else None
    if results is None and isinstance(data, dict):
        results = data.get("data")
    if not isinstance(results, list):
        return {}

    scores: dict[int, float] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        raw_index = item.get("index")
        if raw_index is None:
            raw_index = item.get("document_index")
        raw_score = item.get("relevance_score")
        if raw_score is None:
            raw_score = item.get("score")
        if raw_index is None or raw_score is None:
            continue
        try:
            index = int(raw_index)
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        scores[index] = score
    return scores
