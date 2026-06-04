from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Embedder:
    """Generates text embeddings via LiteLLM (remote) or local models."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        provider: str = "openai",
    ) -> None:
        self.model = model
        self.provider = provider

    async def embed(self, text: str) -> list[float]:
        texts = await self.embed_batch([text])
        return texts[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import litellm

        response = await litellm.aembedding(model=self.model, input=texts)
        return [item["embedding"] for item in response.data]
