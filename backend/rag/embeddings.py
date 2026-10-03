from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class OpenAIEmbeddingProvider:
    """Thin boundary around the OpenAI embeddings API, suitable for test fakes."""

    def __init__(self, model: str, client: OpenAI | None = None):
        self.model = model
        self._client = client

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI()
        return self._client

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        values = list(texts)
        if not values:
            return []
        response = self.client.embeddings.create(model=self.model, input=values)
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]
