from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Literal, Protocol
from dotenv import load_dotenv
load_dotenv()
EmbeddingProviderName = Literal["local_hash", "openai"]
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


class EmbeddingProvider(Protocol):
    name: str
    dimensions: int

    def embed_text(self, text: str) -> list[float]: ...


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class LocalHashEmbeddingProvider:
    name = "local_hash"

    def __init__(self, dimensions: int | None = None):
        self.dimensions = dimensions or int(os.getenv("RAG_EMBEDDING_DIMENSIONS", "128"))
        if self.dimensions < 1:
            raise ValueError("Local hash embedding dimensions must be at least 1.")

    def embed_text(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        vector = [0.0] * self.dimensions
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class OpenAIEmbeddingProvider:
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        dimensions: int | None = None,
    ):
        self.model = model or os.getenv("RAG_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.dimensions = dimensions or int(os.getenv("RAG_OPENAI_EMBEDDING_DIMENSIONS", "1536"))

        if self.dimensions < 1:
            raise ValueError("OpenAI embedding dimensions must be at least 1.")

    def embed_text(self, text: str) -> list[float]:
        if not self.api_key:
            raise ValueError("OpenAI embedding provider requires OPENAI_API_KEY.")

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise ValueError(f"OpenAI client is unavailable: {exc}") from exc

        client = OpenAI(api_key=self.api_key)

        try:
            kwargs = {"model": self.model, "input": text}
            if self.model.startswith("text-embedding-3"):
                kwargs["dimensions"] = self.dimensions

            response = client.embeddings.create(**kwargs)
        except Exception as exc:
            raise ValueError(f"OpenAI embedding request failed: {exc}") from exc

        vector = list(response.data[0].embedding)
        if len(vector) != self.dimensions:
            raise ValueError(
                f"OpenAI embedding dimension mismatch: expected {self.dimensions}, got {len(vector)}."
            )
        return vector


def get_embedding_provider_name() -> str:
    return os.getenv("RAG_EMBEDDING_PROVIDER", "local_hash").strip().lower() or "local_hash"


def get_default_embedding_provider() -> EmbeddingProvider:
    provider_name = get_embedding_provider_name()
    if provider_name == "openai":
        return OpenAIEmbeddingProvider()
    return LocalHashEmbeddingProvider()


def embed_text(text: str, provider: EmbeddingProvider | None = None) -> list[float]:
    active_provider = provider or get_default_embedding_provider()
    return active_provider.embed_text(text)
