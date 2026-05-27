from __future__ import annotations

import math
import re

from rank_bm25 import BM25Okapi

from rag.embeddings import embed_text as provider_embed_text
from rag.embeddings import get_default_embedding_provider

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def embed_text(text: str, dimensions: int = 128) -> list[float]:
    provider = get_default_embedding_provider()
    if provider.name != "local_hash":
        return provider_embed_text(text, provider=provider)
    provider.dimensions = dimensions
    return provider.embed_text(text)


def embed_chunks(chunks: list[dict], dimensions: int = 128, provider=None) -> list[dict]:
    embedded: list[dict] = []
    active_provider = provider or get_default_embedding_provider()
    for chunk in chunks:
        item = dict(chunk)
        item["embedding"] = active_provider.embed_text(f"{chunk.get('title', '')} {chunk.get('content', '')}")
        embedded.append(item)
    return embedded


def bm25_search(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    if not chunks:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    corpus = [tokenize(f"{chunk.get('title', '')} {chunk.get('content', '')}") for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    if not ranked or all(float(score) == 0.0 for _, score in ranked):
        return []
    results: list[dict] = []
    for rank, (idx, score) in enumerate(ranked[:top_k], start=1):
        item = dict(chunks[idx])
        item.update({"score": float(score), "retrieval_method": "bm25", "rank": rank})
        results.append(item)
    return results


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    denom = norm_a * norm_b
    if denom == 0.0:
        return 0.0
    return dot / denom


def vector_search(query: str, embedded_chunks: list[dict], top_k: int = 5, provider=None) -> list[dict]:
    if not embedded_chunks:
        return []
    active_provider = provider or get_default_embedding_provider()
    query_embedding = active_provider.embed_text(query)
    scored: list[tuple[dict, float]] = []
    for chunk in embedded_chunks:
        similarity = cosine_similarity(query_embedding, chunk.get("embedding", []))
        scored.append((chunk, similarity))
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    if not ranked or ranked[0][1] <= 0:
        return []
    results: list[dict] = []
    for rank, (chunk, score) in enumerate(ranked[:top_k], start=1):
        item = {k: v for k, v in chunk.items() if k != "embedding"}
        item.update({"score": float(score), "retrieval_method": "vector", "rank": rank})
        results.append(item)
    return results


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60, top_k: int = 5) -> list[dict]:
    aggregated: dict[str, dict] = {}
    for results in result_lists:
        for idx, result in enumerate(results, start=1):
            chunk_id = result["chunk_id"]
            method = result.get("retrieval_method", "unknown")
            rank = result.get("rank", idx)
            score = 1.0 / (k + rank)
            if chunk_id not in aggregated:
                aggregated[chunk_id] = {
                    "chunk_id": result["chunk_id"], "doc_id": result["doc_id"], "title": result["title"],
                    "content": result["content"], "chunk_index": result["chunk_index"], "category": result["category"],
                    "access_level": result["access_level"], "source": result["source"], "content_hash": result["content_hash"],
                    "metadata": result.get("metadata", {}), "score": 0.0, "retrieval_method": "rrf", "fused_from": [], "ranks": {},
                }
            aggregated[chunk_id]["score"] += score
            if method not in aggregated[chunk_id]["fused_from"]:
                aggregated[chunk_id]["fused_from"].append(method)
            existing = aggregated[chunk_id]["ranks"].get(method)
            if existing is None or rank < existing:
                aggregated[chunk_id]["ranks"][method] = rank
    fused = sorted(aggregated.values(), key=lambda x: x["score"], reverse=True)
    return fused[:top_k]
