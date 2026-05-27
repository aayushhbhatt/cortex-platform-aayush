from __future__ import annotations

import json
import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv
load_dotenv()
from rag.embeddings import get_default_embedding_provider


@dataclass
class VectorSearchResult:
    chunk_id: str
    doc_id: str
    title: str
    content: str
    chunk_index: int
    category: str
    access_level: str
    source: str
    content_hash: str
    score: float
    metadata: dict


class PGVectorStore:
    def __init__(self, database_url: str | None = None, dimensions: int | None = None, embedding_provider=None):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        self.embedding_provider = embedding_provider or get_default_embedding_provider()
        self.dimensions = dimensions or self.embedding_provider.dimensions

    def is_configured(self) -> bool:
        return bool(self.database_url)

    def _connect(self) -> psycopg.Connection:
        if not self.database_url:
            raise ValueError("PGVectorStore requires DATABASE_URL (or database_url argument) to be set.")
        return psycopg.connect(self.database_url)

    @staticmethod
    def _to_vector_literal(embedding: list[float]) -> str:
        return "[" + ",".join(f"{value:.12f}" for value in embedding) + "]"

    def initialize(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        version INTEGER,
                        title TEXT NOT NULL,
                        content TEXT NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        category TEXT NOT NULL,
                        access_level TEXT NOT NULL,
                        source TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        active BOOLEAN NOT NULL DEFAULT TRUE,
                        metadata_json TEXT NOT NULL,
                        embedding vector({self.dimensions}),
                        updated_at TIMESTAMPTZ DEFAULT now()
                    )
                    """
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_doc_id ON chunk_embeddings(doc_id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_active ON chunk_embeddings(active)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_access_level ON chunk_embeddings(access_level)")

    def upsert_chunks(self, chunks: list[dict]) -> int:
        if not chunks:
            return 0
        self.initialize()
        upserted = 0
        with self._connect() as conn:
            with conn.cursor() as cur:
                for chunk in chunks:
                    embedding = self.embedding_provider.embed_text(f"{chunk.get('title', '')} {chunk.get('content', '')}")
                    metadata = dict(chunk.get("metadata", {}) or {})
                    metadata["embedding_provider"] = self.embedding_provider.name
                    if hasattr(self.embedding_provider, "model"):
                        metadata["embedding_model"] = getattr(self.embedding_provider, "model")
                    vector_literal = self._to_vector_literal(embedding)
                    cur.execute(
                        """
                        INSERT INTO chunk_embeddings
                        (chunk_id, doc_id, version, title, content, chunk_index, category, access_level, source, content_hash, active, metadata_json, embedding, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, now())
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            doc_id = EXCLUDED.doc_id,
                            version = EXCLUDED.version,
                            title = EXCLUDED.title,
                            content = EXCLUDED.content,
                            chunk_index = EXCLUDED.chunk_index,
                            category = EXCLUDED.category,
                            access_level = EXCLUDED.access_level,
                            source = EXCLUDED.source,
                            content_hash = EXCLUDED.content_hash,
                            active = EXCLUDED.active,
                            metadata_json = EXCLUDED.metadata_json,
                            embedding = EXCLUDED.embedding,
                            updated_at = now()
                        """,
                        (
                            chunk["chunk_id"],
                            chunk["doc_id"],
                            chunk.get("version"),
                            chunk["title"],
                            chunk["content"],
                            int(chunk["chunk_index"]),
                            chunk["category"],
                            chunk["access_level"],
                            chunk["source"],
                            chunk["content_hash"],
                            bool(chunk.get("active", True)),
                            json.dumps(metadata, sort_keys=True),
                            vector_literal,
                        ),
                    )
                    upserted += 1
        return upserted

    def deactivate_missing_active_chunks(self, active_chunk_ids: set[str]) -> int:
        if not active_chunk_ids:
            return 0
        self.initialize()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chunk_embeddings
                    SET active = FALSE, updated_at = now()
                    WHERE active = TRUE AND NOT (chunk_id = ANY(%s))
                    """,
                    (list(active_chunk_ids),),
                )
                return cur.rowcount or 0

    def _build_search_sql_and_params(
        self,
        vector_literal: str,
        top_k: int,
        access_levels: list[str] | None,
    ) -> tuple[str, list[object]]:
        where_clauses = ["active = TRUE"]
        params: list[object] = [vector_literal]

        if access_levels:
            where_clauses.append("access_level = ANY(%s)")
            params.append(access_levels)

        params.append(vector_literal)
        params.append(top_k)

        sql = f"""
            SELECT
                chunk_id,
                doc_id,
                title,
                content,
                chunk_index,
                category,
                access_level,
                source,
                content_hash,
                metadata_json,
                embedding <=> %s::vector AS distance
            FROM chunk_embeddings
            WHERE {' AND '.join(where_clauses)}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s
        """
        return sql, params

    def search(self, query: str, top_k: int = 5, access_levels: list[str] | None = None) -> list[dict]:
        if not query.strip():
            return []
        self.initialize()
        query_embedding = self.embedding_provider.embed_text(query)
        vector_literal = self._to_vector_literal(query_embedding)

        sql, params = self._build_search_sql_and_params(
            vector_literal=vector_literal,
            top_k=top_k,
            access_levels=access_levels,
        )

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                rows = cur.fetchall()

        results: list[dict] = []
        for rank, row in enumerate(rows, start=1):
            metadata = json.loads(row[9]) if row[9] else {}
            metadata["embedding_provider"] = self.embedding_provider.name
            distance = float(row[10]) if row[10] is not None else 1.0
            score = max(0.0, 1.0 - distance)
            results.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "title": row[2],
                    "content": row[3],
                    "chunk_index": int(row[4]),
                    "category": row[5],
                    "access_level": row[6],
                    "source": row[7],
                    "content_hash": row[8],
                    "metadata": metadata,
                    "score": score,
                    "retrieval_method": "vector",
                    "rank": rank,
                }
            )
        return results
