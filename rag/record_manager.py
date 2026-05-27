from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class IngestionRegistry:
    def __init__(self, db_path: str | Path = "data/cortex_registry.sqlite"):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT NOT NULL,
                    doc_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    category TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    source TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_dir TEXT NOT NULL,
                    documents_seen INTEGER NOT NULL,
                    documents_changed INTEGER NOT NULL,
                    chunks_active INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_version ON documents(doc_id, version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_doc_active ON documents(doc_id, active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_version ON chunks(doc_id, version)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active ON chunks(active)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_chunk_id ON chunks(chunk_id)")

    def get_latest_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ? ORDER BY version DESC LIMIT 1", (doc_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_active_document(self, doc_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ? AND active = 1 ORDER BY version DESC LIMIT 1", (doc_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_next_version(self, doc_id: str) -> int:
        latest = self.get_latest_document(doc_id)
        return 1 if latest is None else int(latest["version"]) + 1

    def deactivate_document_versions(self, doc_id: str) -> None:
        now = self._now()
        with self._connect() as conn:
            conn.execute("UPDATE documents SET active = 0, updated_at = ? WHERE doc_id = ?", (now, doc_id))
            conn.execute("UPDATE chunks SET active = 0, updated_at = ? WHERE doc_id = ?", (now, doc_id))

    def upsert_document_version(self, document: dict, chunks: list[dict]) -> dict:
        doc_id = str(document["doc_id"])
        latest = self.get_latest_document(doc_id)
        if latest and latest["content_hash"] == document["content_hash"]:
            return {"doc_id": doc_id, "version": int(latest["version"]), "status": "unchanged", "chunks_inserted": 0}

        status = "created" if latest is None else "updated"
        version = 1 if latest is None else self.get_next_version(doc_id)
        if latest is not None:
            self.deactivate_document_versions(doc_id)

        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                (doc_id, source, title, category, access_level, content_hash, version, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    doc_id,
                    document["source"],
                    document["title"],
                    document["category"],
                    document["access_level"],
                    document["content_hash"],
                    version,
                    now,
                    now,
                ),
            )

            inserted = 0
            for chunk in chunks:
                chunk_index = int(chunk["chunk_index"])
                stored_chunk_id = f"{doc_id}::v{version}::chunk_{chunk_index:03d}"
                metadata = chunk.get("metadata", {})
                conn.execute(
                    """
                    INSERT INTO chunks
                    (chunk_id, doc_id, version, title, content, chunk_index, category, access_level, source, content_hash, active, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                    """,
                    (
                        stored_chunk_id,
                        doc_id,
                        version,
                        chunk["title"],
                        chunk["content"],
                        chunk_index,
                        chunk["category"],
                        chunk["access_level"],
                        chunk["source"],
                        chunk["content_hash"],
                        json.dumps(metadata, sort_keys=True),
                        now,
                        now,
                    ),
                )
                inserted += 1

        return {"doc_id": doc_id, "version": version, "status": status, "chunks_inserted": inserted}

    def get_active_chunks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, title, content, chunk_index, category, access_level, source, content_hash, metadata_json
                FROM chunks
                WHERE active = 1
                ORDER BY doc_id ASC, chunk_index ASC
                """
            ).fetchall()

        chunks: list[dict] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            chunks.append(item)
        return chunks

    def record_ingestion_run(self, data_dir: str, documents_seen: int, documents_changed: int, chunks_active: int) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs (data_dir, documents_seen, documents_changed, chunks_active, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data_dir, documents_seen, documents_changed, chunks_active, self._now()),
            )
