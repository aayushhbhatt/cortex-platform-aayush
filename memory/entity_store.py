from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from typing import Protocol
from dotenv import load_dotenv
import psycopg

load_dotenv()

@dataclass
class EntityRecord:
    user_id: str
    entity_type: str
    entity_value: str
    attributes: dict
    updated_at: str


class EntityStoreBackend(Protocol):
    def upsert_entity(
        self,
        user_id: str,
        entity_type: str,
        entity_value: str,
        attributes: dict | None = None,
    ) -> EntityRecord: ...

    def get_entities(self, user_id: str, entity_type: str | None = None) -> list[EntityRecord]: ...

    def delete_entities(self, user_id: str, entity_type: str | None = None) -> int: ...


class InMemoryEntityStore:
    """In-memory fallback entity store for tests/local use."""

    def __init__(self) -> None:
        self._records: dict[tuple[str, str, str], EntityRecord] = {}

    def upsert_entity(
        self,
        user_id: str,
        entity_type: str,
        entity_value: str,
        attributes: dict | None = None,
    ) -> EntityRecord:
        record = EntityRecord(
            user_id=user_id,
            entity_type=entity_type,
            entity_value=entity_value,
            attributes=attributes or {},
            updated_at=datetime.now(UTC).isoformat(),
        )
        self._records[(user_id, entity_type, entity_value)] = record
        return record

    def get_entities(self, user_id: str, entity_type: str | None = None) -> list[EntityRecord]:
        records = [
            record
            for (record_user_id, record_entity_type, _), record in self._records.items()
            if record_user_id == user_id and (entity_type is None or record_entity_type == entity_type)
        ]
        records.sort(key=lambda record: (record.entity_type, record.entity_value))
        return records

    def delete_entities(self, user_id: str, entity_type: str | None = None) -> int:
        keys_to_delete = [
            key for key in self._records
            if key[0] == user_id and (entity_type is None or key[1] == entity_type)
        ]
        for key in keys_to_delete:
            del self._records[key]
        return len(keys_to_delete)


class PostgresEntityStore:
    """PostgreSQL-backed entity memory."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = database_url or os.getenv("DATABASE_URL")
        self._initialized = False

    def _get_connection(self) -> psycopg.Connection:
        if not self._database_url:
            raise ValueError("DATABASE_URL is required to use PostgresEntityStore.")
        return psycopg.connect(self._database_url)

    def initialize(self) -> None:
        if self._initialized:
            return
        if not self._database_url:
            raise ValueError("DATABASE_URL is required to initialize PostgresEntityStore.")

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cortex_entities (
                        user_id TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_value TEXT NOT NULL,
                        attributes_json TEXT NOT NULL,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        PRIMARY KEY (user_id, entity_type, entity_value)
                    )
                    """
                )
            conn.commit()
        self._initialized = True

    def upsert_entity(
        self,
        user_id: str,
        entity_type: str,
        entity_value: str,
        attributes: dict | None = None,
    ) -> EntityRecord:
        self.initialize()
        now_iso = datetime.now(UTC).isoformat()
        attributes_payload = json.dumps(attributes or {})

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO cortex_entities (user_id, entity_type, entity_value, attributes_json, updated_at)
                    VALUES (%s, %s, %s, %s, now())
                    ON CONFLICT (user_id, entity_type, entity_value)
                    DO UPDATE SET
                        attributes_json = EXCLUDED.attributes_json,
                        updated_at = now()
                    """,
                    (user_id, entity_type, entity_value, attributes_payload),
                )
            conn.commit()

        return EntityRecord(
            user_id=user_id,
            entity_type=entity_type,
            entity_value=entity_value,
            attributes=attributes or {},
            updated_at=now_iso,
        )

    def get_entities(self, user_id: str, entity_type: str | None = None) -> list[EntityRecord]:
        self.initialize()

        query = """
            SELECT user_id, entity_type, entity_value, attributes_json, updated_at
            FROM cortex_entities
            WHERE user_id = %s
        """
        params: list[str] = [user_id]
        if entity_type is not None:
            query += " AND entity_type = %s"
            params.append(entity_type)
        query += " ORDER BY entity_type, entity_value"

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        return [
            EntityRecord(
                user_id=row[0],
                entity_type=row[1],
                entity_value=row[2],
                attributes=json.loads(row[3]),
                updated_at=row[4].isoformat(),
            )
            for row in rows
        ]

    def delete_entities(self, user_id: str, entity_type: str | None = None) -> int:
        self.initialize()

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                if entity_type is None:
                    cur.execute("DELETE FROM cortex_entities WHERE user_id = %s", (user_id,))
                else:
                    cur.execute("DELETE FROM cortex_entities WHERE user_id = %s AND entity_type = %s", (user_id, entity_type))
                deleted = cur.rowcount
            conn.commit()
        return deleted


_SHARED_IN_MEMORY_ENTITY_STORE: InMemoryEntityStore | None = None


def create_entity_store(prefer_postgres: bool = True) -> EntityStoreBackend:
    """Create Postgres entity store if configured, otherwise shared in-memory fallback."""
    if prefer_postgres and os.getenv("DATABASE_URL"):
        return PostgresEntityStore()

    global _SHARED_IN_MEMORY_ENTITY_STORE
    if _SHARED_IN_MEMORY_ENTITY_STORE is None:
        _SHARED_IN_MEMORY_ENTITY_STORE = InMemoryEntityStore()
    return _SHARED_IN_MEMORY_ENTITY_STORE


def reset_in_memory_entity_store() -> None:
    """Reset the shared in-memory entity store instance (for tests only)."""
    global _SHARED_IN_MEMORY_ENTITY_STORE
    _SHARED_IN_MEMORY_ENTITY_STORE = InMemoryEntityStore()
