from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from typing import Protocol
from dotenv import load_dotenv
import redis

load_dotenv()
@dataclass
class SessionMessage:
    role: str
    content: str
    created_at: str


class SessionMemoryBackend(Protocol):
    def append_message(self, session_id: str, role: str, content: str) -> None: ...

    def get_messages(self, session_id: str, limit: int = 20) -> list[SessionMessage]: ...

    def clear_session(self, session_id: str) -> None: ...


class InMemorySessionMemory:
    """In-memory fallback session memory for tests/local use."""

    def __init__(self) -> None:
        self._messages: dict[str, list[SessionMessage]] = {}

    def append_message(self, session_id: str, role: str, content: str) -> None:
        message = SessionMessage(
            role=role,
            content=content,
            created_at=datetime.now(UTC).isoformat(),
        )
        self._messages.setdefault(session_id, []).append(message)

    def get_messages(self, session_id: str, limit: int = 20) -> list[SessionMessage]:
        if limit <= 0:
            return []
        messages = self._messages.get(session_id, [])
        return messages[-limit:]

    def clear_session(self, session_id: str) -> None:
        self._messages.pop(session_id, None)


class RedisSessionMemory:
    """Redis-backed session memory."""

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 3600) -> None:
        self._redis_url = redis_url or os.getenv("REDIS_URL")
        self._ttl_seconds = ttl_seconds

    def _get_client(self) -> redis.Redis:
        if not self._redis_url:
            raise ValueError("REDIS_URL is required to use RedisSessionMemory.")
        return redis.Redis.from_url(self._redis_url, decode_responses=True)

    @staticmethod
    def _session_key(session_id: str) -> str:
        return f"cortex:session:{session_id}:messages"

    def append_message(self, session_id: str, role: str, content: str) -> None:
        message = SessionMessage(
            role=role,
            content=content,
            created_at=datetime.now(UTC).isoformat(),
        )
        key = self._session_key(session_id)
        client = self._get_client()
        client.rpush(key, json.dumps(asdict(message)))
        client.expire(key, self._ttl_seconds)

    def get_messages(self, session_id: str, limit: int = 20) -> list[SessionMessage]:
        if limit <= 0:
            return []
        key = self._session_key(session_id)
        start = -limit
        raw_messages = self._get_client().lrange(key, start, -1)
        return [SessionMessage(**json.loads(raw)) for raw in raw_messages]

    def clear_session(self, session_id: str) -> None:
        key = self._session_key(session_id)
        self._get_client().delete(key)


_SHARED_IN_MEMORY_SESSION_MEMORY: InMemorySessionMemory | None = None


def create_session_memory(prefer_redis: bool = True) -> SessionMemoryBackend:
    """Create Redis session memory if configured, otherwise shared in-memory fallback."""
    if prefer_redis and os.getenv("REDIS_URL"):
        return RedisSessionMemory()

    global _SHARED_IN_MEMORY_SESSION_MEMORY
    if _SHARED_IN_MEMORY_SESSION_MEMORY is None:
        _SHARED_IN_MEMORY_SESSION_MEMORY = InMemorySessionMemory()
    return _SHARED_IN_MEMORY_SESSION_MEMORY


def reset_in_memory_session_memory() -> None:
    """Reset the shared in-memory session memory instance (for tests only)."""
    global _SHARED_IN_MEMORY_SESSION_MEMORY
    _SHARED_IN_MEMORY_SESSION_MEMORY = InMemorySessionMemory()
