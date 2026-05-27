from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_QUERY_LOG_PATH = "logs/cortex_queries.jsonl"


def get_log_path() -> Path:
    return Path(os.getenv("CORTEX_QUERY_LOG_PATH", DEFAULT_QUERY_LOG_PATH))


def safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return safe_jsonable(asdict(value))
    if isinstance(value, BaseException):
        return {
            "error_type": value.__class__.__name__,
            "message": str(value),
        }
    if isinstance(value, dict):
        return {str(k): safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_jsonable(item) for item in value]
    if isinstance(value, set):
        return [safe_jsonable(item) for item in sorted(value, key=lambda item: str(item))]
    if hasattr(value, "__dict__"):
        return safe_jsonable(vars(value))
    return str(value)


def log_query_event(event: dict[str, Any]) -> dict[str, Any]:
    try:
        final_event = safe_jsonable(dict(event or {}))
        if not isinstance(final_event, dict):
            final_event = {"event": final_event}

        final_event.setdefault("timestamp_utc", datetime.now(UTC).isoformat())

        log_path = get_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(final_event, ensure_ascii=False, sort_keys=True) + "\n")
        return final_event
    except Exception as exc:  # pragma: no cover - defensive failure path
        fallback_event = safe_jsonable(dict(event or {}))
        if not isinstance(fallback_event, dict):
            fallback_event = {"event": fallback_event}
        fallback_event.setdefault("timestamp_utc", datetime.now(UTC).isoformat())
        fallback_event["logging_error"] = {
            "error_type": exc.__class__.__name__,
            "message": str(exc),
        }
        return fallback_event
