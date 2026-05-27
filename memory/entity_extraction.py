from __future__ import annotations

from dataclasses import dataclass, field
import re

from memory.entity_store import create_entity_store


@dataclass
class ExtractedEntity:
    entity_type: str
    entity_value: str
    attributes: dict = field(default_factory=dict)


_SENSITIVE_HINTS = {
    "democrat", "republican", "muslim", "christian", "jewish", "hindu", "buddhist",
    "cancer", "diabetes", "depression", "anxiety", "ethnicity", "race", "pregnant",
    "mental health", "therapy", "psychiatric", "politics", "religion", "sexuality",
    "union", "criminal",
}


_PATTERNS = [
    ("department", re.compile(r"\bmy department is\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("department", re.compile(r"\bi work in\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("role", re.compile(r"\bmy role is\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("role", re.compile(r"\bi am an?\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("project", re.compile(r"\bmy project is\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("project", re.compile(r"\bi am working on\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("project", re.compile(r"\bi work on\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("team", re.compile(r"\bi['’]?m part of the\s+([\w\s\-&,]+?)\s+team\b", re.IGNORECASE)),
    ("team", re.compile(r"\bi am part of the\s+([\w\s\-&,]+?)\s+team\b", re.IGNORECASE)),
    ("team", re.compile(r"\bmy team is\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("preference", re.compile(r"\bi prefer\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("location", re.compile(r"\bmy location is\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("work_style", re.compile(r"\bi usually work from\s+([\w\s\-&,]+)", re.IGNORECASE)),
    ("note", re.compile(r"\bremember that\s+([\w\s\-&,]+)", re.IGNORECASE)),
]

def _normalize(value: str) -> str:
    value = value.strip(" .,!;:\n\t")
    return re.sub(r"\s+", " ", value)


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return any(hint in lowered for hint in _SENSITIVE_HINTS)


def extract_entities_from_text(text: str) -> list[ExtractedEntity]:
    extracted: list[ExtractedEntity] = []
    for entity_type, pattern in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        value = _normalize(match.group(1))
        if not value or _looks_sensitive(value):
            continue
        extracted.append(ExtractedEntity(entity_type=entity_type, entity_value=value))
    unique: dict[tuple[str, str], ExtractedEntity] = {}
    for entity in extracted:
        unique[(entity.entity_type.lower(), entity.entity_value.lower())] = entity
    return list(unique.values())


def persist_extracted_entities(
    user_id: str,
    text: str,
    entity_store=None,
) -> list[dict]:
    entities = extract_entities_from_text(text)
    if not entities:
        return []

    store = entity_store
    if store is None:
        try:
            store = create_entity_store(prefer_postgres=True)
        except Exception:
            return []

    persisted: list[dict] = []
    try:
        for entity in entities:
            row = store.upsert_entity(
                user_id=user_id,
                entity_type=entity.entity_type,
                entity_value=entity.entity_value,
                attributes=entity.attributes,
            )
            persisted.append(
                {
                    "user_id": row.user_id,
                    "entity_type": row.entity_type,
                    "entity_value": row.entity_value,
                    "attributes": row.attributes,
                    "updated_at": row.updated_at,
                }
            )
    except Exception:
        return []
    return persisted
