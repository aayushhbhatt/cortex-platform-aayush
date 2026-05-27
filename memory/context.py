from __future__ import annotations

from dataclasses import dataclass, field
import uuid

from memory.entity_extraction import persist_extracted_entities
from memory.entity_store import create_entity_store
from memory.session_memory import create_session_memory


@dataclass
class MemoryContext:
    user_id: str
    session_id: str
    recent_messages: list[dict] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    context_text: str = ""
    debug: dict = field(default_factory=dict)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _generate_session_id() -> str:
    return f"session_{uuid.uuid4().hex[:12]}"


def build_context_text(recent_messages: list[dict], entities: list[dict], max_chars: int = 2000) -> str:
    if not recent_messages and not entities:
        return ""

    lines: list[str] = []
    if recent_messages:
        lines.append("Recent conversation:")
        for message in recent_messages:
            role = str(message.get("role", "unknown"))
            content = str(message.get("content", ""))
            lines.append(f"- {role}: {content}")

    if entities:
        if lines:
            lines.append("")
        lines.append("Known user facts:")
        for entity in entities:
            entity_type = str(entity.get("entity_type", "fact"))
            entity_value = str(entity.get("entity_value", ""))
            lines.append(f"- {entity_type}: {entity_value}")

    return _truncate("\n".join(lines), max_chars=max_chars)


def build_memory_aware_query(query: str, memory_context: MemoryContext) -> str:
    if not memory_context or (not memory_context.context_text and not memory_context.recent_messages and not memory_context.entities):
        return query

    query_l = query.lower().strip()
    ambiguous_terms = (
        " it", " that", " this", "how do i apply", "what about eligibility", "what about the process", "can you explain more", "tell me more",
    )
    if not any(term in f" {query_l}" for term in ambiguous_terms):
        return query

    topic_keywords = {
        "parental leave": ["parental leave", "primary caregiver", "secondary caregiver", "leave policy"],
        "remote work": ["remote work", "hybrid", "work from home", "home office"],
        "compensation and benefits": ["compensation", "benefits", "401k", "pto", "health insurance"],
        "performance review": ["performance review", "annual review", "mid-year", "rating"],
        "data security": ["data security", "confidential", "restricted data", "sensitive data"],
        "acceptable use": ["acceptable use", "company systems", "password", "mfa"],
        "software request": ["software request", "new software", "it portal", "approved vendor"],
        "company overview": ["company overview", "meridian", "mission", "products"],
        "manager handbook": ["manager handbook", "people leadership", "1:1", "hiring process"],
        "executive strategy": ["executive strategy", "ipo", "apac", "meridianai platform"],
    }

    memory_text = " ".join([
        memory_context.context_text,
        " ".join(str(m.get("content", "")) for m in memory_context.recent_messages),
        " ".join(str(e.get("entity_value", "")) for e in memory_context.entities),
    ]).lower()

    topic = None
    for candidate, keywords in topic_keywords.items():
        if any(k in memory_text for k in keywords):
            topic = candidate
            break
    if not topic:
        return query

    if "how do i apply" in query_l:
        return f"How do I apply for {topic}?"
    if "what about eligibility" in query_l:
        return f"What about eligibility for {topic}?"
    if "what about the process" in query_l:
        return f"What about the process for {topic}?"
    if "tell me more" in query_l or "can you explain more" in query_l:
        return f"Tell me more about {topic}."
    if any(x in query_l for x in [" it", " that", " this"]):
        return f"{query.strip()} about {topic}" if "about" not in query_l else f"{query.strip()} ({topic})"
    return query


def load_memory_context(
    user_id: str,
    session_id: str,
    session_memory=None,
    entity_store=None,
    message_limit: int = 10,
    max_context_chars: int = 2000,
) -> MemoryContext:
    recent_messages: list[dict] = []
    entities: list[dict] = []
    debug: dict = {"message_limit": message_limit, "max_context_chars": max_context_chars}

    memory_backend = session_memory
    if memory_backend is None:
        try:
            memory_backend = create_session_memory(prefer_redis=True)
        except Exception as exc:
            debug["session_memory_error"] = str(exc)
            memory_backend = None

    if memory_backend is not None:
        try:
            messages = memory_backend.get_messages(session_id, limit=message_limit)
            recent_messages = [
                {"role": m.role, "content": m.content, "created_at": m.created_at}
                for m in messages
            ]
        except Exception as exc:
            debug["session_memory_error"] = str(exc)

    entity_backend = entity_store
    if entity_backend is None:
        try:
            entity_backend = create_entity_store(prefer_postgres=True)
        except Exception as exc:
            debug["entity_store_error"] = str(exc)
            entity_backend = None

    if entity_backend is not None:
        try:
            entity_rows = entity_backend.get_entities(user_id)
            entities = [
                {
                    "user_id": row.user_id,
                    "entity_type": row.entity_type,
                    "entity_value": row.entity_value,
                    "attributes": row.attributes,
                    "updated_at": row.updated_at,
                }
                for row in entity_rows
            ]
        except Exception as exc:
            debug["entity_store_error"] = str(exc)

    context_text = build_context_text(recent_messages=recent_messages, entities=entities, max_chars=max_context_chars)
    return MemoryContext(
        user_id=user_id,
        session_id=session_id,
        recent_messages=recent_messages,
        entities=entities,
        context_text=context_text,
        debug=debug,
    )


def append_turn_to_memory(
    session_id: str,
    user_message: str,
    assistant_message: str,
    session_memory=None,
) -> list[dict]:
    memory_backend = session_memory
    if memory_backend is None:
        try:
            memory_backend = create_session_memory(prefer_redis=True)
        except Exception:
            return []

    try:
        existing = memory_backend.get_messages(session_id, limit=2)
        if len(existing) >= 2:
            last_user, last_assistant = existing[-2], existing[-1]
            if (
                last_user.role == "user"
                and last_user.content == user_message
                and last_assistant.role == "assistant"
                and last_assistant.content == assistant_message
            ):
                messages = memory_backend.get_messages(session_id, limit=10)
                return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]

        memory_backend.append_message(session_id, "user", user_message)
        memory_backend.append_message(session_id, "assistant", assistant_message)
        messages = memory_backend.get_messages(session_id, limit=10)
        return [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages]
    except Exception:
        return []


def finalize_agent_memory(user_id: str, session_id: str, query: str, response: str, session_memory=None, entity_store=None) -> dict:
    debug = {"session_append_success": False, "entity_persist_success": False, "persisted_entity_count": 0, "errors": []}
    messages: list[dict] = []
    persisted: list[dict] = []
    try:
        messages = append_turn_to_memory(session_id, query, response, session_memory=session_memory)
        debug["session_append_success"] = True
    except Exception as exc:
        debug["errors"].append(f"session_append: {exc}")

    try:
        persisted = persist_extracted_entities(user_id=user_id, text=query, entity_store=entity_store)
        debug["entity_persist_success"] = True
        debug["persisted_entity_count"] = len(persisted)
    except Exception as exc:
        debug["errors"].append(f"entity_persist: {exc}")

    return {"memory_messages": messages, "persisted_entities": persisted, "memory_debug": debug}
