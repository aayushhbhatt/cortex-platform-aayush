import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.entity_store import InMemoryEntityStore
from memory.session_memory import InMemorySessionMemory


def test_session_memory_append_and_recall_with_fallback() -> None:
    sm = InMemorySessionMemory()
    sm.append_message("s1", "user", "hello")
    sm.append_message("s1", "assistant", "hi")
    msgs = sm.get_messages("s1")
    assert len(msgs) == 2
    assert msgs[0].content == "hello" and msgs[1].content == "hi"


def test_entity_store_upsert_and_recall_with_fallback() -> None:
    es = InMemoryEntityStore()
    es.upsert_entity("u1", "project", "Cortex", {"status": "active"})
    entities = es.get_entities("u1")
    assert entities and entities[0].entity_value == "Cortex"
