import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory.context import load_memory_context
from memory.entity_store import InMemoryEntityStore
from memory.session_memory import InMemorySessionMemory


def test_memory_context_loads_messages_and_entities() -> None:
    sm = InMemorySessionMemory(); es = InMemoryEntityStore()
    sm.append_message("s1", "user", "my project is Cortex")
    es.upsert_entity("u1", "project", "Cortex")
    ctx = load_memory_context("u1", "s1", session_memory=sm, entity_store=es)
    assert ctx.recent_messages and ctx.entities
    assert isinstance(ctx.context_text, str)
    assert len(ctx.context_text) > 0


def test_memory_context_degrades_on_backend_failure() -> None:
    class Bad:
        def get_messages(self, *args, **kwargs):
            raise RuntimeError("down")
        def get_entities(self, *args, **kwargs):
            raise RuntimeError("down")

    ctx = load_memory_context("u1", "s1", session_memory=Bad(), entity_store=Bad())
    assert ctx.recent_messages == [] and ctx.entities == []
