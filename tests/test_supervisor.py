import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.supervisor import build_supervisor_graph, create_initial_state


def test_create_initial_state_contains_required_fields() -> None:
    state = create_initial_state("hello")
    required = {
        "query", "user_id", "user_tier", "intent", "session_id", "request_id", "context", "response",
        "cost_usd", "agent_used", "tool_results", "citations", "used_chunks", "memory_context",
        "memory_context_text", "debug", "trace",
    }
    assert required.issubset(state.keys())


def test_supervisor_routes_supported_agents() -> None:
    graph = build_supervisor_graph()
    assert graph.invoke(create_initial_state("What is the parental leave policy?"))["agent_used"] == "knowledge"
    assert graph.invoke(create_initial_state("Create a support ticket for VPN issue"))["agent_used"] == "action"
    assert graph.invoke(create_initial_state("Find recent AI governance trends"))["agent_used"] == "research"


def test_supervisor_routes_unclear_to_unsupported() -> None:
    final_state = build_supervisor_graph().invoke(create_initial_state("???"))
    assert final_state["intent"] == "unsupported" or final_state["agent_used"] == "unsupported"
    assert final_state["tool_results"] == []
