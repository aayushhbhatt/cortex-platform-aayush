import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.action_agent import action_agent
from agents.knowledge_agent import knowledge_agent
from agents.research_agent import research_agent
from agents.supervisor import create_initial_state


def test_action_agent_create_ticket_uses_tool_and_returns_tool_results() -> None:
    out = action_agent(create_initial_state("Create a support ticket for VPN issue"))
    assert out["agent_used"] == "action"
    assert out["tool_results"]


def test_research_agent_web_search_returns_tool_results() -> None:
    out = research_agent(create_initial_state("Find recent AI governance trends"))
    assert out["agent_used"] == "research"
    assert out["tool_results"]


def test_knowledge_agent_uses_knowledge_tool_and_returns_citations() -> None:
    out = knowledge_agent(create_initial_state("What is the parental leave policy?"))
    assert out["agent_used"] == "knowledge"
    assert isinstance(out.get("citations", []), list)
