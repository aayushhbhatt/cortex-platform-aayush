import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reliability.cost_tracker import CostTracker, estimate_generation_cost, estimate_memory_cost, estimate_retrieval_cost, estimate_tool_cost
from reliability.errors import BudgetExceededError

def test_cost_tracker_adds_events_and_totals():
    t=CostTracker(max_budget_usd=1.0); t.add_cost("retrieval","retrieve",0.01); t.add_cost("generation","answer",0.02)
    assert t.total_cost()==pytest.approx(0.03)

def test_cost_tracker_budget_enforced():
    t=CostTracker(max_budget_usd=0.001); t.add_cost("tools","web_search",0.01)
    with pytest.raises(BudgetExceededError): t.check_budget()

def test_cost_estimator_functions_return_non_negative_values():
    assert estimate_retrieval_cost(3)>=0 and estimate_generation_cost(30,10)>=0 and estimate_tool_cost(1)>=0 and estimate_memory_cost(5)>=0
