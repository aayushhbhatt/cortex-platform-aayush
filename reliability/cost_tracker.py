from dataclasses import asdict, dataclass, field
import os

from reliability.errors import BudgetExceededError
from dotenv import load_dotenv
load_dotenv()

@dataclass
class CostEvent:
    component: str
    operation: str
    estimated_cost_usd: float
    metadata: dict = field(default_factory=dict)


@dataclass
class CostBreakdown:
    total_cost_usd: float
    events: list[CostEvent]
    max_budget_usd: float
    budget_exceeded: bool


def get_max_cost_per_query(default: float = 0.05) -> float:
    """Read MAX_COST_PER_QUERY_USD from env with safe default."""
    raw = os.getenv("MAX_COST_PER_QUERY_USD")
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value <= 0:
        return default
    return value


class CostTracker:
    def __init__(self, max_budget_usd: float | None = None):
        self.max_budget_usd = max_budget_usd if max_budget_usd is not None else get_max_cost_per_query()
        self.events: list[CostEvent] = []

    def add_cost(
        self,
        component: str,
        operation: str,
        estimated_cost_usd: float,
        metadata: dict | None = None,
    ) -> None:
        if not isinstance(component, str) or not component.strip():
            raise ValueError("component must be a non-empty string")
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("operation must be a non-empty string")
        if estimated_cost_usd < 0:
            raise ValueError("estimated_cost_usd must be >= 0")

        self.events.append(
            CostEvent(
                component=component.strip(),
                operation=operation.strip(),
                estimated_cost_usd=float(estimated_cost_usd),
                metadata=metadata or {},
            )
        )

    def total_cost(self) -> float:
        return sum(event.estimated_cost_usd for event in self.events)

    def check_budget(self) -> None:
        """Raise BudgetExceededError if total_cost > max_budget_usd."""
        total = self.total_cost()
        if total > self.max_budget_usd:
            raise BudgetExceededError(
                f"Estimated request cost ${total:.6f} exceeded max budget ${self.max_budget_usd:.6f}."
            )

    def breakdown(self) -> CostBreakdown:
        total = self.total_cost()
        return CostBreakdown(
            total_cost_usd=total,
            events=list(self.events),
            max_budget_usd=self.max_budget_usd,
            budget_exceeded=total > self.max_budget_usd,
        )

    def to_dict(self) -> dict:
        """Return JSON-serializable cost summary."""
        breakdown = self.breakdown()
        return {
            "total_cost_usd": breakdown.total_cost_usd,
            "events": [asdict(event) for event in breakdown.events],
            "max_budget_usd": breakdown.max_budget_usd,
            "budget_exceeded": breakdown.budget_exceeded,
        }


def estimate_retrieval_cost(result_count: int) -> float:
    """Small deterministic retrieval estimate."""
    safe_count = max(0, int(result_count))
    return 0.0001 + (0.00005 * safe_count)


def estimate_generation_cost(context_chars: int, output_chars: int) -> float:
    """Small deterministic generation estimate."""
    safe_context = max(0, int(context_chars))
    safe_output = max(0, int(output_chars))
    return 0.0002 + (0.000001 * (safe_context + safe_output))


def estimate_tool_cost(tool_results_count: int) -> float:
    """Small deterministic tool estimate."""
    safe_count = max(0, int(tool_results_count))
    return 0.0001 * safe_count


def estimate_memory_cost(message_count: int) -> float:
    """Small deterministic memory estimate."""
    safe_count = max(0, int(message_count))
    return 0.00002 * safe_count
