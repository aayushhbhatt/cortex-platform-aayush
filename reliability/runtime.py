from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reliability.circuit_breaker import CircuitBreaker
from reliability.errors import CircuitOpenError, RateLimitExceededError
from reliability.rate_limiter import RateLimiter
from reliability.retry import with_retry
from tools.error_contracts import exception_to_error


class RuntimeReliabilityPolicy:
    def __init__(
        self,
        tool_name: str,
        rate_limiter: RateLimiter | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_attempts: int = 3,
        base_delay: float = 0.05,
        backoff: float = 2.0,
        rate_limit_wait: bool = False,
        rate_limit_timeout: float = 1.0,
    ):
        self.tool_name = tool_name
        self.rate_limiter = rate_limiter or RateLimiter(capacity=10, refill_rate=5)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30)
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.backoff = backoff
        self.rate_limit_wait = rate_limit_wait
        self.rate_limit_timeout = rate_limit_timeout


ACTION_TOOL_POLICY = RuntimeReliabilityPolicy(
    tool_name="create_support_ticket",
    rate_limiter=RateLimiter(capacity=10, refill_rate=5),
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30),
)

RESEARCH_TOOL_POLICY = RuntimeReliabilityPolicy(
    tool_name="web_search",
    rate_limiter=RateLimiter(capacity=100, refill_rate=100),
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30),
)

KNOWLEDGE_TOOL_POLICY = RuntimeReliabilityPolicy(
    tool_name="search_knowledge_base",
    rate_limiter=RateLimiter(capacity=20, refill_rate=10),
    circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=30),
)


def execute_with_reliability(
    *,
    tool_name: str,
    operation: Callable[[], dict[str, Any]],
    policy: RuntimeReliabilityPolicy | None = None,
    fallback_suggestion: str | None = None,
) -> dict:
    """Execute validated operation with runtime reliability protections."""
    runtime_policy = policy or RuntimeReliabilityPolicy(tool_name=tool_name)

    try:
        runtime_policy.rate_limiter.acquire(
            tokens=1.0,
            wait=runtime_policy.rate_limit_wait,
            timeout=runtime_policy.rate_limit_timeout,
        )
    except RateLimitExceededError as exc:
        return exception_to_error(
            tool_name=tool_name,
            exc=exc,
            fallback_suggestion=fallback_suggestion,
        ).model_dump()

    @with_retry(
        max_attempts=runtime_policy.max_attempts,
        base_delay=runtime_policy.base_delay,
        backoff=runtime_policy.backoff,
    )
    def _call_with_circuit() -> dict:
        return runtime_policy.circuit_breaker.call(operation)

    try:
        return _call_with_circuit()
    except CircuitOpenError as exc:
        return exception_to_error(
            tool_name=tool_name,
            exc=exc,
            fallback_suggestion=fallback_suggestion,
        ).model_dump()
    except Exception as exc:
        return exception_to_error(
            tool_name=tool_name,
            exc=exc,
            fallback_suggestion=fallback_suggestion,
        ).model_dump()
