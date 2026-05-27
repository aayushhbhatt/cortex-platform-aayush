import functools
import time

from reliability.errors import CortexError, RateLimitExceededError


def compute_backoff_delay(attempt_index: int, base_delay: float, backoff: float) -> float:
    return base_delay * (backoff ** attempt_index)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.1,
    backoff: float = 2.0,
    retryable_errors: tuple[type[Exception], ...] | None = None,
):
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def should_retry(exc: Exception) -> bool:
        if retryable_errors is not None:
            return isinstance(exc, retryable_errors)
        if isinstance(exc, (TimeoutError, RateLimitExceededError)):
            return True
        if isinstance(exc, CortexError):
            return bool(getattr(exc, "retryable", False))
        return False

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last = attempt == max_attempts - 1
                    if last or not should_retry(exc):
                        raise
                    time.sleep(compute_backoff_delay(attempt, base_delay, backoff))
        return wrapper

    return decorator
