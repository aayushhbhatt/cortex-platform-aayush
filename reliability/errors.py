from dataclasses import dataclass, field
from typing import Literal

ErrorType = Literal[
    "rate_limit",
    "timeout",
    "context_overflow",
    "auth_error",
    "validation_error",
    "provider_error",
    "no_results",
    "all_sources_failed",
    "circuit_open",
    "budget_exceeded",
    "unknown",
]


@dataclass
class ErrorResponse:
    error_type: ErrorType
    message: str
    retryable: bool
    details: dict = field(default_factory=dict)


class CortexError(Exception):
    error_type = "unknown"
    retryable = False


class RateLimitExceededError(CortexError):
    error_type = "rate_limit"
    retryable = True


class TimeoutCortexError(CortexError):
    error_type = "timeout"
    retryable = True


class ContextOverflowError(CortexError):
    error_type = "context_overflow"
    retryable = False


class AuthCortexError(CortexError):
    error_type = "auth_error"
    retryable = False


class ValidationCortexError(CortexError):
    error_type = "validation_error"
    retryable = False


class CircuitOpenError(CortexError):
    error_type = "circuit_open"
    retryable = True


class BudgetExceededError(CortexError):
    error_type = "budget_exceeded"
    retryable = False


def classify_error(exc: Exception) -> ErrorResponse:
    if isinstance(exc, CortexError):
        error_type = exc.error_type
        retryable = exc.retryable
    elif isinstance(exc, TimeoutError):
        error_type = "timeout"
        retryable = True
    elif isinstance(exc, PermissionError):
        error_type = "auth_error"
        retryable = False
    elif isinstance(exc, ValueError):
        error_type = "validation_error"
        retryable = False
    else:
        error_type = "unknown"
        retryable = False

    message = str(exc).strip() or "An unexpected error occurred."
    if len(message) > 300:
        message = message[:300].rstrip() + "..."

    return ErrorResponse(
        error_type=error_type,
        message=message,
        retryable=retryable,
        details={"exception_class": exc.__class__.__name__},
    )
