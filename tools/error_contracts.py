from typing import Any

from pydantic import ValidationError

from reliability.errors import CortexError
from tools.schemas import ErrorType, ToolErrorResponse, ToolSuccessResponse


class ProviderRateLimit(Exception):
    def __init__(self, message: str, retry_after_seconds: float | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ProviderTimeout(Exception):
    pass


class ProviderAuthError(Exception):
    pass


class ProviderError(Exception):
    pass


class NoResultsError(Exception):
    pass


class AllSourcesFailedError(Exception):
    pass


def error_response(*, tool_name: str, error_type: ErrorType, message: str, recoverable: bool, retry_after_seconds: float | None = None, fallback_suggestion: str | None = None, details: dict[str, Any] | None = None) -> ToolErrorResponse:
    return ToolErrorResponse(
        tool_name=tool_name,
        error_type=error_type,
        message=message,
        recoverable=recoverable,
        retry_after_seconds=retry_after_seconds,
        fallback_suggestion=fallback_suggestion,
        details=details or {},
    )


def success_response(*, tool_name: str, data: Any, meta: dict[str, Any] | None = None) -> ToolSuccessResponse:
    return ToolSuccessResponse(tool_name=tool_name, data=data, meta=meta or {})


def validation_error_response(tool_name: str, exc: ValidationError) -> ToolErrorResponse:
    return error_response(
        tool_name=tool_name,
        error_type="validation_error",
        message="Invalid tool input. Fix the provided arguments before retrying.",
        recoverable=False,
        details={"errors": exc.errors()},
    )


def exception_to_error(*, tool_name: str, exc: Exception, fallback_suggestion: str | None = None) -> ToolErrorResponse:
    if isinstance(exc, ProviderRateLimit):
        return error_response(
            tool_name=tool_name,
            error_type="rate_limit",
            message=str(exc) or "Provider rate limit exceeded.",
            recoverable=True,
            retry_after_seconds=exc.retry_after_seconds,
            fallback_suggestion=fallback_suggestion or "Wait and retry the same tool after retry_after_seconds.",
            details={"exception_class": exc.__class__.__name__},
        )
    if isinstance(exc, (ProviderTimeout, TimeoutError)):
        return error_response(tool_name=tool_name, error_type="timeout", message=str(exc) or "Provider timed out.", recoverable=True, fallback_suggestion=fallback_suggestion or "Retry once or use an alternate source.", details={"exception_class": exc.__class__.__name__})
    if isinstance(exc, (ProviderAuthError, PermissionError)):
        return error_response(tool_name=tool_name, error_type="auth_error", message=str(exc) or "Authorization failed.", recoverable=False, fallback_suggestion=fallback_suggestion or "Check credentials or configuration before retrying.", details={"exception_class": exc.__class__.__name__})
    if isinstance(exc, ProviderError):
        return error_response(tool_name=tool_name, error_type="provider_error", message=str(exc) or "Provider error.", recoverable=True, fallback_suggestion=fallback_suggestion or "Retry later or use another provider.", details={"exception_class": exc.__class__.__name__})
    if isinstance(exc, NoResultsError):
        return error_response(tool_name=tool_name, error_type="no_results", message=str(exc) or "No results found.", recoverable=True, fallback_suggestion=fallback_suggestion or "Try a broader query or a different source.", details={"exception_class": exc.__class__.__name__})
    if isinstance(exc, AllSourcesFailedError):
        return error_response(tool_name=tool_name, error_type="all_sources_failed", message=str(exc) or "All sources failed.", recoverable=False, fallback_suggestion=fallback_suggestion or "Escalate to a human or report that no source is available.", details={"exception_class": exc.__class__.__name__})
    if isinstance(exc, CortexError):
        return error_response(tool_name=tool_name, error_type=getattr(exc, "error_type", "unknown"), message=str(exc) or "Cortex reliability error.", recoverable=getattr(exc, "retryable", False), fallback_suggestion=fallback_suggestion, details={"exception_class": exc.__class__.__name__})

    return error_response(tool_name=tool_name, error_type="unknown", message=str(exc).strip() or "An unexpected error occurred.", recoverable=False, fallback_suggestion=fallback_suggestion or "Do not retry blindly. Ask for clarification or escalate.", details={"exception_class": exc.__class__.__name__})
