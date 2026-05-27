from typing import Any

from reliability.errors import ErrorResponse
from tools.error_contracts import ToolErrorResponse, ToolSuccessResponse
from tools.error_contracts import error_response as _error_response_model
from tools.error_contracts import success_response as _success_response_model


def success_response(tool_name: str, data: Any, meta: dict[str, Any] | None = None) -> dict:
    return _success_response_model(tool_name=tool_name, data=data, meta=meta).model_dump()


def error_response(tool_name: str, error: ErrorResponse | ToolErrorResponse) -> dict:
    if isinstance(error, ToolErrorResponse):
        return error.model_dump()
    return _error_response_model(
        tool_name=tool_name,
        error_type=error.error_type,
        message=error.message,
        recoverable=error.retryable,
        details=error.details,
    ).model_dump()


__all__ = ["ToolSuccessResponse", "ToolErrorResponse", "success_response", "error_response"]
