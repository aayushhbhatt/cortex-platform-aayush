from langchain_core.tools import tool
from pydantic import ValidationError

from memory.context import load_memory_context
from memory.entity_store import create_entity_store
from memory.session_memory import create_session_memory
from tools.error_contracts import error_response, exception_to_error, success_response, validation_error_response
from tools.schemas import ForgetUserContextInput, UserContextInput


def get_user_context_impl(user_id: str, session_id: str | None = None, message_limit: int = 10) -> dict:
    tool_name = "get_user_context"
    try:
        inp = UserContextInput.model_validate(
            {"user_id": user_id, "session_id": session_id, "message_limit": message_limit}
        )
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    try:
        resolved_session = inp.session_id or "session_default"
        memory_context = load_memory_context(
            user_id=inp.user_id,
            session_id=resolved_session,
            message_limit=inp.message_limit,
        )
        return success_response(
            tool_name=tool_name,
            data={
                "user_id": inp.user_id,
                "session_id": resolved_session,
                "recent_messages": memory_context.recent_messages,
                "entities": memory_context.entities,
                "context_text": memory_context.context_text,
            },
            meta={"mode": "memory_context", "input_schema": "UserContextInput"},
        ).model_dump()
    except Exception as exc:
        return exception_to_error(tool_name=tool_name, exc=exc).model_dump()


def forget_user_context_impl(user_id: str, entity_type: str | None = None, include_session_messages: bool = False, session_id: str | None = None) -> dict:
    tool_name = "forget_user_context"
    try:
        inp = ForgetUserContextInput.model_validate({
            "user_id": user_id,
            "entity_type": entity_type,
            "include_session_messages": include_session_messages,
            "session_id": session_id,
        })
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    if inp.include_session_messages and not inp.session_id:
        return error_response(
            tool_name=tool_name,
            error_type="validation_error",
            message="session_id is required when include_session_messages is true.",
            recoverable=False,
            details={"field": "session_id"},
        ).model_dump()

    try:
        entity_store = create_entity_store(prefer_postgres=True)
        deleted_entity_count = entity_store.delete_entities(inp.user_id, inp.entity_type)
        session_cleared = False
        if inp.include_session_messages and inp.session_id:
            create_session_memory(prefer_redis=True).clear_session(inp.session_id)
            session_cleared = True
        return success_response(
            tool_name=tool_name,
            data={"user_id": inp.user_id, "entity_type": inp.entity_type, "deleted_entity_count": deleted_entity_count, "session_cleared": session_cleared},
            meta={"input_schema": "ForgetUserContextInput"},
        ).model_dump()
    except Exception as exc:
        return exception_to_error(tool_name=tool_name, exc=exc).model_dump()


@tool
def get_user_context(user_id: str, session_id: str | None = None, message_limit: int = 10) -> dict:
    """Load a user's stored memory context."""
    return get_user_context_impl(user_id=user_id, session_id=session_id, message_limit=message_limit)


@tool
def forget_user_context(user_id: str, entity_type: str | None = None, include_session_messages: bool = False, session_id: str | None = None) -> dict:
    """Forget a user's stored long-term memory facts."""
    return forget_user_context_impl(user_id=user_id, entity_type=entity_type, include_session_messages=include_session_messages, session_id=session_id)
