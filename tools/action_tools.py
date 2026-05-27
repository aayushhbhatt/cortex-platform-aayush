from datetime import datetime
import hashlib
import uuid

from langchain_core.tools import tool
from pydantic import ValidationError

from reliability.runtime import ACTION_TOOL_POLICY, execute_with_reliability
from tools.error_contracts import success_response, validation_error_response
from tools.schemas import EscalateIssueInput, NotificationInput, SupportTicketInput, TicketStatusInput


def _deterministic_suffix(seed: str) -> str:
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:6].upper()


def _create_support_ticket_validated(inp: SupportTicketInput) -> dict:
    ticket_id = f"TKT-{datetime.utcnow().strftime('%Y%m%d')}-{str(uuid.uuid4())[:6].upper()}"
    return success_response(
        tool_name="create_support_ticket",
        data={
            "ticket_id": ticket_id,
            "category": inp.category,
            "priority": inp.priority,
            "title": inp.title,
            "expected_response_time": "2 hours" if inp.priority == "urgent" else "1 business day",
        },
        meta={"mode": "offline_mock", "input_schema": "SupportTicketInput"},
    ).model_dump()


def create_support_ticket_impl(title: str, description: str, category: str, priority: str = "medium", user_id: str = "unknown") -> dict:
    tool_name = "create_support_ticket"
    try:
        inp = SupportTicketInput.model_validate({"title": title, "description": description, "category": category, "priority": priority, "user_id": user_id})
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()
    return execute_with_reliability(tool_name=tool_name, operation=lambda: _create_support_ticket_validated(inp), policy=ACTION_TOOL_POLICY, fallback_suggestion="Try again later or contact the service desk manually.")


def get_ticket_status_impl(ticket_id: str) -> dict:
    tool_name = "get_ticket_status"
    try:
        inp = TicketStatusInput.model_validate({"ticket_id": ticket_id})
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    statuses = ["open", "in_progress", "waiting_on_requester", "resolved"]

    def _op() -> dict:
        idx = int(hashlib.sha256(inp.ticket_id.encode("utf-8")).hexdigest(), 16) % len(statuses)
        return success_response(
            tool_name=tool_name,
            data={"ticket_id": inp.ticket_id, "status": statuses[idx], "last_updated": "offline_mock", "message": "Ticket status lookup is running in offline mock mode."},
            meta={"mode": "offline_mock", "input_schema": "TicketStatusInput"},
        ).model_dump()

    return execute_with_reliability(tool_name=tool_name, operation=_op, policy=ACTION_TOOL_POLICY, fallback_suggestion="Retry ticket status lookup shortly.")


def escalate_issue_impl(issue: str, ticket_id: str | None = None, priority: str = "high", user_id: str = "unknown") -> dict:
    tool_name = "escalate_issue"
    try:
        inp = EscalateIssueInput.model_validate({"issue": issue, "ticket_id": ticket_id, "priority": priority, "user_id": user_id})
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    def _op() -> dict:
        date = datetime.utcnow().strftime("%Y%m%d")
        esc_id = f"ESC-{date}-{_deterministic_suffix(inp.issue + (inp.ticket_id or '') + inp.priority)}"
        return success_response(
            tool_name=tool_name,
            data={"escalation_id": esc_id, "ticket_id": inp.ticket_id, "priority": inp.priority, "issue": inp.issue, "message": "Issue escalation recorded in offline mock mode."},
            meta={"mode": "offline_mock", "input_schema": "EscalateIssueInput"},
        ).model_dump()

    return execute_with_reliability(tool_name=tool_name, operation=_op, policy=ACTION_TOOL_POLICY, fallback_suggestion="Retry escalation request.")


def send_notification_impl(recipient: str, message: str, channel: str = "system", user_id: str = "unknown") -> dict:
    tool_name = "send_notification"
    try:
        inp = NotificationInput.model_validate({"recipient": recipient, "message": message, "channel": channel, "user_id": user_id})
    except ValidationError as exc:
        return validation_error_response(tool_name, exc).model_dump()

    def _op() -> dict:
        date = datetime.utcnow().strftime("%Y%m%d")
        nid = f"NTF-{date}-{_deterministic_suffix(inp.recipient + inp.message + inp.channel)}"
        return success_response(
            tool_name=tool_name,
            data={"notification_id": nid, "recipient": inp.recipient, "channel": inp.channel, "message": inp.message, "delivery_status": "mock_sent"},
            meta={"mode": "offline_mock", "input_schema": "NotificationInput"},
        ).model_dump()

    return execute_with_reliability(tool_name=tool_name, operation=_op, policy=ACTION_TOOL_POLICY, fallback_suggestion="Retry notification request.")


@tool
def create_support_ticket(title: str, description: str, category: str, priority: str = "medium", user_id: str = "unknown") -> dict:
    """Create an internal support ticket.

    When to use: explicit ticket creation/logging requests.
    When NOT to use: read-only ticket lookups/status checks.
    Inputs: title, description, category, priority, user_id.
    Returns: ToolSuccessResponse or ToolErrorResponse.
    Example: create_support_ticket(title='VPN Issue', description='Cannot connect ...', category='IT').
    """
    return create_support_ticket_impl(title, description, category, priority, user_id)


@tool
def get_ticket_status(ticket_id: str) -> dict:
    """Get deterministic offline mock status for an existing ticket.

    When to use: user asks for ticket status/check.
    When NOT to use: creating new tickets.
    Inputs: ticket_id.
    Returns: ToolSuccessResponse or ToolErrorResponse.
    Example: get_ticket_status(ticket_id='TKT-20260524-9EEA16').
    """
    return get_ticket_status_impl(ticket_id)


@tool
def escalate_issue(issue: str, ticket_id: str | None = None, priority: str = "high", user_id: str = "unknown") -> dict:
    """Record deterministic offline escalation.

    When to use: urgent/blocking escalation requests.
    When NOT to use: status lookups.
    Inputs: issue, ticket_id, priority, user_id.
    Returns: ToolSuccessResponse or ToolErrorResponse.
    Example: escalate_issue(issue='VPN outage blocking production', priority='urgent').
    """
    return escalate_issue_impl(issue, ticket_id, priority, user_id)


@tool
def send_notification(recipient: str, message: str, channel: str = "system", user_id: str = "unknown") -> dict:
    """Record deterministic offline notification.

    When to use: notify/alert workflow triggers.
    When NOT to use: creating support tickets.
    Inputs: recipient, message, channel, user_id.
    Returns: ToolSuccessResponse or ToolErrorResponse.
    Example: send_notification(recipient='it-team', message='VPN down', channel='slack').
    """
    return send_notification_impl(recipient, message, channel, user_id)
