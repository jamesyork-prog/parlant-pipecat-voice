
import pytest
from unittest.mock import Mock
from parlant.tools import freshdesk_tools
from parlant.tools.freshdesk_tools import get_ticket, add_note, update_ticket

MOCK_FRESHDESK_DOMAIN = "test.freshdesk.com"
MOCK_FRESHDESK_API_KEY = "test_api_key"

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setattr(freshdesk_tools, "FRESHDESK_DOMAIN", MOCK_FRESHDESK_DOMAIN)
    monkeypatch.setattr(freshdesk_tools, "FRESHDESK_API_KEY", MOCK_FRESHDESK_API_KEY)


@pytest.mark.asyncio
async def test_get_ticket_success(mock_env, httpx_mock):
    ticket_id = 123
    # Real Freshdesk API response structure from ticket 1205974
    mock_response = {
        "id": ticket_id,
        "subject": "ParkWhiz refund request from the online form",
        "description": "<div>ParkWhiz refund request received.</div>",
        "description_text": "ParkWhiz refund request received.",
        "status": 2,
        "priority": 1,
        "source": 2,
        "requester_id": 60068011282,
        "responder_id": None,
        "group_id": 60000422084,
        "product_id": 60000010037,
        "company_id": None,
        "type": None,
        "email_config_id": 60000065244,
        "fr_escalated": False,
        "spam": False,
        "is_escalated": False,
        "due_by": "2025-11-14T15:49:27Z",
        "fr_due_by": "2025-11-08T15:49:27Z",
        "created_at": "2025-11-07T15:49:27Z",
        "updated_at": "2025-11-07T15:49:32Z",
        "tags": ["Refund", "ParkWhizRefundForm"],
        "cc_emails": [],
        "fwd_emails": [],
        "reply_cc_emails": [],
        "custom_fields": {
            "cf_has_location": "No",
            "cf_case_deflected": False,
            "cf_for_internal_use": False,
        },
        "attachments": [],
    }
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}",
        json=mock_response,
        status_code=200,
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id}
    result = await get_ticket(context)

    assert result.data == mock_response
    assert result.metadata["summary"] == f"Fetched ticket details for {ticket_id}"


@pytest.mark.asyncio
async def test_get_ticket_failure(mock_env, httpx_mock):
    ticket_id = 456
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}",
        status_code=404,
        text="Not Found",
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id}
    result = await get_ticket(context)

    assert "error" in result.data
    assert result.data["error"] == "Failed to fetch ticket: 404"
    assert result.metadata["summary"] == f"Error fetching ticket {ticket_id}"


@pytest.mark.asyncio
async def test_add_note_success(mock_env, httpx_mock):
    ticket_id = 789
    note = "This is a test note."
    mock_response = {"id": 1, "body": note}
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes",
        json=mock_response,
        status_code=201,
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id, "note": note}
    result = await add_note(context)

    assert result.data == mock_response
    assert result.metadata["summary"] == f"Added note to ticket {ticket_id}"


@pytest.mark.asyncio
async def test_add_note_failure(mock_env, httpx_mock):
    ticket_id = 101
    note = "This is a failing test note."
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}/notes",
        status_code=500,
        text="Internal Server Error",
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id, "note": note}
    result = await add_note(context)

    assert "error" in result.data
    assert result.data["error"] == "Failed to add note: 500"
    assert result.metadata["summary"] == f"Error adding note to ticket {ticket_id}"


@pytest.mark.asyncio
async def test_update_ticket_success(mock_env, httpx_mock):
    ticket_id = 112
    status = 4  # Closed
    mock_response = {"id": ticket_id, "status": status}
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}",
        json=mock_response,
        status_code=200,
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id, "status": status}
    result = await update_ticket(context)

    assert result.data == mock_response
    assert result.metadata["summary"] == f"Updated ticket {ticket_id}"


@pytest.mark.asyncio
async def test_update_ticket_failure(mock_env, httpx_mock):
    ticket_id = 131
    status = 4  # Closed
    httpx_mock.add_response(
        url=f"https://{MOCK_FRESHDESK_DOMAIN}/api/v2/tickets/{ticket_id}",
        status_code=400,
        text="Bad Request",
    )

    context = Mock()
    context.inputs = {"ticket_id": ticket_id, "status": status}
    result = await update_ticket(context)

    assert "error" in result.data
    assert result.data["error"] == "Failed to update ticket: 400"
    assert result.metadata["summary"] == f"Error updating ticket {ticket_id}"

@pytest.mark.asyncio
async def test_get_ticket_no_credentials(monkeypatch):
    monkeypatch.setattr(freshdesk_tools, "FRESHDESK_DOMAIN", None)
    monkeypatch.setattr(freshdesk_tools, "FRESHDESK_API_KEY", None)
    context = Mock()
    context.inputs = {"ticket_id": 123}
    result = await get_ticket(context)
    assert result.data["error"] == "Freshdesk credentials not configured."
    assert result.metadata["summary"] == "Error: Freshdesk credentials not configured."
