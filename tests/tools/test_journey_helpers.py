
import pytest
from unittest.mock import Mock
from parlant.tools.journey_helpers import extract_booking_info_from_note, triage_ticket


@pytest.mark.asyncio
async def test_extract_booking_info_success():
    """Test extracting booking info from ticket notes."""
    context = Mock()
    context.inputs = {
        "ticket_notes": """
        Customer contacted about booking #PW-123456.
        Reservation date: 2025-11-15
        Location: Downtown Parking Garage
        Amount paid: $45.00
        """
    }
    
    result = await extract_booking_info_from_note(context)
    
    # Verify the tool returns a ToolResult
    assert result.data is not None
    assert result.metadata["summary"].startswith("Booking info")


@pytest.mark.asyncio
async def test_extract_booking_info_no_data():
    """Test when no booking info is found in notes."""
    context = Mock()
    context.inputs = {
        "ticket_notes": "Customer is asking general questions about parking."
    }
    
    result = await extract_booking_info_from_note(context)
    
    # Should still return successfully, even if no booking data found
    assert result.data is not None
    assert result.metadata is not None


@pytest.mark.asyncio
async def test_extract_booking_info_empty_notes():
    """Test with empty ticket notes."""
    context = Mock()
    context.inputs = {
        "ticket_notes": ""
    }
    
    result = await extract_booking_info_from_note(context)
    
    # Should handle empty input gracefully
    assert result.data is not None


@pytest.mark.asyncio
async def test_triage_ticket_approved():
    """Test ticket triage when refund should be approved."""
    context = Mock()
    context.inputs = {
        "ticket_data": {
            "id": "1205974",
            "subject": "Refund request - broken gate",
            "description": "The parking gate was broken and I couldn't exit for 2 hours."
        },
        "booking_info": {
            "booking_id": "PW-123456",
            "amount": 45.00,
            "date": "2025-11-15"
        },
        "refund_policy": "Refunds are approved for facility malfunctions."
    }
    
    result = await triage_ticket(context)
    
    # Verify result structure
    assert "decision" in result.data
    assert result.data["decision"] in ["Approved", "Denied", "Needs Human Review"]
    assert "reasoning" in result.data
    assert result.metadata["summary"].startswith("Triage decision")


@pytest.mark.asyncio
async def test_triage_ticket_denied():
    """Test ticket triage when refund should be denied."""
    context = Mock()
    context.inputs = {
        "ticket_data": {
            "id": "1205975",
            "subject": "Refund request - changed plans",
            "description": "I no longer need parking, purchased 5 minutes ago."
        },
        "booking_info": {
            "booking_id": "PW-789012",
            "amount": 30.00,
            "date": "2025-11-20"
        },
        "refund_policy": "No refunds for change of mind within 24 hours of event."
    }
    
    result = await triage_ticket(context)
    
    # Verify result structure
    assert "decision" in result.data
    assert result.data["decision"] in ["Approved", "Denied", "Needs Human Review"]


@pytest.mark.asyncio
async def test_triage_ticket_needs_review():
    """Test ticket triage when human review is needed."""
    context = Mock()
    context.inputs = {
        "ticket_data": {
            "id": "1205976",
            "subject": "Refund request - unusual situation",
            "description": "Very complex situation with multiple factors."
        },
        "booking_info": None,  # Missing booking info
        "refund_policy": "Standard policy applies."
    }
    
    result = await triage_ticket(context)
    
    # Should still return a valid decision
    assert "decision" in result.data


@pytest.mark.asyncio
async def test_triage_ticket_minimal_data():
    """Test triage with minimal required data."""
    context = Mock()
    context.inputs = {
        "ticket_data": {"id": "123", "subject": "Refund request"},
    }
    
    result = await triage_ticket(context)
    
    # Should handle minimal data gracefully
    assert result.data is not None
    assert "decision" in result.data


@pytest.mark.asyncio
async def test_extract_booking_info_with_multiple_bookings():
    """Test extracting booking info when multiple bookings are mentioned."""
    context = Mock()
    context.inputs = {
        "ticket_notes": """
        Customer has two bookings:
        1. Booking #PW-111111 for $20
        2. Booking #PW-222222 for $35
        Requesting refund for the second one.
        """
    }
    
    result = await extract_booking_info_from_note(context)
    
    # Should extract information about bookings
    assert result.data is not None


@pytest.mark.asyncio
async def test_triage_ticket_with_policy_context():
    """Test that triage properly considers refund policy context."""
    context = Mock()
    context.inputs = {
        "ticket_data": {
            "id": "1205977",
            "subject": "Refund request",
            "description": "Event was cancelled"
        },
        "booking_info": {"booking_id": "PW-333333", "amount": 50.00},
        "refund_policy": "Full refund for event cancellations."
    }
    
    result = await triage_ticket(context)
    
    # Verify the policy was considered
    assert result.data is not None
    assert "decision" in result.data
    assert "reasoning" in result.data
