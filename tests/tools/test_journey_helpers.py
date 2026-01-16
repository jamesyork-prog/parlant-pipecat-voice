
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app_tools.tools.ticket_processing.helpers import extract_booking_info_from_note, triage_ticket


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
            "event_date": "2025-11-15"
        },
        "ticket_notes": ""
    }
    
    # Mock DecisionMaker to return an Approved decision
    mock_decision = {
        "decision": "Approved",
        "reasoning": "Facility malfunction - gate was broken",
        "policy_applied": "Facility Issues Policy",
        "confidence": "high",
        "cancellation_reason": "Amenity missing",
        "booking_info_found": True,
        "method_used": "rules",
        "processing_time_ms": 150
    }
    
    with patch('app_tools.tools.ticket_processing.helpers.DecisionMaker') as MockDecisionMaker:
        mock_instance = MockDecisionMaker.return_value
        mock_instance.make_decision = AsyncMock(return_value=mock_decision)
        
        result = await triage_ticket(context)
        
        # Verify result structure
        assert "decision" in result.data
        assert result.data["decision"] == "Approved"
        assert "reasoning" in result.data
        assert "cancellation_reason" in result.data
        assert result.data["cancellation_reason"] == "Amenity missing"
        assert "method_used" in result.data
        assert "processing_time_ms" in result.data
        assert result.metadata["summary"].startswith("Decision:")


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
            "event_date": "2025-11-20"
        },
        "ticket_notes": ""
    }
    
    # Mock DecisionMaker to return a Denied decision
    mock_decision = {
        "decision": "Denied",
        "reasoning": "Cancellation too close to event date - less than 3 days before event",
        "policy_applied": "Pre-arrival Cancellation Policy",
        "confidence": "high",
        "cancellation_reason": None,
        "booking_info_found": True,
        "method_used": "rules",
        "processing_time_ms": 120
    }
    
    with patch('app_tools.tools.ticket_processing.helpers.DecisionMaker') as MockDecisionMaker:
        mock_instance = MockDecisionMaker.return_value
        mock_instance.make_decision = AsyncMock(return_value=mock_decision)
        
        result = await triage_ticket(context)
        
        # Verify result structure
        assert "decision" in result.data
        assert result.data["decision"] == "Denied"
        assert result.data["cancellation_reason"] is None  # No reason for denied
        assert "method_used" in result.data


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
        "ticket_notes": "Very complex situation with multiple factors."
    }
    
    # Mock DecisionMaker to return Needs Human Review
    mock_decision = {
        "decision": "Needs Human Review",
        "reasoning": "Unable to extract complete booking information from ticket",
        "policy_applied": "Data Validation - Incomplete Information",
        "confidence": "low",
        "cancellation_reason": None,
        "booking_info_found": False,
        "method_used": "extraction_failed",
        "processing_time_ms": 200
    }
    
    with patch('app_tools.tools.ticket_processing.helpers.DecisionMaker') as MockDecisionMaker:
        mock_instance = MockDecisionMaker.return_value
        mock_instance.make_decision = AsyncMock(return_value=mock_decision)
        
        result = await triage_ticket(context)
        
        # Should still return a valid decision
        assert "decision" in result.data
        assert result.data["decision"] == "Needs Human Review"
        assert result.data["booking_info_found"] is False


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
        "booking_info": {
            "booking_id": "PW-333333",
            "amount": 50.00,
            "event_date": "2025-12-01"
        },
        "ticket_notes": "Event was cancelled"
    }
    
    # Mock DecisionMaker to return Approved with LLM analysis
    mock_decision = {
        "decision": "Approved",
        "reasoning": "Event cancellation - full refund per policy",
        "policy_applied": "Event Cancellation Policy",
        "confidence": "high",
        "cancellation_reason": "PW cancellation",
        "booking_info_found": True,
        "method_used": "hybrid",
        "processing_time_ms": 350
    }
    
    with patch('app_tools.tools.ticket_processing.helpers.DecisionMaker') as MockDecisionMaker:
        mock_instance = MockDecisionMaker.return_value
        mock_instance.make_decision = AsyncMock(return_value=mock_decision)
        
        result = await triage_ticket(context)
        
        # Verify the policy was considered
        assert result.data is not None
        assert "decision" in result.data
        assert "reasoning" in result.data
        assert result.data["method_used"] == "hybrid"
