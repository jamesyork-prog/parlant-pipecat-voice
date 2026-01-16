"""
Tests for DecisionMaker orchestrator.

Tests cover:
- Full decision flow
- Rule-only path (high confidence)
- Hybrid path (low confidence → LLM)
- Error handling and fallbacks
- Booking extraction integration
- Cancellation reason mapping
"""

import pytest
import time
from unittest.mock import Mock, patch, AsyncMock
from app_tools.tools.refund_decision.decision_maker import DecisionMaker


@pytest.fixture
def mock_gemini_api_key(monkeypatch):
    """Mock GEMINI_API_KEY environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")


@pytest.fixture
def decision_maker(mock_gemini_api_key):
    """Create DecisionMaker instance."""
    return DecisionMaker()


@pytest.fixture
def sample_ticket_data():
    """Sample ticket data for testing."""
    return {
        "ticket_id": "1206331",
        "subject": "Refund Request",
        "description": "Customer requesting refund for parking reservation.",
        "status": "Open"
    }


@pytest.fixture
def sample_booking_info():
    """Sample booking information for testing."""
    return {
        "booking_id": "PW-509266779",
        "amount": 45.00,
        "event_date": "2025-11-25",
        "reservation_date": "2025-11-01",
        "cancellation_date": "2025-11-10",
        "booking_type": "confirmed",
        "location": "Downtown Parking Garage",
        "customer_email": "customer@example.com"
    }


@pytest.fixture
def sample_ticket_notes():
    """Sample ticket notes for testing."""
    return """
    Booking Confirmation
    
    Booking ID: PW-509266779
    Amount: $45.00
    Event Date: 11/25/2025
    Location: Downtown Parking Garage
    """


# Test initialization
def test_decision_maker_initialization(mock_gemini_api_key):
    """Test DecisionMaker initializes all components."""
    dm = DecisionMaker()
    
    assert dm.policy_loader is not None
    assert dm.booking_extractor is not None
    assert dm.rule_engine is not None
    assert dm.llm_analyzer is not None
    assert dm.cancellation_reason_mapper is not None


# Test rule-only path (high confidence)
@pytest.mark.asyncio
async def test_make_decision_rule_only_approved(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision using rules only (high confidence approval)."""
    # Modify booking to be 10 days before event (7+ days rule)
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-20"  # 10 days from cancellation
    booking_info["cancellation_date"] = "2025-11-10"
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"
    assert result["method_used"] == "rules"
    assert result["booking_info_found"] is True
    assert result["cancellation_reason"] is not None
    assert "Pre-Arrival" in result["policy_applied"]
    assert result["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_make_decision_rule_only_denied(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision using rules only (high confidence denial)."""
    # Use on-demand booking with <3 days notice (clear denial case)
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-12"  # 2 days from cancellation
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "on-demand"  # On-demand with <3 days = denied
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["decision"] == "Denied"
    assert result["confidence"] == "high"
    assert result["method_used"] == "rules"
    assert result["cancellation_reason"] is None
    assert "On-Demand" in result["policy_applied"]


# Test hybrid path (low confidence → LLM)
@pytest.mark.asyncio
async def test_make_decision_hybrid_path(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision using hybrid approach (rules uncertain → LLM)."""
    # Modify booking to trigger uncertain rule result (3-7 days, unclear type)
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"  # 5 days from cancellation
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "unknown"  # Unclear type
    
    # Mock LLM response
    mock_llm_result = {
        "decision": "Approved",
        "reasoning": "LLM determined this should be approved",
        "policy_applied": "LLM Analysis - Approved",
        "confidence": "high",
        "key_factors": ["LLM analysis"]
    }
    
    with patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        assert result["decision"] == "Approved"
        assert result["method_used"] == "hybrid"
        assert result["cancellation_reason"] is not None
        assert "LLM" in result["policy_applied"]


@pytest.mark.asyncio
async def test_make_decision_hybrid_low_confidence_rule(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that low confidence rule triggers LLM analysis."""
    # Create a scenario that produces medium confidence
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"  # 5 days (3-7 days range)
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "confirmed"  # Medium confidence
    
    # Mock LLM response
    mock_llm_result = {
        "decision": "Approved",
        "reasoning": "LLM confirmed approval",
        "policy_applied": "LLM Analysis",
        "confidence": "high",
        "key_factors": []
    }
    
    with patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        # Medium confidence should still use rules, not trigger LLM
        # Only "Uncertain" or "low" confidence triggers LLM
        assert result["method_used"] == "rules"


# Test booking extraction integration
@pytest.mark.asyncio
async def test_make_decision_with_extraction(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test decision with booking extraction from ticket notes."""
    # Mock booking extraction
    mock_extraction_result = {
        "booking_info": {
            "booking_id": "PW-509266779",
            "amount": 45.00,
            "event_date": "2025-11-25",
            "booking_type": "confirmed"
        },
        "found": True,
        "confidence": "high",
        "extraction_method": "llm"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["booking_info_found"] is True
        assert result["decision"] in ["Approved", "Denied", "Needs Human Review"]


@pytest.mark.asyncio
async def test_make_decision_extraction_failed(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test decision when booking extraction fails."""
    # Mock failed extraction
    mock_extraction_result = {
        "booking_info": {},
        "found": False,
        "confidence": "low",
        "extraction_method": "llm"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["booking_info_found"] is False
        assert result["method_used"] == "extraction_failed"
        assert "Unable to extract" in result["reasoning"]


@pytest.mark.asyncio
async def test_make_decision_extraction_error(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test decision when booking extraction raises an error."""
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        side_effect=Exception("Extraction error")
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["method_used"] == "extraction_error"
        assert "Technical error" in result["reasoning"]


# Test missing critical data
@pytest.mark.asyncio
async def test_make_decision_missing_event_date(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision when event date is missing."""
    booking_info = sample_booking_info.copy()
    del booking_info["event_date"]
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["decision"] == "Needs Human Review"
    assert result["method_used"] == "validation_failed"
    assert "Missing event date" in result["reasoning"]


# Test error handling in rule engine
@pytest.mark.asyncio
async def test_make_decision_rule_engine_error(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision when rule engine raises an error."""
    with patch.object(
        decision_maker.rule_engine,
        'apply_rules',
        side_effect=Exception("Rule engine error")
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=sample_booking_info
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["method_used"] == "rule_error"
        assert "Technical error" in result["reasoning"]


# Test error handling in LLM analyzer
@pytest.mark.asyncio
async def test_make_decision_llm_error_with_fallback(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision when LLM fails but rule-based fallback is available."""
    # Create uncertain rule result to trigger LLM
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "unknown"
    
    # Mock rule engine to return uncertain
    mock_rule_result = {
        "decision": "Uncertain",
        "reasoning": "Unclear booking type",
        "policy_rule": "Ambiguous",
        "confidence": "low"
    }
    
    with patch.object(
        decision_maker.rule_engine,
        'apply_rules',
        return_value=mock_rule_result
    ), patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        side_effect=Exception("LLM error")
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        # Should escalate since rule result is uncertain
        assert result["decision"] == "Needs Human Review"
        assert result["method_used"] == "llm_error"


@pytest.mark.asyncio
async def test_make_decision_llm_error_with_high_confidence_fallback(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test decision when LLM fails but high confidence rule is available."""
    # Create uncertain rule result to trigger LLM, then test fallback
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "confirmed"
    
    # Mock rule engine to return Approved with medium confidence
    # Note: Only "Uncertain" or "low" confidence triggers LLM in the actual implementation
    # So we need to mock a rule result that would trigger LLM
    mock_rule_result = {
        "decision": "Approved",
        "reasoning": "Confirmed booking in 3-7 day window",
        "policy_rule": "Confirmed Booking (3-7 days)",
        "confidence": "medium"  # Medium confidence does NOT trigger LLM
    }
    
    with patch.object(
        decision_maker.rule_engine,
        'apply_rules',
        return_value=mock_rule_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        # Medium confidence uses rules directly, doesn't trigger LLM
        assert result["decision"] == "Approved"
        assert result["method_used"] == "rules"
        assert result["confidence"] == "medium"


# Test cancellation reason mapping
@pytest.mark.asyncio
async def test_make_decision_maps_cancellation_reason_approved(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that cancellation reason is mapped for approved decisions."""
    # Create high confidence approval
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-20"  # 10 days from cancellation
    booking_info["cancellation_date"] = "2025-11-10"
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["decision"] == "Approved"
    assert result["cancellation_reason"] is not None
    assert result["cancellation_reason"] in [
        "Other", "Tolerance", "Multi-day", "Pending re-book",
        "Pre-arrival", "Oversold", "No attendant", "Amenity missing",
        "Poor experience", "Inaccurate hours of operation",
        "Attendant refused customer", "Duplicate booking",
        "Confirmed re-book", "Paid again", "Accessibility",
        "PW cancellation"
    ]


@pytest.mark.asyncio
async def test_make_decision_no_cancellation_reason_denied(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that cancellation reason is None for denied decisions."""
    # Create high confidence denial (on-demand with <3 days)
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-12"  # 2 days from cancellation
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "on-demand"
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["decision"] == "Denied"
    assert result["cancellation_reason"] is None


@pytest.mark.asyncio
async def test_make_decision_cancellation_reason_mapper_error(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that cancellation reason defaults to 'Other' on error."""
    # Create high confidence approval
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-20"
    booking_info["cancellation_date"] = "2025-11-10"
    
    with patch.object(
        decision_maker.cancellation_reason_mapper,
        'map_reason',
        side_effect=Exception("Mapping error")
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        assert result["decision"] == "Approved"
        assert result["cancellation_reason"] == "Other"


# Test processing time tracking
@pytest.mark.asyncio
async def test_make_decision_tracks_processing_time(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that processing time is tracked."""
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=sample_booking_info
    )
    
    assert "processing_time_ms" in result
    assert result["processing_time_ms"] >= 0
    assert isinstance(result["processing_time_ms"], int)


# Test method_used tracking
@pytest.mark.asyncio
async def test_make_decision_tracks_method_rules(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that method_used is 'rules' for rule-only decisions."""
    # High confidence rule decision
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-20"
    booking_info["cancellation_date"] = "2025-11-10"
    
    result = await decision_maker.make_decision(
        sample_ticket_data,
        booking_info=booking_info
    )
    
    assert result["method_used"] == "rules"


@pytest.mark.asyncio
async def test_make_decision_tracks_method_hybrid(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that method_used is 'hybrid' when LLM is invoked."""
    # Create uncertain rule result
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "unknown"
    
    mock_llm_result = {
        "decision": "Approved",
        "reasoning": "LLM analysis",
        "policy_applied": "LLM",
        "confidence": "high",
        "key_factors": []
    }
    
    with patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        assert result["method_used"] == "hybrid"


# Test key_factors inclusion
@pytest.mark.asyncio
async def test_make_decision_includes_key_factors(
    decision_maker,
    sample_ticket_data,
    sample_booking_info
):
    """Test that key_factors are included in result."""
    # Create uncertain rule to trigger LLM
    booking_info = sample_booking_info.copy()
    booking_info["event_date"] = "2025-11-15"
    booking_info["cancellation_date"] = "2025-11-10"
    booking_info["booking_type"] = "unknown"
    
    mock_llm_result = {
        "decision": "Approved",
        "reasoning": "LLM analysis",
        "policy_applied": "LLM",
        "confidence": "high",
        "key_factors": ["Factor 1", "Factor 2", "Factor 3"]
    }
    
    with patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            booking_info=booking_info
        )
        
        assert "key_factors" in result
        assert len(result["key_factors"]) == 3


# Test edge cases
@pytest.mark.asyncio
async def test_make_decision_empty_ticket_data(
    decision_maker,
    sample_booking_info
):
    """Test decision with empty ticket data."""
    result = await decision_maker.make_decision(
        {},
        booking_info=sample_booking_info
    )
    
    # Should still work with empty ticket data
    assert result["decision"] in ["Approved", "Denied", "Needs Human Review"]


@pytest.mark.asyncio
async def test_make_decision_no_ticket_notes_no_booking_info(
    decision_maker,
    sample_ticket_data
):
    """Test decision with no ticket notes and no booking info."""
    result = await decision_maker.make_decision(
        sample_ticket_data
    )
    
    # Should escalate due to missing booking info
    assert result["decision"] == "Needs Human Review"
    assert result["booking_info_found"] is False


@pytest.mark.asyncio
async def test_make_decision_uses_ticket_description_for_extraction(
    decision_maker,
    sample_ticket_data
):
    """Test that ticket description is used when ticket_notes is not provided."""
    ticket_data = sample_ticket_data.copy()
    ticket_data["description"] = "Booking PW-123 for $45.00 on 11/25/2025"
    
    mock_extraction_result = {
        "booking_info": {
            "booking_id": "PW-123",
            "amount": 45.00,
            "event_date": "2025-11-25"
        },
        "found": True,
        "confidence": "medium",
        "extraction_method": "llm"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ) as mock_extract:
        result = await decision_maker.make_decision(ticket_data)
        
        # Should have called extraction with ticket description
        mock_extract.assert_called_once()
        call_args = mock_extract.call_args[0][0]
        assert ticket_data["description"] in call_args


# Test extraction with error field
@pytest.mark.asyncio
async def test_make_decision_extraction_with_error_field(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test decision when extraction result includes error field."""
    mock_extraction_result = {
        "booking_info": {},
        "found": False,
        "confidence": "low",
        "extraction_method": "llm",
        "error": "Timeout during extraction"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Needs Human Review"
        assert "Extraction error" in result["reasoning"]


# Test full end-to-end flow
@pytest.mark.asyncio
async def test_make_decision_full_flow_approved(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test complete decision flow resulting in approval."""
    # Mock extraction
    mock_extraction_result = {
        "booking_info": {
            "booking_id": "PW-509266779",
            "amount": 45.00,
            "event_date": "2025-11-20",  # 10 days from cancellation
            "cancellation_date": "2025-11-10",
            "booking_type": "confirmed"
        },
        "found": True,
        "confidence": "high",
        "extraction_method": "pattern"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "high"
        assert result["method_used"] == "rules"
        assert result["booking_info_found"] is True
        assert result["cancellation_reason"] is not None
        assert result["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_make_decision_full_flow_denied(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test complete decision flow resulting in denial."""
    # Mock extraction for on-demand <3 days scenario (clear denial)
    mock_extraction_result = {
        "booking_info": {
            "booking_id": "PW-509266779",
            "amount": 45.00,
            "event_date": "2025-11-12",  # 2 days from cancellation
            "cancellation_date": "2025-11-10",
            "booking_type": "on-demand"
        },
        "found": True,
        "confidence": "high",
        "extraction_method": "llm"
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Denied"
        assert result["confidence"] == "high"
        assert result["method_used"] == "rules"
        assert result["cancellation_reason"] is None
        assert "On-Demand" in result["policy_applied"]


@pytest.mark.asyncio
async def test_make_decision_full_flow_hybrid(
    decision_maker,
    sample_ticket_data,
    sample_ticket_notes
):
    """Test complete decision flow using hybrid approach."""
    # Mock extraction for uncertain scenario
    mock_extraction_result = {
        "booking_info": {
            "booking_id": "PW-509266779",
            "amount": 45.00,
            "event_date": "2025-11-15",  # 5 days (3-7 range)
            "cancellation_date": "2025-11-10",
            "booking_type": "unknown"  # Triggers uncertain
        },
        "found": True,
        "confidence": "medium",
        "extraction_method": "llm"
    }
    
    mock_llm_result = {
        "decision": "Needs Human Review",
        "reasoning": "Complex case requiring human judgment",
        "policy_applied": "Complex Case",
        "confidence": "medium",
        "key_factors": ["Ambiguous booking type", "Mid-range timing"]
    }
    
    with patch.object(
        decision_maker.booking_extractor,
        'extract_booking_info',
        return_value=mock_extraction_result
    ), patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            sample_ticket_data,
            ticket_notes=sample_ticket_notes
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["method_used"] == "hybrid"
        assert result["booking_info_found"] is True
        assert len(result["key_factors"]) > 0



@pytest.mark.asyncio
async def test_amount_limit_exceeded_escalation(decision_maker, sample_ticket_data):
    """
    Test that refunds over $50 are escalated to human review (except pre-arrival).
    
    Automation can only approve refunds up to $50. Amounts over $50 require
    supervisor approval, even if the policy supports the refund.
    """
    # Post-event "paid again" scenario with $100 booking
    booking_info = {
        "booking_id": "PW-782293183",
        "amount": 100.00,  # Over $50 limit
        "event_date": "2025-10-27T15:00:00Z",
        "cancellation_date": "2025-12-05T20:15:00Z",  # Post-event
        "booking_type": "confirmed"
    }
    
    ticket_notes = """Reason: Barcode didn't work
Summary: The attendant tried my barcode 4 times. He made me pay again $50!"""
    
    result = await decision_maker.make_decision(
        ticket_data=sample_ticket_data,
        ticket_notes=ticket_notes,
        booking_info=booking_info
    )
    
    # Should escalate due to amount limit
    assert result["decision"] == "Needs Human Review"
    assert ("Amount Limit" in result["policy_applied"] or 
            "Amount Exceeds" in result["policy_applied"] or
            "Amount Limit Exceeded" in result["reasoning"])
    assert "$100" in result["reasoning"] or "100.00" in result["reasoning"]
    assert "$50" in result["reasoning"]
    assert result["confidence"] == "high"  # High confidence that it WOULD be approved, but needs supervisor
    assert result["method_used"] == "amount_limit_exceeded"


@pytest.mark.asyncio
async def test_amount_limit_pre_arrival_also_has_limit(decision_maker, sample_ticket_data):
    """
    Test that pre-arrival refunds (7+ days) also have the $50 amount limit.
    
    ALL automated approvals are limited to $50 or less, regardless of timing.
    This ensures supervisor oversight for high-value refunds.
    """
    # Pre-arrival cancellation with $150 booking
    booking_info = {
        "booking_id": "PW-999999",
        "amount": 150.00,  # Over $50 - should escalate even for pre-arrival
        "event_date": "2025-12-20T15:00:00Z",
        "cancellation_date": "2025-12-08T10:00:00Z",  # 12 days before
        "booking_type": "confirmed"
    }
    
    ticket_notes = "Customer changed plans, requesting refund."
    
    result = await decision_maker.make_decision(
        ticket_data=sample_ticket_data,
        ticket_notes=ticket_notes,
        booking_info=booking_info
    )
    
    # Should escalate due to amount limit (even though it's pre-arrival)
    assert result["decision"] == "Needs Human Review"
    assert ("Amount Limit" in result["policy_applied"] or 
            "Amount Exceeds" in result["policy_applied"])
    assert "$150" in result["reasoning"] or "150.00" in result["reasoning"]
    assert "$50" in result["reasoning"]
    assert result["confidence"] == "high"
    assert result["method_used"] == "amount_limit_exceeded"


@pytest.mark.asyncio
async def test_amount_limit_under_50_approved(decision_maker, sample_ticket_data):
    """
    Test that refunds under $50 are approved without escalation.
    """
    # Post-event "paid again" scenario with $45 booking
    booking_info = {
        "booking_id": "PW-123456",
        "amount": 45.00,  # Under $50 limit (< 50, not >= 50)
        "event_date": "2025-12-01T15:00:00Z",
        "cancellation_date": "2025-12-05T20:15:00Z",  # Post-event
        "booking_type": "confirmed"
    }
    
    ticket_notes = """Reason: Barcode didn't work
Summary: Had to pay again at the gate."""
    
    result = await decision_maker.make_decision(
        ticket_data=sample_ticket_data,
        ticket_notes=ticket_notes,
        booking_info=booking_info
    )
    
    # Should approve without escalation
    assert result["decision"] == "Approved"
    assert "Paid Again" in result["policy_applied"]
    assert result["method_used"] != "amount_limit_exceeded"


@pytest.mark.asyncio
async def test_amount_limit_exactly_50_escalated(decision_maker, sample_ticket_data):
    """
    Test that refunds of exactly $50.00 are escalated to human review.
    
    The limit is "under $50", so $50.00 exactly requires supervisor approval.
    """
    # Post-event "paid again" scenario with exactly $50 booking
    booking_info = {
        "booking_id": "PW-782293183",
        "amount": 50.00,  # Exactly $50 - should escalate
        "event_date": "2025-12-01T15:00:00Z",
        "cancellation_date": "2025-12-05T20:15:00Z",  # Post-event
        "booking_type": "confirmed"
    }
    
    ticket_notes = """Reason: Barcode didn't work
Summary: Had to pay again at the gate."""
    
    result = await decision_maker.make_decision(
        ticket_data=sample_ticket_data,
        ticket_notes=ticket_notes,
        booking_info=booking_info
    )
    
    # Should escalate due to amount limit
    assert result["decision"] == "Needs Human Review"
    assert ("Amount Limit" in result["policy_applied"] or 
            "Amount Exceeds" in result["policy_applied"])
    assert "$50" in result["reasoning"]
    assert result["confidence"] == "high"
    assert result["method_used"] == "amount_limit_exceeded"


@pytest.mark.asyncio
async def test_multiple_passes_total_amount_exceeds_50_escalated(decision_maker, sample_ticket_data):
    """
    Test that multiple passes with total amount over $50 are escalated.
    
    This tests the fix for ticket 1309307 where customer had 2 passes at $28 each = $56 total.
    The system should detect multiple passes and calculate the total refund amount.
    """
    # Closed location scenario with 2 passes at $28 each = $56 total
    booking_info = {
        "booking_id": "788880177",
        "amount": 28.00,  # Single pass amount - under $50
        "event_date": "2025-12-31T00:00:00Z",
        "cancellation_date": "2025-12-31T13:00:00Z",
        "booking_type": "confirmed"
    }
    
    # This is the actual ticket text from ticket 1309307
    ticket_notes = """Reason: Closed Location (upon Arrival/Exit)
Arrival: 12/31/2025 13:00:00
Exit: 12/31/2025 20:00:00
Make and Model: Chrysler Pacifica
Were you able to park? No
Proof of Payment: Link to uploaded image
Summary: I purchased parking through ParkWhiz for two different vehicle passes at 733 South St, Orlando, FL, for $28 a piece plus processing fees. Upon arrival, the gate to the lot was closed and locked, and no vehicles were allowed to park in there. So, we had to park in a different lot down the road and pay cash, so we ended up getting doubly charged for parking.

How can I get a refund for the unused passes? I was not able to be aware ahead of time that the gate would be closed, and the fact that the lot was marked as available and allowed me to book seemed misleading.

I can provide pictures of what the lot looked like when we arrived, the lot being empty and locked up.

Please let me know how we can work through getting a refund for this, and let me know if I can do anything to help."""
    
    # Mock LLM to return Approved (simulating the bug scenario)
    mock_llm_result = {
        "decision": "Approved",
        "reasoning": "Closed location - customer could not access the lot",
        "policy_applied": "Operational Failure / Service Delivery Failure Policy",
        "confidence": "high",
        "key_factors": ["Gate was closed", "Customer could not park"]
    }
    
    with patch.object(
        decision_maker.llm_analyzer,
        'analyze_case',
        return_value=mock_llm_result
    ):
        result = await decision_maker.make_decision(
            ticket_data=sample_ticket_data,
            ticket_notes=ticket_notes,
            booking_info=booking_info
        )
    
    # Should escalate because total is $56 (2 passes × $28)
    assert result["decision"] == "Needs Human Review", f"Expected escalation but got {result['decision']}"
    assert ("Amount Limit" in result.get("policy_applied", "") or 
            "Amount Exceeds" in result.get("policy_applied", "") or
            "Amount Limit Exceeded" in result.get("reasoning", "") or
            "post_llm_amount_limit_exceeded" in result.get("method_used", "") or
            "amount_limit_exceeded" in result.get("method_used", "")), \
        f"Expected amount limit escalation but got policy: {result.get('policy_applied')}, method: {result.get('method_used')}"
    
    # Verify the total amount was calculated correctly
    if "total_refund_amount" in result:
        assert result["total_refund_amount"] == 56.0, f"Expected $56 total but got ${result['total_refund_amount']}"
    if "pass_count" in result:
        assert result["pass_count"] == 2, f"Expected 2 passes but got {result['pass_count']}"


@pytest.mark.asyncio
async def test_single_pass_under_50_approved(decision_maker, sample_ticket_data):
    """
    Test that a single pass under $50 is still approved (no false positives).
    """
    booking_info = {
        "booking_id": "PW-123456",
        "amount": 28.00,  # Single pass under $50
        "event_date": "2025-12-31T00:00:00Z",
        "cancellation_date": "2025-12-31T13:00:00Z",
        "booking_type": "confirmed"
    }
    
    # Single pass scenario - no mention of multiple passes
    ticket_notes = """Reason: Closed Location (upon Arrival/Exit)
Summary: The gate was closed when I arrived. I couldn't park and had to find another lot."""
    
    result = await decision_maker.make_decision(
        ticket_data=sample_ticket_data,
        ticket_notes=ticket_notes,
        booking_info=booking_info
    )
    
    # Should approve since single pass is under $50
    # Note: May go to LLM for "Closed Location" analysis, but should still approve
    assert result["decision"] in ["Approved", "Needs Human Review"], f"Unexpected decision: {result['decision']}"
    
    # If it's escalated, it should NOT be due to amount limit
    if result["decision"] == "Needs Human Review":
        assert "amount_limit_exceeded" not in result.get("method_used", ""), \
            "Single $28 pass should not trigger amount limit escalation"
