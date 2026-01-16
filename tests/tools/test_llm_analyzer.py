"""Tests for LLMAnalyzer component."""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch
from app_tools.tools.booking_analysis.llm_analyzer import LLMAnalyzer


@pytest.fixture
def mock_gemini_api_key(monkeypatch):
    """Mock GEMINI_API_KEY environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")


@pytest.fixture
def llm_analyzer(mock_gemini_api_key):
    """Create LLMAnalyzer instance."""
    return LLMAnalyzer()


def test_llm_analyzer_initialization(mock_gemini_api_key):
    """Test LLMAnalyzer initializes correctly."""
    analyzer = LLMAnalyzer()
    
    assert analyzer.model_name == "gemini-2.5-flash"
    assert analyzer.client is not None


@pytest.fixture
def sample_ticket_data():
    """Sample ticket data for testing."""
    return {
        "ticket_id": "1206331",
        "subject": "Refund Request - Event Cancelled",
        "description": "Customer is requesting a refund because their event was cancelled.",
        "status": "Open"
    }


@pytest.fixture
def sample_booking_info():
    """Sample booking information for testing."""
    return {
        "booking_id": "PW-509266779",
        "amount": 45.00,
        "event_date": "2025-11-15",
        "reservation_date": "2025-11-01",
        "cancellation_date": "2025-11-10",
        "booking_type": "confirmed",
        "location": "Downtown Parking Garage",
        "customer_email": "customer@example.com"
    }


@pytest.fixture
def sample_policy_text():
    """Sample policy text for testing."""
    return """
# Refund Policy

## Pre-Arrival Cancellations (7+ days)
- Full refund for cancellations 7+ days before event

## Short Notice Cancellations (3-7 days)
- Confirmed bookings: Full refund
- On-demand bookings: No refund

## Last Minute Cancellations (<3 days)
- On-demand bookings: No refund
- Confirmed bookings: Case by case

## Post-Event Cancellations
- No refund after event start
"""


@pytest.fixture
def sample_rule_result():
    """Sample rule-based result for testing."""
    return {
        "decision": "Approved",
        "reasoning": "Cancellation made 5 days before event",
        "policy_rule": "Confirmed Booking (3-7 days)",
        "confidence": "medium"
    }


def create_mock_gemini_response(data):
    """Create a mock Gemini API response."""
    mock_response = Mock()
    mock_response.text = json.dumps(data)
    return mock_response


@pytest.fixture
def mock_approved_response():
    """Mock LLM response for approved decision."""
    return {
        "decision": "Approved",
        "reasoning": "Customer cancelled 5 days before event, which falls within the 3-7 day window for confirmed bookings. Policy allows full refund.",
        "policy_applied": "Confirmed Booking (3-7 days) - Full Refund",
        "confidence": "high",
        "key_factors": [
            "Cancellation timing: 5 days before event",
            "Booking type: Confirmed",
            "Policy allows refund in this timeframe"
        ]
    }


@pytest.fixture
def mock_denied_response():
    """Mock LLM response for denied decision."""
    return {
        "decision": "Denied",
        "reasoning": "Customer cancelled after the event started. Post-event cancellations are not eligible for refunds per policy.",
        "policy_applied": "Post-Event Cancellation - No Refund",
        "confidence": "high",
        "key_factors": [
            "Cancellation after event start",
            "Post-event policy applies",
            "No exceptions mentioned"
        ]
    }


@pytest.fixture
def mock_needs_review_response():
    """Mock LLM response for needs human review decision."""
    return {
        "decision": "Needs Human Review",
        "reasoning": "Case involves partial refund consideration and special circumstances that require human judgment.",
        "policy_applied": "Complex Case - Requires Review",
        "confidence": "medium",
        "key_factors": [
            "Partial refund consideration",
            "Special circumstances mentioned",
            "Policy ambiguous for this scenario"
        ]
    }


def test_llm_analyzer_custom_model(mock_gemini_api_key):
    """Test LLMAnalyzer with custom model name."""
    analyzer = LLMAnalyzer(model_name="gemini-2.0-flash-exp")
    assert analyzer.model_name == "gemini-2.0-flash-exp"


def test_llm_analyzer_no_api_key(monkeypatch):
    """Test LLMAnalyzer raises error without API key."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError) as exc_info:
        LLMAnalyzer()
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_llm_analyzer_env_model(mock_gemini_api_key, monkeypatch):
    """Test LLMAnalyzer uses GEMINI_MODEL env var."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-pro")
    analyzer = LLMAnalyzer()
    assert analyzer.model_name == "gemini-1.5-pro"


@pytest.mark.asyncio
async def test_analyze_case_approved(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, mock_approved_response
):
    """Test LLM analysis returns approved decision."""
    mock_response = create_mock_gemini_response(mock_approved_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "high"
        assert "Confirmed Booking" in result["policy_applied"]
        assert len(result["key_factors"]) == 3
        assert "5 days before event" in result["reasoning"]


@pytest.mark.asyncio
async def test_analyze_case_denied(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, mock_denied_response
):
    """Test LLM analysis returns denied decision."""
    mock_response = create_mock_gemini_response(mock_denied_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Denied"
        assert result["confidence"] == "high"
        assert "Post-Event" in result["policy_applied"]
        assert "started" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_analyze_case_needs_review(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, mock_needs_review_response
):
    """Test LLM analysis returns needs human review decision."""
    mock_response = create_mock_gemini_response(mock_needs_review_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "medium"
        assert "Complex Case" in result["policy_applied"]


@pytest.mark.asyncio
async def test_analyze_case_with_rule_result(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, sample_rule_result, mock_approved_response
):
    """Test LLM analysis with rule-based result context."""
    mock_response = create_mock_gemini_response(mock_approved_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=sample_rule_result
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "high"


# Test timeout handling
@pytest.mark.asyncio
async def test_analyze_case_timeout_with_high_confidence_rule(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test timeout falls back to high confidence rule result."""
    rule_result = {
        "decision": "Approved",
        "reasoning": "Rule-based approval",
        "policy_rule": "Pre-Arrival (7+ days)",
        "confidence": "high"
    }
    
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=asyncio.TimeoutError("Timeout after 10 seconds")):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=rule_result
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "medium"
        assert rule_result["policy_rule"] in result["policy_applied"]
        assert "LLM analysis failed" in result["reasoning"]
        assert "LLM analysis unavailable" in result["key_factors"]


@pytest.mark.asyncio
async def test_analyze_case_timeout_with_medium_confidence_rule(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test timeout falls back to medium confidence rule result."""
    rule_result = {
        "decision": "Denied",
        "reasoning": "On-demand booking with short notice",
        "policy_rule": "On-Demand (<3 days)",
        "confidence": "medium"
    }
    
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=asyncio.TimeoutError()):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=rule_result
        )
        
        assert result["decision"] == "Denied"
        assert result["confidence"] == "medium"
        assert "using rule-based decision" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_analyze_case_timeout_no_rule_result(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test timeout without rule result escalates to human review."""
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=asyncio.TimeoutError()):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "low"
        assert "Escalation - Technical Failure" in result["policy_applied"]
        assert "Unable to complete automated analysis" in result["reasoning"]
        assert "LLM analysis failed" in result["key_factors"]


@pytest.mark.asyncio
async def test_analyze_case_timeout_low_confidence_rule(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test timeout with low confidence rule escalates to human review."""
    rule_result = {
        "decision": "Uncertain",
        "reasoning": "Ambiguous case",
        "policy_rule": "Unknown",
        "confidence": "low"
    }
    
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=asyncio.TimeoutError()):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=rule_result
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "low"
        assert "No high-confidence rule-based decision available" in result["key_factors"]


# Test JSON parsing errors
@pytest.mark.asyncio
async def test_analyze_case_json_parse_error(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test handling of invalid JSON response."""
    mock_response = Mock()
    mock_response.text = "Invalid JSON {not valid}"
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "low"
        assert "Technical Failure" in result["policy_applied"]


@pytest.mark.asyncio
async def test_analyze_case_json_parse_error_with_rule_fallback(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, sample_rule_result
):
    """Test JSON parse error falls back to rule result."""
    mock_response = Mock()
    mock_response.text = "{invalid json"
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=sample_rule_result
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "medium"
        assert sample_rule_result["policy_rule"] in result["policy_applied"]


# Test invalid decision validation
@pytest.mark.asyncio
async def test_analyze_case_invalid_decision_value(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test handling of invalid decision value from LLM."""
    invalid_response = {
        "decision": "Maybe",
        "reasoning": "Not sure",
        "policy_applied": "Unknown",
        "confidence": "medium",
        "key_factors": []
    }
    
    mock_response = create_mock_gemini_response(invalid_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "low"
        assert "Technical Failure" in result["policy_applied"]


@pytest.mark.asyncio
async def test_analyze_case_invalid_confidence_value(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test handling of invalid confidence value from LLM."""
    invalid_response = {
        "decision": "Approved",
        "reasoning": "Valid reasoning",
        "policy_applied": "Some Policy",
        "confidence": "very_high",
        "key_factors": ["factor1"]
    }
    
    mock_response = create_mock_gemini_response(invalid_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "medium"


# Test API errors
@pytest.mark.asyncio
async def test_analyze_case_api_error(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test handling of API errors."""
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=Exception("API Error: Rate limit exceeded")):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text
        )
        
        assert result["decision"] == "Needs Human Review"
        assert result["confidence"] == "low"
        assert "Technical Failure" in result["policy_applied"]


@pytest.mark.asyncio
async def test_analyze_case_api_error_with_rule_fallback(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, sample_rule_result
):
    """Test API error falls back to rule result."""
    with patch.object(llm_analyzer.client.models, 'generate_content',
                     side_effect=Exception("Network error")):
        result = await llm_analyzer.analyze_case(
            sample_ticket_data, sample_booking_info, sample_policy_text,
            rule_result=sample_rule_result
        )
        
        assert result["decision"] == "Approved"
        assert result["confidence"] == "medium"


# Test prompt creation and formatting helpers
def test_create_analysis_prompt_basic(
    llm_analyzer, sample_ticket_data, sample_booking_info, sample_policy_text
):
    """Test basic prompt creation."""
    prompt = llm_analyzer._create_analysis_prompt(
        sample_ticket_data, sample_booking_info, sample_policy_text, None
    )
    
    assert "refund policy expert" in prompt.lower()
    assert sample_policy_text in prompt
    assert sample_ticket_data["ticket_id"] in prompt
    assert sample_booking_info["booking_id"] in prompt
    assert "Approved" in prompt
    assert "Denied" in prompt
    assert "Needs Human Review" in prompt


def test_create_analysis_prompt_with_rule_result(
    llm_analyzer, sample_ticket_data, sample_booking_info,
    sample_policy_text, sample_rule_result
):
    """Test prompt creation with rule result context."""
    prompt = llm_analyzer._create_analysis_prompt(
        sample_ticket_data, sample_booking_info, sample_policy_text, sample_rule_result
    )
    
    assert "Rule-Based Analysis" in prompt
    assert sample_rule_result["decision"] in prompt
    assert sample_rule_result["reasoning"] in prompt
    assert "uncertain about this case" in prompt.lower()


def test_format_booking_info_complete(llm_analyzer, sample_booking_info):
    """Test formatting complete booking information."""
    formatted = llm_analyzer._format_booking_info(sample_booking_info)
    
    assert "PW-509266779" in formatted
    assert "$45.00" in formatted
    assert "2025-11-15" in formatted
    assert "Downtown Parking Garage" in formatted
    assert "confirmed" in formatted.lower()


def test_format_booking_info_empty(llm_analyzer):
    """Test formatting empty booking information."""
    formatted = llm_analyzer._format_booking_info({})
    assert "No booking information" in formatted or "Minimal booking information" in formatted


def test_format_ticket_info_complete(llm_analyzer, sample_ticket_data):
    """Test formatting complete ticket information."""
    formatted = llm_analyzer._format_ticket_info(sample_ticket_data)
    
    assert sample_ticket_data["ticket_id"] in formatted
    assert sample_ticket_data["subject"] in formatted
    assert sample_ticket_data["description"] in formatted
    assert sample_ticket_data["status"] in formatted


def test_format_ticket_info_long_description(llm_analyzer):
    """Test formatting ticket with very long description."""
    long_ticket = {
        "ticket_id": "123",
        "subject": "Test",
        "description": "A" * 2000
    }
    
    formatted = llm_analyzer._format_ticket_info(long_ticket)
    assert "truncated" in formatted.lower()
    assert len(formatted) < 2000


def test_create_fallback_decision_with_high_confidence_rule(llm_analyzer, sample_rule_result):
    """Test fallback uses high confidence rule result."""
    fallback = llm_analyzer._create_fallback_decision("Test error", sample_rule_result)
    
    assert fallback["decision"] == sample_rule_result["decision"]
    assert fallback["confidence"] == "medium"
    assert "LLM analysis failed" in fallback["reasoning"]
    assert sample_rule_result["policy_rule"] in fallback["policy_applied"]


def test_create_fallback_decision_no_rule_result(llm_analyzer):
    """Test fallback without rule result escalates to human review."""
    fallback = llm_analyzer._create_fallback_decision("API failure", None)
    
    assert fallback["decision"] == "Needs Human Review"
    assert fallback["confidence"] == "low"
    assert "Technical Failure" in fallback["policy_applied"]
    assert "Unable to complete automated analysis" in fallback["reasoning"]


# Test edge cases
@pytest.mark.asyncio
async def test_analyze_case_empty_ticket_data(
    llm_analyzer, sample_booking_info, sample_policy_text, mock_approved_response
):
    """Test analysis with empty ticket data."""
    mock_response = create_mock_gemini_response(mock_approved_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case({}, sample_booking_info, sample_policy_text)
        assert result["decision"] == "Approved"


@pytest.mark.asyncio
async def test_analyze_case_empty_booking_info(
    llm_analyzer, sample_ticket_data, sample_policy_text, mock_needs_review_response
):
    """Test analysis with empty booking info."""
    mock_response = create_mock_gemini_response(mock_needs_review_response)
    
    with patch.object(llm_analyzer.client.models, 'generate_content', return_value=mock_response):
        result = await llm_analyzer.analyze_case(sample_ticket_data, {}, sample_policy_text)
        assert result["decision"] == "Needs Human Review"
