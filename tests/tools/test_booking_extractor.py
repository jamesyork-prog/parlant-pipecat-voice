"""
Tests for BookingExtractor component (hybrid pattern + LLM extraction).

Tests cover:
- Extraction with complete booking info
- Extraction with partial booking info
- Extraction with no booking info
- Extraction with multiple bookings
- Mock Gemini API responses
- Pattern fallback behavior
- Error handling and timeouts
"""

import pytest
import json
from unittest.mock import Mock, patch, AsyncMock
from app_tools.tools.booking_analysis.extractor import BookingExtractor


@pytest.fixture
def mock_gemini_api_key(monkeypatch):
    """Mock GEMINI_API_KEY environment variable."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key-12345")


@pytest.fixture
def booking_extractor_no_pattern(mock_gemini_api_key):
    """Create BookingExtractor without pattern fallback for pure LLM testing."""
    return BookingExtractor(use_pattern_fallback=False)


@pytest.fixture
def booking_extractor_with_pattern(mock_gemini_api_key):
    """Create BookingExtractor with pattern fallback enabled."""
    return BookingExtractor(use_pattern_fallback=True)


# Test fixtures for ticket data
@pytest.fixture
def complete_ticket_text():
    """Complete ticket with all booking information."""
    return """
    Booking Confirmation
    
    Booking ID: PW-509266779
    Amount: $45.00
    Reservation Date: 11/01/2025
    Event Date: 11/15/2025
    Location: Downtown Parking Garage
    Customer Email: customer@example.com
    Booking Type: Confirmed
    """


@pytest.fixture
def partial_ticket_text():
    """Partial ticket with some missing fields."""
    return """
    Customer is requesting refund for booking 123456789.
    The parking was scheduled for November 15, 2025.
    """


@pytest.fixture
def no_booking_ticket_text():
    """Ticket with no booking information."""
    return """
    Customer is asking about parking options downtown.
    They want to know about rates and availability.
    No specific booking mentioned.
    """


@pytest.fixture
def multiple_bookings_ticket_text():
    """Ticket with multiple booking references."""
    return """
    Customer has two bookings:
    1. Booking PW-111111111 for $25.00 on 11/10/2025 at Airport Lot
    2. Booking PW-222222222 for $35.00 on 11/15/2025 at Downtown Garage
    
    They want to cancel the second booking (PW-222222222) due to event cancellation.
    """


# Mock Gemini API responses
def create_mock_gemini_response(data):
    """Create a mock Gemini API response."""
    mock_response = Mock()
    mock_response.text = json.dumps(data)
    return mock_response


@pytest.fixture
def mock_complete_extraction():
    """Mock complete booking extraction response."""
    return {
        "booking_id": "PW-509266779",
        "amount": 45.00,
        "reservation_date": "2025-11-01",
        "event_date": "2025-11-15",
        "location": "Downtown Parking Garage",
        "booking_type": "confirmed",
        "customer_email": "customer@example.com",
        "found": True,
        "multiple_bookings": False
    }


@pytest.fixture
def mock_partial_extraction():
    """Mock partial booking extraction response."""
    return {
        "booking_id": "123456789",
        "event_date": "2025-11-15",
        "found": True,
        "multiple_bookings": False
    }


@pytest.fixture
def mock_no_booking_extraction():
    """Mock no booking found response."""
    return {
        "found": False,
        "multiple_bookings": False
    }


@pytest.fixture
def mock_multiple_bookings_extraction():
    """Mock multiple bookings extraction response."""
    return {
        "booking_id": "PW-222222222",
        "amount": 35.00,
        "event_date": "2025-11-15",
        "location": "Downtown Garage",
        "found": True,
        "multiple_bookings": True
    }


# Test initialization
def test_booking_extractor_initialization(mock_gemini_api_key):
    """Test BookingExtractor initializes correctly."""
    extractor = BookingExtractor()
    
    assert extractor.model_name == "gemini-2.5-flash"
    assert extractor.use_pattern_fallback is True
    assert extractor.pattern_extractor is not None


def test_booking_extractor_custom_model(mock_gemini_api_key):
    """Test BookingExtractor with custom model name."""
    extractor = BookingExtractor(model_name="gemini-2.0-flash-exp")
    
    assert extractor.model_name == "gemini-2.0-flash-exp"


def test_booking_extractor_no_api_key(monkeypatch):
    """Test BookingExtractor raises error without API key."""
    # Remove the API key from environment
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    with pytest.raises(ValueError) as exc_info:
        BookingExtractor()
    
    assert "GEMINI_API_KEY" in str(exc_info.value)


def test_booking_extractor_without_pattern_fallback(mock_gemini_api_key):
    """Test BookingExtractor without pattern fallback."""
    extractor = BookingExtractor(use_pattern_fallback=False)
    
    assert extractor.use_pattern_fallback is False
    assert extractor.pattern_extractor is None


# Test extraction with complete booking info
@pytest.mark.asyncio
async def test_extract_complete_booking_llm(
    booking_extractor_no_pattern,
    complete_ticket_text,
    mock_complete_extraction
):
    """Test extraction of complete booking info using LLM."""
    mock_response = create_mock_gemini_response(mock_complete_extraction)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(complete_ticket_text)
        
        assert result["found"] is True
        assert result["extraction_method"] == "llm"
        assert result["confidence"] == "high"
        
        booking_info = result["booking_info"]
        assert booking_info["booking_id"] == "PW-509266779"
        assert booking_info["amount"] == 45.00
        assert booking_info["event_date"] == "2025-11-15"
        assert booking_info["reservation_date"] == "2025-11-01"
        assert booking_info["location"] == "Downtown Parking Garage"
        assert booking_info["booking_type"] == "confirmed"
        assert booking_info["customer_email"] == "customer@example.com"


# Test extraction with partial booking info
@pytest.mark.asyncio
async def test_extract_partial_booking_llm(
    booking_extractor_no_pattern,
    partial_ticket_text,
    mock_partial_extraction
):
    """Test extraction of partial booking info using LLM."""
    mock_response = create_mock_gemini_response(mock_partial_extraction)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(partial_ticket_text)
        
        assert result["found"] is True
        assert result["extraction_method"] == "llm"
        assert result["confidence"] == "medium"
        
        booking_info = result["booking_info"]
        assert booking_info["booking_id"] == "123456789"
        assert booking_info["event_date"] == "2025-11-15"
        assert "amount" not in booking_info  # Not in partial extraction


# Test extraction with no booking info
@pytest.mark.asyncio
async def test_extract_no_booking_llm(
    booking_extractor_no_pattern,
    no_booking_ticket_text,
    mock_no_booking_extraction
):
    """Test extraction when no booking info is found."""
    mock_response = create_mock_gemini_response(mock_no_booking_extraction)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(no_booking_ticket_text)
        
        assert result["found"] is False
        assert result["extraction_method"] == "llm"
        assert result["confidence"] == "low"
        assert result["booking_info"] == {}


# Test extraction with multiple bookings
@pytest.mark.asyncio
async def test_extract_multiple_bookings_llm(
    booking_extractor_no_pattern,
    multiple_bookings_ticket_text,
    mock_multiple_bookings_extraction
):
    """Test extraction when multiple bookings are mentioned."""
    mock_response = create_mock_gemini_response(mock_multiple_bookings_extraction)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(multiple_bookings_ticket_text)
        
        assert result["found"] is True
        assert result["extraction_method"] == "llm"
        assert result.get("multiple_bookings") is True
        
        # Should extract the primary/disputed booking
        booking_info = result["booking_info"]
        assert booking_info["booking_id"] == "PW-222222222"


# Test pattern fallback behavior
@pytest.mark.asyncio
async def test_pattern_fallback_success(
    booking_extractor_with_pattern,
    complete_ticket_text
):
    """Test that pattern extraction is used when it succeeds with high confidence."""
    # Mock pattern extractor to return high confidence result
    mock_pattern_result = {
        "booking_info": {
            "booking_id": "PW-509266779",
            "amount": 45.00,
            "event_date": "2025-11-15"
        },
        "confidence": "high",
        "found": True,
        "extraction_method": "pattern"
    }
    
    with patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_text',
        return_value=mock_pattern_result
    ):
        result = await booking_extractor_with_pattern.extract_booking_info(complete_ticket_text)
        
        assert result["extraction_method"] == "pattern"
        assert result["confidence"] == "high"
        assert result["found"] is True


@pytest.mark.asyncio
async def test_pattern_fallback_to_llm(
    booking_extractor_with_pattern,
    partial_ticket_text,
    mock_partial_extraction
):
    """Test that LLM is used when pattern extraction has low confidence."""
    # Mock pattern extractor to return low confidence
    mock_pattern_result = {
        "booking_info": {"booking_id": "123456789"},
        "confidence": "low",
        "found": True,
        "extraction_method": "pattern"
    }
    
    mock_llm_response = create_mock_gemini_response(mock_partial_extraction)
    
    with patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_text',
        return_value=mock_pattern_result
    ), patch.object(
        booking_extractor_with_pattern.client.models,
        'generate_content',
        return_value=mock_llm_response
    ):
        result = await booking_extractor_with_pattern.extract_booking_info(partial_ticket_text)
        
        # Should fall back to LLM
        assert result["extraction_method"] == "llm"
        assert result["found"] is True


# Test empty input handling
@pytest.mark.asyncio
async def test_extract_empty_text(booking_extractor_no_pattern):
    """Test extraction with empty text."""
    result = await booking_extractor_no_pattern.extract_booking_info("")
    
    assert result["found"] is False
    assert result["confidence"] == "low"
    assert result["extraction_method"] == "none"
    assert result["booking_info"] == {}


@pytest.mark.asyncio
async def test_extract_whitespace_only(booking_extractor_no_pattern):
    """Test extraction with whitespace-only text."""
    result = await booking_extractor_no_pattern.extract_booking_info("   \n\t  ")
    
    assert result["found"] is False
    assert result["confidence"] == "low"
    assert result["extraction_method"] == "none"


# Test confidence calculation
def test_confidence_high(booking_extractor_no_pattern):
    """Test high confidence calculation."""
    result = {
        "booking_id": "PW-123",
        "event_date": "2025-11-15",
        "amount": 45.00,
        "reservation_date": "2025-11-01",
        "location": "Downtown",
        "booking_type": "confirmed",
        "found": True
    }
    confidence = booking_extractor_no_pattern._calculate_confidence(result)
    assert confidence == "high"


def test_confidence_medium_critical_only(booking_extractor_no_pattern):
    """Test medium confidence with critical fields only."""
    result = {
        "booking_id": "PW-123",
        "event_date": "2025-11-15",
        "found": True
    }
    confidence = booking_extractor_no_pattern._calculate_confidence(result)
    assert confidence == "medium"


def test_confidence_medium_one_critical_many_optional(booking_extractor_no_pattern):
    """Test medium confidence with one critical and many optional fields."""
    result = {
        "booking_id": "PW-123",
        "amount": 45.00,
        "location": "Downtown",
        "booking_type": "confirmed",
        "customer_email": "test@example.com",
        "found": True
    }
    confidence = booking_extractor_no_pattern._calculate_confidence(result)
    assert confidence == "medium"


def test_confidence_low_missing_critical(booking_extractor_no_pattern):
    """Test low confidence when missing critical fields."""
    result = {
        "amount": 45.00,
        "location": "Downtown",
        "found": True
    }
    confidence = booking_extractor_no_pattern._calculate_confidence(result)
    assert confidence == "low"


def test_confidence_low_not_found(booking_extractor_no_pattern):
    """Test low confidence when booking not found."""
    result = {"found": False}
    confidence = booking_extractor_no_pattern._calculate_confidence(result)
    assert confidence == "low"


# Test error handling
@pytest.mark.asyncio
async def test_llm_timeout_handling(booking_extractor_no_pattern, complete_ticket_text):
    """Test that LLM timeout is handled gracefully."""
    import asyncio
    
    # Simulate a timeout by raising TimeoutError
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        side_effect=asyncio.TimeoutError("Timeout after 10 seconds")
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(complete_ticket_text)
        
        assert result["found"] is False
        assert result["confidence"] == "low"
        assert result["extraction_method"] == "llm"
        assert "error" in result
        assert "Timeout" in result["error"]


@pytest.mark.asyncio
async def test_llm_json_parse_error(booking_extractor_no_pattern, complete_ticket_text):
    """Test handling of invalid JSON response from LLM."""
    mock_response = Mock()
    mock_response.text = "Invalid JSON {not valid}"
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(complete_ticket_text)
        
        assert result["found"] is False
        assert result["confidence"] == "low"
        assert result["extraction_method"] == "llm"
        assert "error" in result
        assert "JSON parsing error" in result["error"]


@pytest.mark.asyncio
async def test_llm_api_error(booking_extractor_no_pattern, complete_ticket_text):
    """Test handling of LLM API errors."""
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        side_effect=Exception("API Error: Rate limit exceeded")
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(complete_ticket_text)
        
        assert result["found"] is False
        assert result["confidence"] == "low"
        assert result["extraction_method"] == "llm"
        assert "error" in result
        assert "Exception" in result["error"]


# Test prompt creation
def test_create_extraction_prompt(booking_extractor_no_pattern):
    """Test that extraction prompt is created correctly."""
    ticket_notes = "Test ticket with booking PW-123"
    prompt = booking_extractor_no_pattern._create_extraction_prompt(ticket_notes)
    
    assert "Extract booking information" in prompt
    assert "Booking ID" in prompt
    assert "Amount" in prompt
    assert "Event Date" in prompt
    assert ticket_notes in prompt
    assert "ISO format" in prompt
    assert "multiple_bookings" in prompt


# Test HTML vs text detection
@pytest.mark.asyncio
async def test_html_detection_pattern_extractor(booking_extractor_with_pattern):
    """Test that HTML content is detected and routed to HTML extraction."""
    html_content = "<html><body><p>Booking: PW-123</p></body></html>"
    
    mock_html_result = {
        "booking_info": {"booking_id": "PW-123"},
        "confidence": "medium",
        "found": True,
        "extraction_method": "pattern"
    }
    
    with patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_html',
        return_value=mock_html_result
    ) as mock_html, patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_text'
    ) as mock_text:
        result = await booking_extractor_with_pattern.extract_booking_info(html_content)
        
        # Should call HTML extraction, not text
        mock_html.assert_called_once()
        mock_text.assert_not_called()
        assert result["extraction_method"] == "pattern"


@pytest.mark.asyncio
async def test_text_detection_pattern_extractor(booking_extractor_with_pattern):
    """Test that plain text is routed to text extraction."""
    text_content = "Booking: PW-123 for $45.00"
    
    mock_text_result = {
        "booking_info": {"booking_id": "PW-123", "amount": 45.00},
        "confidence": "medium",
        "found": True,
        "extraction_method": "pattern"
    }
    
    with patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_text',
        return_value=mock_text_result
    ) as mock_text, patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_html'
    ) as mock_html:
        result = await booking_extractor_with_pattern.extract_booking_info(text_content)
        
        # Should call text extraction, not HTML
        mock_text.assert_called_once()
        mock_html.assert_not_called()
        assert result["extraction_method"] == "pattern"


# Test pattern extraction error handling
@pytest.mark.asyncio
async def test_pattern_extraction_error_fallback(
    booking_extractor_with_pattern,
    complete_ticket_text,
    mock_complete_extraction
):
    """Test that pattern extraction errors fall back to LLM."""
    mock_llm_response = create_mock_gemini_response(mock_complete_extraction)
    
    with patch.object(
        booking_extractor_with_pattern.pattern_extractor,
        'extract_from_text',
        side_effect=Exception("Pattern extraction failed")
    ), patch.object(
        booking_extractor_with_pattern.client.models,
        'generate_content',
        return_value=mock_llm_response
    ):
        result = await booking_extractor_with_pattern.extract_booking_info(complete_ticket_text)
        
        # Should fall back to LLM
        assert result["extraction_method"] == "llm"
        assert result["found"] is True


# Test edge cases
@pytest.mark.asyncio
async def test_extract_with_null_values_in_response(booking_extractor_no_pattern, complete_ticket_text):
    """Test that null values in LLM response are filtered out."""
    mock_response_with_nulls = {
        "booking_id": "PW-123",
        "amount": None,
        "event_date": "2025-11-15",
        "location": None,
        "found": True,
        "multiple_bookings": False
    }
    
    mock_response = create_mock_gemini_response(mock_response_with_nulls)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(complete_ticket_text)
        
        booking_info = result["booking_info"]
        # Null values should be filtered out
        assert "amount" not in booking_info
        assert "location" not in booking_info
        # Non-null values should be present
        assert booking_info["booking_id"] == "PW-123"
        assert booking_info["event_date"] == "2025-11-15"


@pytest.mark.asyncio
async def test_extract_very_long_ticket(booking_extractor_no_pattern, mock_complete_extraction):
    """Test extraction from very long ticket text."""
    long_ticket = "Customer inquiry: " + ("Lorem ipsum dolor sit amet. " * 1000) + " Booking: PW-123"
    
    mock_response = create_mock_gemini_response(mock_complete_extraction)
    
    with patch.object(
        booking_extractor_no_pattern.client.models,
        'generate_content',
        return_value=mock_response
    ):
        result = await booking_extractor_no_pattern.extract_booking_info(long_ticket)
        
        # Should handle long text without errors
        assert result["found"] is True
        assert result["extraction_method"] == "llm"
