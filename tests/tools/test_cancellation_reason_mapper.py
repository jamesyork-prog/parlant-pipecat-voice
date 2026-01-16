"""
Tests for CancellationReasonMapper component.

Tests cover:
- Mapping for common scenarios (oversold, duplicate, pre-arrival)
- Default to "Other" for ambiguous cases
- All valid ParkWhiz cancellation reasons
- Validation of cancellation reasons
- Case-insensitive keyword matching
"""

import pytest
from app_tools.tools.utils.cancellation_mapper import CancellationReasonMapper


@pytest.fixture
def mapper():
    """Create CancellationReasonMapper instance."""
    return CancellationReasonMapper()


# Test initialization
def test_mapper_initialization(mapper):
    """Test CancellationReasonMapper initializes correctly."""
    assert mapper is not None
    assert len(mapper.VALID_REASONS) == 16
    assert "Other" in mapper.VALID_REASONS
    assert "Oversold" in mapper.VALID_REASONS
    assert len(mapper.keyword_patterns) > 0


def test_valid_reasons_list(mapper):
    """Test that all expected valid reasons are present."""
    expected_reasons = [
        "Other",
        "Tolerance",
        "Multi-day",
        "Pending re-book",
        "Pre-arrival",
        "Oversold",
        "No attendant",
        "Amenity missing",
        "Poor experience",
        "Inaccurate hours of operation",
        "Attendant refused customer",
        "Duplicate booking",
        "Confirmed re-book",
        "Paid again",
        "Accessibility",
        "PW cancellation"
    ]
    
    assert set(mapper.VALID_REASONS) == set(expected_reasons)


# Test common scenario mappings
def test_map_oversold_scenario(mapper):
    """Test mapping for oversold location scenario."""
    reasoning = "The location was oversold and customer could not park"
    policy = "Oversold location policy - full refund"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_oversold_variations(mapper):
    """Test various oversold keyword variations."""
    test_cases = [
        ("Location was over-sold", "Policy applied"),
        ("The garage was overbooked", "Standard policy"),
        ("Facility over-booked for the event", "Refund policy"),
        ("Location sold out, customer turned away", "Full refund")
    ]
    
    for reasoning, policy in test_cases:
        result = mapper.map_reason(reasoning, policy)
        assert result == "Oversold", f"Failed for: {reasoning}"


def test_map_duplicate_booking_scenario(mapper):
    """Test mapping for duplicate booking scenario."""
    reasoning = "Customer has duplicate booking for same time slot"
    policy = "Duplicate pass policy - refund one booking"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Duplicate booking"


def test_map_duplicate_variations(mapper):
    """Test various duplicate booking keyword variations."""
    test_cases = [
        ("Customer duplicated their reservation", "Refund policy"),
        ("This is a double booking situation", "Standard policy"),
        ("Customer booked twice by mistake", "Full refund"),
        ("Duplicated pass detected", "Duplicate policy")
    ]
    
    for reasoning, policy in test_cases:
        result = mapper.map_reason(reasoning, policy)
        assert result == "Duplicate booking", f"Failed for: {reasoning}"


def test_map_pre_arrival_scenario(mapper):
    """Test mapping for pre-arrival cancellation scenario."""
    reasoning = "Customer cancelled 7+ days before event"
    policy = "Pre-arrival cancellation policy - full refund"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Pre-arrival"


def test_map_pre_arrival_variations(mapper):
    """Test various pre-arrival keyword variations."""
    test_cases = [
        ("Cancelled pre-arrival with sufficient notice", "Standard policy"),
        ("Pre arrival cancellation requested", "Refund policy"),
        ("Cancelled before event with 10 days notice", "Full refund"),
        ("7+ days advance cancellation", "Pre-arrival policy")
    ]
    
    for reasoning, policy in test_cases:
        result = mapper.map_reason(reasoning, policy)
        assert result == "Pre-arrival", f"Failed for: {reasoning}"


# Test all valid ParkWhiz cancellation reasons
def test_map_tolerance_reason(mapper):
    """Test mapping to Tolerance reason."""
    reasoning = "Approved as goodwill gesture for customer satisfaction"
    policy = "Tolerance policy applied"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Tolerance"


def test_map_multi_day_reason(mapper):
    """Test mapping to Multi-day reason."""
    reasoning = "Customer has multi-day booking and wants partial refund"
    policy = "Multi-day booking policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Multi-day"


def test_map_pending_rebook_reason(mapper):
    """Test mapping to Pending re-book reason."""
    reasoning = "Customer will rebook for different date"
    policy = "Pending re-book policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Pending re-book"


def test_map_confirmed_rebook_reason(mapper):
    """Test mapping to Confirmed re-book reason."""
    reasoning = "Customer has confirmed rebook for next week"
    policy = "Confirmed re-book policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Confirmed re-book"


def test_map_no_attendant_reason(mapper):
    """Test mapping to No attendant reason."""
    reasoning = "No attendant was present at the location"
    policy = "Attendant missing policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "No attendant"


def test_map_amenity_missing_reason(mapper):
    """Test mapping to Amenity missing reason."""
    reasoning = "Advertised amenity was not available at facility"
    policy = "Amenity missing policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Amenity missing"


def test_map_poor_experience_reason(mapper):
    """Test mapping to Poor experience reason."""
    reasoning = "Customer had poor experience and filed complaint"
    policy = "Service quality policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Poor experience"


def test_map_inaccurate_hours_reason(mapper):
    """Test mapping to Inaccurate hours of operation reason."""
    reasoning = "Location was closed during advertised operating hours"
    policy = "Hours of operation policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Inaccurate hours of operation"


def test_map_attendant_refused_reason(mapper):
    """Test mapping to Attendant refused customer reason."""
    reasoning = "Attendant refused entry to customer despite valid booking"
    policy = "Attendant refusal policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Attendant refused customer"


def test_map_paid_again_reason(mapper):
    """Test mapping to Paid again reason."""
    reasoning = "Customer was charged twice for same booking"
    policy = "Double charge policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Paid again"


def test_map_accessibility_reason(mapper):
    """Test mapping to Accessibility reason."""
    reasoning = "Location did not meet ADA accessibility requirements"
    policy = "Accessibility policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Accessibility"


def test_map_pw_cancellation_reason(mapper):
    """Test mapping to PW cancellation reason."""
    reasoning = "ParkWhiz system cancelled the booking due to platform issue"
    policy = "Platform cancellation policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "PW cancellation"


# Test default to "Other" for ambiguous cases
def test_map_ambiguous_case_defaults_to_other(mapper):
    """Test that ambiguous cases default to Other."""
    reasoning = "Customer requested refund for personal reasons"
    policy = "Standard refund policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Other"


def test_map_empty_reasoning_defaults_to_other(mapper):
    """Test that empty reasoning defaults to Other."""
    result = mapper.map_reason("", "")
    
    assert result == "Other"


def test_map_no_keywords_defaults_to_other(mapper):
    """Test that text with no matching keywords defaults to Other."""
    reasoning = "The customer wants their money back"
    policy = "General policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Other"


def test_map_unrelated_text_defaults_to_other(mapper):
    """Test that unrelated text defaults to Other."""
    reasoning = "Lorem ipsum dolor sit amet consectetur adipiscing elit"
    policy = "Random policy text without keywords"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Other"


# Test case-insensitive matching
def test_map_case_insensitive_uppercase(mapper):
    """Test that uppercase keywords are matched."""
    reasoning = "LOCATION WAS OVERSOLD"
    policy = "OVERSOLD POLICY"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_case_insensitive_mixed(mapper):
    """Test that mixed case keywords are matched."""
    reasoning = "Customer Has DuPlIcAtE BoOkInG"
    policy = "Duplicate Policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Duplicate booking"


def test_map_case_insensitive_lowercase(mapper):
    """Test that lowercase keywords are matched."""
    reasoning = "customer cancelled pre-arrival"
    policy = "pre-arrival policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Pre-arrival"


# Test keyword priority (first match wins)
def test_map_multiple_keywords_first_match(mapper):
    """Test that when multiple keywords match, first pattern match is returned."""
    # This text could match both "Oversold" and "Duplicate booking"
    reasoning = "Location was oversold and customer had duplicate booking"
    policy = "Multiple issues policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    # Should match first pattern found (Oversold comes before Duplicate in patterns)
    assert result == "Oversold"


def test_map_keyword_in_policy_text(mapper):
    """Test that keywords in policy text are also matched."""
    reasoning = "Customer requested refund"
    policy = "Pre-arrival cancellation policy applies"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Pre-arrival"


def test_map_keyword_in_reasoning_only(mapper):
    """Test that keywords in reasoning alone are sufficient."""
    reasoning = "Location was oversold"
    policy = "Standard policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


# Test with booking_info parameter
def test_map_with_booking_info(mapper):
    """Test mapping with optional booking_info parameter."""
    reasoning = "Customer cancelled in advance"
    policy = "Pre-arrival policy"
    booking_info = {
        "booking_id": "PW-123",
        "booking_type": "confirmed"
    }
    
    result = mapper.map_reason(reasoning, policy, booking_info)
    
    assert result == "Pre-arrival"


def test_map_with_none_booking_info(mapper):
    """Test mapping with None booking_info."""
    reasoning = "Location was oversold"
    policy = "Oversold policy"
    
    result = mapper.map_reason(reasoning, policy, None)
    
    assert result == "Oversold"


def test_map_with_empty_booking_info(mapper):
    """Test mapping with empty booking_info dict."""
    reasoning = "Duplicate booking detected"
    policy = "Duplicate policy"
    
    result = mapper.map_reason(reasoning, policy, {})
    
    assert result == "Duplicate booking"


# Test validation method
def test_validate_reason_valid(mapper):
    """Test validation of valid cancellation reasons."""
    for reason in mapper.VALID_REASONS:
        assert mapper.validate_reason(reason) is True


def test_validate_reason_invalid(mapper):
    """Test validation of invalid cancellation reasons."""
    invalid_reasons = [
        "Invalid Reason",
        "Not A Real Reason",
        "Random Text",
        "",
        "other",  # lowercase - should be "Other"
        "oversold"  # lowercase - should be "Oversold"
    ]
    
    for reason in invalid_reasons:
        assert mapper.validate_reason(reason) is False


def test_validate_reason_case_sensitive(mapper):
    """Test that validation is case-sensitive."""
    # Valid reasons must match exact case
    assert mapper.validate_reason("Oversold") is True
    assert mapper.validate_reason("oversold") is False
    assert mapper.validate_reason("OVERSOLD") is False
    
    assert mapper.validate_reason("Other") is True
    assert mapper.validate_reason("other") is False


# Test edge cases
def test_map_with_special_characters(mapper):
    """Test mapping with special characters in text."""
    reasoning = "Customer's booking was over-sold!!! @#$%"
    policy = "Standard policy..."
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_with_newlines_and_tabs(mapper):
    """Test mapping with newlines and tabs in text."""
    reasoning = "Location\nwas\toversold\n\nCustomer\tcould not park"
    policy = "Oversold\tpolicy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_with_very_long_text(mapper):
    """Test mapping with very long text."""
    reasoning = "Customer inquiry: " + ("Lorem ipsum. " * 500) + " Location was oversold."
    policy = "Standard policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_keyword_at_start(mapper):
    """Test keyword matching at start of text."""
    reasoning = "Oversold location caused issue"
    policy = "Policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_keyword_at_end(mapper):
    """Test keyword matching at end of text."""
    reasoning = "Customer could not park because location was oversold"
    policy = "Policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


def test_map_keyword_in_middle(mapper):
    """Test keyword matching in middle of text."""
    reasoning = "The customer reported that the location was oversold and they were turned away"
    policy = "Standard policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Oversold"


# Test partial keyword matches (should not match)
def test_map_partial_keyword_no_match(mapper):
    """Test that partial keywords don't match."""
    # "sold" alone should not match "oversold"
    reasoning = "The item was sold to another customer"
    policy = "Sales policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Other"


def test_map_substring_in_word_no_match(mapper):
    """Test that substrings within words don't match."""
    # "old" in "behold" should not match "oversold"
    reasoning = "Behold the customer's request"
    policy = "Standard policy"
    
    result = mapper.map_reason(reasoning, policy)
    
    assert result == "Other"


# Test all keyword patterns have valid reasons
def test_all_keyword_patterns_map_to_valid_reasons(mapper):
    """Test that all keyword patterns map to valid ParkWhiz reasons."""
    for reason in mapper.keyword_patterns.keys():
        assert reason in mapper.VALID_REASONS, f"Pattern reason '{reason}' not in VALID_REASONS"


# Test comprehensive scenario coverage
def test_map_tolerance_variations(mapper):
    """Test various tolerance/goodwill keyword variations."""
    test_cases = [
        "Approved as tolerance for customer",
        "Goodwill gesture for loyal customer",
        "Exception made for customer satisfaction",
        "Courtesy refund approved"
    ]
    
    for reasoning in test_cases:
        result = mapper.map_reason(reasoning, "Policy")
        assert result == "Tolerance", f"Failed for: {reasoning}"


def test_map_amenity_variations(mapper):
    """Test various amenity missing keyword variations."""
    test_cases = [
        "Advertised amenity was not available",
        "Amenities did not match description",
        "Facility missing key features",
        "Facilities were not as advertised"
    ]
    
    for reasoning in test_cases:
        result = mapper.map_reason(reasoning, "Policy")
        assert result == "Amenity missing", f"Failed for: {reasoning}"


def test_map_poor_experience_variations(mapper):
    """Test various poor experience keyword variations."""
    test_cases = [
        "Customer had poor experience",
        "Customer filed complaint about service",
        "Customer was dissatisfied with parking",
        "Customer unhappy with experience",
        "Bad experience reported"
    ]
    
    for reasoning in test_cases:
        result = mapper.map_reason(reasoning, "Policy")
        assert result == "Poor experience", f"Failed for: {reasoning}"


def test_map_accessibility_variations(mapper):
    """Test various accessibility keyword variations."""
    test_cases = [
        "Location not accessible for wheelchair",
        "ADA requirements not met",
        "Disability access was inadequate",
        "Accessible parking not available"
    ]
    
    for reasoning in test_cases:
        result = mapper.map_reason(reasoning, "Policy")
        assert result == "Accessibility", f"Failed for: {reasoning}"
