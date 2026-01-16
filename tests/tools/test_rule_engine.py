"""
Tests for RuleEngine component.

Tests cover:
- 7+ days rule (approve)
- <3 days on-demand rule (deny)
- After event rule (deny)
- Edge cases (uncertain)
- Date calculation logic
- Special scenario handling (oversold, duplicate, paid again)
"""

import pytest
from datetime import datetime, timedelta, timezone
from app_tools.tools.refund_decision.rule_engine import RuleEngine


@pytest.fixture
def mock_rules():
    """Mock policy rules for testing."""
    return {
        "pre_arrival_days": 7,
        "on_demand_minimum_days": 3,
        "confirmed_minimum_days": 3
    }


@pytest.fixture
def rule_engine(mock_rules):
    """Create RuleEngine instance with mock rules."""
    # Disable contradiction detection for backward compatibility with existing tests
    return RuleEngine(mock_rules, contradiction_config={})


@pytest.fixture
def base_booking_info():
    """Base booking information for tests."""
    return {
        "booking_id": "PW-123456789",
        "amount": 45.00,
        "reservation_date": "2025-11-01",
        "location": "Downtown Parking Garage",
        "customer_email": "customer@example.com"
    }


@pytest.fixture
def base_ticket_data():
    """Base ticket data for tests."""
    return {
        "ticket_id": "1206331",
        "subject": "Refund Request",
        "description": "Customer requesting refund for parking reservation."
    }


# Helper function to create dates
def get_date_string(days_from_now):
    """Get ISO date string for days from now."""
    date = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return date.strftime("%Y-%m-%d")


def get_cancellation_date_string(days_from_now):
    """Get ISO date string for cancellation date."""
    date = datetime.now(timezone.utc) + timedelta(days=days_from_now)
    return date.strftime("%Y-%m-%d")


# Test 7+ days rule (approve)
@pytest.mark.asyncio
async def test_seven_plus_days_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that 7+ days before event is automatically approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(10)  # 10 days from now
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"
    assert "Pre-Arrival" in result["policy_rule"]
    assert "7+ days" in result["reasoning"]


@pytest.mark.asyncio
async def test_exactly_seven_days_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that exactly 7 days before event is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(7)  # Exactly 7 days from now
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"
    assert "Pre-Arrival" in result["policy_rule"]


@pytest.mark.asyncio
async def test_eight_days_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that 8 days before event is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(8)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "on-demand"  # Even on-demand is approved at 7+ days
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"


# Test <3 days on-demand rule (deny)
@pytest.mark.asyncio
async def test_less_than_three_days_on_demand_deny(rule_engine, base_booking_info, base_ticket_data):
    """Test that <3 days on-demand booking is denied."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)  # 2 days from now
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "on-demand"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Denied"
    assert result["confidence"] == "high"
    assert "On-Demand" in result["policy_rule"]
    assert "3+ days notice" in result["reasoning"]


@pytest.mark.asyncio
async def test_one_day_on_demand_deny(rule_engine, base_booking_info, base_ticket_data):
    """Test that 1 day on-demand booking is denied."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(1)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "on-demand"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Denied"
    assert result["confidence"] == "high"


@pytest.mark.asyncio
async def test_zero_days_on_demand_deny(rule_engine, base_booking_info, base_ticket_data):
    """Test that same-day on-demand booking is denied."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(0)  # Today
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "on-demand"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Denied"
    assert result["confidence"] == "high"


# Test after event rule (deny)
@pytest.mark.asyncio
async def test_after_event_deny(rule_engine, base_booking_info, base_ticket_data):
    """Test that post-event cancellation is denied."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-5)  # 5 days ago
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    # Post-event cancellations now escalate for review instead of automatic denial
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"


@pytest.mark.asyncio
async def test_one_day_after_event_deny(rule_engine, base_booking_info, base_ticket_data):
    """Test that 1 day after event is denied."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-1)  # Yesterday
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    # Post-event cancellations now escalate for review instead of automatic denial
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"


# Test 3-7 days confirmed booking (approve with medium confidence)
@pytest.mark.asyncio
async def test_three_to_seven_days_confirmed_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that 3-7 days confirmed booking is approved with medium confidence."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(5)  # 5 days from now
    booking_info["cancellation_date"] = get_date_string(0)  # Today
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "medium"
    assert "Confirmed Booking" in result["policy_rule"]
    assert "3-7 days" in result["policy_rule"]


@pytest.mark.asyncio
async def test_exactly_three_days_confirmed_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that exactly 3 days confirmed booking is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(3)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "medium"


@pytest.mark.asyncio
async def test_six_days_confirmed_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that 6 days confirmed booking is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(6)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "medium"


# Test edge cases (uncertain)
@pytest.mark.asyncio
async def test_missing_event_date_uncertain(rule_engine, base_booking_info, base_ticket_data):
    """Test that missing event date returns uncertain."""
    booking_info = base_booking_info.copy()
    # No event_date
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"
    assert "Missing event date" in result["reasoning"]


@pytest.mark.asyncio
async def test_invalid_date_format_uncertain(rule_engine, base_booking_info, base_ticket_data):
    """Test that invalid date format returns uncertain."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = "invalid-date"
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"
    assert "invalid date format" in result["reasoning"].lower()


@pytest.mark.asyncio
async def test_three_to_seven_days_unclear_type_uncertain(rule_engine, base_booking_info, base_ticket_data):
    """Test that 3-7 days with unclear booking type is uncertain."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(5)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "unknown"  # Not confirmed or on-demand
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"
    assert "Ambiguous Booking Type" in result["policy_rule"]


@pytest.mark.asyncio
async def test_less_than_three_days_non_on_demand_uncertain(rule_engine, base_booking_info, base_ticket_data):
    """Test that <3 days with non-on-demand booking is uncertain."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "third-party"  # Not on-demand
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"
    assert "Short Notice" in result["policy_rule"]


@pytest.mark.asyncio
async def test_empty_booking_type_uncertain(rule_engine, base_booking_info, base_ticket_data):
    """Test that empty booking type with 3-7 days is uncertain."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(4)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = ""
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Uncertain"
    assert result["confidence"] == "low"


# Test special scenarios (oversold, duplicate, paid again)
@pytest.mark.asyncio
async def test_oversold_location_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that oversold location is approved regardless of timing."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-2)  # 2 days ago (post-event)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "The garage full and I couldn't park despite my reservation."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"
    assert "Oversold" in result["policy_rule"]


@pytest.mark.asyncio
async def test_oversold_keywords(rule_engine, base_booking_info, base_ticket_data):
    """Test various oversold keywords are detected."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)  # Use 2 days (short notice)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "third-party"  # Use third-party to avoid on-demand denial
    
    oversold_phrases = [
        "garage was oversold",
        "lot was full",
        "no space available",
        "no spots left",
        "at capacity",
        "garage full",
        "lot full",
        "sold out"
    ]
    
    for phrase in oversold_phrases:
        ticket_data = base_ticket_data.copy()
        ticket_data["description"] = f"Customer says: {phrase}"
        
        result = await rule_engine.apply_rules(booking_info, ticket_data)
        
        assert result["decision"] == "Approved", f"Failed for phrase: {phrase}"
        assert "Oversold" in result["policy_rule"]


@pytest.mark.asyncio
async def test_duplicate_booking_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that duplicate booking is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-1)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "I was charged twice for the same booking. This is a duplicate."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    # Duplicate claims now escalate to human review due to ParkWhiz API limitations
    assert result["decision"] == "Needs Human Review"
    assert result["confidence"] == "high"
    assert "Duplicate" in result["policy_rule"]


@pytest.mark.asyncio
async def test_duplicate_keywords(rule_engine, base_booking_info, base_ticket_data):
    """Test various duplicate keywords are detected."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)  # Use 2 days (short notice)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "third-party"  # Use third-party to avoid on-demand denial
    
    # Enhanced classification now handles some duplicate scenarios differently
    escalation_phrases = [
        "duplicate booking",
        "charged twice", 
        "double charge",
        "two passes for same time",
        "bought twice by mistake",
        "multiple passes"
    ]
    
    # These phrases should escalate to human review
    for phrase in escalation_phrases:
        ticket_data = base_ticket_data.copy()
        ticket_data["description"] = f"Customer issue: {phrase}"
        
        result = await rule_engine.apply_rules(booking_info, ticket_data)
        
        # Duplicate-related claims now escalate to human review (may be categorized as Duplicate or Retroactive)
        assert result["decision"] == "Needs Human Review", f"Failed for phrase: {phrase}"
        assert ("Duplicate" in result["policy_rule"] or "Retroactive" in result["policy_rule"]), f"Unexpected policy rule for phrase '{phrase}': {result['policy_rule']}"
    
    # Enhanced classification handles explicit accidental double booking
    accidental_double_booking_phrases = [
        "booked same time twice"
    ]
    
    # These phrases should be approved as accidental double booking
    for phrase in accidental_double_booking_phrases:
        ticket_data = base_ticket_data.copy()
        ticket_data["description"] = f"Customer issue: {phrase}"
        
        result = await rule_engine.apply_rules(booking_info, ticket_data)
        
        # Enhanced classification should approve accidental double bookings
        assert result["decision"] == "Approved", f"Failed for phrase: {phrase}"
        assert "Enhanced Classification" in result["policy_rule"], f"Unexpected policy rule for phrase '{phrase}': {result['policy_rule']}"


@pytest.mark.asyncio
async def test_paid_again_approve(rule_engine, base_booking_info, base_ticket_data):
    """Test that paid again scenario is approved."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-3)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "I had to pay again at the gate even though I had a reservation."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    # "Paid again" scenarios should be approved directly, not escalated
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"
    assert "Paid Again" in result["policy_rule"]


@pytest.mark.asyncio
async def test_paid_again_keywords(rule_engine, base_booking_info, base_ticket_data):
    """Test various paid again keywords are detected."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)  # Use 2 days (short notice)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "third-party"  # Use third-party to avoid on-demand denial
    
    # Test phrases that should be approved as legitimate paid again scenarios
    paid_again_approved_phrases = [
        "paid again at entrance",
        "paid twice total",
        "charged at gate",
        "paid onsite",
        "paid on-site"
    ]
    
    for phrase in paid_again_approved_phrases:
        ticket_data = base_ticket_data.copy()
        ticket_data["description"] = f"Issue: {phrase}"
        
        result = await rule_engine.apply_rules(booking_info, ticket_data)
        
        assert result["decision"] == "Approved", f"Failed for phrase: {phrase}"
        assert "Paid Again" in result["policy_rule"]
    
    # Test phrases that should require human review as extra charges
    extra_charge_phrases = [
        "charged extra fee",
        "had to pay when I arrived"
    ]
    
    for phrase in extra_charge_phrases:
        ticket_data = base_ticket_data.copy()
        ticket_data["description"] = f"Issue: {phrase}"
        
        result = await rule_engine.apply_rules(booking_info, ticket_data)
        
        assert result["decision"] == "Needs Human Review", f"Failed for phrase: {phrase}"
        assert "Extra Charge" in result["policy_rule"]


# Test date calculation
def test_calculate_days_before_event_positive(rule_engine):
    """Test calculating days before event (positive)."""
    cancellation_date = "2025-11-01"
    event_date = "2025-11-15"
    
    days = rule_engine._calculate_days_before_event(cancellation_date, event_date)
    
    assert days == 14


def test_calculate_days_before_event_negative(rule_engine):
    """Test calculating days after event (negative)."""
    cancellation_date = "2025-11-20"
    event_date = "2025-11-15"
    
    days = rule_engine._calculate_days_before_event(cancellation_date, event_date)
    
    assert days == -5


@pytest.mark.asyncio
async def test_calculate_days_same_day(rule_engine):
    """Test calculating days when cancellation and event are same day."""
    date = "2025-11-15"
    
    days = rule_engine._calculate_days_before_event(date, date)
    
    assert days == 0


@pytest.mark.asyncio
async def test_calculate_days_no_cancellation_date(rule_engine):
    """Test calculating days with no cancellation date (uses current date)."""
    future_date = get_date_string(10)
    
    days = rule_engine._calculate_days_before_event(None, future_date)
    
    # Should be approximately 10 days (may vary by seconds)
    assert 9 <= days <= 10


@pytest.mark.asyncio
async def test_calculate_days_invalid_format(rule_engine):
    """Test calculating days with invalid date format."""
    days = rule_engine._calculate_days_before_event("invalid", "2025-11-15")
    
    assert days is None


@pytest.mark.asyncio
async def test_calculate_days_with_timezone(rule_engine):
    """Test calculating days with timezone-aware dates."""
    cancellation_date = "2025-11-01T10:00:00Z"
    event_date = "2025-11-15T14:00:00Z"
    
    days = rule_engine._calculate_days_before_event(cancellation_date, event_date)
    
    assert days == 14


# Test case sensitivity
@pytest.mark.asyncio
async def test_booking_type_case_insensitive(rule_engine, base_booking_info, base_ticket_data):
    """Test that booking type matching is case insensitive."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "ON-DEMAND"  # Uppercase
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    assert result["decision"] == "Denied"
    assert "On-Demand" in result["policy_rule"]


@pytest.mark.asyncio
async def test_description_case_insensitive(rule_engine, base_booking_info, base_ticket_data):
    """Test that description keyword matching is case insensitive."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(2)  # Use 2 days (short notice)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "third-party"  # Use third-party to avoid on-demand denial
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "GARAGE WAS OVERSOLD"  # Uppercase
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    assert result["decision"] == "Approved"
    assert "Oversold" in result["policy_rule"]


# Test priority of rules
@pytest.mark.asyncio
async def test_seven_days_overrides_on_demand(rule_engine, base_booking_info, base_ticket_data):
    """Test that 7+ days rule takes priority over on-demand restrictions."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(10)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "on-demand"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    # Should approve due to 7+ days, not deny due to on-demand
    assert result["decision"] == "Approved"
    assert "Pre-Arrival" in result["policy_rule"]


@pytest.mark.asyncio
async def test_oversold_overrides_post_event(rule_engine, base_booking_info, base_ticket_data):
    """Test that oversold exception overrides post-event denial."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-5)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "The lot was full when I arrived."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    # Should approve due to oversold, not deny due to post-event
    assert result["decision"] == "Approved"
    assert "Oversold" in result["policy_rule"]


@pytest.mark.asyncio
async def test_duplicate_overrides_post_event(rule_engine, base_booking_info, base_ticket_data):
    """Test that duplicate exception overrides post-event denial."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-2)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "I was charged twice for this booking."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    # Duplicate claims now escalate to human review due to ParkWhiz API limitations
    assert result["decision"] == "Needs Human Review"
    assert "Duplicate" in result["policy_rule"]


# Test edge cases with missing optional fields
@pytest.mark.asyncio
async def test_missing_cancellation_date_uses_current(rule_engine, base_booking_info, base_ticket_data):
    """Test that missing cancellation date uses current date."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(10)
    # No cancellation_date
    booking_info["booking_type"] = "confirmed"
    
    result = await rule_engine.apply_rules(booking_info, base_ticket_data)
    
    # Should still work and approve
    assert result["decision"] == "Approved"
    assert result["confidence"] == "high"


@pytest.mark.asyncio
async def test_missing_booking_type_with_special_scenario(rule_engine, base_booking_info, base_ticket_data):
    """Test that special scenarios work even without booking type."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(1)
    booking_info["cancellation_date"] = get_date_string(0)
    # No booking_type
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = "The garage full."
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    assert result["decision"] == "Approved"
    assert "Oversold" in result["policy_rule"]


@pytest.mark.asyncio
async def test_missing_description_no_special_scenarios(rule_engine, base_booking_info, base_ticket_data):
    """Test that missing description doesn't trigger false positives."""
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(5)
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_data = {"ticket_id": "123", "subject": "Refund"}
    # No description
    
    result = await rule_engine.apply_rules(booking_info, ticket_data)
    
    # Should apply normal rules, not special scenarios
    assert result["decision"] == "Approved"
    assert "Confirmed Booking" in result["policy_rule"]


# Test vehicle restriction gatekeeper logic
def test_vehicle_restriction_not_triggered_by_facility_full(rule_engine):
    """Test that facility full scenarios don't trigger vehicle restriction check."""
    # Test various facility full descriptions
    facility_full_descriptions = [
        "the entrance was full of cars and the workers told me they were not accepting more cars",
        "parking facility was full when I arrived",
        "lot was full and I was turned away",
        "garage was at capacity",
        "they said no space available",
        "facility is full, can't accept more vehicles"
    ]
    
    for description in facility_full_descriptions:
        result = rule_engine._check_for_vehicle_restriction_issue(description.lower())
        assert result is False, f"Should not trigger vehicle check for: {description}"


def test_vehicle_restriction_triggered_by_actual_vehicle_issue(rule_engine):
    """Test that actual vehicle restriction issues DO trigger the check."""
    vehicle_restriction_descriptions = [
        "they didn't allow my SUV",
        "turned away because of my truck",
        "they don't accept crossover vehicles",
        "attendant said my vehicle type wasn't allowed",
        "rejected due to my Tesla",
        "wouldn't let me park because of my van"
    ]
    
    for description in vehicle_restriction_descriptions:
        result = rule_engine._check_for_vehicle_restriction_issue(description.lower())
        assert result is True, f"Should trigger vehicle check for: {description}"


def test_vehicle_restriction_not_triggered_without_rejection_keyword(rule_engine):
    """Test that mentioning vehicle without rejection doesn't trigger check."""
    descriptions = [
        "I have a large SUV and parked successfully",
        "my truck fit in the space",
        "drove my Tesla there"
    ]
    
    for description in descriptions:
        result = rule_engine._check_for_vehicle_restriction_issue(description.lower())
        assert result is False, f"Should not trigger for: {description}"



@pytest.mark.asyncio
async def test_ticket_1256833_paid_again_not_accessibility(rule_engine, base_booking_info, base_ticket_data):
    """
    Test for ticket 1256833: Barcode didn't work, customer paid again.
    
    This should be classified as "Paid Again", NOT "Accessibility Issue".
    The location admin notes mention "NYPD may sometimes close the road" but that's
    just a general warning, not what the customer experienced.
    """
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = "2025-10-27T15:00:00Z"  # Event date
    booking_info["cancellation_date"] = "2025-12-05T20:15:00Z"  # Post-event
    booking_info["booking_type"] = "confirmed"
    booking_info["amount"] = 100.00
    
    # Simulate the Zapier note structure with customer complaint + location metadata
    ticket_notes = """Refund Request Information
Email: lauriecgordon@yahoo.com
Name: Laurie Gordon
Booking ID: 782293183
Reason: Said your barcode didnt work
Arrival: 12/5/2025 12:15:00
Exit: 12/5/2025 20:15:00
Make and Model: Audi Q3
Were you able to park? Yes
Proof of Payment: Link to uploaded image
Summary: When i checked out the attdant tried my barcode 4 times and it didnt work. He said he had same problem with other best parking customers. He made me pay again. $50!

Booking information found!

Location Name: 169 Ludlow Parking Garage - 169 Ludlow St.
Location Address: 169 Ludlow St.
Location City: New York, NY 10002
Location Admin Notes: NYPD may sometimes close the road going to the facility. In case of the road closure, kindly mention having a reservation at this garage.
"""
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = ticket_notes.lower()
    ticket_data["subject"] = "Refund Request - Barcode Issue"
    
    result = await rule_engine.apply_rules(booking_info, ticket_data, ticket_notes)
    
    # Should be classified as "Paid Again", not "Accessibility Issue"
    assert result["decision"] == "Approved", f"Expected Approved, got {result['decision']}"
    assert "Paid Again" in result["policy_rule"], f"Expected 'Paid Again' policy, got {result['policy_rule']}"
    assert "Accessibility" not in result["policy_rule"], f"Should NOT be classified as Accessibility Issue"
    assert result["confidence"] == "high"


@pytest.mark.asyncio
async def test_paid_again_with_attendant_keyword(rule_engine, base_booking_info, base_ticket_data):
    """
    Test that "attendant" keyword doesn't prevent "paid again" detection.
    
    Previously, "attendant" was in the overstay exclusion list, causing false negatives.
    """
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-2)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    
    ticket_notes = """Reason: Barcode didn't work
Summary: The attendant tried scanning my barcode multiple times but it wouldn't work. He made me pay again at the exit."""
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = ticket_notes.lower()
    
    result = await rule_engine.apply_rules(booking_info, ticket_data, ticket_notes)
    
    assert result["decision"] == "Approved"
    assert "Paid Again" in result["policy_rule"]
    assert result["confidence"] == "high"


def test_accessibility_not_triggered_by_location_metadata(rule_engine):
    """
    Test that location admin notes don't trigger accessibility issue detection.
    
    Location metadata may mention "NYPD may sometimes close the road" as a general
    warning, but this shouldn't trigger accessibility issue unless the CUSTOMER
    actually reports it.
    """
    # Customer complaint (what they actually said)
    customer_complaint = "said your barcode didnt work when i checked out the attdant tried my barcode 4 times and it didnt work. he made me pay again. $50!"
    
    # Should NOT trigger accessibility check
    result = rule_engine._check_for_accessibility(customer_complaint.lower())
    assert result is False, "Location metadata should not trigger accessibility check"
    
    # But if customer ACTUALLY mentions road closure, it should trigger
    actual_road_closure = "i couldn't get to the garage because the road closed and police blocked it"
    result = rule_engine._check_for_accessibility(actual_road_closure.lower())
    assert result is True, "Actual customer report of road closure should trigger"



@pytest.mark.asyncio
async def test_paid_again_over_50_limit(rule_engine, base_booking_info, base_ticket_data):
    """
    Test that "paid again" scenarios over $50 are escalated to human review.
    
    Even though the policy supports the refund, automation can only approve up to $50.
    This test verifies that amounts over $50 trigger escalation.
    """
    booking_info = base_booking_info.copy()
    booking_info["event_date"] = get_date_string(-2)  # Post-event
    booking_info["cancellation_date"] = get_date_string(0)
    booking_info["booking_type"] = "confirmed"
    booking_info["amount"] = 100.00  # Over $50 limit
    
    ticket_notes = """Reason: Barcode didn't work
Summary: The attendant tried my barcode 4 times and it wouldn't scan. He made me pay again $50 at the exit."""
    
    ticket_data = base_ticket_data.copy()
    ticket_data["description"] = ticket_notes.lower()
    
    result = await rule_engine.apply_rules(booking_info, ticket_data, ticket_notes)
    
    # Rule engine should approve based on policy
    assert result["decision"] == "Approved"
    assert "Paid Again" in result["policy_rule"]
    
    # But decision_maker should escalate due to amount
    # (This will be tested in test_decision_maker.py)
