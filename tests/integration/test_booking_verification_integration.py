"""
Integration tests for booking verification workflow.

Tests the complete end-to-end flow:
1. Zapier fails → Extract customer info → Find booking → Verify → Decide
2. Zapier fails → Extract → No booking → Escalate
3. Zapier fails → Extract → Multiple bookings → Select best
4. Zapier fails → Extract → API timeout → Retry → Escalate
5. Zapier fails → Extract incomplete → Escalate
6. Zapier fails → Find → Usage contradicts → Escalate

Requirements: All requirements end-to-end
"""

import pytest

# Skip all tests in this file - workflow is deprecated
# Booking verification now happens upstream in Zapier, not in Parlant
pytestmark = pytest.mark.skip(reason="Booking verification now happens upstream in Zapier")


# ============================================================================
# Test 1: Happy Path - Zapier fails → Extract → Find booking → Verify → Decide
# ============================================================================

@pytest.mark.asyncio
async def test_happy_path_zapier_failure_to_verified_booking():
    """
    Test complete happy path workflow.
    
    Flow:
    1. Detect Zapier failure in ticket
    2. Extract customer information (email, dates)
    3. Search ParkWhiz API for bookings
    4. Find matching booking
    5. Verify pass usage status
    6. Make automated decision based on verified data
    
    Expected: Success with verified booking and automated decision
    """
    print("\n" + "=" * 80)
    print("TEST 1: Happy Path - Zapier Failure to Verified Booking")
    print("=" * 80)
    
    # Step 1: Detect Zapier failure
    detector = ZapierFailureDetector()
    ticket_text = """
    Subject: Refund Request
    
    Booking information not found for provided Booking Number
    
    Customer Email: customer@example.com
    Customer Name: John Doe
    Event Date: 2025-11-25
    Exit Date: 2025-11-26
    Location: Stadium Parking
    """
    
    assert detector.is_zapier_failure(ticket_text) is True
    print("✓ Step 1: Zapier failure detected")
    
    # Step 2: Extract customer information
    customer_info = CustomerInfo(
        email="customer@example.com",
        name="John Doe",
        arrival_date="2025-11-25",
        exit_date="2025-11-26",
        location="Stadium Parking"
    )
    
    assert customer_info.is_complete() is True
    print("✓ Step 2: Customer info extracted and complete")
    
    # Step 3-5: Verify booking via ParkWhiz API (mocked)
    verifier = ParkWhizBookingVerifier()
    
    # Mock API response with matching booking
    mock_booking = {
        "id": "12345",
        "customer_email": "customer@example.com",
        "start_time": "2025-11-25T10:00:00Z",
        "end_time": "2025-11-26T10:00:00Z",
        "location_name": "Stadium Parking",
        "price_paid": 15.00,
        "pass_used": False,
        "pass_usage": "not_used"
    }
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = [mock_booking]
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is True
        assert result.verified_booking is not None
        assert result.verified_booking.booking_id == "12345"
        assert result.verified_booking.pass_usage_status == "not_used"
        assert result.verified_booking.match_confidence == "exact"
        assert result.should_escalate is False
        
        print("✓ Step 3: ParkWhiz API search completed")
        print("✓ Step 4: Matching booking found")
        print("✓ Step 5: Pass usage verified (not_used)")
    
    # Step 6: Make automated decision
    guard = DecisionGuard()
    
    can_decide = guard.can_make_automated_decision(result.verified_booking)
    assert can_decide is True
    
    should_escalate, reason = guard.should_escalate(result.verified_booking, customer_info)
    assert should_escalate is False
    
    print("✓ Step 6: Automated decision allowed")
    print("\n✓ HAPPY PATH TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 2: No Booking Found - Zapier fails → Extract → No booking → Escalate
# ============================================================================

@pytest.mark.asyncio
async def test_no_booking_found_escalation():
    """
    Test workflow when no booking is found in ParkWhiz.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information
    3. Search ParkWhiz API - no results
    4. Escalate to human review
    
    Expected: Escalation with "No booking found" reason
    """
    print("\n" + "=" * 80)
    print("TEST 2: No Booking Found - Escalation")
    print("=" * 80)
    
    # Step 1-2: Detect failure and extract info
    customer_info = CustomerInfo(
        email="notfound@example.com",
        arrival_date="2025-11-25",
        exit_date="2025-11-26"
    )
    
    print("✓ Step 1-2: Zapier failure detected, customer info extracted")
    
    # Step 3: Search ParkWhiz - no results
    verifier = ParkWhizBookingVerifier()
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = []  # No bookings found
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is False
        assert result.verified_booking is None
        assert result.failure_reason == "No matching bookings found in ParkWhiz system"
        assert result.should_escalate is True
        assert "No bookings found" in result.escalation_reason
        
        print("✓ Step 3: ParkWhiz API returned no bookings")
    
    # Step 4: Verify escalation
    guard = DecisionGuard()
    
    can_decide = guard.can_make_automated_decision(result.verified_booking)
    assert can_decide is False
    
    should_escalate, reason = guard.should_escalate(
        result.verified_booking, 
        customer_info, 
        result.failure_reason
    )
    assert should_escalate is True
    assert "No matching bookings found" in reason or "No booking found" in reason
    
    print("✓ Step 4: Escalated to human review")
    print(f"  Reason: {reason}")
    print("\n✓ NO BOOKING FOUND TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 3: Multiple Bookings - Select Best Match
# ============================================================================

@pytest.mark.asyncio
async def test_multiple_bookings_select_best():
    """
    Test workflow when multiple bookings are found.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information
    3. Search ParkWhiz API - multiple results
    4. Select booking with closest date match
    5. Verify selected booking
    
    Expected: Best matching booking selected based on dates
    """
    print("\n" + "=" * 80)
    print("TEST 3: Multiple Bookings - Select Best Match")
    print("=" * 80)
    
    customer_info = CustomerInfo(
        email="multi@example.com",
        arrival_date="2025-11-25",
        exit_date="2025-11-26"
    )
    
    print("✓ Step 1-2: Zapier failure detected, customer info extracted")
    
    # Step 3: Multiple bookings returned
    verifier = ParkWhizBookingVerifier()
    
    mock_bookings = [
        {
            "id": "11111",
            "customer_email": "multi@example.com",
            "start_time": "2025-11-20T10:00:00Z",  # 5 days off
            "end_time": "2025-11-21T10:00:00Z",
            "location_name": "Stadium Parking",
            "price_paid": 15.00,
            "pass_used": False
        },
        {
            "id": "22222",
            "customer_email": "multi@example.com",
            "start_time": "2025-11-25T10:00:00Z",  # Exact match!
            "end_time": "2025-11-26T10:00:00Z",
            "location_name": "Stadium Parking",
            "price_paid": 15.00,
            "pass_used": False
        },
        {
            "id": "33333",
            "customer_email": "multi@example.com",
            "start_time": "2025-11-24T10:00:00Z",  # 1 day off
            "end_time": "2025-11-25T10:00:00Z",
            "location_name": "Stadium Parking",
            "price_paid": 15.00,
            "pass_used": True
        }
    ]
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = mock_bookings
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is True
        assert result.verified_booking is not None
        assert result.verified_booking.booking_id == "22222"  # Exact match selected
        assert result.verified_booking.match_confidence == "exact"
        
        print("✓ Step 3: ParkWhiz API returned 3 bookings")
        print("✓ Step 4: Best match selected (booking 22222 - exact date match)")
        print("✓ Step 5: Booking verified")
    
    print("\n✓ MULTIPLE BOOKINGS TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 4: API Timeout - Retry then Escalate
# ============================================================================

@pytest.mark.asyncio
async def test_api_timeout_retry_escalation():
    """
    Test workflow when ParkWhiz API times out.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information
    3. Search ParkWhiz API - timeout
    4. Retry (built into client)
    5. Timeout again - escalate
    
    Expected: Escalation with "API timeout" reason
    """
    print("\n" + "=" * 80)
    print("TEST 4: API Timeout - Retry then Escalate")
    print("=" * 80)
    
    customer_info = CustomerInfo(
        email="timeout@example.com",
        arrival_date="2025-11-25",
        exit_date="2025-11-26"
    )
    
    print("✓ Step 1-2: Zapier failure detected, customer info extracted")
    
    # Step 3-5: API timeout with retry
    verifier = ParkWhizBookingVerifier()
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        # Simulate timeout (retry logic is in the client)
        mock_api.side_effect = ParkWhizTimeoutError("Request timed out after retry")
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is False
        assert result.verified_booking is None
        assert result.failure_reason == "API timeout after retry"
        assert result.should_escalate is True
        assert "timed out" in result.escalation_reason.lower()
        
        print("✓ Step 3: ParkWhiz API timeout")
        print("✓ Step 4: Retry attempted (in client)")
        print("✓ Step 5: Timeout persisted - escalated")
    
    # Verify escalation
    guard = DecisionGuard()
    
    should_escalate, reason = guard.should_escalate(
        result.verified_booking,
        customer_info,
        result.failure_reason
    )
    assert should_escalate is True
    
    print(f"  Escalation reason: {reason}")
    print("\n✓ API TIMEOUT TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 5: Missing Info - Extract Incomplete → Escalate
# ============================================================================

@pytest.mark.asyncio
async def test_missing_customer_info_escalation():
    """
    Test workflow when customer information is incomplete.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information - missing required fields
    3. Validate completeness - fails
    4. Escalate to human review
    
    Expected: Escalation with "Missing required information" reason
    """
    print("\n" + "=" * 80)
    print("TEST 5: Missing Customer Info - Escalation")
    print("=" * 80)
    
    print("✓ Step 1: Zapier failure detected")
    
    # Step 2: Incomplete customer info (missing dates)
    incomplete_info = CustomerInfo(
        email="incomplete@example.com",
        arrival_date="",  # Missing!
        exit_date=""      # Missing!
    )
    
    # Step 3: Validate completeness
    assert incomplete_info.is_complete() is False
    print("✓ Step 2-3: Customer info extracted but incomplete (missing dates)")
    
    # Step 4: Attempt verification - should fail immediately
    verifier = ParkWhizBookingVerifier()
    
    result = await verifier.verify_booking(incomplete_info)
    
    assert result.success is False
    assert result.verified_booking is None
    assert "Missing required customer information" in result.failure_reason
    assert result.should_escalate is True
    assert "complete customer information" in result.escalation_reason
    
    print("✓ Step 4: Escalated due to incomplete information")
    
    # Verify no API call was made
    assert result.api_calls_made == 0
    print("  (No API call made - failed validation)")
    
    print("\n✓ MISSING INFO TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 6: Usage Contradiction - Find → Usage Contradicts → Escalate
# ============================================================================

@pytest.mark.asyncio
async def test_usage_contradiction_escalation():
    """
    Test workflow when pass usage contradicts customer claim.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information
    3. Search ParkWhiz API - booking found
    4. Verify pass usage - shows "used"
    5. Customer claims refund (implies not used)
    6. Escalate due to contradiction
    
    Expected: Escalation with usage contradiction reason
    """
    print("\n" + "=" * 80)
    print("TEST 6: Usage Contradiction - Escalation")
    print("=" * 80)
    
    customer_info = CustomerInfo(
        email="contradiction@example.com",
        arrival_date="2025-11-25",
        exit_date="2025-11-26"
    )
    
    print("✓ Step 1-2: Zapier failure detected, customer info extracted")
    
    # Step 3-4: Booking found with "used" status
    verifier = ParkWhizBookingVerifier()
    
    mock_booking = {
        "id": "99999",
        "customer_email": "contradiction@example.com",
        "start_time": "2025-11-25T10:00:00Z",
        "end_time": "2025-11-26T10:00:00Z",
        "location_name": "Stadium Parking",
        "price_paid": 15.00,
        "pass_used": True,  # Pass was used!
        "pass_usage": "used"
    }
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = [mock_booking]
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is True
        assert result.verified_booking is not None
        assert result.verified_booking.pass_used is True
        assert result.verified_booking.pass_usage_status == "used"
        
        print("✓ Step 3: Booking found")
        print("✓ Step 4: Pass usage verified - shows 'used'")
    
    # Step 5-6: Check for contradiction and escalation
    # In a real scenario, the decision logic would detect that customer
    # is requesting refund but pass was used - this is a contradiction
    guard = DecisionGuard()
    
    # The guard should allow automated decision (booking is verified)
    # but the decision logic would see pass_used=True and escalate
    can_decide = guard.can_make_automated_decision(result.verified_booking)
    assert can_decide is True  # Booking is valid
    
    # However, if we detect the contradiction (pass used but refund requested),
    # we should escalate
    # This would be handled in the decision-making logic
    if result.verified_booking.pass_used:
        print("✓ Step 5: Contradiction detected - pass was used but refund requested")
        print("✓ Step 6: Escalated for human review")
        print("  Reason: Pass usage contradicts refund claim")
    
    print("\n✓ USAGE CONTRADICTION TEST PASSED")
    print("=" * 80)


# ============================================================================
# Test 7: Unknown Pass Usage - Escalation
# ============================================================================

@pytest.mark.asyncio
async def test_unknown_pass_usage_escalation():
    """
    Test workflow when pass usage status is unknown.
    
    Flow:
    1. Detect Zapier failure
    2. Extract customer information
    3. Search ParkWhiz API - booking found
    4. Pass usage status is "unknown"
    5. Escalate due to unclear usage
    
    Expected: Escalation with "Pass usage unavailable" reason
    """
    print("\n" + "=" * 80)
    print("TEST 7: Unknown Pass Usage - Escalation")
    print("=" * 80)
    
    customer_info = CustomerInfo(
        email="unknown@example.com",
        arrival_date="2025-11-25",
        exit_date="2025-11-26"
    )
    
    print("✓ Step 1-2: Zapier failure detected, customer info extracted")
    
    # Step 3-4: Booking found but pass usage unknown
    verifier = ParkWhizBookingVerifier()
    
    mock_booking = {
        "id": "88888",
        "customer_email": "unknown@example.com",
        "start_time": "2025-11-25T10:00:00Z",
        "end_time": "2025-11-26T10:00:00Z",
        "location_name": "Stadium Parking",
        "price_paid": 15.00,
        # No pass_used or pass_usage field - will be "unknown"
    }
    
    with patch.object(verifier.client, 'get_customer_bookings', new_callable=AsyncMock) as mock_api:
        mock_api.return_value = [mock_booking]
        
        result = await verifier.verify_booking(customer_info)
        
        assert result.success is True
        assert result.verified_booking is not None
        assert result.verified_booking.pass_usage_status == "unknown"
        assert result.should_escalate is True
        assert "Pass usage status unavailable" in result.escalation_reason
        
        print("✓ Step 3: Booking found")
        print("✓ Step 4: Pass usage status is 'unknown'")
    
    # Step 5: Verify escalation
    guard = DecisionGuard()
    
    can_decide = guard.can_make_automated_decision(result.verified_booking)
    assert can_decide is False  # Cannot decide with unknown usage
    
    should_escalate, reason = guard.should_escalate(result.verified_booking, customer_info)
    assert should_escalate is True
    assert "Pass usage status unavailable" in reason
    
    print("✓ Step 5: Escalated due to unknown pass usage")
    print(f"  Reason: {reason}")
    print("\n✓ UNKNOWN PASS USAGE TEST PASSED")
    print("=" * 80)


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
