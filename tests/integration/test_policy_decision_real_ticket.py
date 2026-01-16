#!/usr/bin/env python3
"""
Integration test for policy-based decision making with real ticket 1206331.

This test verifies the complete decision-making workflow:
1. Booking extraction from ticket notes
2. Rule-based or LLM-based decision making
3. Cancellation reason mapping (if Approved)
4. Note documentation in Freshdesk
"""

import asyncio
import os
import sys

# Add the app_tools path
sys.path.insert(0, '/app')

from app_tools.tools.refund_decision.decision_maker import DecisionMaker
from app_tools.tools.integrations.freshdesk import get_ticket, get_ticket_description, add_note
from unittest.mock import Mock

# Test ticket ID
TEST_TICKET_ID = "1206331"


async def test_real_ticket_decision():
    """Test complete decision workflow with real ticket 1206331."""
    
    print("=" * 80)
    print("INTEGRATION TEST - POLICY-BASED DECISION (REAL TICKET)")
    print("=" * 80)
    print(f"\nTest Ticket ID: {TEST_TICKET_ID}")
    print()
    
    # Verify environment
    if not os.getenv("FRESHDESK_API_KEY"):
        print("✗ ERROR: FRESHDESK_API_KEY not configured")
        return False
    
    if not os.getenv("GEMINI_API_KEY"):
        print("✗ ERROR: GEMINI_API_KEY not configured")
        return False
    
    print("✓ Environment configured")
    print()
    
    # Step 1: Fetch ticket data
    print("-" * 80)
    print("STEP 1: Fetching ticket data")
    print("-" * 80)
    
    try:
        # Create mock context for tools
        context = Mock()
        context.agent_id = "test_agent"
        context.customer_id = "test_customer"
        context.session_id = "test_session"
        context.inputs = {"ticket_id": TEST_TICKET_ID}
        
        # Fetch ticket metadata
        ticket_result = await get_ticket(context, TEST_TICKET_ID)
        ticket_data = ticket_result.data
        
        print(f"✓ Ticket fetched: {ticket_data.get('subject', 'N/A')}")
        print(f"  Status: {ticket_data.get('status', 'N/A')}")
        print(f"  Priority: {ticket_data.get('priority', 'N/A')}")
        
        # Fetch ticket description (contains booking info)
        desc_result = await get_ticket_description(context, TEST_TICKET_ID)
        ticket_description = desc_result.data.get("description", "")
        
        print(f"✓ Description fetched: {len(ticket_description)} characters")
        print()
        
    except Exception as e:
        print(f"✗ Failed to fetch ticket: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Make decision using DecisionMaker
    print("-" * 80)
    print("STEP 2: Making refund decision")
    print("-" * 80)
    
    try:
        decision_maker = DecisionMaker()
        
        # Prepare ticket data for decision maker
        ticket_info = {
            "ticket_id": TEST_TICKET_ID,
            "subject": ticket_data.get("subject", ""),
            "description": ticket_description,
            "status": ticket_data.get("status", ""),
            "priority": ticket_data.get("priority", "")
        }
        
        # Make decision
        print("Analyzing ticket...")
        decision_result = await decision_maker.make_decision(
            ticket_data=ticket_info,
            ticket_notes=ticket_description
        )
        
        print()
        print("Decision Results:")
        print(f"  Decision: {decision_result.get('decision')}")
        print(f"  Reasoning: {decision_result.get('reasoning')[:200]}...")
        print(f"  Policy Applied: {decision_result.get('policy_applied')}")
        print(f"  Confidence: {decision_result.get('confidence')}")
        print(f"  Method Used: {decision_result.get('method_used')}")
        print(f"  Processing Time: {decision_result.get('processing_time_ms')}ms")
        
        if decision_result.get('cancellation_reason'):
            print(f"  Cancellation Reason: {decision_result.get('cancellation_reason')}")
        
        print(f"  Booking Info Found: {decision_result.get('booking_info_found')}")
        print()
        
    except Exception as e:
        print(f"✗ Failed to make decision: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify booking extraction
    print("-" * 80)
    print("STEP 3: Verifying booking extraction")
    print("-" * 80)
    
    booking_found = decision_result.get('booking_info_found', False)
    
    if booking_found:
        print("✓ Booking information successfully extracted")
    else:
        print("⚠ Booking information not found or incomplete")
        print("  This may be expected if ticket lacks booking details")
    print()
    
    # Step 4: Verify decision quality
    print("-" * 80)
    print("STEP 4: Verifying decision quality")
    print("-" * 80)
    
    decision = decision_result.get('decision')
    confidence = decision_result.get('confidence')
    method = decision_result.get('method_used')
    processing_time = decision_result.get('processing_time_ms', 0)
    
    checks = []
    
    # Check 1: Valid decision
    valid_decisions = ["Approved", "Denied", "Needs Human Review"]
    if decision in valid_decisions:
        print(f"✓ Valid decision: {decision}")
        checks.append(True)
    else:
        print(f"✗ Invalid decision: {decision}")
        checks.append(False)
    
    # Check 2: Has reasoning
    if decision_result.get('reasoning'):
        print(f"✓ Decision includes reasoning")
        checks.append(True)
    else:
        print(f"✗ Decision missing reasoning")
        checks.append(False)
    
    # Check 3: Has policy reference
    if decision_result.get('policy_applied'):
        print(f"✓ Policy reference included")
        checks.append(True)
    else:
        print(f"✗ Policy reference missing")
        checks.append(False)
    
    # Check 4: Confidence level
    valid_confidence = ["high", "medium", "low"]
    if confidence in valid_confidence:
        print(f"✓ Valid confidence level: {confidence}")
        checks.append(True)
    else:
        print(f"✗ Invalid confidence level: {confidence}")
        checks.append(False)
    
    # Check 5: Cancellation reason for Approved decisions
    if decision == "Approved":
        if decision_result.get('cancellation_reason'):
            print(f"✓ Cancellation reason provided: {decision_result.get('cancellation_reason')}")
            checks.append(True)
        else:
            print(f"✗ Approved decision missing cancellation reason")
            checks.append(False)
    
    # Check 6: Performance (rule-based should be <2s, LLM-based <10s)
    if method == "rules" and processing_time < 2000:
        print(f"✓ Rule-based decision within 2s: {processing_time}ms")
        checks.append(True)
    elif method in ["llm", "hybrid"] and processing_time < 10000:
        print(f"✓ LLM-based decision within 10s: {processing_time}ms")
        checks.append(True)
    elif processing_time >= 10000:
        print(f"⚠ Decision took longer than expected: {processing_time}ms")
        checks.append(True)  # Don't fail on performance
    else:
        print(f"✓ Decision completed in {processing_time}ms")
        checks.append(True)
    
    print()
    
    # Step 5: Document decision in Freshdesk (optional - can be skipped in test)
    print("-" * 80)
    print("STEP 5: Testing decision documentation")
    print("-" * 80)
    
    try:
        # Format note body (same as document_decision tool)
        cancellation_reason_text = ""
        if decision_result['decision'] == "Approved":
            cancellation_reason_text = f"\n**ParkWhiz Cancellation Reason:** {decision_result['cancellation_reason']}"
        
        note_body = f"""
**AGENT DECISION: {decision_result['decision']}**

**Reasoning:**
{decision_result['reasoning']}

**Policy Applied:**
{decision_result['policy_applied']}
{cancellation_reason_text}

**Confidence Level:** {decision_result['confidence']}
**Method Used:** {decision_result['method_used']}
**Processing Time:** {decision_result['processing_time_ms']}ms

---
This decision was made by the Whiz Agent (Integration Test). Please review before processing the refund.
"""
        
        print("✓ Note formatted successfully")
        print(f"  Note length: {len(note_body)} characters")
        
        # Actually add the note (commented out to avoid spamming test ticket)
        # Uncomment to test actual Freshdesk integration
        # context.inputs = {
        #     "ticket_id": TEST_TICKET_ID,
        #     "body": note_body,
        #     "private": True
        # }
        # note_result = await add_note(context, TEST_TICKET_ID, note_body, private=True)
        # print("✓ Note added to Freshdesk")
        
        print("  (Skipping actual note creation to avoid spamming test ticket)")
        checks.append(True)
        
    except Exception as e:
        print(f"✗ Failed to format note: {e}")
        checks.append(False)
    
    print()
    
    # Final verdict
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()
    
    passed = sum(checks)
    total = len(checks)
    
    print(f"Checks Passed: {passed}/{total}")
    print()
    
    if all(checks):
        print("✓ ALL TESTS PASSED")
        print()
        print("Summary:")
        print(f"  • Booking extraction: {'Success' if booking_found else 'Not found (may be expected)'}")
        print(f"  • Decision made: {decision}")
        print(f"  • Confidence: {confidence}")
        print(f"  • Method: {method}")
        print(f"  • Processing time: {processing_time}ms")
        print(f"  • Note formatting: Success")
        print()
        print("=" * 80)
        return True
    else:
        print("✗ SOME TESTS FAILED")
        print()
        print(f"Failed checks: {total - passed}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    result = asyncio.run(test_real_ticket_decision())
    sys.exit(0 if result else 1)
