#!/usr/bin/env python3
"""
Integration tests for duplicate booking detection flow.

Tests the complete workflow from ticket detection to refund:
1. Ticket with "paid again" claim is detected
2. Duplicate detection tool is called
3. Decision is made based on detection results
4. Ticket is updated with decision note and tags
5. Refunded bookings are properly tagged

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import asyncio
import sys
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import pytest

# Add the app_tools path
sys.path.insert(0, '/app')

import parlant.sdk as p
from app_tools.tools.ticket_processing.ticket_automation_engine import process_ticket_end_to_end, _is_paid_again_claim


# ============================================================================
# Test Fixtures
# ============================================================================

def create_mock_context(ticket_id: str = "12345"):
    """Create a mock ToolContext for testing."""
    context = Mock(spec=p.ToolContext)
    context.agent_id = "test_agent"
    context.customer_id = "test_customer"
    context.session_id = "test_session"
    context.inputs = {"ticket_id": ticket_id}
    return context


def create_ticket_with_paid_again_claim():
    """Create a mock ticket with 'paid again' claim."""
    return {
        "id": "12345",
        "subject": "Refund request - charged twice",
        "status": 2,
        "priority": 1,
        "description": "I was charged twice for the same parking spot. Please refund one of the charges.",
        "description_text": "I was charged twice for the same parking spot. Please refund one of the charges.",
    }


def create_booking_info_with_email():
    """Create booking info with customer email."""
    return {
        "found": True,
        "booking_info": {
            "customer_email": "customer@example.com",
            "event_date": "2024-01-15",
            "location": "Downtown Garage",
            "booking_id": "ABC123",
        }
    }


# ============================================================================
# Paid Again Claim Detection Tests
# ============================================================================

def test_is_paid_again_claim_detection():
    """Test detection of 'paid again' claims in ticket text."""
    # Test various phrasings
    assert _is_paid_again_claim("I was charged twice for parking")
    assert _is_paid_again_claim("You charged me again for the same booking")
    assert _is_paid_again_claim("Double charged for parking")
    assert _is_paid_again_claim("I paid again by mistake")
    assert _is_paid_again_claim("Duplicate charge on my card")
    assert _is_paid_again_claim("I have two bookings for the same event")
    assert _is_paid_again_claim("Booked twice accidentally")
    
    # Test case insensitivity
    assert _is_paid_again_claim("CHARGED TWICE")
    assert _is_paid_again_claim("Paid Again")
    
    # Test negative cases
    assert not _is_paid_again_claim("I need a refund")
    assert not _is_paid_again_claim("Cancel my booking")
    assert not _is_paid_again_claim("")
    assert not _is_paid_again_claim(None)


# ============================================================================
# Full Workflow Tests - Duplicate Found and Refunded
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_duplicate_found_and_refunded():
    """
    Test complete workflow when duplicate is found and refunded.
    
    Verifies:
    - Ticket with "paid again" claim is detected
    - Duplicate detection is triggered
    - Duplicate is found and refunded
    - Decision is "Approved"
    - Ticket is updated with note and tags
    - "Refunded" tag is added
    """
    print("\n" + "=" * 80)
    print("TEST: Full Workflow - Duplicate Found and Refunded")
    print("=" * 80)
    
    context = create_mock_context("12345")
    
    with patch("app_tools.tools.process_ticket_workflow.get_ticket") as mock_get_ticket, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_description") as mock_get_desc, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_conversations") as mock_get_conv, \
         patch("app_tools.tools.process_ticket_workflow.check_content") as mock_security, \
         patch("app_tools.tools.process_ticket_workflow.extract_booking_info_from_note") as mock_extract, \
         patch("app_tools.tools.process_ticket_workflow.detect_duplicate_bookings") as mock_detect, \
         patch("app_tools.tools.freshdesk_tools.add_note") as mock_add_note, \
         patch("app_tools.tools.freshdesk_tools.update_ticket") as mock_update:
        
        # Setup mocks
        ticket_data = create_ticket_with_paid_again_claim()
        mock_get_ticket.return_value = AsyncMock(data=ticket_data)
        mock_get_desc.return_value = AsyncMock(data={"description_text": ticket_data["description"]})
        mock_get_conv.return_value = AsyncMock(data={"conversations": []})
        mock_security.return_value = AsyncMock(data={"safe": True, "flagged": False})
        mock_extract.return_value = AsyncMock(data=create_booking_info_with_email())
        
        # Mock duplicate detection - found and refunded
        mock_detect.return_value = AsyncMock(data={
            "has_duplicates": True,
            "action_taken": "refunded",
            "refunded_booking_id": "12346",
            "kept_booking_id": "12345",
            "refund_amount": 15.00,
            "explanation": "Found 2 duplicate bookings. Refunded unused booking 12346.",
            "all_booking_ids": ["12345", "12346"]
        })
        
        mock_add_note.return_value = AsyncMock(data={"success": True})
        mock_update.return_value = AsyncMock(data={"success": True})
        
        # Execute workflow
        result = await process_ticket_end_to_end(context, "12345")
        
        # Verify result
        assert result.data["decision"] == "Approved"
        assert result.data["refunded"] is True
        assert result.data["duplicate_detection_used"] is True
        assert "Duplicate booking detected and refunded" in result.data["reasoning"]
        
        # Verify duplicate detection was called
        mock_detect.assert_called_once()
        call_args = mock_detect.call_args
        assert call_args.kwargs["customer_email"] == "customer@example.com"
        assert call_args.kwargs["event_date"] == "2024-01-15"
        
        # Verify note was added
        mock_add_note.assert_called_once()
        note_text = mock_add_note.call_args.args[1]
        assert "Approved" in note_text
        assert "Refunded: Yes" in note_text
        
        # Verify ticket was updated with correct tags
        mock_update.assert_called_once()
        tags = mock_update.call_args.kwargs["tags"]
        assert "Processed by Whiz Agent" in tags
        assert "Refunded" in tags
        assert "refund_approved" in tags
        
        print("✓ Decision: Approved")
        print("✓ Refunded: Yes")
        print("✓ Duplicate detection used: Yes")
        print("✓ Note added with refund details")
        print("✓ Tags updated: Processed by Whiz Agent, Refunded, refund_approved")
        print("=" * 80)
        
        return True


# ============================================================================
# Full Workflow Tests - No Duplicates Found
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_no_duplicates_found():
    """
    Test complete workflow when no duplicates are found.
    
    Verifies:
    - Ticket with "paid again" claim is detected
    - Duplicate detection is triggered
    - No duplicates found
    - Decision is "Denied"
    - Ticket is updated with note and tags
    - No "Refunded" tag is added
    """
    print("\n" + "=" * 80)
    print("TEST: Full Workflow - No Duplicates Found")
    print("=" * 80)
    
    context = create_mock_context("12346")
    
    with patch("app_tools.tools.process_ticket_workflow.get_ticket") as mock_get_ticket, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_description") as mock_get_desc, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_conversations") as mock_get_conv, \
         patch("app_tools.tools.process_ticket_workflow.check_content") as mock_security, \
         patch("app_tools.tools.process_ticket_workflow.extract_booking_info_from_note") as mock_extract, \
         patch("app_tools.tools.process_ticket_workflow.detect_duplicate_bookings") as mock_detect, \
         patch("app_tools.tools.freshdesk_tools.add_note") as mock_add_note, \
         patch("app_tools.tools.freshdesk_tools.update_ticket") as mock_update:
        
        # Setup mocks
        ticket_data = create_ticket_with_paid_again_claim()
        mock_get_ticket.return_value = AsyncMock(data=ticket_data)
        mock_get_desc.return_value = AsyncMock(data={"description_text": ticket_data["description"]})
        mock_get_conv.return_value = AsyncMock(data={"conversations": []})
        mock_security.return_value = AsyncMock(data={"safe": True, "flagged": False})
        mock_extract.return_value = AsyncMock(data=create_booking_info_with_email())
        
        # Mock duplicate detection - no duplicates
        mock_detect.return_value = AsyncMock(data={
            "has_duplicates": False,
            "action_taken": "deny",
            "duplicate_count": 1,
            "explanation": "Found only 1 booking. No duplicates detected.",
            "all_booking_ids": ["12345"]
        })
        
        mock_add_note.return_value = AsyncMock(data={"success": True})
        mock_update.return_value = AsyncMock(data={"success": True})
        
        # Execute workflow
        result = await process_ticket_end_to_end(context, "12346")
        
        # Verify result
        assert result.data["decision"] == "Denied"
        assert result.data["refunded"] is False
        assert result.data["duplicate_detection_used"] is True
        assert "No duplicate bookings found" in result.data["reasoning"]
        
        # Verify note was added
        mock_add_note.assert_called_once()
        note_text = mock_add_note.call_args.args[1]
        assert "Denied" in note_text
        assert "Refunded: No" in note_text
        
        # Verify ticket was updated with correct tags (no "Refunded" tag)
        mock_update.assert_called_once()
        tags = mock_update.call_args.kwargs["tags"]
        assert "Processed by Whiz Agent" in tags
        assert "Refunded" not in tags
        assert "refund_denied" in tags
        
        print("✓ Decision: Denied")
        print("✓ Refunded: No")
        print("✓ Duplicate detection used: Yes")
        print("✓ Note added with denial reason")
        print("✓ Tags updated: Processed by Whiz Agent, refund_denied (no Refunded tag)")
        print("=" * 80)
        
        return True


# ============================================================================
# Full Workflow Tests - 3+ Duplicates (Escalate)
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_three_plus_duplicates_escalate():
    """
    Test complete workflow when 3+ duplicates are found.
    
    Verifies:
    - Ticket with "paid again" claim is detected
    - Duplicate detection is triggered
    - 3+ duplicates found
    - Decision is "Needs Human Review"
    - Ticket is updated with note and tags
    - No "Refunded" tag is added
    """
    print("\n" + "=" * 80)
    print("TEST: Full Workflow - 3+ Duplicates (Escalate)")
    print("=" * 80)
    
    context = create_mock_context("12347")
    
    with patch("app_tools.tools.process_ticket_workflow.get_ticket") as mock_get_ticket, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_description") as mock_get_desc, \
         patch("app_tools.tools.process_ticket_workflow.get_ticket_conversations") as mock_get_conv, \
         patch("app_tools.tools.process_ticket_workflow.check_content") as mock_security, \
         patch("app_tools.tools.process_ticket_workflow.extract_booking_info_from_note") as mock_extract, \
         patch("app_tools.tools.process_ticket_workflow.detect_duplicate_bookings") as mock_detect, \
         patch("app_tools.tools.freshdesk_tools.add_note") as mock_add_note, \
         patch("app_tools.tools.freshdesk_tools.update_ticket") as mock_update:
        
        # Setup mocks
        ticket_data = create_ticket_with_paid_again_claim()
        mock_get_ticket.return_value = AsyncMock(data=ticket_data)
        mock_get_desc.return_value = AsyncMock(data={"description_text": ticket_data["description"]})
        mock_get_conv.return_value = AsyncMock(data={"conversations": []})
        mock_security.return_value = AsyncMock(data={"safe": True, "flagged": False})
        mock_extract.return_value = AsyncMock(data=create_booking_info_with_email())
        
        # Mock duplicate detection - 3+ duplicates
        mock_detect.return_value = AsyncMock(data={
            "has_duplicates": True,
            "action_taken": "escalate",
            "duplicate_count": 3,
            "explanation": "Found 3 duplicate bookings. Too complex for automated refund.",
            "all_booking_ids": ["12345", "12346", "12347"]
        })
        
        mock_add_note.return_value = AsyncMock(data={"success": True})
        mock_update.return_value = AsyncMock(data={"success": True})
        
        # Execute workflow
        result = await process_ticket_end_to_end(context, "12347")
        
        # Verify result
        assert result.data["decision"] == "Needs Human Review"
        assert result.data["refunded"] is False
        assert result.data["duplicate_detection_used"] is True
        assert "requires human review" in result.data["reasoning"]
        
        # Verify note was added
        mock_add_note.assert_called_once()
        note_text = mock_add_note.call_args.args[1]
        assert "Needs Human Review" in note_text
        assert "Refunded: No" in note_text
        
        # Verify ticket was updated with correct tags
        mock_update.assert_called_once()
        tags = mock_update.call_args.kwargs["tags"]
        assert "Processed by Whiz Agent" in tags
        assert "Refunded" not in tags
        assert "needs_human_review" in tags
        
        print("✓ Decision: Needs Human Review")
        print("✓ Refunded: No")
        print("✓ Duplicate detection used: Yes")
        print("✓ Note added with escalation reason")
        print("✓ Tags updated: Processed by Whiz Agent, needs_human_review")
        print("=" * 80)
        
        return True


if __name__ == "__main__":
    # Run with pytest
    import subprocess
    result = subprocess.run(
        ["pytest", __file__, "-v"],
        capture_output=False
    )
    sys.exit(result.returncode)
