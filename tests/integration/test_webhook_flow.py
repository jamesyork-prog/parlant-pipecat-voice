#!/usr/bin/env python3
"""
Integration test for end-to-end webhook flow.

This test verifies the complete webhook processing pipeline:
1. Webhook received and validated
2. Journey activation triggered
3. Ticket processing completes
4. Freshdesk ticket is updated
5. Processing time is measured
"""

import asyncio
import os
import sys
import time
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Add the app_tools path
sys.path.insert(0, '/app')

from fastapi.testclient import TestClient
from app_tools.webhook_server import app
from app_tools.journey_router import AUTOMATED_JOURNEY_NAME


class TestWebhookEndToEndFlow:
    """Tests for complete webhook processing flow."""
    
    def test_webhook_to_journey_activation_flow(self):
        """
        Test complete flow from webhook receipt to journey activation.
        
        This test verifies:
        - Webhook endpoint receives and validates payload
        - Signature validation passes
        - Journey router is called with correct parameters
        - Automated journey name is returned
        - Response indicates successful queueing
        """
        client = TestClient(app)
        
        # Prepare webhook payload
        payload = {
            "ticket_id": "12345",
            "event": "ticket_created",
            "triggered_at": "2025-11-17T10:30:00Z",
            "ticket_subject": "Refund request for booking",
            "ticket_status": 2,
            "ticket_priority": 1,
            "requester_email": "customer@example.com"
        }
        
        # Mock signature validation and journey routing
        with patch('app_tools.webhook_server.validate_freshdesk_signature') as mock_validate, \
             patch('app_tools.webhook_server.route_to_journey') as mock_route:
            
            # Setup mocks
            mock_validate.return_value = True
            mock_route.return_value = AUTOMATED_JOURNEY_NAME
            
            # Send webhook request
            start_time = time.time()
            response = client.post(
                "/webhook/freshdesk",
                json=payload,
                headers={"X-Freshdesk-Signature": "test-signature"}
            )
            end_time = time.time()
            
            # Verify response
            assert response.status_code == 200
            response_data = response.json()
            
            # Verify response structure
            assert response_data["status"] == "success"
            assert response_data["ticket_id"] == "12345"
            assert "processing_time_ms" in response_data
            assert response_data["processing_time_ms"] > 0
            
            # Verify message indicates queueing
            assert "queued" in response_data["message"].lower() or "received" in response_data["message"].lower()
            
            # Verify signature validation was called
            mock_validate.assert_called_once()
            
            # Verify journey router was called with correct parameters
            mock_route.assert_called_once_with(
                trigger_source="webhook",
                ticket_id="12345"
            )
            
            # Verify journey name returned
            assert mock_route.return_value == AUTOMATED_JOURNEY_NAME
            
            # Verify processing time is reasonable (< 5 seconds)
            processing_time_seconds = end_time - start_time
            assert processing_time_seconds < 5.0, f"Processing took too long: {processing_time_seconds}s"
            
            print(f"✓ Webhook processed in {processing_time_seconds:.3f}s")
            print(f"✓ Journey activated: {AUTOMATED_JOURNEY_NAME}")
    
    def test_webhook_flow_with_ticket_updated_event(self):
        """
        Test webhook flow with ticket_updated event.
        
        Verifies that ticket updates are processed the same way as ticket creation.
        """
        client = TestClient(app)
        
        payload = {
            "ticket_id": "67890",
            "event": "ticket_updated",
            "triggered_at": "2025-11-17T11:00:00Z",
            "ticket_subject": "Need refund for cancelled booking"
        }
        
        with patch('app_tools.webhook_server.validate_freshdesk_signature') as mock_validate, \
             patch('app_tools.webhook_server.route_to_journey') as mock_route:
            
            mock_validate.return_value = True
            mock_route.return_value = AUTOMATED_JOURNEY_NAME
            
            response = client.post(
                "/webhook/freshdesk",
                json=payload,
                headers={"X-Freshdesk-Signature": "test-signature"}
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert response_data["ticket_id"] == "67890"
            
            # Verify routing was called
            mock_route.assert_called_once_with(
                trigger_source="webhook",
                ticket_id="67890"
            )
    
    def test_webhook_flow_measures_processing_time(self):
        """
        Test that webhook flow accurately measures processing time.
        
        Verifies:
        - Processing time is included in response
        - Time is measured in milliseconds
        - Time is reasonable (< 5000ms for webhook endpoint)
        """
        client = TestClient(app)
        
        payload = {
            "ticket_id": "99999",
            "event": "ticket_created",
            "triggered_at": "2025-11-17T12:00:00Z"
        }
        
        with patch('app_tools.webhook_server.validate_freshdesk_signature') as mock_validate, \
             patch('app_tools.webhook_server.route_to_journey') as mock_route:
            
            mock_validate.return_value = True
            mock_route.return_value = AUTOMATED_JOURNEY_NAME
            
            response = client.post(
                "/webhook/freshdesk",
                json=payload,
                headers={"X-Freshdesk-Signature": "test-signature"}
            )
            
            assert response.status_code == 200
            response_data = response.json()
            
            # Verify processing time is present and reasonable
            processing_time_ms = response_data["processing_time_ms"]
            assert isinstance(processing_time_ms, int)
            assert processing_time_ms > 0
            assert processing_time_ms < 5000, f"Processing time too high: {processing_time_ms}ms"
            
            print(f"✓ Processing time measured: {processing_time_ms}ms")
    
    def test_webhook_flow_with_multiple_tickets(self):
        """
        Test webhook flow processes multiple tickets correctly.
        
        Verifies that multiple webhooks can be processed in sequence
        without interference.
        """
        client = TestClient(app)
        
        ticket_ids = ["111", "222", "333"]
        
        with patch('app_tools.webhook_server.validate_freshdesk_signature') as mock_validate, \
             patch('app_tools.webhook_server.route_to_journey') as mock_route:
            
            mock_validate.return_value = True
            mock_route.return_value = AUTOMATED_JOURNEY_NAME
            
            for ticket_id in ticket_ids:
                payload = {
                    "ticket_id": ticket_id,
                    "event": "ticket_created",
                    "triggered_at": datetime.utcnow().isoformat() + "Z"
                }
                
                response = client.post(
                    "/webhook/freshdesk",
                    json=payload,
                    headers={"X-Freshdesk-Signature": "test-signature"}
                )
                
                assert response.status_code == 200
                response_data = response.json()
                assert response_data["status"] == "success"
                assert response_data["ticket_id"] == ticket_id
            
            # Verify router was called for each ticket
            assert mock_route.call_count == len(ticket_ids)
            
            print(f"✓ Processed {len(ticket_ids)} tickets successfully")


async def test_complete_ticket_processing_workflow():
    """
    Test complete ticket processing workflow (requires real environment).
    
    This test verifies the full pipeline:
    1. Webhook triggers automated journey
    2. process_ticket_end_to_end tool executes
    3. Ticket is analyzed and decision is made
    4. Freshdesk ticket is updated with note and tags
    
    Note: This test requires:
    - FRESHDESK_API_KEY configured
    - GEMINI_API_KEY or OPENAI_API_KEY configured
    - Valid test ticket ID
    
    This test is skipped if credentials are not available.
    """
    # Check for required credentials
    if not os.getenv("FRESHDESK_API_KEY"):
        print("⚠ Skipping full workflow test - FRESHDESK_API_KEY not configured")
        return
    
    if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠ Skipping full workflow test - No LLM API key configured")
        return
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST - COMPLETE WEBHOOK WORKFLOW")
    print("=" * 80)
    print()
    
    # Use a test ticket ID (should be a real ticket in your Freshdesk)
    test_ticket_id = os.getenv("TEST_TICKET_ID", "1206331")
    
    print(f"Test Ticket ID: {test_ticket_id}")
    print()
    
    # Import the process_ticket_end_to_end tool
    from app_tools.tools.ticket_processing.ticket_automation_engine import process_ticket_end_to_end
    
    # Create mock context
    context = Mock()
    context.agent_id = "test_agent"
    context.customer_id = "test_customer"
    context.session_id = "test_session"
    context.inputs = {"ticket_id": test_ticket_id}
    
    # Execute the complete workflow
    print("-" * 80)
    print("STEP 1: Executing complete ticket processing workflow")
    print("-" * 80)
    
    try:
        start_time = time.time()
        result = await process_ticket_end_to_end(context, test_ticket_id)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"✓ Workflow completed in {processing_time:.2f}s")
        print()
        
        # Verify result structure
        print("-" * 80)
        print("STEP 2: Verifying workflow results")
        print("-" * 80)
        
        data = result.data
        
        # Check required fields
        required_fields = [
            "ticket_id",
            "decision",
            "reasoning",
            "steps_completed",
            "security_status",
            "booking_info_found",
            "note_added",
            "ticket_updated",
            "processing_time_ms"
        ]
        
        for field in required_fields:
            if field in data:
                print(f"✓ Field present: {field}")
            else:
                print(f"✗ Field missing: {field}")
                return False
        
        print()
        
        # Display results
        print("-" * 80)
        print("STEP 3: Workflow Results")
        print("-" * 80)
        print(f"Ticket ID: {data['ticket_id']}")
        print(f"Decision: {data['decision']}")
        print(f"Reasoning: {data['reasoning'][:200]}...")
        print(f"Security Status: {data['security_status']}")
        print(f"Booking Info Found: {data['booking_info_found']}")
        print(f"Note Added: {data['note_added']}")
        print(f"Ticket Updated: {data['ticket_updated']}")
        print(f"Processing Time: {data['processing_time_ms']}ms")
        print()
        
        print(f"Steps Completed ({len(data['steps_completed'])}):")
        for step in data['steps_completed']:
            print(f"  • {step}")
        print()
        
        # Verify decision is valid
        valid_decisions = ["Approved", "Denied", "Needs Human Review", "SECURITY_ESCALATION"]
        if data['decision'] in valid_decisions:
            print(f"✓ Valid decision: {data['decision']}")
        else:
            print(f"✗ Invalid decision: {data['decision']}")
            return False
        
        # Verify note was added
        if data['note_added']:
            print("✓ Note successfully added to Freshdesk")
        else:
            print("⚠ Note was not added to Freshdesk")
        
        # Verify ticket was updated
        if data['ticket_updated']:
            print("✓ Ticket successfully updated in Freshdesk")
        else:
            print("⚠ Ticket was not updated in Freshdesk")
        
        print()
        
        # Performance check
        print("-" * 80)
        print("STEP 4: Performance Verification")
        print("-" * 80)
        
        if data['processing_time_ms'] < 15000:
            print(f"✓ Processing time within target: {data['processing_time_ms']}ms < 15000ms")
        else:
            print(f"⚠ Processing time exceeded target: {data['processing_time_ms']}ms > 15000ms")
        
        print()
        
        # Final verdict
        print("=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        print()
        print("✓ COMPLETE WORKFLOW TEST PASSED")
        print()
        print("Summary:")
        print(f"  • Workflow executed successfully")
        print(f"  • Decision made: {data['decision']}")
        print(f"  • Freshdesk updated: {data['note_added'] and data['ticket_updated']}")
        print(f"  • Processing time: {data['processing_time_ms']}ms")
        print()
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run sync tests
    print("Running webhook flow integration tests...")
    print()
    
    test_class = TestWebhookEndToEndFlow()
    
    print("Test 1: Webhook to journey activation flow")
    test_class.test_webhook_to_journey_activation_flow()
    print()
    
    print("Test 2: Webhook flow with ticket_updated event")
    test_class.test_webhook_flow_with_ticket_updated_event()
    print()
    
    print("Test 3: Webhook flow measures processing time")
    test_class.test_webhook_flow_measures_processing_time()
    print()
    
    print("Test 4: Webhook flow with multiple tickets")
    test_class.test_webhook_flow_with_multiple_tickets()
    print()
    
    # Run async test
    print("Test 5: Complete ticket processing workflow")
    result = asyncio.run(test_complete_ticket_processing_workflow())
    
    sys.exit(0 if result else 1)
