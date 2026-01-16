#!/usr/bin/env python3
"""
Direct test of ticket processing using Parlant SDK.
This runs inside the container with access to the running server.
"""

import asyncio
import os
import sys

# Add the app_tools path
sys.path.insert(0, '/app')

from app_tools.tools.ticket_processing.ticket_automation_engine import process_ticket_end_to_end
from unittest.mock import Mock

async def test_ticket_processing():
    """Test ticket 1206331 processing with Gemini."""
    
    print("=" * 80)
    print("GEMINI INTEGRATION TEST - DIRECT TICKET PROCESSING")
    print("=" * 80)
    print()
    
    # Create mock context
    context = Mock()
    context.agent_id = "test_agent"
    context.customer_id = "test_customer"
    context.session_id = "test_session"
    context.inputs = {"ticket_id": "1206331"}
    
    ticket_id = "1206331"
    
    print(f"Processing ticket: {ticket_id}")
    print(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'not set')}")
    print(f"Gemini Model: {os.getenv('GEMINI_MODEL', 'not set')}")
    print()
    print("-" * 80)
    print("Starting workflow...")
    print("-" * 80)
    print()
    
    try:
        # Call the workflow tool
        result = await process_ticket_end_to_end(context, ticket_id)
        
        print("\n" + "=" * 80)
        print("WORKFLOW RESULTS")
        print("=" * 80)
        print()
        
        data = result.data
        
        # Print summary
        print(f"Ticket ID: {data.get('ticket_id')}")
        print(f"Decision: {data.get('decision')}")
        print(f"Reasoning: {data.get('reasoning')}")
        print()
        
        # Print workflow steps
        print("Steps Completed:")
        for step in data.get('steps_completed', []):
            print(f"  ✓ {step}")
        print()
        
        # Print verification checks
        print("Verification Checks:")
        print(f"  Security Status: {data.get('security_status')}")
        print(f"  Booking Info Found: {data.get('booking_info_found')}")
        print(f"  Note Added: {data.get('note_added')}")
        print(f"  Ticket Updated: {data.get('ticket_updated')}")
        print()
        
        # Print debug info
        if 'debug' in data:
            debug = data['debug']
            print("Debug Information:")
            print(f"  Notes Length: {debug.get('notes_length')} chars")
            print(f"  Booking Found: {debug.get('booking_found')}")
            print(f"  Security Flagged: {debug.get('security_flagged')}")
            print()
        
        # Determine success
        required_steps = [
            "Fetched ticket metadata",
            "Completed security scan",
            "Added analysis note to ticket",
            "Updated ticket tags"
        ]
        
        steps_completed = data.get('steps_completed', [])
        all_steps_present = all(
            any(req in step for step in steps_completed)
            for req in ["ticket metadata", "security scan", "note", "tags"]
        )
        
        print("=" * 80)
        if all_steps_present and data.get('note_added') and data.get('ticket_updated'):
            print("✓ TEST PASSED: All workflow steps completed successfully")
            print("=" * 80)
            return True
        else:
            print("✗ TEST FAILED: Some workflow steps missing")
            print("=" * 80)
            return False
            
    except Exception as e:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED: Exception occurred")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_ticket_processing())
    sys.exit(0 if result else 1)
