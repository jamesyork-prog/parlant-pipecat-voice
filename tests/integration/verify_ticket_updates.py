#!/usr/bin/env python3
"""
Verify that a ticket was actually updated in Freshdesk.
"""

import asyncio
import os
import sys
import httpx
import base64

async def verify_ticket_updates():
    """Verify ticket was updated with notes and tags."""
    
    print("=" * 80)
    print("VERIFYING FRESHDESK TICKET UPDATES")
    print("=" * 80)
    print()
    
    ticket_id = os.getenv("TEST_TICKET_ID", "1234567")
    domain = os.getenv("FRESHDESK_DOMAIN", "your-domain.freshdesk.com")
    api_key = os.getenv("FRESHDESK_API_KEY")
    
    if not api_key:
        print("✗ FRESHDESK_API_KEY not set")
        return False
    
    # Create auth header
    auth_string = f"{api_key}:X"
    auth_bytes = auth_string.encode('ascii')
    auth_b64 = base64.b64encode(auth_bytes).decode('ascii')
    
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Get ticket details
        print(f"Fetching ticket {ticket_id} from Freshdesk...")
        print()
        
        try:
            response = await client.get(
                f"https://{domain}/api/v2/tickets/{ticket_id}",
                headers=headers
            )
            response.raise_for_status()
            ticket = response.json()
            
            print("Ticket Information:")
            print(f"  ID: {ticket.get('id')}")
            print(f"  Subject: {ticket.get('subject')}")
            print(f"  Status: {ticket.get('status')}")
            print(f"  Tags: {ticket.get('tags', [])}")
            print()
            
            # Check for our tags
            tags = ticket.get('tags', [])
            expected_tags = ["needs_human_review", "automated_analysis"]
            
            tags_present = any(tag in tags for tag in expected_tags)
            
            if tags_present:
                print(f"✓ Automated tags found: {[t for t in tags if t in expected_tags]}")
            else:
                print(f"⚠ Expected tags not found. Current tags: {tags}")
            
        except Exception as e:
            print(f"✗ Failed to fetch ticket: {e}")
            return False
        
        # Get ticket conversations/notes
        print()
        print("Fetching ticket notes...")
        print()
        
        try:
            response = await client.get(
                f"https://{domain}/api/v2/tickets/{ticket_id}/conversations",
                headers=headers
            )
            response.raise_for_status()
            conversations = response.json()
            
            # Look for our automated note
            automated_note_found = False
            latest_note = None
            
            for conv in conversations:
                body = conv.get('body_text', '')
                if 'Automated Analysis Complete' in body or 'Whiz AI Agent' in body:
                    automated_note_found = True
                    latest_note = conv
                    break
            
            if automated_note_found and latest_note:
                print("✓ Automated analysis note found!")
                print()
                print("Note Preview:")
                body_text = latest_note.get('body_text', '')
                # Print first 500 chars
                print(body_text[:500])
                if len(body_text) > 500:
                    print("...")
                print()
                
                # Check for key elements in the note
                checks = {
                    "Decision mentioned": "Decision:" in body_text,
                    "Analysis included": "Analysis:" in body_text,
                    "Security status": "Security Status:" in body_text,
                    "Booking info status": "Booking Info:" in body_text,
                    "Agent signature": "Whiz AI Agent" in body_text
                }
                
                print("Note Content Verification:")
                all_present = True
                for check_name, present in checks.items():
                    status = "✓" if present else "✗"
                    print(f"  {status} {check_name}")
                    if not present:
                        all_present = False
                
                print()
                print("=" * 80)
                if tags_present and automated_note_found and all_present:
                    print("✓ VERIFICATION PASSED: Ticket properly updated")
                    print("=" * 80)
                    return True
                else:
                    print("⚠ VERIFICATION PARTIAL: Some elements missing")
                    print("=" * 80)
                    return False
            else:
                print("✗ Automated note not found in recent conversations")
                print()
                print(f"Total conversations: {len(conversations)}")
                if conversations:
                    print("Latest conversation preview:")
                    print(conversations[0].get('body_text', '')[:200])
                return False
                
        except Exception as e:
            print(f"✗ Failed to fetch conversations: {e}")
            return False

if __name__ == "__main__":
    result = asyncio.run(verify_ticket_updates())
    sys.exit(0 if result else 1)
