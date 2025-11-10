# main.py

import asyncio, os, pathlib
import parlant.sdk as p
import psycopg2
from typing import Annotated

# --- Globals ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_ID_FILE = pathlib.Path("/app/data/agent_id.txt")

# Import all tools
from app_tools.tools.freshdesk_tools import get_ticket, add_note, update_ticket
from app_tools.tools.database_logger import log_audit_trail, log_run_metrics, update_customer_context
from app_tools.tools.lakera_security_tool import check_content
from app_tools.tools.journey_helpers import extract_booking_info_from_note, triage_ticket
from app_tools.tools.parkwhiz_tools import get_customer_orders
import uuid
from datetime import datetime


# --- Journeys ---

async def create_ticket_ingestion_journey(agent: p.Agent):
    """
    Freshdesk ticket ingestion journey for automated refund processing.
    Follows Parlant pattern: tool states must be followed by chat states.
    """
    journey = await agent.create_journey(
        title="Freshdesk Ticket Ingestion",
        conditions=["A new Freshdesk ticket needs processing"],
        description="Automates refund request processing with security and policy checking.",
    )
    
    # Fetch ticket
    t1 = await journey.initial_state.transition_to(
        chat_state="Fetching ticket from Freshdesk...",
    )
    t2 = await t1.target.transition_to(tool_state=get_ticket)
    
    # Security scan
    t3 = await t2.target.transition_to(
        chat_state="Running security scan on ticket content...",
    )
    t4 = await t3.target.transition_to(tool_state=check_content)
    
    # Security decision fork
    t5_safe = await t4.target.transition_to(
        chat_state="Content safe. Extracting booking info...",
        condition="Content is safe",
    )
    t5_unsafe = await t4.target.transition_to(
        chat_state="Security threat! Escalating to security team.",
        condition="Content flagged as unsafe",
    )
    
    # Extract booking info
    t6 = await t5_safe.target.transition_to(tool_state=extract_booking_info_from_note)
    
    # Check if found
    t7_found = await t6.target.transition_to(
        chat_state="Booking info found. Analyzing refund eligibility...",
        condition="Booking info extracted",
    )
    t7_missing = await t6.target.transition_to(
        chat_state="No booking info in notes. Querying ParkWhiz API...",
        condition="No booking info found",
    )
    
    # ParkWhiz fallback
    t8 = await t7_missing.target.transition_to(tool_state=get_customer_orders)
    t9 = await t8.target.transition_to(
        chat_state="Data retrieved. Analyzing refund eligibility...",
    )
    
    # Triage (from both paths)
    t10_notes = await t7_found.target.transition_to(tool_state=triage_ticket)
    t10_api = await t9.target.transition_to(tool_state=triage_ticket)
    
    # APPROVED path from notes
    t11_app_n = await t10_notes.target.transition_to(
        chat_state="Refund approved! Adding note...",
        condition="Decision is Approved",
    )
    t12_app_n = await t11_app_n.target.transition_to(tool_state=add_note)
    t13_app_n = await t12_app_n.target.transition_to(
        chat_state="Closing ticket...",
    )
    await t13_app_n.target.transition_to(tool_state=update_ticket)
    
    # APPROVED path from API
    t11_app_a = await t10_api.target.transition_to(
        chat_state="Refund approved! Adding note...",
        condition="Decision is Approved",
    )
    t12_app_a = await t11_app_a.target.transition_to(tool_state=add_note)
    t13_app_a = await t12_app_a.target.transition_to(
        chat_state="Closing ticket...",
    )
    await t13_app_a.target.transition_to(tool_state=update_ticket)
    
    # DENIED path from notes
    t11_den_n = await t10_notes.target.transition_to(
        chat_state="Refund denied. Adding policy explanation...",
        condition="Decision is Denied",
    )
    t12_den_n = await t11_den_n.target.transition_to(tool_state=add_note)
    t13_den_n = await t12_den_n.target.transition_to(
        chat_state="Updating ticket...",
    )
    await t13_den_n.target.transition_to(tool_state=update_ticket)
    
    # DENIED path from API
    t11_den_a = await t10_api.target.transition_to(
        chat_state="Refund denied. Adding policy explanation...",
        condition="Decision is Denied",
    )
    t12_den_a = await t11_den_a.target.transition_to(tool_state=add_note)
    t13_den_a = await t12_den_a.target.transition_to(
        chat_state="Updating ticket...",
    )
    await t13_den_a.target.transition_to(tool_state=update_ticket)
    
    # ESCALATE path from notes
    t11_esc_n = await t10_notes.target.transition_to(
        chat_state="Needs human review. Escalating...",
        condition="Decision is Needs Human Review",
    )
    t12_esc_n = await t11_esc_n.target.transition_to(tool_state=add_note)
    t13_esc_n = await t12_esc_n.target.transition_to(
        chat_state="Assigning to human...",
    )
    await t13_esc_n.target.transition_to(tool_state=update_ticket)
    
    # ESCALATE path from API
    t11_esc_a = await t10_api.target.transition_to(
        chat_state="Needs human review. Escalating...",
        condition="Decision is Needs Human Review",
    )
    t12_esc_a = await t11_esc_a.target.transition_to(tool_state=add_note)
    t13_esc_a = await t12_esc_a.target.transition_to(
        chat_state="Assigning to human...",
    )
    await t13_esc_a.target.transition_to(tool_state=update_ticket)
    
    # ESCALATE from security
    t11_sec = await t5_unsafe.target.transition_to(tool_state=add_note)
    t12_sec = await t11_sec.target.transition_to(
        chat_state="Assigning to security team...",
    )
    await t12_sec.target.transition_to(tool_state=update_ticket)
    
    print(f"✓ Created journey: {journey.title}")


# --- Main Application ---
async def main():
    async with p.Server() as server:
        # Ensure the data directory exists
        AGENT_ID_FILE.parent.mkdir(exist_ok=True)

        agent = await server.create_agent(
            name="Whiz",
            description="An agent that processes incoming tickets from Freshdesk.",
        )

        agent_id = agent.id
        # Persist the id so other processes can read it (voice bridge)
        AGENT_ID_FILE.write_text(agent_id, encoding="utf-8")

        print("\n=== Parlant server is running ===")
        print("UI:   http://127.0.0.1:8800")
        print(f"AGENT_ID: {agent_id}")
        print(f"(Also saved to: {AGENT_ID_FILE})\n")

        # --- Register Journeys and Guidelines here ---
        await create_ticket_ingestion_journey(agent)

        # Add general guidelines
        await agent.create_guideline(
            condition="the customer greets you",
            action="Greet back warmly and ask how you can help today.",
        )

    asyncio.run(main())
