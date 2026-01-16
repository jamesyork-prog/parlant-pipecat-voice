# main.py

import asyncio, os, pathlib
import parlant.sdk as p
import psycopg2
from typing import Annotated
import logging

# --- Globals ---
ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT_ID_FILE = pathlib.Path("/app/data/agent_id.txt")

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Reduce Parlant SDK logging verbosity to avoid duplicate logs
parlant_logger = logging.getLogger('parlant')
parlant_logger.setLevel(logging.WARNING)
parlant_logger.propagate = False  # Prevent duplicate logs

logger = logging.getLogger(__name__)

# Import all tools
from app_tools.tools.integrations.freshdesk import (
    get_ticket,
    get_ticket_description,
    get_ticket_conversations,
    add_note,
    update_ticket,
)
from app_tools.tools.infrastructure.database import log_audit_trail, log_run_metrics, update_customer_context
from app_tools.tools.integrations.lakera import check_content
from app_tools.tools.ticket_processing.helpers import extract_booking_info_from_note, triage_ticket
from app_tools.tools.ticket_processing.manual_trigger import trigger_ticket_processing

from app_tools.tools.integrations.parkwhiz import validate_oauth2_credentials
import uuid
from datetime import datetime


# --- Configuration ---

def validate_llm_config():
    """
    Validates LLM provider configuration and credentials.
    
    Returns:
        str: The configured LLM provider ('gemini' or 'openai')
        
    Raises:
        ValueError: If provider is unknown or required API key is missing
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    logger.info(f"Validating LLM configuration for provider: {provider}")
    
    if provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not configured")
            raise ValueError(
                "GEMINI_API_KEY environment variable is required when LLM_PROVIDER is 'gemini'. "
                "Please add GEMINI_API_KEY to your .env file."
            )
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        logger.info(f"✓ Gemini configuration valid - Model: {model}")
        
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not configured")
            raise ValueError(
                "OPENAI_API_KEY environment variable is required when LLM_PROVIDER is 'openai'. "
                "Please add OPENAI_API_KEY to your .env file."
            )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        logger.info(f"✓ OpenAI configuration valid - Model: {model}")
        
    else:
        logger.error(f"Unknown LLM_PROVIDER: {provider}")
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Valid options are 'gemini' or 'openai'."
        )
    
    return provider


# --- Journeys ---

async def create_interactive_processing_journey(agent: p.Agent):
    """
    Interactive ticket processing journey for chat-based refund processing.
    Follows Parlant pattern: tool states must be followed by chat states.
    Provides step-by-step feedback to users during processing.
    """
    journey = await agent.create_journey(
        title="Interactive Ticket Processing",
        conditions=[
            "The user asks in chat to process a Freshdesk ticket",
            "The user types a ticket number to process interactively",
            "The user wants to analyze a ticket for refund eligibility in chat"
        ],
        description="Interactively processes Freshdesk tickets through the complete refund workflow with step-by-step chat feedback: fetching ticket data, security scanning, extracting booking info, making triage decisions, and updating tickets. This journey provides real-time updates to the user.",
    )
    
    # Fetch ticket metadata first
    # t1 = await journey.initial_state.transition_to(tool_state=get_ticket)
    t1 = await journey.initial_state.transition_to(tool_state=get_ticket)
    t2 = await t1.target.transition_to(
        chat_state="Ticket metadata retrieved. Fetching full description and continuing analysis.",
    )
    
    # Fetch ticket description
    t3 = await t2.target.transition_to(tool_state=get_ticket_description)
    t4 = await t3.target.transition_to(
        chat_state="Description retrieved. Fetching conversation history and continuing analysis.",
    )
    
    # Fetch conversations/notes
    t5 = await t4.target.transition_to(tool_state=get_ticket_conversations)
    t6 = await t5.target.transition_to(
        chat_state="All ticket data gathered. Running security scan and continuing analysis.",
    )
    
    # Security scan
    t7 = await t6.target.transition_to(tool_state=check_content)
    
    # Security decision fork
    t8_safe = await t7.target.transition_to(
        chat_state="Security scan passed. Extracting booking information and continuing analysis.",
        condition="Content is safe",
    )
    t8_unsafe = await t7.target.transition_to(
        chat_state="Security threat detected! Escalating to security team.",
        condition="Content flagged as unsafe",
    )
    
    # Extract booking info
    t9 = await t8_safe.target.transition_to(tool_state=extract_booking_info_from_note)
    
    # Check if booking info found
    t10_found = await t9.target.transition_to(
        chat_state="Booking information found. Analyzing refund eligibility and continuing.",
        condition="Booking info extracted",
    )
    t10_missing = await t9.target.transition_to(
        chat_state="No booking info in notes. Querying ParkWhiz API and continuing.",
        condition="No booking info found",
    )
    
    # Triage from both paths - but merge the results into a single decision flow
    t11_notes = await t10_found.target.transition_to(tool_state=triage_ticket)
    t11_api = await t10_missing.target.transition_to(tool_state=triage_ticket)
    
    # Create a single decision handler that both triage paths feed into
    # This eliminates the duplicate add_note calls by having only one decision flow
    
    # APPROVED decision (single path for both sources)
    t14_approved_notes = await t11_notes.target.transition_to(
        chat_state="Refund approved! Adding approval note and closing ticket.",
        condition="Decision is Approved",
    )
    t14_approved_api = await t11_api.target.transition_to(
        chat_state="Refund approved! Adding approval note and closing ticket.",
        condition="Decision is Approved",
    )
    
    # Single add_note call (no duplication)
    t15_approved_notes = await t14_approved_notes.target.transition_to(tool_state=add_note)
    t15_approved_api = await t14_approved_api.target.transition_to(tool_state=add_note)
    
    t16_approved_notes = await t15_approved_notes.target.transition_to(
        chat_state="Note added. Closing ticket and completing process.",
    )
    t16_approved_api = await t15_approved_api.target.transition_to(
        chat_state="Note added. Closing ticket and completing process.",
    )
    
    t17_approved_notes = await t16_approved_notes.target.transition_to(tool_state=update_ticket)
    t17_approved_api = await t16_approved_api.target.transition_to(tool_state=update_ticket)
    
    await t17_approved_notes.target.transition_to(
        chat_state="Process complete. Ticket has been updated and closed."
    )
    await t17_approved_api.target.transition_to(
        chat_state="Process complete. Ticket has been updated and closed."
    )
    
    # DENIED decision (single path for both sources)
    t14_denied_notes = await t11_notes.target.transition_to(
        chat_state="Refund denied per policy. Adding explanation note and updating ticket.",
        condition="Decision is Denied",
    )
    t14_denied_api = await t11_api.target.transition_to(
        chat_state="Refund denied per policy. Adding explanation note and updating ticket.",
        condition="Decision is Denied",
    )
    
    t15_denied_notes = await t14_denied_notes.target.transition_to(tool_state=add_note)
    t15_denied_api = await t14_denied_api.target.transition_to(tool_state=add_note)
    
    t16_denied_notes = await t15_denied_notes.target.transition_to(
        chat_state="Note added. Updating ticket status and completing process.",
    )
    t16_denied_api = await t15_denied_api.target.transition_to(
        chat_state="Note added. Updating ticket status and completing process.",
    )
    
    t17_denied_notes = await t16_denied_notes.target.transition_to(tool_state=update_ticket)
    t17_denied_api = await t16_denied_api.target.transition_to(tool_state=update_ticket)
    
    await t17_denied_notes.target.transition_to(
        chat_state="Process complete. Ticket has been updated with denial."
    )
    await t17_denied_api.target.transition_to(
        chat_state="Process complete. Ticket has been updated with denial."
    )
    
    # ESCALATE decision (single path for both sources)
    t14_escalate_notes = await t11_notes.target.transition_to(
        chat_state="Case requires human review. Adding escalation note and assigning to agent.",
        condition="Decision is Needs Human Review",
    )
    t14_escalate_api = await t11_api.target.transition_to(
        chat_state="Case requires human review. Adding escalation note and assigning to agent.",
        condition="Decision is Needs Human Review",
    )
    
    t15_escalate_notes = await t14_escalate_notes.target.transition_to(tool_state=add_note)
    t15_escalate_api = await t14_escalate_api.target.transition_to(tool_state=add_note)
    
    t16_escalate_notes = await t15_escalate_notes.target.transition_to(
        chat_state="Note added. Assigning to human agent and completing process.",
    )
    t16_escalate_api = await t15_escalate_api.target.transition_to(
        chat_state="Note added. Assigning to human agent and completing process.",
    )
    
    t17_escalate_notes = await t16_escalate_notes.target.transition_to(tool_state=update_ticket)
    t17_escalate_api = await t16_escalate_api.target.transition_to(tool_state=update_ticket)
    
    await t17_escalate_notes.target.transition_to(
        chat_state="Process complete. Ticket escalated to human agent."
    )
    await t17_escalate_api.target.transition_to(
        chat_state="Process complete. Ticket escalated to human agent."
    )
    
    # ESCALATE from security
    t14_sec = await t8_unsafe.target.transition_to(tool_state=add_note)
    t15_sec = await t14_sec.target.transition_to(
        chat_state="Security note added. Assigning to security team and completing process.",
    )
    t16_sec = await t15_sec.target.transition_to(tool_state=update_ticket)
    await t16_sec.target.transition_to(
        chat_state="Process complete. Security threat escalated."
    )
    
    print(f"✓ Created journey: {journey.title}")

# async def create_automated_processing_journey(agent: p.Agent):
#     """
#     Automated ticket processing journey for webhook-triggered refunds.
#     This journey has no chat states and runs silently in the background.
#     """
#     journey = await agent.create_journey(
#         title="Automated Ticket Processing",
#         conditions=[
#             "A webhook is received to process a Freshdesk ticket",
#             "A ticket needs to be processed automatically without user interaction"
#         ],
#         description="Processes Freshdesk tickets silently in the background. This journey is triggered by webhooks and performs the complete refund workflow: fetching data, security scanning, extracting booking info, making a triage decision, and updating the ticket without any chat-based feedback.",
#     )
    
#     # This journey uses the process_ticket_end_to_end tool, which encapsulates
#     # the entire workflow into a single, non-interactive operation.
    
#     # Import the consolidated tool
#     from app_tools.tools.process_ticket_workflow import process_ticket_end_to_end
    
#     # Initial state transitions directly to the end-to-end processing tool
#     t1 = await journey.initial_state.transition_to(tool_state=process_ticket_end_to_end)
    
#     # The journey ends after the tool completes. The tool itself handles
#     # all sub-steps and updates Freshdesk. The final state of the journey
#     # can simply confirm completion.
#     await t1.target.transition_to(
#         chat_state="Automated processing complete. The ticket has been analyzed and updated in Freshdesk.",
#         # This chat state will not be shown to a user but serves as a final
#         # confirmation in the journey logs.
#     )
    
#     print(f"✓ Created journey: {journey.title}")




# --- Main Application ---
async def main():
    # Validate LLM configuration before starting server
    provider = validate_llm_config()
    
    # Validate ParkWhiz OAuth2 credentials for booking verification
    try:
        validate_oauth2_credentials()
        logger.info("✓ ParkWhiz OAuth2 credentials validated successfully")
    except Exception as e:
        logger.warning(
            f"ParkWhiz OAuth2 validation failed: {e}. "
            "Booking verification will be disabled. "
            "To enable, set PARKWHIZ_CLIENT_ID and PARKWHIZ_CLIENT_SECRET in .env"
        )
    
    # Configure environment variables for Parlant SDK
    if provider == "gemini":
        # Set Google-specific environment variables for Parlant SDK
        # Parlant's Gemini service reads from GOOGLE_API_KEY and GOOGLE_MODEL
        os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")
        os.environ["GOOGLE_MODEL"] = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
        logger.info(f"Initializing Parlant server with Gemini provider (model: {os.environ['GOOGLE_MODEL']})")
        nlp_service_factory = p.NLPServices.gemini
    elif provider == "openai":
        # OpenAI uses default configuration via OPENAI_API_KEY env var
        logger.info("Initializing Parlant server with OpenAI provider")
        nlp_service_factory = p.NLPServices.openai
    
    try:
        async with p.Server(nlp_service=nlp_service_factory) as server:
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
            print(f"LLM Provider: {provider.upper()}")
            print("UI:   http://127.0.0.1:8800")
            print(f"AGENT_ID: {agent_id}")
            print(f"(Also saved to: {AGENT_ID_FILE})\n")

            # --- Register Journeys and Guidelines here ---
            await create_interactive_processing_journey(agent)
            # await create_automated_processing_journey(agent)

            # Add general guidelines
            await agent.create_guideline(
                condition="the user greets you",
                action="Greet back warmly and ask how you can help today.",
            )
            
            await agent.create_guideline(
                condition="a ticket subject contains 'TEST' or 'IGNORE' or similar testing markers",
                action="Continue processing the ticket through the complete journey. These markers are for human agents only. Process the refund request based on the actual ticket content, not the subject line markers.",
            )
    
    except Exception as e:
        error_msg = str(e).lower()
        
        # Gemini-specific error handling
        if provider == "gemini":
            # Rate limit errors (429)
            if "429" in error_msg or "rate" in error_msg or "quota" in error_msg:
                logger.error(
                    "Gemini API rate limit exceeded. "
                    "Consider upgrading to a paid tier for higher limits. "
                    "Free tier: 15 RPM, 1M TPM, 1,500 RPD. "
                    "Paid tier: 2,000 RPM, 4M TPM. "
                    f"Error details: {e}"
                )
                raise RuntimeError(
                    "Gemini rate limit exceeded. Please wait before retrying or upgrade your API tier."
                ) from e
            
            # Authentication errors
            elif "401" in error_msg or "403" in error_msg or "api_key" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
                logger.error(
                    "Gemini API authentication failed. "
                    "Please verify your GEMINI_API_KEY is correct and active. "
                    "Get your API key from: https://aistudio.google.com/apikey "
                    f"Error details: {e}"
                )
                raise RuntimeError(
                    "Gemini authentication failed. Please check your GEMINI_API_KEY."
                ) from e
            
            # Model not found or unavailable
            elif "404" in error_msg or "model" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
                model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
                logger.error(
                    f"Gemini model '{model}' not found or unavailable. "
                    "Available models: gemini-2.5-flash, gemini-2.5-flash-lite, gemini-2.5-pro, gemini-2.0-flash-exp. "
                    f"Error details: {e}"
                )
                raise RuntimeError(
                    f"Gemini model '{model}' not available. Please check GEMINI_MODEL configuration."
                ) from e
            
            # Generic Gemini API errors
            else:
                logger.error(
                    f"Gemini API error occurred: {e}. "
                    "Check your API key, model configuration, and network connectivity. "
                    "Visit https://ai.google.dev/gemini-api/docs for troubleshooting."
                )
                raise RuntimeError(f"Gemini API error: {e}") from e
        
        # OpenAI-specific error handling
        elif provider == "openai":
            if "429" in error_msg or "rate" in error_msg:
                logger.error(f"OpenAI rate limit exceeded: {e}")
                raise RuntimeError("OpenAI rate limit exceeded. Please wait before retrying.") from e
            elif "401" in error_msg or "api_key" in error_msg:
                logger.error(f"OpenAI authentication failed: {e}")
                raise RuntimeError("OpenAI authentication failed. Please check your OPENAI_API_KEY.") from e
            else:
                logger.error(f"OpenAI API error: {e}")
                raise
        
        # Unknown errors
        else:
            logger.error(f"Unexpected error during server initialization: {e}")
            raise


if __name__ == "__main__":
    asyncio.run(main())
