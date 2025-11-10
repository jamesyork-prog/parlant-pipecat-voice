
import parlant.sdk as p


@p.tool
async def extract_booking_info_from_note(context: p.ToolContext) -> p.ToolResult:
    """
    Analyzes Freshdesk ticket notes to extract ParkWhiz booking information.
    
    This is a "thinking tool" that uses the LLM's natural language understanding
    to identify and extract structured booking data from unstructured text.
    
    Args:
        ticket_notes (str): The private notes or description from the Freshdesk ticket
    
    Returns:
        ToolResult with:
        - booking_info (dict|null): Extracted booking information including booking_id,
          amount, date, location, etc. Returns null if no booking info is found.
        - confidence (str): How confident the extraction is (high/medium/low)
        - found_in (str): Where the information was found (notes/description/custom_fields)
    
    Instructions for the LLM:
        Carefully analyze the ticket notes for any ParkWhiz booking information.
        Look for:
        - Booking IDs (usually starting with PW- or similar)
        - Transaction amounts (e.g., $45.00, 45 USD)
        - Reservation dates
        - Parking locations/venues
        - Customer name or email
        
        If you find booking information, extract it as a structured JSON object.
        If multiple bookings are mentioned, focus on the one being disputed.
        If no clear booking information exists, return null for booking_info.
    """
    ticket_notes = context.inputs.get("ticket_notes", "")
    
    # Handle empty notes
    if not ticket_notes or ticket_notes.strip() == "":
        return p.ToolResult(
            data={
                "booking_info": None,
                "confidence": "none",
                "found_in": "none",
                "message": "No ticket notes provided"
            },
            metadata={"summary": "Booking info: not found (empty notes)"}
        )
    
    # The LLM will process this tool and extract the booking information
    # from the ticket_notes based on the instructions in the docstring
    return p.ToolResult(
        data={
            "booking_info": None,  # LLM will populate this
            "confidence": "pending",
            "found_in": "notes",
            "raw_notes": ticket_notes
        },
        metadata={"summary": "Booking info extraction requested"}
    )


@p.tool
async def triage_ticket(context: p.ToolContext) -> p.ToolResult:
    """
    Analyzes all available information to make a refund decision.
    
    This is the main "thinking tool" that synthesizes ticket data, booking information,
    and refund policy to determine the appropriate action.
    
    Args:
        ticket_data (dict): Complete Freshdesk ticket information
        booking_info (dict|null): Extracted ParkWhiz booking information
        refund_policy (str): Relevant refund policy text from the retriever
    
    Returns:
        ToolResult with:
        - decision (str): One of "Approved", "Denied", or "Needs Human Review"
        - reasoning (str): Clear explanation of why this decision was made
        - policy_applied (str): Which specific policy rule was applied
        - confidence (str): Confidence level in the decision (high/medium/low)
        - recommended_action (str): Specific next step to take
    
    Instructions for the LLM:
        You are an expert refund analyst. Analyze ALL provided information:
        
        1. Review the ticket details (subject, description, custom fields)
        2. Examine the booking information (if available)
        3. Apply the refund policy rules
        
        Decision Criteria:
        - APPROVED: Clear policy match, all required info present, straightforward case
        - DENIED: Clearly violates policy, well-documented reason
        - NEEDS HUMAN REVIEW: Missing critical info, edge case, policy ambiguity,
          security concerns, high value transaction, or customer escalation
        
        Always err on the side of "Needs Human Review" if you're uncertain.
        Provide clear, customer-friendly reasoning that can be shared with the customer.
    """
    ticket_data = context.inputs.get("ticket_data", {})
    booking_info = context.inputs.get("booking_info")
    refund_policy = context.inputs.get("refund_policy", "")
    
    ticket_id = ticket_data.get("id", "unknown")
    
    # Handle minimal data case
    if not ticket_data:
        return p.ToolResult(
            data={
                "decision": "Needs Human Review",
                "reasoning": "Insufficient ticket data to make an automated decision",
                "policy_applied": "None",
                "confidence": "none",
                "recommended_action": "escalate_to_human"
            },
            metadata={"summary": "Triage decision: Needs Human Review (no data)"}
        )
    
    # The LLM will process this tool and make the triage decision
    # based on the instructions in the docstring and the provided data
    return p.ToolResult(
        data={
            "decision": "Needs Human Review",  # LLM will determine the actual decision
            "reasoning": "Analyzing ticket information...",
            "policy_applied": refund_policy if refund_policy else "No policy provided",
            "confidence": "pending",
            "recommended_action": "analyze",
            "ticket_id": ticket_id,
            "has_booking_info": booking_info is not None
        },
        metadata={"summary": f"Triage decision requested for ticket {ticket_id}"}
    )
