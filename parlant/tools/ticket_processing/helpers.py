
import parlant.sdk as p
from ..booking_analysis.extractor import BookingExtractor
from ..refund_decision.decision_maker import DecisionMaker
from ..integrations.freshdesk import add_note, update_ticket


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
    
    # Use the optimized BookingExtractor with pattern-based extraction
    try:
        extractor = BookingExtractor(use_pattern_fallback=True)
        result = await extractor.extract_booking_info(ticket_notes)
        
        # Format result for Parlant - include ALL important flags
        return p.ToolResult(
            data={
                "booking_info": result["booking_info"] if result["found"] else None,
                "confidence": result["confidence"],
                "found_in": "notes",
                "extraction_method": result["extraction_method"],
                "found": result["found"],
                "booking_not_found": result.get("booking_not_found", False),
                "booking_mismatch": result.get("booking_mismatch", False),
                "booking_mismatch_details": result.get("booking_mismatch_details"),
                "api_error": result.get("api_error"),
                "multiple_bookings": result.get("multiple_bookings", False)
            },
            metadata={
                "summary": f"Booking info: {'found' if result['found'] else 'not found'} "
                          f"(method: {result['extraction_method']}, confidence: {result['confidence']})"
                          f"{' - BOOKING NOT FOUND IN SYSTEM' if result.get('booking_not_found') else ''}"
                          f"{' - BOOKING MISMATCH DETECTED' if result.get('booking_mismatch') else ''}"
            }
        )
    except Exception as e:
        # Fallback if extraction fails
        return p.ToolResult(
            data={
                "booking_info": None,
                "confidence": "error",
                "found_in": "none",
                "error": str(e)
            },
            metadata={"summary": f"Booking info extraction failed: {str(e)}"}
        )


@p.tool
async def triage_ticket(context: p.ToolContext) -> p.ToolResult:
    """
    Analyzes all available information to make a refund decision using the hybrid
    rule-based and LLM-powered decision-making system.
    
    This tool orchestrates the complete decision flow:
    1. Checks pre-arrival fast path (instant approval if current time < booking start time)
    2. Uses pre-extracted booking info if available, otherwise extracts from ticket notes
    3. Applies deterministic business rules for clear-cut cases
    4. Uses LLM analysis for complex or ambiguous cases
    5. Maps to appropriate ParkWhiz cancellation reason for approved refunds
    
    Args:
        ticket_data (dict): Complete Freshdesk ticket information including:
            - id (str): Ticket ID
            - subject (str): Ticket subject
            - description (str): Ticket description
            - status (str): Current status
        booking_info (dict|null): Optional pre-extracted ParkWhiz booking information
        ticket_notes (str): Optional raw text from ticket notes/conversations
        processing_context (str): Optional context ("webhook" or "chat") for context-aware processing
    
    Returns:
        ToolResult with:
        - decision (str): One of "Approved", "Denied", or "Needs Human Review"
        - reasoning (str): Clear explanation of why this decision was made
        - policy_applied (str): Which specific policy rule was applied
        - confidence (str): Confidence level in the decision (high/medium/low)
        - cancellation_reason (str|null): ParkWhiz cancellation reason (if Approved)
        - booking_info_found (bool): Whether booking info was successfully extracted
        - method_used (str): Decision method ("rules", "llm", "hybrid", or "pre_arrival_gate")
        - processing_time_ms (int): Time taken to make the decision
        - pre_arrival_checked (bool): Whether pre-arrival gate was executed
    """
    ticket_data = context.inputs.get("ticket_data", {})
    booking_info = context.inputs.get("booking_info")
    booking_extraction_result = context.inputs.get("booking_extraction_result")
    ticket_notes = context.inputs.get("ticket_notes", "")
    processing_context = context.inputs.get("processing_context", "webhook")  # Default to webhook for backward compatibility
    
    ticket_id = ticket_data.get("id", "unknown")
    
    # Handle minimal data case
    if not ticket_data:
        return p.ToolResult(
            data={
                "decision": "Needs Human Review",
                "reasoning": "Insufficient ticket data to make an automated decision",
                "policy_applied": "Data Validation - No Ticket Data",
                "confidence": "none",
                "cancellation_reason": None,
                "booking_info_found": False,
                "method_used": "validation_failed",
                "processing_time_ms": 0,
                "pre_arrival_checked": False
            },
            metadata={"summary": "Triage decision: Needs Human Review (no data)"}
        )
    
    # GATE 5: Pre-Arrival Fast Path Check
    # Check if this is a pre-arrival cancellation BEFORE expensive LLM processing
    # This provides instant approval for the simplest refund case
    import time
    start_time = time.perf_counter()
    
    try:
        from ..routing.ticket_router import TicketRouter
        
        # Initialize router with default configuration
        router = TicketRouter()
        
        # Check if pre-arrival gate is enabled
        if router.is_pre_arrival_enabled():
            # Run pre-arrival check with appropriate context
            routing_decision = router.route_ticket(ticket_data)
            
            # Check if pre-arrival gate approved the ticket
            if routing_decision.approval_source == "pre_arrival_gate":
                # Pre-arrival approval - instant approval with <200ms latency
                processing_time = int((time.perf_counter() - start_time) * 1000)
                
                return p.ToolResult(
                    data={
                        "decision": "Approved",
                        "reasoning": routing_decision.approval_reason,
                        "policy_applied": "Pre-Arrival Cancellation Policy",
                        "confidence": "high",
                        "cancellation_reason": "Customer Cancelled",  # Standard ParkWhiz reason for pre-arrival
                        "booking_info_found": True,  # Pre-arrival gate found booking start time
                        "method_used": "pre_arrival_gate",
                        "processing_time_ms": processing_time,
                        "pre_arrival_checked": True,
                        "hours_before_booking": routing_decision.hours_before_booking,
                        "timezone_used": routing_decision.timezone_used,
                        "warnings": routing_decision.warnings if routing_decision.warnings else []
                    },
                    metadata={
                        "summary": f"Decision: Approved (pre-arrival fast path, {processing_time}ms)"
                    }
                )
            
            # Pre-arrival gate determined NOT pre-arrival - continue to standard decision flow
            # This is expected for most tickets (post-arrival, during event, etc.)
        
    except Exception as e:
        # Pre-arrival gate error - log and continue to standard decision flow
        # This ensures graceful degradation if pre-arrival gate fails
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Pre-arrival gate check failed for ticket {ticket_id}: {e}")
        # Continue to standard decision flow below
    
    # Standard Decision Flow (for non-pre-arrival tickets)
    # Use DecisionMaker to orchestrate the decision process
    try:
        decision_maker = DecisionMaker()
        result = await decision_maker.make_decision(
            ticket_data=ticket_data,
            ticket_notes=ticket_notes,
            booking_info=booking_info,
            booking_extraction_result=booking_extraction_result
        )
        
        # Add pre_arrival_checked flag to result
        result["pre_arrival_checked"] = True  # Pre-arrival was checked above
        
        # Format summary for metadata
        decision = result.get("decision", "Unknown")
        method = result.get("method_used", "unknown")
        confidence = result.get("confidence", "unknown")
        processing_time = result.get("processing_time_ms", 0)
        
        summary = (
            f"Decision: {decision} "
            f"(method: {method}, confidence: {confidence}, time: {processing_time}ms)"
        )
        
        # Add cancellation reason to summary if Approved
        if decision == "Approved" and result.get("cancellation_reason"):
            summary += f" - Reason: {result['cancellation_reason']}"
        
        return p.ToolResult(
            data=result,
            metadata={"summary": summary}
        )
        
    except Exception as e:
        # Handle unexpected errors gracefully
        return p.ToolResult(
            data={
                "decision": "Needs Human Review",
                "reasoning": f"Error during decision-making process: {str(e)}. Human review required.",
                "policy_applied": "System Error",
                "confidence": "none",
                "cancellation_reason": None,
                "booking_info_found": False,
                "method_used": "error",
                "processing_time_ms": 0,
                "pre_arrival_checked": True,
                "error": str(e)
            },
            metadata={"summary": f"Triage decision failed: {str(e)}"}
        )



@p.tool
async def document_decision(context: p.ToolContext) -> p.ToolResult:
    """
    Documents the refund decision in Freshdesk with a private note and tag.
    
    MVP Scope: This tool documents the decision but does NOT process the refund
    via ParkWhiz API. This allows the team to verify decision accuracy before
    enabling full automation.
    
    The tool:
    1. Formats a comprehensive private note with decision details
    2. Includes ParkWhiz cancellation reason for Approved decisions
    3. Adds the note to the Freshdesk ticket
    4. Tags the ticket with "Processed by Whiz Agent"
    
    Args:
        ticket_id (str): The Freshdesk ticket ID to document
        decision_result (dict): The decision data from triage_ticket containing:
            - decision (str): "Approved", "Denied", or "Needs Human Review"
            - reasoning (str): Explanation of the decision
            - policy_applied (str): Which policy rule was applied
            - confidence (str): Confidence level (high/medium/low)
            - cancellation_reason (str|null): ParkWhiz cancellation reason (if Approved)
            - method_used (str): Decision method used
            - processing_time_ms (int): Processing time
    
    Returns:
        ToolResult with:
        - documented (bool): Whether the decision was successfully documented
        - ticket_id (str): The ticket ID that was updated
        - note_added (bool): Whether the private note was added
        - tag_added (bool): Whether the tag was added
    """
    ticket_id = context.inputs.get("ticket_id")
    decision_result = context.inputs.get("decision_result", {})
    
    # Validate inputs
    if not ticket_id:
        return p.ToolResult(
            data={
                "documented": False,
                "error": "No ticket_id provided"
            },
            metadata={"summary": "Error: No ticket_id provided"}
        )
    
    if not decision_result:
        return p.ToolResult(
            data={
                "documented": False,
                "error": "No decision_result provided"
            },
            metadata={"summary": "Error: No decision_result provided"}
        )
    
    # Extract decision details
    decision = decision_result.get("decision", "Unknown")
    reasoning = decision_result.get("reasoning", "No reasoning provided")
    policy_applied = decision_result.get("policy_applied", "No policy specified")
    confidence = decision_result.get("confidence", "unknown")
    cancellation_reason = decision_result.get("cancellation_reason")
    method_used = decision_result.get("method_used", "unknown")
    processing_time_ms = decision_result.get("processing_time_ms", 0)
    
    # Format the private note
    note_lines = [
        f"**AGENT DECISION: {decision}**",
        "",
        "**Reasoning:**",
        reasoning,
        "",
        "**Policy Applied:**",
        policy_applied,
    ]
    
    # Add cancellation reason for Approved decisions
    if decision == "Approved" and cancellation_reason:
        note_lines.extend([
            "",
            "**ParkWhiz Cancellation Reason:**",
            cancellation_reason,
        ])
    
    # Add metadata
    note_lines.extend([
        "",
        f"**Confidence Level:** {confidence}",
        f"**Method Used:** {method_used}",
        f"**Processing Time:** {processing_time_ms}ms",
        "",
        "---",
        "This decision was made by the Whiz Agent. Please review before processing the refund.",
        "",
        "[Report AI Inaccuracy](https://forms.gle/1NMxGcYeLkZnM4956)",
    ])
    
    note_body = "\n".join(note_lines)
    
    # Add the private note to the ticket
    try:
        note_result = await add_note(
            context=context,
            ticket_id=ticket_id,
            note=note_body
        )
        
        note_added = "error" not in note_result.data
        
        if not note_added:
            return p.ToolResult(
                data={
                    "documented": False,
                    "ticket_id": ticket_id,
                    "note_added": False,
                    "tag_added": False,
                    "error": f"Failed to add note: {note_result.data.get('error')}"
                },
                metadata={"summary": f"Error: Failed to add note to ticket {ticket_id}"}
            )
    except Exception as e:
        return p.ToolResult(
            data={
                "documented": False,
                "ticket_id": ticket_id,
                "note_added": False,
                "tag_added": False,
                "error": f"Exception adding note: {str(e)}"
            },
            metadata={"summary": f"Error: Exception adding note to ticket {ticket_id}"}
        )
    
    # Add the "Processed by Whiz Agent" tag
    try:
        tag_result = await update_ticket(
            context=context,
            ticket_id=ticket_id,
            tags=["Processed by Whiz Agent"]
        )
        
        tag_added = "error" not in tag_result.data
        
        if not tag_added:
            return p.ToolResult(
                data={
                    "documented": True,  # Note was added successfully
                    "ticket_id": ticket_id,
                    "note_added": True,
                    "tag_added": False,
                    "warning": f"Note added but failed to add tag: {tag_result.data.get('error')}"
                },
                metadata={"summary": f"Documented {decision} decision for ticket {ticket_id} (tag failed)"}
            )
    except Exception as e:
        return p.ToolResult(
            data={
                "documented": True,  # Note was added successfully
                "ticket_id": ticket_id,
                "note_added": True,
                "tag_added": False,
                "warning": f"Note added but exception adding tag: {str(e)}"
            },
            metadata={"summary": f"Documented {decision} decision for ticket {ticket_id} (tag failed)"}
        )
    
    # Success - both note and tag added
    return p.ToolResult(
        data={
            "documented": True,
            "ticket_id": ticket_id,
            "note_added": True,
            "tag_added": True,
            "decision": decision
        },
        metadata={"summary": f"Documented {decision} decision for ticket {ticket_id}"}
    )
