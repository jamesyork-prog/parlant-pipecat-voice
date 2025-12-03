import parlant.sdk as p
import os
from datetime import datetime
from .freshdesk_tools import get_ticket, get_ticket_description, get_ticket_conversations
from .lakera_security_tool import check_content
from .journey_helpers import extract_booking_info_from_note, triage_ticket
from .structured_logger import (
    configure_structured_logging,
    log_journey_start,
    log_journey_end,
    log_tool_execution,
    log_decision_outcome,
    log_error_with_context
)
from .zapier_failure_detector import ZapierFailureDetector
from .customer_info_extractor import CustomerInfoExtractor
from .decision_guard import DecisionGuard
from .verification_note_generator import VerificationNoteGenerator

# Configure structured logging
logger = configure_structured_logging(
    level=os.getenv("WEBHOOK_LOG_LEVEL", "INFO"),
    component_name="process_ticket_workflow"
)


def _is_paid_again_claim(ticket_text: str) -> bool:
    """
    Detect if ticket mentions 'paid again' or similar duplicate charge claims.
    
    Args:
        ticket_text (str): Combined ticket text (subject + description + conversations)
    
    Returns:
        bool: True if ticket contains duplicate charge keywords
    """
    if not ticket_text:
        return False
    
    # Convert to lowercase for case-insensitive matching
    text_lower = ticket_text.lower()
    
    # Keywords that indicate duplicate charge claims
    keywords = [
        "paid again",
        "charged twice",
        "double charged",
        "charged multiple times",
        "billed twice",
        "duplicate charge",
        "charged me again",
        "two bookings",
        "booked twice",
        "duplicate booking",
        "charged 2 times",
        "paid 2 times",
        "paid two times",
        "charged two times",
        "double payment",
        "duplicate payment",
    ]
    
    return any(keyword in text_lower for keyword in keywords)


@p.tool
async def process_ticket_end_to_end(context: p.ToolContext, ticket_id: str) -> p.ToolResult:
    """
    Processes a Freshdesk ticket through the complete refund workflow autonomously.
    
    This tool orchestrates the entire ticket processing pipeline:
    1. Fetches ticket metadata, description, and conversations
    2. Runs security scan on content
    3. Extracts booking information
    4. Makes triage decision (Approved/Denied/Escalate)
    5. Returns the decision and reasoning
    
    Args:
        ticket_id (str): The Freshdesk ticket ID to process
    
    Returns:
        Complete analysis with decision, reasoning, and next steps
    """
    start_time = datetime.utcnow()
    
    # Log journey start
    log_journey_start(logger, ticket_id, "Automated Ticket Processing")
    
    results = {
        "ticket_id": ticket_id,
        "steps_completed": [],
        "errors": []
    }
    
    try:
    
        # Step 1: Fetch ticket metadata
        log_tool_execution(logger, ticket_id, "get_ticket")
        ticket_result = await get_ticket(context, ticket_id)
        if "error" in ticket_result.data:
            log_tool_execution(logger, ticket_id, "get_ticket", success=False, error=str(ticket_result.data))
            return p.ToolResult(
                {"error": "Failed to fetch ticket", "details": ticket_result.data},
                metadata={"summary": f"Error processing ticket {ticket_id}"}
            )
        # Store only essential metadata, not full response
        results["ticket_metadata"] = {
            "id": ticket_result.data.get("id"),
            "subject": ticket_result.data.get("subject"),
            "status": ticket_result.data.get("status")
        }
        results["steps_completed"].append("Fetched ticket metadata")
    
        # Step 2: Fetch description
        log_tool_execution(logger, ticket_id, "get_ticket_description")
        desc_result = await get_ticket_description(context, ticket_id)
        description_text = ""
        if "error" not in desc_result.data:
            description_text = desc_result.data.get('description_text', '')
            results["steps_completed"].append("Fetched ticket description")
        
        # Step 3: Fetch conversations
        log_tool_execution(logger, ticket_id, "get_ticket_conversations")
        conv_result = await get_ticket_conversations(context, ticket_id)
        conversations = []
        if "error" not in conv_result.data:
            conversations = conv_result.data.get('conversations', [])
            results["steps_completed"].append("Fetched conversation history")
    
        # Step 4: Security scan
        # Combine description and subject for security check
        subject = results['ticket_metadata'].get('subject', '')
        content_to_check = f"{subject}\n\n{description_text}".strip()
        
        # Only run security scan if there's actual content
        if content_to_check:
            # Call check_content directly with content parameter
            # Create proper context with content in inputs
            from unittest.mock import Mock
            security_context = Mock()
            security_context.inputs = {'content': content_to_check}
            security_context.agent_id = context.agent_id
            security_context.customer_id = context.customer_id
            security_context.session_id = context.session_id
            
            log_tool_execution(logger, ticket_id, "check_content")
            security_result = await check_content(security_context)
            results["security_scan"] = security_result.data
            results["steps_completed"].append("Completed security scan")
        
            # Only escalate if actually flagged (not just missing data)
            if security_result.data.get("flagged", False):
                results["decision"] = "SECURITY_ESCALATION"
                results["reasoning"] = f"Content flagged by security scan: {security_result.data.get('categories', {})}"
                
                # Log security escalation decision
                log_decision_outcome(logger, ticket_id, "SECURITY_ESCALATION", reasoning=results["reasoning"])
                
                # Still add note and update ticket for security escalation
                note_text = "<h3>⚠️ Security Alert</h3>"
                note_text += f"<p><strong>Status:</strong> Flagged by automated security scan</p>"
                note_text += f"<p><strong>Categories:</strong> {security_result.data.get('categories', {})}</p>"
                note_text += f"<p><strong>Action Required:</strong> Manual security review needed</p>"
                
                from .freshdesk_tools import add_note, update_ticket
                log_tool_execution(logger, ticket_id, "add_note")
                await add_note(context, ticket_id, note_text)
                log_tool_execution(logger, ticket_id, "update_ticket")
                await update_ticket(context, ticket_id, tags=["security_flagged", "needs_review"])
                
                # Log journey end
                processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                log_journey_end(logger, ticket_id, "Automated Ticket Processing", processing_time_ms, "SECURITY_ESCALATION")
                
                return p.ToolResult(
                    {
                        "ticket_id": ticket_id,
                        "decision": "SECURITY_ESCALATION",
                        "reasoning": results["reasoning"],
                        "security_details": security_result.data,
                    },
                    metadata={"summary": f"Ticket {ticket_id}: Security escalation - flagged content"}
                )
        else:
            # No content to scan
            results["security_scan"] = {"safe": True, "flagged": False, "note": "No content to scan"}
            results["steps_completed"].append("Security scan skipped (no content)")
    
        # Step 5: Extract booking info
        # Combine ALL available text sources for booking extraction
        notes_text = f"Subject: {results['ticket_metadata'].get('subject', '')}\n\n"
        notes_text += f"Description: {description_text}\n\n"
        
        # Add all conversations (private notes contain booking info)
        if conversations:
            notes_text += "Private Notes:\n"
            for conv in conversations:
                # Use full body_text, not truncated
                notes_text += f"\n{conv.get('body_text', '')}\n"
        
        from unittest.mock import Mock
        booking_context = Mock()
        booking_context.inputs = {'ticket_notes': notes_text}
        booking_context.agent_id = context.agent_id
        booking_context.customer_id = context.customer_id
        booking_context.session_id = context.session_id
        
        log_tool_execution(logger, ticket_id, "extract_booking_info_from_note")
        booking_result = await extract_booking_info_from_note(booking_context)
        results["booking_info"] = booking_result.data
        results["steps_completed"].append("Extracted booking information")
    
        # Debug: Log what we're analyzing
        results["debug_notes_length"] = len(notes_text)
        results["debug_notes_sample"] = notes_text[:500] if notes_text else "No notes"
        
        # Step 6: Detect Zapier failures and verify booking if needed
        zapier_detector = ZapierFailureDetector()
        booking_info = booking_result.data.get("booking_info", {})
        booking_id = booking_info.get("booking_id")
        
        # Check if Zapier failed or booking ID is invalid
        zapier_failure = zapier_detector.is_zapier_failure(notes_text)
        invalid_booking_id = zapier_detector.is_invalid_booking_id(booking_id)
        
        verification_result = None
        verified_booking = None
        
        if zapier_failure or invalid_booking_id:
            logger.info(
                f"Zapier failure detected for ticket {ticket_id} - booking verification disabled due to API limitations",
                extra={
                    "ticket_id": ticket_id,
                    "zapier_failure": zapier_failure,
                    "invalid_booking_id": invalid_booking_id,
                    "booking_id": booking_id,
                    "reason": "ParkWhiz API cannot search bookings by customer email"
                }
            )
            results["steps_completed"].append("Zapier failure detected - verification unavailable (API limitation)")
        
        # Set booking info source based on verification
        if verified_booking:
            results["booking_info_source"] = "parkwhiz_api_verified"
        elif not booking_result.data.get("booking_info"):
            results["booking_info_source"] = "missing"
        else:
            results["booking_info_source"] = "ticket_notes"
        
        # Step 6.5: Check for "paid again" / duplicate claims
        duplicate_detection_result = None
        if _is_paid_again_claim(notes_text):
            logger.info(f"Detected duplicate/paid-again claim in ticket {ticket_id} - escalating to human review")
            results["steps_completed"].append("Detected duplicate claim - escalating per API limitation")
            
            duplicate_detection_result = type('obj', (object,), {
                'data': {
                    "action_taken": "escalate",
                    "explanation": (
                        "Customer reports duplicate booking or being charged twice. "
                        "Duplicate detection requires manual review because the ParkWhiz API "
                        "does not support searching bookings by customer email. "
                        "A specialist will review the account to locate both bookings."
                    ),
                    "has_duplicates": None,
                    "api_limitation": True
                }
            })()
            
            results["duplicate_detection"] = duplicate_detection_result.data
            results["steps_completed"].append("Duplicate claim flagged for human review")
        
        # Step 7: Make decision based on duplicate detection or triage
        # If duplicate detection was run, use its results for decision
        if duplicate_detection_result:
            detection_data = duplicate_detection_result.data
            action_taken = detection_data.get("action_taken")
            
            if action_taken == "refunded":
                # Duplicate found and refunded - APPROVE
                decision = "Approved"
                reasoning = (
                    f"Duplicate booking detected and refunded. "
                    f"Refunded booking {detection_data.get('refunded_booking_id')}, "
                    f"kept booking {detection_data.get('kept_booking_id')}. "
                    f"{detection_data.get('explanation', '')}"
                )
                confidence = "high"
                policy_applied = "Duplicate Booking Detection"
                refunded = True
                
            elif action_taken == "escalate":
                # 3+ duplicates or error - ESCALATE
                decision = "Needs Human Review"
                reasoning = f"Duplicate detection requires human review. {detection_data.get('explanation', '')}"
                confidence = "medium"
                policy_applied = "Duplicate Booking Detection"
                refunded = False
                
            elif action_taken == "deny":
                # No duplicates found - DENY
                decision = "Denied"
                reasoning = f"No duplicate bookings found. {detection_data.get('explanation', '')}"
                confidence = "high"
                policy_applied = "Duplicate Booking Detection"
                refunded = False
                
            else:
                # Unknown action or error - ESCALATE
                decision = "Needs Human Review"
                reasoning = f"Duplicate detection inconclusive. {detection_data.get('explanation', 'Unknown error')}"
                confidence = "low"
                policy_applied = "Duplicate Booking Detection"
                refunded = False
            
            results["triage_decision"] = {
                "decision": decision,
                "reasoning": reasoning,
                "confidence": confidence,
                "policy_applied": policy_applied,
                "duplicate_detection_used": True,
                "refunded": refunded
            }
            results["steps_completed"].append("Decision made based on duplicate detection")
            
        else:
            # No duplicate detection - check if we should use verified data or escalate
            if verification_result and verification_result.success and verified_booking:
                # Use verified booking data for decision
                decision_guard = DecisionGuard()
                
                # Check if automated decision is safe
                can_decide = decision_guard.can_make_automated_decision(verified_booking)
                should_escalate, escalation_reason = decision_guard.should_escalate(
                    verified_booking,
                    verification_result.customer_info,
                    verification_result.failure_reason
                )
                
                if should_escalate or not can_decide:
                    # Escalate to human review
                    decision = "Needs Human Review"
                    reasoning = escalation_reason or "Booking verified but requires human review"
                    confidence = "medium"
                    policy_applied = "Booking Verification"
                    refunded = False
                    
                    logger.info(
                        f"Escalating ticket {ticket_id} after verification",
                        extra={
                            "ticket_id": ticket_id,
                            "reason": escalation_reason,
                            "booking_id": verified_booking.booking_id
                        }
                    )
                    
                else:
                    # Use standard triage with verified booking data
                    from unittest.mock import Mock
                    triage_context = Mock()
                    
                    # Include full ticket text and VERIFIED booking info
                    full_ticket_data = results['ticket_metadata'].copy()
                    full_ticket_data['description'] = notes_text
                    
                    # Use verified booking data instead of customer-provided
                    verified_booking_info = {
                        "booking_id": verified_booking.booking_id,
                        "customer_email": verified_booking.customer_email,
                        "event_date": verified_booking.arrival_date,
                        "location": verified_booking.location,
                        "pass_used": verified_booking.pass_used,
                        "pass_usage_status": verified_booking.pass_usage_status,
                        "amount_paid": verified_booking.amount_paid,
                        "verified": True  # Flag to indicate this is verified data
                    }
                    
                    triage_context.inputs = {
                        'ticket_data': full_ticket_data,
                        'booking_info': verified_booking_info,
                        'refund_policy': "Standard refund policy applies"
                    }
                    triage_context.agent_id = context.agent_id
                    triage_context.customer_id = context.customer_id
                    triage_context.session_id = context.session_id
                    
                    log_tool_execution(logger, ticket_id, "triage_ticket")
                    triage_result = await triage_ticket(triage_context)
                    
                    decision = triage_result.data.get("decision", "Needs Human Review")
                    reasoning = triage_result.data.get("reasoning", "Analysis complete")
                    confidence = triage_result.data.get("confidence", "N/A")
                    policy_applied = triage_result.data.get("policy_applied", "Standard refund policy")
                    refunded = False
                    
                    # Add note that decision used verified data
                    reasoning += " (Decision based on ParkWhiz verified booking data)"
                    
                    logger.info(
                        f"Decision made using verified booking data for ticket {ticket_id}",
                        extra={
                            "ticket_id": ticket_id,
                            "decision": decision,
                            "booking_id": verified_booking.booking_id
                        }
                    )
                
                results["triage_decision"] = {
                    "decision": decision,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "policy_applied": policy_applied,
                    "duplicate_detection_used": False,
                    "refunded": refunded,
                    "used_verified_data": True
                }
                results["steps_completed"].append("Decision made using verified booking data")
                
            elif verification_result and not verification_result.success:
                # Verification failed - escalate
                decision = "Needs Human Review"
                reasoning = verification_result.failure_reason or "Booking verification failed"
                confidence = "low"
                policy_applied = "Booking Verification"
                refunded = False
                
                logger.info(
                    f"Escalating ticket {ticket_id} due to verification failure",
                    extra={
                        "ticket_id": ticket_id,
                        "failure_reason": verification_result.failure_reason
                    }
                )
                
                results["triage_decision"] = {
                    "decision": decision,
                    "reasoning": reasoning,
                    "confidence": confidence,
                    "policy_applied": policy_applied,
                    "duplicate_detection_used": False,
                    "refunded": refunded,
                    "used_verified_data": False
                }
                results["steps_completed"].append("Escalated due to verification failure")
                
            else:
                # No verification attempted - use standard triage
                from unittest.mock import Mock
                triage_context = Mock()
                
                # Include full ticket text (description + conversations) for rule matching
                full_ticket_data = results['ticket_metadata'].copy()
                full_ticket_data['description'] = notes_text  # Include all notes for keyword matching
                
                triage_context.inputs = {
                    'ticket_data': full_ticket_data,
                    'booking_info': results['booking_info'].get('booking_info'),
                    'refund_policy': "Standard refund policy applies"  # Would come from retriever
                }
                triage_context.agent_id = context.agent_id
                triage_context.customer_id = context.customer_id
                triage_context.session_id = context.session_id
                
                log_tool_execution(logger, ticket_id, "triage_ticket")
                triage_result = await triage_ticket(triage_context)
                results["triage_decision"] = triage_result.data
                results["triage_decision"]["duplicate_detection_used"] = False
                results["triage_decision"]["refunded"] = False
                results["triage_decision"]["used_verified_data"] = False
                results["steps_completed"].append("Completed triage analysis")
                
                # Extract decision info from triage
                decision = triage_result.data.get("decision", "Needs Human Review")
                reasoning = triage_result.data.get("reasoning", "Analysis complete")
                confidence = triage_result.data.get("confidence", "N/A")
                policy_applied = triage_result.data.get("policy_applied", "Standard refund policy")
                refunded = False
        
        # Log decision outcome
        log_decision_outcome(
            logger,
            ticket_id,
            decision,
            confidence=confidence,
            reasoning=reasoning
        )
        
        # Step 8: Add note to ticket with decision (HTML formatted for Freshdesk)
        # Check security status properly
        is_safe = results['security_scan'].get('safe', True) and not results['security_scan'].get('flagged', False)
    
        # Build detailed reasoning based on what was found
        detailed_reasoning = reasoning
        if decision == "Needs Human Review":
            reasons = []
            if results['booking_info_source'] == 'missing':
                reasons.append("No booking information found in ticket notes or description")
            if not is_safe:
                reasons.append("Security scan flagged content for review")
            if confidence == 'pending' or confidence == 'low':
                reasons.append("Insufficient data to make automated decision")
            
            if reasons:
                detailed_reasoning += "<br><br><strong>Specific reasons for human review:</strong><ul>"
                for reason in reasons:
                    detailed_reasoning += f"<li>{reason}</li>"
                detailed_reasoning += "</ul>"
        
        # Use 👾 emoji for automated analysis
        note_text = "<h3>👾 Automated Analysis Complete</h3>"
        note_text += f"<p><strong>Decision:</strong> {decision}</p>"
        note_text += f"<p><strong>Analysis:</strong> {detailed_reasoning}</p>"
        note_text += f"<p><strong>Security Status:</strong> {'✅ Safe' if is_safe else '⚠️ Flagged'}</p>"
        
        # Add booking info status with verification details
        if results['booking_info_source'] == 'parkwhiz_api_verified':
            note_text += f"<p><strong>Booking Info:</strong> ✅ Verified via ParkWhiz API</p>"
            if verified_booking:
                note_text += f"<p><strong>Verified Booking ID:</strong> {verified_booking.booking_id}</p>"
                note_text += f"<p><strong>Pass Usage:</strong> {verified_booking.pass_usage_status.upper()}</p>"
        elif results['booking_info_source'] == 'ticket_notes':
            note_text += f"<p><strong>Booking Info:</strong> ✅ Found in ticket notes</p>"
        else:
            note_text += f"<p><strong>Booking Info:</strong> ❌ Not found in ticket</p>"
        
        note_text += f"<p><strong>Policy Applied:</strong> {policy_applied}</p>"
        note_text += f"<p><strong>Confidence:</strong> {confidence}</p>"
        note_text += f"<p><strong>Refunded:</strong> {'Yes' if refunded else 'No'}</p>"
        note_text += f"<hr><p><em>Automated by Whiz AI Agent</em></p>"
        note_text += f"<p><a href='https://forms.gle/NdDu8GKguHXXmqyYA?entry.ticket_id={ticket_id}' target='_blank'>Report incorrect decision</a></p>"
        
        from .freshdesk_tools import add_note, update_ticket
        
        log_tool_execution(logger, ticket_id, "add_note")
        note_result = await add_note(context, ticket_id, note_text)
        if "error" not in note_result.data:
            results["steps_completed"].append("Added analysis note to ticket")
        
        # Step 9: Update ticket tags based on decision
        tags = ["Processed by Whiz AI"]  # Always add this tag
        
        if decision == "Approved":
            tags.extend(["Refund Approved", "Automated Decision"])
        elif decision == "Denied":
            tags.extend(["Refund Denied", "Automated Decision"])
        else:
            tags.extend(["Needs Human Review", "Automated Analysis"])
        
        # Add "Refunded" tag if a refund was processed
        if refunded:
            tags.append("Refunded")
        
        # Add verification tags
        if results['booking_info_source'] == 'parkwhiz_api_verified':
            tags.append("ParkWhiz Verified")
        
        if verification_result and not verification_result.success:
            tags.append("Verification Failed")
        
        log_tool_execution(logger, ticket_id, "update_ticket")
        update_result = await update_ticket(context, ticket_id, tags=tags)
        if "error" not in update_result.data:
            results["steps_completed"].append("Updated ticket tags")
        
        # Calculate processing time and log journey end
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        log_journey_end(logger, ticket_id, "Automated Ticket Processing", processing_time_ms, decision)
        
        # Return only essential information with minimal debug data
        return p.ToolResult(
            {
                "ticket_id": ticket_id,
                "decision": decision,
                "reasoning": reasoning,
                "policy_applied": policy_applied,
                "confidence": confidence,
                "refunded": refunded,
                "duplicate_detection_used": results["triage_decision"].get("duplicate_detection_used", False),
                "used_verified_data": results["triage_decision"].get("used_verified_data", False),
                "recommended_action": results["triage_decision"].get("recommended_action", ""),
                "steps_completed": results["steps_completed"],
                "security_status": "safe" if (results["security_scan"].get("safe", True) and not results["security_scan"].get("flagged", False)) else "flagged",
                "booking_info_found": results["booking_info_source"] != "missing",
                "booking_info_source": results["booking_info_source"],
                "verification_attempted": verification_result is not None,
                "verification_success": verification_result.success if verification_result else False,
                "note_added": "error" not in note_result.data,
                "ticket_updated": "error" not in update_result.data,
                "processing_time_ms": processing_time_ms,
                "debug": {
                    "notes_length": results.get("debug_notes_length", 0),
                    "notes_sample": results.get("debug_notes_sample", "")[:200],  # Only 200 chars
                    "booking_found": results["booking_info"].get("found", False),
                    "security_flagged": results["security_scan"].get("flagged", False),
                    "paid_again_detected": duplicate_detection_result is not None,
                    "zapier_failure_detected": zapier_failure or invalid_booking_id,
                    "verified_booking_id": verified_booking.booking_id if verified_booking else None
                }
            },
            metadata={"summary": f"Ticket {ticket_id}: {decision} - {'Verified' if verified_booking else 'Not verified'} - {'Refunded' if refunded else 'Not refunded'} - Note added, tags updated"}
        )
    
    except Exception as e:
        # Log error with full context
        log_error_with_context(
            logger,
            e,
            ticket_id=ticket_id,
            journey_name="Automated Ticket Processing",
            tool_name="process_ticket_end_to_end"
        )
        
        # Calculate processing time
        processing_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        log_journey_end(logger, ticket_id, "Automated Ticket Processing", processing_time_ms, "ERROR")
        
        # Re-raise the exception
        raise
