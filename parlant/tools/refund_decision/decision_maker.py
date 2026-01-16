"""
DecisionMaker orchestrator for hybrid refund decision-making.

This module provides the DecisionMaker class that orchestrates the complete
decision-making process by integrating PolicyLoader, BookingExtractor, RuleEngine,
LLMAnalyzer, and CancellationReasonMapper components.
"""

import time
import logging
from typing import Dict, Optional
from .policy_loader import PolicyLoader
from ..booking_analysis.extractor import BookingExtractor
from .rule_engine import RuleEngine
from ..booking_analysis.llm_analyzer import LLMAnalyzer
from ..utils.cancellation_mapper import CancellationReasonMapper
from .escalation_enhancer import EscalationEnhancer
from ..booking_analysis.timeline_analyzer import TimelineAnalyzer, TimelineExtractor
from .scenario_classifier import ScenarioClassifier
from .contradiction_detector import ContradictionDetector
from .webhook_mode_manager import WebhookModeManager

# Configure logger
logger = logging.getLogger(__name__)


class DecisionMaker:
    """
    Orchestrates the hybrid refund decision-making process.
    
    This class integrates all decision-making components to provide a complete
    workflow: extract booking info → apply rules → use LLM if needed → map
    cancellation reason. It tracks processing time and method used for monitoring.
    """
    
    def __init__(self, webhook_config: Optional[Dict] = None):
        """Initialize the DecisionMaker with all required components."""
        # Store webhook config for processing mode detection
        self.webhook_config = webhook_config or {}
        
        # Initialize components
        self.policy_loader = PolicyLoader()
        self.booking_extractor = BookingExtractor()
        self.rule_engine = RuleEngine(self.policy_loader.get_rules(), webhook_config=webhook_config)
        self.llm_analyzer = LLMAnalyzer()
        self.cancellation_reason_mapper = CancellationReasonMapper()
        self.escalation_enhancer = EscalationEnhancer()
        
        # Initialize enhanced classification components
        try:
            self.timeline_analyzer = TimelineAnalyzer(webhook_config)
            self.timeline_extractor = TimelineExtractor()
            logger.info("Timeline analysis components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize timeline analysis components: {e}")
            self.timeline_analyzer = None
            self.timeline_extractor = None
        
        try:
            # Configure scenario classifier with webhook mode settings
            scenario_config = {
                "webhook_mode": self.webhook_config.get("webhook_mode", False),
                "confidence_thresholds": {
                    "webhook_automation": self.webhook_config.get("webhook_confidence_threshold", 0.85),
                    "interactive_chat": self.webhook_config.get("chat_confidence_threshold", 0.70)
                }
            }
            self.scenario_classifier = ScenarioClassifier(scenario_config)
            logger.info("ScenarioClassifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ScenarioClassifier: {e}")
            self.scenario_classifier = None
        
        try:
            self.contradiction_detector = ContradictionDetector(webhook_config)
            logger.info("ContradictionDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContradictionDetector: {e}")
            self.contradiction_detector = None
        
        try:
            self.webhook_mode_manager = WebhookModeManager(webhook_config)
            logger.info("WebhookModeManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebhookModeManager: {e}")
            self.webhook_mode_manager = None
    
    def _calculate_days_before_event(self, cancellation_date: Optional[str], event_date: str) -> Optional[int]:
        """
        Calculate days before event (reuses logic from RuleEngine).
        
        Args:
            cancellation_date: ISO format date or None (uses current date)
            event_date: ISO format date
        
        Returns:
            Number of days before event, or None if calculation fails
        """
        from datetime import datetime, timezone
        
        try:
            # Parse event date
            event_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            event_date_only = event_dt.date()
            
            # Parse or use current date for cancellation
            if cancellation_date:
                cancel_dt = datetime.fromisoformat(cancellation_date.replace('Z', '+00:00'))
                if cancel_dt.tzinfo is None:
                    cancel_dt = cancel_dt.replace(tzinfo=timezone.utc)
                cancel_date_only = cancel_dt.date()
            else:
                cancel_date_only = datetime.now(timezone.utc).date()
            
            # Calculate difference
            return (event_date_only - cancel_date_only).days
        except (ValueError, AttributeError) as e:
            logger.error(f"Date calculation error: {e}")
            return None
    
    async def make_decision(
        self,
        ticket_data: Dict,
        ticket_notes: Optional[str] = None,
        booking_info: Optional[Dict] = None,
        booking_extraction_result: Optional[Dict] = None
    ) -> Dict:
        print(f"!!! MAKE_DECISION CALLED !!! ticket={ticket_data.get('id', ticket_data.get('ticket_id', 'unknown'))}")
        """
        Make a refund decision using hybrid approach.
        
        This method orchestrates the complete decision flow:
        1. Extract booking info (if not provided)
        2. Validate booking info completeness
        3. Apply rule-based logic
        4. Use LLM analysis if rules are uncertain
        5. Map to ParkWhiz cancellation reason (if Approved)
        6. Return final decision with metadata
        
        Args:
            ticket_data: Dictionary containing ticket information:
                - ticket_id (str): Freshdesk ticket ID
                - subject (str): Ticket subject
                - description (str): Ticket description
                - status (str): Current ticket status
            ticket_notes: Optional raw text from ticket notes/conversations
            booking_info: Optional pre-extracted booking information
        
        Returns:
            Dictionary containing:
                - decision (str): "Approved", "Denied", or "Needs Human Review"
                - reasoning (str): Human-readable explanation
                - policy_applied (str): Specific policy rule applied
                - confidence (str): "high", "medium", or "low"
                - cancellation_reason (str): ParkWhiz cancellation reason (if Approved)
                - booking_info_found (bool): Whether booking info was extracted
                - method_used (str): "rules", "llm", or "hybrid"
                - processing_time_ms (int): Time taken to make decision
        """
        start_time = time.time()
        
        ticket_id = ticket_data.get('ticket_id', ticket_data.get('id', 'unknown'))
        logger.info(f"=== DECISION_MAKER CALLED === ticket_id={ticket_id}")
        logger.info(f"Starting decision-making process for ticket: {ticket_id}")
        
        # Step 1: Check if we have pre-extracted booking info with booking_not_found flag
        if booking_extraction_result:
            logger.info("Using pre-extracted booking info from automation engine")
            booking_info_found = booking_extraction_result.get("found", False)
            
            # Check if booking was not found in ParkWhiz system
            # BUT only if we actually extracted a booking ID to check
            if booking_extraction_result.get("booking_not_found") and booking_info and booking_info.get("booking_id"):
                processing_time_ms = int((time.time() - start_time) * 1000)
                logger.warning(f"Booking not found in ParkWhiz system (from pre-extraction). Processing time: {processing_time_ms}ms")
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        "The booking ID provided by the customer was not found in the ParkWhiz system. "
                        "Cannot validate the customer's claim without booking information.<br><br>"
                        "<strong>Action required:</strong> Verify the booking details with the customer."
                    ),
                    "policy_applied": "Data Validation - Booking Not Found in System",
                    "confidence": "high",
                    "confidence_description": self._get_confidence_description("high"),
                    "cancellation_reason": None,
                    "booking_info_found": False,
                    "method_used": "booking_not_found",
                    "processing_time_ms": processing_time_ms
                }
            
            # NEW: Check if booking mismatch was detected
            # This handles cases where booking exists but customer details don't match
            if booking_extraction_result.get("booking_mismatch") and booking_info and booking_info.get("booking_id"):
                processing_time_ms = int((time.time() - start_time) * 1000)
                mismatch_details = booking_extraction_result.get("booking_mismatch_details", {})
                mismatch_summary = mismatch_details.get("summary", "Booking details don't match system records.")
                
                logger.warning(f"Booking mismatch detected (from pre-extraction). Processing time: {processing_time_ms}ms")
                
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        f"<strong>BOOKING MISMATCH DETECTED</strong><br><br>"
                        f"The booking ID exists in the system, but the customer's details don't match "
                        f"the actual booking information. This suggests the customer may have provided "
                        f"incorrect booking details.<br><br>"
                        f"<strong>Action required:</strong> Contact the customer to verify they have "
                        f"provided the correct booking ID and details."
                    ),
                    "policy_applied": "Data Validation - Booking Information Mismatch",
                    "confidence": "high",
                    "confidence_description": self._get_confidence_description("high"),
                    "cancellation_reason": None,
                    "booking_info_found": True,  # Booking was found, but mismatched
                    "method_used": "booking_mismatch",
                    "processing_time_ms": processing_time_ms,
                    "booking_mismatch_details": mismatch_details
                }
            
            # If booking info not found or has low confidence, escalate
            if not booking_info_found or booking_extraction_result.get("confidence") == "low":
                processing_time_ms = int((time.time() - start_time) * 1000)
                error_detail = booking_extraction_result.get("error", "")
                reasoning = (
                    "Unable to extract complete booking information from ticket. "
                    "Missing critical details like booking ID or event date. "
                    "Human review required to gather necessary information."
                )
                if error_detail:
                    reasoning += f" (Extraction error: {error_detail})"
                
                logger.warning(f"Escalating due to incomplete booking info (pre-extracted). Processing time: {processing_time_ms}ms")
                return {
                    "decision": "Needs Human Review",
                    "reasoning": reasoning,
                    "policy_applied": "Data Validation - Incomplete Information",
                    "confidence": "low",
                    "confidence_description": self._get_confidence_description("low"),
                    "cancellation_reason": None,
                    "booking_info_found": False,
                    "method_used": "extraction_failed",
                    "processing_time_ms": processing_time_ms
                }
        
        # Step 2: Extract booking info if not provided
        elif not booking_info:
            logger.info("Booking info not provided, extracting from ticket notes")
            if not ticket_notes:
                # Try to extract from ticket description
                ticket_notes = ticket_data.get("description", "")
                logger.debug("Using ticket description for extraction")
            
            try:
                extraction_result = await self.booking_extractor.extract_booking_info(ticket_notes)
                booking_info = extraction_result.get("booking_info", {})
                booking_info_found = extraction_result.get("found", False)
                
                # Check if extraction had an error
                if extraction_result.get("error"):
                    logger.error(f"Booking extraction error: {extraction_result['error']}")
                
                # Check if booking was not found in ParkWhiz system
                # BUT only if we actually extracted a booking ID to check
                if extraction_result.get("booking_not_found") and booking_info.get("booking_id"):
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    logger.warning(f"Booking not found in ParkWhiz system. Processing time: {processing_time_ms}ms")
                    return {
                        "decision": "Needs Human Review",
                        "reasoning": (
                            "The booking ID provided by the customer was not found in the ParkWhiz system. "
                            "Cannot validate the customer's claim without booking information.<br><br>"
                            "<strong>Action required:</strong> Verify the booking details with the customer."
                        ),
                        "policy_applied": "Data Validation - Booking Not Found in System",
                        "confidence": "high",
                        "confidence_description": self._get_confidence_description("high"),
                        "cancellation_reason": None,
                        "booking_info_found": False,
                        "method_used": "booking_not_found",
                        "processing_time_ms": processing_time_ms
                    }
                
                # NEW: Check if booking mismatch was detected
                # This handles cases where booking exists but customer details don't match
                if extraction_result.get("booking_mismatch") and booking_info.get("booking_id"):
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    mismatch_details = extraction_result.get("booking_mismatch_details", {})
                    mismatch_summary = mismatch_details.get("summary", "Booking details don't match system records.")
                    
                    logger.warning(f"Booking mismatch detected. Processing time: {processing_time_ms}ms")
                    
                    return {
                        "decision": "Needs Human Review",
                        "reasoning": (
                            f"<strong>BOOKING MISMATCH DETECTED</strong><br><br>"
                            f"The booking ID exists in the system, but the customer's details don't match "
                            f"the actual booking information. This suggests the customer may have provided "
                            f"incorrect booking details.<br><br>"
                            f"<strong>Action required:</strong> Contact the customer to verify they have "
                            f"provided the correct booking ID and details."
                        ),
                        "policy_applied": "Data Validation - Booking Information Mismatch",
                        "confidence": "high",
                        "confidence_description": self._get_confidence_description("high"),
                        "cancellation_reason": None,
                        "booking_info_found": True,  # Booking was found, but mismatched
                        "method_used": "booking_mismatch",
                        "processing_time_ms": processing_time_ms,
                        "booking_mismatch_details": mismatch_details
                    }
                
                # If booking info not found or has low confidence, escalate
                if not booking_info_found or extraction_result.get("confidence") == "low":
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    error_detail = extraction_result.get("error", "")
                    reasoning = (
                        "Unable to extract complete booking information from ticket. "
                        "Missing critical details like booking ID or event date. "
                        "Human review required to gather necessary information."
                    )
                    if error_detail:
                        reasoning += f" (Extraction error: {error_detail})"
                    
                    logger.warning(f"Escalating due to incomplete booking info. Processing time: {processing_time_ms}ms")
                    return {
                        "decision": "Needs Human Review",
                        "reasoning": reasoning,
                        "policy_applied": "Data Validation - Incomplete Information",
                        "confidence": "low",
                        "confidence_description": self._get_confidence_description("low"),
                        "cancellation_reason": None,
                        "booking_info_found": False,
                        "method_used": "extraction_failed",
                        "processing_time_ms": processing_time_ms
                    }
            
            except Exception as e:
                # Handle unexpected errors in extraction
                logger.error(f"Unexpected error during booking extraction: {type(e).__name__}: {e}")
                processing_time_ms = int((time.time() - start_time) * 1000)
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        "Technical error occurred while extracting booking information. "
                        "Human review required to process this ticket. "
                        f"Error: {str(e)}"
                    ),
                    "policy_applied": "Technical Error - Extraction Failed",
                    "confidence": "low",
                    "confidence_description": self._get_confidence_description("low"),
                    "cancellation_reason": None,
                    "booking_info_found": False,
                    "method_used": "extraction_error",
                    "processing_time_ms": processing_time_ms
                }
        else:
            logger.info("Using pre-extracted booking info (legacy path)")
            booking_info_found = True
        
        # Step 2: Validate critical fields
        if not booking_info.get("event_date"):
            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.warning(f"Missing event date, escalating. Processing time: {processing_time_ms}ms")
            return {
                "decision": "Needs Human Review",
                "reasoning": (
                    "Missing event date - cannot evaluate refund eligibility without "
                    "knowing when the parking was scheduled. Human review required."
                ),
                "policy_applied": "Data Validation - Missing Event Date",
                "confidence": "low",
                "confidence_description": self._get_confidence_description("low"),
                "cancellation_reason": None,
                "booking_info_found": booking_info_found,
                "method_used": "validation_failed",
                "processing_time_ms": processing_time_ms
            }
        
        # Step 3: Enhanced timeline analysis and scenario classification (NEW)
        timeline_analysis = None
        scenario_classification = None
        
        # Extract timeline data if components are available
        if self.timeline_analyzer and self.timeline_extractor:
            try:
                logger.info("Extracting timeline data from ticket notes")
                timeline_data = self.timeline_extractor.extract_timeline_data(ticket_notes or "")
                
                # Get booking window from booking_info if available
                booking_start = None
                booking_end = None
                
                if booking_info.get("event_date"):
                    try:
                        from datetime import datetime, timedelta
                        event_date_str = booking_info["event_date"]
                        if isinstance(event_date_str, str):
                            booking_start = datetime.fromisoformat(event_date_str.replace('Z', '+00:00'))
                            # Assume 2-hour default booking window if no end time specified
                            booking_end = booking_start + timedelta(hours=2)
                            logger.debug(f"Derived booking window from event_date: {booking_start} to {booking_end}")
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Could not parse event_date for booking window: {e}")
                
                # Merge timeline data with booking info
                final_booking_start = timeline_data.get('booking_start') or booking_start
                final_booking_end = timeline_data.get('booking_end') or booking_end
                arrival_time = timeline_data.get('arrival_time')
                exit_time = timeline_data.get('exit_time')
                
                # Perform timeline analysis if we have sufficient data
                if any([final_booking_start, final_booking_end, arrival_time, exit_time]):
                    timeline_analysis = self.timeline_analyzer.analyze_timeline(
                        booking_start=final_booking_start,
                        booking_end=final_booking_end,
                        arrival_time=arrival_time,
                        exit_time=exit_time
                    )
                    logger.info(f"Timeline analysis complete: early_arrival={timeline_analysis.has_early_arrival}, "
                               f"overstay={timeline_analysis.has_overstay}, "
                               f"consistent={timeline_analysis.is_timeline_consistent}")
                else:
                    logger.info("Insufficient timeline data for analysis")
                    
            except Exception as e:
                logger.error(f"Timeline analysis failed: {type(e).__name__}: {e}")
                # Continue with normal processing
        
        # Perform scenario classification if components are available
        if self.scenario_classifier:
            try:
                logger.info("Running enhanced scenario classification")
                
                # Get customer complaint from ticket
                customer_complaint = ticket_data.get("description", "")
                if ticket_notes:
                    customer_complaint = f"{customer_complaint}\n{ticket_notes}"
                
                # Set webhook mode based on current processing context
                if self.webhook_mode_manager:
                    processing_mode = self.webhook_mode_manager.current_mode
                    if hasattr(self.scenario_classifier, 'webhook_mode'):
                        self.scenario_classifier.webhook_mode = (processing_mode == "webhook_automation")
                
                scenario_classification = self.scenario_classifier.classify_paid_again_scenario(
                    customer_complaint=customer_complaint,
                    timeline_analysis=timeline_analysis,
                    booking_info=booking_info
                )
                
                logger.info(f"Scenario classification: {scenario_classification.scenario_type} "
                           f"(confidence: {scenario_classification.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Scenario classification failed: {type(e).__name__}: {e}")
                # Continue with normal processing
        
        # Step 4: Apply rule-based logic (integrates timeline and scenario classification)
        logger.info("Applying rule-based decision logic")
        try:
            rule_result = await self.rule_engine.apply_rules(booking_info, ticket_data, ticket_notes)
            logger.info(f"=== RULE ENGINE RESULT === decision={rule_result.get('decision')}, confidence={rule_result.get('confidence')}, policy={rule_result.get('policy_rule')}")
            logger.info(f"Rule-based result: {rule_result.get('decision')} "
                       f"(confidence: {rule_result.get('confidence')})")
            
            # Add timeline and scenario classification to rule result if available
            if timeline_analysis and 'timeline_analysis' not in rule_result:
                rule_result['timeline_analysis'] = timeline_analysis
            if scenario_classification and 'scenario_classification' not in rule_result:
                rule_result['scenario_classification'] = scenario_classification.scenario_type
                rule_result['scenario_confidence'] = scenario_classification.confidence
                
        except Exception as e:
            # Handle unexpected errors in rule engine
            logger.error(f"Error in rule engine: {type(e).__name__}: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            return {
                "decision": "Needs Human Review",
                "reasoning": (
                    "Technical error occurred while applying refund rules. "
                    "Human review required to process this ticket. "
                    f"Error: {str(e)}"
                ),
                "policy_applied": "Technical Error - Rule Engine Failed",
                "confidence": "low",
                "confidence_description": self._get_confidence_description("low"),
                "cancellation_reason": None,
                "booking_info_found": booking_info_found,
                "method_used": "rule_error",
                "processing_time_ms": processing_time_ms,
                "timeline_analysis": timeline_analysis,
                "scenario_classification": scenario_classification.scenario_type if scenario_classification else None
            }
        
        # Step 4: Check refund amount limit ($50 for automation)
        # ALL automated approvals are limited to $50 or less, regardless of timing
        # This check now handles multiple passes by detecting patterns in ticket text
        print(f"!!! AMOUNT CHECK !!! rule_decision={rule_result.get('decision')}, booking_amount={booking_info.get('amount', 0)}")
        if rule_result.get("decision") == "Approved":
            single_booking_amount = booking_info.get("amount", 0)
            print(f"!!! INSIDE AMOUNT CHECK !!! amount={single_booking_amount}")
            
            # Detect multiple passes from ticket text
            import re
            combined_text = f"{ticket_data.get('description', '')} {ticket_notes or ''}".lower()
            
            multiple_pass_patterns = [
                r'(\d+)\s*(?:vehicle\s*)?pass(?:es)?',  # "2 passes", "2 vehicle passes"
                r'two\s+(?:different\s+)?(?:vehicle\s+)?pass(?:es)?',  # "two different vehicle passes"
                r'\$\d+(?:\.\d{2})?\s*(?:a\s*piece|each)',  # "$28 a piece", "$28 each"
            ]
            
            pass_count = 1
            for pattern in multiple_pass_patterns:
                match = re.search(pattern, combined_text)
                if match:
                    if match.groups() and match.group(1):
                        try:
                            pass_count = max(pass_count, int(match.group(1)))
                        except ValueError:
                            pass
                    elif 'two' in pattern:
                        pass_count = max(pass_count, 2)
                    break
            
            # Also check for explicit "a piece" or "each" which implies multiple
            if pass_count == 1 and ('a piece' in combined_text or 'each' in combined_text):
                if 'two' in combined_text or '2 ' in combined_text:
                    pass_count = 2
            
            total_refund_amount = single_booking_amount * pass_count
            
            logger.info(f"Checking amount limit: single_amount=${single_booking_amount}, "
                       f"pass_count={pass_count}, total=${total_refund_amount}")
            
            # Check if TOTAL amount exceeds $50 limit
            # Note: $50.00 exactly also requires human review (>= not just >)
            if total_refund_amount >= 50:
                logger.info(f"Total refund amount ${total_refund_amount} exceeds $50 automation limit, escalating to human review")
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Add scenario-specific guidance for human agents
                policy_rule = rule_result.get('policy_rule', '')
                agent_guidance = ""
                
                if "Paid Again" in policy_rule:
                    agent_guidance = (
                        "<br><br><strong>Agent Action Required:</strong><br>"
                        "1. Confirm customer paid again at the same booked location<br>"
                        "2. Ask for receipt or confirmation if available (but not required)<br>"
                        "3. If paid at booked location → Approve refund as 'Paid Again'<br>"
                        "4. If paid at wrong location → Issue OTC credit instead"
                    )
                
                amount_detail = f"${total_refund_amount:.2f}"
                if pass_count > 1:
                    amount_detail = f"${total_refund_amount:.2f} ({pass_count} passes × ${single_booking_amount:.2f})"
                
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        f"{rule_result.get('reasoning')}<br><br>"
                        f"<strong>Amount Limit Exceeded:</strong> This refund is for {amount_detail}. "
                        f"Automated approvals are limited to amounts under $50. "
                        f"A human agent must review and approve this refund amount."
                        f"{agent_guidance}"
                    ),
                    "policy_applied": f"{policy_rule} - Amount Exceeds Automation Limit",
                    "confidence": "high",
                    "confidence_description": self._get_confidence_description("high"),
                    "cancellation_reason": None,
                    "booking_info_found": booking_info_found,
                    "method_used": "amount_limit_exceeded",
                    "processing_time_ms": processing_time_ms,
                    "total_refund_amount": total_refund_amount,
                    "pass_count": pass_count
                }
        
        # Step 5: Check for "Paid Again" vs "Poor Experience" misclassification
        # If customer claims "paid again" but pass was actually used, it's poor experience
        pass_used = booking_info.get("pass_used")
        
        # Also check for common "paid again" claims that are likely poor experience
        customer_complaint = ticket_data.get("description", "").lower()
        ticket_notes = ticket_notes or ""
        combined_text = f"{customer_complaint} {ticket_notes}".lower()
        
        # Look for indicators that this might be poor experience disguised as "paid again"
        poor_experience_indicators = [
            "qr code" in combined_text and "didn't work" in combined_text,
            "no attendant" in combined_text and "pay" in combined_text,
            "scanner" in combined_text and ("broken" in combined_text or "not working" in combined_text),
            "gates" in combined_text and "not" in combined_text and "work" in combined_text
        ]
        
        has_poor_experience_indicators = any(poor_experience_indicators)
        
        if (pass_used is True or has_poor_experience_indicators) and rule_result.get("decision") == "Approved":
            # Check if this was approved as a "Paid Again" scenario
            policy_applied = rule_result.get("policy_applied", "").lower()
            reasoning = rule_result.get("reasoning", "").lower()
            
            if "paid again" in policy_applied or "paid again" in reasoning or has_poor_experience_indicators:
                logger.info("Detected likely 'Poor Experience' case misclassified as 'Paid Again' - converting to Poor Experience Credit")
                rule_result = {
                    "decision": "Poor Experience Credit",
                    "reasoning": (
                        "Customer appears to have successfully used their ParkWhiz pass to park but encountered "
                        "technical difficulties or service issues during their visit. While they may claim to have "
                        "'paid again', the presence of technical issues (QR code problems, no attendant, equipment "
                        "failures) suggests they received parking value from their original purchase but had a poor "
                        "experience. This is classified as a 'Poor Experience' rather than 'Paid Again' scenario."
                    ),
                    "policy_applied": "Poor Experience - Technical difficulties during parking experience",
                    "confidence": "high",
                    "key_factors": [
                        "Technical issues indicate poor experience rather than paid again",
                        "Customer likely received parking value from ParkWhiz purchase", 
                        "QR code/attendant issues are service problems, not pass rejection",
                        "Policy states no refund for poor experience when customer parked"
                    ]
                }
                logger.info("Converted likely Paid Again approval to Poor Experience Credit")
        
        # Step 6: Decide if LLM analysis is needed
        # Check if rule engine already used LLM (e.g., for vehicle classification)
        method_used = rule_result.get("method_used", "rules")
        final_decision = rule_result
        
        # Step 6: Check webhook mode escalation before LLM analysis
        current_mode = self.webhook_mode_manager.current_mode if self.webhook_mode_manager else "interactive_chat"
        
        # Check if webhook mode should escalate instead of using LLM
        if self.webhook_mode_manager and current_mode == "webhook_automation":
            # Prepare additional factors for escalation decision
            additional_factors = {
                "booking_amount": booking_info.get("amount", 0),
                "amount_exceeds_limit": booking_info.get("amount", 0) >= 50,
                "timeline_inconsistent": timeline_analysis and not timeline_analysis.is_timeline_consistent if timeline_analysis else False,
                "multiple_scenarios_detected": scenario_classification and len(scenario_classification.contradictions) > 0 if scenario_classification else False
            }
            
            # Get confidence from rule result or scenario classification
            confidence_score = 0.7  # Default
            if scenario_classification:
                confidence_score = scenario_classification.confidence
            elif rule_result.get("confidence_score"):
                confidence_score = float(rule_result.get("confidence_score"))
            elif rule_result.get("confidence") == "high":
                confidence_score = 0.9
            elif rule_result.get("confidence") == "medium":
                confidence_score = 0.7
            elif rule_result.get("confidence") == "low":
                confidence_score = 0.4
            
            should_escalate = self.webhook_mode_manager.should_escalate(
                confidence=confidence_score,
                processing_mode="webhook_automation",
                scenario_type=scenario_classification.scenario_type if scenario_classification else rule_result.get("scenario_type"),
                additional_factors=additional_factors
            )
            
            if should_escalate:
                escalation_reason = self.webhook_mode_manager.get_escalation_reason(
                    confidence=confidence_score,
                    processing_mode="webhook_automation", 
                    scenario_type=scenario_classification.scenario_type if scenario_classification else rule_result.get("scenario_type"),
                    additional_factors=additional_factors
                )
                
                logger.info(f"Webhook mode escalation triggered: {escalation_reason}")
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                # Enhance webhook escalation reasoning
                base_reasoning = (
                    f"{rule_result.get('reasoning', 'Rule-based analysis completed.')}<br><br>"
                    f"<strong>Webhook Automation Escalation:</strong> {escalation_reason}<br><br>"
                    f"This case requires human review due to webhook automation safety thresholds."
                )
                
                enhanced_reasoning = self._enhance_escalation_reasoning(
                    base_reasoning,
                    "webhook_escalation",
                    rule_result,
                    booking_info,
                    escalation_reason="webhook_confidence_threshold"
                )
                
                return {
                    "decision": "Needs Human Review",
                    "reasoning": enhanced_reasoning,
                    "policy_applied": f"{rule_result.get('policy_rule', 'Webhook Escalation')} - Webhook Safety Threshold",
                    "confidence": "high",  # High confidence in escalation decision
                    "confidence_description": self._get_confidence_description("high"),
                    "cancellation_reason": None,
                    "booking_info_found": booking_info_found,
                    "method_used": "webhook_escalation",
                    "processing_time_ms": processing_time_ms,
                    "timeline_analysis": timeline_analysis,
                    "scenario_classification": scenario_classification.scenario_type if scenario_classification else None
                }
        
        # If rule-based decision is uncertain or has low confidence, use LLM (chat mode) or escalate (webhook mode)
        if rule_result.get("decision") == "Uncertain" or rule_result.get("confidence") == "low":
            if current_mode == "webhook_automation":
                # Webhook mode: escalate uncertain cases instead of using LLM
                logger.info("Webhook mode: escalating uncertain/low confidence case instead of LLM analysis")
                processing_time_ms = int((time.time() - start_time) * 1000)
                
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        f"{rule_result.get('reasoning', 'Rule-based analysis uncertain.')}<br><br>"
                        f"<strong>Webhook Mode Limitation:</strong> Uncertain cases require human review "
                        f"in webhook automation mode to ensure accuracy. LLM fallback is disabled for "
                        f"autonomous processing safety."
                    ),
                    "policy_applied": f"{rule_result.get('policy_rule', 'Uncertain Case')} - Webhook Mode Escalation",
                    "confidence": "high",  # High confidence in escalation decision
                    "confidence_description": self._get_confidence_description("high"),
                    "cancellation_reason": None,
                    "booking_info_found": booking_info_found,
                    "method_used": "webhook_uncertain_escalation",
                    "processing_time_ms": processing_time_ms
                }
            else:
                # Chat mode: use LLM analysis for uncertain cases
                logger.info("Chat mode: rule-based decision uncertain or low confidence, invoking LLM analysis")
                method_used = "hybrid"
            
            try:
                # Get condensed policy text for LLM (more efficient than full policy)
                policy_text = self.policy_loader.get_condensed_policy_text()
                
                # Invoke LLM analysis
                llm_result = await self.llm_analyzer.analyze_case(
                    ticket_data=ticket_data,
                    booking_info=booking_info,
                    policy_text=policy_text,
                    rule_result=rule_result
                )
                
                # Use LLM result as final decision
                final_decision = llm_result
                logger.info(f"LLM analysis result: {llm_result.get('decision')} "
                           f"(confidence: {llm_result.get('confidence')})")
                
                # POST-LLM VALIDATION: Enforce $50 limit for LLM approvals
                # This is a hard-coded safety check to ensure the $50 limit is ALWAYS enforced,
                # regardless of what the LLM decides. The LLM prompt asks it to calculate total
                # amounts, but LLM non-determinism can cause it to miss this check.
                if final_decision.get("decision") == "Approved":
                    # Calculate total refund amount (handle multiple passes)
                    single_booking_amount = booking_info.get("amount", 0)
                    
                    # Check for multiple passes indicator in ticket text
                    combined_text = f"{ticket_data.get('description', '')} {ticket_notes or ''}".lower()
                    
                    # Detect multiple passes from common patterns
                    import re
                    multiple_pass_patterns = [
                        r'(\d+)\s*(?:vehicle\s*)?pass(?:es)?',  # "2 passes", "2 vehicle passes"
                        r'two\s+(?:different\s+)?(?:vehicle\s+)?pass(?:es)?',  # "two different vehicle passes"
                        r'\$\d+(?:\.\d{2})?\s*(?:a\s*piece|each)',  # "$28 a piece", "$28 each"
                    ]
                    
                    pass_count = 1
                    for pattern in multiple_pass_patterns:
                        match = re.search(pattern, combined_text)
                        if match:
                            if match.groups() and match.group(1):
                                try:
                                    pass_count = max(pass_count, int(match.group(1)))
                                except ValueError:
                                    pass
                            elif 'two' in pattern:
                                pass_count = max(pass_count, 2)
                            break
                    
                    # Also check for explicit "a piece" or "each" which implies multiple
                    if pass_count == 1 and ('a piece' in combined_text or 'each' in combined_text):
                        # Look for context clues about quantity
                        if 'two' in combined_text or '2 ' in combined_text:
                            pass_count = 2
                    
                    total_refund_amount = single_booking_amount * pass_count
                    
                    logger.info(f"POST-LLM $50 CHECK: single_amount=${single_booking_amount}, "
                               f"pass_count={pass_count}, total=${total_refund_amount}")
                    
                    if total_refund_amount >= 50:
                        logger.warning(f"POST-LLM OVERRIDE: Total refund ${total_refund_amount} exceeds $50 limit - escalating")
                        processing_time_ms = int((time.time() - start_time) * 1000)
                        
                        return {
                            "decision": "Needs Human Review",
                            "reasoning": (
                                f"{final_decision.get('reasoning', '')}<br><br>"
                                f"<strong>Amount Limit Exceeded (Post-LLM Validation):</strong> "
                                f"The total refund amount is ${total_refund_amount:.2f} "
                                f"({pass_count} pass{'es' if pass_count > 1 else ''} × ${single_booking_amount:.2f}). "
                                f"Automated approvals are limited to amounts under $50. "
                                f"A human agent must review and approve this refund amount."
                            ),
                            "policy_applied": f"{final_decision.get('policy_applied', 'LLM Analysis')} - Amount Exceeds Automation Limit",
                            "confidence": "high",
                            "confidence_description": self._get_confidence_description("high"),
                            "cancellation_reason": None,
                            "booking_info_found": booking_info_found,
                            "method_used": "post_llm_amount_limit_exceeded",
                            "processing_time_ms": processing_time_ms,
                            "total_refund_amount": total_refund_amount,
                            "pass_count": pass_count
                        }
                
                # Check if LLM approved a "Paid Again" that's actually poor experience
                if (pass_used is True or has_poor_experience_indicators) and final_decision.get("decision") == "Approved":
                    policy_applied = final_decision.get("policy_applied", "").lower()
                    reasoning = final_decision.get("reasoning", "").lower()
                    
                    if "paid again" in policy_applied or "paid again" in reasoning:
                        logger.info("LLM approved 'Paid Again' but indicators suggest Poor Experience - converting")
                        final_decision = {
                            "decision": "Poor Experience Credit",
                            "reasoning": (
                                "While the customer reported having to 'pay again', the presence of technical "
                                "issues (QR code problems, no attendant, equipment failures) combined with "
                                "indicators that they successfully parked suggests this is a 'Poor Experience' "
                                "case rather than a legitimate 'Paid Again' scenario. The customer received "
                                "parking value from their ParkWhiz purchase but encountered service issues."
                            ),
                            "policy_applied": "Poor Experience - Technical difficulties during parking experience",
                            "confidence": "high",
                            "key_factors": [
                                "Technical issues indicate poor experience rather than paid again",
                                "Customer likely received parking value from ParkWhiz purchase", 
                                "QR code/attendant issues are service problems, not pass rejection",
                                "Policy states no refund for poor experience when customer parked"
                            ]
                        }
                        logger.info("Converted LLM Paid Again approval to Poor Experience Credit")
            
            except Exception as e:
                # Handle unexpected errors in LLM analysis
                # Fall back to rule-based decision if available
                logger.error(f"Unexpected error in LLM analysis: {type(e).__name__}: {e}")
                
                if rule_result.get("decision") != "Uncertain":
                    # Use rule-based decision as fallback
                    logger.info("Falling back to rule-based decision")
                    final_decision = {
                        "decision": rule_result.get("decision"),
                        "reasoning": f"{rule_result.get('reasoning')} (LLM analysis failed, using rule-based decision)",
                        "policy_applied": rule_result.get("policy_rule", "Rule-Based Fallback"),
                        "confidence": "medium",  # Downgrade confidence
                        "confidence_description": self._get_confidence_description("medium"),
                        "key_factors": [
                            "LLM analysis unavailable",
                            "Using rule-based decision as fallback",
                            f"Error: {str(e)}"
                        ]
                    }
                    method_used = "rules_fallback"
                else:
                    # No fallback available, escalate
                    logger.warning("No fallback available, escalating to human review")
                    processing_time_ms = int((time.time() - start_time) * 1000)
                    return {
                        "decision": "Needs Human Review",
                        "reasoning": (
                            "Unable to complete automated analysis due to technical error. "
                            "This case requires human review to ensure accurate decision-making. "
                            f"Error: {str(e)}"
                        ),
                        "policy_applied": "Technical Error - Analysis Failed",
                        "confidence": "low",
                        "confidence_description": self._get_confidence_description("low"),
                        "cancellation_reason": None,
                        "booking_info_found": booking_info_found,
                        "method_used": "llm_error",
                        "processing_time_ms": processing_time_ms
                    }
        else:
            logger.info("Rule-based decision has sufficient confidence, skipping LLM analysis")
        
        # Step 7: Handle Poor Experience Credit decisions
        credit_amount = None
        if final_decision.get("decision") == "Poor Experience Credit":
            logger.info("Decision is Poor Experience Credit, calculating appropriate credit amount")
            
            # Calculate credit amount based on booking amount
            booking_amount = booking_info.get("amount", 0)
            
            if booking_amount < 10:
                # No credit for bookings under $10
                credit_amount = 0
                credit_message = "No credit offered due to low booking amount (under $10)."
                logger.info(f"No credit for booking amount ${booking_amount} (under $10)")
            elif booking_amount <= 20:
                # $5 credit for bookings $10-$20
                credit_amount = 5.0
                credit_message = f"Offer ${credit_amount:.0f} account credit for the inconvenience."
                logger.info(f"${credit_amount} credit for booking amount ${booking_amount} ($10-$20 range)")
            else:
                # $10 credit for bookings over $20
                credit_amount = 10.0
                credit_message = f"Offer ${credit_amount:.0f} account credit for the inconvenience."
                logger.info(f"${credit_amount} credit for booking amount ${booking_amount} (over $20)")
            
            # Convert to Denied but add credit recommendation
            final_decision["decision"] = "Denied"
            original_reasoning = final_decision.get("reasoning", "")
            final_decision["reasoning"] = (
                f"{original_reasoning}\n\n"
                f"**Recommended Action**: {credit_message} "
                f"This follows the policy for poor experience cases where the customer successfully "
                f"used their pass but encountered technical difficulties or service issues. "
                f"Credit amount is based on booking value: "
                f"No credit for <$10, $5 credit for $10-$20, $10 credit for >$20."
            )
            logger.info(f"Converted Poor Experience Credit to Denied with ${credit_amount} credit recommendation")
        
        # Step 8: Map to ParkWhiz cancellation reason (if Approved)
        cancellation_reason = None
        if final_decision.get("decision") == "Approved":
            logger.info("Decision is Approved, mapping to ParkWhiz cancellation reason")
            try:
                cancellation_reason = self.cancellation_reason_mapper.map_reason(
                    decision_reasoning=final_decision.get("reasoning", ""),
                    policy_applied=final_decision.get("policy_applied", ""),
                    booking_info=booking_info
                )
                logger.info(f"Mapped cancellation reason: {cancellation_reason}")
            except Exception as e:
                # Handle errors in cancellation reason mapping
                # Default to "Other" if mapping fails
                logger.error(f"Error mapping cancellation reason: {e}, defaulting to 'Other'")
                cancellation_reason = "Other"
        
        # Step 9: Calculate processing time and return result
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(f"Decision-making complete. Decision: {final_decision.get('decision')}, "
                   f"Method: {method_used}, Processing time: {processing_time_ms}ms")
        
        # Add confidence description
        confidence = final_decision.get("confidence")
        confidence_description = self._get_confidence_description(confidence)
        
        # Enhance escalation reasoning if this is a "Needs Human Review" decision
        reasoning = final_decision.get("reasoning", "")
        if final_decision.get("decision") == "Needs Human Review":
            # Determine escalation type based on available data
            escalation_type = "general"
            escalation_kwargs = {}
            
            # Check for timeline-based escalation
            if timeline_analysis and not timeline_analysis.is_timeline_consistent:
                escalation_type = "timeline"
                escalation_kwargs["escalation_scenario"] = "timeline_analysis_required"
            
            # Check for contradiction-based escalation
            elif final_decision.get("contradictions") or (scenario_classification and scenario_classification.contradictions):
                escalation_type = "contradiction"
                escalation_kwargs["escalation_scenario"] = "contradiction_detected"
            
            # Check for exit complication escalation
            elif "exit" in reasoning.lower() and "complication" in reasoning.lower():
                escalation_type = "exit_complication"
                escalation_kwargs["exit_complication_type"] = "system_failure"
                escalation_kwargs["customer_parked_successfully"] = booking_info.get("pass_used", False)
            
            # Enhance the reasoning
            enhanced_reasoning = self._enhance_escalation_reasoning(
                reasoning,
                escalation_type,
                final_decision,
                booking_info,
                **escalation_kwargs
            )
            
            # Update the reasoning in final_decision
            reasoning = enhanced_reasoning
        
        result = {
            "decision": final_decision.get("decision"),
            "reasoning": reasoning,
            "policy_applied": final_decision.get("policy_applied", final_decision.get("policy_rule")),
            "confidence": confidence,
            "confidence_description": confidence_description,
            "cancellation_reason": cancellation_reason,
            "booking_info_found": booking_info_found,
            "method_used": method_used,
            "processing_time_ms": processing_time_ms,
            "key_factors": final_decision.get("key_factors", [])
        }
        
        # Add enhanced classification fields (NEW)
        if timeline_analysis:
            result["timeline_analysis"] = {
                "has_early_arrival": timeline_analysis.has_early_arrival,
                "has_overstay": timeline_analysis.has_overstay,
                "early_arrival_minutes": timeline_analysis.early_arrival_minutes,
                "overstay_minutes": timeline_analysis.overstay_minutes,
                "actual_parking_duration": timeline_analysis.actual_parking_duration,
                "booked_duration": timeline_analysis.booked_duration,
                "is_timeline_consistent": timeline_analysis.is_timeline_consistent,
                "timeline_issues": timeline_analysis.timeline_issues
            }
        
        if scenario_classification:
            result["scenario_classification"] = {
                "scenario_type": scenario_classification.scenario_type,
                "confidence": scenario_classification.confidence,
                "recommended_action": scenario_classification.recommended_action,
                "refund_eligible": scenario_classification.refund_eligible,
                "key_factors": scenario_classification.key_factors
            }
        
        # Add contradictions if detected
        contradictions = final_decision.get("contradictions", [])
        if contradictions:
            result["contradictions"] = [
                {
                    "type": c.type if hasattr(c, 'type') else str(c),
                    "description": c.description if hasattr(c, 'description') else str(c),
                    "severity": c.severity if hasattr(c, 'severity') else "medium"
                } for c in contradictions
            ]
        
        # Add processing mode information
        if self.webhook_mode_manager:
            result["processing_mode"] = self.webhook_mode_manager.current_mode
            result["confidence_threshold"] = self.webhook_mode_manager.get_confidence_threshold()
        
        # Add credit amount if applicable
        if credit_amount is not None:
            result["recommended_credit"] = credit_amount
        
        return result
    
    def _get_confidence_description(self, confidence: str) -> str:
        """
        Get a descriptive explanation for the confidence level.
        
        Args:
            confidence: The confidence level ("high", "medium", "low", or other)
            
        Returns:
            Human-readable description of what the confidence level means
        """
        confidence_descriptions = {
            "high": "Clear-cut case with strong policy support",
            "medium": "Reasonable case but some ambiguity", 
            "low": "Uncertain case, borderline, or missing critical information"
        }
        
        return confidence_descriptions.get(confidence, f"Unknown confidence level: {confidence}")
    
    def set_processing_mode(self, mode: str) -> None:
        """
        Set the processing mode for webhook vs chat processing.
        
        Args:
            mode: Processing mode ("webhook_automation" or "interactive_chat")
        """
        if self.webhook_mode_manager:
            self.webhook_mode_manager.set_processing_mode(mode)
            logger.info(f"Processing mode set to: {mode}")
        elif hasattr(self.rule_engine, 'webhook_manager'):
            self.rule_engine.webhook_manager.set_processing_mode(mode)
            logger.info(f"Processing mode set to: {mode} (via rule engine)")
        else:
            logger.warning("WebhookModeManager not available - processing mode not set")
    
    def get_processing_mode(self) -> str:
        """
        Get the current processing mode.
        
        Returns:
            Current processing mode string
        """
        if self.webhook_mode_manager:
            return self.webhook_mode_manager.current_mode
        elif hasattr(self.rule_engine, 'webhook_manager'):
            return self.rule_engine.webhook_manager.current_mode
        else:
            return "interactive_chat"  # Default fallback
    
    def is_webhook_mode(self) -> bool:
        """
        Check if currently in webhook automation mode.
        
        Returns:
            True if in webhook mode, False otherwise
        """
        return self.get_processing_mode() == "webhook_automation"
    
    def _enhance_escalation_reasoning(
        self,
        original_reasoning: str,
        escalation_type: str,
        rule_result: Optional[Dict] = None,
        booking_info: Optional[Dict] = None,
        **kwargs
    ) -> str:
        """
        Enhance escalation reasoning with timeline breakdowns, contradiction examples,
        and agent guidance based on the escalation type.
        
        Args:
            original_reasoning: Original escalation reasoning
            escalation_type: Type of escalation (timeline, contradiction, exit_complication, etc.)
            rule_result: Rule engine result containing additional context
            booking_info: Booking information for context
            **kwargs: Additional context for specific escalation types
            
        Returns:
            Enhanced reasoning with appropriate documentation and guidance
        """
        try:
            enhanced_reasoning = original_reasoning
            
            # Timeline-based escalations
            if escalation_type == "timeline" and rule_result:
                timeline_analysis = rule_result.get("timeline_analysis")
                escalation_scenario = kwargs.get("escalation_scenario", "timeline_analysis_required")
                
                if timeline_analysis:
                    enhanced_reasoning = self.escalation_enhancer.enhance_timeline_escalation(
                        original_reasoning,
                        timeline_analysis,
                        escalation_scenario
                    )
            
            # Contradiction-based escalations
            elif escalation_type == "contradiction" and rule_result:
                contradictions = rule_result.get("contradictions", [])
                escalation_scenario = kwargs.get("escalation_scenario", "contradiction_detected")
                
                if contradictions:
                    enhanced_reasoning = self.escalation_enhancer.enhance_contradiction_escalation(
                        original_reasoning,
                        contradictions,
                        escalation_scenario
                    )
            
            # Exit complication escalations
            elif escalation_type == "exit_complication" and booking_info:
                exit_complication_type = kwargs.get("exit_complication_type", "system_failure")
                customer_parked_successfully = kwargs.get("customer_parked_successfully", False)
                booking_amount = booking_info.get("amount", 0)
                
                enhanced_reasoning = self.escalation_enhancer.enhance_exit_complication_escalation(
                    original_reasoning,
                    exit_complication_type,
                    customer_parked_successfully,
                    booking_amount
                )
            
            # General agent guidance enhancement
            escalation_reason = kwargs.get("escalation_reason", escalation_type)
            missing_information = kwargs.get("missing_information", [])
            suggested_questions = rule_result.get("suggested_questions") if rule_result else None
            
            enhanced_reasoning = self.escalation_enhancer.add_agent_guidance(
                enhanced_reasoning,
                escalation_reason,
                missing_information,
                suggested_questions
            )
            
            logger.info(f"Enhanced escalation reasoning for type: {escalation_type}")
            return enhanced_reasoning
            
        except Exception as e:
            logger.error(f"Failed to enhance escalation reasoning: {e}")
            # Return original reasoning if enhancement fails
            return original_reasoning
