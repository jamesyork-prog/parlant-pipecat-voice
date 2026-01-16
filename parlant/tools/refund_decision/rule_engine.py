"""
RuleEngine component for applying deterministic business rules to refund decisions.

This module provides a RuleEngine class that applies clear-cut business rules
to make refund decisions with high confidence, reducing the need for LLM analysis
in straightforward cases.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from .vehicle_classifier import VehicleClassifier
from .contradiction_detector import ContradictionDetector
from .webhook_mode_manager import WebhookModeManager
from .scenario_classifier import ScenarioClassifier
from ..booking_analysis.timeline_analyzer import TimelineAnalyzer, TimelineExtractor

# Configure logger
logger = logging.getLogger(__name__)


class RuleEngine:
    """
    Applies deterministic business rules for refund decisions.
    
    The RuleEngine evaluates booking information against predefined business rules
    to make quick, consistent decisions for clear-cut cases. When rules produce
    uncertain results, the decision is escalated to LLM analysis.
    """
    
    def __init__(self, rules: Dict, contradiction_config: Optional[Dict] = None, webhook_config: Optional[Dict] = None, timeline_config: Optional[Dict] = None):
        """
        Initialize the RuleEngine with policy rules and enhanced classification components.
        
        Args:
            rules: Dictionary containing refund policy rules from PolicyLoader
            contradiction_config: Optional configuration for contradiction detection
                                 Pass empty dict {} to disable contradiction detection
            webhook_config: Optional configuration for webhook mode management
            timeline_config: Optional configuration for timeline analysis (overstay/early arrival tolerances)
        """
        self.rules = rules
        self.initialization_errors = []
        
        # Initialize webhook mode manager
        try:
            self.webhook_manager = WebhookModeManager(webhook_config)
            logger.info("WebhookModeManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize WebhookModeManager: {e}")
            self.initialization_errors.append(f"WebhookModeManager: {e}")
            # Create default webhook manager as fallback
            self.webhook_manager = WebhookModeManager()
        
        # Initialize timeline analysis components
        try:
            self.timeline_analyzer = TimelineAnalyzer(timeline_config)
            self.timeline_extractor = TimelineExtractor()
            logger.info("Timeline analysis components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize timeline analysis components: {e}")
            self.initialization_errors.append(f"TimelineAnalyzer: {e}")
            self.timeline_analyzer = None
            self.timeline_extractor = None
        
        # Initialize scenario classifier
        try:
            # Merge webhook config with scenario classifier config
            scenario_config = {
                "webhook_mode": self.webhook_manager.is_webhook_mode() if self.webhook_manager else False,
                "confidence_thresholds": {
                    "webhook_automation": self.webhook_manager.webhook_confidence_threshold if self.webhook_manager else 0.85,
                    "interactive_chat": self.webhook_manager.chat_confidence_threshold if self.webhook_manager else 0.70
                }
            }
            if webhook_config:
                scenario_config.update(webhook_config)
            
            self.scenario_classifier = ScenarioClassifier(scenario_config)
            logger.info("ScenarioClassifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ScenarioClassifier: {e}")
            self.initialization_errors.append(f"ScenarioClassifier: {e}")
            self.scenario_classifier = None
        
        # Initialize vehicle classifier with error handling
        try:
            self.vehicle_classifier = VehicleClassifier()
            logger.info("VehicleClassifier initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VehicleClassifier: {e}")
            self.initialization_errors.append(f"VehicleClassifier: {e}")
            self.vehicle_classifier = None
        
        # Initialize contradiction detector with comprehensive fallback mechanism
        self.contradiction_detection_enabled = True
        self.contradiction_detector = None
        
        # Check if contradiction detection should be disabled
        if contradiction_config is not None and len(contradiction_config) == 0:
            # Empty config disables contradiction detection for backward compatibility
            self.contradiction_detection_enabled = False
            logger.info("Contradiction detection disabled via empty configuration")
        elif contradiction_config is not None and contradiction_config.get('disable_contradiction_detection', False):
            # Explicit disable flag
            self.contradiction_detection_enabled = False
            logger.info("Contradiction detection disabled via disable flag")
        else:
            # Try to initialize contradiction detector
            try:
                self.contradiction_detector = ContradictionDetector(contradiction_config)
                logger.info("Contradiction detection enabled and initialized successfully")
            except ImportError as e:
                logger.error(f"Failed to import contradiction detector components: {e}")
                logger.warning("Disabling contradiction detection due to import failure")
                self.contradiction_detection_enabled = False
                self.initialization_errors.append(f"ContradictionDetector import: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize contradiction detector: {e}")
                logger.warning("Disabling contradiction detection due to initialization failure")
                self.contradiction_detection_enabled = False
                self.initialization_errors.append(f"ContradictionDetector: {e}")
        
        # Log initialization status
        if self.initialization_errors:
            logger.warning(f"RuleEngine initialized with {len(self.initialization_errors)} component failures:")
            for error in self.initialization_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("RuleEngine initialized successfully with all components")
        
        # Validate critical components
        if not self.rules:
            logger.warning("No rules provided to RuleEngine - decisions may be limited")
        
        logger.info(f"RuleEngine ready (contradiction_detection: {self.contradiction_detection_enabled}, "
                   f"timeline_analysis: {self.timeline_analyzer is not None}, "
                   f"scenario_classification: {self.scenario_classifier is not None})")
    
    def set_processing_mode(self, mode: str) -> None:
        """
        Set the processing mode for webhook vs chat processing.
        
        Args:
            mode: Processing mode ("webhook_automation" or "interactive_chat")
        """
        if self.webhook_manager:
            self.webhook_manager.set_processing_mode(mode)
            logger.info(f"RuleEngine processing mode set to: {mode}")
            
            # Update scenario classifier webhook mode if available
            if self.scenario_classifier and hasattr(self.scenario_classifier, 'webhook_mode'):
                self.scenario_classifier.webhook_mode = (mode == "webhook_automation")
        else:
            logger.warning("WebhookModeManager not available - cannot set processing mode")
    
    def get_processing_mode(self) -> str:
        """
        Get the current processing mode.
        
        Returns:
            Current processing mode ("webhook_automation" or "interactive_chat")
        """
        if self.webhook_manager:
            return self.webhook_manager.current_mode
        else:
            return "interactive_chat"  # Default fallback
    
    async def apply_rules(
        self,
        booking_info: Dict,
        ticket_data: Dict,
        ticket_notes: str = ""
    ) -> Dict:
        """
        Apply business rules to make a refund decision.
        
        This method evaluates the booking information against deterministic rules
        in priority order:
        1. 7+ days before event → Approve (high confidence)
        2. After event start → Deny (high confidence)
        3. 3-7 days + confirmed booking → Approve (medium confidence)
        4. <3 days + on-demand → Deny (high confidence)
        5. Oversold location → Approve (high confidence)
        6. Duplicate pass → Approve (high confidence)
        7. Edge cases → Uncertain (triggers LLM)
        
        Args:
            booking_info: Dictionary containing booking details:
                - booking_id (str): Booking identifier
                - event_date (str): ISO format date of the event
                - booking_type (str): Type of booking (confirmed, on-demand, etc.)
                - amount (float): Booking amount
                - cancellation_date (str, optional): ISO format cancellation date
            ticket_data: Dictionary containing ticket information:
                - ticket_id (str): Freshdesk ticket ID
                - subject (str): Ticket subject
                - description (str): Ticket description
        
        Returns:
            Dictionary containing:
                - decision (str): "Approved", "Denied", or "Uncertain"
                - reasoning (str): Human-readable explanation
                - policy_rule (str): Specific policy rule applied
                - confidence (str): "high", "medium", or "low"
        """
        logger.info("Applying rule-based decision logic")
        logger.debug(f"Booking info: booking_id={booking_info.get('booking_id')}, "
                    f"event_date={booking_info.get('event_date')}, "
                    f"booking_type={booking_info.get('booking_type')}")
        
        # Validate required fields
        if not booking_info.get("event_date"):
            logger.warning("Missing event date - cannot apply rules")
            return self._create_rule_result(
                decision="Uncertain",
                reasoning="Missing event date - cannot calculate days before event",
                policy_rule="Data Validation",
                confidence="low"
            )
        
        # Calculate days before event
        event_date_raw = booking_info.get("event_date")
        cancellation_date_raw = booking_info.get("cancellation_date")
        
        logger.info(f"Date calculation inputs: event_date='{event_date_raw}', cancellation_date='{cancellation_date_raw}'")
        
        days_before_event = self._calculate_days_before_event(
            cancellation_date_raw,
            event_date_raw
        )
        
        if days_before_event is None:
            logger.error(f"Failed to calculate days before event - invalid date format. event_date='{event_date_raw}', cancellation_date='{cancellation_date_raw}'")
            return {
                "decision": "Uncertain",
                "reasoning": "Unable to calculate days before event - invalid date format",
                "policy_rule": "Data Validation",
                "confidence": "low"
            }
        
        logger.info(f"Days before event: {days_before_event}")
        
        # Extract booking details
        booking_type = booking_info.get("booking_type", "").lower()
        amount = booking_info.get("amount", 0)
        
        # Extract customer's complaint with support for contradiction analysis
        # This preserves original data while extracting complaint text for analysis
        complaint_extraction = self._extract_customer_complaint(ticket_notes)
        customer_complaint = complaint_extraction["complaint_text"]
        
        # Backward compatibility: if ticket_notes is empty, fall back to ticket description
        if not customer_complaint:
            customer_complaint = ticket_data.get("description", "").lower()
            logger.debug("ticket_notes empty, using ticket description as customer complaint")
            # Update extraction result for fallback case
            complaint_extraction = {
                "complaint_text": customer_complaint,
                "original_reason": "",
                "original_summary": "",
                "original_notes": ticket_notes or "",
                "complaint_parts": [customer_complaint] if customer_complaint else []
            }
        
        print(f"DEBUG: Customer complaint extracted: '{customer_complaint[:200] if customer_complaint else 'EMPTY'}'")
        logger.info(f"Customer complaint extracted: '{customer_complaint[:200] if customer_complaint else 'EMPTY'}'")
        ticket_description = ticket_data.get("description", "").lower()
        
        # ENHANCED PRIORITY CHECK: Timeline analysis and scenario classification first
        timeline_analysis = None
        scenario_classification = None
        
        # Step 1: Extract and analyze timeline data if components are available
        if self.timeline_analyzer and self.timeline_extractor:
            try:
                logger.info("Extracting timeline data from ticket notes")
                timeline_data = self.timeline_extractor.extract_timeline_data(ticket_notes)
                
                # Get booking window from booking_info if available
                booking_start = None
                booking_end = None
                
                if booking_info.get("event_date"):
                    try:
                        from datetime import datetime, timedelta
                        event_date_str = booking_info["event_date"]
                        if isinstance(event_date_str, str):
                            booking_start = datetime.fromisoformat(event_date_str)
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
        
        # Step 2: Perform scenario classification if components are available
        if self.scenario_classifier:
            try:
                logger.info("Running enhanced scenario classification")
                
                # Set webhook mode based on current processing context
                processing_mode = self.webhook_manager.current_mode if self.webhook_manager else "interactive_chat"
                if hasattr(self.scenario_classifier, 'webhook_mode'):
                    self.scenario_classifier.webhook_mode = (processing_mode == "webhook_automation")
                
                scenario_classification = self.scenario_classifier.classify_paid_again_scenario(
                    customer_complaint=customer_complaint,
                    timeline_analysis=timeline_analysis,
                    booking_info=booking_info
                )
                
                logger.info(f"Scenario classification: {scenario_classification.scenario_type} "
                           f"(confidence: {scenario_classification.confidence:.2f})")
                
                # Check if scenario classification should override normal rule processing
                # Only escalate for meaningful scenarios, not insufficient data
                if (scenario_classification.recommended_action == "escalate" and 
                    scenario_classification.scenario_type not in ["insufficient_data", "uncertain_requires_analysis"]):
                    logger.info("Scenario classification recommends escalation - bypassing normal rules")
                    return self._create_rule_result(
                        decision="Needs Human Review",
                        reasoning=scenario_classification.reasoning,
                        policy_rule=f"Enhanced Classification - {self._format_scenario_type(scenario_classification.scenario_type)}",
                        confidence="high",
                        scenario_classification=scenario_classification.scenario_type,
                        timeline_analysis=timeline_analysis,
                        contradictions=scenario_classification.contradictions
                    )
                
                # Check webhook mode escalation thresholds ONLY if in webhook mode
                # Skip escalation for insufficient data scenarios - let original rules handle them
                if (self.webhook_manager and 
                    self.webhook_manager.is_webhook_mode() and 
                    scenario_classification.scenario_type not in ["insufficient_data", "uncertain_requires_analysis"] and  # Don't escalate for insufficient data or uncertain cases with low confidence
                    scenario_classification.confidence >= 0.5 and  # Only escalate if we have some meaningful confidence
                    self.webhook_manager.should_escalate(
                        confidence=scenario_classification.confidence,
                        scenario_type=scenario_classification.scenario_type,
                        additional_factors={
                            "timeline_inconsistent": timeline_analysis and not timeline_analysis.is_timeline_consistent,
                            "multiple_scenarios_detected": len(scenario_classification.contradictions) > 0
                        }
                    )):
                    escalation_reason = self.webhook_manager.get_escalation_reason(
                        confidence=scenario_classification.confidence,
                        scenario_type=scenario_classification.scenario_type,
                        additional_factors={
                            "timeline_inconsistent": timeline_analysis and not timeline_analysis.is_timeline_consistent
                        }
                    )
                    
                    logger.info(f"Webhook mode escalation triggered: {escalation_reason}")
                    return self._create_rule_result(
                        decision="Needs Human Review",
                        reasoning=f"Escalated due to webhook automation requirements: {escalation_reason}",
                        policy_rule="Webhook Mode Escalation",
                        confidence="high",
                        scenario_classification=scenario_classification.scenario_type,
                        timeline_analysis=timeline_analysis,
                        escalation_reason=escalation_reason
                    )
                
                # If not in webhook mode or escalation not required, log the scenario classification
                # but continue with original rule processing
                logger.info(f"Scenario classification complete ({scenario_classification.scenario_type}, "
                           f"confidence: {scenario_classification.confidence:.2f}), continuing with original rules")
                
                # Handle high-confidence scenario classifications that can make direct decisions
                # Only override original rules for very high confidence (0.9+) or specific scenario types
                if (scenario_classification.confidence >= 0.9 and 
                    scenario_classification.recommended_action in ["approve_refund", "deny_legitimate_charge"]):
                    
                    decision = "Approved" if scenario_classification.recommended_action == "approve_refund" else "Denied"
                    logger.info(f"High-confidence scenario classification decision: {decision}")
                    
                    return self._create_rule_result(
                        decision=decision,
                        reasoning=scenario_classification.reasoning,
                        policy_rule=f"Enhanced Classification - {self._format_scenario_type(scenario_classification.scenario_type)}",
                        confidence="high",
                        scenario_classification=scenario_classification.scenario_type,
                        timeline_analysis=timeline_analysis,
                        key_factors=scenario_classification.key_factors
                    )
                
                # Handle specific high-priority scenario types that should override rules
                elif (scenario_classification.scenario_type in ["early_arrival_overstay", "accidental_double_booking", 
                                                              "entry_failure_complete_unused", "exit_complication_poor_experience"] and
                      scenario_classification.confidence >= 0.85):
                    
                    decision = "Approved" if scenario_classification.recommended_action == "approve_refund" else "Denied"
                    logger.info(f"High-priority scenario classification decision: {decision} for {scenario_classification.scenario_type}")
                    
                    return self._create_rule_result(
                        decision=decision,
                        reasoning=scenario_classification.reasoning,
                        policy_rule=f"Enhanced Classification - {self._format_scenario_type(scenario_classification.scenario_type)}",
                        confidence="high",
                        scenario_classification=scenario_classification.scenario_type,
                        timeline_analysis=timeline_analysis,
                        key_factors=scenario_classification.key_factors
                    )
                
            except Exception as e:
                logger.error(f"Scenario classification failed: {type(e).__name__}: {e}")
                # Continue with normal processing
        
        # Step 3: Original contradiction detection (if enabled and no scenario classification override)
        if self.contradiction_detection_enabled and self.contradiction_detector is not None:
            logger.info("Running original contradiction detection as fallback check")
            
            # Set timeout for contradiction detection to prevent blocking
            contradiction_timeout = 2  # 2 seconds timeout for performance (reduced from 15s)
            
            try:
                import asyncio
                
                # Pass enhanced complaint extraction data to contradiction detector
                enhanced_ticket_data = {
                    **ticket_data,
                    "complaint_extraction": complaint_extraction
                }
                
                # Run contradiction detection with timeout
                contradiction_result = await asyncio.wait_for(
                    self.contradiction_detector.detect_contradictions(
                        customer_complaint, 
                        enhanced_ticket_data
                    ),
                    timeout=contradiction_timeout
                )
                
                # If contradictions are detected, escalate immediately (bypasses all other rules)
                if contradiction_result.has_contradictions and contradiction_result.escalation_required:
                    logger.info(f"Contradictions detected: {len(contradiction_result.contradictions)} conflicts found")
                    return self._create_rule_result(
                        decision="Needs Human Review",
                        reasoning=contradiction_result.reasoning,
                        policy_rule="Contradiction Detection - Requires Human Review",
                        confidence="high",
                        contradictions=contradiction_result.contradictions,
                        suggested_questions=contradiction_result.suggested_questions,
                        timeline_analysis=timeline_analysis
                    )
                else:
                    logger.info("No contradictions detected - proceeding with normal rule processing")
            
            except asyncio.TimeoutError:
                logger.error(f"Contradiction detection timed out after {contradiction_timeout} seconds")
                # Continue with normal processing but note the timeout
                logger.warning("Continuing with normal rule processing due to contradiction detection timeout")
                # Could optionally escalate here for safety, but continuing allows normal processing
            
            except Exception as e:
                logger.error(f"Contradiction detection failed with error: {e}")
                # Continue with normal processing but log the failure
                logger.warning("Continuing with normal rule processing despite contradiction detection failure")
                
                # Log the error for monitoring
                try:
                    from datetime import datetime
                    error_context = {
                        "timestamp": datetime.now().isoformat(),
                        "ticket_id": ticket_data.get("ticket_id", "unknown"),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "customer_complaint_length": len(customer_complaint) if customer_complaint else 0,
                        "fallback_action": "continue_normal_processing"
                    }
                    logger.error(f"Contradiction detection error context: {error_context}")
                except Exception as logging_error:
                    logger.error(f"Failed to log contradiction detection error context: {logging_error}")
        else:
            if not self.contradiction_detection_enabled:
                logger.info("Contradiction detection disabled - proceeding with normal rule processing")
            else:
                logger.warning("Contradiction detector not available - proceeding with normal rule processing")
        
        # PRIORITY RULE 0: Check for reseller or non-refundable flags
        # These are hard stops - no refunds under any circumstances
        is_reseller = booking_info.get("is_reseller", False)
        is_non_refundable = booking_info.get("is_non_refundable", False)
        
        if is_reseller:
            logger.info("User flagged as reseller - automatic denial per policy")
            return {
                "decision": "Denied",
                "reasoning": (
                    "This user is flagged as a reseller. Per company policy, reseller passes are non-refundable. "
                    "Resellers purchase passes in bulk for resale and are not eligible for refunds."
                ),
                "policy_rule": "Non-Refundable Categories - Reseller Passes",
                "confidence": "high"
            }
        
        if is_non_refundable:
            logger.info("User flagged as non-refundable - automatic denial")
            return {
                "decision": "Denied",
                "reasoning": (
                    "This user is flagged as non-refundable in the system. "
                    "This flag is typically applied to users who have abused the refund policy."
                ),
                "policy_rule": "Non-Refundable User Flag",
                "confidence": "high"
            }
        
        # PRIORITY RULE 0: Check for duplicate claims FIRST (must escalate regardless of timing or scenario classification)
        # This overrides ALL other rules because we cannot auto-detect duplicates due to API limitations
        if self._check_for_duplicate_claim(ticket_data):
            logger.info("Rule matched: Duplicate Claim. Decision: Needs Human Review (API limitation)")
            return {
                "decision": "Needs Human Review",
                "reasoning": (
                    "Customer reports duplicate booking or being charged twice.<br><br>"
                    "Duplicate detection requires manual review because the ParkWhiz API "
                    "does not support searching bookings by customer email.<br><br>"
                    "<strong>Action required:</strong> A specialist will review the customer's account to locate both bookings."
                ),
                "policy_rule": "Duplicate Booking Claim - Requires Manual Review",
                "confidence": "high"
            }
        
        # PRIORITY RULE 2: Check for "Paid Again" scenarios BEFORE cancellation rules
        # These are not cancellations - they are refund requests for double payment
        if self._check_for_paid_again(customer_complaint):
            logger.info("Rule matched: Paid Again (priority check). Decision: Approved")
            
            # Enhanced reasoning with timeline analysis if available
            reasoning = "Customer had to pay again on-site despite having a valid booking."
            if timeline_analysis and timeline_analysis.is_timeline_consistent:
                if not timeline_analysis.has_early_arrival and not timeline_analysis.has_overstay:
                    duration_formatted = self._format_duration(timeline_analysis.actual_parking_duration)
                    reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer parked within their booking window " \
                               f"({duration_formatted}), confirming this was a system failure or duplicate payment scenario."
                else:
                    early_formatted = self._format_duration(timeline_analysis.early_arrival_minutes)
                    overstay_formatted = self._format_duration(timeline_analysis.overstay_minutes)
                    reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer had early arrival ({early_formatted}) " \
                               f"or overstay ({overstay_formatted}), but still paid again on-site, indicating system issues."
            
            return {
                "decision": "Approved",
                "reasoning": reasoning,
                "policy_rule": "Paid Again",
                "confidence": "high"
            }
        
        # PRIORITY RULE 3: Check for "could not find location" - customer navigation issue
        # This is typically a customer issue, not a location issue, but needs human review
        # to determine if signage was adequate or if there was a legitimate access problem
        if self._check_for_location_finding_issue(customer_complaint):
            logger.info("Customer reports 'could not find location' - escalating for signage/access review")
            return self._create_rule_result(
                decision="Needs Human Review",
                reasoning=(
                    "Customer reports being unable to find the parking location. "
                    "This requires human review to determine if:<br>"
                    "<ul>"
                    "<li>Signage was adequate and customer simply had navigation difficulties (typically deny)</li>"
                    "<li>Location signage was poor or misleading (may warrant refund)</li>"
                    "<li>Location was temporarily inaccessible due to construction, closures, etc. (approve)</li>"
                    "</ul><br>"
                    "<strong>Action required:</strong> Review location signage and accessibility. "
                    "Check if customer made reasonable effort to locate the facility."
                ),
                policy_rule="Location Finding Issue - Requires Signage Review",
                confidence="high"
            )
        
        # PRIORITY RULE 4: Check for missing attendant / operational failure
        # These are service delivery failures that require LLM analysis to determine
        # if the customer made a good faith effort vs. just changed their mind
        if self._check_for_operational_failure(ticket_description, ticket_notes):
            logger.info("Operational failure detected (missing attendant/amenity), escalating to LLM for nuanced analysis")
            return {
                "decision": "Uncertain",
                "reasoning": (
                    "Customer reports operational failure (missing attendant, closed facility, etc.). "
                    "This requires LLM analysis to determine if customer made good faith effort to use service "
                    "or if this is a legitimate service delivery failure warranting refund."
                ),
                "policy_rule": "Operational Failure - Requires LLM Analysis",
                "confidence": "low",
                "method_used": "rules"  # Will trigger LLM in decision_maker
            }
        
        # PRIORITY RULE 5: Check for vehicle restriction issues using LLM classification
        # Only triggers if customer EXPLICITLY mentions being turned away due to vehicle type
        # This prevents unnecessary LLM calls for unrelated issues
        # Use customer_complaint (not full ticket_description) to avoid false positives from location metadata
        vehicle_check_result = self._check_for_vehicle_restriction_issue(customer_complaint)
        print(f"DEBUG: Vehicle restriction check result: {vehicle_check_result}")
        logger.info(f"Vehicle restriction check result: {vehicle_check_result} (checked against: '{customer_complaint[:100] if customer_complaint else 'EMPTY'}')")
        if vehicle_check_result:
            logger.info("Customer explicitly mentions vehicle-based rejection, using LLM classifier")
            
            # Extract vehicle and location restrictions from ticket notes
            logger.debug(f"Ticket notes length: {len(ticket_notes) if ticket_notes else 0} chars")
            vehicle = self.vehicle_classifier.extract_vehicle_from_ticket(ticket_notes)
            location_restrictions = self.vehicle_classifier.extract_location_restrictions(ticket_notes)
            
            logger.info(f"Extraction results: vehicle='{vehicle}', restrictions_found={location_restrictions is not None}")
            
            if vehicle and location_restrictions:
                logger.info(f"Classifying vehicle: {vehicle} against restrictions: {location_restrictions[:100]}...")
                
                # Use LLM to classify and compare
                classification = await self.vehicle_classifier.check_vehicle_restriction_mismatch(
                    vehicle_make_model=vehicle,
                    location_restrictions=location_restrictions,
                    ticket_description=ticket_description
                )
                
                # If there's a mismatch (vehicle was incorrectly rejected), approve
                if classification.get("is_mismatch") and classification.get("confidence") in ["high", "medium"]:
                    logger.info(f"Vehicle restriction mismatch confirmed: {classification.get('reasoning')}")
                    
                    # Format restricted categories for display
                    restricted_display = ', '.join(classification.get('restricted_categories', []))
                    
                    return {
                        "decision": "Approved",
                        "reasoning": (
                            f"Customer was turned away due to vehicle restrictions that do not apply to their vehicle.<br><br>"
                            f"<strong>Vehicle:</strong> {vehicle}<br>"
                            f"<strong>Classified as:</strong> {classification.get('vehicle_category').replace('_', ' ').title()}<br>"
                            f"<strong>Location restricts:</strong> {restricted_display}<br><br>"
                            f"<strong>Analysis:</strong> {classification.get('reasoning')}<br><br>"
                            f"<strong>Action required:</strong> Contact location to clarify vehicle restriction policy."
                        ),
                        "policy_rule": "Vehicle Restriction Mismatch",
                        "confidence": classification.get("confidence", "medium"),
                        "method_used": "llm"  # Mark that LLM was used for classification
                    }
                else:
                    logger.info(f"No vehicle restriction mismatch found: {classification.get('reasoning')}")
            else:
                logger.warning(f"Could not extract vehicle ({vehicle}) or restrictions ({location_restrictions is not None})")
                
                # Fallback to simple keyword matching
                logger.info("Falling back to keyword-based vehicle restriction check")
                return {
                    "decision": "Approved",
                    "reasoning": (
                        "Customer reports being turned away due to vehicle restrictions. "
                        "Unable to verify against location's restriction list, but customer's description "
                        "suggests the restriction was not clearly disclosed in the booking.<br><br>"
                        "<strong>Action required:</strong> Manually verify vehicle type against location restrictions."
                    ),
                    "policy_rule": "Undisclosed Vehicle Restriction (Unverified)",
                    "confidence": "medium"
                }
        
        # PRIORITY RULE 6: Check for technical access failures (app errors, QR failures, system issues)
        # Customer couldn't access location due to technical problems - different from "paid extra"
        if self._check_for_technical_access_failure(customer_complaint):
            logger.info("Rule matched: Technical Access Failure. Decision: Needs Human Review (proof verification required)")
            return {
                "decision": "Needs Human Review",
                "reasoning": (
                    "Customer reports being unable to access location due to technical issues "
                    "(app errors, QR code failures, system malfunctions).<br><br>"
                    "<strong>Human review required to:</strong><br>"
                    "<ul>"
                    "<li>Verify customer's claim with proof of payment attempts</li>"
                    "<li>Confirm pass was not used (check entry/exit logs if available)</li>"
                    "<li>Validate technical issue occurred (app logs, system status)</li>"
                    "<li>Ensure customer didn't successfully park elsewhere</li>"
                    "</ul><br>"
                    "<strong>Decision guidance:</strong><br>"
                    "If verified with proof → Approve full refund<br>"
                    "If unverified or suspicious → Deny or offer partial credit"
                ),
                "policy_rule": "Technical Access Failure - Requires Proof Verification",
                "confidence": "high"
            }
        
        # PRIORITY RULE 7: Check for multiple charges/duplicate payment claims (need proof verification)
        # Customer claims they were charged multiple times - requires proof to prevent fraud
        if self._check_for_multiple_charges_claim(customer_complaint):
            logger.info("Rule matched: Multiple Charges Claim. Decision: Needs Human Review (proof verification required)")
            return {
                "decision": "Needs Human Review",
                "reasoning": (
                    "Customer reports being charged multiple times for a single parking transaction.<br><br>"
                    "<strong>Human review required to:</strong><br>"
                    "<ul>"
                    "<li>Verify multiple charges on customer's payment method</li>"
                    "<li>Check if customer successfully parked (may affect refund amount)</li>"
                    "<li>Confirm only one booking should have been created</li>"
                    "<li>Calculate appropriate refund amount (total charges minus legitimate charge)</li>"
                    "</ul><br>"
                    "<strong>Decision guidance:</strong><br>"
                    "If verified with proof → Approve refund of duplicate charges only<br>"
                    "If customer parked → Refund excess charges, keep one legitimate charge<br>"
                    "If unverified → Deny or request additional proof"
                ),
                "policy_rule": "Multiple Charges Claim - Requires Proof Verification",
                "confidence": "high"
            }
        
        # PRIORITY RULE 8: Check for extra charge claims (need proof verification)
        # Customer claims they had to pay additional money - distinguish between system failures and overstay
        if self._check_for_extra_charge_claim(ticket_description):
            # Check if this looks like a system failure vs overstay scenario
            system_failure_indicators = [
                "scan.*error", "qr.*not work", "code.*not work", "scanner.*not work",
                "system.*down", "couldn't scan", "scan.*failed", "barcode.*error",
                "no attendant", "attendant.*not.*there", "machine.*broken"
            ]
            
            is_system_failure = any(re.search(indicator, ticket_description) for indicator in system_failure_indicators)
            
            if is_system_failure:
                logger.info("Rule matched: Extra Charge - System Failure. Decision: Needs Human Review (likely approve)")
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        "Customer reports system failure requiring additional payment. Human review required to:<br>"
                        "<ol>"
                        "<li>Verify system failure (QR code, scanner, attendant issues)</li>"
                        "<li>Confirm customer was within booking window</li>"
                        "<li>Check if booking was actually used or customer paid twice</li>"
                        "</ol><br>"
                        "<strong>Decision guidance:</strong>"
                        "<ul>"
                        "<li>If system failure within booking window → Approve refund (customer paid twice)</li>"
                        "<li>If customer arrived early/late causing issues → Verify timeline before deciding</li>"
                        "</ul>"
                    ),
                    "policy_rule": "System Failure - Paid Again",
                    "confidence": "high"
                }
            else:
                logger.info("Rule matched: Extra Charge Claim. Decision: Needs Human Review (overstay verification required)")
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        "Customer reports having to pay additional charges. Human review required to:<br>"
                        "<ol>"
                        "<li>Verify customer's arrival/entry time against booking start time</li>"
                        "<li>Verify customer's exit time against booking end time</li>"
                        "<li>Request proof of payment if needed</li>"
                        "</ol><br>"
                        "<strong>Decision guidance:</strong>"
                        "<ul>"
                        "<li>If entry was before booking start or exit exceeded booking end → Early arrival or overstay (customer responsibility - deny)</li>"
                        "<li>If entry and exit were both within booking window → Possible system error (investigate further)</li>"
                        "</ul>"
                    ),
                    "policy_rule": "Extra Charge Claim - Requires Timeline Verification",
                    "confidence": "high"
                }
        
        # PRIORITY RULE 9: Check for retroactive booking (booked after start time)
        # This is suspicious and needs human review to clarify customer intent
        if self._check_for_retroactive_booking(ticket_description):
            logger.info("Rule matched: Retroactive Booking. Decision: Needs Human Review (timing clarification needed)")
            
            # Check if this is a "wrong account" scenario (common duplicate booking pattern)
            is_wrong_account = any(phrase in customer_complaint for phrase in [
                'wrong account', 'different account', 'logged in wrong', 'logged on wrong'
            ])
            
            if is_wrong_account:
                reasoning = (
                    "Customer reports accidentally booking with wrong account. This may be a duplicate booking scenario.<br><br>"
                    "<strong>Human review required to:</strong>"
                    "<ol>"
                    "<li>Check if customer has another booking for the same location/date/time under a different account</li>"
                    "<li>Verify which booking (if any) was actually used</li>"
                    "<li>Confirm this is a genuine duplicate, not retroactive booking after using the pass</li>"
                    "</ol><br>"
                    "<strong>Decision guidance:</strong>"
                    "<ul>"
                    "<li><strong>Approve</strong> if customer has a duplicate booking and didn't use this one</li>"
                    "<li><strong>Deny</strong> if customer used this booking or if no duplicate exists</li>"
                    "</ul>"
                )
            else:
                reasoning = (
                    "Customer reports booking wrong time or date. Human review required to:<br>"
                    "<ol>"
                    "<li>Verify if booking was created after the booking start time (retroactive - suspicious)</li>"
                    "<li>Clarify if customer intended to book for a different date/time</li>"
                    "<li>Check if customer's actual arrival/exit times align with their claimed intent</li>"
                    "</ol><br>"
                    "<strong>Decision guidance:</strong>"
                    "<ul>"
                    "<li>May warrant refund if customer can demonstrate they intended different booking time</li>"
                    "<li>May be non-refundable if this was customer error after using the pass</li>"
                    "</ul>"
                )
            
            return {
                "decision": "Needs Human Review",
                "reasoning": reasoning,
                "policy_rule": "Retroactive/Wrong Time Booking - Requires Clarification",
                "confidence": "high"
            }
        
        # Rule 1: 7+ days before event → Approve (Pre-Arrival)
        if days_before_event >= 7:
            logger.info(f"Rule matched: Pre-Arrival (7+ days). Decision: Approved")
            return self._create_rule_result(
                decision="Approved",
                reasoning=f"Cancellation requested {days_before_event} days before event start. "
                         f"Pre-arrival cancellations (7+ days) are automatically approved per policy.",
                policy_rule="Pre-Arrival (7+ days before event)",
                confidence="high",
                timeline_analysis=timeline_analysis,
                scenario_classification=scenario_classification.scenario_type if scenario_classification else None
            )
        
        # Rule 2: After event start → Deny (unless special circumstances)
        if days_before_event < 0:
            logger.info(f"Post-event cancellation detected ({abs(days_before_event)} days after)")
            
            # Check for special circumstances that override post-event denial
            if self._check_for_oversold(ticket_description):
                logger.info("Rule matched: Oversold Location (post-event exception). Decision: Approved")
                return {
                    "decision": "Approved",
                    "reasoning": "Location was oversold/full. Customer was unable to park despite valid booking.",
                    "policy_rule": "Oversold Location",
                    "confidence": "high"
                }
            
            if self._check_for_paid_again(customer_complaint):
                logger.info("Rule matched: Paid Again (post-event exception). Decision: Approved")
                
                # Enhanced reasoning with timeline analysis if available
                reasoning = "Customer had to pay again on-site despite having a valid booking."
                if timeline_analysis and timeline_analysis.is_timeline_consistent:
                    if not timeline_analysis.has_early_arrival and not timeline_analysis.has_overstay:
                        duration_formatted = self._format_duration(timeline_analysis.actual_parking_duration)
                        reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer parked within their booking window " \
                                   f"({duration_formatted}), confirming this was a system failure or duplicate payment scenario."
                    else:
                        early_formatted = self._format_duration(timeline_analysis.early_arrival_minutes)
                        overstay_formatted = self._format_duration(timeline_analysis.overstay_minutes)
                        reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer had early arrival ({early_formatted}) " \
                                   f"or overstay ({overstay_formatted}), but still paid again on-site, indicating system issues."
                
                return {
                    "decision": "Approved",
                    "reasoning": reasoning,
                    "policy_rule": "Paid Again",
                    "confidence": "high"
                }
            
            if self._check_for_closed(customer_complaint):
                logger.info("Rule matched: Closed Location (post-event exception). Decision: Approved")
                return {
                    "decision": "Approved",
                    "reasoning": "Location was closed or inaccessible due to circumstances beyond customer control (gate down, flooded, power out, etc.).",
                    "policy_rule": "Closed Location",
                    "confidence": "high"
                }
            
            if self._check_for_accessibility(customer_complaint):
                logger.info("Rule matched: Accessibility Issue (post-event exception). Decision: Approved")
                return {
                    "decision": "Approved",
                    "reasoning": "Customer was unable to access location due to road closures, police blockades, or other access restrictions.",
                    "policy_rule": "Accessibility Issue",
                    "confidence": "high"
                }
            
            if self._check_for_technical_access_failure(customer_complaint):
                logger.info("Rule matched: Technical Access Failure (post-event exception). Decision: Needs Human Review (proof verification required)")
                return {
                    "decision": "Needs Human Review",
                    "reasoning": (
                        "Customer reports being unable to access location due to technical issues "
                        "(app errors, QR code failures, system malfunctions).<br><br>"
                        "<strong>Human review required to:</strong><br>"
                        "<ul>"
                        "<li>Verify customer's claim with proof of payment attempts</li>"
                        "<li>Confirm pass was not used (check entry/exit logs if available)</li>"
                        "<li>Validate technical issue occurred (app logs, system status)</li>"
                        "<li>Ensure customer didn't successfully park elsewhere</li>"
                        "</ul><br>"
                        "<strong>Decision guidance:</strong><br>"
                        "If verified with proof → Approve full refund<br>"
                        "If unverified or suspicious → Deny or offer partial credit"
                    ),
                    "policy_rule": "Technical Access Failure - Requires Proof Verification",
                    "confidence": "high"
                }
            
            # Post-event without clear special scenario - escalate for review
            # Per policy: Post-event refunds ARE allowed for legitimate service failures
            # If we don't recognize the scenario, let LLM or human review it
            logger.info("Post-event cancellation without recognized special scenario - escalating for review")
            return {
                "decision": "Uncertain",
                "reasoning": f"Cancellation requested {abs(days_before_event)} days after event start. "
                            f"Post-event refunds require evaluation of specific circumstances. "
                            f"This case needs deeper analysis to determine if there was a legitimate service failure.",
                "policy_rule": "Post-Event - Requires Review",
                "confidence": "low"
            }
        
        # Rule 3: <3 days + on-demand → Deny
        if days_before_event < 3 and "on-demand" in booking_type:
            logger.info("Rule matched: On-Demand Cancellation (<3 days). Decision: Denied")
            return self._create_rule_result(
                decision="Denied",
                reasoning=f"On-demand booking with only {days_before_event} days notice. "
                            f"On-demand bookings require 3+ days notice for cancellation.",
                policy_rule="On-Demand Cancellation Policy (<3 days)",
                confidence="high",
                timeline_analysis=timeline_analysis,
                scenario_classification=scenario_classification.scenario_type if scenario_classification else None
            )
        
        # Rule 4: 3-7 days + confirmed booking → Approve (medium confidence)
        if 3 <= days_before_event < 7 and "confirmed" in booking_type:
            logger.info("Rule matched: Confirmed Booking (3-7 days). Decision: Approved")
            return self._create_rule_result(
                decision="Approved",
                reasoning=f"Confirmed booking with {days_before_event} days notice. "
                            f"Meets minimum cancellation window for confirmed bookings.",
                policy_rule="Confirmed Booking (3-7 days notice)",
                confidence="medium",
                timeline_analysis=timeline_analysis,
                scenario_classification=scenario_classification.scenario_type if scenario_classification else None
            )
        
        # Rule 5: Check for special scenarios that always approve
        if self._check_for_oversold(ticket_description):
            logger.info("Rule matched: Oversold Location. Decision: Approved")
            return {
                "decision": "Approved",
                "reasoning": "Location was oversold/full. Customer was unable to park despite valid booking.",
                "policy_rule": "Oversold Location",
                "confidence": "high"
            }
        
        if self._check_for_paid_again(customer_complaint):
            logger.info("Rule matched: Paid Again. Decision: Approved")
            
            # Enhanced reasoning with timeline analysis if available
            reasoning = "Customer had to pay again on-site despite having a valid booking."
            if timeline_analysis and timeline_analysis.is_timeline_consistent:
                if not timeline_analysis.has_early_arrival and not timeline_analysis.has_overstay:
                    duration_formatted = self._format_duration(timeline_analysis.actual_parking_duration)
                    reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer parked within their booking window " \
                               f"({duration_formatted}), confirming this was a system failure or duplicate payment scenario."
                else:
                    early_formatted = self._format_duration(timeline_analysis.early_arrival_minutes)
                    overstay_formatted = self._format_duration(timeline_analysis.overstay_minutes)
                    reasoning += f"<br><br><strong>Timeline Analysis:</strong> Customer had early arrival ({early_formatted}) " \
                               f"or overstay ({overstay_formatted}), but still paid again on-site, indicating system issues."
            
            return {
                "decision": "Approved",
                "reasoning": reasoning,
                "policy_rule": "Paid Again",
                "confidence": "high"
            }
        
        # Rule 6: 3-7 days with unclear booking type → Uncertain (needs LLM)
        if 3 <= days_before_event < 7:
            logger.info("Rule matched: Ambiguous Booking Type (3-7 days). Decision: Uncertain (needs LLM)")
            return {
                "decision": "Uncertain",
                "reasoning": f"Cancellation with {days_before_event} days notice, but booking type is unclear. "
                            f"Requires LLM analysis to determine if refund should be approved.",
                "policy_rule": "Ambiguous Booking Type (3-7 days)",
                "confidence": "low"
            }
        
        # Rule 7: <3 days with non-on-demand booking → Uncertain (needs LLM)
        if days_before_event < 3:
            logger.info("Rule matched: Short Notice Cancellation (<3 days). Decision: Uncertain (needs LLM)")
            return {
                "decision": "Uncertain",
                "reasoning": f"Short notice cancellation ({days_before_event} days) with booking type '{booking_type}'. "
                            f"Requires LLM analysis to evaluate special circumstances.",
                "policy_rule": "Short Notice Cancellation (<3 days)",
                "confidence": "low"
            }
        
        # Default: Uncertain (edge case)
        logger.warning(f"No rule matched - edge case. Days: {days_before_event}, Type: {booking_type}")
        return {
            "decision": "Uncertain",
            "reasoning": f"Edge case: {days_before_event} days before event, booking type '{booking_type}'. "
                        f"Requires LLM analysis for proper evaluation.",
            "policy_rule": "Edge Case - Requires LLM Analysis",
            "confidence": "low"
        }
    
    def _calculate_days_before_event(
        self,
        cancellation_date: Optional[str],
        event_date: str
    ) -> Optional[int]:
        """
        Calculate the number of days between cancellation and event start.
        
        Uses date-only comparison (ignoring time) to avoid timezone issues
        with same-day bookings.
        
        Args:
            cancellation_date: ISO format date string (YYYY-MM-DD) or None (uses current date)
            event_date: ISO format date string (YYYY-MM-DD)
        
        Returns:
            Number of days before event (positive) or after event (negative),
            or None if date parsing fails
        """
        try:
            # Parse event date (extract date only, ignore time)
            event_dt = datetime.fromisoformat(event_date.replace('Z', '+00:00'))
            if event_dt.tzinfo is None:
                event_dt = event_dt.replace(tzinfo=timezone.utc)
            event_date_only = event_dt.date()
            
            # Parse or use current date for cancellation (date only)
            if cancellation_date:
                cancel_dt = datetime.fromisoformat(cancellation_date.replace('Z', '+00:00'))
                if cancel_dt.tzinfo is None:
                    cancel_dt = cancel_dt.replace(tzinfo=timezone.utc)
                cancel_date_only = cancel_dt.date()
            else:
                cancel_date_only = datetime.now(timezone.utc).date()
            
            # Log parsed dates for debugging
            logger.debug(f"Parsed dates: event={event_date_only}, cancellation={cancel_date_only}")
            
            # Calculate difference in days (using date objects, not datetime)
            delta = (event_date_only - cancel_date_only).days
            return delta
        
        except (ValueError, AttributeError) as e:
            # Invalid date format
            logger.error(f"Date parsing error: {type(e).__name__}: {e}")
            return None
    
    def _check_for_oversold(self, ticket_description: str) -> bool:
        """
        Check if ticket mentions oversold/full location.
        
        IMPORTANT: This checks if the customer was UNABLE to park due to the location
        being full/oversold. It should NOT trigger for:
        - "full price" or "full amount" (payment terms)
        - Cases where customer successfully parked (overstay scenarios)
        
        Args:
            ticket_description: Ticket description text (lowercase)
        
        Returns:
            True if oversold indicators found, False otherwise
        """
        # First check if customer was able to park - if yes, NOT oversold
        if "were you able to park? yes" in ticket_description or "were you able to park?yes" in ticket_description:
            logger.debug("Customer was able to park - not an oversold scenario")
            return False
        
        # Exclude false positives from payment-related "full" mentions
        if "full price" in ticket_description or "full amount" in ticket_description or "full hours" in ticket_description:
            logger.debug("'full' refers to payment terms, not facility capacity")
            return False
        
        # Check for genuine oversold/capacity keywords
        oversold_keywords = [
            "oversold", "no space", "no spots", "at capacity",
            "garage full", "lot full", "sold out", "facility was full",
            "parking was full", "lot was full"
        ]
        
        # "turned away" only counts if NOT related to vehicle restrictions
        if "turned away" in ticket_description:
            # Already handled by vehicle restriction check
            return False
        
        return any(keyword in ticket_description for keyword in oversold_keywords)
    
    def _check_for_duplicate_claim(self, ticket_data: Dict) -> bool:
        """
        Check if customer is claiming a duplicate booking.
        
        NOTE: This method only DETECTS duplicate claims for escalation purposes.
        Actual duplicate detection and resolution is NON-FUNCTIONAL due to 
        ParkWhiz API limitations (cannot search bookings by customer email).
        
        All duplicate claims must be escalated to human review.
        
        Args:
            ticket_data: Ticket data dictionary
        
        Returns:
            True if duplicate claim detected (triggers escalation), False otherwise
        """
        ticket_description = ticket_data.get("description", "").lower()
        ticket_subject = ticket_data.get("subject", "").lower()
        
        duplicate_keywords = [
            "duplicate", "charged twice", "double charge", "two passes",
            "bought twice", "multiple passes", "same time", "two bookings",
            "charged 2 times", "billed twice", "double booking"
        ]
        
        # Check both subject and description
        full_text = f"{ticket_subject} {ticket_description}"
        return any(keyword in full_text for keyword in duplicate_keywords)
    
    def _check_for_retroactive_booking(self, ticket_description: str) -> bool:
        """
        Check if booking was created AFTER the booking start time (retroactive).
        
        This is suspicious because you cannot book for a time that has already passed.
        Common scenarios:
        - Customer books at 8:21 PM for a pass starting at 8:00 PM (21 min retroactive)
        - May indicate customer confusion about booking times
        - Needs human review to clarify customer intent
        
        Args:
            ticket_description: Ticket description text (lowercase)
        
        Returns:
            True if retroactive booking indicators found
        """
        # Look for patterns in Zapier notes that show booking created after start time
        # The Zapier note includes both "Booking Created" and "Parking Pass Start Time"
        # This is a simple heuristic - if customer mentions booking "wrong time" or "mistake"
        # combined with timing issues, flag for review
        
        retroactive_indicators = [
            "wrong time", "booked wrong", "wrong date", "mistake",
            "meant to book", "supposed to book", "intended to book",
            "booked for wrong", "incorrect time", "incorrect date"
        ]
        
        return any(indicator in ticket_description for indicator in retroactive_indicators)
    
    def _extract_customer_complaint(self, ticket_notes: str) -> Dict:
        """
        Extract customer's complaint information with support for contradiction analysis.
        
        This method extracts the customer's actual complaint (Reason + Summary) while
        preserving original customer data for agent reference. It supports parsing
        multiple complaint types from single customer input for contradiction detection.
        
        Args:
            ticket_notes: Full ticket notes including Zapier data (can be None or empty)
        
        Returns:
            Dictionary containing:
                - complaint_text (str): Lowercase string containing customer's Reason and Summary
                - original_reason (str): Original Reason field text (preserved for agents)
                - original_summary (str): Original Summary field text (preserved for agents)
                - original_notes (str): Full original ticket notes (preserved for agents)
                - complaint_parts (List[str]): Individual complaint components for analysis
        """
        # Handle None or empty ticket_notes
        if not ticket_notes:
            return {
                "complaint_text": "",
                "original_reason": "",
                "original_summary": "",
                "original_notes": "",
                "complaint_parts": []
            }
        
        complaint_parts = []
        original_reason = ""
        original_summary = ""
        
        # Extract Reason field - stop at next field (Arrival, Make and Model, etc.)
        reason_match = re.search(r'Reason:\s*([^\n]+?)(?=\s*(?:Arrival|Make and Model|Were you able|Summary|$))', ticket_notes, re.IGNORECASE | re.DOTALL)
        if reason_match:
            original_reason = reason_match.group(1).strip()
            complaint_parts.append(original_reason)
        
        # Extract Summary field - stop at "Booking information" or other metadata
        summary_match = re.search(r'Summary:\s*(.+?)(?=\s*(?:Booking information|Refund this booking|Location Name|$))', ticket_notes, re.IGNORECASE | re.DOTALL)
        if summary_match:
            # Clean up the summary - remove extra whitespace and newlines
            original_summary = summary_match.group(1).strip()
            cleaned_summary = re.sub(r'\s+', ' ', original_summary)
            complaint_parts.append(cleaned_summary)
        
        # Combine complaint parts for analysis
        complaint_text = " ".join(complaint_parts).lower()
        
        return {
            "complaint_text": complaint_text,
            "original_reason": original_reason,
            "original_summary": original_summary,
            "original_notes": ticket_notes,
            "complaint_parts": complaint_parts
        }
    
    def _check_for_operational_failure(self, ticket_description: str, ticket_notes: str = "") -> bool:
        """
        Check if customer reports an operational/service delivery failure.
        
        This is a GATEKEEPER function for LLM analysis of edge cases where:
        - The location failed to provide the service (missing attendant, closed, etc.)
        - The customer made a good faith effort to use the service
        - It's unclear if this warrants a refund without understanding context
        
        Common scenarios:
        - "no attendant after 10 mins wait"
        - "facility was closed"
        - "gate wouldn't open"
        - "no one to help me"
        - "missing amenity" (EV charger, handicap access, etc.)
        
        Args:
            ticket_description: Ticket description text (lowercase, can be None)
            ticket_notes: Full ticket notes including Zapier data (lowercase, can be None)
        
        Returns:
            True if customer reports operational failure requiring LLM analysis
        """
        # Handle None values
        ticket_description = ticket_description or ""
        ticket_notes = ticket_notes or ""
        
        # Combine description and notes for checking
        full_text = f"{ticket_description} {ticket_notes}".lower()
        
        # Keywords indicating operational/service failures
        operational_failure_keywords = [
            "no attendant", "missing attendant", "attendant not there", "no one there",
            "facility closed", "location closed", "gate closed", "lot closed",
            "gate wouldn't open", "gate didn't open", "couldn't get in",
            "no one to help", "nobody there", "no staff",
            "missing amenity", "no ev charger", "no handicap", "no elevator",
            "waited", "wait", "after.*min"  # "waited 10 mins", "after 10 mins wait"
        ]
        
        # Check if any operational failure keywords are present
        has_operational_issue = any(
            re.search(keyword, full_text) 
            for keyword in operational_failure_keywords
        )
        
        # Also check the Zapier "Reason" field specifically (only if ticket_notes is not empty)
        if ticket_notes:
            reason_match = re.search(r'Reason:\s*([^\n]+)', ticket_notes, re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).lower()
                # Check if reason mentions missing attendant/amenity
                if any(keyword in reason for keyword in [
                    'missing attendant', 'missing amenity', 'handicap', 'ev charger',
                    'closed', 'no attendant', 'no one'
                ]):
                    logger.info(f"Operational failure detected in Reason field: {reason}")
                    return True
        
        if has_operational_issue:
            logger.info(f"Operational failure keywords detected in ticket text")
        
        return has_operational_issue
    
    def _check_for_vehicle_restriction_issue(self, ticket_description: str) -> bool:
        """
        Check if customer explicitly claims they were turned away due to vehicle restrictions.
        
        This is a GATEKEEPER function - only returns True if customer clearly states
        they were rejected because of their vehicle type. This prevents unnecessary
        LLM calls for unrelated issues.
        
        Common scenarios that should trigger this:
        - "didn't allow crossover vehicles"
        - "turned away because of my SUV"
        - "said they don't accept trucks"
        - "attendant told me my vehicle type wasn't allowed"
        
        Args:
            ticket_description: Ticket description text (lowercase)
        
        Returns:
            True ONLY if customer explicitly mentions vehicle-based rejection
        """
        # FIRST: Check for facility capacity issues - these should NOT trigger vehicle restriction check
        capacity_keywords = [
            "facility was full", "facility is full", "parking was full", "parking is full",
            "lot was full", "lot is full", "garage was full", "garage is full",
            "at capacity", "at max capacity", "full capacity", "no space", "no spaces",
            "full of cars", "filled up", "completely full", "accepting more cars"
        ]
        
        has_capacity_issue = any(re.search(keyword, ticket_description) for keyword in capacity_keywords)
        
        if has_capacity_issue:
            logger.debug("Facility capacity issue detected - skipping vehicle restriction check")
            return False
        
        # MUST have at least one "rejection" keyword
        rejection_keywords = [
            "didn't allow", "don't allow", "not allow", "wouldn't allow", "not allowed",
            "didn't accept", "don't accept", "not accept", "wouldn't accept",
            "turned away", "turned me away", "rejected", "denied entry",
            "wouldn't let", "didn't let", "refused", "wasn't allowed", "weren't allowed"
        ]
        
        has_rejection = any(re.search(keyword, ticket_description) for keyword in rejection_keywords)
        
        if not has_rejection:
            return False
        
        # MUST also mention vehicle-related terms
        vehicle_keywords = [
            "vehicle", "car", "suv", "crossover", "truck", "van",
            "tesla", "sedan", "make and model", "vehicle type"
        ]
        
        has_vehicle_mention = any(re.search(keyword, ticket_description) for keyword in vehicle_keywords)
        
        # Only return True if BOTH rejection AND vehicle are mentioned
        # This ensures we only call the LLM classifier when it's actually a vehicle issue
        return has_rejection and has_vehicle_mention
    
    def _check_for_extra_charge_claim(self, ticket_description: str) -> bool:
        """
        Check if customer claims they had to pay additional charges.
        
        These claims require human review to:
        1. Verify proof of payment
        2. Check if entry time was within booking window
        3. Check if exit time was within booking window
        4. Determine if charge was legitimate or overstay
        
        Args:
            ticket_description: Ticket description text (lowercase)
        
        Returns:
            True if customer claims extra charges (needs human verification)
        """
        extra_charge_keywords = [
            "had to pay", "charged additional", "charged extra", "pay more",
            "additional.*due", "told.*due", "pay.*leave", "pay.*exit",
            "pay.*retrieve", "before they would release"
        ]
        
        return any(keyword in ticket_description for keyword in extra_charge_keywords)
    
    def _check_for_paid_again(self, customer_complaint: str) -> bool:
        """
        Check if customer had to pay again on-site for a DUPLICATE booking.
        
        This distinguishes between:
        - Paid again for duplicate booking (approve) 
        - Paid additional overstay/exit charges (deny - not a refund case)
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if paid-again indicators found AND not an overstay scenario, False otherwise
        """
        # First check for explicit "paid again" / "pay again" / "made me pay" language
        # These are strong indicators of duplicate payment scenarios
        strong_paid_again_indicators = [
            "paid again", "pay again", "made me pay", "charged again",
            "paid twice", "charged twice", "double charge"
        ]
        
        if any(indicator in customer_complaint for indicator in strong_paid_again_indicators):
            logger.debug(f"Strong 'paid again' indicator found in: {customer_complaint[:100]}")
            return True
        
        # Exclusion keywords that indicate overstay/exit charges (NOT duplicate bookings)
        # NOTE: "attendant" removed - attendants can be involved in both scenarios
        # Made more specific to avoid false exclusions
        overstay_keywords = [
            "additional time", "overstay", "over stay", "exceeded", "extra time",
            "stayed longer", "stayed past", "stayed beyond",
            "before they would release", "additional.*due.*time",
            "told.*due.*overstay", "pay.*exit.*overstay", "pay.*retrieve.*late",
            "more time", "longer than", "late departure", "overtime parking"
        ]
        
        # Check if this is an overstay scenario
        if any(keyword in customer_complaint for keyword in overstay_keywords):
            # If overstay keywords present, NOT a duplicate booking scenario
            logger.debug(f"Overstay keywords detected - not a 'paid again' scenario")
            return False
        
        # Keywords that indicate duplicate booking payment (not overstay)
        paid_again_keywords = [
            "charged at gate", "paid onsite", "paid on-site", "paid at location",
            "made me pay", "wouldn't let me park",
            "said i didn't have", "no reservation", "not in system",
            "scan.*error", "qr.*not work", "code.*not work", "scanner.*not work",
            "system.*down", "couldn't scan", "scan.*failed", "barcode.*error",
            "paid ourselves", "paid myself", "paid normal price", "paid full price"
        ]
        
        # More specific check for "had to pay" - only if combined with system/booking issues
        if "had to pay" in customer_complaint:
            # Only treat as "paid again" if combined with booking/system issues
            system_issue_indicators = [
                "no reservation", "not in system", "didn't have", "wouldn't let me park",
                "scan", "qr", "code", "system", "barcode", "reservation"
            ]
            if any(indicator in customer_complaint for indicator in system_issue_indicators):
                return True
            else:
                # "had to pay" without system issues is likely an extra charge scenario
                return False
        
        return any(keyword in customer_complaint for keyword in paid_again_keywords)
    
    def _check_for_closed(self, customer_complaint: str) -> bool:
        """
        Check if location was closed or inaccessible.
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if closed/inaccessible indicators found, False otherwise
        """
        closed_keywords = [
            "closed", "gate down", "flooded", "power out", "no power",
            "elevator broken", "lift broken", "no lights", "lights off",
            "no attendant", "nobody there", "shut down", "not open"
        ]
        return any(keyword in customer_complaint for keyword in closed_keywords)
    
    def _check_for_accessibility(self, customer_complaint: str) -> bool:
        """
        Check if customer couldn't access location due to external factors.
        
        IMPORTANT: Only checks customer's actual complaint, NOT location metadata.
        Location admin notes may mention "NYPD may sometimes close the road" as a 
        general warning, but that doesn't mean the road was actually closed for this customer.
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if accessibility issue indicators found, False otherwise
        """
        accessibility_keywords = [
            "road closed", "street closed", "blocked", "police block",
            "construction", "parade", "barricade", "can't access",
            "couldn't access", "unable to access", "no access", "blocked off",
            "road closure", "detour", "emergency"
        ]
        return any(keyword in customer_complaint for keyword in accessibility_keywords)
    
    def _check_for_location_finding_issue(self, customer_complaint: str) -> bool:
        """
        Check if customer reports being unable to find the parking location.
        
        This is typically a customer navigation issue rather than a location problem,
        but requires human review to determine if signage was adequate.
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if location finding issue indicators found, False otherwise
        """
        location_finding_keywords = [
            "could not find location", "couldn't find location", "unable to find location",
            "can't find location", "cannot find location", "could not locate",
            "couldn't locate", "unable to locate", "can't locate", "cannot locate",
            "could not find the location", "couldn't find the location",
            "could not find parking", "couldn't find parking", "unable to find parking",
            "went round and round", "went around and around", "driving around",
            "drove around", "circled", "looking for", "searching for",
            "no signs", "no signage", "poor signage", "bad signage", "unclear signage",
            "signage needs", "improve signage", "better signage"
        ]
        
        return any(keyword in customer_complaint for keyword in location_finding_keywords)
    
    def _check_for_technical_access_failure(self, customer_complaint: str) -> bool:
        """
        Check if customer was unable to access location due to technical issues.
        
        This covers scenarios where the customer couldn't enter the location due to:
        - App error messages preventing entry
        - QR code failures at entry gates
        - System malfunctions preventing access
        - Scanner/gate technical problems
        
        This is different from "paid extra charges" because the customer never got in at all.
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if technical access failure indicators found, False otherwise
        """
        technical_access_keywords = [
            "error message", "error messages", "app error", "app errors",
            "qr code.*not work", "qr.*not work", "qr.*error", "qr.*failed",
            "qr code.*wouldn't work", "qr.*wouldn't work", "qr.*would not work",
            "code.*not work", "code.*error", "code.*failed",
            "scanner.*not work", "scanner.*error", "scanner.*failed", "scanner.*broken",
            "gate.*not work", "gate.*error", "gate.*failed", "gate.*broken",
            "gate.*wouldn't open", "gate.*would not open", "gate.*wouldn't work",
            "system.*not work", "system.*error", "system.*failed", "system.*down",
            "couldn't scan", "could not scan", "unable to scan", "can't scan",
            "couldn't enter", "could not enter", "unable to enter", "can't enter",
            "wouldn't let.*in", "would not let.*in", "denied entry", "denied access",
            "app.*wouldn't work", "app.*would not work", "app.*not working",
            "technical.*problem", "technical.*issue", "technical.*error",
            "malfunction", "not functioning", "out of order"
        ]
        
        # Use regex for more complex patterns
        import re
        return any(
            re.search(keyword, customer_complaint) 
            for keyword in technical_access_keywords
        )
    
    def _check_for_multiple_charges_claim(self, customer_complaint: str) -> bool:
        """
        Check if customer claims they were charged multiple times for one transaction.
        
        This covers scenarios where the customer reports:
        - Being charged 2, 3, or more times for one parking session
        - Duplicate charges on their payment method
        - Multiple transactions when only one was intended
        
        This requires proof verification to prevent fraud.
        
        Args:
            customer_complaint: Customer's complaint text (Reason + Summary only, lowercase)
        
        Returns:
            True if multiple charges claim detected, False otherwise
        """
        multiple_charges_keywords = [
            "charged.*times", "charged.*twice", "charged.*3.*times", "charged.*three.*times",
            "charged.*multiple", "multiple.*charges", "duplicate.*charges",
            "charged.*2.*times", "charged.*two.*times", "double.*charged",
            "billed.*twice", "billed.*multiple", "billed.*2.*times", "billed.*3.*times",
            "payment.*went.*through.*times", "transaction.*went.*through.*times",
            "charged.*total.*of", "should.*only.*paid", "at.*most.*should.*have.*paid",
            "was.*charged.*and.*charged", "got.*charged.*for.*same",
            "same.*transaction.*charged", "one.*transaction.*charged.*times",
            # Pattern for specific amounts like "$16.75 + $16.75 + $14.65"
            r"\$\d+\.\d+.*\+.*\$\d+\.\d+.*\+.*\$\d+\.\d+",
            r"\$\d+\.\d+.*\+.*\$\d+\.\d+",  # Two charges
            "charged.*\$.*\+.*\$",  # General pattern for multiple dollar amounts
        ]
        
        # Use regex for pattern matching
        import re
        return any(
            re.search(keyword, customer_complaint) 
            for keyword in multiple_charges_keywords
        )
    
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
    
    def _create_rule_result(self, decision: str, reasoning: str, policy_rule: str, confidence: str, **kwargs) -> Dict:
        """
        Create a standardized rule result with confidence description and enhanced classification data.
        
        Args:
            decision: The decision ("Approved", "Denied", "Uncertain", "Needs Human Review")
            reasoning: Human-readable explanation
            policy_rule: Policy rule applied
            confidence: Confidence level ("high", "medium", "low")
            **kwargs: Additional fields to include in the result (timeline_analysis, scenario_classification, etc.)
            
        Returns:
            Dictionary with standardized rule result format including enhanced classification data
        """
        result = {
            "decision": decision,
            "reasoning": reasoning,
            "policy_rule": policy_rule,
            "confidence": confidence,
            "confidence_description": self._get_confidence_description(confidence)
        }
        
        # Add timeline analysis data if provided
        timeline_analysis = kwargs.get('timeline_analysis')
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
        
        # Add scenario classification if provided
        scenario_classification = kwargs.get('scenario_classification')
        if scenario_classification:
            result["scenario_classification"] = scenario_classification
        
        # Add key factors if provided
        key_factors = kwargs.get('key_factors')
        if key_factors:
            result["key_factors"] = key_factors
        
        # Add escalation reason if provided
        escalation_reason = kwargs.get('escalation_reason')
        if escalation_reason:
            result["escalation_reason"] = escalation_reason
        
        # Add contradictions if provided
        contradictions = kwargs.get('contradictions')
        if contradictions:
            result["contradictions"] = contradictions
        
        # Add suggested questions if provided
        suggested_questions = kwargs.get('suggested_questions')
        if suggested_questions:
            result["suggested_questions"] = suggested_questions
        
        # Add any other additional fields
        for key, value in kwargs.items():
            if key not in ['timeline_analysis', 'scenario_classification', 'key_factors', 
                          'escalation_reason', 'contradictions', 'suggested_questions']:
                result[key] = value
        
        return result
    
    def _format_scenario_type(self, scenario_type: str) -> str:
        """
        Convert snake_case scenario types to human-readable format.
        
        Args:
            scenario_type: Snake case scenario type (e.g., "early_arrival_analysis")
            
        Returns:
            Human-readable format (e.g., "Early Arrival Analysis")
        """
        # Define specific mappings for better readability
        scenario_mappings = {
            "early_arrival_analysis": "Early Arrival Analysis",
            "early_arrival_overstay": "Early Arrival Overstay",
            "accidental_double_booking": "Accidental Double Booking",
            "entry_failure_complete_unused": "Entry Failure - Complete Unused",
            "exit_complication_poor_experience": "Exit Complication - Poor Experience",
            "insufficient_data": "Insufficient Data",
            "uncertain_requires_analysis": "Uncertain - Requires Analysis",
            "timeline_inconsistent": "Timeline Inconsistent",
            "paid_again_duplicate": "Paid Again - Duplicate",
            "operational_failure": "Operational Failure",
            "vehicle_restriction_mismatch": "Vehicle Restriction Mismatch"
        }
        
        # Return mapped value if available, otherwise convert snake_case to Title Case
        if scenario_type in scenario_mappings:
            return scenario_mappings[scenario_type]
        else:
            # Fallback: convert snake_case to Title Case
            return scenario_type.replace('_', ' ').title()
    
    def _format_duration(self, minutes: int) -> str:
        """
        Convert minutes to human-readable format (hours and minutes).
        
        Args:
            minutes: Duration in minutes
            
        Returns:
            Human-readable format (e.g., "3 hours 7 minutes", "45 minutes", "2 hours")
        """
        if minutes <= 0:
            return "0 minutes"
        
        hours = minutes // 60
        remaining_minutes = minutes % 60
        
        if hours == 0:
            return f"{remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
        elif remaining_minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            return f"{hours} hour{'s' if hours != 1 else ''} {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
