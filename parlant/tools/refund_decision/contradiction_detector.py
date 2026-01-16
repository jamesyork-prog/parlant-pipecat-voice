"""
ContradictionDetector component for identifying conflicting information in refund requests.

This module provides the ContradictionDetector class that identifies contradictory
information within customer refund requests that should trigger human review rather
than automated processing.

Enhanced for "Paid Again" Classification System with timeline analysis integration.
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from .complaint_parser import ComplaintParser, ComplaintType
from .logic_validator import LogicValidator, Contradiction
from .reasoning_generator import ReasoningGenerator

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ContradictionResult:
    """Result of contradiction detection analysis."""
    has_contradictions: bool
    contradictions: List[Contradiction]
    all_complaints: List[ComplaintType]
    reasoning: str
    suggested_questions: List[str]
    escalation_required: bool


class ContradictionDetector:
    """
    Core component responsible for identifying contradictory information in customer complaints.
    
    This class orchestrates complaint parsing, logic validation, and reasoning generation
    to detect conflicts that require human review rather than automated processing.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ContradictionDetector with required components.
        
        Args:
            config: Optional configuration dictionary for contradiction rules
                   Can include 'config_path' key for external configuration file
        """
        self.config = config or {}
        self.initialization_errors = []
        
        # Initialize performance optimizer for caching and monitoring
        try:
            from .performance_optimizer import PerformanceOptimizer
            self.performance_optimizer = PerformanceOptimizer()
            # Apply performance optimizations to config
            self.config = self.performance_optimizer.optimize_timeout_settings(self.config)
            logger.info("PerformanceOptimizer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PerformanceOptimizer: {e}")
            self.performance_optimizer = None
        
        # Initialize components with error handling and graceful degradation
        try:
            self.complaint_parser = ComplaintParser()
            logger.info("ComplaintParser initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ComplaintParser: {e}")
            self.initialization_errors.append(f"ComplaintParser: {e}")
            self.complaint_parser = None
        
        try:
            self.logic_validator = LogicValidator(self.config)
            logger.info("LogicValidator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LogicValidator: {e}")
            self.initialization_errors.append(f"LogicValidator: {e}")
            # Create fallback validator with basic hardcoded rules
            self.logic_validator = self._create_fallback_validator()
        
        try:
            self.reasoning_generator = ReasoningGenerator()
            logger.info("ReasoningGenerator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ReasoningGenerator: {e}")
            self.initialization_errors.append(f"ReasoningGenerator: {e}")
            self.reasoning_generator = None
        
        # Log initialization status
        if self.initialization_errors:
            logger.warning(f"ContradictionDetector initialized with {len(self.initialization_errors)} component failures")
            for error in self.initialization_errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("ContradictionDetector initialized successfully with all components")
    
    def _create_fallback_validator(self):
        """
        Create a fallback LogicValidator with basic hardcoded rules when configuration fails.
        
        Returns:
            LogicValidator instance with minimal hardcoded rules
        """
        try:
            # Create minimal fallback configuration
            fallback_config = {
                "fallback_mode": True,
                "basic_rules_only": True
            }
            
            # Import here to avoid circular imports
            from .logic_validator import LogicValidator
            
            # Create validator with fallback config
            validator = LogicValidator(fallback_config)
            
            # Override with hardcoded basic rules
            validator.mutually_exclusive_pairs = [
                ("accessibility", "oversold"),
                ("paid_again", "oversold")
            ]
            
            validator.response_conflicts = {
                "accessibility": {
                    "conflicting_responses": ["were you able to park? yes", "were you able to park?yes"],
                    "description": "Customer claims accessibility issues but confirms they were able to park"
                },
                "oversold": {
                    "conflicting_responses": ["were you able to park? yes", "were you able to park?yes"],
                    "description": "Customer claims facility was full but confirms they were able to park"
                }
            }
            
            logger.info("Created fallback LogicValidator with basic hardcoded rules")
            return validator
            
        except Exception as e:
            logger.error(f"Failed to create fallback validator: {e}")
            return None
    
    def detect_timeline_contradictions(
        self,
        customer_complaint: str,
        timeline_analysis: Optional[Dict] = None
    ) -> List[Dict]:
        """
        PATTERN-BASED contradiction detection using keyword matching and timeline logic.
        
        Uses deterministic rules for obvious contradictions:
        - "couldn't park" + timeline shows parking duration
        - "never entered" + pass activation data
        - Mathematical timeline inconsistencies
        
        LLM only used for complex linguistic contradictions that require
        natural language understanding beyond keyword patterns.
        
        Args:
            customer_complaint: Customer's complaint text
            timeline_analysis: Optional timeline analysis results
        
        Returns:
            List of detected contradictions with specific examples
        """
        contradictions = []
        complaint_lower = customer_complaint.lower()
        
        # Pattern 1: "Couldn't park" but timeline shows parking occurred
        no_parking_patterns = [
            "couldn't park", "could not park", "unable to park", "not able to park",
            "never parked", "didn't park", "did not park", "no parking",
            "never entered", "couldn't enter", "could not enter", "unable to enter"
        ]
        
        parking_success_patterns = [
            "retrieved my car", "got my car", "picked up my car", "left my car",
            "parked for", "parking duration", "stayed until", "left at"
        ]
        
        # Check for "no parking" claims with evidence of parking
        has_no_parking_claim = any(pattern in complaint_lower for pattern in no_parking_patterns)
        has_parking_evidence = any(pattern in complaint_lower for pattern in parking_success_patterns)
        
        if has_no_parking_claim and has_parking_evidence:
            contradictions.append({
                "type": "logical_contradiction",
                "severity": "high",
                "description": "Customer claims they couldn't park but also describes retrieving their car after parking",
                "customer_statement_a": next((pattern for pattern in no_parking_patterns if pattern in complaint_lower), ""),
                "customer_statement_b": next((pattern for pattern in parking_success_patterns if pattern in complaint_lower), ""),
                "explanation": "These statements are mutually exclusive - cannot both be unable to park and retrieve car after parking"
            })
        
        # Pattern 2: Timeline mathematical inconsistencies
        if timeline_analysis:
            timeline_contradictions = self._detect_timeline_math_contradictions(
                customer_complaint, timeline_analysis
            )
            contradictions.extend(timeline_contradictions)
        
        return contradictions
    
    def _detect_timeline_math_contradictions(
        self,
        customer_complaint: str,
        timeline_analysis: Dict
    ) -> List[Dict]:
        """
        Detect mathematical inconsistencies in timeline data.
        
        Args:
            customer_complaint: Customer's complaint text
            timeline_analysis: Timeline analysis results
        
        Returns:
            List of timeline-based contradictions
        """
        contradictions = []
        complaint_lower = customer_complaint.lower()
        
        # Extract timeline data
        actual_duration = timeline_analysis.get("actual_parking_duration", 0)
        has_early_arrival = timeline_analysis.get("has_early_arrival", False)
        has_overstay = timeline_analysis.get("has_overstay", False)
        timeline_issues = timeline_analysis.get("timeline_issues", [])
        
        # Pattern 1: Claims no parking but timeline shows significant duration
        no_parking_patterns = [
            "couldn't park", "never parked", "didn't park", "no parking occurred"
        ]
        
        has_no_parking_claim = any(pattern in complaint_lower for pattern in no_parking_patterns)
        
        if has_no_parking_claim and actual_duration > 30:  # More than 30 minutes
            contradictions.append({
                "type": "timeline_math_contradiction",
                "severity": "high", 
                "description": f"Customer claims no parking occurred but timeline shows {actual_duration} minutes of parking",
                "timeline_evidence": f"Actual parking duration: {actual_duration} minutes",
                "customer_claim": "No parking occurred",
                "explanation": "Timeline data contradicts customer's claim of no parking"
            })
        
        # Pattern 2: Claims "quick stop" but timeline shows extended parking
        quick_stop_patterns = [
            "quick stop", "just a minute", "briefly", "short time", "few minutes"
        ]
        
        has_quick_claim = any(pattern in complaint_lower for pattern in quick_stop_patterns)
        
        if has_quick_claim and actual_duration > 120:  # More than 2 hours
            contradictions.append({
                "type": "timeline_math_contradiction",
                "severity": "medium",
                "description": f"Customer claims brief parking but timeline shows {actual_duration} minutes ({actual_duration//60:.1f} hours)",
                "timeline_evidence": f"Actual parking duration: {actual_duration} minutes",
                "customer_claim": "Brief/quick parking",
                "explanation": "Extended parking duration contradicts claim of brief stop"
            })
        
        # Pattern 3: Mathematical timeline inconsistencies from analyzer
        if timeline_issues:
            for issue in timeline_issues:
                contradictions.append({
                    "type": "timeline_math_error",
                    "severity": "high",
                    "description": f"Timeline mathematical inconsistency: {issue}",
                    "timeline_evidence": issue,
                    "explanation": "Timeline data contains mathematical errors or impossible values"
                })
        
        # Pattern 4: Claims on-time arrival but timeline shows early arrival + overstay
        on_time_patterns = [
            "arrived on time", "on-time arrival", "arrived when supposed to"
        ]
        
        has_on_time_claim = any(pattern in complaint_lower for pattern in on_time_patterns)
        
        if has_on_time_claim and has_early_arrival and has_overstay:
            early_minutes = timeline_analysis.get("early_arrival_minutes", 0)
            overstay_minutes = timeline_analysis.get("overstay_minutes", 0)
            
            contradictions.append({
                "type": "timeline_math_contradiction",
                "severity": "medium",
                "description": f"Customer claims on-time arrival but timeline shows {early_minutes} minutes early arrival and {overstay_minutes} minutes overstay",
                "timeline_evidence": f"Early arrival: {early_minutes}min, Overstay: {overstay_minutes}min",
                "customer_claim": "On-time arrival",
                "explanation": "Timeline shows early arrival and overstay, contradicting on-time claim"
            })
        
        return contradictions

    async def detect_contradictions(
        self, 
        customer_complaint: str, 
        ticket_data: Dict,
        timeline_analysis: Optional[Dict] = None
    ) -> ContradictionResult:
        """
        Detect contradictions in customer complaint information.
        
        This method analyzes customer complaints to identify conflicting information
        that should trigger human review rather than automated processing. It supports
        parsing multiple complaint types from single customer input and preserves
        original customer data for agent reference.
        
        Enhanced for "Paid Again" Classification System with timeline analysis integration.
        
        Args:
            customer_complaint: Customer's complaint text (reason + summary)
            ticket_data: Dictionary containing ticket information including responses
                        and optional complaint_extraction data
            timeline_analysis: Optional timeline analysis results for mathematical consistency validation
        
        Returns:
            ContradictionResult with detected conflicts and reasoning
        """
        import time
        import uuid
        import asyncio
        from datetime import datetime
        
        start_time = time.time()
        logger.info("Starting contradiction detection analysis")
        logger.debug(f"Customer complaint length: {len(customer_complaint)} chars")
        
        # Performance optimization: Use fast detection if optimizer is available
        # BUT skip if timeline analysis is provided (enhanced detection needed)
        if self.performance_optimizer and timeline_analysis is None:
            try:
                # Set detector reference for logging compatibility
                self.performance_optimizer.detector = self
                return await self.performance_optimizer.fast_contradiction_detection(
                    customer_complaint,
                    ticket_data,
                    self.complaint_parser,
                    self.logic_validator,
                    self.reasoning_generator
                )
            except Exception as e:
                logger.warning(f"Fast contradiction detection failed: {e}, falling back to standard detection")
        elif timeline_analysis is not None:
            logger.info("Timeline analysis provided - using enhanced detection (bypassing performance optimizer)")
        
        # Performance safeguard: Set timeout for entire detection process
        timeout_seconds = self.config.get('detection_timeout_seconds', 5)  # Reduced from 30 to 5 seconds
        
        try:
            # Wrap the entire detection process in a timeout
            return await asyncio.wait_for(
                self._perform_detection_with_fallbacks(customer_complaint, ticket_data, start_time, timeline_analysis),
                timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(f"Contradiction detection timed out after {timeout_seconds} seconds")
            return self._create_timeout_fallback_result(customer_complaint, ticket_data)
        except Exception as e:
            logger.error(f"Unexpected error in contradiction detection: {e}")
            return self._create_error_fallback_result(str(e), customer_complaint, ticket_data)
    
    async def _perform_detection_with_fallbacks(
        self, 
        customer_complaint: str, 
        ticket_data: Dict, 
        start_time: float,
        timeline_analysis: Optional[Dict] = None
    ) -> ContradictionResult:
        """
        Perform contradiction detection with component-level fallback handling.
        
        Args:
            customer_complaint: Customer's complaint text
            ticket_data: Ticket information dictionary
            start_time: Start time for performance tracking
        
        Returns:
            ContradictionResult with detected conflicts and reasoning
        """
        import time
        import uuid
        from datetime import datetime
        
        # Check if contradiction detection is enabled globally
        detection_enabled = True
        if self.logic_validator and hasattr(self.logic_validator, 'config') and self.logic_validator.config:
            global_settings = getattr(self.logic_validator.config, 'global_settings', {})
            detection_enabled = global_settings.get('enable_contradiction_detection', True)
        
        if not detection_enabled:
            logger.info("Contradiction detection is disabled - returning no contradictions")
            return ContradictionResult(
                has_contradictions=False,
                contradictions=[],
                all_complaints=[],
                reasoning="Contradiction detection is disabled",
                suggested_questions=[],
                escalation_required=False
            )
        
        # Check for critical component failures
        if not self.complaint_parser and not self.logic_validator:
            logger.error("Both ComplaintParser and LogicValidator failed to initialize - cannot perform detection")
            return self._create_component_failure_result("Critical components unavailable", customer_complaint, ticket_data)
        
        # Step 1: Parse complaints with fallback handling
        complaints = []
        parsing_time = 0
        parsing_start = time.time()
        
        if self.complaint_parser:
            try:
                complaints = self.complaint_parser.parse_complaints(customer_complaint)
                parsing_time = (time.time() - parsing_start) * 1000
                logger.info(f"Parsed {len(complaints)} complaint types: {[c.category for c in complaints]}")
            except Exception as e:
                logger.error(f"ComplaintParser failed: {e}")
                # Continue with empty complaints list - logic validator may still detect issues
                complaints = []
                parsing_time = (time.time() - parsing_start) * 1000
                logger.warning("Continuing with empty complaints list due to parser failure")
        else:
            logger.warning("ComplaintParser not available - skipping complaint parsing")
        
        # Step 2: Validate logical consistency with fallback handling
        contradictions = []
        validation_time = 0
        validation_start = time.time()
        
        if self.logic_validator:
            try:
                contradictions = self.logic_validator.validate_consistency(complaints, ticket_data)
                validation_time = (time.time() - validation_start) * 1000
                logger.info(f"Found {len(contradictions)} contradictions from LogicValidator")
            except Exception as e:
                logger.error(f"LogicValidator failed: {e}")
                # Try basic fallback validation
                try:
                    contradictions = self._perform_basic_validation(complaints, ticket_data)
                    validation_time = (time.time() - validation_start) * 1000
                    logger.info(f"Fallback validation found {len(contradictions)} contradictions")
                except Exception as fallback_error:
                    logger.error(f"Fallback validation also failed: {fallback_error}")
                    contradictions = []
                    validation_time = (time.time() - validation_start) * 1000
        else:
            logger.warning("LogicValidator not available - skipping validation")
        
        # Step 2.1: Add timeline-based contradiction detection (deterministic, no LLM)
        timeline_contradiction_start = time.time()
        try:
            timeline_contradictions = self.detect_timeline_contradictions(
                customer_complaint, timeline_analysis
            )
            
            # Convert timeline contradictions to Contradiction objects
            for tc in timeline_contradictions:
                # Create a mock ComplaintType for timeline contradictions
                from .complaint_parser import ComplaintType
                
                mock_complaint = ComplaintType(
                    category="timeline_analysis",
                    confidence=1.0,
                    keywords=[tc.get("customer_statement_a", "")],
                    source_text=tc.get("customer_statement_a", "")
                )
                
                # Create a Contradiction object compatible with existing logic
                timeline_contradiction = Contradiction(
                    type=tc["type"],
                    complaint_a=mock_complaint,
                    complaint_b=None,
                    severity=tc["severity"],
                    description=tc["description"],
                    customer_response=tc.get("customer_claim", "")
                )
                
                # Add timeline-specific attributes if they don't exist
                if not hasattr(timeline_contradiction, 'timeline_evidence'):
                    timeline_contradiction.timeline_evidence = tc.get("timeline_evidence", "")
                if not hasattr(timeline_contradiction, 'explanation'):
                    timeline_contradiction.explanation = tc.get("explanation", "")
                contradictions.append(timeline_contradiction)
            
            timeline_validation_time = (time.time() - timeline_contradiction_start) * 1000
            logger.info(f"Timeline contradiction detection found {len(timeline_contradictions)} additional contradictions in {timeline_validation_time:.1f}ms")
            
        except Exception as e:
            logger.error(f"Timeline contradiction detection failed: {e}")
            # Continue without timeline contradictions rather than failing completely
        
        # Step 3: Generate reasoning with fallback handling
        reasoning_time = 0
        reasoning_start = time.time()
        has_contradictions = len(contradictions) > 0
        escalation_required = has_contradictions
        
        if has_contradictions:
            if self.reasoning_generator:
                try:
                    reasoning_result = self.reasoning_generator.generate_reasoning(contradictions)
                    reasoning = reasoning_result.explanation
                    suggested_questions = reasoning_result.suggested_questions
                except Exception as e:
                    logger.error(f"ReasoningGenerator failed: {e}")
                    # Fallback to basic reasoning
                    reasoning, suggested_questions = self._generate_basic_reasoning(contradictions)
            else:
                logger.warning("ReasoningGenerator not available - using basic reasoning")
                reasoning, suggested_questions = self._generate_basic_reasoning(contradictions)
            
            # Enhance reasoning with original data preservation for agent reference
            complaint_extraction = ticket_data.get("complaint_extraction", {})
            if complaint_extraction:
                original_reason = complaint_extraction.get("original_reason", "")
                original_summary = complaint_extraction.get("original_summary", "")
                
                if original_reason or original_summary:
                    reasoning += "<br><br><strong>Original Customer Input (for agent reference):</strong><br>"
                    if original_reason:
                        reasoning += f"<strong>Reason:</strong> {original_reason}<br>"
                    if original_summary:
                        reasoning += f"<strong>Summary:</strong> {original_summary}"
            
            logger.info("Contradictions detected - escalation required")
        else:
            reasoning = "No contradictions detected in customer complaint"
            suggested_questions = []
            logger.info("No contradictions detected")
        
        reasoning_time = (time.time() - reasoning_start) * 1000
        total_time = (time.time() - start_time) * 1000
        
        # Performance safeguard: Check if processing took too long
        max_processing_time = self.config.get('max_processing_time_ms', 2000)  # Reduced from 10s to 2s
        if total_time > max_processing_time:
            logger.warning(f"Contradiction detection took {total_time:.0f}ms (exceeds {max_processing_time}ms threshold)")
        
        # Create result
        result = ContradictionResult(
            has_contradictions=has_contradictions,
            contradictions=contradictions,
            all_complaints=complaints,
            reasoning=reasoning,
            suggested_questions=suggested_questions,
            escalation_required=escalation_required
        )
        
        # Log comprehensive contradiction event if contradictions detected
        if has_contradictions:
            try:
                event_data = {
                    "event_type": "contradiction_detected",
                    "timestamp": datetime.now().isoformat(),
                    "ticket_id": ticket_data.get("ticket_id", "unknown"),
                    "contradiction_count": len(contradictions),
                    "contradiction_types": [c.type for c in contradictions],
                    "analysis_details": {
                        "complaint_parsing": {
                            "detected_complaints": [
                                {
                                    "category": c.category,
                                    "confidence": c.confidence,
                                    "keywords": c.keywords
                                } for c in complaints
                            ],
                            "complaint_count": len(complaints),
                            "processing_time_ms": parsing_time,
                            "parser_available": self.complaint_parser is not None
                        },
                        "logic_validation": {
                            "rules_applied": (len(getattr(self.logic_validator, 'mutually_exclusive_pairs', [])) + 
                                           len(getattr(self.logic_validator, 'response_conflicts', {}))) if self.logic_validator else 0,
                            "conflicts_found": [
                                {
                                    "type": c.type,
                                    "severity": c.severity,
                                    "description": c.description
                                } for c in contradictions
                            ],
                            "processing_time_ms": validation_time,
                            "validator_available": self.logic_validator is not None
                        },
                        "reasoning_generation": {
                            "reasoning_length": len(reasoning),
                            "suggested_questions_count": len(suggested_questions),
                            "processing_time_ms": reasoning_time,
                            "generator_available": self.reasoning_generator is not None
                        }
                    },
                    "reasoning": reasoning,
                    "customer_context": {
                        "customer_id": ticket_data.get("customer_id"),
                        "original_complaint": customer_complaint,
                        "ticket_metadata": {
                            "subject": ticket_data.get("subject"),
                            "created_at": ticket_data.get("created_at"),
                            "source_system": ticket_data.get("source_system")
                        }
                    },
                    "processing_time_ms": total_time,
                    "component_status": {
                        "complaint_parser": "available" if self.complaint_parser else "failed",
                        "logic_validator": "available" if self.logic_validator else "failed", 
                        "reasoning_generator": "available" if self.reasoning_generator else "failed",
                        "initialization_errors": self.initialization_errors
                    }
                }
                
                self._log_contradiction_event(event_data)
                
                # Store comprehensive audit trail for escalated cases
                if escalation_required:
                    audit_data = {
                        "audit_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "contradiction_escalation",
                        "ticket_id": ticket_data.get("ticket_id", "unknown"),
                        "customer_input": {
                            "original_complaint": customer_complaint,
                            "processed_complaint": customer_complaint.lower().strip(),
                            "complaint_extraction": ticket_data.get("complaint_extraction", {})
                        },
                        "detected_conflicts": {
                            "contradiction_count": len(contradictions),
                            "contradiction_details": [
                                {
                                    "type": c.type,
                                    "severity": c.severity,
                                    "description": c.description,
                                    "involved_complaints": [
                                        c.complaint_a.category if c.complaint_a else None,
                                        c.complaint_b.category if c.complaint_b else None
                                    ]
                                } for c in contradictions
                            ]
                        },
                        "escalation_reasoning": {
                            "reasoning_text": reasoning,
                            "suggested_questions": suggested_questions,
                            "confidence_level": "high"  # High confidence when contradictions detected
                        },
                        "system_context": {
                            "detector_version": "1.0.0",
                            "configuration_version": getattr(self.logic_validator.config, 'version', 'unknown') if self.logic_validator and hasattr(self.logic_validator, 'config') else 'unknown',
                            "processing_time_ms": total_time,
                            "session_id": ticket_data.get("session_id"),
                            "user_agent": ticket_data.get("user_agent"),
                            "source_system": ticket_data.get("source_system"),
                            "component_status": {
                                "complaint_parser": "available" if self.complaint_parser else "failed",
                                "logic_validator": "available" if self.logic_validator else "failed",
                                "reasoning_generator": "available" if self.reasoning_generator else "failed"
                            }
                        },
                        "decision_trail": {
                            "steps": [
                                {
                                    "step_type": "complaint_parsing",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Detected {len(complaints)} complaint types",
                                    "processing_time_ms": parsing_time,
                                    "component_status": "available" if self.complaint_parser else "failed"
                                },
                                {
                                    "step_type": "logic_validation",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Found {len(contradictions)} contradictions",
                                    "processing_time_ms": validation_time,
                                    "component_status": "available" if self.logic_validator else "failed"
                                },
                                {
                                    "step_type": "reasoning_generation",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Generated reasoning and {len(suggested_questions)} questions",
                                    "processing_time_ms": reasoning_time,
                                    "component_status": "available" if self.reasoning_generator else "failed"
                                }
                            ],
                            "final_decision": {
                                "decision": "escalate",
                                "reason": "contradictions_detected",
                                "confidence": "high",
                                "timestamp": datetime.now().isoformat()
                            }
                        },
                        "escalation_required": True
                    }
                    
                    self._store_audit_trail(audit_data)
            except Exception as logging_error:
                logger.error(f"Failed to log contradiction event: {logging_error}")
                # Continue processing even if logging fails
        
        return result
    
    def _perform_basic_validation(self, complaints: List, ticket_data: Dict) -> List:
        """
        Perform basic contradiction validation when LogicValidator fails.
        
        Args:
            complaints: List of parsed complaint types
            ticket_data: Ticket information dictionary
        
        Returns:
            List of basic contradictions detected
        """
        from .logic_validator import Contradiction
        
        contradictions = []
        complaint_categories = {c.category: c for c in complaints}
        
        # Basic mutually exclusive checks
        basic_exclusive_pairs = [
            ("accessibility", "oversold"),
            ("paid_again", "oversold")
        ]
        
        for category_a, category_b in basic_exclusive_pairs:
            if category_a in complaint_categories and category_b in complaint_categories:
                contradiction = Contradiction(
                    type="mutually_exclusive",
                    complaint_a=complaint_categories[category_a],
                    complaint_b=complaint_categories[category_b],
                    severity="high",
                    description=f"Customer claims both {category_a} and {category_b} issues, which are mutually exclusive"
                )
                contradictions.append(contradiction)
        
        # Basic response mismatch checks
        ticket_description = ticket_data.get("description", "").lower()
        ticket_notes = ticket_data.get("notes", "").lower()
        full_text = f"{ticket_description} {ticket_notes}"
        
        if ("were you able to park? yes" in full_text or "were you able to park?yes" in full_text):
            for complaint in complaints:
                if complaint.category in ["accessibility", "oversold"]:
                    contradiction = Contradiction(
                        type="response_mismatch",
                        complaint_a=complaint,
                        customer_response="were you able to park? yes",
                        severity="high",
                        description=f"Customer claims {complaint.category} issues but confirms they were able to park"
                    )
                    contradictions.append(contradiction)
        
        return contradictions
    
    def _generate_basic_reasoning(self, contradictions: List) -> tuple:
        """
        Generate basic reasoning when ReasoningGenerator fails.
        
        Args:
            contradictions: List of detected contradictions
        
        Returns:
            Tuple of (reasoning_text, suggested_questions)
        """
        if not contradictions:
            return "No contradictions detected in customer complaint", []
        
        reasoning_parts = [
            "<strong>Contradictory information detected in customer complaint.</strong><br><br>",
            f"The following conflicts require human review to clarify the customer's actual situation:<br><br>"
        ]
        
        suggested_questions = []
        
        for i, contradiction in enumerate(contradictions, 1):
            reasoning_parts.append(f"<strong>Contradiction {i}:</strong> {contradiction.description}<br><br>")
            
            if contradiction.type == "mutually_exclusive":
                suggested_questions.append(
                    f"Can you clarify which issue was the primary reason you couldn't use your booking: "
                    f"{contradiction.complaint_a.category} or {contradiction.complaint_b.category}?"
                )
            elif contradiction.type == "response_mismatch":
                suggested_questions.append(
                    f"You mentioned {contradiction.complaint_a.category} issues but also indicated you were able to park. "
                    f"Can you explain what happened?"
                )
            elif contradiction.type == "logical_contradiction":
                suggested_questions.append(
                    "You mentioned that you couldn't park, but also described retrieving your car. "
                    "Can you clarify what actually happened with your parking experience?"
                )
            elif contradiction.type in ["timeline_math_contradiction", "timeline_math_error"]:
                suggested_questions.append(
                    "There appears to be some inconsistency in the timeline of your parking experience. "
                    "Can you help us understand exactly what happened and when?"
                )
        
        if not suggested_questions:
            suggested_questions = ["Please clarify the details of your parking experience."]
        
        reasoning_parts.append(
            "<strong>Action required:</strong> Contact the customer to clarify their situation "
            "before making a refund decision."
        )
        
        return "".join(reasoning_parts), suggested_questions
    
    def _create_timeout_fallback_result(self, customer_complaint: str, ticket_data: Dict) -> ContradictionResult:
        """Create fallback result when detection times out."""
        from datetime import datetime
        
        error_event = {
            "event_type": "contradiction_detection_timeout",
            "timestamp": datetime.now().isoformat(),
            "ticket_id": ticket_data.get("ticket_id", "unknown") if ticket_data else "unknown",
            "escalation_required": True
        }
        self._log_contradiction_event(error_event)
        
        return ContradictionResult(
            has_contradictions=True,
            contradictions=[],
            all_complaints=[],
            reasoning="Contradiction detection timed out. Escalating for safety to ensure proper review.",
            suggested_questions=["Please review the customer's complaint manually due to system timeout."],
            escalation_required=True
        )
    
    def _create_error_fallback_result(self, error_msg: str, customer_complaint: str, ticket_data: Dict) -> ContradictionResult:
        """Create fallback result when detection encounters an error."""
        from datetime import datetime
        
        error_event = {
            "event_type": "contradiction_detection_error",
            "timestamp": datetime.now().isoformat(),
            "ticket_id": ticket_data.get("ticket_id", "unknown") if ticket_data else "unknown",
            "error_message": error_msg,
            "escalation_required": True
        }
        self._log_contradiction_event(error_event)
        
        return ContradictionResult(
            has_contradictions=True,
            contradictions=[],
            all_complaints=[],
            reasoning=f"Contradiction detection failed: {error_msg}. Escalating for safety.",
            suggested_questions=["Please review the customer's complaint manually."],
            escalation_required=True
        )
    
    def _create_component_failure_result(self, failure_reason: str, customer_complaint: str, ticket_data: Dict) -> ContradictionResult:
        """Create fallback result when critical components fail."""
        from datetime import datetime
        
        error_event = {
            "event_type": "contradiction_component_failure",
            "timestamp": datetime.now().isoformat(),
            "ticket_id": ticket_data.get("ticket_id", "unknown") if ticket_data else "unknown",
            "failure_reason": failure_reason,
            "escalation_required": True
        }
        self._log_contradiction_event(error_event)
        
        return ContradictionResult(
            has_contradictions=True,
            contradictions=[],
            all_complaints=[],
            reasoning=f"Critical contradiction detection components failed: {failure_reason}. Escalating for safety.",
            suggested_questions=["Please review the customer's complaint manually due to system component failure."],
            escalation_required=True
        )
    
    def reload_configuration(self) -> bool:
        """
        Reload contradiction rules configuration.
        
        Returns:
            True if configuration was reloaded successfully, False otherwise
        """
        return self.logic_validator.reload_configuration()
    
    def add_contradiction_rule(self, rule_id: str, rule_name: str, rule_type: str,
                             complaint_a: str, complaint_b: Optional[str] = None,
                             conflicting_responses: Optional[List[str]] = None,
                             severity: str = "medium", description: str = "",
                             priority: int = 100) -> bool:
        """
        Add a new contradiction rule.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_name: Human-readable name for the rule
            rule_type: Type of rule ("mutually_exclusive", "response_mismatch", "logical_conflict")
            complaint_a: First complaint type
            complaint_b: Second complaint type (for mutually_exclusive and logical_conflict)
            conflicting_responses: List of conflicting responses (for response_mismatch)
            severity: Severity level ("high", "medium", "low")
            description: Human-readable description of the conflict
            priority: Priority for rule application (lower = higher priority)
        
        Returns:
            True if rule was added successfully, False otherwise
        """
        return self.logic_validator.add_contradiction_rule(
            rule_id, rule_name, rule_type, complaint_a, complaint_b,
            conflicting_responses, severity, description, priority
        )
    
    def remove_contradiction_rule(self, rule_id: str) -> bool:
        """
        Remove a contradiction rule.
        
        Args:
            rule_id: ID of rule to remove
        
        Returns:
            True if rule was removed successfully, False otherwise
        """
        return self.logic_validator.remove_contradiction_rule(rule_id)
    
    def get_configuration_info(self) -> Dict:
        """
        Get information about the current contradiction rules configuration.
        
        Returns:
            Dictionary with configuration details
        """
        return self.logic_validator.get_configuration_info()
    
    def save_configuration(self, path: Optional[str] = None) -> bool:
        """
        Save current configuration to file.
        
        Args:
            path: Optional path to save to
        
        Returns:
            True if configuration was saved successfully, False otherwise
        """
        if not self.logic_validator.config_manager or not self.logic_validator.config:
            logger.warning("No configuration available to save")
            return False
        
        try:
            self.logic_validator.config_manager.save_config(
                self.logic_validator.config, path
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def _log_contradiction_event(self, event_data: Dict) -> None:
        """
        Log comprehensive contradiction detection event.
        
        Args:
            event_data: Dictionary containing all event details for logging
        """
        try:
            # Log the event with structured data
            logger.info(
                f"Contradiction Event: {event_data.get('event_type', 'unknown')} "
                f"for ticket {event_data.get('ticket_id', 'unknown')}"
            )
            
            # Log detailed event data as structured information
            logger.debug(f"Event details: {event_data}")
            
        except Exception as e:
            logger.error(f"Failed to log contradiction event: {e}")
    
    def _store_audit_trail(self, audit_data: Dict) -> None:
        """
        Store comprehensive audit trail for contradiction cases.
        
        Args:
            audit_data: Dictionary containing complete audit trail information
        """
        try:
            # Store audit trail (in a real implementation, this would go to a database)
            logger.info(
                f"Audit Trail: {audit_data.get('event_type', 'unknown')} "
                f"for ticket {audit_data.get('ticket_id', 'unknown')} "
                f"with audit ID {audit_data.get('audit_id', 'unknown')}"
            )
            
            # Log audit trail details
            logger.debug(f"Audit trail data: {audit_data}")
            
        except Exception as e:
            logger.error(f"Failed to store audit trail: {e}")