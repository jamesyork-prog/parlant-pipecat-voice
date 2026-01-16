"""
ScenarioClassifier component for rule-based "Paid Again" scenario classification.

This module provides deterministic classification logic that distinguishes between
genuine "Paid Again" scenarios and legitimate additional charges using keyword-based
pattern matching and timeline analysis. Minimizes LLM usage by handling clear-cut
cases with deterministic rules.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from ..booking_analysis.timeline_analyzer import TimelineAnalysis
from .classification_config import ClassificationConfigManager, ScenarioRule

logger = logging.getLogger(__name__)


@dataclass
class ScenarioClassification:
    """
    Result of scenario classification with decision guidance.
    
    This data model contains the classification result and supporting evidence
    for "Paid Again" scenario determination.
    """
    scenario_type: str  # "accidental_double_booking", "early_arrival_overstay", etc.
    confidence: float   # 0.0 to 1.0
    reasoning: str      # Human-readable explanation
    
    # Decision guidance
    recommended_action: str  # "approve_refund", "deny_legitimate_charge", "escalate"
    refund_eligible: bool
    
    # Supporting evidence
    key_factors: List[str]
    timeline_evidence: Optional[TimelineAnalysis] = None
    contradictions: List[str] = None
    
    def __post_init__(self):
        """Initialize contradictions as empty list if None."""
        if self.contradictions is None:
            self.contradictions = []


class ScenarioClassifier:
    """
    Rule-based classification logic for "Paid Again" scenarios.
    
    This class uses deterministic patterns and timeline data to classify scenarios
    with minimal LLM usage. LLM is only invoked when deterministic rules cannot
    provide sufficient confidence.
    """
    
    def __init__(self, config: Optional[Dict] = None, config_manager: Optional[ClassificationConfigManager] = None):
        """
        Initialize scenario classifier with configurable thresholds.
        
        Args:
            config: Configuration dict with classification settings (legacy support)
            config_manager: ClassificationConfigManager instance for extensible rules
        """
        if config is None:
            config = {}
            
        self.webhook_mode = config.get("webhook_mode", False)
        
        # Initialize configuration manager
        if config_manager is None:
            config_path = config.get("config_path")
            self.config_manager = ClassificationConfigManager(config_path)
        else:
            self.config_manager = config_manager
        
        # Load configuration
        # Priority: passed config > configuration file > defaults
        try:
            self.classification_config = self.config_manager.load_config()
            file_confidence_thresholds = {
                "webhook_automation": self.classification_config.confidence_thresholds.webhook_automation,
                "interactive_chat": self.classification_config.confidence_thresholds.interactive_chat
            }
            logger.info(f"Loaded classification config: {len(self.classification_config.scenario_rules)} rules")
        except Exception as e:
            logger.warning(f"Failed to load classification config: {e}. Using defaults.")
            self.classification_config = None
            file_confidence_thresholds = {
                "webhook_automation": 0.85,
                "interactive_chat": 0.70
            }
        
        # Allow passed config to override file config
        passed_thresholds = config.get("confidence_thresholds", {})
        self.confidence_thresholds = {
            "webhook_automation": passed_thresholds.get("webhook_automation", file_confidence_thresholds["webhook_automation"]),
            "interactive_chat": passed_thresholds.get("interactive_chat", file_confidence_thresholds["interactive_chat"])
        }
        
        # Keyword patterns for different scenarios (legacy support)
        self._init_keyword_patterns()
        
        logger.info(f"ScenarioClassifier initialized (webhook_mode={self.webhook_mode})")
    
    def _init_keyword_patterns(self):
        """Initialize keyword patterns for scenario detection."""
        
        # Accidental double booking patterns
        self.double_booking_patterns = [
            r'booked.*twice',
            r'double.*booking',
            r'two.*bookings?',
            r'multiple.*bookings?',
            r'accidentally.*booked.*again',
            r'thought.*first.*didn\'t.*work',
            r'made.*another.*booking',
            r'duplicate.*reservation',
            r'booking.*didn\'t.*go.*through.*so.*booked.*again'
        ]
        
        # Entry failure patterns (complete pass unused)
        self.entry_failure_patterns = [
            r'qr.*code.*(?:didn\'t|wouldn\'t|not).*work',
            r'qr.*(?:rejected|failed|invalid)',
            r'scanner.*(?:rejected|broken|not.*working|down)',
            r'attendant.*said.*(?:pass.*)?invalid',
            r'pass.*rejected',
            r'couldn\'t.*get.*in',
            r'never.*entered.*garage',
            r'gates?.*(?:wouldn\'t|didn\'t).*open',
            r'pass.*wouldn\'t.*scan',
            r'qr.*(?:wouldn\'t|didn\'t).*scan',
            r'had.*to.*pay.*instead',
            r'forced.*to.*pay.*gate.*rate',
            r'pay.*the.*gate.*rate.*instead'
        ]
        
        # Exit complication patterns (used parking but had departure issues)
        self.exit_complication_patterns = [
            r'parked.*but.*(?:couldn\'t|had.*trouble).*(?:exit|leave|get.*out)',
            r'got.*in.*(?:fine|ok).*but.*(?:exit|leaving).*(?:problem|issue)',
            r'used.*pass.*to.*enter.*but.*exit.*(?:failed|broken)',
            r'successfully.*parked.*but.*(?:qr|scanner).*(?:at.*exit|leaving)',
            r'entered.*garage.*but.*had.*to.*pay.*to.*(?:exit|leave)',
            r'parking.*worked.*but.*departure.*(?:issue|problem)',
            r'got.*parking.*but.*exit.*gate.*(?:broken|not.*working)'
        ]
        
        # Early arrival indicators
        self.early_arrival_patterns = [
            r'arrived.*early',
            r'got.*there.*before.*(?:my|the).*time',
            r'came.*before.*(?:booking|reservation).*start',
            r'parking.*started.*before.*(?:my|the).*slot',
            r'early.*arrival',
            r'before.*(?:my|the).*(?:booking|reservation).*time'
        ]
        
        # No parking occurred patterns
        self.no_parking_patterns = [
            r'never.*parked',
            r'didn\'t.*park',
            r'couldn\'t.*use.*(?:the.*)?parking',
            r'no.*parking.*occurred',
            r'pass.*completely.*unused',
            r'never.*got.*to.*use.*(?:the.*)?pass',
            r'pass.*was.*completely.*unused',
            r'never.*used.*(?:the.*)?pass'
        ]
        
        # Timeline contradiction patterns - very specific to avoid false positives
        self.contradiction_patterns = [
            # Claims no parking but describes retrieving car from parking
            (r'(?:couldn\'t|never).*park.*at.*all', r'retrieved.*(?:my|the).*car.*from.*(?:the.*)?garage'),
            (r'(?:didn\'t|never).*use.*(?:the.*)?pass', r'parked.*for.*(?:\d+.*)?(?:hours|minutes)'),
            
            # Claims complete entry failure but describes successful parking duration
            (r'(?:couldn\'t|never).*(?:get.*in|enter).*at.*all', r'parked.*(?:for.*)?(?:\d+.*)?(?:hours|minutes)'),
        ]
    
    def classify_paid_again_scenario(
        self,
        customer_complaint: str,
        timeline_analysis: Optional[TimelineAnalysis] = None,
        booking_info: Optional[Dict] = None
    ) -> ScenarioClassification:
        """
        RULE-BASED classification using keyword patterns and timeline logic.
        
        LLM is ONLY used when:
        - Timeline data is missing/ambiguous AND keywords are unclear
        - Multiple scenarios match with equal confidence
        - Customer language requires complex natural language understanding
        
        For clear cases (overstay math, keyword matches), uses deterministic rules.
        
        Args:
            customer_complaint: Customer's description of the issue
            timeline_analysis: Optional timeline analysis results
            booking_info: Optional booking information
            
        Returns:
            ScenarioClassification with scenario type and confidence
        """
        logger.info("Starting rule-based scenario classification")
        
        if not customer_complaint or not customer_complaint.strip():
            logger.warning("Empty customer complaint provided")
            return self._create_low_confidence_result(
                "insufficient_data",
                "No customer complaint text provided for analysis"
            )
        
        complaint_lower = customer_complaint.lower()
        key_factors = []
        
        # Step 1: Check for multiple booking IDs (only if booking_info indicates multiple bookings)
        # Don't check double booking patterns in text alone as they can be misleading
        double_booking_confidence = 0.0
        if booking_info and booking_info.get("multiple_bookings"):
            double_booking_confidence = self._check_double_booking_indicators(
                customer_complaint, booking_info
            )
        
        if double_booking_confidence >= 0.9:
            logger.info("High confidence accidental double booking detected")
            return ScenarioClassification(
                scenario_type="accidental_double_booking",
                confidence=double_booking_confidence,
                reasoning=(
                    "Customer provided multiple booking IDs or explicitly mentioned "
                    "booking twice for the same time period. This is a clear case of "
                    "accidental double booking where the customer should receive a "
                    "refund for the duplicate booking."
                ),
                recommended_action="approve_refund",
                refund_eligible=True,
                key_factors=[
                    "Multiple booking IDs detected",
                    "Customer explicitly mentioned double booking",
                    "Clear accidental duplicate reservation"
                ],
                timeline_evidence=timeline_analysis
            )
        
        # Step 2: Check for timeline contradictions first (high priority)
        contradictions = self._detect_timeline_contradictions(
            complaint_lower, timeline_analysis
        )
        if contradictions:
            logger.info("Timeline contradictions detected")
            return ScenarioClassification(
                scenario_type="contradictory_statements",
                confidence=0.9,
                reasoning=(
                    "Customer statements contain logical contradictions about their "
                    "parking experience. These inconsistencies require human review "
                    "to clarify the actual sequence of events."
                ),
                recommended_action="escalate",
                refund_eligible=False,
                key_factors=[
                    "Contradictory customer statements detected",
                    "Timeline inconsistencies require clarification"
                ],
                timeline_evidence=timeline_analysis,
                contradictions=contradictions
            )
        
        # Step 3: Check timeline analysis for early arrival overstay
        if timeline_analysis:
            overstay_result = self._check_early_arrival_overstay(
                timeline_analysis, complaint_lower
            )
            if overstay_result:
                logger.info("Early arrival overstay scenario detected")
                return overstay_result
        
        # Step 4: Check for exit complications first (higher priority when both entry and exit issues present)
        exit_complication_confidence = self._check_exit_complication_patterns(complaint_lower)
        if exit_complication_confidence >= 0.8:
            logger.info("Exit complication scenario detected")
            return ScenarioClassification(
                scenario_type="exit_complication_poor_experience",
                confidence=exit_complication_confidence,
                reasoning=(
                    "Customer successfully used their ParkWhiz pass to park but "
                    "encountered technical difficulties when trying to exit. Since "
                    "they received parking value from their purchase, this is a "
                    "poor experience case rather than a refund situation."
                ),
                recommended_action="deny_legitimate_charge",
                refund_eligible=False,
                key_factors=[
                    "Successfully entered and parked",
                    "Technical issues at exit/departure",
                    "Received parking value from ParkWhiz purchase"
                ],
                timeline_evidence=timeline_analysis
            )
        
        # Step 5: Check for entry failure (complete pass unused) - only if no exit complications
        entry_failure_confidence = self._check_entry_failure_patterns(complaint_lower)
        if entry_failure_confidence >= 0.8:
            logger.info("Entry failure scenario detected")
            return ScenarioClassification(
                scenario_type="entry_failure_complete_unused",
                confidence=entry_failure_confidence,
                reasoning=(
                    "Customer was unable to use their ParkWhiz pass at all due to "
                    "technical issues (QR code failure, scanner problems, or attendant "
                    "rejection). Since no parking value was received, a full refund "
                    "is appropriate."
                ),
                recommended_action="approve_refund",
                refund_eligible=True,
                key_factors=[
                    "QR code or scanner technical failure",
                    "Pass rejected at entry",
                    "No parking value received"
                ],
                timeline_evidence=timeline_analysis
            )
        
        # Step 6: Check for no parking scenarios
        no_parking_confidence = self._check_no_parking_patterns(complaint_lower)
        if no_parking_confidence >= 0.8:
            logger.info("No parking scenario detected")
            return ScenarioClassification(
                scenario_type="no_parking_occurred",
                confidence=no_parking_confidence,
                reasoning=(
                    "Customer indicates that no parking occurred and their pass "
                    "was completely unused. If confirmed, this warrants a full refund "
                    "as no parking value was received."
                ),
                recommended_action="approve_refund",
                refund_eligible=True,
                key_factors=[
                    "Customer claims no parking occurred",
                    "Pass appears completely unused",
                    "No parking value received"
                ],
                timeline_evidence=timeline_analysis
            )
        
        # Step 7: Apply configurable rules if available
        if self.classification_config:
            configurable_result = self._apply_configurable_rules(
                customer_complaint, timeline_analysis
            )
            if configurable_result and configurable_result.confidence >= 0.6:
                logger.info(f"Lower-confidence configurable rule matched: {configurable_result.scenario_type}")
                return configurable_result
        
        # Step 8: If no clear pattern matches, return uncertain result
        logger.info("No clear scenario pattern detected, returning uncertain result")
        return self._create_uncertain_result(
            customer_complaint, timeline_analysis, key_factors
        )
    
    def _check_double_booking_indicators(
        self, 
        customer_complaint: str, 
        booking_info: Optional[Dict]
    ) -> float:
        """
        Check for indicators of accidental double booking.
        
        Args:
            customer_complaint: Customer's description
            booking_info: Booking information dict
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        complaint_lower = customer_complaint.lower()
        
        # Check for multiple booking ID patterns in complaint
        # More flexible patterns to catch various ID formats
        booking_id_patterns = [
            r'(?:booking|reservation).*(?:id|number).*\w+.*(?:and|&).*\w+',  # More flexible word patterns
            r'\w{5,}.*(?:and|&).*\w{5,}',  # Two word sequences of 5+ chars
            r'pw-\w+.*(?:and|&).*pw-\w+',  # Two PW- format IDs
            r'\{id1\}.*(?:and|&).*\{id2\}',  # Template placeholders
        ]
        
        for pattern in booking_id_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.95)
                logger.debug(f"Multiple booking IDs detected: {pattern}")
        
        # Check for explicit double booking language
        for pattern in self.double_booking_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.9)
                logger.debug(f"Double booking language detected: {pattern}")
        
        # Check if booking_info contains multiple booking references
        if booking_info and booking_info.get("multiple_bookings"):
            confidence = max(confidence, 0.95)  # Higher confidence for explicit flag
            logger.debug("Multiple bookings flag set in booking_info")
        
        # Special case: if complaint contains words like "twice", "two", "both" with booking context
        # Only apply if we don't have stronger indicators of other scenarios
        double_indicators = [
            r'booked.*twice',
            r'made.*two.*booking',
            r'both.*booking'
        ]
        
        # Only check these if we don't have entry failure patterns
        has_entry_failure = any(re.search(pattern, complaint_lower) for pattern in self.entry_failure_patterns)
        
        if not has_entry_failure:
            for pattern in double_indicators:
                if re.search(pattern, complaint_lower):
                    confidence = max(confidence, 0.8)  # Lower confidence to avoid conflicts
                    logger.debug(f"Double booking indicator detected: {pattern}")
        
        return confidence
    
    def _check_early_arrival_overstay(
        self, 
        timeline_analysis: TimelineAnalysis, 
        complaint_lower: str
    ) -> Optional[ScenarioClassification]:
        """
        Check for early arrival overstay scenario using timeline analysis.
        
        Args:
            timeline_analysis: Timeline analysis results
            complaint_lower: Lowercase customer complaint
            
        Returns:
            ScenarioClassification if early arrival detected, None otherwise
        """
        # Check for early arrival with overstay (strongest case)
        if timeline_analysis.has_early_arrival and timeline_analysis.has_overstay:
            # Check if customer admits to early arrival
            early_arrival_mentioned = any(
                re.search(pattern, complaint_lower) 
                for pattern in self.early_arrival_patterns
            )
            
            # Calculate confidence based on timeline data and customer admission
            confidence = 0.95  # High confidence for mathematical timeline analysis
            
            if early_arrival_mentioned:
                confidence = 0.98  # Even higher if customer admits early arrival
            
            total_extra_minutes = timeline_analysis.early_arrival_minutes + timeline_analysis.overstay_minutes
            
            return ScenarioClassification(
                scenario_type="early_arrival_overstay",
                confidence=confidence,
                reasoning=(
                    f"Customer arrived {timeline_analysis.early_arrival_minutes} minutes early "
                    f"and stayed {timeline_analysis.overstay_minutes} minutes past their booking end time. "
                    f"This created {total_extra_minutes} minutes of additional parking time beyond "
                    f"what was originally booked. The additional charges are legitimate fees for "
                    f"the extra parking time used."
                ),
                recommended_action="deny_legitimate_charge",
                refund_eligible=False,
                key_factors=[
                    f"Early arrival: {timeline_analysis.early_arrival_minutes} minutes",
                    f"Overstay: {timeline_analysis.overstay_minutes} minutes",
                    f"Total extra parking: {total_extra_minutes} minutes",
                    "Mathematical timeline analysis confirms additional charges"
                ],
                timeline_evidence=timeline_analysis
            )
        
        # Check for early arrival without overstay (customer mentions early arrival)
        elif timeline_analysis.has_early_arrival:
            early_arrival_mentioned = any(
                re.search(pattern, complaint_lower) 
                for pattern in self.early_arrival_patterns
            )
            
            if early_arrival_mentioned:
                # Customer admits early arrival - analyze for potential charges
                confidence = 0.9  # High confidence when customer admits early arrival
                
                return ScenarioClassification(
                    scenario_type="early_arrival_analysis",
                    confidence=confidence,
                    reasoning=(
                        f"Customer arrived {timeline_analysis.early_arrival_minutes} minutes early "
                        f"but did not overstay their booking end time. While no overstay charges "
                        f"apply, the early arrival may have resulted in additional parking time "
                        f"that could justify extra fees depending on the facility's policies."
                    ),
                    recommended_action="deny_legitimate_charge",
                    refund_eligible=False,
                    key_factors=[
                        f"Early arrival: {timeline_analysis.early_arrival_minutes} minutes",
                        "Customer admits to early arrival",
                        "Timeline analysis confirms early parking usage",
                        "No overstay detected"
                    ],
                    timeline_evidence=timeline_analysis
                )
        
        return None
    
    def _check_entry_failure_patterns(self, complaint_lower: str) -> float:
        """
        Check for entry failure patterns in customer complaint.
        
        Args:
            complaint_lower: Lowercase customer complaint
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        for pattern in self.entry_failure_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.85)
                logger.debug(f"Entry failure pattern detected: {pattern}")
        
        # Higher confidence if multiple entry failure indicators
        entry_failure_count = sum(
            1 for pattern in self.entry_failure_patterns
            if re.search(pattern, complaint_lower)
        )
        
        if entry_failure_count >= 2:
            confidence = min(confidence + 0.1, 1.0)
        
        return confidence
    
    def _check_exit_complication_patterns(self, complaint_lower: str) -> float:
        """
        Check for exit complication patterns in customer complaint.
        
        Args:
            complaint_lower: Lowercase customer complaint
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        # Check for combination of successful entry + exit problems first (highest priority)
        successful_entry_patterns = [
            r'got.*in.*(?:fine|ok|successfully)',
            r'entered.*(?:garage|lot).*(?:fine|ok|successfully)',
            r'pass.*worked.*(?:to.*)?(?:enter|get.*in)',
            r'got.*in.*(?:fine|ok).*with.*(?:my.*)?pass',
            r'entered.*(?:without.*problems|successfully)',
            r'successfully.*parked.*using'
        ]
        
        exit_problem_patterns = [
            r'(?:but|however).*(?:couldn\'t|had.*trouble).*(?:exit|leave|get.*out)',
            r'(?:but|however).*exit.*(?:scanner|gate).*(?:broken|not.*working|wouldn\'t.*open)',
            r'(?:but|however).*qr.*(?:code.*)?(?:didn\'t.*work|wouldn\'t.*scan).*(?:at.*)?exit',
            r'(?:but|however).*had.*to.*pay.*(?:to.*leave|again)',
            r'(?:but|however).*exit.*(?:problem|issue|failed)'
        ]
        
        successful_entry = any(re.search(p, complaint_lower) for p in successful_entry_patterns)
        exit_problems = any(re.search(p, complaint_lower) for p in exit_problem_patterns)
        
        if successful_entry and exit_problems:
            confidence = max(confidence, 0.95)  # Very high confidence for clear combination
            logger.debug("Successful entry + exit problems combination detected")
        
        # Check for direct exit complication patterns
        for pattern in self.exit_complication_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.85)
                logger.debug(f"Exit complication pattern detected: {pattern}")
        
        # Additional exit-specific patterns
        exit_specific_patterns = [
            r'exit.*(?:scanner|gate).*(?:broken|not.*working)',
            r'leaving.*(?:problem|issue)',
            r'departure.*(?:problem|issue)'
        ]
        
        for pattern in exit_specific_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.8)
                logger.debug(f"Exit-specific pattern detected: {pattern}")
        
        return confidence
    
    def _check_no_parking_patterns(self, complaint_lower: str) -> float:
        """
        Check for no parking patterns in customer complaint.
        
        Args:
            complaint_lower: Lowercase customer complaint
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        confidence = 0.0
        
        for pattern in self.no_parking_patterns:
            if re.search(pattern, complaint_lower):
                confidence = max(confidence, 0.8)
                logger.debug(f"No parking pattern detected: {pattern}")
        
        return confidence
    
    def _detect_timeline_contradictions(
        self, 
        complaint_lower: str, 
        timeline_analysis: Optional[TimelineAnalysis]
    ) -> List[str]:
        """
        Detect logical contradictions in customer statements.
        
        Args:
            complaint_lower: Lowercase customer complaint
            timeline_analysis: Optional timeline analysis
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        # Check for linguistic contradictions
        for negative_pattern, positive_pattern in self.contradiction_patterns:
            if (re.search(negative_pattern, complaint_lower) and 
                re.search(positive_pattern, complaint_lower)):
                contradictions.append(
                    f"Customer claims '{negative_pattern}' but also describes '{positive_pattern}'"
                )
        
        # Check for timeline-based contradictions
        if timeline_analysis and timeline_analysis.actual_parking_duration > 0:
            # If timeline shows parking occurred but customer claims no parking
            no_parking_claims = any(
                re.search(pattern, complaint_lower) 
                for pattern in self.no_parking_patterns
            )
            
            if no_parking_claims:
                contradictions.append(
                    f"Customer claims no parking occurred but timeline shows "
                    f"{timeline_analysis.actual_parking_duration} minutes of parking"
                )
        
        return contradictions
    
    def _apply_configurable_rules(
        self,
        customer_complaint: str,
        timeline_analysis: Optional[TimelineAnalysis]
    ) -> Optional[ScenarioClassification]:
        """
        Apply configurable scenario rules from configuration.
        
        Args:
            customer_complaint: Customer's description of the issue
            timeline_analysis: Optional timeline analysis results
            
        Returns:
            ScenarioClassification if rule matches with sufficient confidence, None otherwise
        """
        if not self.classification_config:
            return None
        
        complaint_lower = customer_complaint.lower()
        best_match = None
        best_confidence = 0.0
        
        # Get all enabled rules sorted by priority
        all_rules = self.classification_config.get_all_enabled_rules()
        
        for rule in all_rules:
            confidence = self._evaluate_rule(rule, complaint_lower, timeline_analysis)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = rule
                
                # Early exit for very high confidence matches
                if confidence >= 0.95:
                    break
        
        if best_match and best_confidence >= 0.6:  # Minimum threshold for configurable rules
            return self._create_rule_based_result(best_match, best_confidence, timeline_analysis)
        
        return None
    
    def _evaluate_rule(
        self,
        rule: ScenarioRule,
        complaint_lower: str,
        timeline_analysis: Optional[TimelineAnalysis]
    ) -> float:
        """
        Evaluate a single configurable rule against the complaint.
        
        Args:
            rule: ScenarioRule to evaluate
            complaint_lower: Lowercase customer complaint
            timeline_analysis: Optional timeline analysis
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not rule.enabled:
            return 0.0
        
        matches = 0
        total_patterns = len(rule.patterns)
        
        # Check each pattern in the rule
        for pattern in rule.patterns:
            try:
                if re.search(pattern, complaint_lower, re.IGNORECASE):
                    matches += 1
            except re.error as e:
                logger.warning(f"Invalid regex pattern in rule {rule.id}: {pattern} - {e}")
                continue
        
        if matches == 0:
            return 0.0
        
        # Calculate confidence based on matches
        match_ratio = matches / total_patterns
        confidence = rule.confidence_base + (rule.confidence_boost * match_ratio)
        
        # Cap at 1.0
        confidence = min(confidence, 1.0)
        
        # Apply timeline-specific boosts for certain scenario types
        if timeline_analysis and confidence > 0.0:
            confidence = self._apply_timeline_boost(rule, confidence, timeline_analysis)
        
        logger.debug(f"Rule {rule.id}: {matches}/{total_patterns} patterns matched, confidence={confidence:.3f}")
        
        return confidence
    
    def _apply_timeline_boost(
        self,
        rule: ScenarioRule,
        base_confidence: float,
        timeline_analysis: TimelineAnalysis
    ) -> float:
        """
        Apply timeline-specific confidence boosts for certain scenario types.
        
        Args:
            rule: ScenarioRule being evaluated
            base_confidence: Base confidence from pattern matching
            timeline_analysis: Timeline analysis results
            
        Returns:
            Adjusted confidence score
        """
        # Early arrival scenarios get boost if timeline confirms early arrival
        if rule.scenario_type == "early_arrival_analysis" and timeline_analysis.has_early_arrival:
            return min(base_confidence + 0.1, 1.0)
        
        # Overstay scenarios get boost if timeline confirms overstay
        if "overstay" in rule.scenario_type and timeline_analysis.has_overstay:
            return min(base_confidence + 0.15, 1.0)
        
        # Contradiction scenarios get boost if timeline is inconsistent
        if rule.scenario_type == "contradictory_statements" and not timeline_analysis.is_timeline_consistent:
            return min(base_confidence + 0.1, 1.0)
        
        return base_confidence
    
    def _create_rule_based_result(
        self,
        rule: ScenarioRule,
        confidence: float,
        timeline_analysis: Optional[TimelineAnalysis]
    ) -> ScenarioClassification:
        """
        Create a ScenarioClassification result from a matched configurable rule.
        
        Args:
            rule: Matched ScenarioRule
            confidence: Calculated confidence score
            timeline_analysis: Optional timeline analysis
            
        Returns:
            ScenarioClassification based on the rule
        """
        # Determine recommended action based on scenario type
        if rule.scenario_type in ["accidental_double_booking", "entry_failure_complete_unused", "no_parking_occurred"]:
            recommended_action = "approve_refund"
            refund_eligible = True
        elif rule.scenario_type in ["early_arrival_overstay", "exit_complication_poor_experience"]:
            recommended_action = "deny_legitimate_charge"
            refund_eligible = False
        elif rule.scenario_type == "contradictory_statements":
            recommended_action = "escalate"
            refund_eligible = False
        else:
            # Unknown scenario type - be conservative
            recommended_action = "escalate"
            refund_eligible = False
        
        # Create reasoning text
        reasoning = f"Configurable rule '{rule.name}' matched with {confidence:.1%} confidence. {rule.description}"
        
        # Create key factors
        key_factors = [
            f"Matched rule: {rule.name}",
            f"Rule confidence: {confidence:.1%}",
            f"Rule priority: {rule.priority}"
        ]
        
        if rule.description:
            key_factors.append(f"Rule description: {rule.description}")
        
        return ScenarioClassification(
            scenario_type=rule.scenario_type,
            confidence=confidence,
            reasoning=reasoning,
            recommended_action=recommended_action,
            refund_eligible=refund_eligible,
            key_factors=key_factors,
            timeline_evidence=timeline_analysis
        )
    
    def _create_low_confidence_result(
        self, 
        scenario_type: str, 
        reasoning: str
    ) -> ScenarioClassification:
        """Create a low confidence classification result."""
        return ScenarioClassification(
            scenario_type=scenario_type,
            confidence=0.3,
            reasoning=reasoning,
            recommended_action="escalate",
            refund_eligible=False,
            key_factors=["Insufficient data for classification"]
        )
    
    def _create_uncertain_result(
        self, 
        customer_complaint: str, 
        timeline_analysis: Optional[TimelineAnalysis],
        key_factors: List[str]
    ) -> ScenarioClassification:
        """Create an uncertain classification result that may trigger LLM analysis."""
        
        # Determine if this should escalate or use LLM based on webhook mode
        current_threshold = self.confidence_thresholds.get(
            "webhook_automation" if self.webhook_mode else "interactive_chat",
            0.85 if self.webhook_mode else 0.70
        )
        
        if self.webhook_mode:
            # In webhook mode, be more conservative and escalate uncertain cases
            recommended_action = "escalate"
            confidence = 0.4  # Below webhook threshold to trigger escalation
        else:
            # In interactive mode, can use LLM for uncertain cases
            recommended_action = "use_llm_analysis"
            confidence = 0.5  # Between chat and webhook thresholds
        
        return ScenarioClassification(
            scenario_type="uncertain_requires_analysis",
            confidence=confidence,
            reasoning=(
                "Customer complaint does not match clear patterns for common "
                "'Paid Again' scenarios. The case requires additional analysis "
                "to determine the appropriate classification and decision."
            ),
            recommended_action=recommended_action,
            refund_eligible=False,
            key_factors=key_factors or [
                "No clear scenario patterns detected",
                "Requires additional analysis for classification"
            ],
            timeline_evidence=timeline_analysis
        )