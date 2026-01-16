"""
EscalationEnhancer component for enhanced escalation documentation and agent guidance.

This module provides the EscalationEnhancer class that enhances escalation reasoning
with timeline breakdowns, contradiction examples, and agent guidance for the
Paid Again Classification System.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TimelineBreakdown:
    """Structured timeline breakdown for escalation documentation."""
    early_arrival_minutes: int
    overstay_minutes: int
    actual_parking_duration: int
    booked_duration: int
    booking_start: Optional[datetime]
    booking_end: Optional[datetime]
    arrival_time: Optional[datetime]
    exit_time: Optional[datetime]
    timeline_issues: List[str]


@dataclass
class ContradictionExample:
    """Structured contradiction example for escalation documentation."""
    contradiction_type: str
    customer_statement: str
    conflicting_evidence: str
    explanation: str
    severity: str


@dataclass
class AgentGuidance:
    """Structured agent guidance for escalation cases."""
    suggested_questions: List[str]
    clarifying_actions: List[str]
    decision_guidance: str
    escalation_priority: str


class EscalationEnhancer:
    """
    Enhances escalation documentation with timeline breakdowns, contradiction examples,
    and agent guidance for the Paid Again Classification System.
    
    This class provides enhanced escalation reasoning that helps human agents
    efficiently resolve escalated cases with full context and actionable guidance.
    """
    
    def __init__(self):
        """Initialize the EscalationEnhancer."""
        logger.info("EscalationEnhancer initialized")
    
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
    
    def enhance_timeline_escalation(
        self,
        original_reasoning: str,
        timeline_analysis: Optional[Dict],
        escalation_scenario: str
    ) -> str:
        """
        Enhance escalation reasoning with timeline breakdown for timeline-based cases.
        
        Args:
            original_reasoning: Original escalation reasoning
            timeline_analysis: Timeline analysis data from rule engine
            escalation_scenario: Type of timeline escalation scenario
            
        Returns:
            Enhanced reasoning with timeline breakdown and agent guidance
        """
        if not timeline_analysis:
            return original_reasoning
        
        # Extract timeline data
        early_arrival = timeline_analysis.get("early_arrival_minutes", 0)
        overstay = timeline_analysis.get("overstay_minutes", 0)
        actual_duration = timeline_analysis.get("actual_parking_duration", 0)
        booked_duration = timeline_analysis.get("booked_duration", 0)
        timeline_issues = timeline_analysis.get("timeline_issues", [])
        
        # Build timeline breakdown
        timeline_breakdown = f"""
<br><br><strong>Timeline Analysis Breakdown:</strong><br>
• Early arrival: {early_arrival} minutes before booking start<br>
• Overstay duration: {overstay} minutes beyond booking end<br>
• Actual parking time: {actual_duration} minutes<br>
• Booked parking time: {booked_duration} minutes<br>
"""
        
        if timeline_issues:
            timeline_breakdown += f"• Timeline issues detected: {', '.join(timeline_issues)}<br>"
        
        # Add scenario-specific guidance
        agent_guidance = self._get_timeline_agent_guidance(escalation_scenario, early_arrival, overstay)
        
        # Combine original reasoning with enhancements
        enhanced_reasoning = f"{original_reasoning}{timeline_breakdown}{agent_guidance}"
        
        logger.info(f"Enhanced timeline escalation reasoning for scenario: {escalation_scenario}")
        return enhanced_reasoning
    
    def enhance_contradiction_escalation(
        self,
        original_reasoning: str,
        contradictions: Optional[List[Dict]],
        escalation_scenario: str
    ) -> str:
        """
        Enhance escalation reasoning with specific contradiction examples and explanations.
        
        Args:
            original_reasoning: Original escalation reasoning
            contradictions: List of detected contradictions
            escalation_scenario: Type of contradiction escalation scenario
            
        Returns:
            Enhanced reasoning with contradiction examples and agent guidance
        """
        if not contradictions:
            return original_reasoning
        
        # Build contradiction examples
        contradiction_breakdown = "<br><br><strong>Detected Contradictions:</strong><br>"
        
        for i, contradiction in enumerate(contradictions, 1):
            contradiction_type = contradiction.get("type", "unknown")
            description = contradiction.get("description", "")
            customer_statement = contradiction.get("customer_statement", "")
            timeline_evidence = contradiction.get("timeline_evidence", "")
            explanation = contradiction.get("explanation", "")
            
            contradiction_breakdown += f"""
<strong>Contradiction {i}:</strong> {contradiction_type.replace('_', ' ').title()}<br>
• Customer claims: "{customer_statement}"<br>
• Evidence shows: "{timeline_evidence}"<br>
• Explanation: {explanation}<br><br>
"""
        
        # Add agent guidance for contradictions
        agent_guidance = self._get_contradiction_agent_guidance(contradictions)
        
        # Combine original reasoning with enhancements
        enhanced_reasoning = f"{original_reasoning}{contradiction_breakdown}{agent_guidance}"
        
        logger.info(f"Enhanced contradiction escalation reasoning with {len(contradictions)} contradictions")
        return enhanced_reasoning
    
    def enhance_exit_complication_escalation(
        self,
        original_reasoning: str,
        exit_complication_type: str,
        customer_parked_successfully: bool,
        booking_amount: float
    ) -> str:
        """
        Enhance escalation reasoning to distinguish between poor experience credits and refund eligibility.
        
        Args:
            original_reasoning: Original escalation reasoning
            exit_complication_type: Type of exit complication
            customer_parked_successfully: Whether customer successfully parked
            booking_amount: Booking amount for credit calculation
            
        Returns:
            Enhanced reasoning with credit vs refund distinction and guidance
        """
        # Determine appropriate resolution
        if customer_parked_successfully:
            # Poor experience credit case
            credit_amount = self._calculate_credit_amount(booking_amount)
            resolution_guidance = f"""
<br><br><strong>Resolution Guidance - Poor Experience Credit:</strong><br>
• Customer successfully used parking but encountered exit difficulties<br>
• Recommend: ${credit_amount:.0f} account credit for inconvenience<br>
• Rationale: Customer received parking value but had poor service experience<br>
• Policy: Credits for service issues when parking was successfully used<br>
"""
        else:
            # Refund eligible case
            resolution_guidance = f"""
<br><br><strong>Resolution Guidance - Refund Eligible:</strong><br>
• Customer unable to complete parking due to exit system failure<br>
• Recommend: Full refund of ${booking_amount:.2f}<br>
• Rationale: Customer could not use the parking service as intended<br>
• Policy: Refunds when service delivery failure prevents parking use<br>
"""
        
        # Add exit complication specific guidance
        agent_guidance = self._get_exit_complication_agent_guidance(exit_complication_type, customer_parked_successfully)
        
        # Combine original reasoning with enhancements
        enhanced_reasoning = f"{original_reasoning}{resolution_guidance}{agent_guidance}"
        
        logger.info(f"Enhanced exit complication escalation reasoning: {exit_complication_type}")
        return enhanced_reasoning
    
    def add_agent_guidance(
        self,
        original_reasoning: str,
        escalation_reason: str,
        missing_information: Optional[List[str]] = None,
        suggested_questions: Optional[List[str]] = None
    ) -> str:
        """
        Add comprehensive agent guidance to escalation reasoning.
        
        Args:
            original_reasoning: Original escalation reasoning
            escalation_reason: Reason for escalation
            missing_information: List of missing information items
            suggested_questions: Pre-defined suggested questions
            
        Returns:
            Enhanced reasoning with agent guidance and suggested questions
        """
        # Generate suggested questions if not provided
        if not suggested_questions:
            suggested_questions = self._generate_suggested_questions(escalation_reason, missing_information)
        
        # Build agent guidance section
        agent_guidance = f"""
<br><br><strong>Agent Action Required:</strong><br>
<strong>Escalation Reason:</strong> {escalation_reason.replace('_', ' ').title()}<br>
"""
        
        if missing_information:
            agent_guidance += f"<strong>Missing Information:</strong> {', '.join(missing_information)}<br>"
        
        if suggested_questions:
            agent_guidance += "<br><strong>Suggested Clarifying Questions:</strong><br>"
            for i, question in enumerate(suggested_questions, 1):
                agent_guidance += f"{i}. {question}<br>"
        
        # Add escalation priority guidance
        priority_guidance = self._get_escalation_priority_guidance(escalation_reason)
        agent_guidance += f"<br><strong>Priority:</strong> {priority_guidance}<br>"
        
        # Combine original reasoning with enhancements
        enhanced_reasoning = f"{original_reasoning}{agent_guidance}"
        
        logger.info(f"Added agent guidance for escalation reason: {escalation_reason}")
        return enhanced_reasoning
    
    def _get_timeline_agent_guidance(self, scenario: str, early_arrival: int, overstay: int) -> str:
        """Generate agent guidance for timeline-based escalations."""
        if scenario == "early_arrival_overstay":
            early_formatted = self._format_duration(early_arrival)
            overstay_formatted = self._format_duration(overstay)
            return f"""
<br><strong>Agent Action Required:</strong><br>
1. Confirm customer arrival time was {early_formatted} before booking start<br>
2. Verify customer stayed {overstay_formatted} past booking end time<br>
3. Explain that early arrival + overstay = legitimate additional charges<br>
4. Decision: Deny refund - customer used more time than booked<br>
"""
        elif scenario == "timeline_inconsistent":
            return """
<br><strong>Agent Action Required:</strong><br>
1. Ask customer to clarify their arrival and departure times<br>
2. Check if timeline makes logical sense (arrival before departure, etc.)<br>
3. Request additional evidence if timeline seems impossible<br>
4. Decision: Resolve timeline inconsistency before making refund decision<br>
"""
        elif scenario == "timeline_math_error":
            return """
<br><strong>Agent Action Required:</strong><br>
1. Review timeline calculations with customer<br>
2. Verify booking window and actual usage times<br>
3. Recalculate overstay charges if applicable<br>
4. Decision: Base decision on corrected timeline analysis<br>
"""
        else:
            return """
<br><strong>Agent Action Required:</strong><br>
1. Review timeline analysis details with customer<br>
2. Clarify any discrepancies in reported vs calculated times<br>
3. Make decision based on verified timeline information<br>
"""
    
    def _get_contradiction_agent_guidance(self, contradictions: List[Dict]) -> str:
        """Generate agent guidance for contradiction-based escalations."""
        guidance = """
<br><strong>Agent Action Required:</strong><br>
1. Review each contradiction with the customer<br>
2. Ask customer to clarify conflicting statements<br>
3. Determine which version of events is accurate<br>
4. Make decision based on clarified information<br>
<br><strong>Suggested Questions:</strong><br>
"""
        
        # Add specific questions based on contradiction types
        contradiction_types = [c.get("type", "") for c in contradictions]
        
        if "parking_vs_no_parking" in contradiction_types:
            guidance += "• Were you able to park successfully or not? Please clarify.<br>"
        
        if "timeline_math_error" in contradiction_types:
            guidance += "• Can you walk me through exactly what time you arrived and left?<br>"
        
        if "mutually_exclusive_claims" in contradiction_types:
            guidance += "• You mentioned two different issues - which was the main problem?<br>"
        
        guidance += "• Can you provide any additional details to help clarify the situation?<br>"
        
        return guidance
    
    def _get_exit_complication_agent_guidance(self, complication_type: str, parked_successfully: bool) -> str:
        """Generate agent guidance for exit complication escalations."""
        base_guidance = f"""
<br><strong>Agent Action Required:</strong><br>
1. Confirm the specific exit issue: {complication_type.replace('_', ' ')}<br>
2. Verify whether customer successfully parked initially<br>
"""
        
        if parked_successfully:
            base_guidance += """3. Explain this is a poor experience case, not a refund case<br>
4. Offer appropriate account credit for the inconvenience<br>
5. Decision: Credit for poor experience, not full refund<br>
"""
        else:
            base_guidance += """3. Confirm customer was unable to use parking due to exit system failure<br>
4. Verify customer did not receive parking value from their purchase<br>
5. Decision: Approve full refund for unused parking service<br>
"""
        
        return base_guidance
    
    def _generate_suggested_questions(self, escalation_reason: str, missing_info: Optional[List[str]]) -> List[str]:
        """Generate suggested questions based on escalation reason and missing information."""
        questions = []
        
        if escalation_reason == "timeline_analysis_required":
            questions.extend([
                "What time did you arrive at the parking location?",
                "What time did you leave the parking facility?",
                "Did you arrive before your booking start time?"
            ])
        
        elif escalation_reason == "contradiction_detected":
            questions.extend([
                "Can you clarify what exactly happened with your parking experience?",
                "Were you able to park successfully or not?",
                "Can you walk me through the timeline of events?"
            ])
        
        elif escalation_reason == "insufficient_information":
            if missing_info:
                questions.extend([f"Can you provide more details about {info}?" for info in missing_info])
            else:
                questions.append("Can you provide more details about your parking experience?")
        
        elif escalation_reason == "webhook_confidence_threshold":
            questions.extend([
                "Can you provide additional details to help clarify your situation?",
                "Are there any other factors that contributed to your refund request?"
            ])
        
        else:
            questions.extend([
                "Can you provide more details about your parking experience?",
                "What specific issue prevented you from using your booking?"
            ])
        
        return questions
    
    def _get_escalation_priority_guidance(self, escalation_reason: str) -> str:
        """Get priority guidance for different escalation reasons."""
        priority_map = {
            "contradiction_detected": "High - Requires immediate clarification",
            "timeline_analysis_required": "Medium - Timeline verification needed",
            "insufficient_information": "Medium - Additional details required",
            "complex_scenario": "Medium - Detailed review required",
            "webhook_confidence_threshold": "Low - Standard escalation review"
        }
        
        return priority_map.get(escalation_reason, "Medium - Standard review required")
    
    def _calculate_credit_amount(self, booking_amount: float) -> float:
        """Calculate appropriate credit amount for poor experience cases."""
        if booking_amount < 10:
            return 0  # No credit for bookings under $10
        elif booking_amount <= 20:
            return 5.0  # $5 credit for bookings $10-$20
        else:
            return 10.0  # $10 credit for bookings over $20