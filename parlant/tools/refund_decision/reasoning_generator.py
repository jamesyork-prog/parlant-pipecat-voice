"""
ReasoningGenerator component for generating human-readable explanations and suggested actions.

This module provides the ReasoningGenerator class that generates detailed reasoning
and suggested clarifying questions for detected contradictions.
"""

import logging
from typing import List
from dataclasses import dataclass
from .logic_validator import Contradiction

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ContradictionReasoning:
    """Result of reasoning generation for contradictions."""
    explanation: str
    suggested_questions: List[str]
    priority_level: str  # "high", "medium", "low"


class ReasoningGenerator:
    """
    Generates human-readable explanations and suggested actions for detected contradictions.
    
    This class creates detailed reasoning that explains why contradictions were detected
    and provides suggested clarifying questions for human agents to ask customers.
    """
    
    def __init__(self):
        """Initialize the ReasoningGenerator."""
        logger.info("ReasoningGenerator initialized")
    
    def generate_reasoning(self, contradictions: List[Contradiction]) -> ContradictionReasoning:
        """
        Generate detailed reasoning and suggested questions for contradictions.
        
        This method analyzes detected contradictions and generates comprehensive
        explanations and actionable guidance for human agents.
        
        Args:
            contradictions: List of detected contradictions
        
        Returns:
            ContradictionReasoning with explanations and agent guidance
        """
        if not contradictions:
            logger.debug("No contradictions to generate reasoning for")
            return ContradictionReasoning(
                explanation="No contradictions detected in customer complaint",
                suggested_questions=[],
                priority_level="low"
            )
        
        logger.info(f"Generating reasoning for {len(contradictions)} contradictions")
        
        # Determine overall priority level
        priority_level = self._determine_priority_level(contradictions)
        
        # Generate explanation
        explanation = self._generate_explanation(contradictions)
        
        # Generate suggested questions
        suggested_questions = self._generate_suggested_questions(contradictions)
        
        return ContradictionReasoning(
            explanation=explanation,
            suggested_questions=suggested_questions,
            priority_level=priority_level
        )
    
    def _determine_priority_level(self, contradictions: List[Contradiction]) -> str:
        """
        Determine the overall priority level based on contradiction severities.
        
        Args:
            contradictions: List of detected contradictions
        
        Returns:
            Priority level string ("high", "medium", "low")
        """
        if any(c.severity == "high" for c in contradictions):
            return "high"
        elif any(c.severity == "medium" for c in contradictions):
            return "medium"
        else:
            return "low"
    
    def _generate_explanation(self, contradictions: List[Contradiction]) -> str:
        """
        Generate a comprehensive explanation of all detected contradictions.
        
        Args:
            contradictions: List of detected contradictions
        
        Returns:
            HTML-formatted explanation string
        """
        explanation_parts = [
            "<strong>Contradictory information detected in customer complaint.</strong><br><br>",
            "The following conflicts require human review to clarify the customer's actual situation:<br><br>"
        ]
        
        for i, contradiction in enumerate(contradictions, 1):
            explanation_parts.append(f"<strong>Contradiction {i}:</strong> {contradiction.description}<br>")
            
            if contradiction.type == "mutually_exclusive":
                explanation_parts.append(
                    f"• Customer claims: {contradiction.complaint_a.category} "
                    f"(keywords: {', '.join(contradiction.complaint_a.keywords[:3])})<br>"
                )
                explanation_parts.append(
                    f"• Customer also claims: {contradiction.complaint_b.category} "
                    f"(keywords: {', '.join(contradiction.complaint_b.keywords[:3])})<br>"
                )
                explanation_parts.append("• These scenarios cannot both be true simultaneously<br><br>")
            
            elif contradiction.type == "response_mismatch":
                explanation_parts.append(
                    f"• Customer claims: {contradiction.complaint_a.category} issue<br>"
                )
                explanation_parts.append(
                    f"• Customer response: {contradiction.customer_response}<br>"
                )
                explanation_parts.append("• The response contradicts the claimed issue<br><br>")
            
            elif contradiction.type == "logical_conflict":
                explanation_parts.append(
                    f"• Customer claims: {contradiction.complaint_a.category} "
                    f"AND {contradiction.complaint_b.category}<br>"
                )
                explanation_parts.append("• These claims create a logical inconsistency<br><br>")
            
            elif contradiction.type == "logical_contradiction":
                explanation_parts.append(
                    f"• Customer statement: {contradiction.description}<br>"
                )
                explanation_parts.append("• This creates a logical inconsistency in the customer's account<br><br>")
            
            elif contradiction.type in ["timeline_math_contradiction", "timeline_math_error"]:
                explanation_parts.append(
                    f"• Timeline issue: {contradiction.description}<br>"
                )
                explanation_parts.append("• Mathematical analysis reveals inconsistencies in the reported timeline<br><br>")
        
        explanation_parts.append(
            "<strong>Action required:</strong> Contact the customer to clarify their situation "
            "before making a refund decision."
        )
        
        return "".join(explanation_parts)
    
    def _generate_suggested_questions(self, contradictions: List[Contradiction]) -> List[str]:
        """
        Generate suggested clarifying questions for human agents.
        
        Args:
            contradictions: List of detected contradictions
        
        Returns:
            List of suggested questions to ask the customer
        """
        questions = []
        
        for contradiction in contradictions:
            if contradiction.type == "mutually_exclusive":
                questions.extend(self._get_mutually_exclusive_questions(contradiction))
            elif contradiction.type == "response_mismatch":
                questions.extend(self._get_response_mismatch_questions(contradiction))
            elif contradiction.type == "logical_conflict":
                questions.extend(self._get_logical_conflict_questions(contradiction))
            elif contradiction.type == "logical_contradiction":
                questions.extend(self._get_logical_contradiction_questions(contradiction))
            elif contradiction.type in ["timeline_math_contradiction", "timeline_math_error"]:
                questions.extend(self._get_timeline_contradiction_questions(contradiction))
        
        # Remove duplicates while preserving order
        unique_questions = []
        seen = set()
        for question in questions:
            if question not in seen:
                unique_questions.append(question)
                seen.add(question)
        
        return unique_questions
    
    def _get_mutually_exclusive_questions(self, contradiction: Contradiction) -> List[str]:
        """Generate questions for mutually exclusive complaint contradictions."""
        category_a = contradiction.complaint_a.category
        category_b = contradiction.complaint_b.category
        
        if category_a == "accessibility" and category_b == "oversold":
            return [
                "Were you unable to reach the parking location due to road closures or access restrictions?",
                "Or was the parking facility full when you arrived at the location?",
                "Please clarify which situation prevented you from parking."
            ]
        elif category_a == "paid_again" and category_b == "oversold":
            return [
                "Did you have to pay additional charges at the parking facility?",
                "Or was the facility full and unable to accommodate your booking?",
                "If you paid additional charges, were you able to park after paying?"
            ]
        elif category_a == "vehicle_restriction" and category_b == "oversold":
            return [
                "Were you turned away because your vehicle type was not allowed?",
                "Or was the facility at capacity and unable to accept more vehicles?",
                "Please clarify the specific reason you were unable to park."
            ]
        else:
            return [
                f"You mentioned both {category_a} and {category_b} issues. Can you clarify which was the primary reason you couldn't use your booking?"
            ]
    
    def _get_response_mismatch_questions(self, contradiction: Contradiction) -> List[str]:
        """Generate questions for response mismatch contradictions."""
        category = contradiction.complaint_a.category
        
        if category == "accessibility":
            return [
                "You mentioned access issues but also indicated you were able to park. Can you clarify what happened?",
                "Were you eventually able to reach the parking location despite the initial access problems?",
                "Did the access issues resolve themselves, or did you find an alternative route?"
            ]
        elif category == "oversold":
            return [
                "You mentioned the facility was full but also indicated you were able to park. Can you explain this?",
                "Were you initially told the facility was full but then found a space?",
                "Did facility staff find a way to accommodate you despite being at capacity?"
            ]
        elif category == "location_finding":
            return [
                "You mentioned difficulty finding the location but also indicated you were able to park. How did this resolve?",
                "Were you eventually able to locate the parking facility?",
                "Did someone provide directions or assistance to help you find the location?"
            ]
        else:
            return [
                f"You mentioned {category} issues but also indicated you were able to park. Can you clarify what happened?"
            ]
    
    def _get_logical_conflict_questions(self, contradiction: Contradiction) -> List[str]:
        """Generate questions for logical conflict contradictions."""
        category_a = contradiction.complaint_a.category
        category_b = contradiction.complaint_b.category
        
        if category_a == "accessibility" and category_b == "paid_again":
            return [
                "You mentioned road closures prevented access, but also that you were charged multiple times. Can you explain this sequence of events?",
                "Did you attempt to use the parking multiple times on different occasions?",
                "Were you charged for bookings you made but couldn't use due to access issues?"
            ]
        elif category_a == "vehicle_restriction" and category_b == "paid_again":
            return [
                "You mentioned being turned away due to vehicle restrictions but also being charged multiple times. Can you clarify?",
                "Were you charged before being told about vehicle restrictions?",
                "Did you have multiple interactions with the parking facility?"
            ]
        else:
            return [
                f"You mentioned both {category_a} and {category_b} issues. Can you walk me through exactly what happened in chronological order?"
            ]
    
    def _get_logical_contradiction_questions(self, contradiction: Contradiction) -> List[str]:
        """Generate questions for logical contradiction contradictions (timeline-based)."""
        return [
            "You mentioned that you couldn't park, but also described retrieving your car. Can you clarify what actually happened with your parking experience?",
            "Did you eventually manage to park despite initial difficulties?",
            "Can you walk me through the timeline of what happened when you arrived at the parking location?"
        ]
    
    def _get_timeline_contradiction_questions(self, contradiction: Contradiction) -> List[str]:
        """Generate questions for timeline mathematical contradiction contradictions."""
        return [
            "There appears to be some inconsistency in the timeline of your parking experience. Can you help us understand exactly what happened and when?",
            "Can you clarify the times when you arrived and left the parking facility?",
            "Were there any unusual circumstances that might explain the timeline discrepancy?"
        ]