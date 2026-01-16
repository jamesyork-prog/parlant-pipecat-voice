"""
CancellationReasonMapper - Maps decision reasoning to ParkWhiz cancellation reasons.

This component maps the refund decision reasoning to appropriate ParkWhiz cancellation
reasons using keyword matching and rule-based logic.
"""

from typing import Dict, Optional


class CancellationReasonMapper:
    """Maps decision reasoning to ParkWhiz cancellation reasons."""
    
    # Valid ParkWhiz cancellation reasons from the dropdown
    VALID_REASONS = [
        "Other",
        "Tolerance",
        "Multi-day",
        "Pending re-book",
        "Pre-arrival",
        "Oversold",
        "No attendant",
        "Amenity missing",
        "Poor experience",
        "Inaccurate hours of operation",
        "Attendant refused customer",
        "Duplicate booking",
        "Confirmed re-book",
        "Paid again",
        "Accessibility",
        "PW cancellation"
    ]
    
    def __init__(self):
        """Initialize the mapper with keyword patterns."""
        # Keyword patterns for mapping (lowercase for case-insensitive matching)
        self.keyword_patterns = {
            "Oversold": ["oversold", "over-sold", "overbooked", "over-booked", "sold out"],
            "Duplicate booking": ["duplicate", "duplicated", "double booking", "booked twice"],
            "Pre-arrival": ["pre-arrival", "pre arrival", "before event", "7+ days", "advance cancellation"],
            "Tolerance": ["tolerance", "goodwill", "customer satisfaction", "exception", "courtesy"],
            "Amenity missing": ["amenity", "amenities", "facility", "facilities", "missing feature"],
            "Poor experience": ["poor experience", "complaint", "dissatisfied", "unhappy", "bad experience"],
            "No attendant": ["no attendant", "attendant missing", "no staff", "unmanned"],
            "Attendant refused customer": ["attendant refused", "refused entry", "denied access"],
            "Inaccurate hours of operation": ["hours", "closed", "operating hours", "wrong hours"],
            "Multi-day": ["multi-day", "multiple days", "multi day"],
            "Pending re-book": ["pending re-book", "pending rebook", "will rebook"],
            "Confirmed re-book": ["confirmed re-book", "confirmed rebook", "rebooked"],
            "Paid again": ["paid again", "double charged", "charged twice", "duplicate payment"],
            "Accessibility": ["accessibility", "accessible", "ada", "disability", "wheelchair"],
            "PW cancellation": ["parkwhiz cancel", "pw cancel", "system cancel", "platform cancel"]
        }
    
    def map_reason(
        self,
        decision_reasoning: str,
        policy_applied: str,
        booking_info: Optional[Dict] = None
    ) -> str:
        """
        Map decision reasoning to ParkWhiz cancellation reason.
        
        Args:
            decision_reasoning: The reasoning text from the decision
            policy_applied: The policy rule that was applied
            booking_info: Optional booking information for additional context
            
        Returns:
            str: One of the valid ParkWhiz cancellation reasons
        """
        # Combine all text for keyword matching (lowercase)
        combined_text = f"{decision_reasoning} {policy_applied}".lower()
        
        # Check each pattern for matches
        for reason, keywords in self.keyword_patterns.items():
            for keyword in keywords:
                if keyword in combined_text:
                    return reason
        
        # Default to "Other" when no clear match
        return "Other"
    
    def validate_reason(self, reason: str) -> bool:
        """
        Validate that a reason is in the list of valid ParkWhiz reasons.
        
        Args:
            reason: The cancellation reason to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        return reason in self.VALID_REASONS
