"""
Booking Mismatch Detector for identifying cases where customer-provided booking details
don't match the actual booking data in the ParkWhiz system.

This module detects scenarios where:
- Customer provides a valid booking ID that exists in the system
- But customer details (name, email, dates) don't match the booking
- Massive temporal discrepancies (years apart)
- Different customer entirely

These cases require human review to verify correct booking information.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BookingMismatch:
    """Represents a detected booking mismatch."""
    mismatch_type: str  # "temporal", "customer", "email", "combined"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    customer_claim: str
    system_data: str
    confidence: float  # 0.0 to 1.0


class BookingMismatchDetector:
    """
    Detects mismatches between customer-provided booking details and actual system data.
    
    This detector identifies cases where customers may have provided:
    - Wrong booking ID
    - Confused booking details
    - Fraudulent claims (rare)
    - Data entry errors
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the booking mismatch detector.
        
        Args:
            config: Configuration dict with thresholds and settings
        """
        if config is None:
            config = {}
        
        # Temporal mismatch thresholds
        self.critical_years_threshold = config.get("critical_years_threshold", 5)  # 5+ years = critical
        self.high_years_threshold = config.get("high_years_threshold", 2)  # 2+ years = high
        self.medium_days_threshold = config.get("medium_days_threshold", 90)  # 90+ days = medium
        
        # Customer mismatch settings
        self.name_similarity_threshold = config.get("name_similarity_threshold", 0.3)  # Levenshtein ratio
        self.email_domain_check = config.get("email_domain_check", True)
        
        logger.info(f"BookingMismatchDetector initialized with thresholds: "
                   f"critical={self.critical_years_threshold}y, high={self.high_years_threshold}y, "
                   f"medium={self.medium_days_threshold}d")
    
    def detect_mismatches(
        self,
        customer_data: Dict,
        system_booking_data: Dict,
        ticket_notes: Optional[str] = None
    ) -> List[BookingMismatch]:
        """
        Detect mismatches between customer claims and system booking data.
        
        Args:
            customer_data: Customer-provided information from ticket:
                - name: Customer name from ticket
                - email: Customer email from ticket  
                - booking_id: Booking ID provided by customer
                - event_date: Date customer claims parking was for
                - arrival_date: Date customer claims they arrived (optional)
            system_booking_data: Actual booking data from ParkWhiz API:
                - user_name: Actual booking owner name
                - user_email: Actual booking owner email
                - booking_created: When booking was actually made
                - pass_start: Actual parking pass start date
                - pass_end: Actual parking pass end date
                - event_name: Actual event name (optional)
                - location: Actual parking location
            ticket_notes: Raw ticket text for additional context
            
        Returns:
            List of detected mismatches, ordered by severity
        """
        logger.info(f"Analyzing booking mismatch for booking ID: {customer_data.get('booking_id')}")
        
        mismatches = []
        
        # 1. Temporal mismatch detection (most critical)
        temporal_mismatch = self._detect_temporal_mismatch(customer_data, system_booking_data)
        if temporal_mismatch:
            mismatches.append(temporal_mismatch)
        
        # 2. Customer identity mismatch detection
        customer_mismatch = self._detect_customer_mismatch(customer_data, system_booking_data)
        if customer_mismatch:
            mismatches.append(customer_mismatch)
        
        # 3. Email mismatch detection
        email_mismatch = self._detect_email_mismatch(customer_data, system_booking_data)
        if email_mismatch:
            mismatches.append(email_mismatch)
        
        # 4. Combined mismatch analysis
        if len(mismatches) >= 2:
            combined_mismatch = self._analyze_combined_mismatches(mismatches, customer_data, system_booking_data)
            if combined_mismatch:
                mismatches.append(combined_mismatch)
        
        # Sort by severity (critical first)
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        mismatches.sort(key=lambda m: severity_order.get(m.severity, 4))
        
        if mismatches:
            logger.warning(f"Detected {len(mismatches)} booking mismatches, highest severity: {mismatches[0].severity}")
        else:
            logger.info("No booking mismatches detected")
        
        return mismatches
    
    def _detect_temporal_mismatch(self, customer_data: Dict, system_data: Dict) -> Optional[BookingMismatch]:
        """
        Detect temporal mismatches between claimed and actual booking dates.
        
        This catches cases where customer claims recent dates
        but the booking is from 2012.
        """
        customer_event_date = customer_data.get("event_date")
        system_pass_start = system_data.get("pass_start") or system_data.get("booking_created")
        
        if not customer_event_date or not system_pass_start:
            logger.debug("Insufficient date data for temporal mismatch detection")
            return None
        
        try:
            # Parse dates - handle various formats
            customer_dt = self._parse_flexible_date(customer_event_date)
            system_dt = self._parse_flexible_date(system_pass_start)
            
            if not customer_dt or not system_dt:
                logger.warning("Could not parse dates for temporal mismatch detection")
                return None
            
            # Calculate time difference
            time_diff = abs((customer_dt - system_dt).days)
            years_diff = time_diff / 365.25
            
            logger.debug(f"Temporal analysis: customer={customer_dt.date()}, system={system_dt.date()}, "
                        f"diff={time_diff} days ({years_diff:.1f} years)")
            
            # Determine severity based on time difference
            if years_diff >= self.critical_years_threshold:
                severity = "critical"
                confidence = 0.95
            elif years_diff >= self.high_years_threshold:
                severity = "high" 
                confidence = 0.85
            elif time_diff >= self.medium_days_threshold:
                severity = "medium"
                confidence = 0.70
            else:
                # Small time difference is normal
                return None
            
            description = (
                f"Massive temporal discrepancy detected: customer claims {customer_dt.strftime('%B %d, %Y')} "
                f"but system shows {system_dt.strftime('%B %d, %Y')} - a difference of {years_diff:.1f} years. "
                f"This suggests the customer may have provided an incorrect booking ID."
            )
            
            return BookingMismatch(
                mismatch_type="temporal",
                severity=severity,
                description=description,
                customer_claim=f"Event date: {customer_dt.strftime('%B %d, %Y')}",
                system_data=f"Actual pass date: {system_dt.strftime('%B %d, %Y')}",
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error in temporal mismatch detection: {e}")
            return None
    
    def _detect_customer_mismatch(self, customer_data: Dict, system_data: Dict) -> Optional[BookingMismatch]:
        """
        Detect customer identity mismatches between ticket submitter and booking owner.
        """
        customer_name = customer_data.get("name", "").strip()
        system_name = system_data.get("user_name", "").strip()
        
        if not customer_name or not system_name:
            logger.debug("Insufficient name data for customer mismatch detection")
            return None
        
        # Calculate name similarity using simple approach
        similarity = self._calculate_name_similarity(customer_name, system_name)
        
        logger.debug(f"Name similarity: '{customer_name}' vs '{system_name}' = {similarity:.2f}")
        
        # If names are very different, it's a mismatch
        if similarity < self.name_similarity_threshold:
            severity = "high" if similarity < 0.1 else "medium"
            confidence = 0.90 if similarity < 0.1 else 0.75
            
            description = (
                f"Customer identity mismatch: ticket submitted by '{customer_name}' "
                f"but booking belongs to '{system_name}'. This suggests the customer "
                f"may have provided someone else's booking ID or incorrect information."
            )
            
            return BookingMismatch(
                mismatch_type="customer",
                severity=severity,
                description=description,
                customer_claim=f"Customer name: {customer_name}",
                system_data=f"Booking owner: {system_name}",
                confidence=confidence
            )
        
        return None
    
    def _detect_email_mismatch(self, customer_data: Dict, system_data: Dict) -> Optional[BookingMismatch]:
        """
        Detect email mismatches between ticket submitter and booking owner.
        
        Uses fuzzy matching to handle cases where people use different emails
        that still clearly belong to the same person (e.g., james.york@gmail.com
        and j.york3@proton.me are likely the same person).
        """
        customer_email = customer_data.get("email", "").strip().lower()
        system_email = system_data.get("user_email", "").strip().lower()
        
        if not customer_email or not system_email:
            logger.debug("Insufficient email data for email mismatch detection")
            return None
        
        # Exact email match check
        if customer_email == system_email:
            return None
        
        # Get account holder name for cross-reference
        system_name = system_data.get("user_name", "").strip().lower()
        customer_name = customer_data.get("name", "").strip().lower()
        
        # Extract name components from emails
        customer_email_name = self._extract_name_from_email(customer_email)
        system_email_name = self._extract_name_from_email(system_email)
        
        logger.debug(f"Email comparison: '{customer_email}' vs '{system_email}'")
        logger.debug(f"Extracted names from emails: '{customer_email_name}' vs '{system_email_name}'")
        logger.debug(f"Account names: customer='{customer_name}', system='{system_name}'")
        
        # Check if emails likely belong to same person using multiple signals
        same_person_confidence = self._calculate_same_person_confidence(
            customer_email=customer_email,
            system_email=system_email,
            customer_email_name=customer_email_name,
            system_email_name=system_email_name,
            customer_name=customer_name,
            system_name=system_name
        )
        
        logger.debug(f"Same person confidence: {same_person_confidence:.2f}")
        
        # If high confidence same person, don't flag as mismatch
        if same_person_confidence >= 0.6:
            logger.info(f"Email mismatch bypassed - likely same person (confidence: {same_person_confidence:.2f})")
            return None
        
        # Determine severity based on how different the emails are
        customer_domain = customer_email.split('@')[-1] if '@' in customer_email else ""
        system_domain = system_email.split('@')[-1] if '@' in system_email else ""
        
        if same_person_confidence >= 0.3:
            # Some indicators of same person, but not conclusive
            severity = "low"
            confidence = 0.50
            description = (
                f"Email mismatch with possible same-person indicators: ticket from '{customer_email}' "
                f"but booking belongs to '{system_email}'. Some name similarities detected - "
                f"may be same person with multiple email addresses."
            )
        elif customer_domain != system_domain:
            # Different domains and no name similarity = high severity
            severity = "high"
            confidence = 0.85
            description = (
                f"Email domain mismatch: ticket from '{customer_email}' "
                f"but booking belongs to '{system_email}'. Different email domains "
                f"and no name correlation suggest this may be a different customer entirely."
            )
        else:
            # Same domain, different username, no name match = medium severity
            severity = "medium"
            confidence = 0.70
            description = (
                f"Email username mismatch: ticket from '{customer_email}' "
                f"but booking belongs to '{system_email}'. Same domain but "
                f"different username may indicate shared account or typo."
            )
        
        return BookingMismatch(
            mismatch_type="email",
            severity=severity,
            description=description,
            customer_claim=f"Customer email: {customer_email}",
            system_data=f"Booking email: {system_email}",
            confidence=confidence
        )
    
    def _extract_name_from_email(self, email: str) -> str:
        """
        Extract likely name components from an email address.
        
        Examples:
            james.york@gmail.com -> "james york"
            j.york3@proton.me -> "j york"
            jyork123@yahoo.com -> "jyork"
            john_doe_99@hotmail.com -> "john doe"
        """
        if not email or '@' not in email:
            return ""
        
        username = email.split('@')[0].lower()
        
        # Remove common numeric suffixes
        import re
        username = re.sub(r'\d+$', '', username)
        
        # Replace common separators with spaces
        username = re.sub(r'[._\-+]', ' ', username)
        
        # Remove any remaining numbers
        username = re.sub(r'\d+', '', username)
        
        # Clean up whitespace
        username = ' '.join(username.split())
        
        return username
    
    def _calculate_same_person_confidence(
        self,
        customer_email: str,
        system_email: str,
        customer_email_name: str,
        system_email_name: str,
        customer_name: str,
        system_name: str
    ) -> float:
        """
        Calculate confidence that two different emails belong to the same person.
        
        Uses multiple signals:
        1. Name extracted from customer email matches system account name
        2. Name extracted from system email matches customer ticket name
        3. Names extracted from both emails share components
        4. Account names match (already checked elsewhere, but reinforces)
        
        Returns:
            Confidence score 0.0 to 1.0
        """
        signals = []
        
        # Signal 1: Customer email name matches system account name
        if customer_email_name and system_name:
            match_score = self._fuzzy_name_match(customer_email_name, system_name)
            if match_score > 0:
                signals.append(("customer_email_vs_system_name", match_score))
                logger.debug(f"Signal: customer email name '{customer_email_name}' vs system name '{system_name}' = {match_score:.2f}")
        
        # Signal 2: System email name matches customer ticket name
        if system_email_name and customer_name:
            match_score = self._fuzzy_name_match(system_email_name, customer_name)
            if match_score > 0:
                signals.append(("system_email_vs_customer_name", match_score))
                logger.debug(f"Signal: system email name '{system_email_name}' vs customer name '{customer_name}' = {match_score:.2f}")
        
        # Signal 3: Email-extracted names share components
        if customer_email_name and system_email_name:
            match_score = self._fuzzy_name_match(customer_email_name, system_email_name)
            if match_score > 0:
                signals.append(("email_names_match", match_score))
                logger.debug(f"Signal: email names '{customer_email_name}' vs '{system_email_name}' = {match_score:.2f}")
        
        # Signal 4: Account names match (strong signal)
        if customer_name and system_name:
            match_score = self._fuzzy_name_match(customer_name, system_name)
            if match_score > 0:
                signals.append(("account_names_match", match_score * 1.5))  # Weight this higher
                logger.debug(f"Signal: account names '{customer_name}' vs '{system_name}' = {match_score:.2f}")
        
        if not signals:
            return 0.0
        
        # Take the best signal (not average, since one strong match is enough)
        best_score = max(score for _, score in signals)
        
        # Boost if multiple signals agree
        if len(signals) >= 2:
            best_score = min(1.0, best_score * 1.2)
        
        return min(1.0, best_score)
    
    def _fuzzy_name_match(self, name1: str, name2: str) -> float:
        """
        Fuzzy match two names, handling initials and partial matches.
        
        Examples that should match:
            "james york" vs "james york" -> 1.0
            "james york" vs "j york" -> 0.7 (initial match)
            "james york" vs "york" -> 0.5 (last name only)
            "james" vs "james smith" -> 0.5 (first name only)
            "james york" vs "john smith" -> 0.0 (no match)
        """
        if not name1 or not name2:
            return 0.0
        
        # Normalize
        name1 = ' '.join(name1.lower().split())
        name2 = ' '.join(name2.lower().split())
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        words1 = name1.split()
        words2 = name2.split()
        
        # Check for shared complete words (first/last name match)
        shared_words = set(words1) & set(words2)
        if shared_words:
            # At least one name component matches exactly
            match_ratio = len(shared_words) / max(len(words1), len(words2))
            return max(0.5, match_ratio)
        
        # Check for initial matches (j vs james, m vs michael)
        for w1 in words1:
            for w2 in words2:
                # One is initial of the other
                if len(w1) == 1 and w2.startswith(w1):
                    return 0.7
                if len(w2) == 1 and w1.startswith(w2):
                    return 0.7
        
        # Check if one name contains the other
        if name1 in name2 or name2 in name1:
            return 0.6
        
        # No match
        return 0.0
    
    def _analyze_combined_mismatches(
        self, 
        mismatches: List[BookingMismatch], 
        customer_data: Dict, 
        system_data: Dict
    ) -> Optional[BookingMismatch]:
        """
        Analyze multiple mismatches together to determine if this is a critical case.
        
        Cases with temporal + customer + email mismatches are extremely suspicious.
        """
        mismatch_types = {m.mismatch_type for m in mismatches}
        
        # If we have temporal + customer identity mismatches, this is critical
        if "temporal" in mismatch_types and ("customer" in mismatch_types or "email" in mismatch_types):
            # Find the temporal mismatch for details
            temporal_mismatch = next(m for m in mismatches if m.mismatch_type == "temporal")
            
            description = (
                f"CRITICAL: Multiple booking mismatches detected - this appears to be a case "
                f"where the customer has provided completely incorrect booking information. "
                f"The booking ID exists but belongs to a different person from a different time period. "
                f"This requires immediate human verification of the correct booking details."
            )
            
            return BookingMismatch(
                mismatch_type="combined",
                severity="critical",
                description=description,
                customer_claim=f"Claims booking from {customer_data.get('name', 'unknown')} for {customer_data.get('event_date', 'unknown date')}",
                system_data=f"Actual booking by {system_data.get('user_name', 'unknown')} for {system_data.get('pass_start', 'unknown date')}",
                confidence=0.98
            )
        
        return None
    
    def _parse_flexible_date(self, date_str: str) -> Optional[datetime]:
        """
        Parse date string in various formats.
        
        Args:
            date_str: Date string in various formats
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        if not date_str:
            return None
        
        # Common date formats to try
        formats = [
            "%Y-%m-%d",  # ISO format
            "%m/%d/%Y",  # US format
            "%m/%d/%Y %H:%M:%S",  # US with time
            "%Y-%m-%d %H:%M:%S",  # ISO with time
            "%A %b %d, %Y, %I:%M %p",  # "Wednesday Jul 18, 2012, 12:10 PM"
            "%A %b %d, %Y",  # "Wednesday Jul 18, 2012"
            "%b %d, %Y",  # "Jul 18, 2012"
            "%B %d, %Y",  # "July 18, 2012"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        
        logger.debug(f"Could not parse date string: '{date_str}'")
        return None
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity between two names using simple approach.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0
        
        # Normalize names (lowercase, remove extra spaces)
        name1 = " ".join(name1.lower().split())
        name2 = " ".join(name2.lower().split())
        
        # Exact match
        if name1 == name2:
            return 1.0
        
        # Check if one name is contained in the other (partial match)
        if name1 in name2 or name2 in name1:
            return 0.7
        
        # Check for common words (first/last name overlap) - this is the most important check
        words1 = set(name1.split())
        words2 = set(name2.split())
        
        if words1 & words2:  # Any common words (shared first or last name)
            overlap = len(words1 & words2)
            total = len(words1 | words2)
            return overlap / total if total > 0 else 0.0
        
        # No common words = very different names
        # For completely different names like "John Smith" vs "Jane Doe", 
        # we should return a very low similarity score
        
        # Check for similar-sounding first letters (very weak similarity)
        first1 = name1.split()[0][0] if name1.split() else ""
        first2 = name2.split()[0][0] if name2.split() else ""
        
        if first1 == first2:
            return 0.1  # Very low similarity for same first letter
        
        # Completely different names
        return 0.0
    
    def should_escalate_for_mismatch(self, mismatches: List[BookingMismatch]) -> Tuple[bool, str]:
        """
        Determine if booking mismatches warrant escalation to human review.
        
        Args:
            mismatches: List of detected mismatches
            
        Returns:
            Tuple of (should_escalate, escalation_reason)
        """
        if not mismatches:
            return False, ""
        
        # Any critical mismatch = immediate escalation
        critical_mismatches = [m for m in mismatches if m.severity == "critical"]
        if critical_mismatches:
            return True, f"Critical booking mismatch detected: {critical_mismatches[0].description}"
        
        # Multiple high-severity mismatches = escalation
        high_mismatches = [m for m in mismatches if m.severity == "high"]
        if len(high_mismatches) >= 2:
            return True, f"Multiple high-severity booking mismatches detected"
        
        # Single high-severity temporal mismatch = escalation
        temporal_high = [m for m in high_mismatches if m.mismatch_type == "temporal"]
        if temporal_high:
            return True, f"Significant temporal mismatch detected: {temporal_high[0].description}"
        
        # Single high-severity customer mismatch = escalation
        customer_high = [m for m in high_mismatches if m.mismatch_type == "customer"]
        if customer_high:
            return True, f"Customer identity mismatch detected: {customer_high[0].description}"
        
        return False, ""
    
    def format_mismatch_summary(self, mismatches: List[BookingMismatch]) -> str:
        """
        Format booking mismatches into a human-readable summary for agents.
        
        Args:
            mismatches: List of detected mismatches
            
        Returns:
            Formatted summary string
        """
        if not mismatches:
            return "No booking mismatches detected."
        
        summary = "**BOOKING MISMATCH DETECTED**\n\n"
        
        for i, mismatch in enumerate(mismatches, 1):
            summary += f"{i}. **{mismatch.mismatch_type.title()} Mismatch** ({mismatch.severity.upper()})\n"
            summary += f"   {mismatch.description}\n"
            summary += f"   • Customer Claims: {mismatch.customer_claim}\n"
            summary += f"   • System Shows: {mismatch.system_data}\n"
            summary += f"   • Confidence: {mismatch.confidence:.0%}\n\n"
        
        summary += "**RECOMMENDED ACTION:**\n"
        summary += "Contact the customer to verify their booking details. They may have:\n"
        summary += "• Provided the wrong booking ID\n"
        summary += "• Confused multiple bookings\n"
        summary += "• Made a data entry error\n"
        summary += "• Submitted someone else's booking information\n\n"
        summary += "Ask them to double-check their booking confirmation email or account."
        
        return summary