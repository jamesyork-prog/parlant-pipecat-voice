"""
Pattern-based booking information extraction using regex and HTML parsing.

This module provides fast, deterministic extraction of booking information from
structured ticket text and HTML. It serves as a performance optimization that
runs before LLM extraction, reducing API calls for well-formatted tickets.
"""

import re
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime
from bs4 import BeautifulSoup

# Configure logger
logger = logging.getLogger(__name__)


# Booking ID patterns - matches various formats
BOOKING_ID_PATTERNS = [
    r'PW-\d+',                                    # PW-12345
    r'Booking\s*(?:ID|#|Number)?\s*:?\s*(\d+)',  # Booking ID: 123, Booking #123
    r'Order\s*(?:ID|#|Number)?\s*:?\s*(\d+)',    # Order ID: 123
    r'Confirmation\s*(?:ID|#|Number)?\s*:?\s*(\d+)',  # Confirmation #123
    r'(?:^|\s)(\d{9,12})(?:\s|$)',               # Standalone 9-12 digit numbers
]

# Date patterns - matches various date formats
DATE_PATTERNS = [
    # ISO format: 2025-11-15, 2025/11/15
    (r'(\d{4}[-/]\d{2}[-/]\d{2})', '%Y-%m-%d'),
    # US format: 11/15/2025, 11-15-2025
    (r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', '%m/%d/%Y'),
    # Written format with day of week: Thursday Dec 04, 2025 (ignore day of week)
    (r'(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})', '%b %d, %Y'),
    # Written format: November 15, 2025
    (r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})', '%B %d, %Y'),
    # Short written: Nov 15, 2025
    (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})', '%b %d, %Y'),
]

# Location patterns - common parking facility indicators
LOCATION_PATTERNS = [
    r'(?:at|location|facility|garage|lot):\s*([^\n,]+)',
    r'(?:parking\s+(?:at|in|near))\s+([^\n,]+)',
    r'(?:address|venue):\s*([^\n,]+)',
]

# User info patterns
USER_INFO_PATTERNS = {
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'amount': r'\$\s*(\d+(?:\.\d{2})?)',
}

# Booking type keywords
BOOKING_TYPE_KEYWORDS = {
    'confirmed': ['confirmed', 'advance', 'pre-booked', 'reservation'],
    'on-demand': ['on-demand', 'same-day', 'instant', 'immediate'],
    'third-party': ['third-party', 'expedia', 'priceline', 'booking.com', 'hotels.com'],
}


class PatternExtractor:
    """
    Fast pattern-based extraction for structured ticket data.
    
    This class attempts to extract booking information using regex patterns
    and HTML parsing before falling back to LLM extraction. It's optimized
    for speed and works well with structured ticket formats.
    """
    
    def __init__(self):
        """Initialize the pattern extractor."""
        self.booking_id_regex = [re.compile(pattern, re.IGNORECASE) for pattern in BOOKING_ID_PATTERNS]
        self.location_regex = [re.compile(pattern, re.IGNORECASE) for pattern in LOCATION_PATTERNS]
        self.email_regex = re.compile(USER_INFO_PATTERNS['email'])
        self.amount_regex = re.compile(USER_INFO_PATTERNS['amount'])
    
    def extract_from_html(self, html_content: str) -> Dict:
        """
        Extract booking information from HTML-formatted ticket content.
        
        Uses BeautifulSoup to parse structured HTML and extract information
        from common ticket template formats.
        
        Args:
            html_content: HTML string from ticket notes
            
        Returns:
            Dict containing:
                - booking_info: Dict with extracted fields
                - confidence: "high" | "medium" | "low"
                - found: bool indicating if booking info was found
                - extraction_method: "pattern"
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract text content for pattern matching
            text_content = soup.get_text(separator='\n', strip=True)
            
            # Try to find structured data in tables or divs
            booking_info = {}
            
            # Look for tables with booking information
            tables = soup.find_all('table')
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text(strip=True).lower()
                        value = cells[1].get_text(strip=True)
                        
                        # Map common labels to fields
                        if 'booking' in label or 'order' in label or 'confirmation' in label:
                            booking_info['booking_id'] = value
                        elif 'amount' in label or 'total' in label or 'price' in label:
                            amount_match = self.amount_regex.search(value)
                            if amount_match:
                                booking_info['amount'] = float(amount_match.group(1))
                        elif 'event' in label or 'parking date' in label or 'start' in label:
                            date_str = self._extract_date(value)
                            if date_str:
                                booking_info['event_date'] = date_str
                        elif 'reservation' in label or 'booked' in label or 'created' in label:
                            date_str = self._extract_date(value)
                            if date_str:
                                booking_info['reservation_date'] = date_str
                        elif 'location' in label or 'facility' in label or 'address' in label:
                            booking_info['location'] = value
                        elif 'email' in label:
                            booking_info['customer_email'] = value
            
            # Fall back to text extraction if table parsing didn't find much
            if len(booking_info) < 2:
                text_result = self.extract_from_text(text_content)
                booking_info.update(text_result['booking_info'])
                # Also get the booking_not_found flag from text extraction
                booking_not_found = text_result.get('booking_not_found', False)
            else:
                # Check for booking not found in HTML content
                booking_not_found = self._check_booking_not_found(text_content)
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(booking_info)
            found = len(booking_info) > 0
            
            return {
                "booking_info": booking_info,
                "confidence": confidence,
                "found": found,
                "extraction_method": "pattern",
                "booking_not_found": booking_not_found
            }
            
        except Exception as e:
            # If HTML parsing fails, fall back to text extraction
            print(f"HTML parsing error: {e}")
            return self.extract_from_text(html_content)
    
    def extract_from_text(self, text_content: str) -> Dict:
        """
        Extract booking information from plain text using regex patterns.
        
        This method extracts the SYSTEM DATA (source of truth) from ticket notes.
        The system data represents the actual booking information from ParkWhiz.
        
        Args:
            text_content: Plain text from ticket notes
            
        Returns:
            Dict containing:
                - booking_info: Dict with extracted fields
                - confidence: "high" | "medium" | "low"
                - found: bool indicating if booking info was found
                - extraction_method: "pattern"
        """
        if not text_content or not text_content.strip():
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "pattern"
            }
        
        booking_info = {}
        
        # Extract booking ID
        booking_id = self._extract_booking_id(text_content)
        if booking_id:
            booking_info['booking_id'] = booking_id
        
        # Extract dates with context-aware labeling from SYSTEM DATA (source of truth)
        # Look for specific labels in system response sections
        booking_created_match = re.search(r'Booking Created:\s*(.+?)(?:\n|$)', text_content, re.IGNORECASE)
        if booking_created_match:
            date_str = self._extract_date(booking_created_match.group(1))
            if date_str:
                booking_info['reservation_date'] = date_str
        
        parking_start_match = re.search(r'Parking Pass Start Time:\s*(.+?)(?:\n|$)', text_content, re.IGNORECASE)
        if parking_start_match:
            date_str = self._extract_date(parking_start_match.group(1))
            if date_str:
                booking_info['event_date'] = date_str
        
        # Extract system user information (actual booking owner)
        # Use a more restrictive pattern that only captures email addresses
        user_email_match = re.search(r'User Email:\s*([^\s\n]+@[^\s\n]+)', text_content, re.IGNORECASE)
        if user_email_match:
            email = user_email_match.group(1).strip()
            if '@' in email:
                booking_info['user_email'] = email
        
        user_name_match = re.search(r'User Name:\s*(.+?)(?:\n|$)', text_content, re.IGNORECASE)
        if user_name_match:
            name = user_name_match.group(1).strip()
            if name:
                booking_info['user_name'] = name
        
        # Also extract customer request information for comparison (but don't use as primary data)
        customer_section_match = re.search(
            r'Refund Request Information(.*?)(?=Booking information found|$)', 
            text_content, 
            re.IGNORECASE | re.DOTALL
        )
        
        if customer_section_match:
            customer_section = customer_section_match.group(1)
            
            # Extract customer's claimed information for mismatch detection
            # Use a more restrictive pattern that only captures email addresses (stops at newline or whitespace)
            customer_email_match = re.search(r'Email:\s*([^\s\n]+@[^\s\n]+)', customer_section, re.IGNORECASE)
            if customer_email_match:
                email = customer_email_match.group(1).strip()
                if '@' in email:
                    booking_info['customer_email'] = email
            
            customer_name_match = re.search(r'Name:\s*(.+?)(?:\n|$)', customer_section, re.IGNORECASE)
            if customer_name_match:
                name = customer_name_match.group(1).strip()
                if name:
                    booking_info['customer_name'] = name
            
            # Extract customer's claimed arrival date for mismatch detection
            arrival_match = re.search(r'Arrival:\s*(.+?)(?:\n|$)', customer_section, re.IGNORECASE)
            if arrival_match:
                date_str = self._extract_date(arrival_match.group(1))
                if date_str:
                    booking_info['customer_claimed_date'] = date_str
        
        # Fallback: extract all dates if specific labels not found
        if 'event_date' not in booking_info:
            dates = self._extract_dates_with_validation(text_content)
            if dates:
                # Try to identify which date is which based on context
                if len(dates) >= 2:
                    # Assume first date is reservation, second is event
                    if 'reservation_date' not in booking_info:
                        booking_info['reservation_date'] = dates[0]
                    booking_info['event_date'] = dates[1]
                elif len(dates) == 1:
                    # Single date - likely event date
                    booking_info['event_date'] = dates[0]
        
        # Extract location (system data)
        location = self._extract_location(text_content)
        if location:
            booking_info['location'] = location
        
        # Extract amount (system data)
        amount_match = self.amount_regex.search(text_content)
        if amount_match:
            booking_info['amount'] = float(amount_match.group(1))
        
        # Infer booking type from keywords
        booking_type = self._infer_booking_type(text_content)
        if booking_type:
            booking_info['booking_type'] = booking_type
        
        # Check for "booking not found" scenario
        booking_not_found = self._check_booking_not_found(text_content)
        
        # Calculate confidence
        confidence = self._calculate_pattern_confidence(booking_info)
        found = len(booking_info) > 0
        
        logger.info(f"Pattern extraction complete: found={found}, confidence={confidence}, "
                   f"fields={list(booking_info.keys())}")
        
        return {
            "booking_info": booking_info,
            "confidence": confidence,
            "found": found,
            "extraction_method": "pattern",
            "booking_not_found": booking_not_found
        }
    
    def _extract_booking_id(self, text: str) -> Optional[str]:
        """Extract booking ID using regex patterns."""
        for regex in self.booking_id_regex:
            match = regex.search(text)
            if match:
                # Return the full match or first group if it exists
                return match.group(1) if match.groups() else match.group(0)
        return None
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract all dates from text and return in ISO format."""
        dates = []
        for pattern, date_format in DATE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_str = match.group(1)
                iso_date = self._parse_date_to_iso(date_str, date_format)
                if iso_date and iso_date not in dates:
                    dates.append(iso_date)
        return dates
    
    def _extract_dates_with_validation(self, text: str) -> List[str]:
        """
        Extract dates from text with validation to avoid billing dates.
        
        This method filters out dates that are likely billing dates rather than
        parking event dates, similar to the LLM validation logic.
        """
        from datetime import datetime, timedelta
        
        all_dates = self._extract_dates(text)
        valid_dates = []
        current_dt = datetime.now()
        
        for date_str in all_dates:
            try:
                date_dt = datetime.fromisoformat(date_str)
                
                # Skip dates that are more than 3 months in the future (likely billing dates)
                if date_dt > current_dt + timedelta(days=90):
                    print(f"Pattern extractor: Skipping future date {date_str} (likely billing date)")
                    continue
                
                # Skip dates that are more than 1 year in the past (likely wrong year)
                if date_dt < current_dt - timedelta(days=365):
                    print(f"Pattern extractor: Skipping old date {date_str} (likely wrong year)")
                    continue
                
                valid_dates.append(date_str)
                
            except ValueError:
                continue
        
        return valid_dates
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract a single date from text."""
        dates = self._extract_dates(text)
        return dates[0] if dates else None
    
    def _parse_date_to_iso(self, date_str: str, date_format: str) -> Optional[str]:
        """Parse date string to ISO format (YYYY-MM-DD)."""
        try:
            # Handle flexible separators
            if date_format == '%Y-%m-%d':
                date_str = date_str.replace('/', '-')
            elif date_format == '%m/%d/%Y':
                date_str = date_str.replace('-', '/')
            
            dt = datetime.strptime(date_str, date_format)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract location using regex patterns."""
        for regex in self.location_regex:
            match = regex.search(text)
            if match:
                location = match.group(1).strip()
                # Clean up location (remove extra whitespace, limit length)
                location = ' '.join(location.split())
                return location[:200]  # Limit to 200 chars
        return None
    
    def _infer_booking_type(self, text: str) -> Optional[str]:
        """Infer booking type from keywords in text."""
        text_lower = text.lower()
        
        for booking_type, keywords in BOOKING_TYPE_KEYWORDS.items():
            if any(keyword in text_lower for keyword in keywords):
                return booking_type
        
        return None
    
    def _check_booking_not_found(self, text: str) -> bool:
        """
        Check if the text indicates that booking information was not found.
        
        Args:
            text: Text content to check
            
        Returns:
            True if booking not found indicators are present
        """
        booking_not_found_phrases = [
            "booking information not found",
            "booking not found",
            "could not find booking",
            "unable to locate booking",
            "booking does not exist",
            "invalid booking number",
            "booking id not found"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in booking_not_found_phrases)
    
    def _calculate_pattern_confidence(self, booking_info: Dict) -> str:
        """
        Calculate confidence level for pattern-based extraction.
        
        Pattern extraction is generally less confident than LLM extraction
        because it can't understand context as well.
        
        Args:
            booking_info: Extracted booking information
            
        Returns:
            "high" | "medium" | "low"
        """
        if not booking_info:
            return "low"
        
        # Count critical fields
        critical_fields = ["booking_id", "event_date"]
        critical_count = sum(1 for field in critical_fields if field in booking_info)
        
        # Count optional fields
        optional_fields = ["amount", "reservation_date", "location", "booking_type", "customer_email"]
        optional_count = sum(1 for field in optional_fields if field in booking_info)
        
        # Pattern extraction is more conservative with confidence
        # High confidence: both critical + at least 4 optional fields
        if critical_count == 2 and optional_count >= 4:
            return "high"
        
        # Medium confidence: both critical + at least 2 optional fields
        elif critical_count == 2 and optional_count >= 2:
            return "medium"
        
        # Low confidence: missing critical fields or very incomplete
        else:
            return "low"
