"""
Pre-arrival date parser for booking start time extraction.

This module provides multi-format date parsing with fallback strategies
for extracting booking start times from diverse Zapier date formats.

Supports multiple formats with priority-based fallback:
1. Primary: "Wednesday Dec 17, 2025, 07:45 PM" (Zapier standard)
2. ISO 8601: "2025-12-17T19:45:00" 
3. US Format: "12/17/2025 7:45 PM"

Performance: <100ms even with all format attempts
"""

from datetime import datetime
from typing import Optional, List, Dict
import re
import time
import logging

logger = logging.getLogger(__name__)


class PreArrivalDateParser:
    """
    Parses booking start times from diverse Zapier date formats.
    
    Supports multiple formats with priority-based fallback:
    1. Primary: "Wednesday Dec 17, 2025, 07:45 PM" (Zapier standard)
    2. ISO 8601: "2025-12-17T19:45:00" 
    3. US Format: "12/17/2025 7:45 PM"
    
    Performance: <100ms even with all format attempts
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize parser with configurable format list.
        
        Args:
            config: Optional configuration with date_formats list
        """
        self.config = config or {}
        self.date_formats = self._load_date_formats()
        self.last_successful_format: Optional[str] = None  # Track last successful format for metrics
    
    def parse_booking_start_time(self, ticket_data: Dict) -> Optional[datetime]:
        """
        Parse booking start time from ticket data.
        
        Args:
            ticket_data: Zapier webhook payload
            
        Returns:
            Naive datetime object (timezone applied separately), or None if parsing fails
            
        Processing time: <100ms with all format attempts
        """
        start = time.perf_counter()
        
        try:
            # Extract start time field with robust error handling
            start_time_raw = ticket_data.get("Parking Pass Start Time", "")
            
            # Handle None and non-string types gracefully
            if start_time_raw is None:
                logger.warning("Parking Pass Start Time field is None")
                return None
            
            if not isinstance(start_time_raw, str):
                logger.warning(f"Parking Pass Start Time field is not a string: {type(start_time_raw)}")
                return None
            
            start_time_str = start_time_raw.strip()
            
            if not start_time_str:
                logger.warning("Parking Pass Start Time field missing or empty")
                return None
            
            # Normalize whitespace and common formatting issues
            start_time_str = self._normalize_date_string(start_time_str)
            
            # Try each format in priority order
            for format_str in self.date_formats:
                try:
                    parsed_dt = datetime.strptime(start_time_str, format_str)
                    
                    processing_ms = int((time.perf_counter() - start) * 1000)
                    logger.info(f"Date parsed successfully with format '{format_str}' in {processing_ms}ms")
                    
                    # Track successful format for metrics
                    self.last_successful_format = format_str
                    
                    return parsed_dt
                    
                except ValueError:
                    continue  # Try next format
            
            # All formats failed
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Date parsing failed for '{start_time_str}' after {processing_ms}ms")
            self.last_successful_format = None
            return None
            
        except Exception as e:
            # Catch any unexpected errors to ensure robustness
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Unexpected error in date parsing after {processing_ms}ms: {e}")
            return None
    
    def _load_date_formats(self) -> List[str]:
        """Load date format strings from configuration."""
        default_formats = [
            "%A %b %d, %Y, %I:%M %p",  # "Wednesday Dec 17, 2025, 07:45 PM"
            "%Y-%m-%dT%H:%M:%S",       # "2025-12-17T19:45:00"
            "%m/%d/%Y %I:%M %p",       # "12/17/2025 7:45 PM"
            "%b %d, %Y %I:%M:%S %p",   # "Dec 17, 2025 7:45:00 PM"
            "%Y-%m-%d %H:%M:%S"        # "2025-12-17 19:45:00"
        ]
        return self.config.get("date_formats", default_formats)
    
    def _normalize_date_string(self, date_str: str) -> str:
        """
        Normalize date string for consistent parsing.
        
        Args:
            date_str: Raw date string from Zapier
            
        Returns:
            Normalized date string
        """
        # Remove extra whitespace (including tabs and newlines)
        normalized = re.sub(r'\s+', ' ', date_str.strip())
        
        # Handle common formatting variations
        # "Dec." → "Dec", "PM." → "PM", "P.M." → "PM", "A.M." → "AM", etc.
        normalized = re.sub(r'\.(?=\s|$)', '', normalized)
        
        # Handle P.M. and A.M. specifically
        normalized = re.sub(r'P\.M\.?', 'PM', normalized, flags=re.IGNORECASE)
        normalized = re.sub(r'A\.M\.?', 'AM', normalized, flags=re.IGNORECASE)
        
        # Handle abbreviated day names by expanding them to full names
        day_abbreviations = {
            'Mon': 'Monday', 'Tue': 'Tuesday', 'Wed': 'Wednesday', 
            'Thu': 'Thursday', 'Fri': 'Friday', 'Sat': 'Saturday', 'Sun': 'Sunday'
        }
        
        for abbrev, full in day_abbreviations.items():
            # Replace abbreviated day names at the beginning of the string
            if normalized.startswith(abbrev + ' '):
                normalized = normalized.replace(abbrev + ' ', full + ' ', 1)
                break
        
        return normalized