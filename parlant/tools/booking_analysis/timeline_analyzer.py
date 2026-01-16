"""
Timeline analysis components for deterministic overstay and early arrival detection.

This module provides pure mathematical calculations for timeline analysis without
LLM calls, focusing on deterministic rule-based logic for webhook automation.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
import logging
from ..refund_decision.classification_config import ClassificationConfigManager

logger = logging.getLogger(__name__)


@dataclass
class TimelineAnalysis:
    """
    Structured timeline analysis results with calculated durations and scenario classification.
    
    This data model contains all timeline-related calculations needed for
    "Paid Again" classification decisions.
    """
    booking_start: Optional[datetime]
    booking_end: Optional[datetime]
    arrival_time: Optional[datetime]
    exit_time: Optional[datetime]
    
    # Calculated fields (deterministic mathematical calculations)
    early_arrival_minutes: int = 0  # 0 if on-time or late
    overstay_minutes: int = 0      # 0 if no overstay
    actual_parking_duration: int = 0  # Total minutes parked
    booked_duration: int = 0       # Total minutes booked
    
    # Analysis results
    has_early_arrival: bool = False
    has_overstay: bool = False
    is_timeline_consistent: bool = True
    timeline_issues: List[str] = None
    
    def __post_init__(self):
        """Initialize timeline_issues as empty list if None."""
        if self.timeline_issues is None:
            self.timeline_issues = []


class TimelineAnalyzer:
    """
    Deterministic timeline analyzer using pure mathematical calculations.
    
    This class performs overstay and early arrival detection using only
    date arithmetic and threshold comparisons. NO LLM calls are made.
    """
    
    def __init__(self, config: Optional[Dict] = None, config_manager: Optional[ClassificationConfigManager] = None):
        """
        Initialize timeline analyzer with configurable thresholds.
        
        Args:
            config: Configuration dict with tolerance settings (legacy support)
            config_manager: ClassificationConfigManager instance for configurable thresholds
        """
        if config is None:
            config = {}
        
        # Initialize configuration manager
        if config_manager is None:
            config_path = config.get("config_path")
            self.config_manager = ClassificationConfigManager(config_path)
        else:
            self.config_manager = config_manager
        
        # Load thresholds from configuration
        try:
            classification_config = self.config_manager.load_config()
            self.overstay_tolerance_minutes = classification_config.tolerance_thresholds.overstay_tolerance_minutes
            self.early_arrival_tolerance_minutes = classification_config.tolerance_thresholds.early_arrival_tolerance_minutes
            logger.info("Loaded tolerance thresholds from configuration")
        except Exception as e:
            logger.warning(f"Failed to load tolerance thresholds from config: {e}. Using defaults.")
            self.overstay_tolerance_minutes = config.get("overstay_tolerance_minutes", 15)
            self.early_arrival_tolerance_minutes = config.get("early_arrival_tolerance_minutes", 30)
        
        logger.info(f"TimelineAnalyzer initialized with overstay_tolerance={self.overstay_tolerance_minutes}min, "
                   f"early_arrival_tolerance={self.early_arrival_tolerance_minutes}min")
    
    def analyze_timeline(
        self, 
        booking_start: Optional[datetime],
        booking_end: Optional[datetime],
        arrival_time: Optional[datetime],
        exit_time: Optional[datetime]
    ) -> TimelineAnalysis:
        """
        DETERMINISTIC timeline analysis using pure mathematical calculations.
        NO LLM calls - uses only date arithmetic and threshold comparisons.
        
        Args:
            booking_start: When the booking window starts
            booking_end: When the booking window ends
            arrival_time: When customer actually arrived
            exit_time: When customer actually left
        
        Returns:
            TimelineAnalysis with calculated durations and scenario classification
        """
        logger.info(f"Timeline Analysis Input - Booking: {booking_start} to {booking_end}, "
                   f"Actual: {arrival_time} to {exit_time}")
        
        # Log the raw datetime values for debugging
        if booking_start and booking_end:
            booking_hours = (booking_end - booking_start).total_seconds() / 3600
            logger.debug(f"Booking window duration: {booking_hours:.1f} hours")
        
        if arrival_time and exit_time:
            actual_hours = (exit_time - arrival_time).total_seconds() / 3600
            logger.debug(f"Actual parking duration: {actual_hours:.1f} hours")
        
        # Initialize analysis result
        analysis = TimelineAnalysis(
            booking_start=booking_start,
            booking_end=booking_end,
            arrival_time=arrival_time,
            exit_time=exit_time,
            timeline_issues=[]  # Initialize explicitly
        )
        
        # Calculate booked duration if we have booking window
        if booking_start and booking_end:
            booked_delta = booking_end - booking_start
            analysis.booked_duration = int(booked_delta.total_seconds() / 60)
            logger.debug(f"Booked duration: {analysis.booked_duration} minutes")
            
            # Validate reasonable booking duration (warn if >= 12 hours)
            if analysis.booked_duration >= 12 * 60:  # 12 hours or more
                logger.warning(f"Unusually long booking duration: {analysis.booked_duration} minutes "
                             f"({analysis.booked_duration / 60:.1f} hours). "
                             f"Booking window: {booking_start} to {booking_end}")
                issue_msg = f"Unusually long booking duration: {analysis.booked_duration / 60:.1f} hours"
                analysis.timeline_issues.append(issue_msg)
        
        # Calculate actual parking duration if we have arrival and exit
        if arrival_time and exit_time:
            actual_delta = exit_time - arrival_time
            analysis.actual_parking_duration = int(actual_delta.total_seconds() / 60)
            logger.debug(f"Actual parking duration: {analysis.actual_parking_duration} minutes")
            
            # Validate reasonable actual parking duration
            if analysis.actual_parking_duration >= 12 * 60:  # 12 hours or more
                logger.warning(f"Unusually long actual parking duration: {analysis.actual_parking_duration} minutes "
                             f"({analysis.actual_parking_duration / 60:.1f} hours). "
                             f"Parked from: {arrival_time} to {exit_time}")
                analysis.timeline_issues.append(f"Unusually long actual parking: {analysis.actual_parking_duration / 60:.1f} hours")
        
        # Calculate early arrival if we have both booking start and arrival
        if booking_start and arrival_time:
            if arrival_time < booking_start:
                early_delta = booking_start - arrival_time
                analysis.early_arrival_minutes = int(early_delta.total_seconds() / 60)
                analysis.has_early_arrival = analysis.early_arrival_minutes > self.early_arrival_tolerance_minutes
                logger.debug(f"Early arrival: {analysis.early_arrival_minutes} minutes "
                           f"(threshold: {self.early_arrival_tolerance_minutes})")
            else:
                # Customer arrived on time or late
                late_delta = arrival_time - booking_start
                late_minutes = int(late_delta.total_seconds() / 60)
                logger.debug(f"Customer arrived {late_minutes} minutes after booking start (on-time or late)")
        
        # Calculate overstay - this is the key calculation for "Paid Again" scenarios
        analysis = self._calculate_overstay(analysis)
        
        # Validate timeline consistency
        analysis = self._validate_timeline_consistency(analysis)
        
        logger.info(f"Timeline analysis complete: early_arrival={analysis.has_early_arrival}, "
                   f"overstay={analysis.has_overstay}, consistent={analysis.is_timeline_consistent}")
        
        return analysis
    
    def _calculate_overstay(self, analysis: TimelineAnalysis) -> TimelineAnalysis:
        """
        Calculate overstay duration using deterministic logic.
        
        Overstay occurs when:
        1. Customer arrives early AND stays past booking end, OR
        2. Customer arrives on-time but stays past booking end
        
        Args:
            analysis: Timeline analysis in progress
            
        Returns:
            Updated analysis with overstay calculations
        """
        # Need booking end and exit time to calculate overstay
        if not analysis.booking_end or not analysis.exit_time:
            logger.debug("Cannot calculate overstay: missing booking_end or exit_time")
            return analysis
        
        # Check if customer stayed past booking end
        if analysis.exit_time > analysis.booking_end:
            overstay_delta = analysis.exit_time - analysis.booking_end
            analysis.overstay_minutes = int(overstay_delta.total_seconds() / 60)
            analysis.has_overstay = analysis.overstay_minutes > self.overstay_tolerance_minutes
            
            logger.debug(f"Overstay calculation: {analysis.overstay_minutes} minutes past booking end "
                        f"(threshold: {self.overstay_tolerance_minutes})")
            
            # Special case: Early arrival + overstay = legitimate additional charge
            if analysis.has_early_arrival and analysis.has_overstay:
                total_extra_minutes = analysis.early_arrival_minutes + analysis.overstay_minutes
                logger.info(f"Early arrival + overstay detected: {analysis.early_arrival_minutes}min early + "
                           f"{analysis.overstay_minutes}min overstay = {total_extra_minutes}min total extra parking")
        
        return analysis
    
    def _validate_timeline_consistency(self, analysis: TimelineAnalysis) -> TimelineAnalysis:
        """
        Validate timeline for mathematical consistency and logical errors.
        
        Args:
            analysis: Timeline analysis in progress
            
        Returns:
            Updated analysis with consistency validation
        """
        issues = []
        
        # Check if arrival is after exit (impossible)
        if analysis.arrival_time and analysis.exit_time:
            if analysis.arrival_time >= analysis.exit_time:
                issues.append("Arrival time is after or equal to exit time")
                analysis.is_timeline_consistent = False
        
        # Check if booking window is valid
        if analysis.booking_start and analysis.booking_end:
            if analysis.booking_start >= analysis.booking_end:
                issues.append("Booking start time is after or equal to booking end time")
                analysis.is_timeline_consistent = False
        
        # Check for extremely long parking durations (likely data error)
        if analysis.actual_parking_duration > 24 * 60:  # More than 24 hours
            issues.append(f"Extremely long parking duration: {analysis.actual_parking_duration} minutes")
            analysis.is_timeline_consistent = False
        
        # Check for negative durations (should not happen with proper datetime handling)
        if analysis.early_arrival_minutes < 0 or analysis.overstay_minutes < 0:
            issues.append("Negative duration calculated (data error)")
            analysis.is_timeline_consistent = False
        
        # Check if actual parking is much longer than booked (potential data issue)
        if (analysis.actual_parking_duration > 0 and analysis.booked_duration > 0 and 
            analysis.actual_parking_duration > analysis.booked_duration * 3):
            issues.append(f"Actual parking ({analysis.actual_parking_duration}min) is much longer than booked ({analysis.booked_duration}min)")
            # Don't mark as inconsistent - this could be legitimate overstay
        
        # Append new issues to existing ones (don't overwrite)
        if analysis.timeline_issues is None:
            analysis.timeline_issues = []
        analysis.timeline_issues.extend(issues)
        
        if issues:
            logger.warning(f"Timeline consistency issues: {issues}")
        
        if analysis.timeline_issues:
            logger.debug(f"All timeline issues: {analysis.timeline_issues}")
        else:
            logger.debug("Timeline is mathematically consistent")
        
        return analysis


class TimelineExtractor:
    """
    Extracts arrival and exit times from ticket text using regex patterns.
    
    This class uses deterministic pattern matching to find timeline data
    in various ticket formats without LLM calls.
    """
    
    def __init__(self):
        """Initialize timeline extractor with regex patterns."""
        # Arrival time patterns
        self.arrival_patterns = [
            r'Arrival:\s*([^\n]+)',
            r'Arrived:\s*([^\n]+)',
            r'Entry Time:\s*([^\n]+)',
            r'Check-in:\s*([^\n]+)',
            r'Parking Pass Start Time:\s*([^\n]+)',
            r'Started parking:\s*([^\n]+)',
            r'Entered at:\s*([^\n]+)',
        ]
        
        # Exit time patterns
        self.exit_patterns = [
            r'Exit:\s*([^\n]+)',
            r'Departure:\s*([^\n]+)',
            r'Left:\s*([^\n]+)',
            r'Check-out:\s*([^\n]+)',
            r'Parking Pass End Time:\s*([^\n]+)',
            r'Finished parking:\s*([^\n]+)',
            r'Exited at:\s*([^\n]+)',
        ]
        
        # Booking window patterns
        self.booking_start_patterns = [
            r'Booking Start:\s*([^\n]+)',
            r'Reserved from:\s*([^\n]+)',
            r'Booking window starts:\s*([^\n]+)',
            r'Pass valid from:\s*([^\n]+)',
            r'Parking Pass Start Time:\s*([^\n]+)',
        ]
        
        self.booking_end_patterns = [
            r'Booking End:\s*([^\n]+)',
            r'Reserved until:\s*([^\n]+)',
            r'Booking window ends:\s*([^\n]+)',
            r'Pass valid until:\s*([^\n]+)',
            r'Parking Pass End Time:\s*([^\n]+)',
        ]
        
        # Date/time parsing patterns
        self.datetime_patterns = [
            # ISO format with time: 2025-12-13 14:30:00, 2025-12-13T14:30:00
            (r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}(?::\d{2})?)', '%Y-%m-%d %H:%M:%S'),
            (r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2})', '%Y-%m-%d %H:%M'),
            # US format with time: 12/13/2025 2:30 PM, 12/13/2025 14:30
            (r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM))', '%m/%d/%Y %I:%M %p'),
            (r'(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})', '%m/%d/%Y %H:%M'),
            # Written format with time: Dec 13, 2025 2:30 PM
            (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\s+\d{1,2}:\d{2}\s*(?:AM|PM))', '%b %d, %Y %I:%M %p'),
            # Date only patterns (will assume start/end of day)
            (r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4})', '%b %d, %Y'),
            (r'(\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'(\d{1,2}/\d{1,2}/\d{4})', '%m/%d/%Y'),
        ]
        
        logger.info("TimelineExtractor initialized with regex patterns")
    
    def extract_timeline_data(self, ticket_text: str) -> Dict[str, Optional[datetime]]:
        """
        Extract timeline data from ticket text using regex patterns.
        
        Args:
            ticket_text: Raw text from ticket notes/conversations
            
        Returns:
            Dict containing:
                - arrival_time: When customer arrived (if found)
                - exit_time: When customer left (if found)
                - booking_start: When booking window starts (if found)
                - booking_end: When booking window ends (if found)
        """
        logger.debug(f"Extracting timeline data from {len(ticket_text)} characters of text")
        
        result = {
            'arrival_time': None,
            'exit_time': None,
            'booking_start': None,
            'booking_end': None
        }
        
        if not ticket_text or not ticket_text.strip():
            logger.warning("Empty ticket text provided for timeline extraction")
            return result
        
        # Extract arrival time
        result['arrival_time'] = self._extract_time_by_patterns(ticket_text, self.arrival_patterns, "arrival")
        
        # Extract exit time
        result['exit_time'] = self._extract_time_by_patterns(ticket_text, self.exit_patterns, "exit")
        
        # Extract booking window start
        result['booking_start'] = self._extract_time_by_patterns(ticket_text, self.booking_start_patterns, "booking_start")
        
        # Extract booking window end
        result['booking_end'] = self._extract_time_by_patterns(ticket_text, self.booking_end_patterns, "booking_end")
        
        # Log what we found
        found_fields = [k for k, v in result.items() if v is not None]
        logger.info(f"Timeline extraction complete. Found: {found_fields}")
        
        return result
    
    def _extract_time_by_patterns(self, text: str, patterns: List[str], field_name: str) -> Optional[datetime]:
        """
        Extract a datetime using a list of regex patterns.
        
        Args:
            text: Text to search in
            patterns: List of regex patterns to try
            field_name: Name of field being extracted (for logging)
            
        Returns:
            Parsed datetime or None if not found
        """
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_str = match.group(1).strip()
                parsed_time = self._parse_datetime_string(time_str)
                if parsed_time:
                    logger.debug(f"Found {field_name}: {time_str} -> {parsed_time}")
                    return parsed_time
                else:
                    logger.debug(f"Found {field_name} text '{time_str}' but could not parse as datetime")
        
        logger.debug(f"No {field_name} found in text")
        return None
    
    def _parse_datetime_string(self, time_str: str) -> Optional[datetime]:
        """
        Parse a datetime string using various format patterns.
        
        Args:
            time_str: String containing date/time information
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        # Clean up the string
        time_str = time_str.strip()
        
        # Try each datetime pattern
        for pattern, date_format in self.datetime_patterns:
            match = re.search(pattern, time_str, re.IGNORECASE)
            if match:
                datetime_part = match.group(1)
                try:
                    # Handle T separator in ISO format
                    if 'T' in datetime_part:
                        datetime_part = datetime_part.replace('T', ' ')
                    
                    # Parse the datetime
                    if date_format.endswith('%S'):
                        # Try with seconds first
                        try:
                            return datetime.strptime(datetime_part, date_format)
                        except ValueError:
                            # Fall back to format without seconds
                            date_format_no_sec = date_format.replace(':%S', '')
                            return datetime.strptime(datetime_part, date_format_no_sec)
                    else:
                        return datetime.strptime(datetime_part, date_format)
                        
                except ValueError as e:
                    logger.debug(f"Failed to parse '{datetime_part}' with format '{date_format}': {e}")
                    continue
        
        logger.debug(f"Could not parse datetime string: '{time_str}'")
        return None