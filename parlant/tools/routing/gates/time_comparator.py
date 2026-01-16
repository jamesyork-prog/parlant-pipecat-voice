"""
Time comparison logic for pre-arrival determination.

This module implements the PreArrivalComparator class that performs timezone-aware
time comparison to determine if a cancellation request is made before the booking
start time (pre-arrival).
"""

import time
import logging
from datetime import datetime
from typing import Optional, Dict, List

from ..models import PreArrivalResult

logger = logging.getLogger(__name__)


class PreArrivalComparator:
    """
    Performs timezone-aware time comparison for pre-arrival determination.
    
    Handles:
    - Boundary conditions (current == start → NOT pre-arrival)
    - DST transitions
    - Far future/past bookings
    - Suspicious timing patterns
    
    Performance: <1ms for time arithmetic
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize comparator with boundary condition configuration.
        
        Args:
            config: Optional configuration for boundary handling
        """
        self.config = config or {}
        self.max_future_months = self.config.get("max_future_months", 12)
        self.max_past_days = self.config.get("max_past_days", 7)
    
    def compare_times(self, booking_start: datetime, current_time: datetime) -> PreArrivalResult:
        """
        Compare current time with booking start time.
        
        Args:
            booking_start: Timezone-aware booking start datetime
            current_time: Timezone-aware current datetime
            
        Returns:
            PreArrivalResult with comparison outcome and metrics
            
        Processing time: <1ms (simple arithmetic)
        """
        start = time.perf_counter()
        
        # Calculate time difference
        time_diff = booking_start - current_time
        hours_before = time_diff.total_seconds() / 3600
        
        # Determine pre-arrival status
        # Boundary condition: current == start is NOT pre-arrival
        is_pre_arrival = time_diff.total_seconds() > 0
        
        # Check for suspicious patterns and get warning flags
        suspicious_warnings = self._check_suspicious_timing(hours_before, booking_start, current_time)
        
        processing_ms = int((time.perf_counter() - start) * 1000)
        
        return PreArrivalResult(
            is_pre_arrival=is_pre_arrival,
            hours_before_booking=hours_before,  # Always include hours_before_booking
            current_time=current_time,
            booking_start_time=booking_start,
            timezone_used=booking_start.tzinfo.zone if hasattr(booking_start.tzinfo, 'zone') else str(booking_start.tzinfo),
            processing_time_ms=processing_ms
        )
    
    def _check_suspicious_timing(self, hours_before: float, booking_start: datetime, current_time: datetime) -> List[str]:
        """
        Check for suspicious timing patterns and log warnings.
        
        Detects and logs:
        - Far past bookings (>7 days ago) - potential data error
        - Far future bookings (>12 months) - flag for review
        - Very short notice (<15 minutes) - unusual but valid
        - Extremely long durations (>7 days) - potential data error (requires booking end time)
        
        Args:
            hours_before: Hours between current time and booking start
            booking_start: Booking start datetime
            current_time: Current datetime
            
        Returns:
            List of warning flags for suspicious patterns detected
        """
        warnings = []
        
        # Far past booking (>7 days ago) - Requirement 17.1
        if hours_before < -24 * self.max_past_days:
            days_past = abs(hours_before) / 24
            warning_msg = (
                f"SUSPICIOUS_TIMING: Far past booking detected - "
                f"booking was {days_past:.1f} days ago (>{self.max_past_days} days threshold). "
                f"Booking start: {booking_start.isoformat()}, "
                f"Current time: {current_time.isoformat()}. "
                f"Potential data error - flagging for manual review."
            )
            logger.warning(warning_msg)
            warnings.append("far_past_booking")
        
        # Far future booking (>12 months) - Requirement 17.2
        elif hours_before > 24 * 30 * self.max_future_months:
            months_future = hours_before / (24 * 30)
            warning_msg = (
                f"SUSPICIOUS_TIMING: Far future booking detected - "
                f"booking is {months_future:.1f} months in future (>{self.max_future_months} months threshold). "
                f"Booking start: {booking_start.isoformat()}, "
                f"Current time: {current_time.isoformat()}. "
                f"Flagging for review but processing as pre-arrival."
            )
            logger.warning(warning_msg)
            warnings.append("far_future_booking")
        
        # Very short booking window (<15 minutes in future) - Requirement 17.3
        elif 0 < hours_before < 0.25:  # 0.25 hours = 15 minutes
            minutes_before = hours_before * 60
            info_msg = (
                f"SHORT_NOTICE: Very short notice booking detected - "
                f"booking starts in {minutes_before:.1f} minutes (<15 minutes threshold). "
                f"Booking start: {booking_start.isoformat()}, "
                f"Current time: {current_time.isoformat()}. "
                f"Unusual but valid - continuing processing."
            )
            logger.info(info_msg)
            warnings.append("short_notice_booking")
        
        # Log comprehensive context for all suspicious patterns - Requirement 17.5
        if warnings:
            context_msg = (
                f"TIMING_CONTEXT: Suspicious pattern context - "
                f"Hours before booking: {hours_before:.2f}, "
                f"Timezone: {booking_start.tzinfo.zone if hasattr(booking_start.tzinfo, 'zone') else str(booking_start.tzinfo)}, "
                f"DST active: {self._check_dst_status(booking_start)}, "
                f"Thresholds: max_past_days={self.max_past_days}, max_future_months={self.max_future_months}, "
                f"Patterns detected: {', '.join(warnings)}"
            )
            logger.info(context_msg)
        
        return warnings
    
    def _check_dst_status(self, dt: datetime) -> str:
        """
        Check if DST is active for the given datetime.
        
        Args:
            dt: Timezone-aware datetime to check
            
        Returns:
            String indicating DST status
        """
        try:
            if hasattr(dt, 'dst') and dt.dst() is not None:
                return "yes" if dt.dst().total_seconds() > 0 else "no"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def check_booking_duration(self, booking_start: datetime, booking_end: datetime) -> List[str]:
        """
        Check for extremely long booking durations.
        
        This method implements Requirement 17.4 but requires booking end time data
        which is not currently available in the ticket structure. This method is
        provided for future use when booking end time becomes available.
        
        Args:
            booking_start: Booking start datetime
            booking_end: Booking end datetime
            
        Returns:
            List of warning flags for long duration patterns
        """
        warnings = []
        
        # Calculate booking duration
        duration = booking_end - booking_start
        duration_hours = duration.total_seconds() / 3600
        duration_days = duration_hours / 24
        
        # Extremely long duration (>7 days) - Requirement 17.4
        if duration_days > 7:
            warning_msg = (
                f"SUSPICIOUS_DURATION: Extremely long booking duration detected - "
                f"duration is {duration_days:.1f} days (>7 days threshold). "
                f"Booking start: {booking_start.isoformat()}, "
                f"Booking end: {booking_end.isoformat()}. "
                f"Potential data error - flagging for manual review."
            )
            logger.warning(warning_msg)
            warnings.append("extremely_long_duration")
        
        # Very short duration (<15 minutes) for additional context
        elif duration_hours < 0.25:  # 15 minutes
            minutes_duration = duration_hours * 60
            info_msg = (
                f"SHORT_DURATION: Very short booking duration detected - "
                f"duration is {minutes_duration:.1f} minutes (<15 minutes). "
                f"Booking start: {booking_start.isoformat()}, "
                f"Booking end: {booking_end.isoformat()}. "
                f"Unusual but may be valid for short-term parking."
            )
            logger.info(info_msg)
            warnings.append("very_short_duration")
        
        return warnings