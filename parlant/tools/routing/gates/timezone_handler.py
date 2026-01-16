"""
Pre-arrival timezone handler for booking location timezone conversion.

This module provides timezone conversion with inference fallback and DST handling
for converting booking start times to the correct local timezone.

Supports:
- IANA timezone identifiers (America/New_York, etc.)
- City-to-timezone inference fallback
- DST transition handling
- UTC fallback for missing data

Performance: <10ms per conversion using cached timezone data
"""

from datetime import datetime, timezone
import pytz
from typing import Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)


class TimezoneHandler:
    """
    Handles timezone conversion with inference fallback.
    
    Supports:
    - IANA timezone identifiers (America/New_York, etc.)
    - City-to-timezone inference fallback
    - DST transition handling
    - UTC fallback for missing data
    
    Performance: <10ms per conversion using cached timezone data
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize timezone handler with city mapping configuration.
        
        Args:
            config: Optional configuration with city_to_timezone_mapping
        """
        self.config = config or {}
        self.city_mapping = self._load_city_mapping()
        self.timezone_cache = {}  # Cache pytz timezone objects
    
    def apply_timezone_to_datetime(self, dt: datetime, ticket_data: Dict) -> Optional[datetime]:
        """
        Apply timezone to naive datetime object.
        
        Args:
            dt: Naive datetime from date parser
            ticket_data: Zapier webhook payload with timezone fields
            
        Returns:
            Timezone-aware datetime, or None if timezone handling fails
            
        Processing time: <10ms using cached timezone data
        """
        start = time.perf_counter()
        
        try:
            # Get timezone for booking location
            tz = self._determine_timezone(ticket_data)
            
            if tz is None:
                logger.error("Unable to determine timezone - escalating to next gate")
                return None
            
            # Apply timezone to naive datetime (handles DST automatically)
            localized_dt = tz.localize(dt)
            
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.info(f"Timezone applied successfully in {processing_ms}ms: {tz.zone}")
            
            return localized_dt
            
        except Exception as e:
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Timezone application failed after {processing_ms}ms: {e}")
            return None
    
    def get_current_time_in_timezone(self, tz: pytz.BaseTzInfo) -> datetime:
        """
        Get current time in specified timezone.
        
        Args:
            tz: pytz timezone object
            
        Returns:
            Current time in specified timezone
            
        Processing time: <5ms
        """
        start = time.perf_counter()
        
        try:
            utc_now = datetime.now(timezone.utc)
            current_in_tz = utc_now.astimezone(tz)
            
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.debug(f"Current time in {tz.zone} calculated in {processing_ms}ms")
            
            return current_in_tz
            
        except Exception as e:
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(f"Failed to get current time in timezone {tz.zone} after {processing_ms}ms: {e}")
            # Fallback to UTC
            return datetime.now(timezone.utc)
    
    def _determine_timezone(self, ticket_data: Dict) -> Optional[pytz.BaseTzInfo]:
        """
        Determine timezone using multiple strategies.
        
        Priority:
        1. Direct IANA timezone from "Location Time Zone" field
        2. Inference from "Location City, State" field
        3. UTC fallback with warning
        
        Args:
            ticket_data: Zapier webhook payload
            
        Returns:
            pytz timezone object, or None if all methods fail
        """
        # Strategy 1: Direct IANA timezone
        tz_str = ticket_data.get("Location Time Zone", "")
        if isinstance(tz_str, str) and tz_str.strip():
            tz = self._get_timezone_from_string(tz_str.strip())
            if tz:
                logger.debug(f"Using direct timezone: {tz_str}")
                return tz
            else:
                logger.warning(f"Invalid IANA timezone identifier: {tz_str}")
        
        # Strategy 2: City-to-timezone inference
        location_str = ticket_data.get("Location City, State", "")
        if isinstance(location_str, str) and location_str.strip():
            tz = self._infer_timezone_from_city(location_str.strip())
            if tz:
                logger.info(f"Timezone inferred from city: {location_str} → {tz.zone}")
                return tz
            else:
                logger.warning(f"Failed to infer timezone from location: {location_str}")
        
        # Strategy 3: UTC fallback
        logger.warning("Timezone determination failed - defaulting to UTC")
        return pytz.UTC
    
    def _get_timezone_from_string(self, tz_str: str) -> Optional[pytz.BaseTzInfo]:
        """
        Get pytz timezone from IANA string with caching.
        
        Args:
            tz_str: IANA timezone identifier
            
        Returns:
            pytz timezone object, or None if invalid
        """
        if not tz_str:
            return None
            
        if tz_str in self.timezone_cache:
            return self.timezone_cache[tz_str]
        
        try:
            tz = pytz.timezone(tz_str)
            self.timezone_cache[tz_str] = tz
            logger.debug(f"Cached timezone: {tz_str}")
            return tz
        except pytz.UnknownTimeZoneError:
            logger.warning(f"Unknown timezone: {tz_str}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting timezone {tz_str}: {e}")
            return None
    
    def _infer_timezone_from_city(self, location_str: str) -> Optional[pytz.BaseTzInfo]:
        """
        Infer timezone from city/state string.
        
        Args:
            location_str: City, state string (e.g., "New York, NY")
            
        Returns:
            pytz timezone object, or None if inference fails
        """
        if not location_str:
            logger.debug("Empty location string provided for timezone inference")
            return None
            
        # Normalize location string for lookup
        location_key = location_str.lower().strip()
        
        # Normalize whitespace: multiple spaces -> single space, spaces around commas
        location_key = ' '.join(location_key.split())
        location_key = location_key.replace(' ,', ',').replace(', ', ', ')
        
        logger.debug(f"Attempting timezone inference for normalized location: '{location_key}' (original: '{location_str}')")
        
        # Check direct mapping
        if location_key in self.city_mapping:
            tz_str = self.city_mapping[location_key]
            tz = self._get_timezone_from_string(tz_str)
            if tz:
                logger.info(f"Direct city match: '{location_str}' → {tz_str}")
                return tz
            else:
                logger.warning(f"Direct city match found but timezone invalid: '{location_str}' → {tz_str}")
        
        # Check partial matches (e.g., "New York" matches "New York, NY")
        for city_key, tz_str in self.city_mapping.items():
            # Split both the input and mapping key into parts
            location_parts = [part.strip().lower() for part in location_key.split(',')]
            city_parts = [part.strip().lower() for part in city_key.split(',')]
            
            # Check if any part of the input matches any part of the mapping
            # Use more sophisticated matching to avoid false positives
            for loc_part in location_parts:
                for city_part in city_parts:
                    if len(loc_part) > 2 and len(city_part) > 2:
                        # Check for exact substring match or word boundary match
                        if (loc_part in city_part or city_part in loc_part or
                            any(word == city_part for word in loc_part.split()) or
                            any(word == loc_part for word in city_part.split())):
                            
                            tz = self._get_timezone_from_string(tz_str)
                            if tz:
                                logger.info(f"Partial city match: '{location_str}' → '{city_key}' → {tz_str}")
                                return tz
                            else:
                                logger.warning(f"Partial city match found but timezone invalid: '{location_str}' → '{city_key}' → {tz_str}")
        
        logger.info(f"No timezone mapping found for location: '{location_str}' (normalized: '{location_key}')")
        return None
    
    def _load_city_mapping(self) -> Dict[str, str]:
        """Load city-to-timezone mapping from configuration."""
        default_mapping = {
            "new york, ny": "America/New_York",
            "new york": "America/New_York",
            "manhattan, ny": "America/New_York",
            "brooklyn, ny": "America/New_York",
            "queens, ny": "America/New_York",
            "bronx, ny": "America/New_York",
            "staten island, ny": "America/New_York",
            "los angeles, ca": "America/Los_Angeles",
            "los angeles": "America/Los_Angeles",
            "san francisco, ca": "America/Los_Angeles",
            "san francisco": "America/Los_Angeles",
            "san diego, ca": "America/Los_Angeles",
            "oakland, ca": "America/Los_Angeles",
            "sacramento, ca": "America/Los_Angeles",
            "chicago, il": "America/Chicago",
            "chicago": "America/Chicago",
            "dallas, tx": "America/Chicago",
            "houston, tx": "America/Chicago",
            "austin, tx": "America/Chicago",
            "san antonio, tx": "America/Chicago",
            "denver, co": "America/Denver",
            "colorado springs, co": "America/Denver",
            "phoenix, az": "America/Phoenix",
            "tucson, az": "America/Phoenix",
            "seattle, wa": "America/Los_Angeles",
            "portland, or": "America/Los_Angeles",
            "las vegas, nv": "America/Los_Angeles",
            "miami, fl": "America/New_York",
            "orlando, fl": "America/New_York",
            "tampa, fl": "America/New_York",
            "jacksonville, fl": "America/New_York",
            "atlanta, ga": "America/New_York",
            "savannah, ga": "America/New_York",
            "boston, ma": "America/New_York",
            "cambridge, ma": "America/New_York",
            "washington, dc": "America/New_York",
            "philadelphia, pa": "America/New_York",
            "pittsburgh, pa": "America/New_York",
            "baltimore, md": "America/New_York",
            "detroit, mi": "America/New_York",
            "cleveland, oh": "America/New_York",
            "columbus, oh": "America/New_York",
            "cincinnati, oh": "America/New_York",
            "indianapolis, in": "America/New_York",
            "milwaukee, wi": "America/Chicago",
            "minneapolis, mn": "America/Chicago",
            "st. paul, mn": "America/Chicago",
            "kansas city, mo": "America/Chicago",
            "st. louis, mo": "America/Chicago",
            "nashville, tn": "America/Chicago",
            "memphis, tn": "America/Chicago",
            "new orleans, la": "America/Chicago",
            "oklahoma city, ok": "America/Chicago",
            "tulsa, ok": "America/Chicago",
            "little rock, ar": "America/Chicago",
            "birmingham, al": "America/Chicago",
            "jackson, ms": "America/Chicago",
            "charlotte, nc": "America/New_York",
            "raleigh, nc": "America/New_York",
            "charleston, sc": "America/New_York",
            "columbia, sc": "America/New_York",
            "richmond, va": "America/New_York",
            "norfolk, va": "America/New_York",
            "louisville, ky": "America/New_York",
            "lexington, ky": "America/New_York"
        }
        return self.config.get("city_to_timezone_mapping", default_mapping)