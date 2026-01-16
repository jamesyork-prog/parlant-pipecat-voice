"""
Flexible webhook payload parser for pre-arrival gate.

This module provides robust payload parsing that handles:
- Nested webhook payloads (e.g., "freshdesk_webhook" wrapper)
- Multiple field name variations for booking data
- Backward compatibility with previous payload formats
- Graceful handling of missing/extra fields

The parser normalizes diverse payload formats into a consistent structure
for downstream processing by the pre-arrival gate.
"""

import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)


class PayloadParser:
    """
    Flexible webhook payload parser with support for multiple formats.
    
    Handles:
    - Nested payload structures (e.g., "freshdesk_webhook" wrapper)
    - Multiple field name variations
    - Backward compatibility
    - Missing/extra field handling
    """
    
    # Field name variations for booking start time
    START_TIME_FIELD_VARIATIONS = [
        "Parking Pass Start Time",
        "parking_pass_start_time",
        "start_time",
        "booking_start_time",
        "reservation_start_time",
        "pass_start_time",
    ]
    
    # Field name variations for timezone
    TIMEZONE_FIELD_VARIATIONS = [
        "Location Time Zone",
        "location_time_zone",
        "timezone",
        "time_zone",
        "tz",
        "location_timezone",
    ]
    
    # Field name variations for location
    LOCATION_FIELD_VARIATIONS = [
        "Location City, State",
        "location_city_state",
        "location",
        "city_state",
        "venue_location",
        "parking_location",
    ]
    
    # Field name variations for ticket ID
    TICKET_ID_FIELD_VARIATIONS = [
        "ticket_id",
        "id",
        "ticket_number",
        "freshdesk_ticket_id",
        "fd_ticket_id",
    ]
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize payload parser with optional configuration.
        
        Args:
            config: Optional configuration for custom field mappings
        """
        self.config = config or {}
        
        # Allow custom field variations from config
        self.start_time_fields = self.config.get(
            "start_time_field_variations",
            self.START_TIME_FIELD_VARIATIONS
        )
        self.timezone_fields = self.config.get(
            "timezone_field_variations",
            self.TIMEZONE_FIELD_VARIATIONS
        )
        self.location_fields = self.config.get(
            "location_field_variations",
            self.LOCATION_FIELD_VARIATIONS
        )
        self.ticket_id_fields = self.config.get(
            "ticket_id_field_variations",
            self.TICKET_ID_FIELD_VARIATIONS
        )
        
        # Track payload format variations for monitoring
        self.format_stats = {
            "nested_payloads": 0,
            "flat_payloads": 0,
            "field_variations_used": {},
            "missing_fields": {},
            "extra_fields_ignored": 0,
        }
    
    def parse_payload(self, raw_payload: Dict) -> Dict:
        """
        Parse webhook payload into normalized format.
        
        Handles nested payloads, field name variations, and missing fields.
        
        Args:
            raw_payload: Raw webhook payload from Freshdesk/Zapier
            
        Returns:
            Normalized payload dictionary with standard field names
        """
        logger.debug(f"Parsing webhook payload with keys: {list(raw_payload.keys())}")
        
        # Step 1: Extract inner payload if nested
        payload = self._extract_inner_payload(raw_payload)
        
        # Step 2: Extract fields with variation support
        normalized = {}
        
        # Extract ticket ID
        ticket_id = self._extract_field_with_variations(
            payload,
            self.ticket_id_fields,
            "ticket_id"
        )
        if ticket_id:
            normalized["ticket_id"] = ticket_id
        else:
            logger.warning("Ticket ID not found in payload")
            self.format_stats["missing_fields"]["ticket_id"] = \
                self.format_stats["missing_fields"].get("ticket_id", 0) + 1
        
        # Extract booking start time
        start_time = self._extract_field_with_variations(
            payload,
            self.start_time_fields,
            "Parking Pass Start Time"
        )
        if start_time:
            normalized["Parking Pass Start Time"] = start_time
        else:
            logger.warning("Booking start time not found in payload")
            self.format_stats["missing_fields"]["start_time"] = \
                self.format_stats["missing_fields"].get("start_time", 0) + 1
        
        # Extract timezone
        timezone = self._extract_field_with_variations(
            payload,
            self.timezone_fields,
            "Location Time Zone"
        )
        if timezone:
            normalized["Location Time Zone"] = timezone
        else:
            logger.debug("Timezone not found in payload (will attempt inference)")
            self.format_stats["missing_fields"]["timezone"] = \
                self.format_stats["missing_fields"].get("timezone", 0) + 1
        
        # Extract location
        location = self._extract_field_with_variations(
            payload,
            self.location_fields,
            "Location City, State"
        )
        if location:
            normalized["Location City, State"] = location
        else:
            logger.debug("Location not found in payload (will use UTC fallback if needed)")
            self.format_stats["missing_fields"]["location"] = \
                self.format_stats["missing_fields"].get("location", 0) + 1
        
        # Step 3: Copy any additional fields from original payload
        # This maintains backward compatibility with existing code
        for key, value in payload.items():
            if key not in normalized:
                normalized[key] = value
                # Track extra fields for monitoring (but don't log each one)
                self.format_stats["extra_fields_ignored"] += 1
        
        logger.info(
            f"Payload parsed successfully: "
            f"nested={self.format_stats['nested_payloads'] > self.format_stats['flat_payloads']}, "
            f"fields_found={len(normalized)}"
        )
        
        return normalized
    
    def _extract_inner_payload(self, raw_payload: Dict) -> Dict:
        """
        Extract inner payload from nested structure.
        
        Handles common nesting patterns:
        - {"freshdesk_webhook": {...actual data...}}
        - {"webhook": {...actual data...}}
        - {"data": {...actual data...}}
        - {"payload": {...actual data...}}
        
        Args:
            raw_payload: Raw webhook payload
            
        Returns:
            Inner payload or original if not nested
        """
        # Check for common wrapper keys
        wrapper_keys = ["freshdesk_webhook", "webhook", "data", "payload", "ticket"]
        
        for wrapper_key in wrapper_keys:
            if wrapper_key in raw_payload:
                inner = raw_payload[wrapper_key]
                if isinstance(inner, dict):
                    logger.info(f"Extracted nested payload from '{wrapper_key}' wrapper")
                    self.format_stats["nested_payloads"] += 1
                    return inner
        
        # No nesting detected - return original
        logger.debug("No nesting detected in payload")
        self.format_stats["flat_payloads"] += 1
        return raw_payload
    
    def _extract_field_with_variations(
        self,
        payload: Dict,
        field_variations: List[str],
        canonical_name: str
    ) -> Optional[Any]:
        """
        Extract field value trying multiple name variations.
        
        Args:
            payload: Payload dictionary
            field_variations: List of possible field names to try
            canonical_name: Canonical field name for logging
            
        Returns:
            Field value if found, None otherwise
        """
        for field_name in field_variations:
            if field_name in payload:
                value = payload[field_name]
                
                # Log if using non-standard field name
                if field_name != field_variations[0]:
                    logger.info(
                        f"Using field variation '{field_name}' for '{canonical_name}'"
                    )
                    # Track which variations are being used
                    variation_key = f"{canonical_name}:{field_name}"
                    self.format_stats["field_variations_used"][variation_key] = \
                        self.format_stats["field_variations_used"].get(variation_key, 0) + 1
                
                return value
        
        return None
    
    def get_format_stats(self) -> Dict:
        """
        Get statistics about payload format variations encountered.
        
        Returns:
            Dictionary with format statistics for monitoring
        """
        return {
            "nested_payloads": self.format_stats["nested_payloads"],
            "flat_payloads": self.format_stats["flat_payloads"],
            "field_variations_used": dict(self.format_stats["field_variations_used"]),
            "missing_fields": dict(self.format_stats["missing_fields"]),
            "extra_fields_ignored": self.format_stats["extra_fields_ignored"],
            "total_payloads_processed": (
                self.format_stats["nested_payloads"] + 
                self.format_stats["flat_payloads"]
            ),
        }
    
    def reset_stats(self) -> None:
        """Reset format statistics."""
        self.format_stats = {
            "nested_payloads": 0,
            "flat_payloads": 0,
            "field_variations_used": {},
            "missing_fields": {},
            "extra_fields_ignored": 0,
        }
    
    def validate_critical_fields(self, normalized_payload: Dict) -> tuple[bool, List[str]]:
        """
        Validate that critical fields are present in normalized payload.
        
        Critical fields:
        - Parking Pass Start Time (required for pre-arrival check)
        
        Optional fields (can use fallbacks):
        - Location Time Zone (can infer or use UTC)
        - Location City, State (can use UTC fallback)
        - ticket_id (can use "unknown")
        
        Args:
            normalized_payload: Normalized payload from parse_payload()
            
        Returns:
            Tuple of (is_valid, list_of_missing_critical_fields)
        """
        missing_critical = []
        
        # Check for critical field: booking start time
        if "Parking Pass Start Time" not in normalized_payload:
            missing_critical.append("Parking Pass Start Time")
        elif not normalized_payload["Parking Pass Start Time"]:
            # Field exists but is empty
            missing_critical.append("Parking Pass Start Time (empty)")
        
        is_valid = len(missing_critical) == 0
        
        if not is_valid:
            logger.error(
                f"Payload validation failed: missing critical fields: {missing_critical}"
            )
        
        return is_valid, missing_critical
    
    def get_data_quality_report(self, normalized_payload: Dict) -> Dict:
        """
        Generate data quality report for normalized payload.
        
        Reports on:
        - Critical fields present/missing
        - Optional fields present/missing
        - Data completeness percentage
        - Recommendations for improving data quality
        
        Args:
            normalized_payload: Normalized payload from parse_payload()
            
        Returns:
            Dictionary with data quality metrics and recommendations
        """
        report = {
            "critical_fields": {},
            "optional_fields": {},
            "completeness_percent": 0.0,
            "issues": [],
            "recommendations": [],
        }
        
        # Check critical fields
        has_start_time = bool(
            "Parking Pass Start Time" in normalized_payload and 
            normalized_payload["Parking Pass Start Time"]
        )
        report["critical_fields"]["Parking Pass Start Time"] = has_start_time
        
        if not has_start_time:
            report["issues"].append("Missing critical field: Parking Pass Start Time")
            report["recommendations"].append(
                "Ensure webhook includes booking start time field"
            )
        
        # Check optional fields
        has_timezone = bool(
            "Location Time Zone" in normalized_payload and 
            normalized_payload["Location Time Zone"]
        )
        report["optional_fields"]["Location Time Zone"] = has_timezone
        
        if not has_timezone:
            report["issues"].append("Missing optional field: Location Time Zone")
            report["recommendations"].append(
                "Include timezone field to avoid UTC fallback and improve accuracy"
            )
        
        has_location = bool(
            "Location City, State" in normalized_payload and 
            normalized_payload["Location City, State"]
        )
        report["optional_fields"]["Location City, State"] = has_location
        
        if not has_location:
            report["issues"].append("Missing optional field: Location City, State")
            report["recommendations"].append(
                "Include location field to enable timezone inference"
            )
        
        has_ticket_id = bool(
            "ticket_id" in normalized_payload and 
            normalized_payload["ticket_id"]
        )
        report["optional_fields"]["ticket_id"] = has_ticket_id
        
        if not has_ticket_id:
            report["issues"].append("Missing optional field: ticket_id")
            report["recommendations"].append(
                "Include ticket ID for better audit logging"
            )
        
        # Calculate completeness
        total_fields = 4  # 1 critical + 3 optional
        present_fields = sum([
            has_start_time,
            has_timezone,
            has_location,
            has_ticket_id
        ])
        report["completeness_percent"] = (present_fields / total_fields) * 100
        
        return report
    
    def can_process_with_partial_data(self, normalized_payload: Dict) -> tuple[bool, str]:
        """
        Determine if pre-arrival processing can continue with partial data.
        
        Processing can continue if:
        - Critical field (Parking Pass Start Time) is present
        - Optional fields can use fallbacks (timezone inference, UTC fallback)
        
        Args:
            normalized_payload: Normalized payload from parse_payload()
            
        Returns:
            Tuple of (can_process, reason_or_warning)
        """
        # Check critical field
        has_start_time = bool(
            "Parking Pass Start Time" in normalized_payload and 
            normalized_payload["Parking Pass Start Time"]
        )
        
        if not has_start_time:
            return False, "Cannot process: missing critical field 'Parking Pass Start Time'"
        
        # Check optional fields and build warning message
        warnings = []
        
        has_timezone = bool(
            "Location Time Zone" in normalized_payload and 
            normalized_payload["Location Time Zone"]
        )
        has_location = bool(
            "Location City, State" in normalized_payload and 
            normalized_payload["Location City, State"]
        )
        
        if not has_timezone and not has_location:
            warnings.append("Will use UTC fallback (timezone and location missing)")
        elif not has_timezone:
            warnings.append("Will attempt timezone inference from location")
        
        if warnings:
            warning_msg = "Can process with partial data: " + "; ".join(warnings)
            logger.info(warning_msg)
            return True, warning_msg
        
        return True, "Can process with complete data"
    
    def log_payload_format(self, raw_payload: Dict) -> None:
        """
        Log payload format details for debugging and monitoring.
        
        Args:
            raw_payload: Raw webhook payload
        """
        logger.info(
            f"Webhook payload format: "
            f"top_level_keys={list(raw_payload.keys())}, "
            f"nested={'freshdesk_webhook' in raw_payload or 'webhook' in raw_payload}"
        )
        
        # Log field presence for monitoring
        payload = self._extract_inner_payload(raw_payload)
        
        has_start_time = any(field in payload for field in self.start_time_fields)
        has_timezone = any(field in payload for field in self.timezone_fields)
        has_location = any(field in payload for field in self.location_fields)
        
        logger.debug(
            f"Payload field presence: "
            f"start_time={has_start_time}, "
            f"timezone={has_timezone}, "
            f"location={has_location}"
        )
