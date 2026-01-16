"""
Product Type Gate implementation.

This gate determines if a ticket should be instantly rejected based on
the product type (on-demand, season package, etc.).
"""

import time
import logging
from datetime import datetime
from typing import Dict, Optional

from ..models import ProductTypeDecision, ProductTypeGateConfig, ProductTypeMetrics
from ..audit_logger import get_gate_audit_logger

logger = logging.getLogger(__name__)


class ProductTypeGate:
    """
    Gate 4: Product type inference and refundability check.
    
    Instantly rejects on-demand and season packages
    Passes transient and event bookings to next gate
    
    Performance: <50ms per ticket
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize product type gate with configuration.
        
        Args:
            config: Optional configuration for duration thresholds, etc.
        """
        self.config = config or {}
        self.gate_config = ProductTypeGateConfig()
        self.metrics = ProductTypeMetrics()
        self.audit_logger = get_gate_audit_logger()
        
        # Override defaults with provided config
        if "on_demand_threshold_hours" in self.config:
            self.gate_config.on_demand_threshold_hours = self.config["on_demand_threshold_hours"]
        if "season_package_detection" in self.config:
            self.gate_config.season_package_detection = self.config["season_package_detection"]
        
        # Override default LLM cost if provided
        if "average_llm_cost_per_ticket" in self.config:
            self.metrics.average_llm_cost_per_ticket = self.config["average_llm_cost_per_ticket"]
    
    def check_product_type(self, ticket_data: Dict) -> ProductTypeDecision:
        """
        Infer product type from available ticket fields and check refundability.
        
        Inference priority:
        1. Event Name/Venue Name → Event (refundable based on other criteria)
        2. Booking duration <4 hours + no event data → On-Demand (REJECT)
        3. Season package indicators → Season Package (REJECT unless duplicate)
        4. Default → Transient (refundable based on other criteria)
        
        Args:
            ticket_data: Zapier webhook payload
            
        Returns:
            ProductTypeDecision with routing instruction
            
        Processing time: <50ms
        """
        start = time.perf_counter()
        
        # Check for event booking indicators
        event_name = ticket_data.get("Event Name", "") or ""
        venue_name = ticket_data.get("Venue Name", "") or ""
        
        if event_name.strip() or venue_name.strip():
            processing_time_ms = int((time.perf_counter() - start) * 1000)
            
            # Log Product Type pass-through decision
            self.audit_logger.log_product_type_decision(
                ticket_id=ticket_data.get("id", "unknown"),
                product_type="event",
                inference_method="event_name_venue_populated",
                booking_duration_hours=None,
                policy_rule="event_booking_pass_through",
                decision="passed",
                processing_time_ms=processing_time_ms
            )
            
            self.metrics.add_pass_through("event", processing_time_ms)
            return ProductTypeDecision(
                should_continue=True,
                product_type="event",
                inference_method="event_name_venue_populated",
                processing_time_ms=processing_time_ms
            )
        
        # Check for on-demand based on booking duration
        duration_calculation_failed = False
        duration_hours = None
        try:
            duration_hours = self._calculate_booking_duration(ticket_data)
            
            if duration_hours is not None and duration_hours < self.gate_config.on_demand_threshold_hours:
                processing_time_ms = int((time.perf_counter() - start) * 1000)
                rejection_reason = self._format_on_demand_rejection()
                inference_method = f"booking_duration_{duration_hours:.1f}h_below_threshold"
                
                # Log Product Type rejection decision
                self.audit_logger.log_product_type_decision(
                    ticket_id=ticket_data.get("id", "unknown"),
                    product_type="on-demand",
                    inference_method=inference_method,
                    booking_duration_hours=duration_hours,
                    policy_rule="on_demand_instant_rejection",
                    decision="rejected",
                    processing_time_ms=processing_time_ms,
                    rejection_reason=rejection_reason
                )
                
                self.metrics.add_rejection("on-demand", processing_time_ms)
                return ProductTypeDecision(
                    should_continue=False,
                    product_type="on-demand",
                    inference_method=inference_method,
                    rejection_reason=rejection_reason,
                    processing_time_ms=processing_time_ms
                )
            elif duration_hours is None:
                # Duration calculation failed
                duration_calculation_failed = True
        except Exception as e:
            logger.warning(f"Duration calculation failed: {e}")
            duration_calculation_failed = True
        
        # Track duration calculation failure if it occurred
        if duration_calculation_failed:
            self.metrics.add_duration_failure()
            
            # Log data validation warning for duration calculation failure
            self.audit_logger.log_data_warning(
                ticket_id=ticket_data.get("id", "unknown"),
                warning_type="calculation_error",
                field_name="Parking Pass Duration",
                field_value="calculation_failed",
                warning_message="Duration calculation failed - defaulting to transient"
            )
        
        # Check for season package (TBD: need to understand API fields for detection)
        # For now, this is a placeholder - will implement once API fields are known
        # if self._is_season_package(ticket_data):
        #     if self._has_duplicate_indicators(ticket_data):
        #         processing_time_ms = int((time.perf_counter() - start) * 1000)
        #         self.metrics.add_pass_through("season_package_duplicate", processing_time_ms)
        #         return ProductTypeDecision(
        #             should_continue=True,
        #             product_type="season_package_duplicate",
        #             inference_method="season_package_with_duplicate_claim",
        #             processing_time_ms=processing_time_ms
        #         )
        #     else:
        #         processing_time_ms = int((time.perf_counter() - start) * 1000)
        #         self.metrics.add_rejection("season_package", processing_time_ms)
        #         return ProductTypeDecision(
        #             should_continue=False,
        #             product_type="season_package",
        #             inference_method="season_package_detected",
        #             rejection_reason=self._format_season_package_rejection(),
        #             processing_time_ms=processing_time_ms
        #         )
        
        # Default to transient (safest assumption)
        processing_time_ms = int((time.perf_counter() - start) * 1000)
        
        # Log Product Type default decision
        self.audit_logger.log_product_type_decision(
            ticket_id=ticket_data.get("id", "unknown"),
            product_type="transient",
            inference_method="default_transient",
            booking_duration_hours=None,
            policy_rule="default_transient_assumption",
            decision="passed",
            processing_time_ms=processing_time_ms
        )
        
        self.metrics.add_pass_through("transient", processing_time_ms)
        return ProductTypeDecision(
            should_continue=True,
            product_type="transient",
            inference_method="default_transient",
            processing_time_ms=processing_time_ms
        )
    
    def get_metrics(self) -> ProductTypeMetrics:
        """Get current metrics for this gate."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics to initial state."""
        self.metrics = ProductTypeMetrics()
        # Preserve LLM cost setting
        if "average_llm_cost_per_ticket" in self.config:
            self.metrics.average_llm_cost_per_ticket = self.config["average_llm_cost_per_ticket"]
    
    def _calculate_booking_duration(self, ticket_data: Dict) -> Optional[float]:
        """
        Calculate booking duration in hours from start/end times.
        
        Handles various edge cases:
        - Missing or empty time fields
        - Invalid date formats
        - Malformed date strings
        - Boundary conditions (exactly at threshold)
        
        Args:
            ticket_data: Zapier webhook payload with timing fields
            
        Returns:
            Duration in hours, or None if calculation fails
            
        Note: Audit logging for failures is handled by the caller
        """
        try:
            # Extract time fields with robust handling
            start_str = ticket_data.get("Parking Pass Start Time")
            end_str = ticket_data.get("Parking Pass End Time")
            
            # Handle None values
            if start_str is None or end_str is None:
                logger.warning("Missing start or end time fields - defaulting to transient")
                return None
            
            # Convert to string and strip whitespace
            start_str = str(start_str).strip()
            end_str = str(end_str).strip()
            
            # Handle empty strings
            if not start_str or not end_str:
                logger.warning("Empty start or end time fields - defaulting to transient")
                return None
            
            # Parse dates with standard format: "Wednesday Dec 17, 2025, 07:45 PM"
            try:
                start = datetime.strptime(start_str, "%A %b %d, %Y, %I:%M %p")
            except ValueError as e:
                logger.warning(f"Invalid start time format '{start_str}': {e} - defaulting to transient")
                return None
            
            try:
                end = datetime.strptime(end_str, "%A %b %d, %Y, %I:%M %p")
            except ValueError as e:
                logger.warning(f"Invalid end time format '{end_str}': {e} - defaulting to transient")
                return None
            
            # Calculate duration
            duration = end - start
            duration_hours = duration.total_seconds() / 3600
            
            # Validate duration is positive
            if duration_hours <= 0:
                logger.warning(f"Invalid duration: end time {end_str} is before start time {start_str} - defaulting to transient")
                return None
            
            # Log successful calculation for debugging
            logger.debug(f"Calculated booking duration: {duration_hours:.2f} hours")
            
            return duration_hours
            
        except Exception as e:
            # Catch any unexpected errors and log them
            logger.warning(f"Unexpected error in duration calculation: {e} - defaulting to transient")
            return None
    
    def _format_on_demand_rejection(self) -> str:
        """Format customer-friendly rejection message for on-demand products."""
        return (
            "Non-refundable. On-demand parking products allow immediate booking without advance reservation. "
            "Per ParkWhiz policy, refunds are not available for on-demand bookings. "
            "For questions about this policy, please contact ParkWhiz customer support."
        )
    
    def _format_season_package_rejection(self) -> str:
        """Format customer-friendly rejection message for season packages."""
        return (
            "Non-refundable. Season packages are discounted multi-event bundles sold as a complete package. "
            "Per ParkWhiz policy, individual events within a season package cannot be refunded separately. "
            "Exception: If you accidentally purchased duplicate season packages, please indicate this in your request for manual review."
        )