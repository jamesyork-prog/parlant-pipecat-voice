"""
Audit logging integration for MOR and Product Type gates.

This module provides comprehensive audit logging that meets all requirements
from Requirement 12: audit logging for MOR/product type decisions.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .models import PreArrivalAuditLog

logger = logging.getLogger(__name__)


@dataclass
class MORAuditEntry:
    """Audit log entry for MOR gate decisions."""
    ticket_id: str
    partner_value: str
    vendor_detected: str
    policy_rule: str
    decision: str  # "rejected", "passed", "passed_with_flag"
    processing_time_ms: int
    timestamp: str
    rejection_reason: Optional[str] = None
    special_flag: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ProductTypeAuditEntry:
    """Audit log entry for Product Type gate decisions."""
    ticket_id: str
    product_type: str
    inference_method: str
    booking_duration_hours: Optional[float]
    policy_rule: str
    decision: str  # "rejected", "passed"
    processing_time_ms: int
    timestamp: str
    rejection_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DataValidationWarning:
    """Audit log entry for data validation warnings."""
    ticket_id: str
    warning_type: str  # "missing_field", "invalid_format", "unknown_value"
    field_name: str
    field_value: Any
    warning_message: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return asdict(self)


class GateAuditLogger:
    """
    Audit logger for MOR and Product Type gate decisions.
    
    This logger captures all decision points, data validation issues,
    and processing metrics required for compliance verification and
    audit trail reconstruction.
    
    Integrates with existing agent_audit_log table structure.
    """
    
    def __init__(self):
        """Initialize gate audit logger."""
        self.logger = logging.getLogger(f"{__name__}.GateAuditLogger")
        self.logger.info("GateAuditLogger initialized")
    
    def log_mor_decision(
        self,
        ticket_id: str,
        partner_value: str,
        vendor_detected: str,
        policy_rule: str,
        decision: str,
        processing_time_ms: int,
        rejection_reason: Optional[str] = None,
        special_flag: Optional[str] = None
    ) -> None:
        """
        Log MOR gate decision with all required audit fields.
        
        Implements Requirement 12.1: MOR rejections log ticket ID, Partner value,
        vendor detected, policy rule, timestamp
        
        Args:
            ticket_id: Unique ticket identifier
            partner_value: Original Partner field value from ticket
            vendor_detected: Normalized vendor name detected by gate
            policy_rule: Policy rule that determined the decision
            decision: Gate decision (rejected/passed/passed_with_flag)
            processing_time_ms: Gate processing time in milliseconds
            rejection_reason: Customer-friendly rejection message (if rejected)
            special_flag: Special handling flag (e.g., groupon_credit_only)
        """
        entry = MORAuditEntry(
            ticket_id=ticket_id,
            partner_value=partner_value,
            vendor_detected=vendor_detected,
            policy_rule=policy_rule,
            decision=decision,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            rejection_reason=rejection_reason,
            special_flag=special_flag
        )
        
        # Log structured audit information
        if decision == "rejected":
            self.logger.info(
                f"MOR gate rejection: {vendor_detected}",
                extra={
                    "event_type": "mor_gate_rejection",
                    "ticket_id": ticket_id,
                    "partner_value": partner_value,
                    "vendor_detected": vendor_detected,
                    "policy_rule": policy_rule,
                    "processing_time_ms": processing_time_ms,
                    "rejection_reason": rejection_reason
                }
            )
        elif decision == "passed_with_flag":
            self.logger.info(
                f"MOR: {vendor_detected} (pass with {special_flag})",
                extra={
                    "event_type": "mor_gate_pass_with_flag",
                    "ticket_id": ticket_id,
                    "partner_value": partner_value,
                    "vendor_detected": vendor_detected,
                    "policy_rule": policy_rule,
                    "processing_time_ms": processing_time_ms,
                    "special_flag": special_flag
                }
            )
        else:  # passed
            self.logger.info(
                f"MOR: {partner_value} (pass)",
                extra={
                    "event_type": "mor_gate_pass",
                    "ticket_id": ticket_id,
                    "partner_value": partner_value,
                    "vendor_detected": vendor_detected,
                    "policy_rule": policy_rule,
                    "processing_time_ms": processing_time_ms
                }
            )
        
        self._log_to_database("mor_gate_decision", entry.to_dict(), ticket_id)
    
    def log_product_type_decision(
        self,
        ticket_id: str,
        product_type: str,
        inference_method: str,
        booking_duration_hours: Optional[float],
        policy_rule: str,
        decision: str,
        processing_time_ms: int,
        rejection_reason: Optional[str] = None
    ) -> None:
        """
        Log Product Type gate decision with all required audit fields.
        
        Implements Requirement 12.2: Product Type rejections log ticket ID, product type,
        inference method, duration, policy rule
        
        Args:
            ticket_id: Unique ticket identifier
            product_type: Inferred product type (event/on-demand/transient/season_package)
            inference_method: Method used to infer product type
            booking_duration_hours: Calculated booking duration (if applicable)
            policy_rule: Policy rule that determined the decision
            decision: Gate decision (rejected/passed)
            processing_time_ms: Gate processing time in milliseconds
            rejection_reason: Customer-friendly rejection message (if rejected)
        """
        entry = ProductTypeAuditEntry(
            ticket_id=ticket_id,
            product_type=product_type,
            inference_method=inference_method,
            booking_duration_hours=booking_duration_hours,
            policy_rule=policy_rule,
            decision=decision,
            processing_time_ms=processing_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            rejection_reason=rejection_reason
        )
        
        # Log structured audit information
        if decision == "rejected":
            self.logger.info(
                f"Product Type gate rejection: {product_type}",
                extra={
                    "event_type": "product_type_gate_rejection",
                    "ticket_id": ticket_id,
                    "product_type": product_type,
                    "inference_method": inference_method,
                    "booking_duration_hours": booking_duration_hours,
                    "policy_rule": policy_rule,
                    "processing_time_ms": processing_time_ms,
                    "rejection_reason": rejection_reason
                }
            )
        else:  # passed
            self.logger.info(
                f"Product Type: {product_type} (pass)",
                extra={
                    "event_type": "product_type_gate_pass",
                    "ticket_id": ticket_id,
                    "product_type": product_type,
                    "inference_method": inference_method,
                    "booking_duration_hours": booking_duration_hours,
                    "policy_rule": policy_rule,
                    "processing_time_ms": processing_time_ms
                }
            )
        
        self._log_to_database("product_type_gate_decision", entry.to_dict(), ticket_id)
    
    def log_data_warning(
        self,
        ticket_id: str,
        warning_type: str,
        field_name: str,
        field_value: Any,
        warning_message: str
    ) -> None:
        """
        Log data validation warning.
        
        Implements Requirement 12.4: Data validation warnings captured for
        missing fields, invalid formats, unknowns
        
        Args:
            ticket_id: Unique ticket identifier
            warning_type: Type of warning (missing_field/invalid_format/unknown_value)
            field_name: Name of the problematic field
            field_value: Value that caused the warning
            warning_message: Human-readable warning description
        """
        entry = DataValidationWarning(
            ticket_id=ticket_id,
            warning_type=warning_type,
            field_name=field_name,
            field_value=field_value,
            warning_message=warning_message,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Log structured warning
        self.logger.warning(
            f"Data validation warning: {warning_message}",
            extra={
                "event_type": "data_validation_warning",
                "ticket_id": ticket_id,
                "warning_type": warning_type,
                "field_name": field_name,
                "field_value": field_value
            }
        )
        
        self._log_to_database("data_validation_warning", entry.to_dict(), ticket_id)
    
    def log_pre_arrival_decision(
        self,
        audit_entry: PreArrivalAuditLog
    ) -> None:
        """
        Log pre-arrival gate decision with comprehensive audit information.
        
        Implements Requirements 12.1, 12.2, 12.3, 12.4, 12.5:
        - Pre-arrival approvals: ticket ID, booking start, current time, timezone, hours before
        - Non-pre-arrival: ticket ID, booking start, current time, time difference  
        - Timezone inference: original field, inference method, final timezone
        - Data validation warnings: field name, issue, fallback action
        - Performance metrics: parsing time, timezone time, comparison time, total time
        
        Args:
            audit_entry: PreArrivalAuditLog with all decision details
        """
        ticket_id = audit_entry.ticket_id
        
        # Determine event type based on decision
        if audit_entry.is_pre_arrival:
            event_type = "pre_arrival_approval"
            hours_str = f"{audit_entry.hours_before_booking:.1f}" if audit_entry.hours_before_booking is not None else "unknown"
            log_message = (
                f"Pre-arrival APPROVED: {hours_str} hours before booking "
                f"(timezone: {audit_entry.timezone_used}, total: {audit_entry.total_time_ms}ms)"
            )
        else:
            event_type = "pre_arrival_not_applicable"
            hours_str = f"{audit_entry.hours_before_booking:.1f}" if audit_entry.hours_before_booking is not None else "unknown"
            log_message = (
                f"Pre-arrival check: NOT pre-arrival, time difference: {hours_str} hours "
                f"(timezone: {audit_entry.timezone_used}, total: {audit_entry.total_time_ms}ms)"
            )
        
        # Create structured log entry with all required fields
        log_extra = {
            "event_type": event_type,
            "ticket_id": ticket_id,
            
            # Core decision fields (Requirements 12.1, 12.2)
            "booking_start_time": audit_entry.parsed_start_time.isoformat() if audit_entry.parsed_start_time else None,
            "current_time": audit_entry.current_time.isoformat() if audit_entry.current_time else None,
            "timezone_used": audit_entry.timezone_used,
            "hours_before_booking": audit_entry.hours_before_booking,
            "is_pre_arrival": audit_entry.is_pre_arrival,
            "decision": audit_entry.decision,
            "reasoning": audit_entry.reasoning,
            
            # Timezone inference details (Requirement 12.3)
            "raw_timezone_field": audit_entry.raw_timezone,
            "raw_location_field": audit_entry.raw_location,
            "timezone_source": audit_entry.timezone_source,  # "direct", "inferred", "utc_fallback"
            
            # Data validation details (Requirement 12.4)
            "raw_start_time": audit_entry.raw_start_time,
            "data_quality_issues": audit_entry.data_quality_issues,
            "warnings": audit_entry.warnings,
            "suspicious_patterns": audit_entry.suspicious_patterns,
            
            # Performance metrics (Requirement 12.5)
            "parsing_time_ms": audit_entry.parsing_time_ms,
            "timezone_time_ms": audit_entry.timezone_time_ms,
            "comparison_time_ms": audit_entry.comparison_time_ms,
            "total_time_ms": audit_entry.total_time_ms,
            
            # Timestamp
            "timestamp": audit_entry.timestamp.isoformat()
        }
        
        # Log at appropriate level based on outcome
        if audit_entry.is_pre_arrival:
            self.logger.info(log_message, extra=log_extra)
        else:
            self.logger.debug(log_message, extra=log_extra)
        
        # Log data quality issues as warnings if present
        if audit_entry.data_quality_issues:
            for issue in audit_entry.data_quality_issues:
                self.logger.warning(
                    f"Pre-arrival data quality issue: {issue}",
                    extra={
                        "event_type": "pre_arrival_data_quality_warning",
                        "ticket_id": ticket_id,
                        "issue": issue
                    }
                )
        
        # Log suspicious patterns if detected
        if audit_entry.suspicious_patterns:
            for pattern in audit_entry.suspicious_patterns:
                self.logger.warning(
                    f"Pre-arrival suspicious pattern: {pattern}",
                    extra={
                        "event_type": "pre_arrival_suspicious_pattern",
                        "ticket_id": ticket_id,
                        "pattern": pattern
                    }
                )
        
        # Convert audit entry to dict for database logging
        audit_dict = {
            "ticket_id": ticket_id,
            "timestamp": audit_entry.timestamp.isoformat(),
            "raw_start_time": audit_entry.raw_start_time,
            "raw_timezone": audit_entry.raw_timezone,
            "raw_location": audit_entry.raw_location,
            "parsed_start_time": audit_entry.parsed_start_time.isoformat() if audit_entry.parsed_start_time else None,
            "timezone_used": audit_entry.timezone_used,
            "timezone_source": audit_entry.timezone_source,
            "current_time": audit_entry.current_time.isoformat() if audit_entry.current_time else None,
            "is_pre_arrival": audit_entry.is_pre_arrival,
            "hours_before_booking": audit_entry.hours_before_booking,
            "decision": audit_entry.decision,
            "reasoning": audit_entry.reasoning,
            "parsing_time_ms": audit_entry.parsing_time_ms,
            "timezone_time_ms": audit_entry.timezone_time_ms,
            "comparison_time_ms": audit_entry.comparison_time_ms,
            "total_time_ms": audit_entry.total_time_ms,
            "warnings": audit_entry.warnings,
            "data_quality_issues": audit_entry.data_quality_issues,
            "suspicious_patterns": audit_entry.suspicious_patterns
        }
        
        self._log_to_database(event_type, audit_dict, ticket_id)
    
    def log_pre_arrival_timezone_inference(
        self,
        ticket_id: str,
        original_timezone_field: Optional[str],
        original_location_field: Optional[str],
        inference_method: str,
        final_timezone: str,
        inference_success: bool
    ) -> None:
        """
        Log timezone inference details for pre-arrival processing.
        
        Implements Requirement 12.3: Timezone inference logging with original field,
        inference method, and final timezone.
        
        Args:
            ticket_id: Unique ticket identifier
            original_timezone_field: Original "Location Time Zone" field value
            original_location_field: Original "Location City, State" field value  
            inference_method: Method used ("direct", "city_mapping", "utc_fallback")
            final_timezone: Final timezone used for processing
            inference_success: Whether inference was successful
        """
        log_message = (
            f"Pre-arrival timezone inference: {inference_method} → {final_timezone} "
            f"(success: {inference_success})"
        )
        
        log_extra = {
            "event_type": "pre_arrival_timezone_inference",
            "ticket_id": ticket_id,
            "original_timezone_field": original_timezone_field,
            "original_location_field": original_location_field,
            "inference_method": inference_method,
            "final_timezone": final_timezone,
            "inference_success": inference_success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if inference_success:
            self.logger.info(log_message, extra=log_extra)
        else:
            self.logger.warning(log_message, extra=log_extra)
        
        self._log_to_database("pre_arrival_timezone_inference", log_extra, ticket_id)
    
    def log_pre_arrival_performance_metrics(
        self,
        ticket_id: str,
        parsing_time_ms: int,
        timezone_time_ms: int,
        comparison_time_ms: int,
        total_time_ms: int,
        target_ms: int = 200
    ) -> None:
        """
        Log performance metrics for pre-arrival processing.
        
        Implements Requirement 12.5: Performance metrics logging with component timing.
        
        Args:
            ticket_id: Unique ticket identifier
            parsing_time_ms: Date parsing time in milliseconds
            timezone_time_ms: Timezone handling time in milliseconds
            comparison_time_ms: Time comparison time in milliseconds
            total_time_ms: Total processing time in milliseconds
            target_ms: Target processing time for performance evaluation
        """
        performance_status = "within_target" if total_time_ms <= target_ms else "exceeds_target"
        
        log_message = (
            f"Pre-arrival performance: {total_time_ms}ms total "
            f"(parse: {parsing_time_ms}ms, tz: {timezone_time_ms}ms, compare: {comparison_time_ms}ms) "
            f"- {performance_status}"
        )
        
        log_extra = {
            "event_type": "pre_arrival_performance_metrics",
            "ticket_id": ticket_id,
            "parsing_time_ms": parsing_time_ms,
            "timezone_time_ms": timezone_time_ms,
            "comparison_time_ms": comparison_time_ms,
            "total_time_ms": total_time_ms,
            "target_ms": target_ms,
            "performance_status": performance_status,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Log as warning if significantly exceeding target
        if total_time_ms > target_ms * 1.5:
            self.logger.warning(log_message, extra=log_extra)
        else:
            self.logger.info(log_message, extra=log_extra)
        
        self._log_to_database("pre_arrival_performance_metrics", log_extra, ticket_id)
    
    def _log_to_database(self, event_type: str, event_details: Dict[str, Any], ticket_id: str) -> None:
        """
        Log audit entry to database.
        
        Integrates with existing agent_audit_log table structure.
        In production, this would execute:
        
        INSERT INTO agent_audit_log (run_id, event_type, event_details, status)
        VALUES (f'{event_type}_{ticket_id}', event_type, event_details, 'completed')
        
        Args:
            event_type: Type of audit event
            event_details: Detailed event data
            ticket_id: Ticket identifier for run_id generation
        """
        try:
            # TODO: Replace with actual database integration
            # For now, just log the database operation that would occur
            run_id = f"{event_type}_{ticket_id}"
            
            self.logger.debug(
                f"Database audit log entry",
                extra={
                    "run_id": run_id,
                    "event_type": event_type,
                    "event_details_json": json.dumps(event_details, default=str),
                    "status": "completed"
                }
            )
            
            # In production, this would be:
            # cursor.execute("""
            #     INSERT INTO agent_audit_log (run_id, event_type, event_details, status)
            #     VALUES (%s, %s, %s, %s)
            # """, (run_id, event_type, json.dumps(event_details, default=str), "completed"))
            
        except Exception as e:
            self.logger.error(f"Failed to log audit entry to database: {e}")
            # Don't raise - audit logging failure shouldn't break gate processing


# Global audit logger instance
_gate_audit_logger: Optional[GateAuditLogger] = None


def get_gate_audit_logger() -> GateAuditLogger:
    """
    Get the global gate audit logger instance.
    
    Returns:
        The global GateAuditLogger instance
    """
    global _gate_audit_logger
    if _gate_audit_logger is None:
        _gate_audit_logger = GateAuditLogger()
    return _gate_audit_logger