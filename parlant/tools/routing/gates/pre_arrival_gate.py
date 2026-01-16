"""
Pre-arrival gate orchestrator for instant approval of pre-arrival cancellations.

This module provides the main PreArrivalGate class that coordinates date parsing,
timezone handling, and time comparison to provide <200ms pre-arrival determination.

The gate operates as Gate 5 in the routing pipeline, after MOR, flags, and product
type checks but before scenario classification. It provides instant approval for
cancellations made before the booking start time.

Performance: <200ms target, <100ms for common cases
"""

import time
import logging
from typing import Optional, Dict, List
from datetime import datetime

from .date_parser import PreArrivalDateParser
from .timezone_handler import TimezoneHandler
from .time_comparator import PreArrivalComparator
from .payload_parser import PayloadParser
from ..models import PreArrivalDecision, PreArrivalResult, PreArrivalAuditLog, PreArrivalMetrics
from ..audit_logger import get_gate_audit_logger

logger = logging.getLogger(__name__)


class PreArrivalGate:
    """
    Gate 5: Pre-arrival fast path for instant approvals.
    
    Coordinates date parsing, timezone handling, and time comparison
    to provide <200ms pre-arrival determination.
    
    Performance: <200ms target, <100ms for common cases
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize pre-arrival gate with component configuration.
        
        Args:
            config: Optional configuration for all components
        """
        self.config = config or {}
        
        # Initialize payload parser for flexible webhook handling
        self.payload_parser = PayloadParser(self.config.get("payload_parser"))
        
        # Initialize components with their respective configurations
        self.date_parser = PreArrivalDateParser(self.config.get("date_parser"))
        self.timezone_handler = TimezoneHandler(self.config.get("timezone_handler"))
        self.comparator = PreArrivalComparator(self.config.get("comparator"))
        
        # Initialize audit logger
        self.audit_logger = get_gate_audit_logger()
        
        # Performance tracking
        self.target_processing_ms = self.config.get("target_processing_ms", 200)
        self.optimized_path_ms = self.config.get("optimized_path_ms", 100)
        
        # Initialize metrics tracking
        self.metrics = PreArrivalMetrics()
        self.processing_times: List[float] = []  # Track all processing times for percentile calculation
        self.hours_before_booking_values: List[float] = []  # Track hours before booking for analytics
        self.average_llm_cost = self.config.get("average_llm_cost", 0.12)  # Default $0.12 per LLM call
    
    def check_pre_arrival(self, ticket_data: Dict, context: Optional[str] = None) -> PreArrivalDecision:
        """
        Main entry point for pre-arrival determination with comprehensive audit logging.
        
        Processing flow:
        1. Parse and normalize webhook payload (handle nested structures and field variations)
        2. Parse booking start time from normalized ticket data
        3. Apply timezone to parsed datetime
        4. Get current time in booking timezone
        5. Compare times and determine pre-arrival status
        6. Return instant approval or continue to next gate
        7. Log comprehensive audit information
        
        Args:
            ticket_data: Zapier webhook payload (may be nested or have field variations)
            context: Processing context ("webhook" or "chat") for context-aware error handling
            
        Returns:
            PreArrivalDecision with routing instruction
            
        Processing time: <200ms target
        """
        start = time.perf_counter()
        warnings = []
        data_quality_issues = []
        suspicious_patterns = []
        
        # Step 0: Parse and normalize payload to handle variations
        try:
            # Log original payload format for monitoring
            self.payload_parser.log_payload_format(ticket_data)
            
            # Normalize payload
            normalized_data = self.payload_parser.parse_payload(ticket_data)
            
            # Validate critical fields are present
            is_valid, missing_fields = self.payload_parser.validate_critical_fields(normalized_data)
            
            if not is_valid:
                data_quality_issues.append(f"Missing critical fields: {', '.join(missing_fields)}")
                logger.warning(f"Payload validation failed: {missing_fields}")
                
                # Track metrics for validation failure
                self._track_parsing_failure(
                    total_processing_ms=int((time.perf_counter() - start) * 1000),
                    context=context
                )
                
                return PreArrivalDecision(
                    should_continue=True,  # Escalate to next gate
                    processing_time_ms=int((time.perf_counter() - start) * 1000),
                    warnings=[f"Payload validation failed: missing {', '.join(missing_fields)}"]
                )
            
            # Check if we can process with partial data
            can_process, process_msg = self.payload_parser.can_process_with_partial_data(normalized_data)
            
            if not can_process:
                data_quality_issues.append(process_msg)
                logger.warning(f"Cannot process with partial data: {process_msg}")
                
                # Track metrics for validation failure
                self._track_parsing_failure(
                    total_processing_ms=int((time.perf_counter() - start) * 1000),
                    context=context
                )
                
                return PreArrivalDecision(
                    should_continue=True,  # Escalate to next gate
                    processing_time_ms=int((time.perf_counter() - start) * 1000),
                    warnings=[process_msg]
                )
            
            # Log data quality report for monitoring
            quality_report = self.payload_parser.get_data_quality_report(normalized_data)
            if quality_report["issues"]:
                logger.info(
                    f"Data quality issues detected: {quality_report['issues']}, "
                    f"completeness: {quality_report['completeness_percent']:.1f}%"
                )
                data_quality_issues.extend(quality_report["issues"])
            
            # Add partial data warning if applicable
            if "partial data" in process_msg.lower():
                warnings.append(process_msg)
            
            # Use normalized data for rest of processing
            ticket_data = normalized_data
            
        except Exception as e:
            logger.error(f"Payload parsing error: {e}", exc_info=True)
            data_quality_issues.append(f"Payload parsing error: {str(e)}")
            
            # Track metrics for parsing failure
            self._track_parsing_failure(
                total_processing_ms=int((time.perf_counter() - start) * 1000),
                context=context
            )
            
            return PreArrivalDecision(
                should_continue=True,  # Escalate to next gate on payload error
                processing_time_ms=int((time.perf_counter() - start) * 1000),
                warnings=[f"Payload parsing failed: {str(e)}"]
            )
        
        # Extract ticket ID for audit logging
        ticket_id = ticket_data.get("ticket_id", "unknown")
        
        # Initialize audit log entry
        audit_entry = PreArrivalAuditLog(
            ticket_id=ticket_id,
            timestamp=datetime.utcnow(),
            raw_start_time=ticket_data.get("Parking Pass Start Time", ""),
            raw_timezone=ticket_data.get("Location Time Zone", ""),
            raw_location=ticket_data.get("Location City, State", "")
        )
        
        try:
            logger.info(f"Starting pre-arrival check for ticket {ticket_id} with context: {context or 'default'}")
            
            # Step 1: Parse booking start time
            parse_start = time.perf_counter()
            booking_start_naive = self.date_parser.parse_booking_start_time(ticket_data)
            parse_time_ms = int((time.perf_counter() - parse_start) * 1000)
            audit_entry.parsing_time_ms = parse_time_ms
            
            if booking_start_naive is None:
                data_quality_issues.append(f"Date parsing failed for: '{audit_entry.raw_start_time}'")
                
                # Context-specific error handling for date parsing failure
                if context == "webhook":
                    # Webhook: Log and escalate to next gate (graceful degradation)
                    warning_msg = "Date parsing failed - escalating to next gate"
                    warnings.append(warning_msg)
                    logger.warning(f"Webhook pre-arrival: {warning_msg} for ticket {ticket_id}")
                elif context == "chat":
                    # Chat: Inform user and offer manual entry option
                    warning_msg = "Date parsing failed - user interaction required"
                    warnings.append(warning_msg)
                    warnings.append("User should be prompted to manually enter booking start time")
                    logger.info(f"Chat pre-arrival: {warning_msg} for ticket {ticket_id}")
                else:
                    # Default: Log and escalate (safe fallback)
                    warning_msg = "Date parsing failed - escalating to next gate"
                    warnings.append(warning_msg)
                    logger.warning(f"Pre-arrival gate: {warning_msg} for ticket {ticket_id}")
                
                # Track metrics for parsing failure with context
                self._track_parsing_failure(
                    total_processing_ms=int((time.perf_counter() - start) * 1000),
                    context=context
                )
                
                # Complete audit entry for parsing failure
                audit_entry.data_quality_issues = data_quality_issues
                audit_entry.warnings = warnings
                audit_entry.total_time_ms = int((time.perf_counter() - start) * 1000)
                
                # Log audit information for parsing failure with context
                self.audit_logger.log_pre_arrival_decision(audit_entry)
                
                logger.warning(
                    f"Pre-arrival gate: {warning_msg} (processing time: {audit_entry.total_time_ms}ms, "
                    f"context: {context or 'default'})"
                )
                
                return PreArrivalDecision(
                    should_continue=True,  # Escalate to next gate
                    processing_time_ms=audit_entry.total_time_ms,
                    warnings=warnings
                )
            
            audit_entry.parsed_start_time = booking_start_naive
            
            # Step 2: Apply timezone
            tz_start = time.perf_counter()
            booking_start_tz = self.timezone_handler.apply_timezone_to_datetime(
                booking_start_naive, ticket_data
            )
            tz_time_ms = int((time.perf_counter() - tz_start) * 1000)
            audit_entry.timezone_time_ms = tz_time_ms
            
            if booking_start_tz is None:
                data_quality_issues.append("Timezone handling failed - unable to determine timezone")
                
                # Context-specific error handling for timezone failure
                if context == "webhook":
                    # Webhook: Use UTC fallback and continue (graceful degradation)
                    warning_msg = "Timezone handling failed - using UTC fallback"
                    warnings.append(warning_msg)
                    logger.warning(f"Webhook pre-arrival: {warning_msg} for ticket {ticket_id}, attempting UTC fallback")
                    
                    # Attempt UTC fallback for webhook context
                    import pytz
                    try:
                        booking_start_tz = pytz.UTC.localize(booking_start_naive)
                        audit_entry.timezone_used = "UTC"
                        audit_entry.timezone_source = "utc_fallback"
                        warnings.append("Using UTC timezone as fallback - times may be inaccurate")
                        logger.info(f"Webhook pre-arrival: UTC fallback successful for ticket {ticket_id}")
                        # Continue processing with UTC fallback
                    except Exception as utc_error:
                        # UTC fallback failed - escalate
                        warning_msg = "Timezone handling and UTC fallback failed - escalating to next gate"
                        warnings.append(warning_msg)
                        logger.error(f"Webhook pre-arrival: UTC fallback failed for ticket {ticket_id}: {utc_error}")
                        
                        # Track metrics for timezone failure with context
                        self._track_timezone_failure(
                            total_processing_ms=int((time.perf_counter() - start) * 1000),
                            context=context
                        )
                        
                        # Complete audit entry for timezone failure
                        audit_entry.data_quality_issues = data_quality_issues
                        audit_entry.warnings = warnings
                        audit_entry.total_time_ms = int((time.perf_counter() - start) * 1000)
                        
                        # Log audit information for timezone failure
                        self.audit_logger.log_pre_arrival_decision(audit_entry)
                        
                        return PreArrivalDecision(
                            should_continue=True,  # Escalate to next gate
                            processing_time_ms=audit_entry.total_time_ms,
                            warnings=warnings
                        )
                elif context == "chat":
                    # Chat: Ask user for timezone confirmation
                    warning_msg = "Timezone handling failed - user confirmation required"
                    warnings.append(warning_msg)
                    warnings.append("User should be asked to confirm timezone or provide location details")
                    logger.info(f"Chat pre-arrival: {warning_msg} for ticket {ticket_id}")
                    
                    # Track metrics for timezone failure with context
                    self._track_timezone_failure(
                        total_processing_ms=int((time.perf_counter() - start) * 1000),
                        context=context
                    )
                    
                    # Complete audit entry for timezone failure
                    audit_entry.data_quality_issues = data_quality_issues
                    audit_entry.warnings = warnings
                    audit_entry.total_time_ms = int((time.perf_counter() - start) * 1000)
                    
                    # Log audit information for timezone failure
                    self.audit_logger.log_pre_arrival_decision(audit_entry)
                    
                    return PreArrivalDecision(
                        should_continue=True,  # Escalate to next gate (user interaction needed)
                        processing_time_ms=audit_entry.total_time_ms,
                        warnings=warnings
                    )
                else:
                    # Default: Escalate to next gate (safe fallback)
                    warning_msg = "Timezone handling failed - escalating to next gate"
                    warnings.append(warning_msg)
                    logger.warning(f"Pre-arrival gate: {warning_msg} for ticket {ticket_id}")
                    
                    # Track metrics for timezone failure with context
                    self._track_timezone_failure(
                        total_processing_ms=int((time.perf_counter() - start) * 1000),
                        context=context
                    )
                    
                    # Complete audit entry for timezone failure
                    audit_entry.data_quality_issues = data_quality_issues
                    audit_entry.warnings = warnings
                    audit_entry.total_time_ms = int((time.perf_counter() - start) * 1000)
                    
                    # Log audit information for timezone failure
                    self.audit_logger.log_pre_arrival_decision(audit_entry)
                    
                    return PreArrivalDecision(
                        should_continue=True,  # Escalate to next gate
                        processing_time_ms=audit_entry.total_time_ms,
                        warnings=warnings
                    )
            
            # Only continue if we have a valid timezone (either from handler or UTC fallback)
            if booking_start_tz is None:
                # This should not happen after the above logic, but safety check
                warning_msg = "Timezone handling failed after all fallback attempts"
                warnings.append(warning_msg)
                
                # Track metrics for timezone failure
                self._track_timezone_failure(
                    total_processing_ms=int((time.perf_counter() - start) * 1000),
                    context=context
                )
                
                # Complete audit entry
                audit_entry.data_quality_issues = data_quality_issues
                audit_entry.warnings = warnings
                audit_entry.total_time_ms = int((time.perf_counter() - start) * 1000)
                self.audit_logger.log_pre_arrival_decision(audit_entry)
                
                logger.warning(f"Pre-arrival gate: {warning_msg} (processing time: {audit_entry.total_time_ms}ms)")
                
                return PreArrivalDecision(
                    should_continue=True,
                    processing_time_ms=audit_entry.total_time_ms,
                    warnings=warnings
                )
            
            # Determine timezone source for audit logging
            timezone_source = self._determine_timezone_source(ticket_data, booking_start_tz)
            audit_entry.timezone_used = booking_start_tz.tzinfo.zone if hasattr(booking_start_tz.tzinfo, 'zone') else str(booking_start_tz.tzinfo)
            audit_entry.timezone_source = timezone_source
            
            # Log timezone inference details
            self.audit_logger.log_pre_arrival_timezone_inference(
                ticket_id=ticket_id,
                original_timezone_field=audit_entry.raw_timezone,
                original_location_field=audit_entry.raw_location,
                inference_method=timezone_source,
                final_timezone=audit_entry.timezone_used,
                inference_success=True
            )
            
            # Step 3: Get current time in booking timezone
            current_time_tz = self.timezone_handler.get_current_time_in_timezone(
                booking_start_tz.tzinfo
            )
            audit_entry.current_time = current_time_tz
            
            # Step 4: Compare times
            compare_start = time.perf_counter()
            comparison_result = self.comparator.compare_times(booking_start_tz, current_time_tz)
            compare_time_ms = int((time.perf_counter() - compare_start) * 1000)
            audit_entry.comparison_time_ms = compare_time_ms
            
            # Check for suspicious patterns
            suspicious_patterns.extend(self._check_suspicious_patterns(comparison_result))
            
            # Step 5: Generate decision and complete audit entry
            total_processing_ms = int((time.perf_counter() - start) * 1000)
            audit_entry.total_time_ms = total_processing_ms
            audit_entry.is_pre_arrival = comparison_result.is_pre_arrival
            audit_entry.hours_before_booking = comparison_result.hours_before_booking
            audit_entry.warnings = warnings
            audit_entry.data_quality_issues = data_quality_issues
            audit_entry.suspicious_patterns = suspicious_patterns
            
            # Log performance metrics
            self.audit_logger.log_pre_arrival_performance_metrics(
                ticket_id=ticket_id,
                parsing_time_ms=parse_time_ms,
                timezone_time_ms=tz_time_ms,
                comparison_time_ms=compare_time_ms,
                total_time_ms=total_processing_ms,
                target_ms=self.target_processing_ms
            )
            
            if comparison_result.is_pre_arrival:
                # Instant approval
                approval_reasoning = self._format_pre_arrival_approval(comparison_result)
                audit_entry.decision = "Approved"
                audit_entry.reasoning = approval_reasoning
                
                # Track metrics for pre-arrival approval with context
                self._track_pre_arrival_approval(
                    total_processing_ms=total_processing_ms,
                    hours_before_booking=comparison_result.hours_before_booking,
                    timezone_source=timezone_source,
                    format_used=self.date_parser.last_successful_format,
                    context=context
                )
                
                # Log comprehensive audit information for approval
                self.audit_logger.log_pre_arrival_decision(audit_entry)
                
                logger.info(
                    f"Pre-arrival APPROVED for ticket {ticket_id}: {comparison_result.hours_before_booking:.1f} hours before booking "
                    f"(processing time: {total_processing_ms}ms, timezone: {audit_entry.timezone_used})"
                )
                
                return PreArrivalDecision(
                    should_continue=False,  # Stop processing - approved
                    decision="Approved",
                    reasoning=approval_reasoning,
                    hours_before_booking=comparison_result.hours_before_booking,
                    timezone_used=audit_entry.timezone_used,
                    processing_time_ms=total_processing_ms,
                    warnings=warnings
                )
            else:
                # Not pre-arrival - continue to next gate
                audit_entry.decision = None  # No decision made, continuing
                audit_entry.reasoning = "Not pre-arrival - continuing to next gate"
                
                # Track metrics for not pre-arrival with context
                self._track_not_pre_arrival(
                    total_processing_ms=total_processing_ms,
                    timezone_source=timezone_source,
                    format_used=self.date_parser.last_successful_format,
                    context=context
                )
                
                # Log comprehensive audit information for non-pre-arrival
                self.audit_logger.log_pre_arrival_decision(audit_entry)
                
                logger.info(
                    f"Pre-arrival check for ticket {ticket_id}: NOT pre-arrival, continuing to next gate "
                    f"(processing time: {total_processing_ms}ms, timezone: {audit_entry.timezone_used})"
                )
                
                return PreArrivalDecision(
                    should_continue=True,
                    processing_time_ms=total_processing_ms,
                    warnings=warnings
                )
        
        except Exception as e:
            processing_ms = int((time.perf_counter() - start) * 1000)
            error_msg = f"Pre-arrival gate error: {str(e)}"
            warnings.append(error_msg)
            data_quality_issues.append(f"Processing exception: {str(e)}")
            
            # Complete audit entry for exception
            audit_entry.warnings = warnings
            audit_entry.data_quality_issues = data_quality_issues
            audit_entry.total_time_ms = processing_ms
            
            # Log audit information for exception
            self.audit_logger.log_pre_arrival_decision(audit_entry)
            
            logger.error(f"{error_msg} for ticket {ticket_id} (processing time: {processing_ms}ms)", exc_info=True)
            
            return PreArrivalDecision(
                should_continue=True,  # Escalate on error (safe)
                processing_time_ms=processing_ms,
                warnings=warnings
            )
    
    def _format_pre_arrival_approval(self, result: PreArrivalResult) -> str:
        """
        Format customer-friendly approval message.
        
        Args:
            result: PreArrivalResult with timing details
            
        Returns:
            Formatted approval reasoning
        """
        hours = result.hours_before_booking
        
        # Format time description based on duration
        if hours >= 24:
            days = hours / 24
            time_desc = f"{days:.1f} days"
        elif hours >= 1:
            time_desc = f"{hours:.1f} hours"
        else:
            minutes = hours * 60
            time_desc = f"{minutes:.0f} minutes"
        
        # Base approval message with timing details
        base_message = (
            f"Approved: Pre-arrival cancellation. "
            f"Cancellation made {time_desc} before parking start time. "
            f"Per ParkWhiz policy, pre-arrival cancellations are automatically approved."
        )
        
        # Add timezone caveat if UTC fallback was used (customer-friendly language)
        if result.timezone_used == "UTC":
            base_message += " (Note: Location data was incomplete - times calculated using standard time)"
        
        return base_message
    
    def _determine_timezone_source(self, ticket_data: Dict, booking_start_tz: datetime) -> str:
        """
        Determine how the timezone was obtained for audit logging.
        
        Args:
            ticket_data: Original ticket data
            booking_start_tz: Timezone-aware datetime result
            
        Returns:
            Timezone source method ("direct", "inferred", "utc_fallback")
        """
        timezone_field = ticket_data.get("Location Time Zone", "").strip()
        location_field = ticket_data.get("Location City, State", "").strip()
        
        if timezone_field:
            return "direct"
        elif location_field:
            return "inferred"
        else:
            return "utc_fallback"
    
    def _check_suspicious_patterns(self, comparison_result: PreArrivalResult) -> List[str]:
        """
        Check for suspicious timing patterns and return list of detected patterns.
        
        Args:
            comparison_result: Result from time comparison
            
        Returns:
            List of suspicious pattern descriptions
        """
        patterns = []
        
        if comparison_result.hours_before_booking is None:
            return patterns
        
        hours = comparison_result.hours_before_booking
        
        # Far past booking (>7 days ago)
        if hours < -24 * 7:
            days_past = abs(hours) / 24
            patterns.append(f"Booking was {days_past:.1f} days ago - potential data error")
        
        # Far future booking (>12 months)
        if hours > 24 * 30 * 12:
            months_future = hours / (24 * 30)
            patterns.append(f"Booking is {months_future:.1f} months in future - flagging for review")
        
        # Very short booking window (<15 minutes in future)
        if 0 < hours < 0.25:
            minutes_before = hours * 60
            patterns.append(f"Short notice booking: starts in {minutes_before:.1f} minutes")
        
        return patterns
    def _log_performance_metrics(self, processing_ms: int, result: PreArrivalResult, context: Optional[str]):
        """
        Log performance metrics and warnings for monitoring.
        
        Args:
            processing_ms: Total processing time in milliseconds
            result: PreArrivalResult with component timing
            context: Processing context ("webhook" or "chat")
        """
        # Performance warning if exceeding targets
        if processing_ms > self.target_processing_ms:
            logger.warning(
                f"Pre-arrival gate performance warning: {processing_ms}ms > {self.target_processing_ms}ms target "
                f"(context: {context or 'default'}, timezone: {result.timezone_used})"
            )
        
        # Log optimized path success
        elif processing_ms < self.optimized_path_ms:
            logger.debug(
                f"Pre-arrival gate optimized path: {processing_ms}ms < {self.optimized_path_ms}ms "
                f"(context: {context or 'default'}, timezone: {result.timezone_used})"
            )
        
        # Context-specific performance logging
        if context == "webhook" and processing_ms > 5000:  # 5 seconds
            logger.warning(
                f"Webhook pre-arrival processing slow: {processing_ms}ms "
                f"(approaching 8s timeout threshold)"
            )
        elif context == "chat" and processing_ms > 10000:  # 10 seconds
            logger.warning(
                f"Chat pre-arrival processing slow: {processing_ms}ms "
                f"(may impact user experience)"
            )
        
        # Log comprehensive metrics for analysis
        logger.info(
            f"Pre-arrival metrics: total={processing_ms}ms, "
            f"comparison={result.processing_time_ms}ms, "
            f"timezone={result.timezone_used}, "
            f"context={context or 'default'}, "
            f"result={'pre_arrival' if result.is_pre_arrival else 'not_pre_arrival'}"
        )
    
    def get_gate_info(self) -> Dict:
        """
        Get information about the pre-arrival gate configuration and status.
        
        Returns:
            Dictionary with gate configuration and component status
        """
        return {
            "gate_name": "pre_arrival",
            "gate_position": 5,
            "target_processing_ms": self.target_processing_ms,
            "optimized_path_ms": self.optimized_path_ms,
            "components": {
                "payload_parser": {
                    "format_stats": self.payload_parser.get_format_stats()
                },
                "date_parser": {
                    "formats_supported": len(self.date_parser.date_formats),
                    "primary_format": self.date_parser.date_formats[0] if self.date_parser.date_formats else None
                },
                "timezone_handler": {
                    "city_mappings": len(self.timezone_handler.city_mapping),
                    "timezone_cache_size": len(self.timezone_handler.timezone_cache)
                },
                "comparator": {
                    "max_future_months": self.comparator.max_future_months,
                    "max_past_days": self.comparator.max_past_days
                }
            },
            "configuration": self.config
        }
    
    def _track_pre_arrival_approval(
        self, 
        total_processing_ms: int, 
        hours_before_booking: float,
        timezone_source: str,
        format_used: Optional[str],
        context: Optional[str] = None
    ) -> None:
        """
        Track metrics for a pre-arrival approval with context-aware tracking.
        
        Args:
            total_processing_ms: Total processing time in milliseconds
            hours_before_booking: Hours before booking start time
            timezone_source: Source of timezone ("direct", "inferred", "utc_fallback")
            format_used: Date format that successfully parsed the date
            context: Processing context ("webhook" or "chat")
        """
        # Track overall metrics
        self.metrics.total_processed += 1
        self.metrics.pre_arrival_approved += 1
        
        # Track context-specific counts
        if context == "webhook":
            self.metrics.webhook_processed += 1
            self.metrics.webhook_approved += 1
            self.metrics.webhook_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("webhook")
            # Track webhook-specific cost savings
            self.metrics.webhook_cost_savings += self.average_llm_cost
        elif context == "chat":
            self.metrics.chat_processed += 1
            self.metrics.chat_approved += 1
            self.metrics.chat_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("chat")
            # Track chat-specific cost savings
            self.metrics.chat_cost_savings += self.average_llm_cost
        
        # Track processing time (overall)
        self.processing_times.append(total_processing_ms)
        self._update_processing_time_metrics()
        
        # Track hours before booking
        if hours_before_booking is not None:
            self.hours_before_booking_values.append(hours_before_booking)
            self._update_hours_before_booking_avg()
        
        # Track timezone inference
        if timezone_source == "inferred":
            self.metrics.timezone_inference_used += 1
        elif timezone_source == "utc_fallback":
            self.metrics.utc_fallback_used += 1
        
        # Track date format success
        if format_used:
            if format_used == self.date_parser.date_formats[0]:
                self.metrics.primary_format_success += 1
            else:
                self.metrics.fallback_format_success += 1
        
        # Calculate cost savings (LLM call avoided) - overall
        self.metrics.cost_savings_estimate += self.average_llm_cost
        
        # Recalculate derived metrics
        self.metrics.calculate_derived_metrics()
        
        # Log context-aware metrics
        logger.debug(
            f"Pre-arrival approval tracked: context={context or 'default'}, "
            f"processing_ms={total_processing_ms}, hours_before={hours_before_booking:.1f}, "
            f"timezone_source={timezone_source}"
        )

    
    def _track_not_pre_arrival(
        self, 
        total_processing_ms: int,
        timezone_source: str,
        format_used: Optional[str],
        context: Optional[str] = None
    ) -> None:
        """
        Track metrics for a not-pre-arrival result with context-aware tracking.
        
        Args:
            total_processing_ms: Total processing time in milliseconds
            timezone_source: Source of timezone ("direct", "inferred", "utc_fallback")
            format_used: Date format that successfully parsed the date
            context: Processing context ("webhook" or "chat")
        """
        # Track overall metrics
        self.metrics.total_processed += 1
        self.metrics.not_pre_arrival += 1
        
        # Track context-specific counts
        if context == "webhook":
            self.metrics.webhook_processed += 1
            self.metrics.webhook_not_pre_arrival += 1
            self.metrics.webhook_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("webhook")
        elif context == "chat":
            self.metrics.chat_processed += 1
            self.metrics.chat_not_pre_arrival += 1
            self.metrics.chat_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("chat")
        
        # Track processing time (overall)
        self.processing_times.append(total_processing_ms)
        self._update_processing_time_metrics()
        
        # Track timezone inference
        if timezone_source == "inferred":
            self.metrics.timezone_inference_used += 1
        elif timezone_source == "utc_fallback":
            self.metrics.utc_fallback_used += 1
        
        # Track date format success
        if format_used:
            if format_used == self.date_parser.date_formats[0]:
                self.metrics.primary_format_success += 1
            else:
                self.metrics.fallback_format_success += 1
        
        # Recalculate derived metrics
        self.metrics.calculate_derived_metrics()
        
        # Log context-aware metrics
        logger.debug(
            f"Not pre-arrival tracked: context={context or 'default'}, "
            f"processing_ms={total_processing_ms}, timezone_source={timezone_source}"
        )

    
    def _track_parsing_failure(self, total_processing_ms: int, context: Optional[str] = None) -> None:
        """
        Track metrics for a date parsing failure with context-aware tracking.
        
        Args:
            total_processing_ms: Total processing time in milliseconds
            context: Processing context ("webhook" or "chat")
        """
        # Track overall metrics
        self.metrics.total_processed += 1
        self.metrics.parsing_failures += 1
        
        # Track context-specific counts
        if context == "webhook":
            self.metrics.webhook_processed += 1
            self.metrics.webhook_parsing_failures += 1
            self.metrics.webhook_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("webhook")
        elif context == "chat":
            self.metrics.chat_processed += 1
            self.metrics.chat_parsing_failures += 1
            self.metrics.chat_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("chat")
        
        # Track processing time (overall)
        self.processing_times.append(total_processing_ms)
        self._update_processing_time_metrics()
        
        # Recalculate derived metrics
        self.metrics.calculate_derived_metrics()
        
        # Log context-aware metrics
        logger.debug(
            f"Parsing failure tracked: context={context or 'default'}, "
            f"processing_ms={total_processing_ms}"
        )

    
    def _track_timezone_failure(self, total_processing_ms: int, context: Optional[str] = None) -> None:
        """
        Track metrics for a timezone handling failure with context-aware tracking.
        
        Args:
            total_processing_ms: Total processing time in milliseconds
            context: Processing context ("webhook" or "chat")
        """
        # Track overall metrics
        self.metrics.total_processed += 1
        self.metrics.timezone_failures += 1
        
        # Track context-specific counts
        if context == "webhook":
            self.metrics.webhook_processed += 1
            self.metrics.webhook_timezone_failures += 1
            self.metrics.webhook_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("webhook")
        elif context == "chat":
            self.metrics.chat_processed += 1
            self.metrics.chat_timezone_failures += 1
            self.metrics.chat_processing_times.append(total_processing_ms)
            self.metrics.update_context_performance_metrics("chat")
        
        # Track processing time (overall)
        self.processing_times.append(total_processing_ms)
        self._update_processing_time_metrics()
        
        # Recalculate derived metrics
        self.metrics.calculate_derived_metrics()
        
        # Log context-aware metrics
        logger.debug(
            f"Timezone failure tracked: context={context or 'default'}, "
            f"processing_ms={total_processing_ms}"
        )

    
    def _update_processing_time_metrics(self) -> None:
        """Update processing time metrics from collected data."""
        if not self.processing_times:
            return
        
        # Calculate average
        self.metrics.avg_processing_ms = sum(self.processing_times) / len(self.processing_times)
        
        # Calculate 95th percentile
        sorted_times = sorted(self.processing_times)
        p95_index = int(len(sorted_times) * 0.95)
        self.metrics.p95_processing_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        # Track maximum
        self.metrics.max_processing_ms = max(self.processing_times)
    
    def _update_hours_before_booking_avg(self) -> None:
        """Update average hours before booking from collected data."""
        if not self.hours_before_booking_values:
            return
        
        self.metrics.hours_before_booking_avg = sum(self.hours_before_booking_values) / len(self.hours_before_booking_values)
    
    def track_webhook_timeout(self, processing_ms: int) -> None:
        """
        Track a webhook timeout event.
        
        Args:
            processing_ms: Processing time when timeout occurred
        """
        self.metrics.webhook_timeout_count += 1
        self.metrics.webhook_processed += 1
        self.metrics.total_processed += 1
        
        # Track processing time
        self.metrics.webhook_processing_times.append(processing_ms)
        self.processing_times.append(processing_ms)
        
        # Update metrics
        self.metrics.update_context_performance_metrics("webhook")
        self._update_processing_time_metrics()
        self.metrics.calculate_derived_metrics()
        
        logger.warning(
            f"Webhook timeout tracked: processing_ms={processing_ms}, "
            f"total_timeouts={self.metrics.webhook_timeout_count}, "
            f"timeout_rate={self.metrics.webhook_timeout_rate:.2f}%"
        )
    
    def get_metrics(self) -> PreArrivalMetrics:
        """
        Get current metrics for the pre-arrival gate.
        
        Returns:
            PreArrivalMetrics with current statistics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.metrics = PreArrivalMetrics()
        self.processing_times = []
        self.hours_before_booking_values = []
    
    def get_metrics_summary(self) -> Dict:
        """
        Get a human-readable summary of metrics with context-aware breakdown.
        
        Returns:
            Dictionary with formatted metrics summary including context-specific data
        """
        return {
            "processing_counts": {
                "total_processed": self.metrics.total_processed,
                "pre_arrival_approved": self.metrics.pre_arrival_approved,
                "not_pre_arrival": self.metrics.not_pre_arrival,
                "parsing_failures": self.metrics.parsing_failures,
                "timezone_failures": self.metrics.timezone_failures,
                "approval_rate_percent": (
                    (self.metrics.pre_arrival_approved / self.metrics.total_processed * 100)
                    if self.metrics.total_processed > 0 else 0.0
                )
            },
            "webhook_context": {
                "processed": self.metrics.webhook_processed,
                "approved": self.metrics.webhook_approved,
                "not_pre_arrival": self.metrics.webhook_not_pre_arrival,
                "parsing_failures": self.metrics.webhook_parsing_failures,
                "timezone_failures": self.metrics.webhook_timezone_failures,
                "timeout_count": self.metrics.webhook_timeout_count,
                "timeout_rate_percent": round(self.metrics.webhook_timeout_rate, 2),
                "approval_rate_percent": (
                    (self.metrics.webhook_approved / self.metrics.webhook_processed * 100)
                    if self.metrics.webhook_processed > 0 else 0.0
                ),
                "avg_processing_ms": round(self.metrics.webhook_avg_processing_ms, 2),
                "p95_processing_ms": round(self.metrics.webhook_p95_processing_ms, 2),
                "max_processing_ms": round(self.metrics.webhook_max_processing_ms, 2),
                "cost_savings_usd": round(self.metrics.webhook_cost_savings, 2)
            },
            "chat_context": {
                "processed": self.metrics.chat_processed,
                "approved": self.metrics.chat_approved,
                "not_pre_arrival": self.metrics.chat_not_pre_arrival,
                "parsing_failures": self.metrics.chat_parsing_failures,
                "timezone_failures": self.metrics.chat_timezone_failures,
                "approval_rate_percent": (
                    (self.metrics.chat_approved / self.metrics.chat_processed * 100)
                    if self.metrics.chat_processed > 0 else 0.0
                ),
                "avg_processing_ms": round(self.metrics.chat_avg_processing_ms, 2),
                "p95_processing_ms": round(self.metrics.chat_p95_processing_ms, 2),
                "max_processing_ms": round(self.metrics.chat_max_processing_ms, 2),
                "cost_savings_usd": round(self.metrics.chat_cost_savings, 2)
            },
            "performance_metrics": {
                "avg_processing_ms": round(self.metrics.avg_processing_ms, 2),
                "p95_processing_ms": round(self.metrics.p95_processing_ms, 2),
                "max_processing_ms": round(self.metrics.max_processing_ms, 2),
                "target_ms": self.target_processing_ms,
                "meets_target": self.metrics.p95_processing_ms < self.target_processing_ms
            },
            "timezone_metrics": {
                "timezone_inference_used": self.metrics.timezone_inference_used,
                "utc_fallback_used": self.metrics.utc_fallback_used,
                "timezone_inference_success_rate": round(self.metrics.timezone_inference_success_rate * 100, 2)
            },
            "date_parsing_metrics": {
                "primary_format_success": self.metrics.primary_format_success,
                "fallback_format_success": self.metrics.fallback_format_success,
                "parsing_success_rate": round(self.metrics.parsing_success_rate * 100, 2)
            },
            "business_metrics": {
                "hours_before_booking_avg": round(self.metrics.hours_before_booking_avg, 2),
                "cost_savings_estimate_usd": round(self.metrics.cost_savings_estimate, 2),
                "average_llm_cost_per_ticket": self.average_llm_cost
            }
        }
    
    def check_alert_conditions(self) -> List[Dict]:
        """
        Check for alert conditions with context-aware thresholds.
        
        Returns:
            List of alert dictionaries with severity, message, and context
        """
        alerts = []
        
        # Alert thresholds (configurable)
        webhook_timeout_threshold = self.config.get("webhook_timeout_threshold_percent", 5.0)
        webhook_performance_threshold_ms = self.config.get("webhook_performance_threshold_ms", 5000)
        chat_performance_threshold_ms = self.config.get("chat_performance_threshold_ms", 10000)
        parsing_failure_threshold = self.config.get("parsing_failure_threshold_percent", 5.0)
        timezone_fallback_threshold = self.config.get("timezone_fallback_threshold_percent", 10.0)
        
        # Check webhook timeout rate
        if self.metrics.webhook_processed > 0 and self.metrics.webhook_timeout_rate > webhook_timeout_threshold:
            alerts.append({
                "severity": "high",
                "context": "webhook",
                "metric": "timeout_rate",
                "message": (
                    f"Webhook timeout rate ({self.metrics.webhook_timeout_rate:.2f}%) exceeds threshold "
                    f"({webhook_timeout_threshold}%). {self.metrics.webhook_timeout_count} timeouts out of "
                    f"{self.metrics.webhook_processed} webhook requests."
                ),
                "current_value": self.metrics.webhook_timeout_rate,
                "threshold": webhook_timeout_threshold,
                "recommendation": "Investigate webhook processing performance. Consider optimizing date parsing or timezone handling."
            })
        
        # Check webhook performance (p95 latency)
        if self.metrics.webhook_processed > 0 and self.metrics.webhook_p95_processing_ms > webhook_performance_threshold_ms:
            alerts.append({
                "severity": "medium",
                "context": "webhook",
                "metric": "p95_latency",
                "message": (
                    f"Webhook p95 processing time ({self.metrics.webhook_p95_processing_ms:.0f}ms) exceeds threshold "
                    f"({webhook_performance_threshold_ms}ms). Risk of approaching 8-second timeout limit."
                ),
                "current_value": self.metrics.webhook_p95_processing_ms,
                "threshold": webhook_performance_threshold_ms,
                "recommendation": "Review slow webhook processing paths. Check for timezone cache misses or complex date formats."
            })
        
        # Check chat performance (p95 latency)
        if self.metrics.chat_processed > 0 and self.metrics.chat_p95_processing_ms > chat_performance_threshold_ms:
            alerts.append({
                "severity": "low",
                "context": "chat",
                "metric": "p95_latency",
                "message": (
                    f"Chat p95 processing time ({self.metrics.chat_p95_processing_ms:.0f}ms) exceeds threshold "
                    f"({chat_performance_threshold_ms}ms). May impact user experience."
                ),
                "current_value": self.metrics.chat_p95_processing_ms,
                "threshold": chat_performance_threshold_ms,
                "recommendation": "Review chat processing performance. Consider optimizing for interactive use cases."
            })
        
        # Check parsing failure rate (overall)
        if self.metrics.total_processed > 0:
            parsing_failure_rate = (self.metrics.parsing_failures / self.metrics.total_processed) * 100
            if parsing_failure_rate > parsing_failure_threshold:
                # Determine which context has more failures
                webhook_parsing_rate = (
                    (self.metrics.webhook_parsing_failures / self.metrics.webhook_processed * 100)
                    if self.metrics.webhook_processed > 0 else 0.0
                )
                chat_parsing_rate = (
                    (self.metrics.chat_parsing_failures / self.metrics.chat_processed * 100)
                    if self.metrics.chat_processed > 0 else 0.0
                )
                
                primary_context = "webhook" if webhook_parsing_rate > chat_parsing_rate else "chat"
                
                alerts.append({
                    "severity": "medium",
                    "context": primary_context,
                    "metric": "parsing_failure_rate",
                    "message": (
                        f"Date parsing failure rate ({parsing_failure_rate:.2f}%) exceeds threshold "
                        f"({parsing_failure_threshold}%). {self.metrics.parsing_failures} failures out of "
                        f"{self.metrics.total_processed} tickets. Primary context: {primary_context} "
                        f"(webhook: {webhook_parsing_rate:.2f}%, chat: {chat_parsing_rate:.2f}%)."
                    ),
                    "current_value": parsing_failure_rate,
                    "threshold": parsing_failure_threshold,
                    "recommendation": "Review date format configuration. Check for new date formats in production data."
                })
        
        # Check timezone fallback rate (overall)
        total_timezone_attempts = self.metrics.timezone_inference_used + self.metrics.utc_fallback_used
        if total_timezone_attempts > 0:
            utc_fallback_rate = (self.metrics.utc_fallback_used / total_timezone_attempts) * 100
            if utc_fallback_rate > timezone_fallback_threshold:
                alerts.append({
                    "severity": "low",
                    "context": "all",
                    "metric": "utc_fallback_rate",
                    "message": (
                        f"UTC fallback rate ({utc_fallback_rate:.2f}%) exceeds threshold "
                        f"({timezone_fallback_threshold}%). {self.metrics.utc_fallback_used} UTC fallbacks out of "
                        f"{total_timezone_attempts} timezone inferences."
                    ),
                    "current_value": utc_fallback_rate,
                    "threshold": timezone_fallback_threshold,
                    "recommendation": "Review city-to-timezone mapping. Consider expanding location coverage."
                })
        
        # Check webhook vs chat success rate disparity
        if self.metrics.webhook_processed > 10 and self.metrics.chat_processed > 10:
            webhook_success_rate = (
                (self.metrics.webhook_approved / self.metrics.webhook_processed * 100)
                if self.metrics.webhook_processed > 0 else 0.0
            )
            chat_success_rate = (
                (self.metrics.chat_approved / self.metrics.chat_processed * 100)
                if self.metrics.chat_processed > 0 else 0.0
            )
            
            # Alert if success rates differ by more than 20%
            if abs(webhook_success_rate - chat_success_rate) > 20:
                alerts.append({
                    "severity": "medium",
                    "context": "comparison",
                    "metric": "success_rate_disparity",
                    "message": (
                        f"Significant disparity in success rates between contexts: "
                        f"webhook={webhook_success_rate:.2f}%, chat={chat_success_rate:.2f}%. "
                        f"Difference: {abs(webhook_success_rate - chat_success_rate):.2f}%."
                    ),
                    "current_value": abs(webhook_success_rate - chat_success_rate),
                    "threshold": 20.0,
                    "recommendation": "Investigate context-specific processing differences. Verify identical logic execution."
                })
        
        return alerts
    
    def log_alerts(self) -> None:
        """
        Check alert conditions and log any triggered alerts with context information.
        """
        alerts = self.check_alert_conditions()
        
        if not alerts:
            logger.debug("No alert conditions triggered")
            return
        
        for alert in alerts:
            severity = alert["severity"]
            context = alert["context"]
            metric = alert["metric"]
            message = alert["message"]
            recommendation = alert.get("recommendation", "")
            
            log_message = (
                f"[{severity.upper()}] Pre-Arrival Alert - Context: {context}, Metric: {metric}\n"
                f"{message}\n"
                f"Recommendation: {recommendation}"
            )
            
            if severity == "high":
                logger.error(log_message)
            elif severity == "medium":
                logger.warning(log_message)
            else:
                logger.info(log_message)


    def get_metrics_for_alerting(self) -> Dict[str, float]:
        """
        Get current metrics in format needed for alerting system.
        
        Returns:
            Dictionary with metric names and current values for alert evaluation
        """
        # Calculate rates
        parsing_failure_rate = 0.0
        utc_fallback_rate = 0.0
        suspicious_pattern_rate = 0.0
        
        if self.metrics.total_processed > 0:
            parsing_failure_rate = (self.metrics.parsing_failures / self.metrics.total_processed) * 100
            
            # Calculate UTC fallback rate
            total_timezone_attempts = self.metrics.timezone_inference_used + self.metrics.utc_fallback_used
            if total_timezone_attempts > 0:
                utc_fallback_rate = (self.metrics.utc_fallback_used / total_timezone_attempts) * 100
        
        # Calculate webhook timeout rate
        webhook_timeout_rate = 0.0
        if self.metrics.webhook_processed > 0:
            webhook_timeout_rate = (self.metrics.webhook_timeout_count / self.metrics.webhook_processed) * 100
        
        return {
            # Parsing metrics (Requirement 20.1)
            "parsing_failure_rate_percent": parsing_failure_rate,
            
            # Timezone metrics (Requirement 20.2)
            "utc_fallback_rate_percent": utc_fallback_rate,
            "timezone_inference_success_rate_percent": self.metrics.timezone_inference_success_rate * 100,
            
            # Performance metrics (Requirement 20.3)
            "processing_latency_p95_ms": self.metrics.p95_processing_ms,
            "processing_latency_avg_ms": self.metrics.avg_processing_ms,
            "processing_latency_max_ms": self.metrics.max_processing_ms,
            
            # Webhook-specific performance
            "webhook_processing_latency_p95_ms": self.metrics.webhook_p95_processing_ms,
            "webhook_timeout_rate_percent": webhook_timeout_rate,
            
            # Chat-specific performance
            "chat_processing_latency_p95_ms": self.metrics.chat_p95_processing_ms,
            
            # Suspicious patterns (Requirement 20.4)
            "suspicious_pattern_rate_percent": suspicious_pattern_rate,
            
            # Data quality metrics
            "timezone_failure_rate_percent": (
                (self.metrics.timezone_failures / self.metrics.total_processed * 100)
                if self.metrics.total_processed > 0 else 0.0
            ),
            
            # Success metrics
            "approval_rate_percent": (
                (self.metrics.pre_arrival_approved / self.metrics.total_processed * 100)
                if self.metrics.total_processed > 0 else 0.0
            ),
            
            # Volume metrics
            "total_processed": float(self.metrics.total_processed),
            "webhook_processed": float(self.metrics.webhook_processed),
            "chat_processed": float(self.metrics.chat_processed)
        }
