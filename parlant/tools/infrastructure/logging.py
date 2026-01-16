"""
Structured logging configuration for webhook automation.

This module provides JSON-formatted structured logging with consistent
fields across all webhook and journey processing operations.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Optional, Dict, Any


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter that outputs logs in JSON format with structured fields.
    
    Each log entry includes:
    - timestamp: ISO 8601 formatted timestamp
    - level: Log level (INFO, WARNING, ERROR, etc.)
    - component: The module/component generating the log
    - event: A short description of what happened
    - Additional context fields (ticket_id, processing_time_ms, etc.)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.
        
        Args:
            record: The log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log structure
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "component": record.name,
            "event": record.getMessage(),
        }
        
        # Add ticket_id if present in extra fields
        if hasattr(record, "ticket_id"):
            log_data["ticket_id"] = record.ticket_id
        
        # Add processing_time_ms if present
        if hasattr(record, "processing_time_ms"):
            log_data["processing_time_ms"] = record.processing_time_ms
        
        # Add journey_name if present
        if hasattr(record, "journey_name"):
            log_data["journey_name"] = record.journey_name
        
        # Add decision if present
        if hasattr(record, "decision"):
            log_data["decision"] = record.decision
        
        # Add tool_name if present
        if hasattr(record, "tool_name"):
            log_data["tool_name"] = record.tool_name
        
        # Add event_type if present
        if hasattr(record, "event_type"):
            log_data["event_type"] = record.event_type
        
        # Add source_ip if present
        if hasattr(record, "source_ip"):
            log_data["source_ip"] = record.source_ip
        
        # Add signature_valid if present
        if hasattr(record, "signature_valid"):
            log_data["signature_valid"] = record.signature_valid
        
        # Add error details if this is an error log
        if record.exc_info:
            log_data["error"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info) if record.exc_info else None
            }
        
        # Add any additional extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def configure_structured_logging(
    level: str = "INFO",
    component_name: Optional[str] = None
) -> logging.Logger:
    """
    Configure structured JSON logging for a component.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        component_name: Name of the component (defaults to root logger)
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(component_name) if component_name else logging.getLogger()
    
    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Set log level
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler with structured formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))
    handler.setFormatter(StructuredFormatter())
    
    # Add handler to logger
    logger.addHandler(handler)
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def log_webhook_received(
    logger: logging.Logger,
    ticket_id: str,
    event_type: str,
    source_ip: str
) -> None:
    """
    Log webhook received event with structured fields.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        event_type: Type of webhook event
        source_ip: Source IP address
    """
    logger.info(
        "Webhook received",
        extra={
            "ticket_id": ticket_id,
            "event_type": event_type,
            "source_ip": source_ip
        }
    )


def log_signature_validation(
    logger: logging.Logger,
    ticket_id: str,
    is_valid: bool,
    source_ip: str
) -> None:
    """
    Log webhook signature validation result.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        is_valid: Whether signature validation passed
        source_ip: Source IP address
    """
    level = logging.INFO if is_valid else logging.WARNING
    message = "Signature validation passed" if is_valid else "Signature validation failed"
    
    logger.log(
        level,
        message,
        extra={
            "ticket_id": ticket_id,
            "signature_valid": is_valid,
            "source_ip": source_ip
        }
    )


def log_routing_decision(
    logger: logging.Logger,
    ticket_id: str,
    trigger_source: str,
    journey_name: str
) -> None:
    """
    Log journey routing decision.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        trigger_source: Source that triggered processing (webhook/chat)
        journey_name: Name of journey being activated
    """
    logger.info(
        "Journey routing decision made",
        extra={
            "ticket_id": ticket_id,
            "trigger_source": trigger_source,
            "journey_name": journey_name
        }
    )


def log_journey_activation(
    logger: logging.Logger,
    ticket_id: str,
    journey_name: str,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """
    Log journey activation attempt.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        journey_name: Name of journey being activated
        success: Whether activation succeeded
        error: Error message if activation failed
    """
    if success:
        logger.info(
            "Journey activated successfully",
            extra={
                "ticket_id": ticket_id,
                "journey_name": journey_name
            }
        )
    else:
        logger.error(
            f"Journey activation failed: {error}",
            extra={
                "ticket_id": ticket_id,
                "journey_name": journey_name,
                "error_message": error
            }
        )


def log_journey_start(
    logger: logging.Logger,
    ticket_id: str,
    journey_name: str
) -> None:
    """
    Log journey execution start.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        journey_name: Name of journey starting
    """
    logger.info(
        "Journey execution started",
        extra={
            "ticket_id": ticket_id,
            "journey_name": journey_name
        }
    )


def log_journey_end(
    logger: logging.Logger,
    ticket_id: str,
    journey_name: str,
    processing_time_ms: int,
    decision: Optional[str] = None
) -> None:
    """
    Log journey execution completion.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        journey_name: Name of journey that completed
        processing_time_ms: Total processing time in milliseconds
        decision: Final decision (Approved/Denied/Escalated)
    """
    logger.info(
        "Journey execution completed",
        extra={
            "ticket_id": ticket_id,
            "journey_name": journey_name,
            "processing_time_ms": processing_time_ms,
            "decision": decision
        }
    )


def log_tool_execution(
    logger: logging.Logger,
    ticket_id: str,
    tool_name: str,
    success: bool = True,
    error: Optional[str] = None
) -> None:
    """
    Log tool execution.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        tool_name: Name of tool being executed
        success: Whether execution succeeded
        error: Error message if execution failed
    """
    if success:
        logger.info(
            f"Tool executed: {tool_name}",
            extra={
                "ticket_id": ticket_id,
                "tool_name": tool_name
            }
        )
    else:
        logger.error(
            f"Tool execution failed: {tool_name} - {error}",
            extra={
                "ticket_id": ticket_id,
                "tool_name": tool_name,
                "error_message": error
            }
        )


def log_decision_outcome(
    logger: logging.Logger,
    ticket_id: str,
    decision: str,
    confidence: Optional[float] = None,
    reasoning: Optional[str] = None
) -> None:
    """
    Log refund decision outcome.
    
    Args:
        logger: Logger instance
        ticket_id: Freshdesk ticket ID
        decision: Decision made (Approved/Denied/Needs Human Review)
        confidence: Confidence score if available
        reasoning: Decision reasoning
    """
    extra_fields = {
        "ticket_id": ticket_id,
        "decision": decision
    }
    
    if confidence is not None:
        extra_fields["confidence"] = confidence
    
    if reasoning:
        extra_fields["reasoning"] = reasoning
    
    logger.info(
        f"Decision made: {decision}",
        extra=extra_fields
    )


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    ticket_id: Optional[str] = None,
    journey_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log an error with full context and stack trace.
    
    Args:
        logger: Logger instance
        error: The exception that occurred
        ticket_id: Freshdesk ticket ID if applicable
        journey_name: Journey name if applicable
        tool_name: Tool name if applicable
        context: Additional context dictionary
    """
    extra_fields = {}
    
    if ticket_id:
        extra_fields["ticket_id"] = ticket_id
    
    if journey_name:
        extra_fields["journey_name"] = journey_name
    
    if tool_name:
        extra_fields["tool_name"] = tool_name
    
    if context:
        extra_fields.update(context)
    
    logger.error(
        f"Error occurred: {str(error)}",
        extra=extra_fields,
        exc_info=True
    )


def log_performance_warning(
    logger: logging.Logger,
    operation: str,
    duration_ms: int,
    threshold_ms: int,
    ticket_id: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log a performance warning for slow operations.
    
    Args:
        logger: Logger instance
        operation: Name of the operation that was slow
        duration_ms: Actual duration in milliseconds
        threshold_ms: Threshold that was exceeded
        ticket_id: Freshdesk ticket ID if applicable
        context: Additional context dictionary
    """
    extra_fields = {
        "operation": operation,
        "duration_ms": duration_ms,
        "threshold_ms": threshold_ms
    }
    
    if ticket_id:
        extra_fields["ticket_id"] = ticket_id
    
    if context:
        extra_fields.update(context)
    
    logger.warning(
        f"Slow operation detected: {operation} took {duration_ms}ms (threshold: {threshold_ms}ms)",
        extra=extra_fields
    )


def log_api_call(
    logger: logging.Logger,
    api_name: str,
    latency_ms: int,
    success: bool = True,
    ticket_id: Optional[str] = None,
    error: Optional[str] = None
) -> None:
    """
    Log an API call with latency information.
    
    Args:
        logger: Logger instance
        api_name: Name of the API called
        latency_ms: Latency in milliseconds
        success: Whether the call succeeded
        ticket_id: Freshdesk ticket ID if applicable
        error: Error message if call failed
    """
    extra_fields = {
        "api_name": api_name,
        "latency_ms": latency_ms,
        "success": success
    }
    
    if ticket_id:
        extra_fields["ticket_id"] = ticket_id
    
    if error:
        extra_fields["error_message"] = error
    
    level = logging.INFO if success else logging.ERROR
    message = f"API call: {api_name} ({latency_ms}ms)"
    if not success:
        message += f" - Failed: {error}"
    
    logger.log(level, message, extra=extra_fields)



def log_error_rate_alert(
    logger: logging.Logger,
    error_type: str,
    rate_percent: float,
    threshold_percent: float,
    severity: str = "medium"
) -> None:
    """
    Log an error rate threshold alert.
    
    Args:
        logger: Logger instance
        error_type: Type of error exceeding threshold
        rate_percent: Current error rate percentage
        threshold_percent: Threshold that was exceeded
        severity: Alert severity (low/medium/high)
    """
    logger.warning(
        f"Error rate alert: {error_type} error rate ({rate_percent}%) exceeds threshold ({threshold_percent}%)",
        extra={
            "alert_type": "error_rate_threshold",
            "error_type": error_type,
            "rate_percent": rate_percent,
            "threshold_percent": threshold_percent,
            "severity": severity
        }
    )
