"""
FastAPI webhook server for receiving Freshdesk webhooks.

This module provides HTTP endpoints for:
- Receiving Freshdesk webhook events
- Health checks for monitoring
- Test endpoint for manual testing
"""

import os
import logging
from datetime import datetime
from typing import Optional, Literal, Dict, Any, Set, Union
from collections import defaultdict, deque
from fastapi import FastAPI, Request, HTTPException, status
from pydantic import BaseModel, Field

# Import webhook validator and journey router
from app_tools.tools.infrastructure.webhook_validator import validate_freshdesk_signature
from app_tools.journey_router import route_to_journey
from app_tools.tools.ticket_processing.activator import activate_journey

# Import structured logging
from app_tools.tools.infrastructure.logging import (
    configure_structured_logging,
    log_webhook_received,
    log_signature_validation,
    log_routing_decision,
    log_journey_activation,
    log_error_with_context,
    log_performance_warning
)

# Import metrics tracking
from app_tools.tools.infrastructure.metrics import get_metrics_tracker

# Import webhook configuration
from app_tools.tools.infrastructure.webhook_config import get_validated_config

# Load and validate configuration
webhook_config = get_validated_config()

# Configure structured logging with config
logger = configure_structured_logging(
    level=webhook_config.log_level,
    component_name="webhook_server"
)

# Log configuration on startup
logger.info(
    "Webhook server configuration loaded",
    extra={
        "enabled": webhook_config.enabled,
        "port": webhook_config.port,
        "path": webhook_config.path,
        "events": webhook_config.events,
        "rate_limit": webhook_config.rate_limit,
        "rate_limit_window": webhook_config.rate_limit_window,
        "deduplication_window": webhook_config.deduplication_window
    }
)

# Get metrics tracker instance
metrics = get_metrics_tracker()

# Initialize FastAPI app
app = FastAPI(
    title="Parlant Webhook Server",
    description="Receives and processes Freshdesk webhook events",
    version="1.0.0"
)

# Rate limiting storage: {ip_address: [(timestamp1, timestamp2, ...)]}
rate_limit_storage = defaultdict(list)

# Event deduplication storage: stores (ticket_id, event_type, timestamp) tuples
# Using deque with maxlen to automatically limit size
event_deduplication_storage = deque(maxlen=100)

# Supported event types from configuration
SUPPORTED_EVENT_TYPES: Set[str] = set(webhook_config.events)

# Refund-related keywords for filtering ticket updates
REFUND_KEYWORDS = {
    "refund", "refunds", "reimbursement", "reimburse", "money back",
    "cancel", "cancellation", "cancelled", "canceled",
    "chargeback", "charge back", "dispute"
}


def check_rate_limit(client_ip: str) -> bool:
    """
    Check if the client has exceeded the rate limit.
    
    Args:
        client_ip: The IP address of the client
        
    Returns:
        True if within rate limit, False if exceeded
    """
    now = datetime.utcnow()
    
    # Get request timestamps for this IP
    timestamps = rate_limit_storage[client_ip]
    
    # Remove timestamps older than the rate limit window
    cutoff_time = now.timestamp() - webhook_config.rate_limit_window
    timestamps[:] = [ts for ts in timestamps if ts > cutoff_time]
    
    # Check if limit exceeded
    if len(timestamps) >= webhook_config.rate_limit:
        return False
    
    # Add current timestamp
    timestamps.append(now.timestamp())
    
    return True


def is_duplicate_event(ticket_id: str, event_type: str) -> bool:
    """
    Check if this event was already processed recently.
    
    Args:
        ticket_id: The ticket ID
        event_type: The event type
        
    Returns:
        True if this is a duplicate event, False otherwise
    """
    now = datetime.utcnow()
    cutoff_time = now.timestamp() - webhook_config.deduplication_window
    
    # Check if this event exists in recent history
    for stored_ticket_id, stored_event_type, stored_timestamp in event_deduplication_storage:
        if (stored_ticket_id == ticket_id and 
            stored_event_type == event_type and 
            stored_timestamp > cutoff_time):
            return True
    
    # Not a duplicate - add to storage
    event_deduplication_storage.append((ticket_id, event_type, now.timestamp()))
    
    return False


def is_supported_event_type(event_type: str) -> bool:
    """
    Check if the event type is supported for processing.
    
    Args:
        event_type: The event type from the webhook
        
    Returns:
        True if supported, False otherwise
    """
    return event_type in SUPPORTED_EVENT_TYPES


def is_refund_related(payload: Dict[str, Any]) -> bool:
    """
    Check if a ticket update is refund-related.
    
    This function checks various fields in the payload for refund-related keywords:
    - ticket_subject
    - ticket_description (if present)
    - ticket_tags (if present)
    - custom_fields (if present)
    
    Args:
        payload: The webhook payload dictionary
        
    Returns:
        True if the update is refund-related, False otherwise
    """
    # For ticket_created events, always process (they might be refund requests)
    event_type = payload.get("event", "")
    if event_type == "ticket_created":
        return True
    
    # For ticket_updated events, check if it's refund-related
    if event_type == "ticket_updated":
        # Check subject
        subject = payload.get("ticket_subject", "").lower()
        if any(keyword in subject for keyword in REFUND_KEYWORDS):
            return True
        
        # Check description if present
        description = payload.get("ticket_description", "").lower()
        if any(keyword in description for keyword in REFUND_KEYWORDS):
            return True
        
        # Check tags if present
        tags = payload.get("ticket_tags", [])
        if isinstance(tags, list):
            tags_str = " ".join(str(tag).lower() for tag in tags)
            if any(keyword in tags_str for keyword in REFUND_KEYWORDS):
                return True
        
        # Check custom fields if present
        custom_fields = payload.get("custom_fields", {})
        if isinstance(custom_fields, dict):
            custom_fields_str = " ".join(str(v).lower() for v in custom_fields.values())
            if any(keyword in custom_fields_str for keyword in REFUND_KEYWORDS):
                return True
        
        # Not refund-related
        return False
    
    # For other event types, don't process
    return False


class FreshdeskWebhookPayload(BaseModel):
    """Freshdesk webhook payload structure"""
    ticket_id: Union[str, int] = Field(..., description="The ticket ID from Freshdesk")
    ticket_subject: Optional[str] = Field(None, description="Subject of the ticket")
    ticket_description: Optional[str] = Field(None, description="Description of the ticket")
    ticket_url: Optional[str] = Field(None, description="URL to the ticket")
    ticket_portal_url: Optional[str] = Field(None, description="Portal URL for the ticket")
    ticket_status: Optional[str] = Field(None, description="Status of the ticket")
    ticket_priority: Optional[str] = Field(None, description="Priority level of the ticket")
    ticket_contact_name: Optional[str] = Field(None, description="Name of the ticket contact")
    ticket_contact_email: Optional[str] = Field(None, description="Email of the ticket contact")
    # Optional fields that may not always be present
    event: Optional[str] = Field(None, description="The type of event that triggered the webhook")
    triggered_at: Optional[str] = Field(None, description="ISO 8601 timestamp of when the event occurred")


class WebhookResponse(BaseModel):
    """Response sent back to Freshdesk"""
    status: Literal["success", "error"]
    message: str
    ticket_id: Optional[str] = None
    processing_time_ms: Optional[int] = None


@app.get("/webhook/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    
    Returns:
        dict: Health status information
    """
    logger.info("Health check requested")
    return {
        "status": "healthy",
        "service": "webhook_server",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/webhook/metrics")
async def get_metrics():
    """
    Metrics endpoint for monitoring and observability.
    
    Returns comprehensive metrics including:
    - Webhook processing statistics
    - Journey execution metrics
    - Error rates
    - Performance metrics
    
    Returns:
        dict: Comprehensive metrics summary
    """
    logger.info("Metrics requested")
    
    # Check for error rate alerts
    alerts = metrics.get_error_rate_alerts()
    
    # Log any alerts
    if alerts:
        from app_tools.tools.infrastructure.logging import log_error_rate_alert
        for alert in alerts:
            log_error_rate_alert(
                logger,
                error_type=alert["error_type"],
                rate_percent=alert["rate_percent"],
                threshold_percent=alert["threshold_percent"],
                severity=alert["severity"]
            )
    
    summary = metrics.get_summary()
    summary["alerts"] = alerts
    
    return summary


@app.post("/webhook/test")
async def test_webhook(request: Request):
    """
    Test endpoint for manual webhook testing.
    
    Accepts any JSON payload and logs it for debugging purposes.
    Does not perform signature validation or trigger journey processing.
    
    Args:
        request: The incoming HTTP request
        
    Returns:
        dict: Test response with received payload
    """
    try:
        payload = await request.json()
        logger.info(f"Test webhook received: {payload}")
        
        return {
            "status": "success",
            "message": "Test webhook received successfully",
            "received_payload": payload,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error processing test webhook: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid payload: {str(e)}"
        )


@app.post("/webhook/freshdesk", response_model=WebhookResponse)
async def handle_freshdesk_webhook(request: Request):
    """
    Main webhook endpoint for receiving Freshdesk events.
    
    This endpoint:
    1. Receives webhook POST requests from Freshdesk
    2. Validates the webhook signature
    3. Parses the payload
    4. Routes to appropriate journey (to be implemented in task 5)
    
    Args:
        request: The incoming HTTP request with webhook payload
        
    Returns:
        WebhookResponse: Success or error response
        
    Raises:
        HTTPException: For validation errors or processing failures
    """
    start_time = datetime.utcnow()
    ticket_id = None  # Initialize ticket_id for error handling
    
    try:
        # Check rate limit
        client_ip = request.client.host if request.client else "unknown"
        
        if not check_rate_limit(client_ip):
            logger.warning(
                "Rate limit exceeded",
                extra={
                    "source_ip": client_ip,
                    "limit": webhook_config.rate_limit,
                    "window_seconds": webhook_config.rate_limit_window
                }
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {webhook_config.rate_limit} requests per {webhook_config.rate_limit_window} seconds"
            )
        
        # Parse the JSON payload first to get ticket_id
        payload_dict = await request.json()
        
        # Freshdesk wraps the payload in a "freshdesk_webhook" object
        if "freshdesk_webhook" in payload_dict:
            payload_dict = payload_dict["freshdesk_webhook"]
        
        # Extract ticket_id early for logging
        ticket_id = str(payload_dict.get("ticket_id", "unknown"))
        
        # Get raw body for signature validation
        raw_body = await request.body()
        
        # Extract signature header
        signature = request.headers.get("X-Freshdesk-Signature")
        
        # Get webhook secret from configuration
        webhook_secret = webhook_config.secret
        
        # Validate signature if secret is configured and not disabled
        if webhook_secret and webhook_secret != "disabled-for-testing":
            is_valid = validate_freshdesk_signature(
                payload=raw_body,
                signature=signature,
                secret=webhook_secret
            )
            
            # Log signature validation result
            log_signature_validation(logger, ticket_id, is_valid, client_ip)
            
            if not is_valid:
                # Record validation failure
                metrics.record_validation_failure()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid webhook signature"
                )
        else:
            logger.warning(
                "Webhook signature validation disabled - skipping validation",
                extra={
                    "ticket_id": ticket_id,
                    "source_ip": client_ip
                }
            )
        
        # Validate payload structure using Pydantic
        try:
            payload = FreshdeskWebhookPayload(**payload_dict)
        except Exception as validation_error:
            log_error_with_context(
                logger,
                validation_error,
                ticket_id=ticket_id,
                context={"error_type": "payload_validation"}
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid payload structure: {str(validation_error)}"
            )
        
        # Extract key information (ticket_id already extracted above)
        # Default to "ticket_created" if event type is not provided (e.g., test webhooks)
        event_type = payload.event or "ticket_created"
        
        # Log webhook received with structured fields
        log_webhook_received(logger, ticket_id, event_type, client_ip)
        
        # Task 6.1: Filter unsupported event types
        if not is_supported_event_type(event_type):
            logger.info(
                "Unsupported event type ignored",
                extra={
                    "ticket_id": ticket_id,
                    "event_type": event_type,
                    "supported_types": list(SUPPORTED_EVENT_TYPES)
                }
            )
            return WebhookResponse(
                status="success",
                message=f"Event type '{event_type}' ignored (not supported)",
                ticket_id=ticket_id,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        
        # Task 6.2: Check for duplicate events
        if is_duplicate_event(ticket_id, event_type):
            logger.info(
                "Duplicate event detected",
                extra={
                    "ticket_id": ticket_id,
                    "event_type": event_type,
                    "deduplication_window_seconds": webhook_config.deduplication_window
                }
            )
            return WebhookResponse(
                status="success",
                message=f"Duplicate event ignored (processed within {webhook_config.deduplication_window}s)",
                ticket_id=ticket_id,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        
        # Task 6.3: Check if update is refund-related
        # Add the event_type to payload_dict for is_refund_related check
        payload_dict_with_event = {**payload_dict, "event": event_type}
        if not is_refund_related(payload_dict_with_event):
            logger.info(
                "Non-refund-related update ignored",
                extra={
                    "ticket_id": ticket_id,
                    "event_type": event_type
                }
            )
            return WebhookResponse(
                status="success",
                message="Non-refund-related update ignored",
                ticket_id=ticket_id,
                processing_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000)
            )
        
        logger.info(
            "Webhook passed all filters",
            extra={
                "ticket_id": ticket_id,
                "event_type": event_type
            }
        )
        
        # Route to appropriate journey based on trigger source
        journey_name = route_to_journey(
            trigger_source="webhook",
            ticket_id=ticket_id
        )
        
        # Log routing decision with structured fields
        log_routing_decision(logger, ticket_id, "webhook", journey_name)
        
        # Activate the selected journey with Parlant agent
        try:
            activation_result = await activate_journey(
                ticket_id=ticket_id,
                journey_name=journey_name,
                payload=payload_dict
            )
            
            if activation_result.get("success"):
                log_journey_activation(
                    logger,
                    ticket_id,
                    journey_name,
                    activation_result.get("session_id", "unknown")
                )
            else:
                error_detail = activation_result.get("error", "Unknown error")
                logger.error(
                    "Journey activation failed",
                    extra={
                        "ticket_id": ticket_id,
                        "journey_name": journey_name,
                        "error": error_detail
                    }
                )
                # Error already logged above
        except Exception as e:
            logger.error(
                "Journey activation exception",
                extra={
                    "ticket_id": ticket_id,
                    "journey_name": journey_name,
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            )
        
        # Calculate processing time
        processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        
        # Record successful webhook processing
        metrics.record_webhook_success(processing_time)
        
        logger.info(
            "Webhook processed successfully",
            extra={
                "ticket_id": ticket_id,
                "processing_time_ms": processing_time
            }
        )
        
        # Log performance warning if processing took too long
        if processing_time > 15000:
            metrics.record_slow_operation()
            log_performance_warning(
                logger,
                operation="webhook_processing",
                duration_ms=processing_time,
                threshold_ms=15000,
                ticket_id=ticket_id
            )
        
        return WebhookResponse(
            status="success",
            message=f"Webhook received and queued for processing",
            ticket_id=ticket_id,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        # Record webhook failure
        metrics.record_webhook_failure()
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Record webhook failure and error
        metrics.record_webhook_failure()
        metrics.record_error("webhook_processing_error")
        
        # Log error with full context
        log_error_with_context(
            logger,
            e,
            ticket_id=ticket_id if 'ticket_id' in locals() else None,
            context={"error_type": "webhook_processing"}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Get port from configuration
    port = webhook_config.port
    
    logger.info(f"Starting webhook server on port {port}")
    
    # Check if webhooks are enabled
    if not webhook_config.enabled:
        logger.warning("Webhooks are disabled in configuration (WEBHOOK_ENABLED=false)")
        logger.warning("Server will start but will not process webhooks")
    
    uvicorn.run(app, host="0.0.0.0", port=port)
