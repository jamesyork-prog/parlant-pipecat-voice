"""
Fast webhook handler for async webhook orchestration.

This module provides the WebhookHandler class that implements:
- HMAC signature verification for Freshdesk webhooks
- Timestamp validation to prevent replay attacks
- Fast rejection paths for invalid requests
- Sub-50ms response times at 99th percentile
"""

import hmac
import hashlib
import time
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from .prometheus_metrics import get_webhook_metrics_collector
from .tracing import trace_span, SpanType, get_current_trace_context, set_current_trace_context, TraceContext

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of webhook validation."""
    VALID = "valid"
    INVALID_SIGNATURE = "invalid_signature"
    MISSING_SIGNATURE = "missing_signature"
    INVALID_TIMESTAMP = "invalid_timestamp"
    MALFORMED_PAYLOAD = "malformed_payload"
    REPLAY_ATTACK = "replay_attack"


@dataclass
class WebhookRequest:
    """Webhook request data structure."""
    headers: Dict[str, str]
    body: bytes
    signature: Optional[str]
    timestamp: datetime
    
    def payload_hash(self) -> str:
        """Generate SHA-256 hash of body for idempotency."""
        return hashlib.sha256(self.body).hexdigest()


@dataclass
class WebhookResponse:
    """Webhook response data structure."""
    status_code: int
    message: str
    task_id: Optional[str] = None
    processing_time_ms: Optional[int] = None


class WebhookHandler:
    """
    Fast webhook handler with signature validation and replay protection.
    
    Implements requirements:
    - 1.3: Invalid signature rejection within 10ms
    - 1.4: Malformed payload rejection within 10ms  
    - 1.5: No LLM calls during request handling
    """
    
    def __init__(self, webhook_secret: str, max_timestamp_age: int = 300):
        """
        Initialize webhook handler.
        
        Args:
            webhook_secret: HMAC secret for signature validation
            max_timestamp_age: Maximum age of timestamp in seconds (default: 5 minutes)
        """
        self.webhook_secret = webhook_secret.encode('utf-8')
        self.max_timestamp_age = max_timestamp_age
        
    def validate_signature(self, payload: bytes, signature: Optional[str]) -> Tuple[bool, float]:
        """
        Verify Freshdesk webhook signature using HMAC-SHA256.
        
        Implements fast rejection path for invalid signatures.
        
        Args:
            payload: Raw request body as bytes
            signature: X-Freshdesk-Signature header value
            
        Returns:
            Tuple of (is_valid, validation_time_ms)
        """
        start_time = time.perf_counter()
        
        # Fast rejection for missing signature
        if signature is None or signature == "":
            validation_time = (time.perf_counter() - start_time) * 1000
            return False, validation_time
        
        try:
            # Compute expected signature using HMAC-SHA256
            expected_signature = hmac.new(
                key=self.webhook_secret,
                msg=payload,
                digestmod=hashlib.sha256
            ).hexdigest()
            
            # Constant-time comparison to prevent timing attacks
            is_valid = hmac.compare_digest(signature, expected_signature)
            
            validation_time = (time.perf_counter() - start_time) * 1000
            return is_valid, validation_time
            
        except Exception as e:
            validation_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error validating webhook signature: {e}")
            return False, validation_time
    
    def validate_timestamp(self, timestamp_header: Optional[str]) -> Tuple[bool, float]:
        """
        Validate webhook timestamp to prevent replay attacks.
        
        Args:
            timestamp_header: X-Freshdesk-Timestamp header value
            
        Returns:
            Tuple of (is_valid, validation_time_ms)
        """
        start_time = time.perf_counter()
        
        # Fast rejection for missing timestamp
        if timestamp_header is None or timestamp_header == "":
            validation_time = (time.perf_counter() - start_time) * 1000
            return False, validation_time
        
        try:
            # Parse timestamp (assuming Unix timestamp)
            webhook_timestamp = float(timestamp_header)
            current_timestamp = time.time()
            
            # Check if timestamp is within acceptable range
            age = current_timestamp - webhook_timestamp
            is_valid = 0 <= age <= self.max_timestamp_age
            
            validation_time = (time.perf_counter() - start_time) * 1000
            return is_valid, validation_time
            
        except (ValueError, TypeError) as e:
            validation_time = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Invalid timestamp format: {timestamp_header}, error: {e}")
            return False, validation_time
    
    def validate_payload_structure(self, payload: bytes) -> Tuple[ValidationResult, float, Optional[Dict[str, Any]]]:
        """
        Validate payload structure and parse JSON.
        
        Implements fast rejection path for malformed payloads.
        
        Args:
            payload: Raw request body as bytes
            
        Returns:
            Tuple of (validation_result, validation_time_ms, parsed_payload)
        """
        start_time = time.perf_counter()
        
        try:
            import json
            
            # Fast rejection for empty payload
            if not payload:
                validation_time = (time.perf_counter() - start_time) * 1000
                return ValidationResult.MALFORMED_PAYLOAD, validation_time, None
            
            # Parse JSON payload
            parsed_payload = json.loads(payload.decode('utf-8'))
            
            # Basic structure validation - must have ticket_id
            if not isinstance(parsed_payload, dict):
                validation_time = (time.perf_counter() - start_time) * 1000
                return ValidationResult.MALFORMED_PAYLOAD, validation_time, None
            
            # Handle Freshdesk wrapper
            if "freshdesk_webhook" in parsed_payload:
                parsed_payload = parsed_payload["freshdesk_webhook"]
            
            # Validate required fields
            if "ticket_id" not in parsed_payload:
                validation_time = (time.perf_counter() - start_time) * 1000
                return ValidationResult.MALFORMED_PAYLOAD, validation_time, None
            
            validation_time = (time.perf_counter() - start_time) * 1000
            return ValidationResult.VALID, validation_time, parsed_payload
            
        except json.JSONDecodeError as e:
            validation_time = (time.perf_counter() - start_time) * 1000
            logger.warning(f"Invalid JSON payload: {e}")
            return ValidationResult.MALFORMED_PAYLOAD, validation_time, None
        except Exception as e:
            validation_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error validating payload structure: {e}")
            return ValidationResult.MALFORMED_PAYLOAD, validation_time, None
    
    def handle_webhook(self, request: WebhookRequest) -> WebhookResponse:
        """
        Process webhook with fast validation and rejection paths.
        
        Implements requirements:
        - 1.1: <50ms response time at 99th percentile
        - 1.3: Invalid signature rejection within 10ms
        - 1.4: Malformed payload rejection within 10ms
        - 1.5: No LLM calls during request handling
        - 5.5: End-to-end tracing from webhook to completion
        
        Args:
            request: WebhookRequest containing headers, body, and metadata
            
        Returns:
            WebhookResponse with status and timing information
        """
        # Extract or create trace context from headers
        trace_context = TraceContext.from_headers(request.headers)
        if not trace_context:
            # Create new trace for this webhook
            trace_context = TraceContext(
                trace_id=f"webhook-{int(time.time() * 1000)}-{hash(request.body) % 10000:04d}",
                span_id=f"webhook-{int(time.time() * 1000000) % 1000000:06d}"
            )
        
        # Start webhook processing span
        with trace_span(
            "webhook_processing",
            SpanType.WEBHOOK,
            trace_context,
            component="webhook_handler",
            http_method="POST",
            content_length=len(request.body)
        ) as span:
            start_time = time.perf_counter()
            metrics_collector = get_webhook_metrics_collector()
            validation_times = {}
            
            # Extract ticket ID for tracing if possible
            try:
                import json
                payload_data = json.loads(request.body.decode('utf-8'))
                ticket_id = payload_data.get("ticket_id")
                if ticket_id:
                    span.set_tag("ticket_id", str(ticket_id))
            except:
                pass  # Ignore JSON parsing errors for tracing
            
            # Fast path 1: Signature validation
            with trace_span("signature_validation", SpanType.WEBHOOK, signature_length=len(request.signature or "")) as sig_span:
                signature_valid, signature_time = self.validate_signature(
                    request.body, 
                    request.signature
                )
                sig_span.set_tag("validation_result", signature_valid)
                sig_span.set_tag("validation_time_ms", signature_time)
            
            validation_times["signature"] = signature_time
            
            if not signature_valid:
                total_time = (time.perf_counter() - start_time) * 1000
                
                # Add error information to span
                span.set_tag("error", True)
                span.set_tag("error.type", "invalid_signature")
                span.set_tag("http.status_code", 401)
                span.log("Invalid webhook signature rejected", level="warning")
                
                # Record metrics for signature failure
                metrics_collector.record_webhook_request(
                    status_code=401,
                    validation_result="invalid_signature",
                    response_time_ms=total_time,
                    validation_times=validation_times
                )
                
                logger.warning(
                    "Invalid webhook signature rejected",
                    extra={
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "validation_time_ms": signature_time,
                        "total_time_ms": total_time
                    }
                )
                return WebhookResponse(
                    status_code=401,
                    message="Invalid webhook signature",
                    processing_time_ms=int(total_time)
                )
            
            # Fast path 2: Timestamp validation (replay attack prevention)
            timestamp_header = request.headers.get("X-Freshdesk-Timestamp")
            with trace_span("timestamp_validation", SpanType.WEBHOOK, timestamp=timestamp_header) as ts_span:
                timestamp_valid, timestamp_time = self.validate_timestamp(timestamp_header)
                ts_span.set_tag("validation_result", timestamp_valid)
                ts_span.set_tag("validation_time_ms", timestamp_time)
            
            validation_times["timestamp"] = timestamp_time
            
            if not timestamp_valid:
                total_time = (time.perf_counter() - start_time) * 1000
                
                # Add error information to span
                span.set_tag("error", True)
                span.set_tag("error.type", "invalid_timestamp")
                span.set_tag("http.status_code", 400)
                span.log("Invalid timestamp rejected (potential replay attack)", level="warning")
                
                # Record metrics for timestamp failure
                metrics_collector.record_webhook_request(
                    status_code=400,
                    validation_result="invalid_timestamp",
                    response_time_ms=total_time,
                    validation_times=validation_times
                )
                
                logger.warning(
                    "Invalid timestamp rejected (potential replay attack)",
                    extra={
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "timestamp_header": timestamp_header,
                        "validation_time_ms": timestamp_time,
                        "total_time_ms": total_time
                    }
                )
                return WebhookResponse(
                    status_code=400,
                    message="Invalid or expired timestamp",
                    processing_time_ms=int(total_time)
                )
            
            # Fast path 3: Payload structure validation
            with trace_span("payload_validation", SpanType.WEBHOOK, payload_size=len(request.body)) as payload_span:
                payload_result, payload_time, parsed_payload = self.validate_payload_structure(request.body)
                payload_span.set_tag("validation_result", payload_result.value)
                payload_span.set_tag("validation_time_ms", payload_time)
                
                if parsed_payload:
                    payload_span.set_tag("ticket_id", parsed_payload.get("ticket_id"))
            
            validation_times["payload"] = payload_time
            
            if payload_result != ValidationResult.VALID:
                total_time = (time.perf_counter() - start_time) * 1000
                
                # Add error information to span
                span.set_tag("error", True)
                span.set_tag("error.type", "malformed_payload")
                span.set_tag("http.status_code", 400)
                span.log("Malformed payload rejected", level="warning")
                
                # Record metrics for payload failure
                metrics_collector.record_webhook_request(
                    status_code=400,
                    validation_result="malformed_payload",
                    response_time_ms=total_time,
                    validation_times=validation_times
                )
                
                logger.warning(
                    "Malformed payload rejected",
                    extra={
                        "trace_id": span.trace_id,
                        "span_id": span.span_id,
                        "validation_result": payload_result.value,
                        "validation_time_ms": payload_time,
                        "total_time_ms": total_time
                    }
                )
                return WebhookResponse(
                    status_code=400,
                    message="Malformed payload structure",
                    processing_time_ms=int(total_time)
                )
            
            # All validations passed - webhook is valid
            total_time = (time.perf_counter() - start_time) * 1000
            
            # Add success information to span
            span.set_tag("http.status_code", 202)
            span.set_tag("processing_time_ms", total_time)
            span.set_tag("validation.signature_time_ms", signature_time)
            span.set_tag("validation.timestamp_time_ms", timestamp_time)
            span.set_tag("validation.payload_time_ms", payload_time)
            span.log("Webhook validation successful", level="info")
            
            # Record metrics for successful validation
            metrics_collector.record_webhook_request(
                status_code=202,
                validation_result="valid",
                response_time_ms=total_time,
                validation_times=validation_times
            )
            
            logger.info(
                "Webhook validation successful",
                extra={
                    "trace_id": span.trace_id,
                    "span_id": span.span_id,
                    "ticket_id": parsed_payload.get("ticket_id") if parsed_payload else None,
                    "signature_time_ms": signature_time,
                    "timestamp_time_ms": timestamp_time,
                    "payload_time_ms": payload_time,
                    "total_time_ms": total_time
                }
            )
            
            # Return success response with trace information
            response = WebhookResponse(
                status_code=202,
                message="Webhook validated successfully",
                processing_time_ms=int(total_time)
            )
            
            # Add trace ID to response for correlation
            response.task_id = f"trace-{span.trace_id}"
            
            return response