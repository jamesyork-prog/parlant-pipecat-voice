"""
End-to-end tracing system for async webhook orchestration.

This module provides comprehensive tracing capabilities for:
- Trace ID generation and propagation
- Tracing through webhook handler to worker completion
- Trace correlation with Freshdesk updates
- Distributed tracing across components

Implements requirements:
- 5.5: End-to-end tracing from webhook to Freshdesk update
"""

import uuid
import time
import logging
import json
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextvars import ContextVar
from contextlib import contextmanager

from .prometheus_metrics import get_webhook_metrics_collector

logger = logging.getLogger(__name__)


class SpanType(Enum):
    """Types of trace spans."""
    WEBHOOK = "webhook"
    WORKER = "worker"
    ROUTING = "routing"
    DECISION = "decision"
    LLM_CALL = "llm_call"
    DATABASE = "database"
    FRESHDESK = "freshdesk"
    IDEMPOTENCY = "idempotency"
    QUEUE = "queue"


class SpanStatus(Enum):
    """Status of a trace span."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class TraceContext:
    """Context information for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceContext':
        """Create from dictionary."""
        return cls(**data)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id
        }
        
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        
        if self.baggage:
            headers["X-Trace-Baggage"] = json.dumps(self.baggage)
        
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create from HTTP headers."""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        parent_span_id = headers.get("X-Parent-Span-Id")
        
        baggage = {}
        baggage_header = headers.get("X-Trace-Baggage")
        if baggage_header:
            try:
                baggage = json.loads(baggage_header)
            except json.JSONDecodeError:
                logger.warning(f"Invalid trace baggage header: {baggage_header}")
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            baggage=baggage
        )


@dataclass
class Span:
    """A trace span representing a unit of work."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    span_type: SpanType
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """Finish the span with optional status."""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
            self.status = status
    
    def set_tag(self, key: str, value: Union[str, int, float, bool]):
        """Set a tag on the span."""
        self.tags[key] = value
    
    def log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "span_type": self.span_type.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "logs": self.logs
        }


class TraceCollector:
    """
    Collects and manages trace spans for the async webhook orchestration system.
    
    Provides functionality for:
    - Span creation and management
    - Trace correlation across components
    - Metrics emission for tracing
    - Trace export for monitoring systems
    """
    
    def __init__(self):
        """Initialize the trace collector."""
        self.active_spans: Dict[str, Span] = {}
        self.completed_spans: List[Span] = []
        self.max_completed_spans = 1000  # Keep last 1000 completed spans
        self.metrics_collector = get_webhook_metrics_collector()
        
        logger.info("TraceCollector initialized")
    
    def create_trace_id(self) -> str:
        """Generate a new trace ID."""
        return str(uuid.uuid4())
    
    def create_span_id(self) -> str:
        """Generate a new span ID."""
        return str(uuid.uuid4())[:16]  # Shorter span IDs
    
    def start_span(
        self,
        operation_name: str,
        span_type: SpanType,
        trace_context: Optional[TraceContext] = None,
        parent_span_id: Optional[str] = None
    ) -> Span:
        """
        Start a new trace span.
        
        Args:
            operation_name: Name of the operation being traced
            span_type: Type of span (webhook, worker, etc.)
            trace_context: Optional existing trace context
            parent_span_id: Optional parent span ID
            
        Returns:
            New Span instance
        """
        # Generate IDs
        if trace_context:
            trace_id = trace_context.trace_id
            parent_span_id = parent_span_id or trace_context.span_id
        else:
            trace_id = self.create_trace_id()
        
        span_id = self.create_span_id()
        
        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            span_type=span_type,
            start_time=time.time()
        )
        
        # Store active span
        self.active_spans[span_id] = span
        
        # Record metrics
        self.metrics_collector.record_trace_span(span_type.value, operation_name)
        
        logger.debug(f"Started span {span_id} for operation {operation_name}")
        return span
    
    def finish_span(self, span_id: str, status: SpanStatus = SpanStatus.OK):
        """
        Finish a trace span.
        
        Args:
            span_id: ID of the span to finish
            status: Final status of the span
        """
        span = self.active_spans.get(span_id)
        if not span:
            logger.warning(f"Attempted to finish unknown span: {span_id}")
            return
        
        # Finish the span
        span.finish(status)
        
        # Move to completed spans
        self.completed_spans.append(span)
        del self.active_spans[span_id]
        
        # Trim completed spans if needed
        if len(self.completed_spans) > self.max_completed_spans:
            self.completed_spans = self.completed_spans[-self.max_completed_spans:]
        
        # Record trace duration metrics if this is a root span
        if not span.parent_span_id:
            trace_type = f"{span.span_type.value}_to_completion"
            self.metrics_collector.record_trace_duration(trace_type, span.duration_ms / 1000)
        
        logger.debug(f"Finished span {span_id} with status {status.value}")
    
    def get_span(self, span_id: str) -> Optional[Span]:
        """Get an active span by ID."""
        return self.active_spans.get(span_id)
    
    def correlate_spans(
        self,
        from_span_id: str,
        to_span_id: str,
        correlation_type: str = "follows"
    ) -> bool:
        """
        Correlate two spans to show their relationship.
        
        Args:
            from_span_id: Source span ID
            to_span_id: Target span ID
            correlation_type: Type of correlation (follows, triggers, etc.)
            
        Returns:
            True if correlation was successful
        """
        from_span = self.active_spans.get(from_span_id)
        to_span = self.active_spans.get(to_span_id)
        
        if not from_span or not to_span:
            # Check completed spans too
            all_spans = {**self.active_spans}
            for span in self.completed_spans:
                all_spans[span.span_id] = span
            
            from_span = all_spans.get(from_span_id)
            to_span = all_spans.get(to_span_id)
        
        if not from_span or not to_span:
            logger.warning(f"Cannot correlate spans {from_span_id} -> {to_span_id}: spans not found")
            self.metrics_collector.record_trace_correlation(
                from_component="unknown",
                to_component="unknown",
                success=False,
                failure_reason="spans_not_found"
            )
            return False
        
        # Add correlation tags
        from_span.set_tag(f"correlation.{correlation_type}.to", to_span_id)
        to_span.set_tag(f"correlation.{correlation_type}.from", from_span_id)
        
        # Record successful correlation
        self.metrics_collector.record_trace_correlation(
            from_component=from_span.span_type.value,
            to_component=to_span.span_type.value,
            success=True
        )
        
        logger.debug(f"Correlated spans {from_span_id} -> {to_span_id} ({correlation_type})")
        return True
    
    def get_trace_spans(self, trace_id: str) -> List[Span]:
        """
        Get all spans for a specific trace.
        
        Args:
            trace_id: Trace ID to search for
            
        Returns:
            List of spans belonging to the trace
        """
        spans = []
        
        # Check active spans
        for span in self.active_spans.values():
            if span.trace_id == trace_id:
                spans.append(span)
        
        # Check completed spans
        for span in self.completed_spans:
            if span.trace_id == trace_id:
                spans.append(span)
        
        # Sort by start time
        spans.sort(key=lambda s: s.start_time)
        return spans
    
    def export_trace(self, trace_id: str) -> Dict[str, Any]:
        """
        Export a complete trace for external systems.
        
        Args:
            trace_id: Trace ID to export
            
        Returns:
            Dictionary containing trace data
        """
        spans = self.get_trace_spans(trace_id)
        
        if not spans:
            return {"error": f"No spans found for trace {trace_id}"}
        
        # Calculate trace duration
        start_time = min(span.start_time for span in spans)
        end_times = [span.end_time for span in spans if span.end_time]
        end_time = max(end_times) if end_times else None
        
        trace_duration_ms = None
        if end_time:
            trace_duration_ms = (end_time - start_time) * 1000
        
        return {
            "trace_id": trace_id,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": trace_duration_ms,
            "span_count": len(spans),
            "spans": [span.to_dict() for span in spans],
            "status": "completed" if end_time else "active"
        }
    
    def get_active_traces(self) -> List[str]:
        """Get list of active trace IDs."""
        trace_ids = set()
        for span in self.active_spans.values():
            trace_ids.add(span.trace_id)
        return list(trace_ids)
    
    def get_trace_summary(self) -> Dict[str, Any]:
        """Get summary of tracing activity."""
        active_trace_count = len(self.get_active_traces())
        completed_trace_count = len(set(span.trace_id for span in self.completed_spans))
        
        return {
            "active_spans": len(self.active_spans),
            "completed_spans": len(self.completed_spans),
            "active_traces": active_trace_count,
            "completed_traces": completed_trace_count,
            "total_spans_processed": len(self.completed_spans) + len(self.active_spans)
        }
    
    def clear_all_spans(self):
        """
        Clear all active and completed spans.
        
        This method is primarily intended for testing to ensure clean state
        between test runs.
        """
        self.active_spans.clear()
        self.completed_spans.clear()
        logger.debug("Cleared all spans from trace collector")


# Global trace collector instance
_trace_collector: Optional[TraceCollector] = None

# Context variable for current trace context
_current_trace_context: ContextVar[Optional[TraceContext]] = ContextVar(
    'current_trace_context', 
    default=None
)


def get_trace_collector() -> TraceCollector:
    """
    Get the global trace collector instance.
    
    Returns:
        The global TraceCollector instance
    """
    global _trace_collector
    if _trace_collector is None:
        _trace_collector = TraceCollector()
    return _trace_collector


def get_current_trace_context() -> Optional[TraceContext]:
    """Get the current trace context from context variables."""
    return _current_trace_context.get()


def set_current_trace_context(context: Optional[TraceContext]):
    """Set the current trace context in context variables."""
    _current_trace_context.set(context)


@contextmanager
def trace_span(
    operation_name: str,
    span_type: SpanType,
    trace_context: Optional[TraceContext] = None,
    **tags
):
    """
    Context manager for creating and managing trace spans.
    
    Args:
        operation_name: Name of the operation being traced
        span_type: Type of span
        trace_context: Optional trace context
        **tags: Additional tags to set on the span
        
    Yields:
        Span instance
    """
    collector = get_trace_collector()
    
    # Use current trace context if none provided
    if trace_context is None:
        trace_context = get_current_trace_context()
    
    # Start span
    span = collector.start_span(operation_name, span_type, trace_context)
    
    # Set tags
    for key, value in tags.items():
        span.set_tag(key, value)
    
    # Create new trace context for this span
    new_context = TraceContext(
        trace_id=span.trace_id,
        span_id=span.span_id,
        parent_span_id=span.parent_span_id
    )
    
    # Set as current context
    token = _current_trace_context.set(new_context)
    
    try:
        yield span
        collector.finish_span(span.span_id, SpanStatus.OK)
    except Exception as e:
        span.log(f"Error: {str(e)}", level="error")
        span.set_tag("error", True)
        span.set_tag("error.message", str(e))
        collector.finish_span(span.span_id, SpanStatus.ERROR)
        raise
    finally:
        # Restore previous context
        _current_trace_context.reset(token)


def create_trace_context_from_headers(headers: Dict[str, str]) -> Optional[TraceContext]:
    """
    Create trace context from HTTP headers.
    
    Args:
        headers: HTTP headers dictionary
        
    Returns:
        TraceContext if headers contain trace information, None otherwise
    """
    return TraceContext.from_headers(headers)


def inject_trace_context_into_headers(
    headers: Dict[str, str],
    trace_context: Optional[TraceContext] = None
) -> Dict[str, str]:
    """
    Inject trace context into HTTP headers.
    
    Args:
        headers: Existing headers dictionary
        trace_context: Trace context to inject (uses current if None)
        
    Returns:
        Headers dictionary with trace information
    """
    if trace_context is None:
        trace_context = get_current_trace_context()
    
    if trace_context:
        headers.update(trace_context.to_headers())
    
    return headers


def correlate_webhook_to_worker(webhook_span_id: str, worker_span_id: str) -> bool:
    """
    Correlate webhook processing to worker processing.
    
    Args:
        webhook_span_id: Webhook handler span ID
        worker_span_id: Worker processing span ID
        
    Returns:
        True if correlation was successful
    """
    collector = get_trace_collector()
    return collector.correlate_spans(webhook_span_id, worker_span_id, "triggers")


def correlate_worker_to_freshdesk(worker_span_id: str, freshdesk_span_id: str) -> bool:
    """
    Correlate worker processing to Freshdesk update.
    
    Args:
        worker_span_id: Worker processing span ID
        freshdesk_span_id: Freshdesk update span ID
        
    Returns:
        True if correlation was successful
    """
    collector = get_trace_collector()
    return collector.correlate_spans(worker_span_id, freshdesk_span_id, "updates")


def get_trace_for_ticket(ticket_id: str) -> Optional[Dict[str, Any]]:
    """
    Get trace information for a specific ticket.
    
    Args:
        ticket_id: Freshdesk ticket ID
        
    Returns:
        Trace data if found, None otherwise
    """
    collector = get_trace_collector()
    
    # Search for spans with ticket_id tag
    for span in list(collector.active_spans.values()) + collector.completed_spans:
        if span.tags.get("ticket_id") == ticket_id:
            return collector.export_trace(span.trace_id)
    
    return None


def export_traces_for_monitoring() -> List[Dict[str, Any]]:
    """
    Export all completed traces for monitoring systems.
    
    Returns:
        List of trace data dictionaries
    """
    collector = get_trace_collector()
    
    # Get all unique trace IDs from completed spans
    trace_ids = set(span.trace_id for span in collector.completed_spans)
    
    traces = []
    for trace_id in trace_ids:
        trace_data = collector.export_trace(trace_id)
        if "error" not in trace_data:
            traces.append(trace_data)
    
    return traces