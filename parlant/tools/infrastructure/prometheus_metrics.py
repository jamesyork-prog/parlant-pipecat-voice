"""
Prometheus metrics collection system for async webhook orchestration.

This module provides comprehensive metrics collection for:
- Webhook handler performance (response times, success rates, error types)
- Worker processing metrics (task completion tracking, processing times)
- Queue depth and system health metrics
- End-to-end tracing support

Implements requirements:
- 5.1: Webhook request metrics (response time, success rate, error types)
- 5.2: Worker processing metrics (processing time, retry counts, completion status)
"""

import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from prometheus_client import (
    CollectorRegistry, Gauge, Counter, Histogram, Info, 
    generate_latest, start_http_server
)
from dataclasses import dataclass
from enum import Enum

from .metrics import get_metrics_tracker, MetricsTracker

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics for categorization."""
    WEBHOOK = "webhook"
    WORKER = "worker"
    QUEUE = "queue"
    SYSTEM = "system"
    TRACING = "tracing"


@dataclass
class MetricLabels:
    """Standard metric labels for consistency."""
    component: str = "async-webhook-orchestration"
    version: str = "1.0.0"
    environment: str = "production"


class WebhookMetricsCollector:
    """
    Prometheus metrics collector for async webhook orchestration.
    
    Implements comprehensive metrics collection for webhook processing,
    worker operations, and system health monitoring.
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize the metrics collector.
        
        Args:
            registry: Optional Prometheus registry. Creates new one if None.
        """
        self.registry = registry or CollectorRegistry()
        self.metrics_tracker = get_metrics_tracker()
        self.labels = MetricLabels()
        
        # Initialize all Prometheus metrics
        self._setup_webhook_metrics()
        self._setup_worker_metrics()
        self._setup_queue_metrics()
        self._setup_system_metrics()
        self._setup_tracing_metrics()
        
        logger.info("WebhookMetricsCollector initialized with Prometheus metrics")
    
    def _setup_webhook_metrics(self):
        """Set up webhook handler performance metrics."""
        
        # Webhook request metrics
        self.webhook_requests_total = Counter(
            'webhook_requests_total',
            'Total number of webhook requests received',
            ['status_code', 'validation_result'],
            registry=self.registry
        )
        
        self.webhook_response_time_ms = Histogram(
            'webhook_response_time_ms',
            'Webhook response time in milliseconds',
            buckets=[1, 5, 10, 25, 50, 75, 100, 150, 200, 300, 500],
            registry=self.registry
        )
        
        self.webhook_validation_time_ms = Histogram(
            'webhook_validation_time_ms',
            'Webhook validation time in milliseconds',
            ['validation_type'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 15, 20, 30],
            registry=self.registry
        )
        
        # Webhook error metrics
        self.webhook_signature_failures_total = Counter(
            'webhook_signature_failures_total',
            'Total number of webhook signature validation failures',
            registry=self.registry
        )
        
        self.webhook_timestamp_failures_total = Counter(
            'webhook_timestamp_failures_total',
            'Total number of webhook timestamp validation failures',
            registry=self.registry
        )
        
        self.webhook_payload_failures_total = Counter(
            'webhook_payload_failures_total',
            'Total number of webhook payload validation failures',
            registry=self.registry
        )
        
        # Idempotency metrics
        self.idempotency_hits_total = Counter(
            'idempotency_hits_total',
            'Total number of duplicate webhook requests detected',
            registry=self.registry
        )
        
        self.idempotency_store_errors_total = Counter(
            'idempotency_store_errors_total',
            'Total number of idempotency store errors',
            ['error_type'],
            registry=self.registry
        )
        
        # Performance percentiles (gauges updated from histograms)
        self.webhook_response_time_p50 = Gauge(
            'webhook_response_time_ms_p50',
            'Webhook response time 50th percentile',
            registry=self.registry
        )
        
        self.webhook_response_time_p95 = Gauge(
            'webhook_response_time_ms_p95',
            'Webhook response time 95th percentile',
            registry=self.registry
        )
        
        self.webhook_response_time_p99 = Gauge(
            'webhook_response_time_ms_p99',
            'Webhook response time 99th percentile',
            registry=self.registry
        )
    
    def _setup_worker_metrics(self):
        """Set up worker processing metrics."""
        
        # Task processing metrics
        self.task_processing_time_seconds = Histogram(
            'task_processing_time_seconds',
            'Task processing time in seconds',
            ['processing_path', 'decision_type'],
            buckets=[1, 5, 10, 30, 60, 120, 180, 240, 300],
            registry=self.registry
        )
        
        self.task_completions_total = Counter(
            'task_completions_total',
            'Total number of completed tasks',
            ['status', 'processing_path', 'decision_type'],
            registry=self.registry
        )
        
        self.task_retries_total = Counter(
            'task_retries_total',
            'Total number of task retry attempts',
            ['error_type', 'retry_count'],
            registry=self.registry
        )
        
        # Worker health metrics
        self.active_workers = Gauge(
            'active_workers',
            'Number of active worker processes',
            registry=self.registry
        )
        
        self.worker_processed_count = Counter(
            'worker_processed_count_total',
            'Total tasks processed by each worker',
            ['worker_id'],
            registry=self.registry
        )
        
        self.worker_error_count = Counter(
            'worker_error_count_total',
            'Total errors encountered by each worker',
            ['worker_id', 'error_type'],
            registry=self.registry
        )
        
        # Load balancing metrics
        self.worker_registrations = Counter(
            'worker_registrations_total',
            'Total worker registrations',
            ['worker_id'],
            registry=self.registry
        )
        
        self.worker_unregistrations = Counter(
            'worker_unregistrations_total',
            'Total worker unregistrations',
            ['worker_id'],
            registry=self.registry
        )
        
        self.worker_capacity = Gauge(
            'worker_capacity',
            'Worker processing capacity',
            ['worker_id'],
            registry=self.registry
        )
        
        self.worker_load_factor = Gauge(
            'worker_load_factor',
            'Current worker load factor (0.0 to 1.0)',
            ['worker_id'],
            registry=self.registry
        )
        
        self.task_assignments = Counter(
            'task_assignments_total',
            'Total task assignments to workers',
            ['worker_id'],
            registry=self.registry
        )
        
        self.task_reassignments = Counter(
            'task_reassignments_total',
            'Total task reassignments between workers',
            ['from_worker', 'reason'],
            registry=self.registry
        )
        
        self.worker_health_changes = Counter(
            'worker_health_changes_total',
            'Total worker health status changes',
            ['worker_id', 'status'],
            registry=self.registry
        )
        
        # Throughput scaling metrics
        self.scaling_test_results = Counter(
            'scaling_test_results_total',
            'Total scaling test results',
            ['is_linear'],
            registry=self.registry
        )
        
        self.scaling_efficiency = Gauge(
            'scaling_efficiency',
            'Latest scaling efficiency measurement',
            registry=self.registry
        )
        
        self.linear_scaling_coefficient = Gauge(
            'linear_scaling_coefficient',
            'Latest linear scaling coefficient (R²)',
            registry=self.registry
        )
        
        self.throughput_per_worker = Gauge(
            'throughput_per_worker',
            'Throughput per worker in tasks per minute',
            ['worker_count'],
            registry=self.registry
        )
        
        # Decision pipeline metrics
        self.decision_pipeline_time_seconds = Histogram(
            'decision_pipeline_time_seconds',
            'Decision pipeline processing time in seconds',
            ['pipeline_stage'],
            buckets=[0.1, 0.5, 1, 5, 10, 15, 30, 60, 120, 180],
            registry=self.registry
        )
        
        self.llm_call_time_seconds = Histogram(
            'llm_call_time_seconds',
            'LLM API call time in seconds',
            ['model', 'call_type'],
            buckets=[0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60],
            registry=self.registry
        )
        
        self.freshdesk_update_time_seconds = Histogram(
            'freshdesk_update_time_seconds',
            'Freshdesk ticket update time in seconds',
            ['operation_type'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 15, 20],
            registry=self.registry
        )
    
    def _setup_queue_metrics(self):
        """Set up queue depth and system health metrics."""
        
        # Queue metrics
        self.queue_depth = Gauge(
            'queue_depth',
            'Number of tasks waiting for processing',
            ['priority'],
            registry=self.registry
        )
        
        self.queue_processing_rate = Gauge(
            'queue_processing_rate_per_minute',
            'Tasks processed per minute',
            registry=self.registry
        )
        
        self.dead_letter_queue_size = Gauge(
            'dead_letter_queue_size',
            'Number of tasks in dead letter queue',
            registry=self.registry
        )
        
        # Task lifecycle metrics
        self.task_enqueue_time_seconds = Histogram(
            'task_enqueue_time_seconds',
            'Time to enqueue a task',
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            registry=self.registry
        )
        
        self.task_wait_time_seconds = Histogram(
            'task_wait_time_seconds',
            'Time tasks wait in queue before processing',
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry
        )
    
    def _setup_system_metrics(self):
        """Set up system health and dependency metrics."""
        
        # Dependency health
        self.redis_connection_status = Gauge(
            'redis_connection_status',
            'Redis connection status (1=connected, 0=disconnected)',
            registry=self.registry
        )
        
        self.hatchet_connection_status = Gauge(
            'hatchet_connection_status',
            'Hatchet connection status (1=connected, 0=disconnected)',
            registry=self.registry
        )
        
        self.database_connection_status = Gauge(
            'database_connection_status',
            'Database connection status (1=connected, 0=disconnected)',
            registry=self.registry
        )
        
        # Graceful degradation metrics
        self.fallback_mode_active = Gauge(
            'fallback_mode_active',
            'Whether fallback mode is active (1=active, 0=normal)',
            ['fallback_type'],
            registry=self.registry
        )
        
        self.synchronous_processing_count = Counter(
            'synchronous_processing_count_total',
            'Number of requests processed synchronously due to fallback',
            ['fallback_reason'],
            registry=self.registry
        )
        
        # System information
        self.system_info = Info(
            'webhook_orchestration_system_info',
            'System information for webhook orchestration',
            registry=self.registry
        )
        
        # Initialize system info
        self.system_info.info({
            'component': self.labels.component,
            'version': self.labels.version,
            'environment': self.labels.environment,
            'started_at': datetime.now(timezone.utc).isoformat()
        })
    
    def _setup_tracing_metrics(self):
        """Set up end-to-end tracing metrics."""
        
        # Trace metrics
        self.trace_spans_total = Counter(
            'trace_spans_total',
            'Total number of trace spans created',
            ['span_type', 'operation'],
            registry=self.registry
        )
        
        self.trace_duration_seconds = Histogram(
            'trace_duration_seconds',
            'End-to-end trace duration in seconds',
            ['trace_type'],
            buckets=[1, 5, 10, 30, 60, 120, 180, 240, 300, 600],
            registry=self.registry
        )
        
        self.trace_correlation_success_total = Counter(
            'trace_correlation_success_total',
            'Successful trace correlations between components',
            ['from_component', 'to_component'],
            registry=self.registry
        )
        
        self.trace_correlation_failures_total = Counter(
            'trace_correlation_failures_total',
            'Failed trace correlations between components',
            ['from_component', 'to_component', 'failure_reason'],
            registry=self.registry
        )
    
    # Webhook metrics recording methods
    
    def record_webhook_request(
        self,
        status_code: int,
        validation_result: str,
        response_time_ms: float,
        validation_times: Optional[Dict[str, float]] = None
    ):
        """
        Record webhook request metrics.
        
        Args:
            status_code: HTTP status code returned
            validation_result: Result of validation (valid, invalid_signature, etc.)
            response_time_ms: Total response time in milliseconds
            validation_times: Optional dict of validation step times
        """
        # Record request count
        self.webhook_requests_total.labels(
            status_code=str(status_code),
            validation_result=validation_result
        ).inc()
        
        # Record response time
        self.webhook_response_time_ms.observe(response_time_ms)
        
        # Record validation times if provided
        if validation_times:
            for validation_type, time_ms in validation_times.items():
                self.webhook_validation_time_ms.labels(
                    validation_type=validation_type
                ).observe(time_ms)
        
        # Record specific failure types
        if validation_result == "invalid_signature":
            self.webhook_signature_failures_total.inc()
        elif validation_result == "invalid_timestamp":
            self.webhook_timestamp_failures_total.inc()
        elif validation_result == "malformed_payload":
            self.webhook_payload_failures_total.inc()
    
    def record_idempotency_hit(self):
        """Record an idempotency hit (duplicate request detected)."""
        self.idempotency_hits_total.inc()
    
    def record_idempotency_error(self, error_type: str):
        """
        Record an idempotency store error.
        
        Args:
            error_type: Type of error (connection_failed, timeout, etc.)
        """
        self.idempotency_store_errors_total.labels(error_type=error_type).inc()
    
    # Worker metrics recording methods
    
    def record_task_completion(
        self,
        status: str,
        processing_path: str,
        decision_type: str,
        processing_time_seconds: float
    ):
        """
        Record task completion metrics.
        
        Args:
            status: Task completion status (success, failed, timeout)
            processing_path: Processing path used (fast_path, complex)
            decision_type: Type of decision made (approved, denied, escalated)
            processing_time_seconds: Processing time in seconds
        """
        self.task_completions_total.labels(
            status=status,
            processing_path=processing_path,
            decision_type=decision_type
        ).inc()
        
        self.task_processing_time_seconds.labels(
            processing_path=processing_path,
            decision_type=decision_type
        ).observe(processing_time_seconds)
    
    def record_task_retry(self, error_type: str, retry_count: int):
        """
        Record task retry attempt.
        
        Args:
            error_type: Type of error that caused retry
            retry_count: Current retry attempt number
        """
        self.task_retries_total.labels(
            error_type=error_type,
            retry_count=str(retry_count)
        ).inc()
    
    def record_worker_registration(self, worker_id: str, capacity: int):
        """Record worker registration event."""
        try:
            self.worker_registrations.labels(worker_id=worker_id).inc()
            self.worker_capacity.labels(worker_id=worker_id).set(capacity)
            logger.debug(f"Recorded worker registration: {worker_id} (capacity: {capacity})")
        except Exception as e:
            logger.error(f"Failed to record worker registration metrics: {e}")
    
    def record_worker_unregistration(self, worker_id: str, reassigned_tasks: int):
        """Record worker unregistration event."""
        try:
            self.worker_unregistrations.labels(worker_id=worker_id).inc()
            self.task_reassignments.labels(
                from_worker=worker_id,
                reason="worker_removed"
            ).inc(reassigned_tasks)
            logger.debug(f"Recorded worker unregistration: {worker_id} (reassigned: {reassigned_tasks})")
        except Exception as e:
            logger.error(f"Failed to record worker unregistration metrics: {e}")
    
    def record_worker_health_change(self, worker_id: str, status: str):
        """Record worker health status change."""
        try:
            self.worker_health_changes.labels(
                worker_id=worker_id,
                status=status
            ).inc()
            logger.debug(f"Recorded worker health change: {worker_id} -> {status}")
        except Exception as e:
            logger.error(f"Failed to record worker health change metrics: {e}")
    
    def record_task_assignment(self, task_id: str, worker_id: str, load_factor: float):
        """Record task assignment to worker."""
        try:
            self.task_assignments.labels(worker_id=worker_id).inc()
            self.worker_load_factor.labels(worker_id=worker_id).set(load_factor)
            logger.debug(f"Recorded task assignment: {task_id} -> {worker_id} (load: {load_factor:.2f})")
        except Exception as e:
            logger.error(f"Failed to record task assignment metrics: {e}")
    
    def record_worker_stats(self, worker_id: str, processed_count: int, error_count: int):
        """
        Record worker statistics.
        
        Args:
            worker_id: Unique worker identifier
            processed_count: Total tasks processed by worker
            error_count: Total errors encountered by worker
        """
        # Set processed count (counter will be incremented from 0)
        current_processed = self.worker_processed_count.labels(worker_id=worker_id)._value._value
        if processed_count > current_processed:
            self.worker_processed_count.labels(worker_id=worker_id).inc(
                processed_count - current_processed
            )
    
    def record_scaling_test_result(
        self,
        linear_coefficient: float,
        scaling_efficiency: float,
        is_linear: bool,
        test_duration: float
    ):
        """Record scaling test result metrics."""
        try:
            self.scaling_test_results.labels(is_linear=str(is_linear)).inc()
            self.scaling_efficiency.set(scaling_efficiency)
            self.linear_scaling_coefficient.set(linear_coefficient)
            logger.debug(f"Recorded scaling test result: efficiency={scaling_efficiency:.2f}, linear={is_linear}")
        except Exception as e:
            logger.error(f"Failed to record scaling test result metrics: {e}")
    
    def record_throughput_measurement(
        self,
        worker_count: int,
        throughput_per_minute: float,
        throughput_per_worker: float,
        error_rate: float
    ):
        """Record throughput measurement metrics."""
        try:
            self.throughput_per_worker.labels(worker_count=str(worker_count)).set(throughput_per_worker)
            logger.debug(f"Recorded throughput measurement: {worker_count} workers, {throughput_per_worker:.1f} tasks/min/worker")
        except Exception as e:
            logger.error(f"Failed to record throughput measurement metrics: {e}")
    
    def record_decision_pipeline_time(self, pipeline_stage: str, time_seconds: float):
        """
        Record decision pipeline processing time.
        
        Args:
            pipeline_stage: Stage of pipeline (routing, rule_engine, llm, persistence)
            time_seconds: Processing time in seconds
        """
        self.decision_pipeline_time_seconds.labels(
            pipeline_stage=pipeline_stage
        ).observe(time_seconds)
    
    def record_llm_call(self, model: str, call_type: str, time_seconds: float):
        """
        Record LLM API call metrics.
        
        Args:
            model: LLM model used (gpt-4, gemini-pro, etc.)
            call_type: Type of call (decision, reasoning, classification)
            time_seconds: Call duration in seconds
        """
        self.llm_call_time_seconds.labels(
            model=model,
            call_type=call_type
        ).observe(time_seconds)
    
    def record_freshdesk_update(self, operation_type: str, time_seconds: float):
        """
        Record Freshdesk update metrics.
        
        Args:
            operation_type: Type of operation (add_note, update_status)
            time_seconds: Operation duration in seconds
        """
        self.freshdesk_update_time_seconds.labels(
            operation_type=operation_type
        ).observe(time_seconds)
    
    # Queue metrics recording methods
    
    def update_queue_depth(self, depth: int, priority: str = "normal"):
        """
        Update queue depth metric.
        
        Args:
            depth: Current queue depth
            priority: Queue priority level
        """
        self.queue_depth.labels(priority=priority).set(depth)
    
    def update_processing_rate(self, rate_per_minute: float):
        """
        Update queue processing rate.
        
        Args:
            rate_per_minute: Tasks processed per minute
        """
        self.queue_processing_rate.set(rate_per_minute)
    
    def update_dead_letter_queue_size(self, size: int):
        """
        Update dead letter queue size.
        
        Args:
            size: Current dead letter queue size
        """
        self.dead_letter_queue_size.set(size)
    
    def record_task_enqueue_time(self, time_seconds: float):
        """
        Record task enqueue time.
        
        Args:
            time_seconds: Time to enqueue task in seconds
        """
        self.task_enqueue_time_seconds.observe(time_seconds)
    
    def record_task_wait_time(self, wait_time_seconds: float):
        """
        Record task wait time in queue.
        
        Args:
            wait_time_seconds: Time task waited in queue
        """
        self.task_wait_time_seconds.observe(wait_time_seconds)
    
    # System health recording methods
    
    def update_dependency_status(self, dependency: str, is_connected: bool):
        """
        Update dependency connection status.
        
        Args:
            dependency: Dependency name (redis, hatchet, database)
            is_connected: Whether dependency is connected
        """
        status_value = 1 if is_connected else 0
        
        if dependency == "redis":
            self.redis_connection_status.set(status_value)
        elif dependency == "hatchet":
            self.hatchet_connection_status.set(status_value)
        elif dependency == "database":
            self.database_connection_status.set(status_value)
    
    def update_fallback_mode(self, fallback_type: str, is_active: bool):
        """
        Update fallback mode status.
        
        Args:
            fallback_type: Type of fallback (hatchet_unavailable, redis_unavailable)
            is_active: Whether fallback mode is active
        """
        self.fallback_mode_active.labels(fallback_type=fallback_type).set(
            1 if is_active else 0
        )
    
    def record_synchronous_processing(self, fallback_reason: str):
        """
        Record synchronous processing due to fallback.
        
        Args:
            fallback_reason: Reason for fallback (hatchet_down, redis_down)
        """
        self.synchronous_processing_count.labels(fallback_reason=fallback_reason).inc()
    
    # Tracing metrics recording methods
    
    def record_trace_span(self, span_type: str, operation: str):
        """
        Record trace span creation.
        
        Args:
            span_type: Type of span (webhook, worker, api_call)
            operation: Operation being traced
        """
        self.trace_spans_total.labels(
            span_type=span_type,
            operation=operation
        ).inc()
    
    def record_trace_duration(self, trace_type: str, duration_seconds: float):
        """
        Record end-to-end trace duration.
        
        Args:
            trace_type: Type of trace (webhook_to_completion, webhook_to_freshdesk)
            duration_seconds: Total trace duration
        """
        self.trace_duration_seconds.labels(trace_type=trace_type).observe(duration_seconds)
    
    def record_trace_correlation(
        self,
        from_component: str,
        to_component: str,
        success: bool,
        failure_reason: Optional[str] = None
    ):
        """
        Record trace correlation between components.
        
        Args:
            from_component: Source component
            to_component: Target component
            success: Whether correlation was successful
            failure_reason: Reason for failure if not successful
        """
        if success:
            self.trace_correlation_success_total.labels(
                from_component=from_component,
                to_component=to_component
            ).inc()
        else:
            self.trace_correlation_failures_total.labels(
                from_component=from_component,
                to_component=to_component,
                failure_reason=failure_reason or "unknown"
            ).inc()
    
    # Metrics export and health methods
    
    def update_percentiles_from_tracker(self):
        """Update percentile gauges from the internal metrics tracker."""
        percentiles = self.metrics_tracker.get_webhook_processing_percentiles()
        
        if percentiles["p50"] is not None:
            self.webhook_response_time_p50.set(percentiles["p50"])
        if percentiles["p95"] is not None:
            self.webhook_response_time_p95.set(percentiles["p95"])
        if percentiles["p99"] is not None:
            self.webhook_response_time_p99.set(percentiles["p99"])
    
    def update_worker_count(self, count: int):
        """
        Update active worker count.
        
        Args:
            count: Number of active workers
        """
        self.active_workers.set(count)
    
    def get_metrics(self) -> str:
        """
        Get all metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        # Update percentiles from internal tracker
        self.update_percentiles_from_tracker()
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get health summary for monitoring.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "webhook_success_rate": self.metrics_tracker.get_webhook_success_rate(),
            "active_workers": self.active_workers._value._value,
            "queue_depth": sum(
                gauge._value._value for gauge in self.queue_depth._metrics.values()
            ),
            "dead_letter_queue_size": self.dead_letter_queue_size._value._value,
            "redis_connected": bool(self.redis_connection_status._value._value),
            "hatchet_connected": bool(self.hatchet_connection_status._value._value),
            "database_connected": bool(self.database_connection_status._value._value),
            "fallback_modes_active": [
                label_values["fallback_type"] 
                for label_values, gauge in self.fallback_mode_active._metrics.items()
                if gauge._value._value > 0
            ]
        }


# Global metrics collector instance
_webhook_metrics_collector: Optional[WebhookMetricsCollector] = None


def get_webhook_metrics_collector() -> WebhookMetricsCollector:
    """
    Get the global webhook metrics collector instance.
    
    Returns:
        The global WebhookMetricsCollector instance
    """
    global _webhook_metrics_collector
    if _webhook_metrics_collector is None:
        _webhook_metrics_collector = WebhookMetricsCollector()
    return _webhook_metrics_collector


def start_metrics_server(port: int = 8080) -> None:
    """
    Start Prometheus metrics HTTP server.
    
    Args:
        port: Port to serve metrics on (default: 8080)
    """
    collector = get_webhook_metrics_collector()
    start_http_server(port, registry=collector.registry)
    logger.info(f"Prometheus metrics server started on port {port}")


def get_prometheus_metrics() -> str:
    """
    Get webhook orchestration metrics in Prometheus format.
    
    Returns:
        Metrics in Prometheus text format
    """
    collector = get_webhook_metrics_collector()
    return collector.get_metrics()


def get_health_status() -> Dict[str, Any]:
    """
    Get system health status for monitoring.
    
    Returns:
        Dictionary with health status information
    """
    collector = get_webhook_metrics_collector()
    return collector.get_health_summary()