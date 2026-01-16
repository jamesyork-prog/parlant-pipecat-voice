"""
Metrics tracking for webhook automation and journey execution.

This module provides in-memory metrics collection for:
- Webhook processing (success/failure counts)
- Processing time percentiles (p50, p95, p99)
- Journey activation counts by type
- Decision distribution (Approved/Denied/Escalated)
- Error rates by type
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Literal, Any
from collections import defaultdict, deque
from dataclasses import dataclass, field
import statistics


@dataclass
class ProcessingTimeMetrics:
    """Metrics for processing time tracking."""
    times: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def add_time(self, time_ms: int) -> None:
        """Add a processing time measurement."""
        self.times.append(time_ms)
    
    def get_percentile(self, percentile: int) -> Optional[float]:
        """
        Calculate a percentile of processing times.
        
        Args:
            percentile: The percentile to calculate (50, 95, 99)
            
        Returns:
            The percentile value in milliseconds, or None if no data
        """
        if not self.times:
            return None
        
        sorted_times = sorted(self.times)
        index = int(len(sorted_times) * (percentile / 100.0))
        
        # Handle edge case where index equals length
        if index >= len(sorted_times):
            index = len(sorted_times) - 1
        
        return sorted_times[index]
    
    def get_average(self) -> Optional[float]:
        """Get average processing time."""
        if not self.times:
            return None
        return statistics.mean(self.times)
    
    def get_count(self) -> int:
        """Get total number of measurements."""
        return len(self.times)


class MetricsTracker:
    """
    In-memory metrics tracker for webhook automation.
    
    This class tracks various metrics related to webhook processing,
    journey execution, and error rates. Metrics are stored in memory
    and can be queried for monitoring and alerting.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        # Webhook metrics
        self.webhook_success_count = 0
        self.webhook_failure_count = 0
        self.webhook_processing_times = ProcessingTimeMetrics()
        
        # Journey activation metrics by type
        self.journey_activation_counts: Dict[str, int] = defaultdict(int)
        
        # Decision distribution
        self.decision_counts: Dict[str, int] = defaultdict(int)
        
        # Journey execution times by type
        self.journey_execution_times: Dict[str, ProcessingTimeMetrics] = defaultdict(
            ProcessingTimeMetrics
        )
        
        # Error metrics by type
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.validation_failure_count = 0
        self.journey_failure_count = 0
        
        # API call latency tracking
        self.api_call_latencies: Dict[str, ProcessingTimeMetrics] = defaultdict(
            ProcessingTimeMetrics
        )
        
        # Performance warnings
        self.slow_operation_count = 0
        
        # Timestamp of tracker initialization
        self.start_time = datetime.utcnow()
    
    # Webhook metrics
    
    def record_webhook_success(self, processing_time_ms: int) -> None:
        """
        Record a successful webhook processing.
        
        Args:
            processing_time_ms: Time taken to process the webhook in milliseconds
        """
        self.webhook_success_count += 1
        self.webhook_processing_times.add_time(processing_time_ms)
    
    def record_webhook_failure(self) -> None:
        """Record a failed webhook processing."""
        self.webhook_failure_count += 1
    
    def get_webhook_success_rate(self) -> float:
        """
        Calculate webhook success rate.
        
        Returns:
            Success rate as a percentage (0-100), or 0 if no webhooks processed
        """
        total = self.webhook_success_count + self.webhook_failure_count
        if total == 0:
            return 0.0
        return (self.webhook_success_count / total) * 100
    
    def get_webhook_processing_percentiles(self) -> Dict[str, Optional[float]]:
        """
        Get webhook processing time percentiles.
        
        Returns:
            Dictionary with p50, p95, p99 values in milliseconds
        """
        return {
            "p50": self.webhook_processing_times.get_percentile(50),
            "p95": self.webhook_processing_times.get_percentile(95),
            "p99": self.webhook_processing_times.get_percentile(99),
            "average": self.webhook_processing_times.get_average(),
            "count": self.webhook_processing_times.get_count()
        }
    
    # Journey metrics
    
    def record_journey_activation(self, journey_name: str) -> None:
        """
        Record a journey activation.
        
        Args:
            journey_name: Name of the journey that was activated
        """
        self.journey_activation_counts[journey_name] += 1
    
    def record_journey_execution(
        self,
        journey_name: str,
        execution_time_ms: int,
        decision: Optional[str] = None
    ) -> None:
        """
        Record a journey execution completion.
        
        Args:
            journey_name: Name of the journey that executed
            execution_time_ms: Time taken to execute the journey in milliseconds
            decision: The decision made (Approved/Denied/Escalated/etc.)
        """
        self.journey_execution_times[journey_name].add_time(execution_time_ms)
        
        if decision:
            self.decision_counts[decision] += 1
    
    def get_journey_activation_counts(self) -> Dict[str, int]:
        """
        Get journey activation counts by type.
        
        Returns:
            Dictionary mapping journey names to activation counts
        """
        return dict(self.journey_activation_counts)
    
    def get_journey_execution_percentiles(
        self,
        journey_name: str
    ) -> Dict[str, Optional[float]]:
        """
        Get execution time percentiles for a specific journey.
        
        Args:
            journey_name: Name of the journey
            
        Returns:
            Dictionary with p50, p95, p99 values in milliseconds
        """
        metrics = self.journey_execution_times[journey_name]
        return {
            "p50": metrics.get_percentile(50),
            "p95": metrics.get_percentile(95),
            "p99": metrics.get_percentile(99),
            "average": metrics.get_average(),
            "count": metrics.get_count()
        }
    
    def get_decision_distribution(self) -> Dict[str, int]:
        """
        Get distribution of decisions made.
        
        Returns:
            Dictionary mapping decision types to counts
        """
        return dict(self.decision_counts)
    
    # Error metrics
    
    def record_error(self, error_type: str) -> None:
        """
        Record an error occurrence.
        
        Args:
            error_type: Type/category of the error
        """
        self.error_counts[error_type] += 1
    
    def record_validation_failure(self) -> None:
        """Record a webhook validation failure."""
        self.validation_failure_count += 1
        self.record_error("validation_failure")
    
    def record_journey_failure(self) -> None:
        """Record a journey execution failure."""
        self.journey_failure_count += 1
        self.record_error("journey_failure")
    
    def get_error_counts(self) -> Dict[str, int]:
        """
        Get error counts by type.
        
        Returns:
            Dictionary mapping error types to counts
        """
        return dict(self.error_counts)
    
    def get_validation_failure_rate(self) -> float:
        """
        Calculate validation failure rate.
        
        Returns:
            Failure rate as a percentage (0-100), or 0 if no webhooks processed
        """
        total = self.webhook_success_count + self.webhook_failure_count
        if total == 0:
            return 0.0
        return (self.validation_failure_count / total) * 100
    
    def get_journey_failure_rate(self) -> float:
        """
        Calculate journey failure rate.
        
        Returns:
            Failure rate as a percentage (0-100), or 0 if no journeys activated
        """
        total_activations = sum(self.journey_activation_counts.values())
        if total_activations == 0:
            return 0.0
        return (self.journey_failure_count / total_activations) * 100
    
    def check_error_rate_threshold(
        self,
        error_type: str,
        threshold_percent: float = 10.0
    ) -> bool:
        """
        Check if error rate exceeds a threshold.
        
        Args:
            error_type: Type of error to check
            threshold_percent: Threshold percentage (0-100)
            
        Returns:
            True if threshold exceeded, False otherwise
        """
        if error_type == "validation":
            rate = self.get_validation_failure_rate()
        elif error_type == "journey":
            rate = self.get_journey_failure_rate()
        elif error_type == "webhook":
            total = self.webhook_success_count + self.webhook_failure_count
            if total == 0:
                return False
            rate = (self.webhook_failure_count / total) * 100
        else:
            # For custom error types, calculate rate against total webhooks
            total = self.webhook_success_count + self.webhook_failure_count
            if total == 0:
                return False
            error_count = self.error_counts.get(error_type, 0)
            rate = (error_count / total) * 100
        
        return rate > threshold_percent
    
    def get_error_rate_alerts(
        self,
        thresholds: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get list of error rate alerts based on thresholds.
        
        Args:
            thresholds: Dictionary mapping error types to threshold percentages.
                       Defaults to {"validation": 5.0, "journey": 10.0, "webhook": 10.0}
        
        Returns:
            List of alert dictionaries with error type, rate, and threshold
        """
        if thresholds is None:
            thresholds = {
                "validation": 5.0,
                "journey": 10.0,
                "webhook": 10.0
            }
        
        alerts = []
        
        for error_type, threshold in thresholds.items():
            if self.check_error_rate_threshold(error_type, threshold):
                if error_type == "validation":
                    rate = self.get_validation_failure_rate()
                elif error_type == "journey":
                    rate = self.get_journey_failure_rate()
                elif error_type == "webhook":
                    total = self.webhook_success_count + self.webhook_failure_count
                    rate = (self.webhook_failure_count / total) * 100 if total > 0 else 0
                else:
                    total = self.webhook_success_count + self.webhook_failure_count
                    error_count = self.error_counts.get(error_type, 0)
                    rate = (error_count / total) * 100 if total > 0 else 0
                
                alerts.append({
                    "error_type": error_type,
                    "rate_percent": round(rate, 2),
                    "threshold_percent": threshold,
                    "severity": "high" if rate > threshold * 2 else "medium"
                })
        
        return alerts
    
    # Performance metrics
    
    def record_api_call_latency(self, api_name: str, latency_ms: int) -> None:
        """
        Record API call latency.
        
        Args:
            api_name: Name of the API (e.g., "freshdesk", "parkwhiz")
            latency_ms: Latency in milliseconds
        """
        self.api_call_latencies[api_name].add_time(latency_ms)
    
    def get_api_call_latencies(self, api_name: str) -> Dict[str, Optional[float]]:
        """
        Get API call latency percentiles.
        
        Args:
            api_name: Name of the API
            
        Returns:
            Dictionary with p50, p95, p99 values in milliseconds
        """
        metrics = self.api_call_latencies[api_name]
        return {
            "p50": metrics.get_percentile(50),
            "p95": metrics.get_percentile(95),
            "p99": metrics.get_percentile(99),
            "average": metrics.get_average(),
            "count": metrics.get_count()
        }
    
    def record_slow_operation(self) -> None:
        """Record a slow operation (processing time > threshold)."""
        self.slow_operation_count += 1
    
    # Summary methods
    
    def get_summary(self) -> Dict:
        """
        Get a comprehensive summary of all metrics.
        
        Returns:
            Dictionary containing all tracked metrics
        """
        uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime_seconds,
            "webhook_metrics": {
                "success_count": self.webhook_success_count,
                "failure_count": self.webhook_failure_count,
                "success_rate_percent": self.get_webhook_success_rate(),
                "processing_times": self.get_webhook_processing_percentiles()
            },
            "journey_metrics": {
                "activation_counts": self.get_journey_activation_counts(),
                "execution_times": {
                    journey_name: self.get_journey_execution_percentiles(journey_name)
                    for journey_name in self.journey_execution_times.keys()
                },
                "decision_distribution": self.get_decision_distribution()
            },
            "error_metrics": {
                "error_counts": self.get_error_counts(),
                "validation_failure_count": self.validation_failure_count,
                "validation_failure_rate_percent": self.get_validation_failure_rate(),
                "journey_failure_count": self.journey_failure_count,
                "journey_failure_rate_percent": self.get_journey_failure_rate()
            },
            "performance_metrics": {
                "slow_operation_count": self.slow_operation_count,
                "api_latencies": {
                    api_name: self.get_api_call_latencies(api_name)
                    for api_name in self.api_call_latencies.keys()
                }
            }
        }
    
    def reset(self) -> None:
        """Reset all metrics to initial state."""
        self.__init__()


# Global metrics tracker instance
_metrics_tracker: Optional[MetricsTracker] = None


def get_metrics_tracker() -> MetricsTracker:
    """
    Get the global metrics tracker instance.
    
    Returns:
        The global MetricsTracker instance
    """
    global _metrics_tracker
    if _metrics_tracker is None:
        _metrics_tracker = MetricsTracker()
    return _metrics_tracker


def reset_metrics() -> None:
    """Reset the global metrics tracker."""
    global _metrics_tracker
    if _metrics_tracker is not None:
        _metrics_tracker.reset()



class PerformanceTimer:
    """
    Context manager for timing operations and recording metrics.
    
    Usage:
        with PerformanceTimer(metrics, "webhook_processing") as timer:
            # Do work
            pass
        # Automatically records processing time
    """
    
    def __init__(
        self,
        metrics_tracker: MetricsTracker,
        operation_name: str,
        threshold_ms: Optional[int] = None,
        record_as: Optional[str] = None
    ):
        """
        Initialize the performance timer.
        
        Args:
            metrics_tracker: The metrics tracker to record to
            operation_name: Name of the operation being timed
            threshold_ms: Optional threshold for slow operation warnings
            record_as: Optional category to record the time under
        """
        self.metrics = metrics_tracker
        self.operation_name = operation_name
        self.threshold_ms = threshold_ms
        self.record_as = record_as
        self.start_time = None
        self.end_time = None
        self.duration_ms = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record metrics."""
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        
        # Record slow operation if threshold exceeded
        if self.threshold_ms and self.duration_ms > self.threshold_ms:
            self.metrics.record_slow_operation()
        
        # Record to specific category if specified
        if self.record_as:
            if self.record_as == "webhook":
                self.metrics.webhook_processing_times.add_time(self.duration_ms)
            elif self.record_as.startswith("api:"):
                api_name = self.record_as[4:]
                self.metrics.record_api_call_latency(api_name, self.duration_ms)
        
        return False  # Don't suppress exceptions
    
    def get_duration_ms(self) -> Optional[int]:
        """Get the duration in milliseconds."""
        return self.duration_ms
