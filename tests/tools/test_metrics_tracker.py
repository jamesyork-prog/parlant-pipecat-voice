"""
Tests for metrics tracking functionality.

This module tests the MetricsTracker class and its integration with
webhook processing and journey execution.
"""

import pytest
from app_tools.tools.infrastructure.metrics import (
    MetricsTracker,
    ProcessingTimeMetrics,
    PerformanceTimer,
    get_metrics_tracker,
    reset_metrics
)


class TestProcessingTimeMetrics:
    """Tests for ProcessingTimeMetrics class."""
    
    def test_add_time(self):
        """Test adding processing times."""
        metrics = ProcessingTimeMetrics()
        
        metrics.add_time(100)
        metrics.add_time(200)
        metrics.add_time(300)
        
        assert metrics.get_count() == 3
    
    def test_get_percentile(self):
        """Test percentile calculation."""
        metrics = ProcessingTimeMetrics()
        
        # Add times: 100, 200, 300, 400, 500
        for i in range(1, 6):
            metrics.add_time(i * 100)
        
        # p50 should be around 300 (median)
        p50 = metrics.get_percentile(50)
        assert p50 == 300
        
        # p95 should be around 500
        p95 = metrics.get_percentile(95)
        assert p95 == 500
    
    def test_get_average(self):
        """Test average calculation."""
        metrics = ProcessingTimeMetrics()
        
        metrics.add_time(100)
        metrics.add_time(200)
        metrics.add_time(300)
        
        assert metrics.get_average() == 200.0
    
    def test_empty_metrics(self):
        """Test metrics with no data."""
        metrics = ProcessingTimeMetrics()
        
        assert metrics.get_percentile(50) is None
        assert metrics.get_average() is None
        assert metrics.get_count() == 0


class TestMetricsTracker:
    """Tests for MetricsTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.metrics = MetricsTracker()
    
    def test_record_webhook_success(self):
        """Test recording successful webhook processing."""
        self.metrics.record_webhook_success(150)
        
        assert self.metrics.webhook_success_count == 1
        assert self.metrics.webhook_processing_times.get_count() == 1
    
    def test_record_webhook_failure(self):
        """Test recording failed webhook processing."""
        self.metrics.record_webhook_failure()
        
        assert self.metrics.webhook_failure_count == 1
    
    def test_webhook_success_rate(self):
        """Test webhook success rate calculation."""
        self.metrics.record_webhook_success(100)
        self.metrics.record_webhook_success(150)
        self.metrics.record_webhook_failure()
        
        # 2 successes out of 3 total = 66.67%
        rate = self.metrics.get_webhook_success_rate()
        assert rate == pytest.approx(66.67, rel=0.01)
    
    def test_webhook_processing_percentiles(self):
        """Test webhook processing time percentiles."""
        for i in range(1, 11):
            self.metrics.record_webhook_success(i * 100)
        
        percentiles = self.metrics.get_webhook_processing_percentiles()
        
        # p50 should be around the median (550 for 10 values)
        assert percentiles["p50"] == 600  # 50% of 10 items = index 5 = 600
        assert percentiles["count"] == 10
        assert percentiles["average"] == 550.0
    
    def test_record_journey_activation(self):
        """Test recording journey activations."""
        self.metrics.record_journey_activation("Automated Ticket Processing")
        self.metrics.record_journey_activation("Automated Ticket Processing")
        self.metrics.record_journey_activation("Interactive Ticket Processing")
        
        counts = self.metrics.get_journey_activation_counts()
        
        assert counts["Automated Ticket Processing"] == 2
        assert counts["Interactive Ticket Processing"] == 1
    
    def test_record_journey_execution(self):
        """Test recording journey execution."""
        self.metrics.record_journey_execution(
            "Automated Ticket Processing",
            1500,
            "Approved"
        )
        
        # Check execution time was recorded
        percentiles = self.metrics.get_journey_execution_percentiles(
            "Automated Ticket Processing"
        )
        assert percentiles["count"] == 1
        assert percentiles["average"] == 1500.0
        
        # Check decision was recorded
        decisions = self.metrics.get_decision_distribution()
        assert decisions["Approved"] == 1
    
    def test_decision_distribution(self):
        """Test decision distribution tracking."""
        self.metrics.record_journey_execution("Journey1", 1000, "Approved")
        self.metrics.record_journey_execution("Journey2", 1000, "Approved")
        self.metrics.record_journey_execution("Journey3", 1000, "Denied")
        self.metrics.record_journey_execution("Journey4", 1000, "Escalated")
        
        distribution = self.metrics.get_decision_distribution()
        
        assert distribution["Approved"] == 2
        assert distribution["Denied"] == 1
        assert distribution["Escalated"] == 1
    
    def test_record_error(self):
        """Test error recording."""
        self.metrics.record_error("validation_failure")
        self.metrics.record_error("validation_failure")
        self.metrics.record_error("journey_failure")
        
        errors = self.metrics.get_error_counts()
        
        assert errors["validation_failure"] == 2
        assert errors["journey_failure"] == 1
    
    def test_record_validation_failure(self):
        """Test validation failure recording."""
        self.metrics.record_validation_failure()
        
        assert self.metrics.validation_failure_count == 1
        assert self.metrics.error_counts["validation_failure"] == 1
    
    def test_record_journey_failure(self):
        """Test journey failure recording."""
        self.metrics.record_journey_failure()
        
        assert self.metrics.journey_failure_count == 1
        assert self.metrics.error_counts["journey_failure"] == 1
    
    def test_validation_failure_rate(self):
        """Test validation failure rate calculation."""
        self.metrics.record_webhook_success(100)
        self.metrics.record_webhook_success(100)
        self.metrics.record_validation_failure()
        self.metrics.record_webhook_failure()
        
        # 1 validation failure out of 3 total = 33.33%
        rate = self.metrics.get_validation_failure_rate()
        assert rate == pytest.approx(33.33, rel=0.01)
    
    def test_journey_failure_rate(self):
        """Test journey failure rate calculation."""
        self.metrics.record_journey_activation("Journey1")
        self.metrics.record_journey_activation("Journey2")
        self.metrics.record_journey_activation("Journey3")
        self.metrics.record_journey_failure()
        
        # 1 failure out of 3 activations = 33.33%
        rate = self.metrics.get_journey_failure_rate()
        assert rate == pytest.approx(33.33, rel=0.01)
    
    def test_record_api_call_latency(self):
        """Test API call latency recording."""
        self.metrics.record_api_call_latency("freshdesk", 250)
        self.metrics.record_api_call_latency("freshdesk", 300)
        self.metrics.record_api_call_latency("parkwhiz", 150)
        
        freshdesk_latencies = self.metrics.get_api_call_latencies("freshdesk")
        assert freshdesk_latencies["count"] == 2
        assert freshdesk_latencies["average"] == 275.0
        
        parkwhiz_latencies = self.metrics.get_api_call_latencies("parkwhiz")
        assert parkwhiz_latencies["count"] == 1
        assert parkwhiz_latencies["average"] == 150.0
    
    def test_record_slow_operation(self):
        """Test slow operation recording."""
        self.metrics.record_slow_operation()
        self.metrics.record_slow_operation()
        
        assert self.metrics.slow_operation_count == 2
    
    def test_check_error_rate_threshold(self):
        """Test error rate threshold checking."""
        # Set up scenario with 20% validation failure rate
        for _ in range(8):
            self.metrics.record_webhook_success(100)
        for _ in range(2):
            self.metrics.record_validation_failure()
            self.metrics.record_webhook_failure()
        
        # Should exceed 10% threshold
        assert self.metrics.check_error_rate_threshold("validation", 10.0) is True
        
        # Should not exceed 25% threshold
        assert self.metrics.check_error_rate_threshold("validation", 25.0) is False
    
    def test_get_error_rate_alerts(self):
        """Test error rate alert generation."""
        # Create high validation failure rate
        for _ in range(8):
            self.metrics.record_webhook_success(100)
        for _ in range(2):
            self.metrics.record_validation_failure()
            self.metrics.record_webhook_failure()
        
        alerts = self.metrics.get_error_rate_alerts({"validation": 5.0})
        
        assert len(alerts) == 1
        assert alerts[0]["error_type"] == "validation"
        assert alerts[0]["rate_percent"] > 5.0
    
    def test_get_summary(self):
        """Test comprehensive metrics summary."""
        # Add some data
        self.metrics.record_webhook_success(150)
        self.metrics.record_webhook_failure()
        self.metrics.record_journey_activation("Journey1")
        self.metrics.record_journey_execution("Journey1", 1500, "Approved")
        self.metrics.record_error("test_error")
        
        summary = self.metrics.get_summary()
        
        assert "webhook_metrics" in summary
        assert "journey_metrics" in summary
        assert "error_metrics" in summary
        assert "performance_metrics" in summary
        assert "uptime_seconds" in summary
    
    def test_reset(self):
        """Test metrics reset."""
        self.metrics.record_webhook_success(100)
        self.metrics.record_journey_activation("Journey1")
        
        self.metrics.reset()
        
        assert self.metrics.webhook_success_count == 0
        assert len(self.metrics.journey_activation_counts) == 0


class TestPerformanceTimer:
    """Tests for PerformanceTimer context manager."""
    
    def test_performance_timer_basic(self):
        """Test basic performance timer usage."""
        metrics = MetricsTracker()
        
        with PerformanceTimer(metrics, "test_operation") as timer:
            # Simulate some work
            import time
            time.sleep(0.01)  # 10ms
        
        assert timer.get_duration_ms() is not None
        assert timer.get_duration_ms() >= 10
    
    def test_performance_timer_with_threshold(self):
        """Test performance timer with slow operation threshold."""
        metrics = MetricsTracker()
        
        with PerformanceTimer(metrics, "test_operation", threshold_ms=5):
            import time
            time.sleep(0.01)  # 10ms - exceeds 5ms threshold
        
        # Should have recorded a slow operation
        assert metrics.slow_operation_count == 1
    
    def test_performance_timer_record_as_webhook(self):
        """Test performance timer recording as webhook."""
        metrics = MetricsTracker()
        
        with PerformanceTimer(metrics, "test_operation", record_as="webhook"):
            import time
            time.sleep(0.01)
        
        # Should have recorded webhook processing time
        assert metrics.webhook_processing_times.get_count() == 1


class TestGlobalMetricsTracker:
    """Tests for global metrics tracker singleton."""
    
    def test_get_metrics_tracker(self):
        """Test getting global metrics tracker."""
        tracker1 = get_metrics_tracker()
        tracker2 = get_metrics_tracker()
        
        # Should return same instance
        assert tracker1 is tracker2
    
    def test_reset_metrics(self):
        """Test resetting global metrics."""
        tracker = get_metrics_tracker()
        tracker.record_webhook_success(100)
        
        reset_metrics()
        
        # Should have reset
        assert tracker.webhook_success_count == 0
