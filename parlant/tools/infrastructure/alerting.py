"""
Alerting configuration and management for async webhook orchestration.

This module provides alerting capabilities for:
- Queue depth alerting thresholds
- Retry rate monitoring and alerts
- Performance degradation alerts
- System health monitoring

Implements requirements:
- 5.3: Queue depth alerting thresholds
- 5.4: Retry rate monitoring and alerts
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

from .prometheus_metrics import get_webhook_metrics_collector
from .metrics import get_metrics_tracker

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertThreshold:
    """Configuration for an alert threshold."""
    name: str
    description: str
    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte", "eq", "ne"
    duration_seconds: int  # How long condition must persist
    severity: AlertSeverity
    enabled: bool = True
    
    def evaluate(self, current_value: float, duration_exceeded: bool) -> bool:
        """
        Evaluate if the threshold condition is met.
        
        Args:
            current_value: Current metric value
            duration_exceeded: Whether duration threshold is exceeded
            
        Returns:
            True if alert should be triggered
        """
        if not self.enabled:
            return False
        
        # Check value threshold
        value_condition_met = False
        if self.comparison == "gt":
            value_condition_met = current_value > self.threshold_value
        elif self.comparison == "lt":
            value_condition_met = current_value < self.threshold_value
        elif self.comparison == "gte":
            value_condition_met = current_value >= self.threshold_value
        elif self.comparison == "lte":
            value_condition_met = current_value <= self.threshold_value
        elif self.comparison == "eq":
            value_condition_met = current_value == self.threshold_value
        elif self.comparison == "ne":
            value_condition_met = current_value != self.threshold_value
        
        # Both value and duration conditions must be met
        return value_condition_met and duration_exceeded


@dataclass
class Alert:
    """An active or resolved alert."""
    id: str
    threshold_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "threshold_name": self.threshold_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "metadata": self.metadata
        }


class AlertManager:
    """
    Manages alerting for the async webhook orchestration system.
    
    Provides functionality for:
    - Threshold configuration and evaluation
    - Alert generation and management
    - Alert notification and escalation
    - Alert history and reporting
    """
    
    def __init__(self):
        """Initialize the alert manager."""
        self.thresholds: Dict[str, AlertThreshold] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.threshold_states: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        
        logger.info("AlertManager initialized with default thresholds")
    
    def _setup_default_thresholds(self):
        """Set up default alerting thresholds."""
        
        # Queue depth alerts (Requirement 5.3)
        self.add_threshold(AlertThreshold(
            name="queue_depth_high",
            description="Queue depth exceeds 100 tasks",
            metric_name="queue_depth",
            threshold_value=100,
            comparison="gt",
            duration_seconds=60,  # 1 minute
            severity=AlertSeverity.MEDIUM
        ))
        
        self.add_threshold(AlertThreshold(
            name="queue_depth_critical",
            description="Queue depth exceeds 500 tasks",
            metric_name="queue_depth",
            threshold_value=500,
            comparison="gt",
            duration_seconds=30,  # 30 seconds
            severity=AlertSeverity.CRITICAL
        ))
        
        # Retry rate alerts (Requirement 5.4)
        self.add_threshold(AlertThreshold(
            name="retry_rate_high",
            description="Retry rate exceeds 10% over 5 minutes",
            metric_name="retry_rate_percent",
            threshold_value=10.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=AlertSeverity.MEDIUM
        ))
        
        self.add_threshold(AlertThreshold(
            name="retry_rate_critical",
            description="Retry rate exceeds 25% over 2 minutes",
            metric_name="retry_rate_percent",
            threshold_value=25.0,
            comparison="gt",
            duration_seconds=120,  # 2 minutes
            severity=AlertSeverity.HIGH
        ))
        
        # Performance degradation alerts
        self.add_threshold(AlertThreshold(
            name="webhook_response_time_high",
            description="Webhook response time P99 exceeds 75ms",
            metric_name="webhook_response_time_p99",
            threshold_value=75.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=AlertSeverity.MEDIUM
        ))
        
        self.add_threshold(AlertThreshold(
            name="webhook_response_time_critical",
            description="Webhook response time P99 exceeds 100ms",
            metric_name="webhook_response_time_p99",
            threshold_value=100.0,
            comparison="gt",
            duration_seconds=120,  # 2 minutes
            severity=AlertSeverity.HIGH
        ))
        
        # Success rate alerts
        self.add_threshold(AlertThreshold(
            name="webhook_success_rate_low",
            description="Webhook success rate below 95%",
            metric_name="webhook_success_rate",
            threshold_value=95.0,
            comparison="lt",
            duration_seconds=300,  # 5 minutes
            severity=AlertSeverity.MEDIUM
        ))
        
        self.add_threshold(AlertThreshold(
            name="webhook_success_rate_critical",
            description="Webhook success rate below 90%",
            metric_name="webhook_success_rate",
            threshold_value=90.0,
            comparison="lt",
            duration_seconds=120,  # 2 minutes
            severity=AlertSeverity.HIGH
        ))
        
        # Dead letter queue alerts
        self.add_threshold(AlertThreshold(
            name="dead_letter_queue_high",
            description="Dead letter queue size exceeds 10 tasks",
            metric_name="dead_letter_queue_size",
            threshold_value=10,
            comparison="gt",
            duration_seconds=600,  # 10 minutes
            severity=AlertSeverity.MEDIUM
        ))
        
        self.add_threshold(AlertThreshold(
            name="dead_letter_queue_critical",
            description="Dead letter queue size exceeds 50 tasks",
            metric_name="dead_letter_queue_size",
            threshold_value=50,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=AlertSeverity.HIGH
        ))
        
        # Worker health alerts
        self.add_threshold(AlertThreshold(
            name="active_workers_low",
            description="Active worker count below 2",
            metric_name="active_workers",
            threshold_value=2,
            comparison="lt",
            duration_seconds=120,  # 2 minutes
            severity=AlertSeverity.HIGH
        ))
        
        self.add_threshold(AlertThreshold(
            name="active_workers_none",
            description="No active workers available",
            metric_name="active_workers",
            threshold_value=0,
            comparison="eq",
            duration_seconds=30,  # 30 seconds
            severity=AlertSeverity.CRITICAL
        ))
        
        # System dependency alerts
        self.add_threshold(AlertThreshold(
            name="redis_disconnected",
            description="Redis connection is down",
            metric_name="redis_connected",
            threshold_value=0,
            comparison="eq",
            duration_seconds=60,  # 1 minute
            severity=AlertSeverity.HIGH
        ))
        
        self.add_threshold(AlertThreshold(
            name="hatchet_disconnected",
            description="Hatchet connection is down",
            metric_name="hatchet_connected",
            threshold_value=0,
            comparison="eq",
            duration_seconds=60,  # 1 minute
            severity=AlertSeverity.HIGH
        ))
        
        self.add_threshold(AlertThreshold(
            name="database_disconnected",
            description="Database connection is down",
            metric_name="database_connected",
            threshold_value=0,
            comparison="eq",
            duration_seconds=30,  # 30 seconds
            severity=AlertSeverity.CRITICAL
        ))
    
    def add_threshold(self, threshold: AlertThreshold):
        """
        Add an alert threshold.
        
        Args:
            threshold: AlertThreshold configuration
        """
        self.thresholds[threshold.name] = threshold
        self.threshold_states[threshold.name] = {
            "first_breach_time": None,
            "last_check_time": None,
            "last_value": None
        }
        
        logger.info(f"Added alert threshold: {threshold.name}")
    
    def remove_threshold(self, threshold_name: str):
        """
        Remove an alert threshold.
        
        Args:
            threshold_name: Name of threshold to remove
        """
        if threshold_name in self.thresholds:
            del self.thresholds[threshold_name]
            del self.threshold_states[threshold_name]
            logger.info(f"Removed alert threshold: {threshold_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """
        Add a notification handler for alerts.
        
        Args:
            handler: Function to call when alerts are triggered
        """
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    def evaluate_thresholds(self) -> List[Alert]:
        """
        Evaluate all thresholds against current metrics.
        
        Returns:
            List of newly triggered alerts
        """
        current_time = datetime.now(timezone.utc)
        new_alerts = []
        
        # Get current metrics
        current_metrics = self._get_current_metrics()
        
        for threshold_name, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
            
            state = self.threshold_states[threshold_name]
            current_value = current_metrics.get(threshold.metric_name)
            
            if current_value is None:
                logger.warning(f"Metric {threshold.metric_name} not available for threshold {threshold_name}")
                continue
            
            # Update state
            state["last_check_time"] = current_time
            state["last_value"] = current_value
            
            # Check if threshold is breached
            value_condition_met = self._evaluate_threshold_condition(threshold, current_value)
            
            if value_condition_met:
                # Threshold breached
                if state["first_breach_time"] is None:
                    state["first_breach_time"] = current_time
                
                # Check if duration threshold is met
                breach_duration = (current_time - state["first_breach_time"]).total_seconds()
                duration_exceeded = breach_duration >= threshold.duration_seconds
                
                if duration_exceeded and threshold_name not in self.active_alerts:
                    # Trigger new alert
                    alert = self._create_alert(threshold, current_value, current_time)
                    self.active_alerts[alert.id] = alert
                    self.alert_history.append(alert)
                    new_alerts.append(alert)
                    
                    # Send notifications
                    self._send_notifications(alert)
                    
                    logger.warning(f"Alert triggered: {alert.message}")
            
            else:
                # Threshold not breached
                state["first_breach_time"] = None
                
                # Check if we should resolve an active alert
                if threshold_name in self.active_alerts:
                    alert_id = None
                    for aid, alert in self.active_alerts.items():
                        if alert.threshold_name == threshold_name:
                            alert_id = aid
                            break
                    
                    if alert_id:
                        self._resolve_alert(alert_id, current_time)
        
        return new_alerts
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current metric values for threshold evaluation."""
        metrics_tracker = get_metrics_tracker()
        metrics_collector = get_webhook_metrics_collector()
        
        # Get basic metrics
        webhook_percentiles = metrics_tracker.get_webhook_processing_percentiles()
        success_rate = metrics_tracker.get_webhook_success_rate()
        
        # Get health status
        health_status = metrics_collector.get_health_summary()
        
        return {
            # Queue metrics
            "queue_depth": health_status.get("queue_depth", 0),
            "dead_letter_queue_size": health_status.get("dead_letter_queue_size", 0),
            
            # Performance metrics
            "webhook_response_time_p99": webhook_percentiles.get("p99", 0) or 0,
            "webhook_response_time_p95": webhook_percentiles.get("p95", 0) or 0,
            "webhook_success_rate": success_rate,
            
            # Worker metrics
            "active_workers": health_status.get("active_workers", 0),
            
            # Dependency health (1 = connected, 0 = disconnected)
            "redis_connected": 1 if health_status.get("redis_connected", False) else 0,
            "hatchet_connected": 1 if health_status.get("hatchet_connected", False) else 0,
            "database_connected": 1 if health_status.get("database_connected", False) else 0,
            
            # Retry rate calculation
            "retry_rate_percent": self._calculate_retry_rate()
        }
    
    def _calculate_retry_rate(self) -> float:
        """Calculate current retry rate percentage."""
        metrics_tracker = get_metrics_tracker()
        
        # Get total webhooks and total retries
        total_webhooks = metrics_tracker.webhook_success_count + metrics_tracker.webhook_failure_count
        
        # Calculate retry rate from error counts (simplified)
        if total_webhooks == 0:
            return 0.0
        
        # Estimate retry rate based on failure rate
        failure_rate = (metrics_tracker.webhook_failure_count / total_webhooks) * 100
        
        # Assume retries are proportional to failures
        retry_rate = failure_rate * 0.5  # Simplified calculation
        
        return retry_rate
    
    def _evaluate_threshold_condition(self, threshold: AlertThreshold, current_value: float) -> bool:
        """Evaluate threshold condition."""
        if threshold.comparison == "gt":
            return current_value > threshold.threshold_value
        elif threshold.comparison == "lt":
            return current_value < threshold.threshold_value
        elif threshold.comparison == "gte":
            return current_value >= threshold.threshold_value
        elif threshold.comparison == "lte":
            return current_value <= threshold.threshold_value
        elif threshold.comparison == "eq":
            return current_value == threshold.threshold_value
        elif threshold.comparison == "ne":
            return current_value != threshold.threshold_value
        else:
            return False
    
    def _create_alert(self, threshold: AlertThreshold, current_value: float, triggered_at: datetime) -> Alert:
        """Create a new alert."""
        alert_id = f"{threshold.name}_{int(triggered_at.timestamp())}"
        
        message = f"{threshold.description} (current: {current_value}, threshold: {threshold.threshold_value})"
        
        return Alert(
            id=alert_id,
            threshold_name=threshold.name,
            severity=threshold.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            triggered_at=triggered_at,
            metadata={
                "metric_name": threshold.metric_name,
                "comparison": threshold.comparison,
                "duration_seconds": threshold.duration_seconds
            }
        )
    
    def _resolve_alert(self, alert_id: str, resolved_at: datetime):
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = resolved_at
            
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert.message}")
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending alert notification: {e}")
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: ID of alert to acknowledge
            
        Returns:
            True if alert was acknowledged
        """
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = datetime.now(timezone.utc)
            
            logger.info(f"Alert acknowledged: {alert.message}")
            return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """
        Get list of active alerts.
        
        Args:
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """
        Get alert history for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of alerts from the time period
        """
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if alert.triggered_at >= cutoff_time
        ]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """
        Get summary of alerting status.
        
        Returns:
            Dictionary with alert summary information
        """
        active_alerts = self.get_active_alerts()
        recent_alerts = self.get_alert_history(24)
        
        severity_counts = {severity.value: 0 for severity in AlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        return {
            "active_alert_count": len(active_alerts),
            "active_alerts_by_severity": severity_counts,
            "recent_alert_count_24h": len(recent_alerts),
            "threshold_count": len(self.thresholds),
            "enabled_threshold_count": len([t for t in self.thresholds.values() if t.enabled]),
            "notification_handler_count": len(self.notification_handlers),
            "last_evaluation": datetime.now(timezone.utc).isoformat()
        }
    
    def export_alerts_for_monitoring(self) -> Dict[str, Any]:
        """
        Export alert data for external monitoring systems.
        
        Returns:
            Dictionary with alert data for monitoring
        """
        return {
            "active_alerts": [alert.to_dict() for alert in self.get_active_alerts()],
            "alert_summary": self.get_alert_summary(),
            "thresholds": {
                name: {
                    "description": threshold.description,
                    "metric_name": threshold.metric_name,
                    "threshold_value": threshold.threshold_value,
                    "comparison": threshold.comparison,
                    "duration_seconds": threshold.duration_seconds,
                    "severity": threshold.severity.value,
                    "enabled": threshold.enabled
                }
                for name, threshold in self.thresholds.items()
            }
        }


# Global alert manager instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Get the global alert manager instance.
    
    Returns:
        The global AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def evaluate_alerts() -> List[Alert]:
    """
    Evaluate all alert thresholds and return new alerts.
    
    Returns:
        List of newly triggered alerts
    """
    manager = get_alert_manager()
    return manager.evaluate_thresholds()


def add_alert_notification_handler(handler: Callable[[Alert], None]):
    """
    Add a notification handler for alerts.
    
    Args:
        handler: Function to call when alerts are triggered
    """
    manager = get_alert_manager()
    manager.add_notification_handler(handler)


# Default notification handlers
def log_alert_handler(alert: Alert):
    """Default alert handler that logs alerts."""
    logger.warning(f"ALERT [{alert.severity.value.upper()}]: {alert.message}")


def console_alert_handler(alert: Alert):
    """Alert handler that prints to console."""
    print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.triggered_at}: {alert.message}")


# Initialize default handlers
def setup_default_alert_handlers():
    """Set up default alert notification handlers."""
    add_alert_notification_handler(log_alert_handler)
    add_alert_notification_handler(console_alert_handler)