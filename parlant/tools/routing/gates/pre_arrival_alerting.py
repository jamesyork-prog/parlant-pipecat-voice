"""
Health monitoring and alerting for Pre-Arrival gate.

This module provides comprehensive alerting for:
- High parsing failure rates (>5% of tickets)
- High timezone inference fallback rates (>10% of tickets)
- Processing latency exceeding targets (>300ms)
- Suspicious booking pattern increases
- Actionable troubleshooting information

Implements requirements:
- 20.1: Alert on high parsing failure rate
- 20.2: Alert on high timezone inference fallback rate
- 20.3: Alert on processing latency exceeding targets
- 20.4: Alert on suspicious booking pattern increases
- 20.5: Add actionable troubleshooting information to alerts
"""

import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class PreArrivalAlertSeverity(Enum):
    """Alert severity levels for pre-arrival gate."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PreArrivalAlertType(Enum):
    """Types of pre-arrival alerts."""
    PARSING_FAILURE_RATE = "parsing_failure_rate"
    TIMEZONE_FALLBACK_RATE = "timezone_fallback_rate"
    PROCESSING_LATENCY = "processing_latency"
    SUSPICIOUS_PATTERNS = "suspicious_patterns"
    DATA_QUALITY = "data_quality"


@dataclass
class PreArrivalAlertThreshold:
    """Configuration for a pre-arrival alert threshold."""
    name: str
    alert_type: PreArrivalAlertType
    description: str
    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte"
    duration_seconds: int  # How long condition must persist
    severity: PreArrivalAlertSeverity
    troubleshooting_guide: str
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
        
        # Both value and duration conditions must be met
        return value_condition_met and duration_exceeded


@dataclass
class PreArrivalAlert:
    """An active or resolved pre-arrival alert."""
    id: str
    alert_type: PreArrivalAlertType
    threshold_name: str
    severity: PreArrivalAlertSeverity
    message: str
    current_value: float
    threshold_value: float
    triggered_at: datetime
    troubleshooting_guide: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "alert_type": self.alert_type.value,
            "threshold_name": self.threshold_name,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "triggered_at": self.triggered_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "troubleshooting_guide": self.troubleshooting_guide,
            "metadata": self.metadata
        }


class PreArrivalAlertManager:
    """
    Manages alerting for the Pre-Arrival gate.
    
    Provides functionality for:
    - Threshold configuration and evaluation
    - Alert generation with troubleshooting guides
    - Alert notification and escalation
    - Alert history and reporting
    """
    
    def __init__(self):
        """Initialize the pre-arrival alert manager."""
        self.thresholds: Dict[str, PreArrivalAlertThreshold] = {}
        self.active_alerts: Dict[str, PreArrivalAlert] = {}
        self.alert_history: List[PreArrivalAlert] = []
        self.threshold_states: Dict[str, Dict[str, Any]] = {}
        self.notification_handlers: List[Callable[[PreArrivalAlert], None]] = []
        
        # Initialize default thresholds
        self._setup_default_thresholds()
        
        logger.info("PreArrivalAlertManager initialized with default thresholds")
    
    def _setup_default_thresholds(self):
        """Set up default alerting thresholds for pre-arrival gate."""
        
        # Parsing failure rate alerts (Requirement 20.1)
        self.add_threshold(PreArrivalAlertThreshold(
            name="parsing_failure_rate_warning",
            alert_type=PreArrivalAlertType.PARSING_FAILURE_RATE,
            description="Date parsing failure rate exceeds 5%",
            metric_name="parsing_failure_rate_percent",
            threshold_value=5.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=PreArrivalAlertSeverity.WARNING,
            troubleshooting_guide="""
TROUBLESHOOTING: High Date Parsing Failure Rate

Possible Causes:
1. New date format from Zapier not in configuration
2. Data quality issues in Freshdesk tickets
3. Zapier integration changes

Immediate Actions:
1. Check recent failed parsing logs for common patterns
2. Review router_config.json date_formats configuration
3. Examine sample failed tickets in Freshdesk

Resolution Steps:
1. Identify the new date format from logs
2. Add new format to router_config.json date_formats list
3. Hot reload configuration (no deployment needed)
4. Monitor parsing success rate for improvement

Prevention:
- Set up automated alerts for Zapier integration changes
- Maintain comprehensive date format test suite
- Document all known date format variations
            """.strip()
        ))
        
        self.add_threshold(PreArrivalAlertThreshold(
            name="parsing_failure_rate_critical",
            alert_type=PreArrivalAlertType.PARSING_FAILURE_RATE,
            description="Date parsing failure rate exceeds 15%",
            metric_name="parsing_failure_rate_percent",
            threshold_value=15.0,
            comparison="gt",
            duration_seconds=120,  # 2 minutes
            severity=PreArrivalAlertSeverity.CRITICAL,
            troubleshooting_guide="""
CRITICAL: Very High Date Parsing Failure Rate

IMMEDIATE ACTION REQUIRED

This indicates a major issue with date parsing that is affecting
a significant portion of tickets. Pre-arrival gate is degraded.

Emergency Response:
1. Check if Zapier integration has changed date format
2. Review last 10 failed parsing attempts in logs
3. Consider temporarily disabling pre-arrival gate if >25% failure rate
4. Escalate to engineering team immediately

Quick Fix Options:
1. Rollback recent configuration changes if any
2. Add emergency date format to configuration
3. Enable verbose logging for date parser
4. Monitor Freshdesk ticket format changes

Contact: engineering-oncall@company.com
Runbook: docs/runbooks/pre-arrival-parsing-failures.md
            """.strip()
        ))
        
        # Timezone inference fallback rate alerts (Requirement 20.2)
        self.add_threshold(PreArrivalAlertThreshold(
            name="timezone_fallback_rate_warning",
            alert_type=PreArrivalAlertType.TIMEZONE_FALLBACK_RATE,
            description="Timezone UTC fallback rate exceeds 10%",
            metric_name="utc_fallback_rate_percent",
            threshold_value=10.0,
            comparison="gt",
            duration_seconds=600,  # 10 minutes
            severity=PreArrivalAlertSeverity.WARNING,
            troubleshooting_guide="""
TROUBLESHOOTING: High Timezone UTC Fallback Rate

Possible Causes:
1. Missing timezone data in Zapier tickets
2. New locations not in city-to-timezone mapping
3. Invalid IANA timezone identifiers from source

Immediate Actions:
1. Check recent UTC fallback logs for location patterns
2. Review city_to_timezone_mapping in configuration
3. Examine timezone fields in sample tickets

Resolution Steps:
1. Identify common locations triggering UTC fallback
2. Add missing cities to city_to_timezone_mapping
3. Update configuration and hot reload
4. Monitor timezone inference success rate

Impact:
- Pre-arrival decisions still work but may be less accurate
- Customers in non-UTC timezones may see incorrect timing
- Audit logs will show UTC fallback warnings

Prevention:
- Expand city-to-timezone mapping proactively
- Monitor new location patterns in tickets
- Validate timezone data quality at source
            """.strip()
        ))
        
        self.add_threshold(PreArrivalAlertThreshold(
            name="timezone_fallback_rate_critical",
            alert_type=PreArrivalAlertType.TIMEZONE_FALLBACK_RATE,
            description="Timezone UTC fallback rate exceeds 25%",
            metric_name="utc_fallback_rate_percent",
            threshold_value=25.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=PreArrivalAlertSeverity.ERROR,
            troubleshooting_guide="""
ERROR: Very High Timezone UTC Fallback Rate

This indicates a major data quality issue with timezone information.
Pre-arrival accuracy is significantly degraded.

Emergency Response:
1. Check if Zapier stopped sending timezone data
2. Verify "Location Time Zone" field in recent tickets
3. Review Zapier integration configuration
4. Check for Freshdesk field mapping changes

Quick Fix Options:
1. Expand city-to-timezone mapping with common locations
2. Enable verbose timezone inference logging
3. Consider adding timezone validation at webhook entry
4. Escalate to data engineering team

Impact Assessment:
- Decisions may be incorrect for non-UTC timezones
- Customer experience degraded for international bookings
- Manual review recommended for affected tickets

Contact: data-engineering@company.com
Runbook: docs/runbooks/timezone-data-quality.md
            """.strip()
        ))
        
        # Processing latency alerts (Requirement 20.3)
        self.add_threshold(PreArrivalAlertThreshold(
            name="processing_latency_warning",
            alert_type=PreArrivalAlertType.PROCESSING_LATENCY,
            description="Pre-arrival processing latency P95 exceeds 300ms",
            metric_name="processing_latency_p95_ms",
            threshold_value=300.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=PreArrivalAlertSeverity.WARNING,
            troubleshooting_guide="""
TROUBLESHOOTING: High Pre-Arrival Processing Latency

Target: <200ms P95, Current: >300ms P95

Possible Causes:
1. Timezone cache not working effectively
2. Complex date format attempts taking too long
3. System resource contention
4. Database connection issues

Immediate Actions:
1. Check component-level latency breakdown in logs
2. Review timezone cache hit rate
3. Monitor system CPU and memory usage
4. Check database connection pool status

Resolution Steps:
1. Identify slow component (parsing, timezone, comparison)
2. Review date format priority ordering
3. Verify timezone caching is enabled
4. Check for slow database queries
5. Consider scaling resources if needed

Performance Optimization:
- Ensure most common date formats are first in list
- Verify pytz timezone cache is working
- Check for unnecessary logging in hot path
- Profile slow tickets to identify patterns

Prevention:
- Set up continuous performance monitoring
- Maintain performance regression test suite
- Document performance optimization techniques
            """.strip()
        ))
        
        self.add_threshold(PreArrivalAlertThreshold(
            name="processing_latency_critical",
            alert_type=PreArrivalAlertType.PROCESSING_LATENCY,
            description="Pre-arrival processing latency P95 exceeds 500ms",
            metric_name="processing_latency_p95_ms",
            threshold_value=500.0,
            comparison="gt",
            duration_seconds=120,  # 2 minutes
            severity=PreArrivalAlertSeverity.ERROR,
            troubleshooting_guide="""
ERROR: Critical Pre-Arrival Processing Latency

Target: <200ms P95, Current: >500ms P95

IMMEDIATE ACTION REQUIRED

This latency level indicates a serious performance issue that
is degrading the fast path benefit of the pre-arrival gate.

Emergency Response:
1. Check if system is under unusual load
2. Review recent code or configuration changes
3. Check for database connection issues
4. Monitor system resources (CPU, memory, disk I/O)

Quick Fix Options:
1. Restart pre-arrival gate service if memory leak suspected
2. Increase timezone cache size
3. Reduce date format list to most common formats only
4. Enable circuit breaker if latency continues

Impact Assessment:
- Fast path benefit significantly reduced
- Webhook timeouts may increase
- Customer experience degraded
- Cost savings reduced due to slower processing

Escalation:
- Notify engineering team immediately
- Consider temporary gate disable if >1000ms
- Prepare rollback plan

Contact: engineering-oncall@company.com
Runbook: docs/runbooks/pre-arrival-performance-degradation.md
            """.strip()
        ))
        
        # Suspicious booking pattern alerts (Requirement 20.4)
        self.add_threshold(PreArrivalAlertThreshold(
            name="suspicious_patterns_warning",
            alert_type=PreArrivalAlertType.SUSPICIOUS_PATTERNS,
            description="Suspicious booking pattern rate exceeds 5%",
            metric_name="suspicious_pattern_rate_percent",
            threshold_value=5.0,
            comparison="gt",
            duration_seconds=600,  # 10 minutes
            severity=PreArrivalAlertSeverity.WARNING,
            troubleshooting_guide="""
TROUBLESHOOTING: Increased Suspicious Booking Patterns

Suspicious patterns include:
- Bookings >12 months in future
- Bookings >7 days in past
- Very short notice (<15 minutes)
- Extremely long durations (>7 days)

Possible Causes:
1. Data quality issues in booking system
2. Test data in production environment
3. Fraudulent booking attempts
4. System clock synchronization issues

Immediate Actions:
1. Review suspicious pattern logs for common characteristics
2. Check sample tickets flagged as suspicious
3. Verify booking data source integrity
4. Check for test data in production

Resolution Steps:
1. Identify pattern type (far future, far past, etc.)
2. Determine if data issue or legitimate edge cases
3. Adjust suspicious pattern thresholds if needed
4. Coordinate with booking system team if data issue

Impact:
- Tickets are still processed correctly
- Manual review flags added to decisions
- No customer impact, just increased review workload

Prevention:
- Implement data validation at booking creation
- Set up automated test data detection
- Monitor booking data quality metrics
- Document legitimate edge cases
            """.strip()
        ))
        
        self.add_threshold(PreArrivalAlertThreshold(
            name="suspicious_patterns_critical",
            alert_type=PreArrivalAlertType.SUSPICIOUS_PATTERNS,
            description="Suspicious booking pattern rate exceeds 15%",
            metric_name="suspicious_pattern_rate_percent",
            threshold_value=15.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=PreArrivalAlertSeverity.ERROR,
            troubleshooting_guide="""
ERROR: Very High Suspicious Booking Pattern Rate

This indicates a major data quality issue or potential security incident.

Emergency Response:
1. Check if test data is being sent to production
2. Review for potential fraudulent activity patterns
3. Verify booking system data integrity
4. Check for system clock synchronization issues

Quick Fix Options:
1. Filter out test data at webhook entry if identified
2. Coordinate with booking system team immediately
3. Enable enhanced logging for suspicious patterns
4. Consider temporary manual review for all pre-arrivals

Security Considerations:
- Check for coordinated fraudulent booking attempts
- Review IP addresses of suspicious bookings
- Coordinate with fraud prevention team
- Document patterns for security analysis

Impact Assessment:
- High manual review workload
- Potential fraud risk if not addressed
- Data quality issues affecting decisions
- Customer trust impact if fraud-related

Contact: fraud-prevention@company.com, data-engineering@company.com
Runbook: docs/runbooks/suspicious-booking-patterns.md
            """.strip()
        ))
        
        # Webhook timeout rate alert
        self.add_threshold(PreArrivalAlertThreshold(
            name="webhook_timeout_rate_warning",
            alert_type=PreArrivalAlertType.PROCESSING_LATENCY,
            description="Webhook timeout rate exceeds 5%",
            metric_name="webhook_timeout_rate_percent",
            threshold_value=5.0,
            comparison="gt",
            duration_seconds=300,  # 5 minutes
            severity=PreArrivalAlertSeverity.WARNING,
            troubleshooting_guide="""
TROUBLESHOOTING: High Webhook Timeout Rate

Target: <1% timeout rate, Current: >5%

Possible Causes:
1. Processing latency exceeding 8-second limit
2. Freshdesk API slow response times
3. Network connectivity issues
4. System resource contention

Immediate Actions:
1. Check processing latency metrics
2. Review Freshdesk API response times
3. Monitor network connectivity
4. Check system resource usage

Resolution Steps:
1. Identify bottleneck (gate processing vs ticket updates)
2. Optimize slow components
3. Consider increasing timeout threshold if appropriate
4. Review webhook payload size and complexity

Impact:
- Tickets require manual review
- Timeout notes added to tickets
- Customer experience degraded
- Manual workload increased

Prevention:
- Maintain processing latency <5 seconds
- Monitor Freshdesk API performance
- Set up network monitoring
- Optimize ticket update operations
            """.strip()
        ))
    
    def add_threshold(self, threshold: PreArrivalAlertThreshold):
        """
        Add an alert threshold.
        
        Args:
            threshold: PreArrivalAlertThreshold configuration
        """
        self.thresholds[threshold.name] = threshold
        self.threshold_states[threshold.name] = {
            "first_breach_time": None,
            "last_check_time": None,
            "last_value": None
        }
        
        logger.info(f"Added pre-arrival alert threshold: {threshold.name}")
    
    def remove_threshold(self, threshold_name: str):
        """
        Remove an alert threshold.
        
        Args:
            threshold_name: Name of threshold to remove
        """
        if threshold_name in self.thresholds:
            del self.thresholds[threshold_name]
            del self.threshold_states[threshold_name]
            logger.info(f"Removed pre-arrival alert threshold: {threshold_name}")
    
    def add_notification_handler(self, handler: Callable[[PreArrivalAlert], None]):
        """
        Add a notification handler for alerts.
        
        Args:
            handler: Function to call when alerts are triggered
        """
        self.notification_handlers.append(handler)
        logger.info("Added pre-arrival notification handler")
    
    def evaluate_thresholds(self, current_metrics: Dict[str, float]) -> List[PreArrivalAlert]:
        """
        Evaluate all thresholds against current metrics.
        
        Args:
            current_metrics: Dictionary of current metric values
            
        Returns:
            List of newly triggered alerts
        """
        current_time = datetime.now(timezone.utc)
        new_alerts = []
        
        for threshold_name, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue
            
            state = self.threshold_states[threshold_name]
            current_value = current_metrics.get(threshold.metric_name)
            
            if current_value is None:
                logger.debug(f"Metric {threshold.metric_name} not available for threshold {threshold_name}")
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
                    
                    logger.warning(
                        f"Pre-arrival alert triggered: {alert.message}",
                        extra={
                            "alert_type": alert.alert_type.value,
                            "severity": alert.severity.value,
                            "current_value": current_value,
                            "threshold_value": threshold.threshold_value
                        }
                    )
            
            else:
                # Threshold not breached
                state["first_breach_time"] = None
                
                # Check if we should resolve an active alert
                alert_id = None
                for aid, alert in self.active_alerts.items():
                    if alert.threshold_name == threshold_name:
                        alert_id = aid
                        break
                
                if alert_id:
                    self._resolve_alert(alert_id, current_time)
        
        return new_alerts
    
    def _evaluate_threshold_condition(self, threshold: PreArrivalAlertThreshold, current_value: float) -> bool:
        """Evaluate threshold condition."""
        if threshold.comparison == "gt":
            return current_value > threshold.threshold_value
        elif threshold.comparison == "lt":
            return current_value < threshold.threshold_value
        elif threshold.comparison == "gte":
            return current_value >= threshold.threshold_value
        elif threshold.comparison == "lte":
            return current_value <= threshold.threshold_value
        else:
            return False
    
    def _create_alert(
        self, 
        threshold: PreArrivalAlertThreshold, 
        current_value: float, 
        triggered_at: datetime
    ) -> PreArrivalAlert:
        """Create a new alert."""
        alert_id = f"pre_arrival_{threshold.name}_{int(triggered_at.timestamp())}"
        
        message = (
            f"Pre-Arrival Gate: {threshold.description} "
            f"(current: {current_value:.2f}, threshold: {threshold.threshold_value:.2f})"
        )
        
        return PreArrivalAlert(
            id=alert_id,
            alert_type=threshold.alert_type,
            threshold_name=threshold.name,
            severity=threshold.severity,
            message=message,
            current_value=current_value,
            threshold_value=threshold.threshold_value,
            triggered_at=triggered_at,
            troubleshooting_guide=threshold.troubleshooting_guide,
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
            alert.resolved_at = resolved_at
            
            del self.active_alerts[alert_id]
            
            logger.info(
                f"Pre-arrival alert resolved: {alert.message}",
                extra={
                    "alert_type": alert.alert_type.value,
                    "duration_seconds": (resolved_at - alert.triggered_at).total_seconds()
                }
            )
    
    def _send_notifications(self, alert: PreArrivalAlert):
        """Send notifications for an alert."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error sending pre-arrival alert notification: {e}")
    
    def get_active_alerts(
        self, 
        alert_type: Optional[PreArrivalAlertType] = None,
        severity: Optional[PreArrivalAlertSeverity] = None
    ) -> List[PreArrivalAlert]:
        """
        Get list of active alerts.
        
        Args:
            alert_type: Optional alert type filter
            severity: Optional severity filter
            
        Returns:
            List of active alerts
        """
        alerts = list(self.active_alerts.values())
        
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)
    
    def get_alert_history(self, hours: int = 24) -> List[PreArrivalAlert]:
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
        
        severity_counts = {severity.value: 0 for severity in PreArrivalAlertSeverity}
        for alert in active_alerts:
            severity_counts[alert.severity.value] += 1
        
        type_counts = {alert_type.value: 0 for alert_type in PreArrivalAlertType}
        for alert in active_alerts:
            type_counts[alert.alert_type.value] += 1
        
        return {
            "active_alert_count": len(active_alerts),
            "active_alerts_by_severity": severity_counts,
            "active_alerts_by_type": type_counts,
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
                    "alert_type": threshold.alert_type.value,
                    "description": threshold.description,
                    "metric_name": threshold.metric_name,
                    "threshold_value": threshold.threshold_value,
                    "comparison": threshold.comparison,
                    "duration_seconds": threshold.duration_seconds,
                    "severity": threshold.severity.value,
                    "enabled": threshold.enabled,
                    "troubleshooting_guide": threshold.troubleshooting_guide
                }
                for name, threshold in self.thresholds.items()
            }
        }


# Global alert manager instance
_pre_arrival_alert_manager: Optional[PreArrivalAlertManager] = None


def get_pre_arrival_alert_manager() -> PreArrivalAlertManager:
    """
    Get the global pre-arrival alert manager instance.
    
    Returns:
        The global PreArrivalAlertManager instance
    """
    global _pre_arrival_alert_manager
    if _pre_arrival_alert_manager is None:
        _pre_arrival_alert_manager = PreArrivalAlertManager()
    return _pre_arrival_alert_manager


def evaluate_pre_arrival_alerts(current_metrics: Dict[str, float]) -> List[PreArrivalAlert]:
    """
    Evaluate all pre-arrival alert thresholds and return new alerts.
    
    Args:
        current_metrics: Dictionary of current metric values
        
    Returns:
        List of newly triggered alerts
    """
    manager = get_pre_arrival_alert_manager()
    return manager.evaluate_thresholds(current_metrics)


def add_pre_arrival_notification_handler(handler: Callable[[PreArrivalAlert], None]):
    """
    Add a notification handler for pre-arrival alerts.
    
    Args:
        handler: Function to call when alerts are triggered
    """
    manager = get_pre_arrival_alert_manager()
    manager.add_notification_handler(handler)


# Default notification handlers
def log_pre_arrival_alert_handler(alert: PreArrivalAlert):
    """Default alert handler that logs alerts with troubleshooting guide."""
    logger.warning(
        f"PRE-ARRIVAL ALERT [{alert.severity.value.upper()}]: {alert.message}\n\n"
        f"Troubleshooting Guide:\n{alert.troubleshooting_guide}"
    )


def console_pre_arrival_alert_handler(alert: PreArrivalAlert):
    """Alert handler that prints to console with troubleshooting guide."""
    print(f"\n{'='*80}")
    print(f"🚨 PRE-ARRIVAL ALERT [{alert.severity.value.upper()}]")
    print(f"{'='*80}")
    print(f"Time: {alert.triggered_at}")
    print(f"Type: {alert.alert_type.value}")
    print(f"Message: {alert.message}")
    print(f"\n{alert.troubleshooting_guide}")
    print(f"{'='*80}\n")


# Initialize default handlers
def setup_default_pre_arrival_alert_handlers():
    """Set up default alert notification handlers for pre-arrival gate."""
    add_pre_arrival_notification_handler(log_pre_arrival_alert_handler)
    add_pre_arrival_notification_handler(console_pre_arrival_alert_handler)
