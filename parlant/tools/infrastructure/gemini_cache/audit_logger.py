"""
Comprehensive Audit Logger for Gemini Context Caching.

This module provides comprehensive audit logging for all cache effectiveness operations:
- Cache metrics logging with correlation IDs
- Content loading operations and failures
- Prompt construction timing and structure
- Cost calculations and savings tracking
- Performance metrics and optimization events
- Structured logging with complete troubleshooting trails

Requirements: 14.1, 14.2, 14.3, 14.4, 14.5
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum

from .models import CacheMetrics, StaticContent, CacheConfiguration, ContentLoadingStatus
from ..database import get_db_connection, return_db_connection

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of audit events for cache operations."""
    CACHE_RESPONSE_PROCESSED = "cache_response_processed"
    PROMPT_CONSTRUCTED = "prompt_constructed"
    CONTENT_LOADING_SUCCESS = "content_loading_success"
    CONTENT_LOADING_FAILURE = "content_loading_failure"
    CONTENT_CHANGE_DETECTED = "content_change_detected"
    FALLBACK_MODE_ACTIVATED = "fallback_mode_activated"
    FALLBACK_MODE_DEACTIVATED = "fallback_mode_deactivated"
    COST_CALCULATION = "cost_calculation"
    PERFORMANCE_ALERT = "performance_alert"
    CACHE_EFFECTIVENESS_ALERT = "cache_effectiveness_alert"
    CONFIGURATION_LOADED = "configuration_loaded"
    CONFIGURATION_ERROR = "configuration_error"


class AuditSeverity(Enum):
    """Severity levels for audit events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Structured audit event for cache operations."""
    event_id: str
    correlation_id: str
    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime
    component: str
    operation: str
    details: Dict[str, Any]
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for logging."""
        return {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "component": self.component,
            "operation": self.operation,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "request_id": self.request_id,
        }


class CacheAuditLogger:
    """
    Comprehensive audit logger for Gemini Context Caching operations.
    
    Provides structured logging with correlation IDs for complete request tracing,
    detailed cache effectiveness metrics, performance monitoring, and troubleshooting support.
    """
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        """
        Initialize cache audit logger.
        
        Args:
            config: Cache configuration for audit settings
        """
        self.config = config
        self._correlation_id_stack: List[str] = []
        self._current_correlation_id: Optional[str] = None
        
        # Audit settings
        self.log_to_database = True
        self.log_to_file = True
        self.retention_days = 30
        
        # Performance tracking
        self._operation_start_times: Dict[str, float] = {}
        
        logger.info("CacheAuditLogger initialized")
    
    def start_correlation(self, operation: str = "cache_operation") -> str:
        """
        Start a new correlation context for request tracing.
        
        Args:
            operation: Name of the operation being traced
            
        Returns:
            Correlation ID for this context
        """
        correlation_id = f"cache_{uuid.uuid4().hex[:12]}"
        self._correlation_id_stack.append(correlation_id)
        self._current_correlation_id = correlation_id
        
        logger.debug(f"Started correlation context: {correlation_id} for {operation}")
        return correlation_id
    
    def end_correlation(self) -> Optional[str]:
        """
        End the current correlation context.
        
        Returns:
            The ended correlation ID, or None if no context was active
        """
        if self._correlation_id_stack:
            ended_id = self._correlation_id_stack.pop()
            self._current_correlation_id = (
                self._correlation_id_stack[-1] if self._correlation_id_stack else None
            )
            logger.debug(f"Ended correlation context: {ended_id}")
            return ended_id
        return None
    
    def get_correlation_id(self) -> str:
        """
        Get current correlation ID, creating one if none exists.
        
        Returns:
            Current correlation ID
        """
        if not self._current_correlation_id:
            return self.start_correlation("auto_generated")
        return self._current_correlation_id
    
    def log_cache_response_processed(
        self,
        metrics: CacheMetrics,
        request_id: str,
        processing_duration_ms: Optional[float] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log cache response processing with detailed metrics.
        
        Requirement 14.1: Log cached_content_token_count and cost savings
        Requirement 14.4: Log cache effectiveness and savings per request
        
        Args:
            metrics: Cache metrics from response processing
            request_id: Request identifier
            processing_duration_ms: Time taken to process response
            additional_details: Additional context
        """
        details = {
            "request_id": request_id,
            "cached_tokens": metrics.cached_tokens,
            "dynamic_tokens": metrics.dynamic_tokens,
            "total_tokens": metrics.total_tokens,
            "cost_with_cache": metrics.cost_with_cache,
            "cost_without_cache": metrics.cost_without_cache,
            "savings_usd": metrics.savings,
            "cache_effectiveness_percent": metrics.cache_effectiveness,
            "timestamp": metrics.timestamp.isoformat(),
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.CACHE_RESPONSE_PROCESSED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            component="RequestBuilder",
            operation="process_response",
            details=details,
            duration_ms=processing_duration_ms,
            request_id=request_id,
        )
        
        self._log_event(event)
        
        # Alert on low cache effectiveness
        if metrics.cache_effectiveness < 70:
            self.log_cache_effectiveness_alert(
                metrics.cache_effectiveness,
                request_id,
                "Low cache effectiveness detected"
            )
    
    def log_prompt_constructed(
        self,
        request_id: str,
        static_content_tokens: int,
        dynamic_content_tokens: int,
        total_prompt_tokens: int,
        construction_duration_ms: float,
        prompt_structure_valid: bool = True,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log prompt construction with structure and timing details.
        
        Requirement 14.2: Log prompt structure and static content size
        
        Args:
            request_id: Request identifier
            static_content_tokens: Tokens in static content
            dynamic_content_tokens: Tokens in dynamic content
            total_prompt_tokens: Total prompt tokens
            construction_duration_ms: Time taken to construct prompt
            prompt_structure_valid: Whether prompt structure is valid
            additional_details: Additional context
        """
        details = {
            "request_id": request_id,
            "static_content_tokens": static_content_tokens,
            "dynamic_content_tokens": dynamic_content_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "static_content_ratio": (static_content_tokens / total_prompt_tokens * 100) if total_prompt_tokens > 0 else 0,
            "prompt_structure_valid": prompt_structure_valid,
            "construction_duration_ms": construction_duration_ms,
        }
        
        if additional_details:
            details.update(additional_details)
        
        # Determine severity based on performance and structure
        severity = AuditSeverity.INFO
        if construction_duration_ms > 10:  # Target is <10ms
            severity = AuditSeverity.WARNING
        elif not prompt_structure_valid:
            severity = AuditSeverity.WARNING
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.PROMPT_CONSTRUCTED,
            severity=severity,
            timestamp=datetime.now(),
            component="RequestBuilder",
            operation="build_request",
            details=details,
            duration_ms=construction_duration_ms,
            request_id=request_id,
        )
        
        self._log_event(event)
        
        # Performance alert if construction is slow
        if construction_duration_ms > 10:
            self.log_performance_alert(
                "slow_prompt_construction",
                f"Prompt construction took {construction_duration_ms:.1f}ms (target: <10ms)",
                {"construction_duration_ms": construction_duration_ms, "request_id": request_id}
            )
    
    def log_content_loading_success(
        self,
        static_content: StaticContent,
        loading_duration_ms: float,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log successful content loading with detailed metrics.
        
        Args:
            static_content: Loaded static content
            loading_duration_ms: Time taken to load content
            additional_details: Additional context
        """
        details = {
            "total_tokens": static_content.total_tokens,
            "content_hash": static_content.content_hash,
            "loaded_at": static_content.loaded_at.isoformat(),
            "loading_duration_ms": loading_duration_ms,
            "content_types_loaded": [],
            "content_types_failed": [],
            "content_types_fallback": [],
        }
        
        # Analyze loading statuses
        for content_type, status in static_content.loading_statuses.items():
            if status.status.value == "loaded":
                details["content_types_loaded"].append({
                    "type": content_type.value,
                    "token_count": status.token_count,
                    "file_path": status.file_path,
                })
            elif status.status.value == "failed":
                details["content_types_failed"].append({
                    "type": content_type.value,
                    "error": status.error_message,
                    "file_path": status.file_path,
                })
            elif status.status.value == "fallback":
                details["content_types_fallback"].append({
                    "type": content_type.value,
                    "reason": status.error_message,
                    "token_count": status.token_count,
                })
        
        if additional_details:
            details.update(additional_details)
        
        # Determine severity based on loading results
        severity = AuditSeverity.INFO
        if details["content_types_failed"] or details["content_types_fallback"]:
            severity = AuditSeverity.WARNING
        if loading_duration_ms > 3000:  # Target is <3s
            severity = AuditSeverity.WARNING
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.CONTENT_LOADING_SUCCESS,
            severity=severity,
            timestamp=datetime.now(),
            component="ContentLoader",
            operation="initialize_content",
            details=details,
            duration_ms=loading_duration_ms,
        )
        
        self._log_event(event)
        
        # Performance alert if loading is slow
        if loading_duration_ms > 3000:
            self.log_performance_alert(
                "slow_content_loading",
                f"Content loading took {loading_duration_ms:.1f}ms (target: <3000ms)",
                {"loading_duration_ms": loading_duration_ms}
            )
    
    def log_content_loading_failure(
        self,
        error_message: str,
        failure_reason: str,
        fallback_actions: List[str],
        loading_duration_ms: Optional[float] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log content loading failure with detailed error information.
        
        Requirement 14.3: Log failure reason and fallback actions
        
        Args:
            error_message: Error that occurred
            failure_reason: Detailed reason for failure
            fallback_actions: Actions taken to handle failure
            loading_duration_ms: Time taken before failure
            additional_details: Additional context
        """
        details = {
            "failure_reason": failure_reason,
            "fallback_actions": fallback_actions,
            "fallback_actions_count": len(fallback_actions),
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.CONTENT_LOADING_FAILURE,
            severity=AuditSeverity.ERROR,
            timestamp=datetime.now(),
            component="ContentLoader",
            operation="initialize_content",
            details=details,
            duration_ms=loading_duration_ms,
            error_message=error_message,
        )
        
        self._log_event(event)
    
    def log_content_change_detected(
        self,
        changed_files: List[str],
        change_type: str,
        reload_triggered: bool,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log content change detection and reload actions.
        
        Args:
            changed_files: List of files that changed
            change_type: Type of change detected
            reload_triggered: Whether content reload was triggered
            additional_details: Additional context
        """
        details = {
            "changed_files": changed_files,
            "change_type": change_type,
            "reload_triggered": reload_triggered,
            "changed_files_count": len(changed_files),
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.CONTENT_CHANGE_DETECTED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            component="ContentLoader",
            operation="check_content_changed",
            details=details,
        )
        
        self._log_event(event)
    
    def log_fallback_mode_activated(
        self,
        reason: str,
        failed_content_types: List[str],
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log fallback mode activation with detailed context.
        
        Args:
            reason: Reason for activating fallback mode
            failed_content_types: Content types that failed to load
            additional_details: Additional context
        """
        details = {
            "activation_reason": reason,
            "failed_content_types": failed_content_types,
            "failed_content_count": len(failed_content_types),
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.FALLBACK_MODE_ACTIVATED,
            severity=AuditSeverity.WARNING,
            timestamp=datetime.now(),
            component="ContentLoader",
            operation="activate_fallback_mode",
            details=details,
        )
        
        self._log_event(event)
    
    def log_fallback_mode_deactivated(
        self,
        duration_minutes: float,
        restoration_attempts: int,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log fallback mode deactivation with restoration metrics.
        
        Args:
            duration_minutes: How long fallback mode was active
            restoration_attempts: Number of restoration attempts made
            additional_details: Additional context
        """
        details = {
            "fallback_duration_minutes": duration_minutes,
            "restoration_attempts": restoration_attempts,
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.FALLBACK_MODE_DEACTIVATED,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            component="ContentLoader",
            operation="deactivate_fallback_mode",
            details=details,
        )
        
        self._log_event(event)
    
    def log_cost_calculation(
        self,
        cached_tokens: int,
        dynamic_tokens: int,
        total_cost_with_cache: float,
        total_cost_without_cache: float,
        savings: float,
        discount_percentage: float,
        request_id: Optional[str] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log detailed cost calculation for transparency and validation.
        
        Requirement 14.4: Log cache effectiveness and savings per request
        
        Args:
            cached_tokens: Number of cached tokens
            dynamic_tokens: Number of dynamic tokens
            total_cost_with_cache: Total cost with caching
            total_cost_without_cache: Total cost without caching
            savings: Cost savings amount
            discount_percentage: Cache discount percentage applied
            request_id: Associated request ID
            additional_details: Additional context
        """
        details = {
            "cached_tokens": cached_tokens,
            "dynamic_tokens": dynamic_tokens,
            "total_tokens": cached_tokens + dynamic_tokens,
            "total_cost_with_cache_usd": total_cost_with_cache,
            "total_cost_without_cache_usd": total_cost_without_cache,
            "savings_usd": savings,
            "savings_percentage": (savings / total_cost_without_cache * 100) if total_cost_without_cache > 0 else 0,
            "discount_percentage": discount_percentage,
            "cache_ratio": (cached_tokens / (cached_tokens + dynamic_tokens) * 100) if (cached_tokens + dynamic_tokens) > 0 else 0,
        }
        
        if additional_details:
            details.update(additional_details)
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.COST_CALCULATION,
            severity=AuditSeverity.INFO,
            timestamp=datetime.now(),
            component="RequestBuilder",
            operation="calculate_costs",
            details=details,
            request_id=request_id,
        )
        
        self._log_event(event)
    
    def log_performance_alert(
        self,
        alert_type: str,
        message: str,
        performance_data: Dict[str, Any],
        severity: AuditSeverity = AuditSeverity.WARNING
    ):
        """
        Log performance alerts for optimization analysis.
        
        Requirement 14.5: Provide complete trail for optimization
        
        Args:
            alert_type: Type of performance alert
            message: Alert message
            performance_data: Performance metrics data
            severity: Alert severity
        """
        details = {
            "alert_type": alert_type,
            "performance_data": performance_data,
        }
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.PERFORMANCE_ALERT,
            severity=severity,
            timestamp=datetime.now(),
            component="PerformanceMonitor",
            operation="check_performance",
            details=details,
            error_message=message,
        )
        
        self._log_event(event)
    
    def log_cache_effectiveness_alert(
        self,
        effectiveness_percentage: float,
        request_id: str,
        message: str,
        threshold: float = 70.0
    ):
        """
        Log cache effectiveness alerts for optimization.
        
        Args:
            effectiveness_percentage: Current cache effectiveness
            request_id: Associated request ID
            message: Alert message
            threshold: Effectiveness threshold that was breached
        """
        details = {
            "effectiveness_percentage": effectiveness_percentage,
            "threshold": threshold,
            "effectiveness_gap": threshold - effectiveness_percentage,
            "request_id": request_id,
        }
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=AuditEventType.CACHE_EFFECTIVENESS_ALERT,
            severity=AuditSeverity.WARNING,
            timestamp=datetime.now(),
            component="CacheMonitor",
            operation="check_effectiveness",
            details=details,
            error_message=message,
            request_id=request_id,
        )
        
        self._log_event(event)
    
    def log_configuration_loaded(
        self,
        config_source: str,
        config_valid: bool,
        validation_errors: Optional[List[str]] = None,
        additional_details: Optional[Dict[str, Any]] = None
    ):
        """
        Log configuration loading and validation results.
        
        Args:
            config_source: Source of configuration
            config_valid: Whether configuration is valid
            validation_errors: Any validation errors found
            additional_details: Additional context
        """
        details = {
            "config_source": config_source,
            "config_valid": config_valid,
            "validation_errors": validation_errors or [],
            "validation_error_count": len(validation_errors) if validation_errors else 0,
        }
        
        if additional_details:
            details.update(additional_details)
        
        severity = AuditSeverity.INFO if config_valid else AuditSeverity.ERROR
        event_type = AuditEventType.CONFIGURATION_LOADED if config_valid else AuditEventType.CONFIGURATION_ERROR
        
        event = AuditEvent(
            event_id=self._generate_event_id(),
            correlation_id=self.get_correlation_id(),
            event_type=event_type,
            severity=severity,
            timestamp=datetime.now(),
            component="ConfigurationManager",
            operation="load_config",
            details=details,
            error_message="; ".join(validation_errors) if validation_errors else None,
        )
        
        self._log_event(event)
    
    def get_audit_trail(
        self,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get audit trail for troubleshooting and analysis.
        
        Requirement 14.5: Provide complete trail for optimization
        
        Args:
            correlation_id: Filter by correlation ID
            request_id: Filter by request ID
            start_time: Filter by start time
            end_time: Filter by end time
            event_types: Filter by event types
            
        Returns:
            List of audit events matching criteria
        """
        try:
            if not self.log_to_database:
                logger.warning("Database logging disabled, cannot retrieve audit trail")
                return []
            
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for audit trail")
                return []
            
            cursor = conn.cursor()
            
            # Build query conditions
            conditions = []
            params = []
            
            if correlation_id:
                conditions.append("event_details->>'correlation_id' = %s")
                params.append(correlation_id)
            
            if request_id:
                conditions.append("event_details->>'request_id' = %s")
                params.append(request_id)
            
            if start_time:
                conditions.append("timestamp >= %s")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= %s")
                params.append(end_time)
            
            if event_types:
                event_type_values = [et.value for et in event_types]
                conditions.append("event_details->>'event_type' = ANY(%s)")
                params.append(event_type_values)
            
            # Add cache-specific filter
            conditions.append("event_details->>'component' LIKE 'Cache%' OR event_details->>'component' LIKE '%Cache%'")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor.execute(f"""
                SELECT event_details, timestamp
                FROM agent_audit_log 
                WHERE {where_clause}
                ORDER BY timestamp ASC
            """, params)
            
            results = cursor.fetchall()
            cursor.close()
            return_db_connection(conn)
            
            # Format results
            audit_trail = []
            for row in results:
                event_data = row[0]
                event_data["db_timestamp"] = row[1].isoformat()
                audit_trail.append(event_data)
            
            return audit_trail
            
        except Exception as e:
            logger.error(f"Failed to get audit trail: {e}")
            if conn:
                return_db_connection(conn)
            return []
    
    def get_performance_summary(
        self,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get performance summary for optimization analysis.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Performance summary with optimization recommendations
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get performance-related events
        performance_events = self.get_audit_trail(
            start_time=start_time,
            end_time=end_time,
            event_types=[
                AuditEventType.PROMPT_CONSTRUCTED,
                AuditEventType.CONTENT_LOADING_SUCCESS,
                AuditEventType.PERFORMANCE_ALERT,
            ]
        )
        
        # Analyze performance metrics
        prompt_construction_times = []
        content_loading_times = []
        performance_alerts = []
        
        for event in performance_events:
            if event.get("event_type") == AuditEventType.PROMPT_CONSTRUCTED.value:
                if "construction_duration_ms" in event.get("details", {}):
                    prompt_construction_times.append(event["details"]["construction_duration_ms"])
            elif event.get("event_type") == AuditEventType.CONTENT_LOADING_SUCCESS.value:
                if "loading_duration_ms" in event.get("details", {}):
                    content_loading_times.append(event["details"]["loading_duration_ms"])
            elif event.get("event_type") == AuditEventType.PERFORMANCE_ALERT.value:
                performance_alerts.append(event)
        
        # Calculate statistics
        summary = {
            "analysis_period": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "hours": hours,
            },
            "prompt_construction": {
                "total_operations": len(prompt_construction_times),
                "avg_duration_ms": sum(prompt_construction_times) / len(prompt_construction_times) if prompt_construction_times else 0,
                "max_duration_ms": max(prompt_construction_times) if prompt_construction_times else 0,
                "slow_operations": len([t for t in prompt_construction_times if t > 10]),
                "target_ms": 10,
            },
            "content_loading": {
                "total_operations": len(content_loading_times),
                "avg_duration_ms": sum(content_loading_times) / len(content_loading_times) if content_loading_times else 0,
                "max_duration_ms": max(content_loading_times) if content_loading_times else 0,
                "slow_operations": len([t for t in content_loading_times if t > 3000]),
                "target_ms": 3000,
            },
            "alerts": {
                "total_alerts": len(performance_alerts),
                "alert_types": {},
            },
        }
        
        # Analyze alert types
        for alert in performance_alerts:
            alert_type = alert.get("details", {}).get("alert_type", "unknown")
            if alert_type not in summary["alerts"]["alert_types"]:
                summary["alerts"]["alert_types"][alert_type] = 0
            summary["alerts"]["alert_types"][alert_type] += 1
        
        return summary
    
    def cleanup_old_logs(self, retention_days: Optional[int] = None):
        """
        Clean up old audit logs based on retention policy.
        
        Args:
            retention_days: Days to retain logs (uses instance default if None)
        """
        if not self.log_to_database:
            logger.debug("Database logging disabled, no cleanup needed")
            return
        
        retention = retention_days or self.retention_days
        cutoff_date = datetime.now() - timedelta(days=retention)
        
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for log cleanup")
                return
            
            cursor = conn.cursor()
            
            # Delete old cache-related audit logs
            cursor.execute("""
                DELETE FROM agent_audit_log 
                WHERE timestamp < %s 
                AND (
                    event_details->>'component' LIKE 'Cache%' 
                    OR event_details->>'component' LIKE '%Cache%'
                    OR event_details->>'event_type' LIKE 'cache_%'
                )
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Cleaned up {deleted_count} old cache audit logs (retention: {retention} days)")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")
            if conn:
                return_db_connection(conn)
    
    def _log_event(self, event: AuditEvent):
        """
        Log audit event to configured destinations.
        
        Args:
            event: Audit event to log
        """
        event_dict = event.to_dict()
        
        # Log to application logger with appropriate level
        log_level = {
            AuditSeverity.DEBUG: logging.DEBUG,
            AuditSeverity.INFO: logging.INFO,
            AuditSeverity.WARNING: logging.WARNING,
            AuditSeverity.ERROR: logging.ERROR,
            AuditSeverity.CRITICAL: logging.CRITICAL,
        }.get(event.severity, logging.INFO)
        
        logger.log(
            log_level,
            f"CACHE_AUDIT [{event.event_type.value}] {event.operation}: "
            f"{event_dict.get('error_message', 'Success')} "
            f"(correlation: {event.correlation_id})"
        )
        
        # Log to database if enabled
        if self.log_to_database:
            self._log_to_database(event_dict)
        
        # Log to file if enabled (structured JSON)
        if self.log_to_file:
            self._log_to_file(event_dict)
    
    def _log_to_database(self, event_dict: Dict[str, Any]):
        """Log event to database."""
        try:
            conn = get_db_connection()
            if not conn:
                logger.warning("Failed to get database connection for audit logging")
                return
            
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status, timestamp
                ) VALUES (%s, %s, %s, %s, %s)
            """, (
                event_dict["correlation_id"],
                f"cache_{event_dict['event_type']}",
                json.dumps(event_dict),
                "success" if event_dict["severity"] in ["debug", "info"] else "error",
                datetime.fromisoformat(event_dict["timestamp"]),
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
        except Exception as e:
            logger.error(f"Failed to log audit event to database: {e}")
            if conn:
                return_db_connection(conn)
    
    def _log_to_file(self, event_dict: Dict[str, Any]):
        """Log event to structured file."""
        try:
            # In a production system, this would write to a structured log file
            # For now, we use the standard logger with JSON formatting
            logger.info(f"CACHE_AUDIT_JSON: {json.dumps(event_dict)}")
            
        except Exception as e:
            logger.error(f"Failed to log audit event to file: {e}")
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        return f"evt_{uuid.uuid4().hex[:16]}"


# Global cache audit logger instance
_cache_audit_logger: Optional[CacheAuditLogger] = None


def get_cache_audit_logger(config: Optional[CacheConfiguration] = None) -> CacheAuditLogger:
    """
    Get the global cache audit logger instance.
    
    Args:
        config: Cache configuration (used only on first call)
        
    Returns:
        The global CacheAuditLogger instance
    """
    global _cache_audit_logger
    if _cache_audit_logger is None:
        _cache_audit_logger = CacheAuditLogger(config)
    return _cache_audit_logger


def reset_cache_audit_logger():
    """Reset the global cache audit logger (for testing)."""
    global _cache_audit_logger
    _cache_audit_logger = None


class AuditContext:
    """
    Context manager for audit correlation tracking.
    
    Usage:
        with AuditContext("content_loading") as audit:
            # ... perform operations ...
            audit.log_content_loading_success(...)
    """
    
    def __init__(self, operation: str, config: Optional[CacheConfiguration] = None):
        """
        Initialize audit context.
        
        Args:
            operation: Name of the operation being audited
            config: Cache configuration
        """
        self.operation = operation
        self.audit_logger = get_cache_audit_logger(config)
        self.correlation_id: Optional[str] = None
    
    def __enter__(self) -> CacheAuditLogger:
        """Start audit context."""
        self.correlation_id = self.audit_logger.start_correlation(self.operation)
        return self.audit_logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End audit context."""
        if exc_type is not None:
            # Log any unhandled exceptions
            self.audit_logger.log_performance_alert(
                "unhandled_exception",
                f"Unhandled exception in {self.operation}: {exc_val}",
                {
                    "exception_type": exc_type.__name__, 
                    "exception_message": str(exc_val)
                },
                AuditSeverity.ERROR
            )
        
        self.audit_logger.end_correlation()