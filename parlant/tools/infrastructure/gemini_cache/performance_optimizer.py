"""
Performance Optimizer for Gemini Context Caching.

This module provides performance optimization and monitoring:
- Content loading optimization for <3 second startup
- Prompt construction optimization for <10ms per request
- Memory usage optimization (<15MB total)
- Performance degradation detection and alerts

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import logging
import time
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceAlert:
    """Performance alert information."""
    alert_type: str
    severity: str  # "info", "warning", "error", "critical"
    message: str
    timestamp: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    threshold: Optional[float] = None
    current_value: Optional[float] = None


@dataclass
class PerformanceOptimizationResult:
    """Result of performance optimization."""
    optimization_applied: bool
    optimization_type: str
    before_value: float
    after_value: float
    improvement_percent: float
    details: str


class PerformanceOptimizer:
    """
    Optimizes cache system performance and detects degradation.
    
    Provides:
    - Content loading optimization
    - Prompt construction optimization
    - Memory usage monitoring
    - Performance degradation detection
    - Automated alerting
    """
    
    def __init__(
        self,
        content_loading_target_ms: float = 3000,
        prompt_construction_target_ms: float = 10,
        memory_target_mb: float = 15,
        alert_threshold_percent: float = 50  # Alert when >50% over target
    ):
        """
        Initialize performance optimizer.
        
        Args:
            content_loading_target_ms: Target content loading time in ms
            prompt_construction_target_ms: Target prompt construction time in ms
            memory_target_mb: Target memory usage in MB
            alert_threshold_percent: Percentage over target to trigger alerts
        """
        self.content_loading_target_ms = content_loading_target_ms
        self.prompt_construction_target_ms = prompt_construction_target_ms
        self.memory_target_mb = memory_target_mb
        self.alert_threshold_percent = alert_threshold_percent
        
        # Performance tracking
        self._content_loading_times: List[float] = []
        self._prompt_construction_times: List[float] = []
        self._memory_samples: List[float] = []
        
        # Alert tracking
        self._alerts: List[PerformanceAlert] = []
        self._last_alert_time: Dict[str, datetime] = {}
        self._alert_cooldown_seconds = 300  # 5 minutes between same alert type
        
        # Optimization tracking
        self._optimizations_applied: List[PerformanceOptimizationResult] = []
    
    def record_content_loading_time(self, duration_ms: float):
        """
        Record content loading time and check for performance issues.
        
        Args:
            duration_ms: Content loading duration in milliseconds
        """
        self._content_loading_times.append(duration_ms)
        
        # Keep only recent samples (last 100)
        if len(self._content_loading_times) > 100:
            self._content_loading_times = self._content_loading_times[-100:]
        
        # Check if performance is degraded
        if duration_ms > self.content_loading_target_ms:
            excess_percent = ((duration_ms - self.content_loading_target_ms) / 
                            self.content_loading_target_ms * 100)
            
            if excess_percent > self.alert_threshold_percent:
                self._create_alert(
                    alert_type="slow_content_loading",
                    severity="warning",
                    message=f"Content loading took {duration_ms:.1f}ms (target: {self.content_loading_target_ms:.1f}ms)",
                    metrics={
                        "duration_ms": duration_ms,
                        "target_ms": self.content_loading_target_ms,
                        "excess_percent": excess_percent
                    },
                    threshold=self.content_loading_target_ms,
                    current_value=duration_ms
                )
    
    def record_prompt_construction_time(self, duration_ms: float):
        """
        Record prompt construction time and check for performance issues.
        
        Args:
            duration_ms: Prompt construction duration in milliseconds
        """
        self._prompt_construction_times.append(duration_ms)
        
        # Keep only recent samples (last 1000)
        if len(self._prompt_construction_times) > 1000:
            self._prompt_construction_times = self._prompt_construction_times[-1000:]
        
        # Check if performance is degraded
        if duration_ms > self.prompt_construction_target_ms:
            excess_percent = ((duration_ms - self.prompt_construction_target_ms) / 
                            self.prompt_construction_target_ms * 100)
            
            if excess_percent > self.alert_threshold_percent:
                self._create_alert(
                    alert_type="slow_prompt_construction",
                    severity="warning",
                    message=f"Prompt construction took {duration_ms:.2f}ms (target: {self.prompt_construction_target_ms:.2f}ms)",
                    metrics={
                        "duration_ms": duration_ms,
                        "target_ms": self.prompt_construction_target_ms,
                        "excess_percent": excess_percent
                    },
                    threshold=self.prompt_construction_target_ms,
                    current_value=duration_ms
                )
    
    def record_memory_usage(self, memory_mb: float):
        """
        Record memory usage and check for issues.
        
        Args:
            memory_mb: Memory usage in megabytes
        """
        self._memory_samples.append(memory_mb)
        
        # Keep only recent samples (last 100)
        if len(self._memory_samples) > 100:
            self._memory_samples = self._memory_samples[-100:]
        
        # Check if memory usage is excessive
        if memory_mb > self.memory_target_mb:
            excess_percent = ((memory_mb - self.memory_target_mb) / 
                            self.memory_target_mb * 100)
            
            if excess_percent > self.alert_threshold_percent:
                severity = "error" if memory_mb > self.memory_target_mb * 1.5 else "warning"
                self._create_alert(
                    alert_type="high_memory_usage",
                    severity=severity,
                    message=f"Memory usage {memory_mb:.2f}MB exceeds target {self.memory_target_mb:.2f}MB",
                    metrics={
                        "memory_mb": memory_mb,
                        "target_mb": self.memory_target_mb,
                        "excess_percent": excess_percent
                    },
                    threshold=self.memory_target_mb,
                    current_value=memory_mb
                )
    
    def check_performance_degradation(self) -> List[PerformanceAlert]:
        """
        Check for performance degradation across all metrics.
        
        Returns:
            List of active performance alerts
        """
        alerts = []
        
        # Check content loading performance
        if self._content_loading_times:
            avg_loading_time = sum(self._content_loading_times) / len(self._content_loading_times)
            if avg_loading_time > self.content_loading_target_ms * 1.2:  # 20% over target
                alerts.append(PerformanceAlert(
                    alert_type="degraded_content_loading",
                    severity="warning",
                    message=f"Average content loading time {avg_loading_time:.1f}ms exceeds target by 20%",
                    timestamp=datetime.now(),
                    metrics={
                        "avg_time_ms": avg_loading_time,
                        "target_ms": self.content_loading_target_ms,
                        "sample_count": len(self._content_loading_times)
                    },
                    threshold=self.content_loading_target_ms,
                    current_value=avg_loading_time
                ))
        
        # Check prompt construction performance
        if self._prompt_construction_times:
            avg_construction_time = sum(self._prompt_construction_times) / len(self._prompt_construction_times)
            if avg_construction_time > self.prompt_construction_target_ms * 1.2:  # 20% over target
                alerts.append(PerformanceAlert(
                    alert_type="degraded_prompt_construction",
                    severity="warning",
                    message=f"Average prompt construction time {avg_construction_time:.2f}ms exceeds target by 20%",
                    timestamp=datetime.now(),
                    metrics={
                        "avg_time_ms": avg_construction_time,
                        "target_ms": self.prompt_construction_target_ms,
                        "sample_count": len(self._prompt_construction_times)
                    },
                    threshold=self.prompt_construction_target_ms,
                    current_value=avg_construction_time
                ))
        
        # Check memory usage
        if self._memory_samples:
            avg_memory = sum(self._memory_samples) / len(self._memory_samples)
            if avg_memory > self.memory_target_mb:
                severity = "error" if avg_memory > self.memory_target_mb * 1.5 else "warning"
                alerts.append(PerformanceAlert(
                    alert_type="high_average_memory",
                    severity=severity,
                    message=f"Average memory usage {avg_memory:.2f}MB exceeds target {self.memory_target_mb:.2f}MB",
                    timestamp=datetime.now(),
                    metrics={
                        "avg_memory_mb": avg_memory,
                        "target_mb": self.memory_target_mb,
                        "sample_count": len(self._memory_samples)
                    },
                    threshold=self.memory_target_mb,
                    current_value=avg_memory
                ))
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dictionary with performance metrics and status
        """
        summary = {
            "content_loading": self._get_content_loading_summary(),
            "prompt_construction": self._get_prompt_construction_summary(),
            "memory_usage": self._get_memory_usage_summary(),
            "alerts": [self._alert_to_dict(alert) for alert in self._alerts[-10:]],  # Last 10 alerts
            "optimizations": [self._optimization_to_dict(opt) for opt in self._optimizations_applied[-10:]],
            "overall_status": self._get_overall_status()
        }
        
        return summary
    
    def _get_content_loading_summary(self) -> Dict[str, Any]:
        """Get content loading performance summary."""
        if not self._content_loading_times:
            return {
                "status": "no_data",
                "sample_count": 0
            }
        
        avg_time = sum(self._content_loading_times) / len(self._content_loading_times)
        max_time = max(self._content_loading_times)
        min_time = min(self._content_loading_times)
        
        # Calculate p95
        sorted_times = sorted(self._content_loading_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        
        status = "good" if avg_time <= self.content_loading_target_ms else "degraded"
        
        return {
            "status": status,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "p95_time_ms": p95_time,
            "target_ms": self.content_loading_target_ms,
            "sample_count": len(self._content_loading_times),
            "within_target": avg_time <= self.content_loading_target_ms
        }
    
    def _get_prompt_construction_summary(self) -> Dict[str, Any]:
        """Get prompt construction performance summary."""
        if not self._prompt_construction_times:
            return {
                "status": "no_data",
                "sample_count": 0
            }
        
        avg_time = sum(self._prompt_construction_times) / len(self._prompt_construction_times)
        max_time = max(self._prompt_construction_times)
        min_time = min(self._prompt_construction_times)
        
        # Calculate p95
        sorted_times = sorted(self._prompt_construction_times)
        p95_index = int(0.95 * len(sorted_times))
        p95_time = sorted_times[min(p95_index, len(sorted_times) - 1)]
        
        status = "good" if avg_time <= self.prompt_construction_target_ms else "degraded"
        
        return {
            "status": status,
            "avg_time_ms": avg_time,
            "min_time_ms": min_time,
            "max_time_ms": max_time,
            "p95_time_ms": p95_time,
            "target_ms": self.prompt_construction_target_ms,
            "sample_count": len(self._prompt_construction_times),
            "within_target": avg_time <= self.prompt_construction_target_ms
        }
    
    def _get_memory_usage_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self._memory_samples:
            return {
                "status": "no_data",
                "sample_count": 0
            }
        
        avg_memory = sum(self._memory_samples) / len(self._memory_samples)
        max_memory = max(self._memory_samples)
        min_memory = min(self._memory_samples)
        
        status = "good" if avg_memory <= self.memory_target_mb else "high"
        if avg_memory > self.memory_target_mb * 1.5:
            status = "critical"
        
        return {
            "status": status,
            "avg_memory_mb": avg_memory,
            "min_memory_mb": min_memory,
            "max_memory_mb": max_memory,
            "target_mb": self.memory_target_mb,
            "sample_count": len(self._memory_samples),
            "within_target": avg_memory <= self.memory_target_mb
        }
    
    def _get_overall_status(self) -> str:
        """Get overall performance status."""
        content_summary = self._get_content_loading_summary()
        prompt_summary = self._get_prompt_construction_summary()
        memory_summary = self._get_memory_usage_summary()
        
        # Check if any metric is critical
        if memory_summary.get("status") == "critical":
            return "critical"
        
        # Check if any metric is degraded
        if (content_summary.get("status") == "degraded" or 
            prompt_summary.get("status") == "degraded" or
            memory_summary.get("status") == "high"):
            return "degraded"
        
        # Check if we have data
        if (content_summary.get("status") == "no_data" and
            prompt_summary.get("status") == "no_data" and
            memory_summary.get("status") == "no_data"):
            return "no_data"
        
        return "good"
    
    def _create_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        metrics: Dict[str, Any],
        threshold: Optional[float] = None,
        current_value: Optional[float] = None
    ):
        """Create and store a performance alert."""
        # Check cooldown
        if alert_type in self._last_alert_time:
            time_since_last = datetime.now() - self._last_alert_time[alert_type]
            if time_since_last.total_seconds() < self._alert_cooldown_seconds:
                return  # Skip alert due to cooldown
        
        alert = PerformanceAlert(
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            metrics=metrics,
            threshold=threshold,
            current_value=current_value
        )
        
        self._alerts.append(alert)
        self._last_alert_time[alert_type] = datetime.now()
        
        # Log alert
        log_level = {
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"Performance Alert [{severity.upper()}]: {message}")
        
        # Keep only recent alerts (last 100)
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]
    
    def _alert_to_dict(self, alert: PerformanceAlert) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metrics": alert.metrics,
            "threshold": alert.threshold,
            "current_value": alert.current_value
        }
    
    def _optimization_to_dict(self, opt: PerformanceOptimizationResult) -> Dict[str, Any]:
        """Convert optimization result to dictionary."""
        return {
            "optimization_applied": opt.optimization_applied,
            "optimization_type": opt.optimization_type,
            "before_value": opt.before_value,
            "after_value": opt.after_value,
            "improvement_percent": opt.improvement_percent,
            "details": opt.details
        }
    
    def suggest_optimizations(self) -> List[str]:
        """
        Suggest performance optimizations based on current metrics.
        
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        # Check content loading performance
        content_summary = self._get_content_loading_summary()
        if content_summary.get("status") == "degraded":
            suggestions.append(
                "Content loading is slow. Consider: "
                "1) Reducing static content size, "
                "2) Caching loaded content in memory, "
                "3) Using async file I/O"
            )
        
        # Check prompt construction performance
        prompt_summary = self._get_prompt_construction_summary()
        if prompt_summary.get("status") == "degraded":
            suggestions.append(
                "Prompt construction is slow. Consider: "
                "1) Pre-building templates, "
                "2) Reducing dynamic content formatting, "
                "3) Using string builders instead of concatenation"
            )
        
        # Check memory usage
        memory_summary = self._get_memory_usage_summary()
        if memory_summary.get("status") in ["high", "critical"]:
            suggestions.append(
                "Memory usage is high. Consider: "
                "1) Reducing static content size, "
                "2) Implementing content compression, "
                "3) Clearing old metrics more frequently"
            )
        
        return suggestions
    
    def reset_metrics(self):
        """Reset all performance metrics (for testing or maintenance)."""
        self._content_loading_times.clear()
        self._prompt_construction_times.clear()
        self._memory_samples.clear()
        self._alerts.clear()
        self._last_alert_time.clear()
        self._optimizations_applied.clear()
        logger.info("Performance metrics reset")


# Global performance optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def reset_performance_optimizer():
    """Reset global performance optimizer (for testing)."""
    global _global_optimizer
    _global_optimizer = None
