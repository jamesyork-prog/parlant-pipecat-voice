"""
Metrics Tracker for Gemini Context Caching.

This module tracks cache effectiveness and cost savings:
- Records metrics for each request
- Calculates aggregate statistics
- Provides reporting data for dashboards
- Tracks performance metrics (content loading, prompt construction)
- Monitors cache effectiveness and cost savings

Requirements: 10.1-10.5, 15.1-15.5
"""

import logging
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .models import CacheMetrics

logger = logging.getLogger(__name__)


@dataclass
class AggregateMetrics:
    """Aggregate metrics for a time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    total_cached_tokens: int = 0
    total_dynamic_tokens: int = 0
    total_savings_usd: float = 0.0
    avg_cache_effectiveness: float = 0.0
    avg_savings_per_request: float = 0.0
    cache_hit_rate: float = 0.0
    min_cache_effectiveness: float = 100.0
    max_cache_effectiveness: float = 0.0


@dataclass
class PerformanceMetrics:
    """Performance metrics for cache operations."""
    content_loading_times: List[float] = field(default_factory=list)
    prompt_construction_times: List[float] = field(default_factory=list)
    total_content_loads: int = 0
    total_prompt_constructions: int = 0
    slow_content_loads: int = 0  # > 3 seconds
    slow_prompt_constructions: int = 0  # > 10ms
    
    def add_content_loading_time(self, duration_ms: float):
        """Record content loading time."""
        self.content_loading_times.append(duration_ms)
        self.total_content_loads += 1
        if duration_ms > 3000:  # 3 seconds in ms
            self.slow_content_loads += 1
    
    def add_prompt_construction_time(self, duration_ms: float):
        """Record prompt construction time."""
        self.prompt_construction_times.append(duration_ms)
        self.total_prompt_constructions += 1
        if duration_ms > 10:  # 10ms
            self.slow_prompt_constructions += 1
    
    def get_avg_content_loading_time(self) -> float:
        """Get average content loading time in ms."""
        if not self.content_loading_times:
            return 0.0
        return statistics.mean(self.content_loading_times)
    
    def get_avg_prompt_construction_time(self) -> float:
        """Get average prompt construction time in ms."""
        if not self.prompt_construction_times:
            return 0.0
        return statistics.mean(self.prompt_construction_times)
    
    def get_p95_content_loading_time(self) -> float:
        """Get 95th percentile content loading time in ms."""
        if not self.content_loading_times:
            return 0.0
        sorted_times = sorted(self.content_loading_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    def get_p95_prompt_construction_time(self) -> float:
        """Get 95th percentile prompt construction time in ms."""
        if not self.prompt_construction_times:
            return 0.0
        sorted_times = sorted(self.prompt_construction_times)
        index = int(0.95 * len(sorted_times))
        return sorted_times[min(index, len(sorted_times) - 1)]


class CacheMetricsTracker:
    """
    Tracks implicit cache effectiveness and cost savings.
    
    Monitors cache hit rates, cost savings, optimization opportunities,
    and performance metrics for content loading and prompt construction.
    """
    
    def __init__(self, config_or_retention_hours=24):
        """
        Initialize metrics tracker.
        
        Args:
            config_or_retention_hours: Either a CacheConfiguration object or 
                                       hours to retain detailed metrics (int)
        """
        # Handle both CacheConfiguration and int for backward compatibility
        if hasattr(config_or_retention_hours, 'retention_hours'):
            # It's a CacheConfiguration object
            self.retention_hours = getattr(config_or_retention_hours, 'retention_hours', 24)
        elif isinstance(config_or_retention_hours, int):
            self.retention_hours = config_or_retention_hours
        else:
            # Default fallback
            self.retention_hours = 24
        self._metrics: List[CacheMetrics] = []
        self._performance_metrics = PerformanceMetrics()
        
        # Cumulative counters
        self._total_requests = 0
        self._total_savings = 0.0
        self._total_cached_tokens = 0
        self._total_dynamic_tokens = 0
        
        # Alert tracking
        self._low_effectiveness_count = 0
        self._last_alert_time: Optional[datetime] = None
        
        # Performance tracking
        self._startup_time: Optional[datetime] = None
        self._last_performance_alert: Optional[datetime] = None
    
    def record_content_loading_time(self, duration_ms: float):
        """
        Record content loading time.
        
        Args:
            duration_ms: Duration in milliseconds
        """
        self._performance_metrics.add_content_loading_time(duration_ms)
        
        # Alert if content loading is slow (> 3 seconds)
        if duration_ms > 3000:
            logger.warning(f"Slow content loading: {duration_ms:.1f}ms (target: <3000ms)")
            
        logger.debug(f"Content loading time: {duration_ms:.1f}ms")
    
    def record_prompt_construction_time(self, duration_ms: float):
        """
        Record prompt construction time.
        
        Args:
            duration_ms: Duration in milliseconds
        """
        self._performance_metrics.add_prompt_construction_time(duration_ms)
        
        # Alert if prompt construction is slow (> 10ms)
        if duration_ms > 10:
            logger.warning(f"Slow prompt construction: {duration_ms:.1f}ms (target: <10ms)")
            
        logger.debug(f"Prompt construction time: {duration_ms:.1f}ms")
    
    def record_startup_time(self):
        """Record system startup time."""
        self._startup_time = datetime.now()
        logger.info("Cache system startup time recorded")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics summary.
        
        Returns:
            Dictionary with performance statistics
        """
        return {
            "content_loading": {
                "total_loads": self._performance_metrics.total_content_loads,
                "avg_time_ms": self._performance_metrics.get_avg_content_loading_time(),
                "p95_time_ms": self._performance_metrics.get_p95_content_loading_time(),
                "slow_loads": self._performance_metrics.slow_content_loads,
                "slow_load_rate": (
                    self._performance_metrics.slow_content_loads / 
                    max(1, self._performance_metrics.total_content_loads) * 100
                )
            },
            "prompt_construction": {
                "total_constructions": self._performance_metrics.total_prompt_constructions,
                "avg_time_ms": self._performance_metrics.get_avg_prompt_construction_time(),
                "p95_time_ms": self._performance_metrics.get_p95_prompt_construction_time(),
                "slow_constructions": self._performance_metrics.slow_prompt_constructions,
                "slow_construction_rate": (
                    self._performance_metrics.slow_prompt_constructions / 
                    max(1, self._performance_metrics.total_prompt_constructions) * 100
                )
            },
            "startup_time": self._startup_time.isoformat() if self._startup_time else None
        }
    
    def record_request(self, metrics: CacheMetrics):
        """
        Record metrics for a single request.
        
        Args:
            metrics: Cache metrics for the request
        """
        self._metrics.append(metrics)
        
        # Update cumulative counters
        self._total_requests += 1
        self._total_savings += metrics.savings
        self._total_cached_tokens += metrics.cached_tokens
        self._total_dynamic_tokens += metrics.dynamic_tokens
        
        # Check for low effectiveness
        if metrics.cache_effectiveness < 70:
            self._low_effectiveness_count += 1
            logger.warning(
                f"Low cache effectiveness: {metrics.cache_effectiveness:.1f}% "
                f"(request: {metrics.request_id})"
            )
        
        # Cleanup old metrics
        self._cleanup_old_metrics()
    
    def track_request(self, **kwargs):
        """
        Track a request with keyword arguments (convenience method).
        
        This is an alias that accepts keyword arguments for simpler tracking
        when you don't have a full CacheMetrics object.
        
        Args:
            ticket_id: Ticket ID being processed
            cache_used: Whether caching was used
            processing_time_ms: Processing time in milliseconds
            decision: Decision made (optional)
            confidence: Confidence level (optional)
        """
        # Log the tracking info without requiring a full CacheMetrics object
        logger.info(
            f"Request tracked: ticket_id={kwargs.get('ticket_id')}, "
            f"cache_used={kwargs.get('cache_used')}, "
            f"processing_time_ms={kwargs.get('processing_time_ms')}, "
            f"decision={kwargs.get('decision')}"
        )
        
        # Update simple counters
        self._total_requests += 1
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Get summary of cache effectiveness.
        
        Returns:
            Dictionary with summary statistics including performance metrics
        """
        if not self._metrics:
            base_summary = {
                "status": "no_data",
                "total_requests": self._total_requests,
                "total_savings_usd": self._total_savings,
            }
        else:
            cache_effectiveness_rates = [m.cache_effectiveness for m in self._metrics]
            savings_amounts = [m.savings for m in self._metrics]
            
            base_summary = {
                "status": "active",
                "total_requests": self._total_requests,
                "recent_requests": len(self._metrics),
                "avg_cache_effectiveness": statistics.mean(cache_effectiveness_rates),
                "total_savings_usd": self._total_savings,
                "recent_savings_usd": sum(savings_amounts),
                "avg_savings_per_request": statistics.mean(savings_amounts) if savings_amounts else 0,
                "total_cached_tokens": self._total_cached_tokens,
                "total_dynamic_tokens": self._total_dynamic_tokens,
                "cache_hit_rate": self._calculate_cache_hit_rate(),
                "low_effectiveness_count": self._low_effectiveness_count,
            }
        
        # Add performance metrics
        base_summary["performance_metrics"] = self.get_performance_metrics()
        
        return base_summary
    
    def get_hourly_metrics(self, hours: int = 24) -> List[AggregateMetrics]:
        """
        Get hourly aggregate metrics.
        
        Args:
            hours: Number of hours to include
            
        Returns:
            List of hourly aggregate metrics
        """
        now = datetime.now()
        hourly_metrics = []
        
        for hour_offset in range(hours):
            period_end = now - timedelta(hours=hour_offset)
            period_start = period_end - timedelta(hours=1)
            
            # Filter metrics for this hour
            hour_metrics = [
                m for m in self._metrics
                if period_start <= m.timestamp < period_end
            ]
            
            if hour_metrics:
                agg = self._aggregate_metrics(hour_metrics, period_start, period_end)
                hourly_metrics.append(agg)
        
        return hourly_metrics
    
    def get_daily_summary(self) -> AggregateMetrics:
        """
        Get daily aggregate metrics.
        
        Returns:
            Aggregate metrics for the last 24 hours
        """
        now = datetime.now()
        period_start = now - timedelta(hours=24)
        
        day_metrics = [
            m for m in self._metrics
            if m.timestamp >= period_start
        ]
        
        return self._aggregate_metrics(day_metrics, period_start, now)
    
    def get_cost_report(self) -> Dict[str, Any]:
        """
        Get detailed cost report.
        
        Returns:
            Dictionary with cost breakdown and projections
        """
        daily = self.get_daily_summary()
        
        # Project monthly savings
        monthly_projection = daily.total_savings_usd * 30
        yearly_projection = daily.total_savings_usd * 365
        
        return {
            "daily_savings_usd": daily.total_savings_usd,
            "monthly_projection_usd": monthly_projection,
            "yearly_projection_usd": yearly_projection,
            "daily_requests": daily.total_requests,
            "avg_savings_per_request": daily.avg_savings_per_request,
            "cache_effectiveness_percent": daily.avg_cache_effectiveness,
            "total_cached_tokens": daily.total_cached_tokens,
            "total_dynamic_tokens": daily.total_dynamic_tokens,
        }
    
    def check_alerts(self, threshold: float = 70.0) -> List[Dict[str, Any]]:
        """
        Check for alert conditions including performance issues.
        
        Args:
            threshold: Cache effectiveness threshold for alerts
            
        Returns:
            List of active alerts
        """
        alerts = []
        
        # Check recent effectiveness
        recent_metrics = self._metrics[-10:] if len(self._metrics) >= 10 else self._metrics
        if recent_metrics:
            avg_effectiveness = statistics.mean([m.cache_effectiveness for m in recent_metrics])
            if avg_effectiveness < threshold:
                alerts.append({
                    "type": "low_cache_effectiveness",
                    "severity": "warning",
                    "message": f"Cache effectiveness dropped to {avg_effectiveness:.1f}%",
                    "threshold": threshold,
                    "current_value": avg_effectiveness,
                })
        
        # Check for consecutive low effectiveness
        if self._low_effectiveness_count >= 5:
            alerts.append({
                "type": "consecutive_low_effectiveness",
                "severity": "error",
                "message": f"{self._low_effectiveness_count} requests with low cache effectiveness",
                "count": self._low_effectiveness_count,
            })
        
        # Check performance alerts
        perf_metrics = self.get_performance_metrics()
        
        # Content loading performance
        if perf_metrics["content_loading"]["slow_load_rate"] > 20:  # >20% slow loads
            alerts.append({
                "type": "slow_content_loading",
                "severity": "warning",
                "message": f"High rate of slow content loading: {perf_metrics['content_loading']['slow_load_rate']:.1f}%",
                "avg_time_ms": perf_metrics["content_loading"]["avg_time_ms"],
                "p95_time_ms": perf_metrics["content_loading"]["p95_time_ms"],
            })
        
        # Prompt construction performance
        if perf_metrics["prompt_construction"]["slow_construction_rate"] > 10:  # >10% slow constructions
            alerts.append({
                "type": "slow_prompt_construction",
                "severity": "warning", 
                "message": f"High rate of slow prompt construction: {perf_metrics['prompt_construction']['slow_construction_rate']:.1f}%",
                "avg_time_ms": perf_metrics["prompt_construction"]["avg_time_ms"],
                "p95_time_ms": perf_metrics["prompt_construction"]["p95_time_ms"],
            })
        
        return alerts
    
    def reset_counters(self):
        """Reset cumulative counters (for testing or maintenance)."""
        self._total_requests = 0
        self._total_savings = 0.0
        self._total_cached_tokens = 0
        self._total_dynamic_tokens = 0
        self._low_effectiveness_count = 0
        self._metrics.clear()
        self._performance_metrics = PerformanceMetrics()
        self._startup_time = None
        logger.info("Metrics counters reset")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for monitoring dashboards.
        
        Returns:
            Dictionary with all metrics formatted for dashboard display
        """
        summary = self.get_summary_metrics()
        cost_report = self.get_cost_report()
        alerts = self.check_alerts()
        daily_summary = self.get_daily_summary()
        
        return {
            "overview": {
                "status": summary["status"],
                "total_requests": summary["total_requests"],
                "cache_effectiveness": summary.get("avg_cache_effectiveness", 0),
                "total_savings_usd": summary["total_savings_usd"],
                "cache_hit_rate": summary.get("cache_hit_rate", 0),
            },
            "performance": summary["performance_metrics"],
            "cost_analysis": cost_report,
            "alerts": alerts,
            "daily_summary": {
                "requests": daily_summary.total_requests,
                "savings": daily_summary.total_savings_usd,
                "effectiveness": daily_summary.avg_cache_effectiveness,
                "cached_tokens": daily_summary.total_cached_tokens,
                "dynamic_tokens": daily_summary.total_dynamic_tokens,
            },
            "trends": {
                "hourly_metrics": self.get_hourly_metrics(24),
                "low_effectiveness_count": summary.get("low_effectiveness_count", 0),
            }
        }
    
    def _cleanup_old_metrics(self):
        """Remove metrics older than retention period."""
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        original_count = len(self._metrics)
        self._metrics = [m for m in self._metrics if m.timestamp > cutoff]
        
        removed = original_count - len(self._metrics)
        if removed > 0:
            logger.debug(f"Cleaned up {removed} old metrics entries")
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate percentage of requests with cached tokens."""
        if not self._metrics:
            return 0.0
        
        hits = len([m for m in self._metrics if m.cached_tokens > 0])
        return (hits / len(self._metrics)) * 100
    
    def _aggregate_metrics(
        self,
        metrics: List[CacheMetrics],
        period_start: datetime,
        period_end: datetime
    ) -> AggregateMetrics:
        """Aggregate a list of metrics into summary."""
        if not metrics:
            return AggregateMetrics(
                period_start=period_start,
                period_end=period_end,
            )
        
        effectiveness_rates = [m.cache_effectiveness for m in metrics]
        savings_amounts = [m.savings for m in metrics]
        
        return AggregateMetrics(
            period_start=period_start,
            period_end=period_end,
            total_requests=len(metrics),
            total_cached_tokens=sum(m.cached_tokens for m in metrics),
            total_dynamic_tokens=sum(m.dynamic_tokens for m in metrics),
            total_savings_usd=sum(savings_amounts),
            avg_cache_effectiveness=statistics.mean(effectiveness_rates),
            avg_savings_per_request=statistics.mean(savings_amounts),
            cache_hit_rate=len([m for m in metrics if m.cached_tokens > 0]) / len(metrics) * 100,
            min_cache_effectiveness=min(effectiveness_rates),
            max_cache_effectiveness=max(effectiveness_rates),
        )


class PerformanceTimer:
    """
    Context manager for timing cache operations.
    
    Usage:
        with PerformanceTimer(tracker, "content_loading") as timer:
            # ... do content loading work ...
        # timer automatically records the duration
    """
    
    def __init__(self, tracker: CacheMetricsTracker, operation_type: str):
        """
        Initialize performance timer.
        
        Args:
            tracker: Metrics tracker to record timing to
            operation_type: Type of operation ("content_loading" or "prompt_construction")
        """
        self.tracker = tracker
        self.operation_type = operation_type
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time() * 1000  # Convert to milliseconds
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration."""
        if self.start_time is not None:
            end_time = time.time() * 1000
            self.duration_ms = end_time - self.start_time
            
            if self.operation_type == "content_loading":
                self.tracker.record_content_loading_time(self.duration_ms)
            elif self.operation_type == "prompt_construction":
                self.tracker.record_prompt_construction_time(self.duration_ms)
            else:
                logger.warning(f"Unknown operation type for timing: {self.operation_type}")
    
    def get_duration_ms(self) -> Optional[float]:
        """Get the recorded duration in milliseconds."""
        return self.duration_ms
