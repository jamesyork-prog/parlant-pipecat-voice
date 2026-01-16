"""
Metrics tracking for MOR and Product Type gates.

This module provides comprehensive metrics collection and analysis for the
routing gates, including cost savings calculations, performance monitoring,
and dashboard-ready data aggregation.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .models import MORMetrics, ProductTypeMetrics


@dataclass
class CombinedGateMetrics:
    """Combined metrics from both MOR and Product Type gates."""
    
    # Overall statistics
    total_tickets_processed: int = 0
    total_tickets_rejected: int = 0
    total_processing_time_ms: int = 0
    
    # Cost savings
    total_cost_savings: float = 0.0
    average_llm_cost_per_ticket: float = 0.12
    
    # Performance metrics
    average_processing_time_ms: float = 0.0
    rejection_rate_percent: float = 0.0
    
    # Gate-specific metrics
    mor_metrics: Optional[MORMetrics] = None
    product_type_metrics: Optional[ProductTypeMetrics] = None
    
    # Timestamp
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_from_gates(self, mor_metrics: MORMetrics, product_type_metrics: ProductTypeMetrics) -> None:
        """Update combined metrics from individual gate metrics."""
        self.mor_metrics = mor_metrics
        self.product_type_metrics = product_type_metrics
        
        # Calculate totals
        self.total_tickets_processed = mor_metrics.total_processed + product_type_metrics.total_processed
        self.total_tickets_rejected = mor_metrics.rejection_count + product_type_metrics.rejection_count
        self.total_processing_time_ms = mor_metrics.total_processing_time_ms + product_type_metrics.total_processing_time_ms
        
        # Calculate cost savings
        self.total_cost_savings = mor_metrics.get_cost_savings() + product_type_metrics.get_cost_savings()
        
        # Calculate averages
        if self.total_tickets_processed > 0:
            self.average_processing_time_ms = self.total_processing_time_ms / self.total_tickets_processed
            self.rejection_rate_percent = (self.total_tickets_rejected / self.total_tickets_processed) * 100
        else:
            self.average_processing_time_ms = 0.0
            self.rejection_rate_percent = 0.0
        
        self.last_updated = datetime.utcnow()
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        return {
            "overview": {
                "total_tickets_processed": self.total_tickets_processed,
                "total_tickets_rejected": self.total_tickets_rejected,
                "rejection_rate_percent": round(self.rejection_rate_percent, 2),
                "total_cost_savings_usd": round(self.total_cost_savings, 2),
                "average_processing_time_ms": round(self.average_processing_time_ms, 2),
                "last_updated": self.last_updated.isoformat()
            },
            "mor_gate": self._get_mor_dashboard_data(),
            "product_type_gate": self._get_product_type_dashboard_data(),
            "cost_analysis": self._get_cost_analysis()
        }
    
    def _get_mor_dashboard_data(self) -> Dict[str, Any]:
        """Get MOR gate metrics for dashboard."""
        if not self.mor_metrics:
            return {}
        
        return {
            "total_processed": self.mor_metrics.total_processed,
            "rejection_count": self.mor_metrics.rejection_count,
            "rejection_rate_percent": round(self.mor_metrics.get_rejection_rate(), 2),
            "cost_savings_usd": round(self.mor_metrics.get_cost_savings(), 2),
            "average_processing_time_ms": round(self.mor_metrics.get_average_processing_time(), 2),
            "vendor_rejections": dict(self.mor_metrics.vendor_rejection_counts),
            "pass_through_counts": {
                "parkwhiz": self.mor_metrics.parkwhiz_pass_count,
                "groupon_special": self.mor_metrics.groupon_special_count,
                "unknown_partner": self.mor_metrics.unknown_partner_count
            }
        }
    
    def _get_product_type_dashboard_data(self) -> Dict[str, Any]:
        """Get Product Type gate metrics for dashboard."""
        if not self.product_type_metrics:
            return {}
        
        return {
            "total_processed": self.product_type_metrics.total_processed,
            "rejection_count": self.product_type_metrics.rejection_count,
            "rejection_rate_percent": round(self.product_type_metrics.get_rejection_rate(), 2),
            "cost_savings_usd": round(self.product_type_metrics.get_cost_savings(), 2),
            "average_processing_time_ms": round(self.product_type_metrics.get_average_processing_time(), 2),
            "product_type_distribution": dict(self.product_type_metrics.product_type_counts),
            "rejection_breakdown": {
                "on_demand": self.product_type_metrics.on_demand_rejections,
                "season_package": self.product_type_metrics.season_package_rejections
            },
            "pass_through_counts": {
                "event": self.product_type_metrics.event_pass_count,
                "transient": self.product_type_metrics.transient_pass_count
            },
            "duration_calculation_failures": self.product_type_metrics.duration_calculation_failures
        }
    
    def _get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis."""
        daily_savings = self.total_cost_savings
        weekly_savings = daily_savings * 7
        monthly_savings = daily_savings * 30
        yearly_savings = daily_savings * 365
        
        return {
            "cost_per_ticket_usd": self.average_llm_cost_per_ticket,
            "tickets_saved_from_llm": self.total_tickets_rejected,
            "savings_projections": {
                "daily_usd": round(daily_savings, 2),
                "weekly_usd": round(weekly_savings, 2),
                "monthly_usd": round(monthly_savings, 2),
                "yearly_usd": round(yearly_savings, 2)
            },
            "efficiency_metrics": {
                "processing_time_saved_ms": self.total_tickets_rejected * 10000,  # Assume 10s saved per rejection
                "processing_time_saved_hours": round((self.total_tickets_rejected * 10000) / (1000 * 60 * 60), 2)
            }
        }


class GateMetricsTracker:
    """
    Centralized metrics tracker for MOR and Product Type gates.
    
    This class aggregates metrics from both gates and provides
    dashboard-ready data for monitoring and analysis.
    """
    
    def __init__(self):
        """Initialize the metrics tracker."""
        self.combined_metrics = CombinedGateMetrics()
        self.start_time = datetime.utcnow()
        
        # Historical data (last 24 hours)
        self.hourly_metrics: List[CombinedGateMetrics] = []
        self.max_hourly_records = 24
    
    def update_metrics(self, mor_metrics: MORMetrics, product_type_metrics: ProductTypeMetrics) -> None:
        """
        Update metrics from both gates.
        
        Args:
            mor_metrics: Current MOR gate metrics
            product_type_metrics: Current Product Type gate metrics
        """
        self.combined_metrics.update_from_gates(mor_metrics, product_type_metrics)
        
        # Store hourly snapshot
        current_hour = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
        
        # Check if we need a new hourly record
        if not self.hourly_metrics or self.hourly_metrics[-1].last_updated.hour != current_hour.hour:
            # Create snapshot of current metrics
            snapshot = CombinedGateMetrics()
            snapshot.update_from_gates(mor_metrics, product_type_metrics)
            self.hourly_metrics.append(snapshot)
            
            # Keep only last 24 hours
            if len(self.hourly_metrics) > self.max_hourly_records:
                self.hourly_metrics = self.hourly_metrics[-self.max_hourly_records:]
    
    def get_current_metrics(self) -> CombinedGateMetrics:
        """Get current combined metrics."""
        return self.combined_metrics
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        dashboard_data = self.combined_metrics.get_dashboard_data()
        
        # Add historical trends
        dashboard_data["trends"] = self._get_trend_data()
        dashboard_data["uptime_seconds"] = (datetime.utcnow() - self.start_time).total_seconds()
        
        return dashboard_data
    
    def _get_trend_data(self) -> Dict[str, Any]:
        """Get trend data for the last 24 hours."""
        if len(self.hourly_metrics) < 2:
            return {
                "hourly_rejection_counts": [],
                "hourly_cost_savings": [],
                "hourly_processing_times": []
            }
        
        # Extract trends
        rejection_counts = [m.total_tickets_rejected for m in self.hourly_metrics]
        cost_savings = [m.total_cost_savings for m in self.hourly_metrics]
        processing_times = [m.average_processing_time_ms for m in self.hourly_metrics]
        timestamps = [m.last_updated.isoformat() for m in self.hourly_metrics]
        
        return {
            "timestamps": timestamps,
            "hourly_rejection_counts": rejection_counts,
            "hourly_cost_savings": cost_savings,
            "hourly_processing_times": processing_times,
            "trend_analysis": {
                "rejection_trend": self._calculate_trend(rejection_counts),
                "cost_savings_trend": self._calculate_trend(cost_savings),
                "performance_trend": self._calculate_trend(processing_times, lower_is_better=True)
            }
        }
    
    def _calculate_trend(self, values: List[float], lower_is_better: bool = False) -> str:
        """
        Calculate trend direction from a list of values.
        
        Args:
            values: List of numeric values
            lower_is_better: If True, lower values are considered better
            
        Returns:
            Trend direction: "improving", "declining", or "stable"
        """
        if len(values) < 2:
            return "stable"
        
        # Compare last 3 values to previous 3 values (if available)
        recent_count = min(3, len(values) // 2)
        if recent_count < 1:
            return "stable"
        
        recent_avg = sum(values[-recent_count:]) / recent_count
        previous_avg = sum(values[-2*recent_count:-recent_count]) / recent_count if len(values) >= 2*recent_count else values[0]
        
        # Calculate percentage change
        if previous_avg == 0:
            return "stable"
        
        change_percent = ((recent_avg - previous_avg) / previous_avg) * 100
        
        # Determine trend (5% threshold for significance)
        if abs(change_percent) < 5:
            return "stable"
        elif change_percent > 0:
            return "declining" if lower_is_better else "improving"
        else:
            return "improving" if lower_is_better else "declining"
    
    def get_vendor_analysis(self) -> Dict[str, Any]:
        """Get detailed vendor analysis for MOR rejections."""
        if not self.combined_metrics.mor_metrics:
            return {}
        
        vendor_counts = self.combined_metrics.mor_metrics.vendor_rejection_counts
        total_rejections = sum(vendor_counts.values())
        
        if total_rejections == 0:
            return {}
        
        # Calculate vendor percentages and rankings
        vendor_analysis = []
        for vendor, count in sorted(vendor_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_rejections) * 100
            cost_savings = count * self.combined_metrics.average_llm_cost_per_ticket
            
            vendor_analysis.append({
                "vendor": vendor,
                "rejection_count": count,
                "percentage": round(percentage, 2),
                "cost_savings_usd": round(cost_savings, 2)
            })
        
        return {
            "total_vendor_rejections": total_rejections,
            "vendor_breakdown": vendor_analysis,
            "top_vendor": vendor_analysis[0]["vendor"] if vendor_analysis else None,
            "vendor_diversity": len(vendor_counts)  # Number of different vendors seen
        }
    
    def get_product_type_analysis(self) -> Dict[str, Any]:
        """Get detailed product type analysis."""
        if not self.combined_metrics.product_type_metrics:
            return {}
        
        type_counts = self.combined_metrics.product_type_metrics.product_type_counts
        total_processed = sum(type_counts.values())
        
        if total_processed == 0:
            return {}
        
        # Calculate type percentages
        type_analysis = []
        for product_type, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_processed) * 100
            
            type_analysis.append({
                "product_type": product_type,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "total_processed": total_processed,
            "type_distribution": type_analysis,
            "most_common_type": type_analysis[0]["product_type"] if type_analysis else None,
            "rejection_breakdown": {
                "on_demand_rejections": self.combined_metrics.product_type_metrics.on_demand_rejections,
                "season_package_rejections": self.combined_metrics.product_type_metrics.season_package_rejections,
                "total_rejections": self.combined_metrics.product_type_metrics.rejection_count
            },
            "data_quality": {
                "duration_calculation_failures": self.combined_metrics.product_type_metrics.duration_calculation_failures,
                "failure_rate_percent": round(
                    (self.combined_metrics.product_type_metrics.duration_calculation_failures / total_processed) * 100, 2
                ) if total_processed > 0 else 0
            }
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics to initial state."""
        self.combined_metrics = CombinedGateMetrics()
        self.hourly_metrics = []
        self.start_time = datetime.utcnow()


# Global metrics tracker instance
_gate_metrics_tracker: Optional[GateMetricsTracker] = None


def get_gate_metrics_tracker() -> GateMetricsTracker:
    """
    Get the global gate metrics tracker instance.
    
    Returns:
        The global GateMetricsTracker instance
    """
    global _gate_metrics_tracker
    if _gate_metrics_tracker is None:
        _gate_metrics_tracker = GateMetricsTracker()
    return _gate_metrics_tracker


def reset_gate_metrics() -> None:
    """Reset the global gate metrics tracker."""
    global _gate_metrics_tracker
    if _gate_metrics_tracker is not None:
        _gate_metrics_tracker.reset_metrics()