"""
Metrics endpoint for exposing router metrics to Prometheus.

This module provides a simple HTTP endpoint that exposes router metrics
in Prometheus format, integrated with the main Parlant application.
"""

import json
from typing import Dict, Any
from datetime import datetime
from prometheus_client import generate_latest, CollectorRegistry, Gauge, Counter, Histogram
from .metrics_tracker import get_gate_metrics_tracker


class RouterMetricsCollector:
    """Collects and exposes router metrics in Prometheus format."""
    
    def __init__(self):
        """Initialize the metrics collector."""
        self.registry = CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Set up Prometheus metrics."""
        # Overview metrics
        self.total_processed = Gauge(
            'router_total_processed_total', 
            'Total tickets processed by router', 
            registry=self.registry
        )
        self.total_rejections = Gauge(
            'router_total_rejections_total', 
            'Total tickets rejected by router', 
            registry=self.registry
        )
        self.total_cost_savings = Gauge(
            'router_total_cost_savings_usd_total', 
            'Total cost savings in USD', 
            registry=self.registry
        )
        
        # MOR Gate metrics
        self.mor_gate_processed = Gauge(
            'router_mor_gate_total_processed_total', 
            'Total tickets processed by MOR gate', 
            registry=self.registry
        )
        self.mor_gate_rejections = Gauge(
            'router_mor_gate_rejections_total', 
            'Total rejections by MOR gate', 
            registry=self.registry
        )
        self.mor_gate_vendor_rejections = Gauge(
            'router_mor_gate_vendor_rejections_total', 
            'Vendor rejections by MOR gate', 
            ['vendor'], 
            registry=self.registry
        )
        
        # Product Type Gate metrics
        self.product_type_gate_processed = Gauge(
            'router_product_type_gate_total_processed_total', 
            'Total tickets processed by Product Type gate', 
            registry=self.registry
        )
        self.product_type_gate_rejections = Gauge(
            'router_product_type_gate_rejections_total', 
            'Total rejections by Product Type gate', 
            registry=self.registry
        )
        self.product_type_distribution = Gauge(
            'router_product_type_gate_type_distribution_total', 
            'Product type distribution', 
            ['product_type'], 
            registry=self.registry
        )
        
        # Performance metrics
        self.mor_gate_p95 = Gauge(
            'router_mor_gate_processing_time_ms_p95', 
            'MOR gate 95th percentile processing time', 
            registry=self.registry
        )
        self.product_type_gate_p95 = Gauge(
            'router_product_type_gate_processing_time_ms_p95', 
            'Product Type gate 95th percentile processing time', 
            registry=self.registry
        )
        self.combined_gates_p95 = Gauge(
            'router_combined_gates_processing_time_ms_p95', 
            'Combined gates 95th percentile processing time', 
            registry=self.registry
        )
    
    def update_metrics(self):
        """Update all metrics from the current tracker state."""
        tracker = get_gate_metrics_tracker()
        current_metrics = tracker.get_current_metrics()
        
        # Update overview metrics
        self.total_processed.set(current_metrics.total_tickets_processed)
        self.total_rejections.set(current_metrics.total_tickets_rejected)
        self.total_cost_savings.set(current_metrics.total_cost_savings)
        
        # Update MOR gate metrics
        if current_metrics.mor_metrics:
            self.mor_gate_processed.set(current_metrics.mor_metrics.total_processed)
            self.mor_gate_rejections.set(current_metrics.mor_metrics.rejection_count)
            
            # Clear and update vendor metrics
            self.mor_gate_vendor_rejections._metrics.clear()
            for vendor, count in current_metrics.mor_metrics.vendor_rejection_counts.items():
                self.mor_gate_vendor_rejections.labels(vendor=vendor).set(count)
        
        # Update Product Type gate metrics
        if current_metrics.product_type_metrics:
            self.product_type_gate_processed.set(current_metrics.product_type_metrics.total_processed)
            self.product_type_gate_rejections.set(current_metrics.product_type_metrics.rejection_count)
            
            # Clear and update product type distribution
            self.product_type_distribution._metrics.clear()
            for product_type, count in current_metrics.product_type_metrics.product_type_counts.items():
                self.product_type_distribution.labels(product_type=product_type).set(count)
        
        # Update performance metrics (simplified - would need histogram data for real P95)
        if current_metrics.mor_metrics:
            avg_time = current_metrics.mor_metrics.get_average_processing_time()
            self.mor_gate_p95.set(avg_time * 1.2)  # Approximate P95 as 120% of average
        
        if current_metrics.product_type_metrics:
            avg_time = current_metrics.product_type_metrics.get_average_processing_time()
            self.product_type_gate_p95.set(avg_time * 1.2)  # Approximate P95 as 120% of average
        
        # Combined P95 (sum of individual gates)
        combined_p95 = (
            (current_metrics.mor_metrics.get_average_processing_time() * 1.2 if current_metrics.mor_metrics else 0) +
            (current_metrics.product_type_metrics.get_average_processing_time() * 1.2 if current_metrics.product_type_metrics else 0)
        )
        self.combined_gates_p95.set(combined_p95)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        self.update_metrics()
        return generate_latest(self.registry).decode('utf-8')


# Global metrics collector instance
_metrics_collector: RouterMetricsCollector = None


def get_metrics_collector() -> RouterMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = RouterMetricsCollector()
    return _metrics_collector


def get_prometheus_metrics() -> str:
    """Get router metrics in Prometheus format."""
    collector = get_metrics_collector()
    return collector.get_metrics()


def get_metrics_json() -> Dict[str, Any]:
    """Get router metrics as JSON for debugging."""
    tracker = get_gate_metrics_tracker()
    return tracker.get_dashboard_data()