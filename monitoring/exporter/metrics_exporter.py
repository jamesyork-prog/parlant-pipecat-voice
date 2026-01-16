#!/usr/bin/env python3
"""
Prometheus metrics exporter for MOR/Product Type Router.

This service collects metrics from the router's database tables and
exposes them in Prometheus format for monitoring and alerting.
"""

import os
import time
import logging
import psycopg2
from typing import Dict, Any, Optional
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
from prometheus_client.core import CollectorRegistry
import json
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
registry = CollectorRegistry()

# Overview metrics
total_processed = Gauge('router_total_processed_total', 'Total tickets processed by router', registry=registry)
total_rejections = Gauge('router_total_rejections_total', 'Total tickets rejected by router', registry=registry)
total_cost_savings = Gauge('router_total_cost_savings_usd_total', 'Total cost savings in USD', registry=registry)

# MOR Gate metrics
mor_gate_processed = Gauge('router_mor_gate_total_processed_total', 'Total tickets processed by MOR gate', registry=registry)
mor_gate_rejections = Gauge('router_mor_gate_rejections_total', 'Total rejections by MOR gate', registry=registry)
mor_gate_vendor_rejections = Gauge('router_mor_gate_vendor_rejections_total', 'Vendor rejections by MOR gate', ['vendor'], registry=registry)
mor_gate_unknown_partner = Gauge('router_mor_gate_unknown_partner_total', 'Unknown partner count in MOR gate', registry=registry)
mor_gate_processing_time = Histogram('router_mor_gate_processing_time_ms', 'MOR gate processing time in milliseconds', 
                                   buckets=[1, 5, 10, 15, 20, 25, 30, 50, 100], registry=registry)

# Product Type Gate metrics
product_type_gate_processed = Gauge('router_product_type_gate_total_processed_total', 'Total tickets processed by Product Type gate', registry=registry)
product_type_gate_rejections = Gauge('router_product_type_gate_rejections_total', 'Total rejections by Product Type gate', registry=registry)
product_type_distribution = Gauge('router_product_type_gate_type_distribution_total', 'Product type distribution', ['product_type'], registry=registry)
product_type_duration_failures = Gauge('router_product_type_gate_duration_failures_total', 'Duration calculation failures', registry=registry)
product_type_processing_time = Histogram('router_product_type_gate_processing_time_ms', 'Product Type gate processing time in milliseconds',
                                        buckets=[1, 5, 10, 20, 30, 50, 75, 100, 150], registry=registry)

# Combined metrics
combined_processing_time = Histogram('router_combined_gates_processing_time_ms', 'Combined gates processing time in milliseconds',
                                   buckets=[1, 10, 25, 50, 75, 100, 150, 200, 300], registry=registry)

# Performance percentiles (calculated from histograms)
mor_gate_p95 = Gauge('router_mor_gate_processing_time_ms_p95', 'MOR gate 95th percentile processing time', registry=registry)
product_type_gate_p95 = Gauge('router_product_type_gate_processing_time_ms_p95', 'Product Type gate 95th percentile processing time', registry=registry)
combined_gates_p95 = Gauge('router_combined_gates_processing_time_ms_p95', 'Combined gates 95th percentile processing time', registry=registry)

# System info
router_info = Info('router_system_info', 'Router system information', registry=registry)


class RouterMetricsExporter:
    """Exports router metrics to Prometheus."""
    
    def __init__(self):
        """Initialize the metrics exporter."""
        self.db_config = {
            'host': os.getenv('POSTGRES_HOST', 'postgres'),
            'port': int(os.getenv('POSTGRES_PORT', 5432)),
            'database': os.getenv('POSTGRES_DB', 'WhizDB'),
            'user': os.getenv('POSTGRES_USER', 'admin'),
            'password': os.getenv('POSTGRES_PASSWORD', 'whiz')
        }
        
        self.metrics_port = int(os.getenv('METRICS_PORT', 8080))
        self.metrics_interval = int(os.getenv('METRICS_INTERVAL', 30))
        
        # Initialize system info
        router_info.info({
            'version': '1.0.0',
            'component': 'mor-product-type-router',
            'exporter_version': '1.0.0'
        })
        
        logger.info(f"Initialized metrics exporter on port {self.metrics_port}")
    
    def get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get database connection."""
        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return None
    
    def collect_metrics(self) -> None:
        """Collect all router metrics from database."""
        conn = self.get_db_connection()
        if not conn:
            logger.error("Cannot collect metrics - no database connection")
            return
        
        try:
            with conn.cursor() as cursor:
                # Collect MOR gate metrics
                self._collect_mor_gate_metrics(cursor)
                
                # Collect Product Type gate metrics
                self._collect_product_type_gate_metrics(cursor)
                
                # Collect combined metrics
                self._collect_combined_metrics(cursor)
                
                # Calculate performance percentiles
                self._calculate_performance_percentiles(cursor)
                
            logger.info("Successfully collected router metrics")
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        finally:
            conn.close()
    
    def _collect_mor_gate_metrics(self, cursor) -> None:
        """Collect MOR gate specific metrics."""
        try:
            # Get MOR gate summary from audit logs
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_processed,
                    COUNT(*) FILTER (WHERE decision = 'reject') as rejections,
                    AVG(processing_time_ms) as avg_processing_time
                FROM routing_audit_log 
                WHERE gate_name = 'mor_gate' 
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            result = cursor.fetchone()
            if result:
                processed, rejections, avg_time = result
                mor_gate_processed.set(processed or 0)
                mor_gate_rejections.set(rejections or 0)
                
                # Update processing time histogram (simulate distribution)
                if avg_time:
                    mor_gate_processing_time.observe(float(avg_time))
            
            # Get vendor rejection breakdown
            cursor.execute("""
                SELECT 
                    metadata->>'vendor' as vendor,
                    COUNT(*) as count
                FROM routing_audit_log 
                WHERE gate_name = 'mor_gate' 
                AND decision = 'reject'
                AND metadata->>'vendor' IS NOT NULL
                AND created_at > NOW() - INTERVAL '24 hours'
                GROUP BY metadata->>'vendor'
            """)
            
            # Clear existing vendor metrics
            mor_gate_vendor_rejections._metrics.clear()
            
            for vendor, count in cursor.fetchall():
                mor_gate_vendor_rejections.labels(vendor=vendor).set(count)
            
            # Get unknown partner count
            cursor.execute("""
                SELECT COUNT(*) 
                FROM routing_audit_log 
                WHERE gate_name = 'mor_gate' 
                AND metadata->>'warning' LIKE '%Unknown Partner%'
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            unknown_count = cursor.fetchone()[0]
            mor_gate_unknown_partner.set(unknown_count or 0)
            
        except Exception as e:
            logger.error(f"Error collecting MOR gate metrics: {e}")
    
    def _collect_product_type_gate_metrics(self, cursor) -> None:
        """Collect Product Type gate specific metrics."""
        try:
            # Get Product Type gate summary
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_processed,
                    COUNT(*) FILTER (WHERE decision = 'reject') as rejections,
                    AVG(processing_time_ms) as avg_processing_time
                FROM routing_audit_log 
                WHERE gate_name = 'product_type_gate' 
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            result = cursor.fetchone()
            if result:
                processed, rejections, avg_time = result
                product_type_gate_processed.set(processed or 0)
                product_type_gate_rejections.set(rejections or 0)
                
                # Update processing time histogram
                if avg_time:
                    product_type_processing_time.observe(float(avg_time))
            
            # Get product type distribution
            cursor.execute("""
                SELECT 
                    metadata->>'product_type' as product_type,
                    COUNT(*) as count
                FROM routing_audit_log 
                WHERE gate_name = 'product_type_gate' 
                AND metadata->>'product_type' IS NOT NULL
                AND created_at > NOW() - INTERVAL '24 hours'
                GROUP BY metadata->>'product_type'
            """)
            
            # Clear existing product type metrics
            product_type_distribution._metrics.clear()
            
            for product_type, count in cursor.fetchall():
                product_type_distribution.labels(product_type=product_type).set(count)
            
            # Get duration calculation failures
            cursor.execute("""
                SELECT COUNT(*) 
                FROM routing_audit_log 
                WHERE gate_name = 'product_type_gate' 
                AND metadata->>'warning' LIKE '%Duration calculation%'
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            failure_count = cursor.fetchone()[0]
            product_type_duration_failures.set(failure_count or 0)
            
        except Exception as e:
            logger.error(f"Error collecting Product Type gate metrics: {e}")
    
    def _collect_combined_metrics(self, cursor) -> None:
        """Collect combined router metrics."""
        try:
            # Get overall router statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_processed,
                    COUNT(*) FILTER (WHERE decision = 'reject') as total_rejections,
                    SUM(CASE 
                        WHEN decision = 'reject' THEN 0.12 
                        ELSE 0 
                    END) as cost_savings
                FROM routing_audit_log 
                WHERE gate_name IN ('mor_gate', 'product_type_gate')
                AND created_at > NOW() - INTERVAL '24 hours'
            """)
            
            result = cursor.fetchone()
            if result:
                processed, rejections, savings = result
                total_processed.set(processed or 0)
                total_rejections.set(rejections or 0)
                total_cost_savings.set(float(savings or 0))
            
            # Get combined processing times (simulate)
            cursor.execute("""
                SELECT AVG(processing_time_ms) 
                FROM routing_audit_log 
                WHERE gate_name IN ('mor_gate', 'product_type_gate')
                AND created_at > NOW() - INTERVAL '1 hour'
            """)
            
            avg_combined_time = cursor.fetchone()[0]
            if avg_combined_time:
                # Simulate combined processing time (sum of both gates)
                combined_processing_time.observe(float(avg_combined_time) * 2)
            
        except Exception as e:
            logger.error(f"Error collecting combined metrics: {e}")
    
    def _calculate_performance_percentiles(self, cursor) -> None:
        """Calculate performance percentiles from historical data."""
        try:
            # Calculate MOR gate P95
            cursor.execute("""
                SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_time_ms)
                FROM routing_audit_log 
                WHERE gate_name = 'mor_gate' 
                AND processing_time_ms IS NOT NULL
                AND created_at > NOW() - INTERVAL '1 hour'
            """)
            
            mor_p95 = cursor.fetchone()[0]
            if mor_p95:
                mor_gate_p95.set(float(mor_p95))
            
            # Calculate Product Type gate P95
            cursor.execute("""
                SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_time_ms)
                FROM routing_audit_log 
                WHERE gate_name = 'product_type_gate' 
                AND processing_time_ms IS NOT NULL
                AND created_at > NOW() - INTERVAL '1 hour'
            """)
            
            pt_p95 = cursor.fetchone()[0]
            if pt_p95:
                product_type_gate_p95.set(float(pt_p95))
            
            # Calculate combined P95 (approximate)
            if mor_p95 and pt_p95:
                combined_gates_p95.set(float(mor_p95) + float(pt_p95))
            
        except Exception as e:
            logger.error(f"Error calculating performance percentiles: {e}")
    
    def run(self) -> None:
        """Run the metrics exporter."""
        # Start Prometheus HTTP server
        start_http_server(self.metrics_port, registry=registry)
        logger.info(f"Metrics server started on port {self.metrics_port}")
        
        # Collect metrics periodically
        while True:
            try:
                self.collect_metrics()
                time.sleep(self.metrics_interval)
            except KeyboardInterrupt:
                logger.info("Shutting down metrics exporter")
                break
            except Exception as e:
                logger.error(f"Unexpected error in metrics collection: {e}")
                time.sleep(self.metrics_interval)


if __name__ == '__main__':
    exporter = RouterMetricsExporter()
    exporter.run()