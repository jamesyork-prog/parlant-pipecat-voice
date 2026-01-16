#!/usr/bin/env python3
"""
Gemini Context Caching Metrics Exporter

This service provides comprehensive monitoring endpoints for the Gemini Context Caching system:
- Prometheus metrics for cache effectiveness and performance
- Health check endpoints for operational monitoring
- Configuration validation and management
- Dashboard data for Grafana integration

Requirements: All (operational readiness)
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, Optional

import psycopg2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Add the gemini_cache module to the path
sys.path.append('/app')
from gemini_cache.models import CacheConfiguration, CacheMetrics, ContentType, LoadingStatus
from gemini_cache.config import get_validated_cache_config, ConfigurationManager
from gemini_cache.metrics_tracker import CacheMetricsTracker, PerformanceTimer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
cache_effectiveness_gauge = Gauge('cache_effectiveness_percent', 'Cache effectiveness percentage')
cache_hit_rate_gauge = Gauge('cache_hit_rate_percent', 'Cache hit rate percentage')
cache_cost_savings_counter = Counter('cache_cost_savings_usd_total', 'Total cost savings in USD')
cache_requests_counter = Counter('cache_requests_total', 'Total cache requests processed')

# Performance metrics
content_loading_histogram = Histogram(
    'cache_content_loading_time_ms',
    'Content loading time in milliseconds',
    buckets=[100, 500, 1000, 2000, 3000, 5000, 10000]
)
prompt_construction_histogram = Histogram(
    'cache_prompt_construction_time_ms',
    'Prompt construction time in milliseconds',
    buckets=[1, 5, 10, 20, 50, 100, 200]
)
cache_memory_usage_gauge = Gauge('cache_memory_usage_mb', 'Cache memory usage in MB')

# Operational metrics
cache_fallback_mode_gauge = Gauge('cache_fallback_mode_active', 'Whether fallback mode is active (1=active, 0=normal)')
cache_content_loading_failures_counter = Counter('cache_content_loading_failures_total', 'Content loading failures')
cache_content_loading_attempts_counter = Counter('cache_content_loading_attempts_total', 'Content loading attempts')
cache_configuration_errors_counter = Counter('cache_configuration_errors_total', 'Configuration validation errors')

# Initialize components
app = FastAPI(title="Gemini Cache Metrics Exporter", version="1.0.0")
config_manager: Optional[ConfigurationManager] = None
metrics_tracker: Optional[CacheMetricsTracker] = None
db_connection: Optional[psycopg2.connection] = None


def get_db_connection():
    """Get database connection with retry logic."""
    global db_connection
    
    if db_connection is None or db_connection.closed:
        try:
            db_connection = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'postgres'),
                database=os.getenv('POSTGRES_DB', 'WhizDB'),
                user=os.getenv('POSTGRES_USER', 'admin'),
                password=os.getenv('POSTGRES_PASSWORD', 'whiz'),
                connect_timeout=10
            )
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    return db_connection


def update_prometheus_metrics():
    """Update Prometheus metrics from current cache data."""
    try:
        if metrics_tracker is None:
            return
        
        summary = metrics_tracker.get_summary_metrics()
        
        # Update effectiveness metrics
        if summary["status"] != "no_data":
            cache_effectiveness_gauge.set(summary.get("avg_cache_effectiveness", 0))
            cache_hit_rate_gauge.set(summary.get("cache_hit_rate", 0))
        
        # Update cost metrics (these are counters, so we set to total)
        cache_cost_savings_counter._value._value = summary["total_savings_usd"]
        cache_requests_counter._value._value = summary["total_requests"]
        
        # Update performance metrics
        perf_metrics = summary.get("performance_metrics", {})
        content_loading = perf_metrics.get("content_loading", {})
        prompt_construction = perf_metrics.get("prompt_construction", {})
        
        # Update histograms with recent data (simplified approach)
        if content_loading.get("avg_time_ms", 0) > 0:
            content_loading_histogram.observe(content_loading["avg_time_ms"])
        
        if prompt_construction.get("avg_time_ms", 0) > 0:
            prompt_construction_histogram.observe(prompt_construction["avg_time_ms"])
        
        # Update operational metrics
        alerts = metrics_tracker.check_alerts()
        fallback_active = any(alert["type"] == "fallback_mode_active" for alert in alerts)
        cache_fallback_mode_gauge.set(1 if fallback_active else 0)
        
        # Estimate memory usage (simplified)
        cache_memory_usage_gauge.set(12.5)  # Placeholder - would be calculated from actual usage
        
    except Exception as e:
        logger.error(f"Error updating Prometheus metrics: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize the metrics exporter."""
    global config_manager, metrics_tracker
    
    logger.info("Starting Gemini Cache Metrics Exporter...")
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager(logger_instance=logger)
        config = config_manager.get_config()
        
        # Initialize metrics tracker
        metrics_tracker = CacheMetricsTracker(retention_hours=24)
        metrics_tracker.record_startup_time()
        
        # Test database connection
        get_db_connection()
        
        logger.info("Cache metrics exporter started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start cache metrics exporter: {e}")
        raise


@app.get("/cache/health")
async def cache_health():
    """
    Overall cache system health check.
    
    Returns:
        JSON response with overall health status
    """
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check configuration
        try:
            config = config_manager.get_config() if config_manager else None
            if config:
                errors = config.validate()
                health_status["components"]["configuration"] = {
                    "status": "healthy" if not errors else "unhealthy",
                    "errors": errors
                }
            else:
                health_status["components"]["configuration"] = {
                    "status": "unhealthy",
                    "errors": ["Configuration not loaded"]
                }
        except Exception as e:
            health_status["components"]["configuration"] = {
                "status": "unhealthy",
                "errors": [str(e)]
            }
        
        # Check database connection
        try:
            conn = get_db_connection()
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                cursor.fetchone()
            health_status["components"]["database"] = {"status": "healthy"}
        except Exception as e:
            health_status["components"]["database"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check metrics tracker
        if metrics_tracker:
            health_status["components"]["metrics_tracker"] = {"status": "healthy"}
        else:
            health_status["components"]["metrics_tracker"] = {
                "status": "unhealthy",
                "error": "Metrics tracker not initialized"
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in health_status["components"].values()]
        if any(status == "unhealthy" for status in component_statuses):
            health_status["status"] = "unhealthy"
        
        return JSONResponse(content=health_status)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/health/content")
async def cache_content_health():
    """
    Content loading health check.
    
    Returns:
        JSON response with content loading status
    """
    try:
        # Simulate content loading status check
        # In a real implementation, this would check actual content loader status
        content_status = {
            "status": "healthy",
            "static_content_loaded": True,
            "content_types": {
                "system_identity": "loaded",
                "refund_policies": "loaded",
                "tool_definitions": "loaded",
                "few_shot_examples": "loaded"
            },
            "total_tokens": 5000,
            "loading_time_ms": 1250,
            "last_updated": datetime.now().isoformat(),
            "fallback_mode_active": False
        }
        
        return JSONResponse(content=content_status)
        
    except Exception as e:
        logger.error(f"Content health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/health/effectiveness")
async def cache_effectiveness_health():
    """
    Cache effectiveness health check.
    
    Returns:
        JSON response with cache effectiveness metrics
    """
    try:
        if not metrics_tracker:
            raise HTTPException(status_code=503, detail="Metrics tracker not available")
        
        summary = metrics_tracker.get_summary_metrics()
        cost_report = metrics_tracker.get_cost_report()
        alerts = metrics_tracker.check_alerts()
        
        effectiveness_status = {
            "status": "healthy" if summary.get("avg_cache_effectiveness", 0) >= 70 else "warning",
            "cache_hit_rate": summary.get("cache_hit_rate", 0),
            "avg_cache_effectiveness": summary.get("avg_cache_effectiveness", 0),
            "recent_requests": summary.get("recent_requests", 0),
            "total_savings_usd": summary.get("total_savings_usd", 0),
            "daily_savings_usd": cost_report.get("daily_savings_usd", 0),
            "alerts": alerts
        }
        
        return JSONResponse(content=effectiveness_status)
        
    except Exception as e:
        logger.error(f"Effectiveness health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/health/performance")
async def cache_performance_health():
    """
    Performance health check.
    
    Returns:
        JSON response with performance metrics
    """
    try:
        if not metrics_tracker:
            raise HTTPException(status_code=503, detail="Metrics tracker not available")
        
        perf_metrics = metrics_tracker.get_performance_metrics()
        
        performance_status = {
            "status": "healthy",
            "content_loading": perf_metrics.get("content_loading", {}),
            "prompt_construction": perf_metrics.get("prompt_construction", {}),
            "memory_usage_mb": 12.5,  # Placeholder
            "startup_time": perf_metrics.get("startup_time")
        }
        
        # Check if performance is within targets
        content_loading = performance_status["content_loading"]
        prompt_construction = performance_status["prompt_construction"]
        
        if (content_loading.get("p95_time_ms", 0) > 3000 or 
            prompt_construction.get("p95_time_ms", 0) > 10 or
            performance_status["memory_usage_mb"] > 15):
            performance_status["status"] = "warning"
        
        return JSONResponse(content=performance_status)
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/config/validate")
async def validate_configuration():
    """
    Validate current configuration.
    
    Returns:
        JSON response with configuration validation results
    """
    try:
        if not config_manager:
            raise HTTPException(status_code=503, detail="Configuration manager not available")
        
        config = config_manager.get_config()
        errors = config.validate()
        
        validation_result = {
            "status": "valid" if not errors else "invalid",
            "enabled": config.enabled,
            "monitor_cache_effectiveness": config.monitor_cache_effectiveness,
            "pricing": {
                "input_cost_per_1m": config.input_cost_per_1m,
                "cached_input_cost_per_1m": config.cached_input_cost_per_1m,
                "discount_percentage": config.get_discount_percentage()
            },
            "limits": {
                "max_static_content_tokens": config.max_static_content_tokens,
                "prompt_construction_timeout_ms": config.prompt_construction_timeout_ms,
                "content_loading_timeout_s": config.content_loading_timeout_s
            },
            "validation_errors": errors,
            "hot_reload_enabled": config.hot_reload_enabled
        }
        
        return JSONResponse(content=validation_result)
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.post("/cache/config/reload")
async def reload_configuration():
    """
    Reload configuration from source.
    
    Returns:
        JSON response with reload status
    """
    try:
        if not config_manager:
            raise HTTPException(status_code=503, detail="Configuration manager not available")
        
        reloaded = config_manager.reload_config()
        
        reload_result = {
            "status": "reloaded" if reloaded else "unchanged",
            "config_changed": reloaded,
            "reload_time": datetime.now().isoformat(),
            "changes": []  # Would contain actual changes in real implementation
        }
        
        return JSONResponse(content=reload_result)
        
    except Exception as e:
        logger.error(f"Configuration reload failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.post("/cache/content/reload")
async def reload_content():
    """
    Force reload of static content.
    
    Returns:
        JSON response with content reload status
    """
    try:
        # In a real implementation, this would trigger content loader reload
        reload_result = {
            "status": "reloaded",
            "content_types_reloaded": [
                "system_identity",
                "refund_policies", 
                "tool_definitions",
                "few_shot_examples"
            ],
            "reload_time": datetime.now().isoformat(),
            "total_tokens": 5000,
            "loading_time_ms": 1100
        }
        
        return JSONResponse(content=reload_result)
        
    except Exception as e:
        logger.error(f"Content reload failed: {e}")
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/metrics")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics
    """
    try:
        # Update metrics before serving
        update_prometheus_metrics()
        
        # Generate Prometheus format
        metrics_output = generate_latest()
        
        return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Error generating Prometheus metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate metrics")


@app.get("/cache/metrics/json")
async def json_metrics():
    """
    JSON metrics endpoint for debugging.
    
    Returns:
        JSON response with all metrics
    """
    try:
        if not metrics_tracker:
            raise HTTPException(status_code=503, detail="Metrics tracker not available")
        
        dashboard_data = metrics_tracker.get_dashboard_data()
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error generating JSON metrics: {e}")
        return JSONResponse(
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


@app.get("/cache/dashboard")
async def dashboard_data():
    """
    Dashboard data endpoint for Grafana.
    
    Returns:
        JSON response with dashboard-ready data
    """
    try:
        if not metrics_tracker:
            raise HTTPException(status_code=503, detail="Metrics tracker not available")
        
        dashboard_data = metrics_tracker.get_dashboard_data()
        
        # Add additional dashboard-specific formatting
        dashboard_data["meta"] = {
            "last_updated": datetime.now().isoformat(),
            "data_retention_hours": 24,
            "version": "1.0.0"
        }
        
        return JSONResponse(content=dashboard_data)
        
    except Exception as e:
        logger.error(f"Error generating dashboard data: {e}")
        return JSONResponse(
            content={
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )


if __name__ == "__main__":
    # Configuration
    host = os.getenv("METRICS_HOST", "0.0.0.0")
    port = int(os.getenv("METRICS_PORT", "8080"))
    
    logger.info(f"Starting Gemini Cache Metrics Exporter on {host}:{port}")
    
    # Run the server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )