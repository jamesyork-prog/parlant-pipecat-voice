"""
HTTP endpoint for exposing async webhook orchestration metrics.

This module provides HTTP endpoints for Prometheus metrics collection
and health status monitoring for the async webhook orchestration system.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from .prometheus_metrics import get_webhook_metrics_collector, get_health_status
from .idempotency_store import IdempotencyStore
from .task_queue import TaskQueue

logger = logging.getLogger(__name__)


class WebhookMetricsEndpoint:
    """
    HTTP endpoint handler for webhook orchestration metrics.
    
    Provides endpoints for:
    - Prometheus metrics export
    - Health status checks
    - System status monitoring
    """
    
    def __init__(
        self,
        idempotency_store: Optional[IdempotencyStore] = None,
        task_queue: Optional[TaskQueue] = None
    ):
        """
        Initialize metrics endpoint.
        
        Args:
            idempotency_store: Optional idempotency store for health checks
            task_queue: Optional task queue for queue metrics
        """
        self.metrics_collector = get_webhook_metrics_collector()
        self.idempotency_store = idempotency_store
        self.task_queue = task_queue
        
        logger.info("WebhookMetricsEndpoint initialized")
    
    async def get_prometheus_metrics(self) -> str:
        """
        Get metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        try:
            # Update system health metrics before export
            await self._update_system_health_metrics()
            
            # Get metrics from collector
            metrics_text = self.metrics_collector.get_metrics()
            
            logger.debug("Prometheus metrics exported successfully")
            return metrics_text
            
        except Exception as e:
            logger.error(f"Error exporting Prometheus metrics: {e}")
            # Return empty metrics on error to avoid breaking monitoring
            return ""
    
    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get system health status.
        
        Returns:
            Dictionary with health status information
        """
        try:
            # Update system health metrics
            await self._update_system_health_metrics()
            
            # Get health summary from metrics collector
            health_summary = get_health_status()
            
            # Add additional health information
            health_summary.update({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "healthy" if self._is_system_healthy(health_summary) else "degraded",
                "version": "1.0.0",
                "component": "async-webhook-orchestration"
            })
            
            return health_summary
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error",
                "error": str(e),
                "component": "async-webhook-orchestration"
            }
    
    async def get_metrics_json(self) -> Dict[str, Any]:
        """
        Get metrics in JSON format for debugging.
        
        Returns:
            Dictionary with all metrics in JSON format
        """
        try:
            # Update system health metrics
            await self._update_system_health_metrics()
            
            # Get health status
            health_status = await self.get_health_status()
            
            # Get queue metrics if available
            queue_metrics = {}
            if self.task_queue:
                try:
                    queue_metrics = await self._get_queue_metrics()
                except Exception as e:
                    logger.warning(f"Error getting queue metrics: {e}")
                    queue_metrics = {"error": str(e)}
            
            # Get idempotency metrics if available
            idempotency_metrics = {}
            if self.idempotency_store:
                try:
                    idempotency_metrics = await self._get_idempotency_metrics()
                except Exception as e:
                    logger.warning(f"Error getting idempotency metrics: {e}")
                    idempotency_metrics = {"error": str(e)}
            
            return {
                "health": health_status,
                "queue": queue_metrics,
                "idempotency": idempotency_metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting JSON metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def _update_system_health_metrics(self):
        """Update system health metrics in the collector."""
        try:
            # Check Redis connection
            redis_connected = True
            if self.idempotency_store:
                try:
                    redis_connected = await self.idempotency_store.is_available()
                except Exception:
                    redis_connected = False
            
            self.metrics_collector.update_dependency_status("redis", redis_connected)
            
            # Check task queue (Hatchet) connection
            hatchet_connected = True
            if self.task_queue:
                try:
                    # Simple health check - try to get queue status
                    hatchet_connected = await self.task_queue.is_healthy()
                except Exception:
                    hatchet_connected = False
            
            self.metrics_collector.update_dependency_status("hatchet", hatchet_connected)
            
            # Database connection is checked in the worker components
            # For now, assume it's connected if we can get here
            self.metrics_collector.update_dependency_status("database", True)
            
            # Update queue depth if available
            if self.task_queue:
                try:
                    queue_depth = await self.task_queue.get_queue_depth()
                    self.metrics_collector.update_queue_depth(queue_depth)
                    
                    dead_letter_size = await self.task_queue.get_dead_letter_queue_size()
                    self.metrics_collector.update_dead_letter_queue_size(dead_letter_size)
                except Exception as e:
                    logger.warning(f"Error updating queue metrics: {e}")
            
        except Exception as e:
            logger.error(f"Error updating system health metrics: {e}")
    
    async def _get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue-specific metrics."""
        if not self.task_queue:
            return {"error": "Task queue not available"}
        
        try:
            return {
                "queue_depth": await self.task_queue.get_queue_depth(),
                "dead_letter_queue_size": await self.task_queue.get_dead_letter_queue_size(),
                "processing_rate": await self.task_queue.get_processing_rate(),
                "active_workers": await self.task_queue.get_active_worker_count()
            }
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_idempotency_metrics(self) -> Dict[str, Any]:
        """Get idempotency store metrics."""
        if not self.idempotency_store:
            return {"error": "Idempotency store not available"}
        
        try:
            return {
                "is_available": await self.idempotency_store.is_available(),
                "connection_status": "connected" if await self.idempotency_store.is_available() else "disconnected"
            }
        except Exception as e:
            return {"error": str(e)}
    
    def _is_system_healthy(self, health_summary: Dict[str, Any]) -> bool:
        """
        Determine if the system is healthy based on health summary.
        
        Args:
            health_summary: Health summary from metrics collector
            
        Returns:
            True if system is healthy, False if degraded
        """
        try:
            # Check webhook success rate
            success_rate = health_summary.get("webhook_success_rate", 0)
            if success_rate < 95.0:  # Less than 95% success rate
                return False
            
            # Check if any critical dependencies are down
            if not health_summary.get("database_connected", True):
                return False
            
            # Check queue depth
            queue_depth = health_summary.get("queue_depth", 0)
            if queue_depth > 500:  # Queue is backing up
                return False
            
            # Check dead letter queue
            dead_letter_size = health_summary.get("dead_letter_queue_size", 0)
            if dead_letter_size > 50:  # Too many failed tasks
                return False
            
            # System is healthy
            return True
            
        except Exception as e:
            logger.error(f"Error determining system health: {e}")
            return False


# Global metrics endpoint instance
_metrics_endpoint: Optional[WebhookMetricsEndpoint] = None


def get_metrics_endpoint(
    idempotency_store: Optional[IdempotencyStore] = None,
    task_queue: Optional[TaskQueue] = None
) -> WebhookMetricsEndpoint:
    """
    Get the global metrics endpoint instance.
    
    Args:
        idempotency_store: Optional idempotency store for health checks
        task_queue: Optional task queue for queue metrics
        
    Returns:
        The global WebhookMetricsEndpoint instance
    """
    global _metrics_endpoint
    if _metrics_endpoint is None:
        _metrics_endpoint = WebhookMetricsEndpoint(idempotency_store, task_queue)
    return _metrics_endpoint


async def handle_metrics_request(path: str) -> tuple[str, int, Dict[str, str]]:
    """
    Handle HTTP metrics requests.
    
    Args:
        path: Request path (/metrics, /health, /metrics.json)
        
    Returns:
        Tuple of (response_body, status_code, headers)
    """
    endpoint = get_metrics_endpoint()
    
    try:
        if path == "/metrics":
            # Prometheus metrics
            metrics_text = await endpoint.get_prometheus_metrics()
            return metrics_text, 200, {"Content-Type": "text/plain; charset=utf-8"}
        
        elif path == "/health":
            # Health status
            health_status = await endpoint.get_health_status()
            status_code = 200 if health_status.get("status") == "healthy" else 503
            return json.dumps(health_status, indent=2), status_code, {"Content-Type": "application/json"}
        
        elif path == "/metrics.json":
            # JSON metrics for debugging
            json_metrics = await endpoint.get_metrics_json()
            return json.dumps(json_metrics, indent=2), 200, {"Content-Type": "application/json"}
        
        else:
            # Not found
            return "Not Found", 404, {"Content-Type": "text/plain"}
    
    except Exception as e:
        logger.error(f"Error handling metrics request {path}: {e}")
        error_response = {"error": str(e), "path": path}
        return json.dumps(error_response), 500, {"Content-Type": "application/json"}