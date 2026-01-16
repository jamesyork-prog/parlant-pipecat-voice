"""
Worker load balancer for horizontal scaling support.

This module provides the WorkerLoadBalancer class that implements:
- Even task distribution across multiple workers
- Worker health monitoring and task reassignment
- Worker removal handling without data loss
- Load balancing strategies for optimal performance
"""

import asyncio
import time
import logging
import uuid
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

from .task_queue import Task, TaskStatus, TaskPriority
from .prometheus_metrics import get_webhook_metrics_collector

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    REMOVED = "removed"
    STARTING = "starting"


@dataclass
class WorkerInfo:
    """Information about a worker instance."""
    worker_id: str
    status: WorkerStatus
    last_heartbeat: datetime
    task_count: int = 0
    processing_capacity: int = 10  # Max concurrent tasks
    current_tasks: Set[str] = field(default_factory=set)
    total_processed: int = 0
    error_count: int = 0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def is_available(self) -> bool:
        """Check if worker is available for new tasks."""
        return (
            self.status == WorkerStatus.HEALTHY and
            self.task_count < self.processing_capacity
        )
    
    @property
    def load_factor(self) -> float:
        """Calculate current load factor (0.0 to 1.0)."""
        if self.processing_capacity == 0:
            return 1.0
        return self.task_count / self.processing_capacity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "worker_id": self.worker_id,
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "task_count": self.task_count,
            "processing_capacity": self.processing_capacity,
            "current_tasks": list(self.current_tasks),
            "total_processed": self.total_processed,
            "error_count": self.error_count,
            "started_at": self.started_at.isoformat(),
            "load_factor": self.load_factor,
            "is_available": self.is_available
        }


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"


class WorkerLoadBalancer:
    """
    Worker load balancer for horizontal scaling.
    
    Implements requirements:
    - 7.1: Even task distribution across multiple workers
    - 7.3: Worker removal handling without data loss
    - Worker health monitoring and task reassignment
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_LOADED,
        health_check_interval: int = 30,
        heartbeat_timeout: int = 90
    ):
        """
        Initialize worker load balancer.
        
        Args:
            strategy: Load balancing strategy to use
            health_check_interval: Seconds between health checks
            heartbeat_timeout: Seconds before marking worker unhealthy
        """
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.heartbeat_timeout = heartbeat_timeout
        
        # Worker registry
        self.workers: Dict[str, WorkerInfo] = {}
        self.round_robin_index = 0
        
        # Task assignment tracking
        self.task_assignments: Dict[str, str] = {}  # task_id -> worker_id
        self.pending_reassignments: List[Task] = []
        
        # Health monitoring
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._is_monitoring = False
        
        logger.info(f"WorkerLoadBalancer initialized with strategy: {strategy.value}")
    
    async def start_monitoring(self):
        """Start worker health monitoring."""
        if self._is_monitoring:
            logger.warning("Health monitoring already started")
            return
        
        self._is_monitoring = True
        self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Worker health monitoring started")
    
    async def stop_monitoring(self):
        """Stop worker health monitoring."""
        self._is_monitoring = False
        
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
            self._health_monitor_task = None
        
        logger.info("Worker health monitoring stopped")
    
    async def register_worker(
        self,
        worker_id: str,
        processing_capacity: int = 10
    ) -> bool:
        """
        Register a new worker with the load balancer.
        
        Args:
            worker_id: Unique worker identifier
            processing_capacity: Maximum concurrent tasks for this worker
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already registered, updating info")
            
            worker_info = WorkerInfo(
                worker_id=worker_id,
                status=WorkerStatus.STARTING,
                last_heartbeat=datetime.now(timezone.utc),
                processing_capacity=processing_capacity
            )
            
            self.workers[worker_id] = worker_info
            
            # Record worker registration metrics
            metrics_collector = get_webhook_metrics_collector()
            metrics_collector.record_worker_registration(worker_id, processing_capacity)
            
            logger.info(f"Worker {worker_id} registered with capacity {processing_capacity}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register worker {worker_id}: {e}")
            return False
    
    async def unregister_worker(self, worker_id: str) -> bool:
        """
        Unregister a worker and reassign its tasks.
        
        Implements requirement 7.3: Worker removal handling without data loss.
        
        Args:
            worker_id: Worker identifier to remove
            
        Returns:
            True if unregistration successful, False otherwise
        """
        try:
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Worker {worker_id} not found for unregistration")
                return False
            
            logger.info(f"Unregistering worker {worker_id} with {len(worker.current_tasks)} active tasks")
            
            # Mark worker as removed
            worker.status = WorkerStatus.REMOVED
            
            # Reassign all current tasks to other workers
            tasks_to_reassign = list(worker.current_tasks)
            reassignment_count = 0
            
            for task_id in tasks_to_reassign:
                if await self._reassign_task(task_id, worker_id):
                    reassignment_count += 1
            
            # Remove worker from registry
            del self.workers[worker_id]
            
            # Record worker unregistration metrics
            metrics_collector = get_webhook_metrics_collector()
            metrics_collector.record_worker_unregistration(worker_id, reassignment_count)
            
            logger.info(f"Worker {worker_id} unregistered, reassigned {reassignment_count} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unregister worker {worker_id}: {e}")
            return False
    
    async def worker_heartbeat(self, worker_id: str, stats: Dict[str, Any] = None) -> bool:
        """
        Record worker heartbeat and update stats.
        
        Args:
            worker_id: Worker identifier
            stats: Optional worker statistics
            
        Returns:
            True if heartbeat recorded, False otherwise
        """
        try:
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Heartbeat from unregistered worker {worker_id}")
                return False
            
            # Update heartbeat timestamp
            worker.last_heartbeat = datetime.now(timezone.utc)
            
            # Update worker status to healthy if it was starting
            if worker.status == WorkerStatus.STARTING:
                worker.status = WorkerStatus.HEALTHY
                logger.info(f"Worker {worker_id} is now healthy")
            
            # Update stats if provided
            if stats:
                worker.task_count = stats.get("task_count", worker.task_count)
                worker.total_processed = stats.get("total_processed", worker.total_processed)
                worker.error_count = stats.get("error_count", worker.error_count)
                
                # Update current tasks set
                current_tasks = stats.get("current_tasks", [])
                worker.current_tasks = set(current_tasks)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record heartbeat for worker {worker_id}: {e}")
            return False
    
    async def assign_task(self, task: Task) -> Optional[str]:
        """
        Assign task to the best available worker.
        
        Implements requirement 7.1: Even task distribution across multiple workers.
        
        Args:
            task: Task to assign
            
        Returns:
            Worker ID if assignment successful, None otherwise
        """
        try:
            # Get available workers
            available_workers = [
                worker for worker in self.workers.values()
                if worker.is_available
            ]
            
            if not available_workers:
                logger.warning("No available workers for task assignment")
                return None
            
            # Select worker based on strategy
            selected_worker = await self._select_worker(available_workers, task)
            
            if not selected_worker:
                logger.error("Failed to select worker for task assignment")
                return None
            
            # Assign task to worker
            selected_worker.current_tasks.add(task.task_id)
            selected_worker.task_count += 1
            self.task_assignments[task.task_id] = selected_worker.worker_id
            
            # Record assignment metrics
            metrics_collector = get_webhook_metrics_collector()
            metrics_collector.record_task_assignment(
                task.task_id,
                selected_worker.worker_id,
                selected_worker.load_factor
            )
            
            logger.info(f"Task {task.task_id} assigned to worker {selected_worker.worker_id} "
                       f"(load: {selected_worker.load_factor:.2f})")
            
            return selected_worker.worker_id
            
        except Exception as e:
            logger.error(f"Failed to assign task {task.task_id}: {e}")
            return None
    
    async def task_completed(self, task_id: str, worker_id: str) -> bool:
        """
        Mark task as completed and update worker stats.
        
        Args:
            task_id: Completed task identifier
            worker_id: Worker that completed the task
            
        Returns:
            True if completion recorded, False otherwise
        """
        try:
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Task completion from unknown worker {worker_id}")
                return False
            
            # Remove task from worker
            worker.current_tasks.discard(task_id)
            worker.task_count = max(0, worker.task_count - 1)
            worker.total_processed += 1
            
            # Remove from assignments
            self.task_assignments.pop(task_id, None)
            
            logger.debug(f"Task {task_id} completed by worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record task completion {task_id}: {e}")
            return False
    
    async def task_failed(self, task_id: str, worker_id: str) -> bool:
        """
        Mark task as failed and update worker stats.
        
        Args:
            task_id: Failed task identifier
            worker_id: Worker that failed the task
            
        Returns:
            True if failure recorded, False otherwise
        """
        try:
            worker = self.workers.get(worker_id)
            if not worker:
                logger.warning(f"Task failure from unknown worker {worker_id}")
                return False
            
            # Remove task from worker
            worker.current_tasks.discard(task_id)
            worker.task_count = max(0, worker.task_count - 1)
            worker.error_count += 1
            
            # Remove from assignments
            self.task_assignments.pop(task_id, None)
            
            logger.debug(f"Task {task_id} failed on worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record task failure {task_id}: {e}")
            return False
    
    async def _select_worker(self, available_workers: List[WorkerInfo], task: Task) -> Optional[WorkerInfo]:
        """
        Select the best worker for task assignment based on strategy.
        
        Args:
            available_workers: List of available workers
            task: Task to assign
            
        Returns:
            Selected worker or None
        """
        if not available_workers:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            # Round robin selection
            worker = available_workers[self.round_robin_index % len(available_workers)]
            self.round_robin_index += 1
            return worker
        
        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            # Select worker with lowest load factor
            return min(available_workers, key=lambda w: w.load_factor)
        
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            # Weighted round robin based on processing capacity
            total_capacity = sum(w.processing_capacity for w in available_workers)
            if total_capacity == 0:
                return available_workers[0]
            
            # Select based on capacity weight
            weights = [w.processing_capacity / total_capacity for w in available_workers]
            
            # Simple weighted selection (could be improved with proper weighted round robin)
            import random
            return random.choices(available_workers, weights=weights)[0]
        
        else:
            # Default to first available worker
            return available_workers[0]
    
    async def _reassign_task(self, task_id: str, failed_worker_id: str) -> bool:
        """
        Reassign a task from a failed worker to another worker.
        
        Args:
            task_id: Task to reassign
            failed_worker_id: Worker that failed or was removed
            
        Returns:
            True if reassignment successful, False otherwise
        """
        try:
            # Find available workers (excluding the failed one)
            available_workers = [
                worker for worker in self.workers.values()
                if worker.is_available and worker.worker_id != failed_worker_id
            ]
            
            if not available_workers:
                logger.error(f"No available workers to reassign task {task_id}")
                return False
            
            # Create a dummy task for selection (we don't have the full task object)
            dummy_task = Task(
                task_id=task_id,
                payload={},
                created_at=datetime.now(timezone.utc),
                retry_count=0,
                max_retries=3,
                priority=TaskPriority.NORMAL.value,
                worker_id=None,
                status=TaskStatus.PENDING
            )
            
            # Select new worker
            selected_worker = await self._select_worker(available_workers, dummy_task)
            
            if not selected_worker:
                logger.error(f"Failed to select worker for task reassignment {task_id}")
                return False
            
            # Update assignments
            selected_worker.current_tasks.add(task_id)
            selected_worker.task_count += 1
            self.task_assignments[task_id] = selected_worker.worker_id
            
            logger.info(f"Task {task_id} reassigned from {failed_worker_id} to {selected_worker.worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reassign task {task_id}: {e}")
            return False
    
    async def _health_monitor_loop(self):
        """Health monitoring loop to check worker status."""
        while self._is_monitoring:
            try:
                await self._check_worker_health()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _check_worker_health(self):
        """Check health of all registered workers."""
        current_time = datetime.now(timezone.utc)
        unhealthy_workers = []
        
        for worker_id, worker in self.workers.items():
            if worker.status == WorkerStatus.REMOVED:
                continue
            
            # Check if worker has missed heartbeat
            time_since_heartbeat = (current_time - worker.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > self.heartbeat_timeout:
                if worker.status == WorkerStatus.HEALTHY:
                    logger.warning(f"Worker {worker_id} marked unhealthy (no heartbeat for {time_since_heartbeat:.1f}s)")
                    worker.status = WorkerStatus.UNHEALTHY
                    unhealthy_workers.append(worker)
                    
                    # Record unhealthy worker metrics
                    metrics_collector = get_webhook_metrics_collector()
                    metrics_collector.record_worker_health_change(worker_id, "unhealthy")
        
        # Reassign tasks from unhealthy workers
        for worker in unhealthy_workers:
            tasks_to_reassign = list(worker.current_tasks)
            for task_id in tasks_to_reassign:
                await self._reassign_task(task_id, worker.worker_id)
            
            # Clear tasks from unhealthy worker
            worker.current_tasks.clear()
            worker.task_count = 0
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """
        Get load balancer statistics for monitoring.
        
        Returns:
            Dictionary with load balancer statistics
        """
        healthy_workers = [w for w in self.workers.values() if w.status == WorkerStatus.HEALTHY]
        unhealthy_workers = [w for w in self.workers.values() if w.status == WorkerStatus.UNHEALTHY]
        
        total_capacity = sum(w.processing_capacity for w in healthy_workers)
        total_load = sum(w.task_count for w in healthy_workers)
        
        return {
            "strategy": self.strategy.value,
            "total_workers": len(self.workers),
            "healthy_workers": len(healthy_workers),
            "unhealthy_workers": len(unhealthy_workers),
            "total_capacity": total_capacity,
            "total_load": total_load,
            "overall_load_factor": total_load / total_capacity if total_capacity > 0 else 0.0,
            "active_assignments": len(self.task_assignments),
            "pending_reassignments": len(self.pending_reassignments),
            "workers": [worker.to_dict() for worker in self.workers.values()]
        }
    
    def get_worker_distribution(self) -> Dict[str, int]:
        """
        Get current task distribution across workers.
        
        Returns:
            Dictionary mapping worker_id to task count
        """
        return {
            worker_id: worker.task_count
            for worker_id, worker in self.workers.items()
            if worker.status != WorkerStatus.REMOVED
        }