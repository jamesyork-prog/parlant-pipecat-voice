"""
Task queue interface for Hatchet integration.

This module provides the TaskQueue class that implements:
- Task creation with unique identifiers
- Hatchet workflow orchestration integration
- Fallback to synchronous processing when Hatchet unavailable
- FIFO queue processing with priority support
- Worker load balancing for horizontal scaling
"""

import uuid
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone

from .priority_queue_manager import PriorityQueueManager

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class Task:
    """Task data structure for queue processing."""
    task_id: str
    payload: Dict[str, Any]
    created_at: datetime
    retry_count: int
    max_retries: int
    priority: int
    worker_id: Optional[str]
    status: TaskStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data


@dataclass
class TaskResult:
    """Result of task processing."""
    task_id: str
    success: bool
    result_data: Optional[Dict[str, Any]]
    error: Optional[str]
    processing_time: float
    retry_count: int
    completed_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        data = asdict(self)
        data['completed_at'] = self.completed_at.isoformat()
        return data


class TaskQueue:
    """
    Task queue interface for Hatchet integration.
    
    Implements requirements:
    - 1.2: Task creation with unique identifiers
    - 6.1: Fallback to synchronous processing when Hatchet unavailable
    - 7.4: FIFO ordering for tasks of equal priority
    - 7.1: Even task distribution across multiple workers (via load balancer)
    """
    
    def __init__(self, hatchet_client=None, fallback_processor=None, load_balancer=None):
        """
        Initialize task queue.
        
        Args:
            hatchet_client: Hatchet client for workflow orchestration
            fallback_processor: Synchronous processor for fallback mode
            load_balancer: WorkerLoadBalancer for horizontal scaling
        """
        self.hatchet_client = hatchet_client
        self.fallback_processor = fallback_processor
        self.load_balancer = load_balancer
        self._available = hatchet_client is not None
        self._local_queue: List[Task] = []  # For fallback mode
        self._processing_tasks: Dict[str, Task] = {}
        
        # Priority queue manager for FIFO processing
        self._priority_queue_manager = PriorityQueueManager()
        
    async def enqueue(self, task_data: Dict[str, Any], priority: int = TaskPriority.NORMAL.value) -> str:
        """
        Add task to queue with optional priority.
        
        Implements requirements:
        - 1.2: Return unique task identifier
        - 6.1: Fallback when Hatchet unavailable
        - 7.1: Even task distribution via load balancer
        
        Args:
            task_data: Task payload data
            priority: Task priority (default: NORMAL)
            
        Returns:
            Unique task identifier
        """
        # Generate unique task ID
        task_id = str(uuid.uuid4())
        
        # Create task object
        task = Task(
            task_id=task_id,
            payload=task_data,
            created_at=datetime.now(timezone.utc),
            retry_count=0,
            max_retries=3,  # Default max retries
            priority=priority,
            worker_id=None,
            status=TaskStatus.PENDING
        )
        
        if self._available and self.hatchet_client:
            try:
                # Enqueue with Hatchet and load balancer
                await self._enqueue_with_hatchet(task)
                logger.info(f"Task {task_id} enqueued with Hatchet")
                return task_id
            except Exception as e:
                logger.error(f"Failed to enqueue task with Hatchet: {e}")
                self._available = False
                # Fall through to fallback mode
        
        # Fallback mode - add to local queue with load balancing
        await self._enqueue_fallback(task)
        logger.warning(f"Task {task_id} enqueued in fallback mode (Hatchet unavailable)")
        return task_id
    
    async def _enqueue_with_hatchet(self, task: Task):
        """Enqueue task with Hatchet workflow orchestration."""
        if not self.hatchet_client:
            raise Exception("Hatchet client not available")
        
        # Use load balancer to assign worker if available
        if self.load_balancer:
            assigned_worker = await self.load_balancer.assign_task(task)
            if assigned_worker:
                task.worker_id = assigned_worker
                logger.info(f"Task {task.task_id} assigned to worker {assigned_worker}")
        
        # Convert task to Hatchet workflow input
        workflow_input = {
            "task_id": task.task_id,
            "payload": task.payload,
            "priority": task.priority,
            "created_at": task.created_at.isoformat(),
            "max_retries": task.max_retries,
            "assigned_worker": task.worker_id
        }
        
        # Submit to Hatchet workflow
        # Note: This is a placeholder - actual Hatchet integration would use
        # the real Hatchet client API
        await self.hatchet_client.admin.run_workflow(
            "webhook-processing-workflow",
            workflow_input
        )
    
    async def _enqueue_fallback(self, task: Task):
        """Enqueue task in fallback mode using priority queue manager."""
        # Use load balancer to assign worker if available
        if self.load_balancer:
            assigned_worker = await self.load_balancer.assign_task(task)
            if assigned_worker:
                task.worker_id = assigned_worker
                logger.info(f"Task {task.task_id} assigned to worker {assigned_worker} in fallback mode")
        
        # Use priority queue manager for FIFO ordering within priority levels
        success = self._priority_queue_manager.enqueue(task)
        if not success:
            logger.error(f"Failed to enqueue task {task.task_id} in priority queue manager")
            return
        
        # Also add to local queue for backward compatibility
        # Insert task maintaining FIFO order within priority levels
        inserted = False
        for i, existing_task in enumerate(self._local_queue):
            if existing_task.priority < task.priority:
                self._local_queue.insert(i, task)
                inserted = True
                break
            elif existing_task.priority == task.priority:
                # Same priority - continue to find the end of this priority group
                # to maintain FIFO order
                continue
        
        if not inserted:
            self._local_queue.append(task)
        
        # If fallback processor is available, process immediately
        if self.fallback_processor:
            asyncio.create_task(self._process_fallback_task(task))
    
    async def _process_fallback_task(self, task: Task):
        """Process task in fallback mode."""
        try:
            task.status = TaskStatus.IN_PROGRESS
            self._processing_tasks[task.task_id] = task
            
            # Remove from queue
            if task in self._local_queue:
                self._local_queue.remove(task)
            
            start_time = time.time()
            
            # Process with fallback processor
            result = await self.fallback_processor(task.payload)
            
            processing_time = time.time() - start_time
            
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                success=True,
                result_data=result,
                error=None,
                processing_time=processing_time,
                retry_count=task.retry_count,
                completed_at=datetime.now(timezone.utc)
            )
            
            task.status = TaskStatus.COMPLETED
            await self._complete_task(task.task_id, task_result)
            
        except Exception as e:
            logger.error(f"Fallback task processing failed for {task.task_id}: {e}")
            
            # Handle retry logic
            task.retry_count += 1
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.PENDING
                await self._enqueue_fallback(task)  # Re-enqueue for retry
            else:
                task.status = TaskStatus.DEAD_LETTER
                await self._dead_letter_task(task.task_id, str(e))
        finally:
            # Remove from processing tasks
            self._processing_tasks.pop(task.task_id, None)
    
    async def dequeue(self, worker_id: str) -> Optional[Task]:
        """
        Get next task for worker processing.
        
        Implements requirement 7.4: FIFO ordering for equal priority tasks.
        
        Args:
            worker_id: Identifier of the worker requesting a task
            
        Returns:
            Next task to process, or None if queue is empty
        """
        if self._available and self.hatchet_client:
            # In Hatchet mode, workers pull tasks through Hatchet
            # This method would not typically be called
            logger.warning("dequeue() called in Hatchet mode - tasks are pushed to workers")
            return None
        
        # Use priority queue manager for proper FIFO ordering
        task = self._priority_queue_manager.dequeue()
        if task:
            task.worker_id = worker_id
            task.status = TaskStatus.IN_PROGRESS
            self._processing_tasks[task.task_id] = task
            
            # Also remove from local queue for backward compatibility
            for i, local_task in enumerate(self._local_queue):
                if local_task.task_id == task.task_id:
                    self._local_queue.pop(i)
                    break
            
            return task
        
        return None
    
    async def complete_task(self, task_id: str, result: TaskResult) -> bool:
        """
        Mark task as completed with result.
        
        Args:
            task_id: Task identifier
            result: Task processing result
            
        Returns:
            True if completed successfully, False otherwise
        """
        success = await self._complete_task(task_id, result)
        
        # Notify load balancer of task completion
        if self.load_balancer and result.success:
            task = self._processing_tasks.get(task_id)
            if task and task.worker_id:
                await self.load_balancer.task_completed(task_id, task.worker_id)
        
        return success
    
    async def _complete_task(self, task_id: str, result: TaskResult) -> bool:
        """Internal method to complete task."""
        try:
            # Remove from processing tasks
            task = self._processing_tasks.pop(task_id, None)
            if task:
                task.status = TaskStatus.COMPLETED
            
            logger.info(f"Task {task_id} completed successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to complete task {task_id}: {e}")
            return False
    
    async def retry_task(self, task_id: str, delay: int) -> bool:
        """
        Schedule task retry with delay.
        
        Args:
            task_id: Task identifier
            delay: Delay in seconds before retry
            
        Returns:
            True if retry scheduled, False otherwise
        """
        try:
            task = self._processing_tasks.get(task_id)
            if not task:
                logger.error(f"Task {task_id} not found for retry")
                return False
            
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            
            # Schedule retry after delay
            await asyncio.sleep(delay)
            
            if self._available and self.hatchet_client:
                await self._enqueue_with_hatchet(task)
            else:
                await self._enqueue_fallback(task)
            
            logger.info(f"Task {task_id} scheduled for retry (attempt {task.retry_count})")
            return True
        except Exception as e:
            logger.error(f"Failed to retry task {task_id}: {e}")
            return False
    
    async def dead_letter(self, task_id: str, reason: str) -> bool:
        """
        Move task to dead letter queue.
        
        Args:
            task_id: Task identifier
            reason: Reason for dead lettering
            
        Returns:
            True if moved to dead letter queue, False otherwise
        """
        success = await self._dead_letter_task(task_id, reason)
        
        # Notify load balancer of task failure
        if self.load_balancer:
            task = self._processing_tasks.get(task_id)
            if task and task.worker_id:
                await self.load_balancer.task_failed(task_id, task.worker_id)
        
        return success
    
    async def _dead_letter_task(self, task_id: str, reason: str) -> bool:
        """Internal method to move task to dead letter queue."""
        try:
            task = self._processing_tasks.pop(task_id, None)
            if task:
                task.status = TaskStatus.DEAD_LETTER
            
            logger.error(f"Task {task_id} moved to dead letter queue: {reason}")
            # In a real implementation, this would store the task in a dead letter queue
            # for manual investigation and potential reprocessing
            return True
        except Exception as e:
            logger.error(f"Failed to dead letter task {task_id}: {e}")
            return False
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics for monitoring.
        
        Returns:
            Dictionary with queue statistics
        """
        base_stats = {}
        
        if self._available and self.hatchet_client:
            # In Hatchet mode, stats would come from Hatchet
            base_stats = {
                "mode": "hatchet",
                "available": True,
                "pending_tasks": "unknown",  # Would query Hatchet
                "processing_tasks": "unknown"
            }
        else:
            # Fallback mode stats
            pending_count = len([t for t in self._local_queue if t.status == TaskStatus.PENDING])
            processing_count = len(self._processing_tasks)
            
            base_stats = {
                "mode": "fallback",
                "available": False,
                "pending_tasks": pending_count,
                "processing_tasks": processing_count,
                "total_local_tasks": len(self._local_queue)
            }
        
        # Add priority queue manager stats
        priority_stats = self._priority_queue_manager.get_queue_stats()
        base_stats.update({
            "priority_queue_total": priority_stats.total_tasks,
            "tasks_by_priority": priority_stats.tasks_by_priority,
            "oldest_task_age_seconds": priority_stats.oldest_task_age_seconds,
            "average_wait_time_seconds": priority_stats.average_wait_time_seconds,
            "fifo_violations": priority_stats.fifo_violations
        })
        
        # Add load balancer stats if available
        if self.load_balancer:
            base_stats["load_balancer"] = self.load_balancer.get_load_balancer_stats()
            base_stats["worker_distribution"] = self.load_balancer.get_worker_distribution()
        
        return base_stats
    
    async def verify_fifo_ordering(self) -> Dict[str, Any]:
        """
        Verify FIFO ordering within priority levels.
        
        Returns:
            Dictionary with verification results
        """
        is_valid, violations = self._priority_queue_manager.verify_fifo_ordering()
        
        return {
            "is_valid": is_valid,
            "violations": violations,
            "total_violations": len(violations)
        }
    
    async def get_tasks_by_priority(self) -> Dict[int, List[Task]]:
        """
        Get all tasks organized by priority level.
        
        Returns:
            Dictionary mapping priority to list of tasks
        """
        return self._priority_queue_manager.get_tasks_by_priority()
    
    async def reorder_tasks(self, task_ids: List[str], new_priority: int) -> bool:
        """
        Change priority of multiple tasks while maintaining FIFO order.
        
        Args:
            task_ids: List of task IDs to reorder
            new_priority: New priority level for the tasks
            
        Returns:
            True if reordering successful, False otherwise
        """
        return self._priority_queue_manager.reorder_tasks(task_ids, new_priority)
    
    async def is_available(self) -> bool:
        """
        Check if Hatchet is available.
        
        Returns:
            True if Hatchet is available, False if in fallback mode
        """
        if not self.hatchet_client:
            return False
        
        try:
            # In a real implementation, this would ping Hatchet
            # For now, we assume it's available if client exists
            self._available = True
            return True
        except Exception as e:
            logger.error(f"Hatchet health check failed: {e}")
            self._available = False
            return False