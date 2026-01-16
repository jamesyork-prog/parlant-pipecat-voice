"""
Priority queue manager for FIFO processing with priority support.

This module provides the PriorityQueueManager class that implements:
- Priority-based task processing with FIFO for equal priority
- Queue ordering verification and monitoring
- Future priority support architecture
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import deque

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
class QueueStats:
    """Statistics for queue monitoring."""
    total_tasks: int = 0
    tasks_by_priority: Dict[int, int] = field(default_factory=dict)
    oldest_task_age_seconds: float = 0.0
    average_wait_time_seconds: float = 0.0
    fifo_violations: int = 0


class PriorityQueueManager:
    """
    Priority queue manager for FIFO processing with priority support.
    
    Implements requirements:
    - 7.4: FIFO ordering for tasks of equal priority
    - Priority-based task processing
    - Queue ordering verification and monitoring
    - Future priority support architecture
    """
    
    def __init__(self):
        """Initialize the priority queue manager."""
        # Use separate deques for each priority level to maintain FIFO order
        self._priority_queues: Dict[int, deque] = {}
        self._task_metadata: Dict[str, Dict[str, Any]] = {}  # task_id -> metadata
        self._enqueue_times: Dict[str, datetime] = {}  # task_id -> enqueue_time
        self._stats = QueueStats()
        
        logger.info("PriorityQueueManager initialized")
    
    def enqueue(self, task) -> bool:
        """
        Add task to the appropriate priority queue.
        
        Args:
            task: Task to enqueue
            
        Returns:
            True if enqueued successfully, False otherwise
        """
        try:
            priority = task.priority
            
            # Create priority queue if it doesn't exist
            if priority not in self._priority_queues:
                self._priority_queues[priority] = deque()
            
            # Add task to the end of the priority queue (FIFO)
            self._priority_queues[priority].append(task)
            
            # Store metadata
            self._task_metadata[task.task_id] = {
                "priority": priority,
                "enqueue_time": datetime.now(timezone.utc),
                "position_in_queue": len(self._priority_queues[priority]) - 1
            }
            self._enqueue_times[task.task_id] = datetime.now(timezone.utc)
            
            # Update stats
            self._update_stats()
            
            logger.debug(f"Task {task.task_id} enqueued with priority {priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False
    
    def dequeue(self):
        """
        Get the next task following priority and FIFO rules.
        
        Returns:
            Next task to process, or None if queue is empty
        """
        try:
            # Find the highest priority that has tasks
            if not self._priority_queues:
                return None
            
            # Get priorities in descending order (highest first)
            priorities = sorted(self._priority_queues.keys(), reverse=True)
            
            for priority in priorities:
                queue = self._priority_queues[priority]
                if queue:
                    # Get the first task (FIFO within priority)
                    task = queue.popleft()
                    
                    # Clean up empty queues
                    if not queue:
                        del self._priority_queues[priority]
                    
                    # Update metadata
                    if task.task_id in self._task_metadata:
                        del self._task_metadata[task.task_id]
                    if task.task_id in self._enqueue_times:
                        del self._enqueue_times[task.task_id]
                    
                    # Update stats
                    self._update_stats()
                    
                    logger.debug(f"Task {task.task_id} dequeued with priority {priority}")
                    return task
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to dequeue task: {e}")
            return None
    
    def peek_next(self):
        """
        Peek at the next task without removing it.
        
        Returns:
            Next task that would be dequeued, or None if queue is empty
        """
        try:
            if not self._priority_queues:
                return None
            
            # Find the highest priority that has tasks
            priorities = sorted(self._priority_queues.keys(), reverse=True)
            
            for priority in priorities:
                queue = self._priority_queues[priority]
                if queue:
                    return queue[0]  # First task in FIFO order
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to peek next task: {e}")
            return None
    
    def remove_task(self, task_id: str) -> bool:
        """
        Remove a specific task from the queue.
        
        Args:
            task_id: ID of task to remove
            
        Returns:
            True if task was found and removed, False otherwise
        """
        try:
            # Find the task in the appropriate priority queue
            task_meta = self._task_metadata.get(task_id)
            if not task_meta:
                logger.warning(f"Task {task_id} not found in queue")
                return False
            
            priority = task_meta["priority"]
            if priority not in self._priority_queues:
                logger.warning(f"Priority queue {priority} not found for task {task_id}")
                return False
            
            queue = self._priority_queues[priority]
            
            # Find and remove the task
            for i, task in enumerate(queue):
                if task.task_id == task_id:
                    del queue[i]
                    
                    # Clean up empty queues
                    if not queue:
                        del self._priority_queues[priority]
                    
                    # Clean up metadata
                    del self._task_metadata[task_id]
                    if task_id in self._enqueue_times:
                        del self._enqueue_times[task_id]
                    
                    # Update stats
                    self._update_stats()
                    
                    logger.debug(f"Task {task_id} removed from priority {priority} queue")
                    return True
            
            logger.warning(f"Task {task_id} not found in priority {priority} queue")
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove task {task_id}: {e}")
            return False
    
    def get_queue_size(self, priority: Optional[int] = None) -> int:
        """
        Get the size of the queue(s).
        
        Args:
            priority: Specific priority level, or None for total size
            
        Returns:
            Number of tasks in the specified queue(s)
        """
        try:
            if priority is not None:
                return len(self._priority_queues.get(priority, deque()))
            else:
                return sum(len(queue) for queue in self._priority_queues.values())
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0
    
    def get_tasks_by_priority(self) -> Dict[int, List]:
        """
        Get all tasks organized by priority.
        
        Returns:
            Dictionary mapping priority to list of tasks
        """
        try:
            result = {}
            for priority, queue in self._priority_queues.items():
                result[priority] = list(queue)
            return result
        except Exception as e:
            logger.error(f"Failed to get tasks by priority: {e}")
            return {}
    
    def verify_fifo_ordering(self) -> Tuple[bool, List[str]]:
        """
        Verify that tasks within each priority level are in FIFO order.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        try:
            for priority, queue in self._priority_queues.items():
                if len(queue) <= 1:
                    continue  # Single task or empty queue is always valid
                
                # Check that enqueue times are in ascending order (FIFO)
                prev_time = None
                for i, task in enumerate(queue):
                    enqueue_time = self._enqueue_times.get(task.task_id)
                    if enqueue_time is None:
                        violations.append(
                            f"Priority {priority}, position {i}: Task {task.task_id} missing enqueue time"
                        )
                        continue
                    
                    if prev_time is not None and enqueue_time < prev_time:
                        violations.append(
                            f"Priority {priority}, position {i}: Task {task.task_id} "
                            f"enqueued at {enqueue_time} but previous task at {prev_time} (FIFO violation)"
                        )
                    
                    prev_time = enqueue_time
            
            is_valid = len(violations) == 0
            if not is_valid:
                self._stats.fifo_violations += len(violations)
                logger.warning(f"FIFO ordering violations detected: {violations}")
            
            return is_valid, violations
            
        except Exception as e:
            logger.error(f"Failed to verify FIFO ordering: {e}")
            return False, [f"Verification error: {str(e)}"]
    
    def get_queue_stats(self) -> QueueStats:
        """
        Get queue statistics for monitoring.
        
        Returns:
            QueueStats object with current statistics
        """
        self._update_stats()
        return self._stats
    
    def _update_stats(self):
        """Update internal statistics."""
        try:
            # Total tasks
            self._stats.total_tasks = sum(len(queue) for queue in self._priority_queues.values())
            
            # Tasks by priority
            self._stats.tasks_by_priority = {
                priority: len(queue) for priority, queue in self._priority_queues.items()
            }
            
            # Oldest task age
            if self._enqueue_times:
                current_time = datetime.now(timezone.utc)
                oldest_time = min(self._enqueue_times.values())
                self._stats.oldest_task_age_seconds = (current_time - oldest_time).total_seconds()
            else:
                self._stats.oldest_task_age_seconds = 0.0
            
            # Average wait time
            if self._enqueue_times:
                current_time = datetime.now(timezone.utc)
                wait_times = [
                    (current_time - enqueue_time).total_seconds()
                    for enqueue_time in self._enqueue_times.values()
                ]
                self._stats.average_wait_time_seconds = sum(wait_times) / len(wait_times)
            else:
                self._stats.average_wait_time_seconds = 0.0
                
        except Exception as e:
            logger.error(f"Failed to update stats: {e}")
    
    def clear(self):
        """Clear all queues and reset state."""
        try:
            self._priority_queues.clear()
            self._task_metadata.clear()
            self._enqueue_times.clear()
            self._stats = QueueStats()
            logger.info("Priority queue manager cleared")
        except Exception as e:
            logger.error(f"Failed to clear queue manager: {e}")
    
    def is_empty(self) -> bool:
        """
        Check if all queues are empty.
        
        Returns:
            True if no tasks in any queue, False otherwise
        """
        return self.get_queue_size() == 0
    
    def get_priority_levels(self) -> List[int]:
        """
        Get all active priority levels.
        
        Returns:
            List of priority levels that have tasks
        """
        return sorted(self._priority_queues.keys(), reverse=True)
    
    def reorder_tasks(self, task_ids: List[str], new_priority: int) -> bool:
        """
        Change priority of multiple tasks while maintaining FIFO order.
        
        Args:
            task_ids: List of task IDs to reorder
            new_priority: New priority level for the tasks
            
        Returns:
            True if reordering successful, False otherwise
        """
        try:
            # Find and remove tasks from their current queues
            tasks_to_reorder = []
            
            for task_id in task_ids:
                task_meta = self._task_metadata.get(task_id)
                if not task_meta:
                    logger.warning(f"Task {task_id} not found for reordering")
                    continue
                
                old_priority = task_meta["priority"]
                if old_priority not in self._priority_queues:
                    continue
                
                queue = self._priority_queues[old_priority]
                
                # Find the task in the queue
                for i, task in enumerate(queue):
                    if task.task_id == task_id:
                        # Remove from old queue
                        removed_task = queue[i]
                        del queue[i]
                        
                        # Update task priority
                        removed_task.priority = new_priority
                        tasks_to_reorder.append(removed_task)
                        
                        # Clean up empty queues
                        if not queue:
                            del self._priority_queues[old_priority]
                        
                        break
            
            # Add tasks to new priority queue in the same relative order
            if new_priority not in self._priority_queues:
                self._priority_queues[new_priority] = deque()
            
            new_queue = self._priority_queues[new_priority]
            for task in tasks_to_reorder:
                new_queue.append(task)
                
                # Update metadata
                self._task_metadata[task.task_id]["priority"] = new_priority
            
            # Update stats
            self._update_stats()
            
            logger.info(f"Reordered {len(tasks_to_reorder)} tasks to priority {new_priority}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reorder tasks: {e}")
            return False