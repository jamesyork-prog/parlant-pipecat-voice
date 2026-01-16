"""
Dead letter queue handling for async webhook orchestration.

This module provides dead letter queue functionality for:
- Storing unrecoverable failures requiring manual intervention
- Classifying permanent vs transient failure types
- Monitoring and alerting for dead letter queue items
- Recovery and reprocessing capabilities
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum

from .retry_strategy import FailureType, classify_error_for_retry

logger = logging.getLogger(__name__)


class DeadLetterReason(Enum):
    """Reasons for moving tasks to dead letter queue."""
    MAX_RETRIES_EXCEEDED = "max_retries_exceeded"
    PERMANENT_FAILURE = "permanent_failure"
    PROCESSING_TIMEOUT = "processing_timeout"
    WORKER_CRASH = "worker_crash"
    SYSTEM_ERROR = "system_error"
    MANUAL_INTERVENTION = "manual_intervention"


@dataclass
class DeadLetterItem:
    """Item stored in dead letter queue."""
    task_id: str
    original_payload: Dict[str, Any]
    failure_reason: DeadLetterReason
    error_message: str
    failure_type: FailureType
    retry_count: int
    max_retries: int
    created_at: datetime
    dead_lettered_at: datetime
    worker_id: Optional[str] = None
    processing_time: Optional[float] = None
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    last_recovery_attempt: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['dead_lettered_at'] = self.dead_lettered_at.isoformat()
        data['failure_reason'] = self.failure_reason.value
        data['failure_type'] = self.failure_type.value
        if self.last_recovery_attempt:
            data['last_recovery_attempt'] = self.last_recovery_attempt.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeadLetterItem':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['dead_lettered_at'] = datetime.fromisoformat(data['dead_lettered_at'])
        data['failure_reason'] = DeadLetterReason(data['failure_reason'])
        data['failure_type'] = FailureType(data['failure_type'])
        if data.get('last_recovery_attempt'):
            data['last_recovery_attempt'] = datetime.fromisoformat(data['last_recovery_attempt'])
        return cls(**data)


class DeadLetterQueue:
    """
    Dead letter queue for unrecoverable task failures.
    
    Implements requirements:
    - 4.4: Dead letter queue for transient failures exceeding max retries
    - 4.5: Immediate dead lettering for permanent failures
    - Monitoring and alerting for dead letter queue items
    """
    
    def __init__(self, storage_backend=None, alert_threshold: int = 10):
        """
        Initialize dead letter queue.
        
        Args:
            storage_backend: Storage backend for persistence (Redis, DB, etc.)
            alert_threshold: Number of items that triggers alerting
        """
        self.storage_backend = storage_backend
        self.alert_threshold = alert_threshold
        self._items: Dict[str, DeadLetterItem] = {}  # In-memory fallback
        self._stats = {
            "total_items": 0,
            "items_by_reason": {},
            "items_by_type": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
    
    async def add_item(
        self,
        task_id: str,
        original_payload: Dict[str, Any],
        error: Exception,
        retry_count: int,
        max_retries: int,
        worker_id: Optional[str] = None,
        processing_time: Optional[float] = None,
        stack_trace: Optional[str] = None
    ) -> DeadLetterItem:
        """
        Add item to dead letter queue.
        
        Args:
            task_id: Unique task identifier
            original_payload: Original task payload
            error: Exception that caused the failure
            retry_count: Number of retries attempted
            max_retries: Maximum retries allowed
            worker_id: ID of worker that processed the task
            processing_time: Time spent processing before failure
            stack_trace: Full stack trace of the error
            
        Returns:
            DeadLetterItem that was created
        """
        # Classify the failure
        failure_type = classify_error_for_retry(error)
        
        # Determine reason for dead lettering
        if failure_type == FailureType.PERMANENT_FAILURE:
            reason = DeadLetterReason.PERMANENT_FAILURE
        elif retry_count >= max_retries:
            reason = DeadLetterReason.MAX_RETRIES_EXCEEDED
        elif "timeout" in str(error).lower():
            reason = DeadLetterReason.PROCESSING_TIMEOUT
        elif "worker" in str(error).lower() and "crash" in str(error).lower():
            reason = DeadLetterReason.WORKER_CRASH
        else:
            reason = DeadLetterReason.SYSTEM_ERROR
        
        # Create dead letter item
        item = DeadLetterItem(
            task_id=task_id,
            original_payload=original_payload,
            failure_reason=reason,
            error_message=str(error),
            failure_type=failure_type,
            retry_count=retry_count,
            max_retries=max_retries,
            created_at=original_payload.get('created_at', datetime.now(timezone.utc)),
            dead_lettered_at=datetime.now(timezone.utc),
            worker_id=worker_id,
            processing_time=processing_time,
            stack_trace=stack_trace
        )
        
        # Store the item
        await self._store_item(item)
        
        # Update statistics
        self._update_stats(item)
        
        # Check if alerting is needed
        await self._check_alert_threshold()
        
        logger.error(
            f"Task {task_id} moved to dead letter queue",
            extra={
                "task_id": task_id,
                "failure_reason": reason.value,
                "failure_type": failure_type.value,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "error_message": str(error)
            }
        )
        
        return item
    
    async def _store_item(self, item: DeadLetterItem):
        """Store item in backend storage."""
        if self.storage_backend:
            try:
                await self.storage_backend.store_dead_letter_item(item.to_dict())
            except Exception as e:
                logger.error(f"Failed to store dead letter item in backend: {e}")
                # Fall back to in-memory storage
                self._items[item.task_id] = item
        else:
            # In-memory storage
            self._items[item.task_id] = item
    
    def _update_stats(self, item: DeadLetterItem):
        """Update internal statistics."""
        self._stats["total_items"] += 1
        
        # Update by reason
        reason_key = item.failure_reason.value
        self._stats["items_by_reason"][reason_key] = (
            self._stats["items_by_reason"].get(reason_key, 0) + 1
        )
        
        # Update by type
        type_key = item.failure_type.value
        self._stats["items_by_type"][type_key] = (
            self._stats["items_by_type"].get(type_key, 0) + 1
        )
    
    async def _check_alert_threshold(self):
        """Check if alert threshold is exceeded and trigger alerts."""
        current_count = await self.get_item_count()
        
        if current_count >= self.alert_threshold:
            await self._trigger_alert(
                f"Dead letter queue has {current_count} items (threshold: {self.alert_threshold})"
            )
    
    async def _trigger_alert(self, message: str):
        """Trigger alert for dead letter queue issues."""
        logger.critical(f"DEAD_LETTER_QUEUE_ALERT: {message}")
        
        # In a real implementation, this would integrate with alerting systems
        # like PagerDuty, Slack, email, etc.
        
        # For now, we'll emit a metric that monitoring can pick up
        try:
            # This would integrate with your metrics system
            # metrics.increment("dead_letter_queue.alert_triggered")
            pass
        except Exception as e:
            logger.error(f"Failed to emit dead letter queue alert metric: {e}")
    
    async def get_items(
        self,
        limit: int = 100,
        offset: int = 0,
        failure_reason: Optional[DeadLetterReason] = None,
        failure_type: Optional[FailureType] = None
    ) -> List[DeadLetterItem]:
        """
        Get items from dead letter queue with filtering.
        
        Args:
            limit: Maximum number of items to return
            offset: Number of items to skip
            failure_reason: Filter by failure reason
            failure_type: Filter by failure type
            
        Returns:
            List of dead letter items
        """
        if self.storage_backend:
            try:
                filters = {}
                if failure_reason:
                    filters['failure_reason'] = failure_reason.value
                if failure_type:
                    filters['failure_type'] = failure_type.value
                
                items_data = await self.storage_backend.get_dead_letter_items(
                    limit=limit,
                    offset=offset,
                    filters=filters
                )
                return [DeadLetterItem.from_dict(data) for data in items_data]
            except Exception as e:
                logger.error(f"Failed to get items from backend: {e}")
                # Fall back to in-memory
        
        # In-memory filtering
        items = list(self._items.values())
        
        # Apply filters
        if failure_reason:
            items = [item for item in items if item.failure_reason == failure_reason]
        if failure_type:
            items = [item for item in items if item.failure_type == failure_type]
        
        # Sort by dead_lettered_at (newest first)
        items.sort(key=lambda x: x.dead_lettered_at, reverse=True)
        
        # Apply pagination
        return items[offset:offset + limit]
    
    async def get_item(self, task_id: str) -> Optional[DeadLetterItem]:
        """
        Get specific item from dead letter queue.
        
        Args:
            task_id: Task identifier
            
        Returns:
            DeadLetterItem if found, None otherwise
        """
        if self.storage_backend:
            try:
                item_data = await self.storage_backend.get_dead_letter_item(task_id)
                if item_data:
                    return DeadLetterItem.from_dict(item_data)
            except Exception as e:
                logger.error(f"Failed to get item from backend: {e}")
        
        return self._items.get(task_id)
    
    async def remove_item(self, task_id: str) -> bool:
        """
        Remove item from dead letter queue.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if item was removed, False if not found
        """
        if self.storage_backend:
            try:
                success = await self.storage_backend.remove_dead_letter_item(task_id)
                if success:
                    logger.info(f"Removed dead letter item {task_id}")
                    return True
            except Exception as e:
                logger.error(f"Failed to remove item from backend: {e}")
        
        # In-memory removal
        if task_id in self._items:
            del self._items[task_id]
            logger.info(f"Removed dead letter item {task_id} from memory")
            return True
        
        return False
    
    async def retry_item(self, task_id: str, task_queue) -> bool:
        """
        Retry a dead letter item by re-enqueueing it.
        
        Args:
            task_id: Task identifier
            task_queue: TaskQueue instance to re-enqueue to
            
        Returns:
            True if successfully re-enqueued, False otherwise
        """
        item = await self.get_item(task_id)
        if not item:
            logger.error(f"Dead letter item {task_id} not found for retry")
            return False
        
        try:
            # Update recovery attempt tracking
            item.recovery_attempts += 1
            item.last_recovery_attempt = datetime.now(timezone.utc)
            await self._store_item(item)
            
            # Re-enqueue the original payload
            new_task_id = await task_queue.enqueue(item.original_payload)
            
            # Remove from dead letter queue
            await self.remove_item(task_id)
            
            self._stats["recovery_attempts"] += 1
            self._stats["successful_recoveries"] += 1
            
            logger.info(
                f"Successfully retried dead letter item {task_id} as new task {new_task_id}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to retry dead letter item {task_id}: {e}")
            return False
    
    async def get_item_count(self) -> int:
        """
        Get total number of items in dead letter queue.
        
        Returns:
            Number of items
        """
        if self.storage_backend:
            try:
                return await self.storage_backend.get_dead_letter_count()
            except Exception as e:
                logger.error(f"Failed to get count from backend: {e}")
        
        return len(self._items)
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get dead letter queue statistics.
        
        Returns:
            Dictionary with statistics
        """
        current_count = await self.get_item_count()
        
        # Get recent items for analysis
        recent_items = await self.get_items(limit=50)
        
        # Calculate time-based stats
        now = datetime.now(timezone.utc)
        last_24h = [
            item for item in recent_items
            if (now - item.dead_lettered_at).total_seconds() < 86400
        ]
        last_hour = [
            item for item in recent_items
            if (now - item.dead_lettered_at).total_seconds() < 3600
        ]
        
        stats = {
            "current_count": current_count,
            "alert_threshold": self.alert_threshold,
            "items_last_24h": len(last_24h),
            "items_last_hour": len(last_hour),
            "total_processed": self._stats["total_items"],
            "recovery_attempts": self._stats["recovery_attempts"],
            "successful_recoveries": self._stats["successful_recoveries"],
            "recovery_success_rate": (
                self._stats["successful_recoveries"] / max(1, self._stats["recovery_attempts"])
            ),
            "items_by_reason": self._stats["items_by_reason"].copy(),
            "items_by_type": self._stats["items_by_type"].copy()
        }
        
        return stats
    
    async def cleanup_old_items(self, max_age_days: int = 30) -> int:
        """
        Clean up old items from dead letter queue.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            Number of items cleaned up
        """
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        cleaned_count = 0
        
        if self.storage_backend:
            try:
                cleaned_count = await self.storage_backend.cleanup_old_dead_letter_items(cutoff_time)
            except Exception as e:
                logger.error(f"Failed to cleanup old items from backend: {e}")
        
        # In-memory cleanup
        items_to_remove = []
        for task_id, item in self._items.items():
            if item.dead_lettered_at.timestamp() < cutoff_time:
                items_to_remove.append(task_id)
        
        for task_id in items_to_remove:
            del self._items[task_id]
            cleaned_count += 1
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old dead letter items")
        
        return cleaned_count


# Global dead letter queue instance
dead_letter_queue = DeadLetterQueue()


async def add_to_dead_letter_queue(
    task_id: str,
    payload: Dict[str, Any],
    error: Exception,
    retry_count: int,
    max_retries: int,
    **kwargs
) -> DeadLetterItem:
    """
    Convenience function to add item to dead letter queue.
    
    Args:
        task_id: Task identifier
        payload: Original task payload
        error: Exception that caused failure
        retry_count: Number of retries attempted
        max_retries: Maximum retries allowed
        **kwargs: Additional metadata
        
    Returns:
        DeadLetterItem that was created
    """
    return await dead_letter_queue.add_item(
        task_id=task_id,
        original_payload=payload,
        error=error,
        retry_count=retry_count,
        max_retries=max_retries,
        **kwargs
    )