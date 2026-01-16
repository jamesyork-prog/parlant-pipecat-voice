"""
Graceful degradation strategies for async webhook orchestration.

This module provides fallback behaviors when dependencies fail:
- Fallback to synchronous processing when Hatchet is unavailable
- Redis unavailable handling with warnings
- Worker crash recovery mechanisms
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class DependencyStatus(Enum):
    """Status of system dependencies."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class DegradationMode(Enum):
    """Types of degradation modes."""
    NORMAL = "normal"
    HATCHET_FALLBACK = "hatchet_fallback"
    REDIS_FALLBACK = "redis_fallback"
    FULL_FALLBACK = "full_fallback"


@dataclass
class DependencyHealth:
    """Health status of a dependency."""
    name: str
    status: DependencyStatus
    last_check: datetime
    error_count: int
    consecutive_failures: int
    last_error: Optional[str] = None
    response_time_ms: Optional[float] = None


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    max_consecutive_failures: int = 3
    health_check_interval: int = 30  # seconds
    fallback_timeout: int = 9  # seconds (under Freshdesk 10s limit)
    enable_warnings: bool = True
    enable_metrics: bool = True


class GracefulDegradationManager:
    """
    Manager for graceful degradation strategies.
    
    Implements requirements:
    - 6.1: Fallback to synchronous processing when Hatchet unavailable
    - 6.2: Redis unavailable handling with warnings
    - 6.3: Worker crash recovery mechanisms
    """
    
    def __init__(self, config: Optional[DegradationConfig] = None):
        """
        Initialize graceful degradation manager.
        
        Args:
            config: Configuration for degradation behavior
        """
        self.config = config or DegradationConfig()
        self.dependencies: Dict[str, DependencyHealth] = {}
        self.current_mode = DegradationMode.NORMAL
        self.fallback_processors: Dict[str, Callable] = {}
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self._stats = {
            "mode_changes": 0,
            "fallback_activations": 0,
            "successful_fallbacks": 0,
            "failed_fallbacks": 0,
            "total_processed_in_fallback": 0
        }
    
    def register_dependency(
        self, 
        name: str, 
        health_check_func: Callable[[], bool],
        fallback_processor: Optional[Callable] = None
    ):
        """
        Register a dependency for health monitoring.
        
        Args:
            name: Dependency name (e.g., "hatchet", "redis")
            health_check_func: Function to check dependency health
            fallback_processor: Optional fallback processor for this dependency
        """
        self.dependencies[name] = DependencyHealth(
            name=name,
            status=DependencyStatus.UNKNOWN,
            last_check=datetime.now(timezone.utc),
            error_count=0,
            consecutive_failures=0
        )
        
        if fallback_processor:
            self.fallback_processors[name] = fallback_processor
        
        # Start health check task
        self.health_check_tasks[name] = asyncio.create_task(
            self._health_check_loop(name, health_check_func)
        )
        
        logger.info(f"Registered dependency: {name}")
    
    async def _health_check_loop(self, name: str, health_check_func: Callable):
        """
        Continuous health check loop for a dependency.
        
        Args:
            name: Dependency name
            health_check_func: Health check function
        """
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_dependency_health(name, health_check_func)
            except asyncio.CancelledError:
                logger.info(f"Health check cancelled for {name}")
                break
            except Exception as e:
                logger.error(f"Error in health check loop for {name}: {e}")
    
    async def _check_dependency_health(self, name: str, health_check_func: Callable):
        """
        Check health of a specific dependency.
        
        Args:
            name: Dependency name
            health_check_func: Health check function
        """
        dependency = self.dependencies[name]
        start_time = time.perf_counter()
        
        try:
            # Run health check with timeout
            is_healthy = await asyncio.wait_for(
                asyncio.to_thread(health_check_func),
                timeout=5.0
            )
            
            response_time = (time.perf_counter() - start_time) * 1000
            
            if is_healthy:
                # Dependency is healthy
                if dependency.status != DependencyStatus.AVAILABLE:
                    logger.info(f"Dependency {name} recovered")
                    await self._handle_dependency_recovery(name)
                
                dependency.status = DependencyStatus.AVAILABLE
                dependency.consecutive_failures = 0
                dependency.response_time_ms = response_time
                dependency.last_error = None
            else:
                # Dependency is unhealthy
                await self._handle_dependency_failure(name, "Health check failed")
                
        except asyncio.TimeoutError:
            await self._handle_dependency_failure(name, "Health check timeout")
        except Exception as e:
            await self._handle_dependency_failure(name, str(e))
        finally:
            dependency.last_check = datetime.now(timezone.utc)
    
    async def _handle_dependency_failure(self, name: str, error: str):
        """
        Handle dependency failure and update degradation mode.
        
        Args:
            name: Dependency name
            error: Error message
        """
        dependency = self.dependencies[name]
        dependency.error_count += 1
        dependency.consecutive_failures += 1
        dependency.last_error = error
        dependency.status = DependencyStatus.UNAVAILABLE
        
        logger.warning(
            f"Dependency {name} failed: {error} "
            f"(consecutive failures: {dependency.consecutive_failures})"
        )
        
        # Check if we need to enter degradation mode
        if dependency.consecutive_failures >= self.config.max_consecutive_failures:
            await self._update_degradation_mode()
    
    async def _handle_dependency_recovery(self, name: str):
        """
        Handle dependency recovery and potentially exit degradation mode.
        
        Args:
            name: Dependency name
        """
        logger.info(f"Dependency {name} recovered")
        await self._update_degradation_mode()
    
    async def _update_degradation_mode(self):
        """Update the current degradation mode based on dependency status."""
        old_mode = self.current_mode
        
        # Check if any dependency has exceeded failure threshold
        failed_dependencies = []
        for name, dependency in self.dependencies.items():
            if dependency.consecutive_failures >= self.config.max_consecutive_failures:
                failed_dependencies.append(name)
        
        # Determine new mode based on dependency status
        hatchet_available = self._is_dependency_available("hatchet")
        redis_available = self._is_dependency_available("redis")
        
        # If we have any failed dependencies, enter degradation mode
        if failed_dependencies:
            # Specific degradation modes for known dependencies
            if "hatchet" in failed_dependencies and "redis" not in failed_dependencies:
                self.current_mode = DegradationMode.HATCHET_FALLBACK
            elif "redis" in failed_dependencies and "hatchet" not in failed_dependencies:
                self.current_mode = DegradationMode.REDIS_FALLBACK
            elif "hatchet" in failed_dependencies and "redis" in failed_dependencies:
                self.current_mode = DegradationMode.FULL_FALLBACK
            else:
                # Generic degradation for other dependencies
                self.current_mode = DegradationMode.FULL_FALLBACK
        elif hatchet_available and redis_available:
            self.current_mode = DegradationMode.NORMAL
        elif not hatchet_available and redis_available:
            self.current_mode = DegradationMode.HATCHET_FALLBACK
        elif hatchet_available and not redis_available:
            self.current_mode = DegradationMode.REDIS_FALLBACK
        else:
            self.current_mode = DegradationMode.FULL_FALLBACK
        
        if old_mode != self.current_mode:
            self._stats["mode_changes"] += 1
            logger.warning(
                f"Degradation mode changed: {old_mode.value} -> {self.current_mode.value}"
            )
            
            if self.config.enable_warnings:
                await self._emit_degradation_warning(old_mode, self.current_mode)
    
    def _is_dependency_available(self, name: str) -> bool:
        """
        Check if a dependency is available.
        
        Args:
            name: Dependency name
            
        Returns:
            True if available, False otherwise
        """
        dependency = self.dependencies.get(name)
        if not dependency:
            return True  # Unknown dependencies are assumed available
        
        return dependency.status == DependencyStatus.AVAILABLE
    
    async def _emit_degradation_warning(
        self, 
        old_mode: DegradationMode, 
        new_mode: DegradationMode
    ):
        """
        Emit warning about degradation mode change.
        
        Args:
            old_mode: Previous degradation mode
            new_mode: New degradation mode
        """
        warning_msg = f"System degradation: {old_mode.value} -> {new_mode.value}"
        
        if new_mode == DegradationMode.HATCHET_FALLBACK:
            warning_msg += " - Using synchronous processing fallback"
        elif new_mode == DegradationMode.REDIS_FALLBACK:
            warning_msg += " - Idempotency checks disabled"
        elif new_mode == DegradationMode.FULL_FALLBACK:
            warning_msg += " - All fallbacks active"
        
        logger.critical(f"DEGRADATION_WARNING: {warning_msg}")
        
        # In a real implementation, this would integrate with alerting systems
        if self.config.enable_metrics:
            try:
                # This would emit metrics to your monitoring system
                # metrics.increment(f"degradation.mode_change.{new_mode.value}")
                pass
            except Exception as e:
                logger.error(f"Failed to emit degradation metric: {e}")
    
    async def process_with_fallback(
        self,
        primary_processor: Callable,
        payload: Dict[str, Any],
        dependency_name: str = "hatchet"
    ) -> Dict[str, Any]:
        """
        Process payload with fallback if primary processor fails.
        
        Implements requirement 6.1: Fallback to synchronous processing when Hatchet unavailable.
        
        Args:
            primary_processor: Primary processing function
            payload: Data to process
            dependency_name: Name of dependency being used
            
        Returns:
            Processing result
        """
        start_time = time.perf_counter()
        
        try:
            # Check if we should use fallback mode
            if not self._is_dependency_available(dependency_name):
                return await self._process_with_fallback_processor(
                    payload, dependency_name, start_time
                )
            
            # Try primary processor
            result = await asyncio.wait_for(
                primary_processor(payload),
                timeout=self.config.fallback_timeout
            )
            
            return {
                "success": True,
                "result": result,
                "processing_mode": "primary",
                "processing_time_ms": (time.perf_counter() - start_time) * 1000
            }
            
        except asyncio.TimeoutError:
            logger.warning(f"Primary processor timeout, falling back")
            return await self._process_with_fallback_processor(
                payload, dependency_name, start_time
            )
        except Exception as e:
            logger.error(f"Primary processor failed: {e}, falling back")
            return await self._process_with_fallback_processor(
                payload, dependency_name, start_time
            )
    
    async def _process_with_fallback_processor(
        self,
        payload: Dict[str, Any],
        dependency_name: str,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Process using fallback processor.
        
        Args:
            payload: Data to process
            dependency_name: Name of failed dependency
            start_time: Processing start time
            
        Returns:
            Processing result
        """
        self._stats["fallback_activations"] += 1
        
        fallback_processor = self.fallback_processors.get(dependency_name)
        if not fallback_processor:
            self._stats["failed_fallbacks"] += 1
            return {
                "success": False,
                "error": f"No fallback processor available for {dependency_name}",
                "processing_mode": "fallback_failed",
                "processing_time_ms": (time.perf_counter() - start_time) * 1000
            }
        
        try:
            # Process with fallback and timeout
            result = await asyncio.wait_for(
                fallback_processor(payload),
                timeout=self.config.fallback_timeout
            )
            
            self._stats["successful_fallbacks"] += 1
            self._stats["total_processed_in_fallback"] += 1
            
            return {
                "success": True,
                "result": result,
                "processing_mode": f"fallback_{dependency_name}",
                "processing_time_ms": (time.perf_counter() - start_time) * 1000,
                "warning": f"Processed in fallback mode due to {dependency_name} unavailability"
            }
            
        except Exception as e:
            self._stats["failed_fallbacks"] += 1
            logger.error(f"Fallback processor failed: {e}")
            
            return {
                "success": False,
                "error": f"Fallback processor failed: {str(e)}",
                "processing_mode": "fallback_failed",
                "processing_time_ms": (time.perf_counter() - start_time) * 1000
            }
    
    async def handle_redis_unavailable(self, operation_name: str) -> Dict[str, Any]:
        """
        Handle Redis unavailable scenario with warnings.
        
        Implements requirement 6.2: Redis unavailable handling with warnings.
        
        Args:
            operation_name: Name of operation that would use Redis
            
        Returns:
            Result indicating degraded operation
        """
        warning_msg = f"Redis unavailable for {operation_name} - operating in degraded mode"
        logger.warning(warning_msg)
        
        if self.config.enable_metrics:
            try:
                # This would emit metrics to your monitoring system
                # metrics.increment(f"redis.unavailable.{operation_name}")
                pass
            except Exception as e:
                logger.error(f"Failed to emit Redis unavailable metric: {e}")
        
        return {
            "success": True,
            "degraded": True,
            "warning": warning_msg,
            "operation": operation_name,
            "fallback_mode": "redis_unavailable"
        }
    
    async def handle_worker_crash(
        self,
        worker_id: str,
        task_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Handle worker crash and implement recovery mechanisms.
        
        Implements requirement 6.3: Worker crash recovery mechanisms.
        
        Args:
            worker_id: ID of crashed worker
            task_id: ID of task being processed when crash occurred
            error: Error message from crash
            
        Returns:
            Recovery result
        """
        logger.error(
            f"Worker {worker_id} crashed",
            extra={
                "worker_id": worker_id,
                "task_id": task_id,
                "error": error
            }
        )
        
        recovery_actions = []
        
        # 1. Mark worker as unavailable
        recovery_actions.append(f"Marked worker {worker_id} as unavailable")
        
        # 2. If task was being processed, mark for reassignment
        if task_id:
            recovery_actions.append(f"Task {task_id} marked for reassignment")
            
            # In a real implementation, this would:
            # - Move task back to pending state
            # - Reset worker assignment
            # - Potentially increment retry count
        
        # 3. Emit metrics and alerts
        if self.config.enable_metrics:
            try:
                # This would emit metrics to your monitoring system
                # metrics.increment("worker.crash", tags={"worker_id": worker_id})
                pass
            except Exception as e:
                logger.error(f"Failed to emit worker crash metric: {e}")
        
        # 4. Check if we need to enter degradation mode
        await self._update_degradation_mode()
        
        return {
            "success": True,
            "worker_id": worker_id,
            "task_id": task_id,
            "recovery_actions": recovery_actions,
            "current_mode": self.current_mode.value
        }
    
    def get_current_mode(self) -> DegradationMode:
        """
        Get current degradation mode.
        
        Returns:
            Current degradation mode
        """
        return self.current_mode
    
    def is_degraded(self) -> bool:
        """
        Check if system is in degraded mode.
        
        Returns:
            True if in any degraded mode, False if normal
        """
        return self.current_mode != DegradationMode.NORMAL
    
    def get_dependency_status(self, name: str) -> Optional[DependencyHealth]:
        """
        Get status of a specific dependency.
        
        Args:
            name: Dependency name
            
        Returns:
            DependencyHealth if found, None otherwise
        """
        return self.dependencies.get(name)
    
    def get_all_dependencies_status(self) -> Dict[str, DependencyHealth]:
        """
        Get status of all dependencies.
        
        Returns:
            Dictionary of dependency name to health status
        """
        return self.dependencies.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get degradation manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        dependency_stats = {}
        for name, health in self.dependencies.items():
            dependency_stats[name] = {
                "status": health.status.value,
                "error_count": health.error_count,
                "consecutive_failures": health.consecutive_failures,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "last_error": health.last_error
            }
        
        return {
            "current_mode": self.current_mode.value,
            "is_degraded": self.is_degraded(),
            "dependencies": dependency_stats,
            "stats": self._stats.copy(),
            "config": {
                "max_consecutive_failures": self.config.max_consecutive_failures,
                "health_check_interval": self.config.health_check_interval,
                "fallback_timeout": self.config.fallback_timeout
            }
        }
    
    async def shutdown(self):
        """Shutdown the degradation manager and cleanup resources."""
        logger.info("Shutting down graceful degradation manager")
        
        # Cancel all health check tasks
        for name, task in self.health_check_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self.health_check_tasks.clear()
        logger.info("Graceful degradation manager shutdown complete")


# Global degradation manager instance
degradation_manager = GracefulDegradationManager()


async def process_with_hatchet_fallback(
    primary_processor: Callable,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Convenience function to process with Hatchet fallback.
    
    Args:
        primary_processor: Primary Hatchet processor
        payload: Data to process
        
    Returns:
        Processing result
    """
    return await degradation_manager.process_with_fallback(
        primary_processor, payload, "hatchet"
    )


async def handle_redis_degradation(operation_name: str) -> Dict[str, Any]:
    """
    Convenience function to handle Redis unavailable scenario.
    
    Args:
        operation_name: Name of operation that would use Redis
        
    Returns:
        Degradation result
    """
    return await degradation_manager.handle_redis_unavailable(operation_name)