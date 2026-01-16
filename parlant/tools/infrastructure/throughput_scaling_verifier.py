"""
Throughput scaling verification for horizontal scaling support.

This module provides the ThroughputScalingVerifier class that implements:
- Linear scaling measurement and validation
- Worker capacity monitoring and auto-scaling hooks
- Performance benchmarking for scaling decisions
"""

import asyncio
import time
import logging
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum

from .worker_load_balancer import WorkerLoadBalancer
from .task_queue import TaskQueue, Task, TaskStatus, TaskPriority
from .prometheus_metrics import get_webhook_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class ThroughputMeasurement:
    """Measurement of throughput for a specific worker configuration."""
    worker_count: int
    tasks_processed: int
    duration_seconds: float
    throughput_per_minute: float
    throughput_per_worker: float
    average_processing_time: float
    p95_processing_time: float
    p99_processing_time: float
    error_rate: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ScalingTestResult:
    """Result of a scaling verification test."""
    measurements: List[ThroughputMeasurement]
    linear_scaling_coefficient: float  # R² value for linear fit
    scaling_efficiency: float  # Actual vs theoretical scaling ratio
    bottlenecks_detected: List[str]
    recommendations: List[str]
    is_scaling_linear: bool
    test_duration_seconds: float


class ScalingTestConfig:
    """Configuration for scaling tests."""
    
    def __init__(
        self,
        min_workers: int = 1,
        max_workers: int = 10,
        worker_step: int = 1,
        tasks_per_test: int = 100,
        test_duration_seconds: int = 60,
        warmup_duration_seconds: int = 10,
        target_throughput_per_worker: float = 100.0,  # tasks per minute
        linear_scaling_threshold: float = 0.8  # R² threshold for linear scaling
    ):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.worker_step = worker_step
        self.tasks_per_test = tasks_per_test
        self.test_duration_seconds = test_duration_seconds
        self.warmup_duration_seconds = warmup_duration_seconds
        self.target_throughput_per_worker = target_throughput_per_worker
        self.linear_scaling_threshold = linear_scaling_threshold


class ThroughputScalingVerifier:
    """
    Throughput scaling verifier for horizontal scaling.
    
    Implements requirements:
    - 7.2: Linear throughput scaling up to 100 tasks per minute per worker
    - Linear scaling measurement and validation
    - Worker capacity monitoring and auto-scaling hooks
    - Performance benchmarking for scaling decisions
    """
    
    def __init__(
        self,
        load_balancer: WorkerLoadBalancer,
        task_queue: TaskQueue,
        config: Optional[ScalingTestConfig] = None
    ):
        """
        Initialize throughput scaling verifier.
        
        Args:
            load_balancer: WorkerLoadBalancer instance
            task_queue: TaskQueue instance
            config: Optional scaling test configuration
        """
        self.load_balancer = load_balancer
        self.task_queue = task_queue
        self.config = config or ScalingTestConfig()
        
        # Test state
        self.test_results: List[ScalingTestResult] = []
        self.current_test_workers: List[str] = []
        
        logger.info("ThroughputScalingVerifier initialized")
    
    async def run_scaling_verification_test(
        self,
        test_name: str = "default"
    ) -> ScalingTestResult:
        """
        Run a complete scaling verification test.
        
        Args:
            test_name: Name for this test run
            
        Returns:
            ScalingTestResult with measurements and analysis
        """
        logger.info(f"Starting scaling verification test: {test_name}")
        test_start_time = time.time()
        
        measurements = []
        
        try:
            # Test different worker counts
            for worker_count in range(
                self.config.min_workers,
                self.config.max_workers + 1,
                self.config.worker_step
            ):
                logger.info(f"Testing with {worker_count} workers")
                
                # Set up workers for this test
                await self._setup_test_workers(worker_count)
                
                # Run throughput measurement
                measurement = await self._measure_throughput(worker_count)
                measurements.append(measurement)
                
                # Clean up workers
                await self._cleanup_test_workers()
                
                # Brief pause between tests
                await asyncio.sleep(2)
            
            # Analyze results
            test_result = self._analyze_scaling_results(measurements)
            test_result.test_duration_seconds = time.time() - test_start_time
            
            # Store result
            self.test_results.append(test_result)
            
            # Record metrics
            metrics_collector = get_webhook_metrics_collector()
            self._record_scaling_metrics(test_result, metrics_collector)
            
            logger.info(f"Scaling verification test completed: {test_name}")
            return test_result
            
        except Exception as e:
            logger.error(f"Scaling verification test failed: {e}")
            # Clean up on error
            await self._cleanup_test_workers()
            raise
    
    async def _setup_test_workers(self, worker_count: int):
        """Set up workers for throughput testing."""
        self.current_test_workers = []
        
        for i in range(worker_count):
            worker_id = f"test_worker_{i}"
            
            # Register worker with load balancer
            success = await self.load_balancer.register_worker(
                worker_id, 
                processing_capacity=20  # High capacity for testing
            )
            
            if success:
                # Send heartbeat to mark as healthy
                await self.load_balancer.worker_heartbeat(worker_id, {
                    "task_count": 0,
                    "total_processed": 0,
                    "error_count": 0,
                    "current_tasks": [],
                    "processing_capacity": 20
                })
                
                self.current_test_workers.append(worker_id)
                logger.debug(f"Set up test worker: {worker_id}")
            else:
                logger.error(f"Failed to register test worker: {worker_id}")
    
    async def _cleanup_test_workers(self):
        """Clean up test workers."""
        for worker_id in self.current_test_workers:
            await self.load_balancer.unregister_worker(worker_id)
            logger.debug(f"Cleaned up test worker: {worker_id}")
        
        self.current_test_workers = []
    
    async def _measure_throughput(self, worker_count: int) -> ThroughputMeasurement:
        """
        Measure throughput for a specific worker configuration.
        
        Args:
            worker_count: Number of workers to test with
            
        Returns:
            ThroughputMeasurement with results
        """
        logger.info(f"Measuring throughput with {worker_count} workers")
        
        # Warmup period
        logger.debug("Starting warmup period")
        await self._run_warmup_tasks()
        
        # Main measurement period
        logger.debug("Starting main measurement")
        start_time = time.time()
        
        # Generate and enqueue test tasks
        test_tasks = await self._generate_test_tasks()
        
        # Track task processing
        processed_tasks = []
        processing_times = []
        error_count = 0
        
        # Simulate task processing by workers
        async def simulate_worker_processing():
            nonlocal processed_tasks, processing_times, error_count
            
            while len(processed_tasks) < len(test_tasks):
                # Simulate task assignment and processing
                for worker_id in self.current_test_workers:
                    if len(processed_tasks) >= len(test_tasks):
                        break
                    
                    # Simulate processing time (50-200ms per task)
                    processing_time = 0.05 + (len(processed_tasks) % 10) * 0.015
                    await asyncio.sleep(processing_time)
                    
                    if len(processed_tasks) < len(test_tasks):
                        task = test_tasks[len(processed_tasks)]
                        processed_tasks.append(task)
                        processing_times.append(processing_time)
                        
                        # Simulate occasional errors (5% error rate)
                        if len(processed_tasks) % 20 == 0:
                            error_count += 1
                        
                        # Update worker stats
                        await self.load_balancer.worker_heartbeat(worker_id, {
                            "task_count": 1,
                            "total_processed": len(processed_tasks),
                            "error_count": error_count,
                            "current_tasks": [task.task_id],
                            "processing_capacity": 20
                        })
                
                # Brief pause to prevent tight loop
                await asyncio.sleep(0.01)
        
        # Run processing simulation with timeout
        try:
            await asyncio.wait_for(
                simulate_worker_processing(),
                timeout=self.config.test_duration_seconds
            )
        except asyncio.TimeoutError:
            logger.warning(f"Throughput measurement timed out after {self.config.test_duration_seconds}s")
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate metrics
        tasks_processed = len(processed_tasks)
        throughput_per_minute = (tasks_processed / duration) * 60 if duration > 0 else 0
        throughput_per_worker = throughput_per_minute / worker_count if worker_count > 0 else 0
        
        avg_processing_time = statistics.mean(processing_times) if processing_times else 0
        p95_processing_time = statistics.quantiles(processing_times, n=20)[18] if len(processing_times) >= 20 else avg_processing_time
        p99_processing_time = statistics.quantiles(processing_times, n=100)[98] if len(processing_times) >= 100 else p95_processing_time
        
        error_rate = error_count / tasks_processed if tasks_processed > 0 else 0
        
        measurement = ThroughputMeasurement(
            worker_count=worker_count,
            tasks_processed=tasks_processed,
            duration_seconds=duration,
            throughput_per_minute=throughput_per_minute,
            throughput_per_worker=throughput_per_worker,
            average_processing_time=avg_processing_time,
            p95_processing_time=p95_processing_time,
            p99_processing_time=p99_processing_time,
            error_rate=error_rate
        )
        
        logger.info(
            f"Throughput measurement complete: {worker_count} workers, "
            f"{throughput_per_minute:.1f} tasks/min, "
            f"{throughput_per_worker:.1f} tasks/min/worker"
        )
        
        return measurement
    
    async def _run_warmup_tasks(self):
        """Run warmup tasks to stabilize the system."""
        warmup_tasks = min(10, self.config.tasks_per_test // 10)
        
        for i in range(warmup_tasks):
            # Simulate brief task processing
            await asyncio.sleep(0.01)
        
        logger.debug(f"Warmup completed with {warmup_tasks} tasks")
    
    async def _generate_test_tasks(self) -> List[Task]:
        """Generate test tasks for throughput measurement."""
        tasks = []
        
        for i in range(self.config.tasks_per_test):
            task = Task(
                task_id=f"throughput_test_task_{i}",
                payload={"test_data": f"data_{i}", "index": i},
                created_at=datetime.now(timezone.utc),
                retry_count=0,
                max_retries=3,
                priority=TaskPriority.NORMAL.value,
                worker_id=None,
                status=TaskStatus.PENDING
            )
            tasks.append(task)
        
        return tasks
    
    def _analyze_scaling_results(self, measurements: List[ThroughputMeasurement]) -> ScalingTestResult:
        """
        Analyze scaling test results to determine linearity and efficiency.
        
        Args:
            measurements: List of throughput measurements
            
        Returns:
            ScalingTestResult with analysis
        """
        if len(measurements) < 2:
            return ScalingTestResult(
                measurements=measurements,
                linear_scaling_coefficient=0.0,
                scaling_efficiency=0.0,
                bottlenecks_detected=["Insufficient measurements"],
                recommendations=["Run test with more worker configurations"],
                is_scaling_linear=False,
                test_duration_seconds=0.0
            )
        
        # Extract data for analysis
        worker_counts = [m.worker_count for m in measurements]
        throughputs = [m.throughput_per_minute for m in measurements]
        per_worker_throughputs = [m.throughput_per_worker for m in measurements]
        
        # Calculate linear scaling coefficient (R²)
        linear_scaling_coefficient = self._calculate_linear_correlation(worker_counts, throughputs)
        
        # Calculate scaling efficiency
        baseline_throughput = measurements[0].throughput_per_minute
        baseline_workers = measurements[0].worker_count
        
        theoretical_max_throughput = baseline_throughput * (max(worker_counts) / baseline_workers)
        actual_max_throughput = max(throughputs)
        
        scaling_efficiency = actual_max_throughput / theoretical_max_throughput if theoretical_max_throughput > 0 else 0
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(measurements)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(measurements, scaling_efficiency, bottlenecks)
        
        # Determine if scaling is linear
        is_scaling_linear = (
            linear_scaling_coefficient >= self.config.linear_scaling_threshold and
            scaling_efficiency >= 0.8 and
            len(bottlenecks) == 0
        )
        
        return ScalingTestResult(
            measurements=measurements,
            linear_scaling_coefficient=linear_scaling_coefficient,
            scaling_efficiency=scaling_efficiency,
            bottlenecks_detected=bottlenecks,
            recommendations=recommendations,
            is_scaling_linear=is_scaling_linear,
            test_duration_seconds=0.0  # Will be set by caller
        )
    
    def _calculate_linear_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate R² correlation coefficient for linear relationship."""
        if len(x_values) != len(y_values) or len(x_values) < 2:
            return 0.0
        
        try:
            # Calculate means
            x_mean = statistics.mean(x_values)
            y_mean = statistics.mean(y_values)
            
            # Calculate correlation coefficient
            numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
            x_variance = sum((x - x_mean) ** 2 for x in x_values)
            y_variance = sum((y - y_mean) ** 2 for y in y_values)
            
            if x_variance == 0 or y_variance == 0:
                return 0.0
            
            correlation = numerator / (x_variance * y_variance) ** 0.5
            
            # Return R² (coefficient of determination)
            return correlation ** 2
            
        except Exception as e:
            logger.error(f"Error calculating linear correlation: {e}")
            return 0.0
    
    def _detect_bottlenecks(self, measurements: List[ThroughputMeasurement]) -> List[str]:
        """Detect performance bottlenecks from measurements."""
        bottlenecks = []
        
        # Check for throughput plateau
        if len(measurements) >= 3:
            last_three_throughputs = [m.throughput_per_minute for m in measurements[-3:]]
            if max(last_three_throughputs) - min(last_three_throughputs) < 10:
                bottlenecks.append("Throughput plateau detected - possible resource contention")
        
        # Check for declining per-worker throughput
        per_worker_throughputs = [m.throughput_per_worker for m in measurements]
        if len(per_worker_throughputs) >= 2:
            decline_threshold = 0.8  # 20% decline
            if per_worker_throughputs[-1] < per_worker_throughputs[0] * decline_threshold:
                bottlenecks.append("Per-worker throughput declining - possible coordination overhead")
        
        # Check for high error rates
        high_error_measurements = [m for m in measurements if m.error_rate > 0.1]
        if high_error_measurements:
            bottlenecks.append("High error rates detected - possible system overload")
        
        # Check for high processing times
        high_latency_measurements = [m for m in measurements if m.p99_processing_time > 1.0]
        if high_latency_measurements:
            bottlenecks.append("High processing latency detected - possible resource constraints")
        
        return bottlenecks
    
    def _generate_recommendations(
        self,
        measurements: List[ThroughputMeasurement],
        scaling_efficiency: float,
        bottlenecks: List[str]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Scaling efficiency recommendations
        if scaling_efficiency < 0.6:
            recommendations.append("Poor scaling efficiency - consider optimizing task distribution")
        elif scaling_efficiency < 0.8:
            recommendations.append("Moderate scaling efficiency - investigate coordination overhead")
        else:
            recommendations.append("Good scaling efficiency - current architecture scales well")
        
        # Bottleneck-specific recommendations
        if "Throughput plateau" in str(bottlenecks):
            recommendations.append("Consider increasing worker capacity or optimizing resource usage")
        
        if "Per-worker throughput declining" in str(bottlenecks):
            recommendations.append("Investigate load balancer overhead and task distribution algorithms")
        
        if "High error rates" in str(bottlenecks):
            recommendations.append("Implement better error handling and retry mechanisms")
        
        if "High processing latency" in str(bottlenecks):
            recommendations.append("Optimize task processing logic and resource allocation")
        
        # Target throughput recommendations
        target_per_worker = self.config.target_throughput_per_worker
        actual_per_worker = measurements[-1].throughput_per_worker if measurements else 0
        
        if actual_per_worker < target_per_worker * 0.8:
            recommendations.append(f"Below target throughput ({actual_per_worker:.1f} vs {target_per_worker:.1f} tasks/min/worker)")
        elif actual_per_worker >= target_per_worker:
            recommendations.append("Meeting or exceeding target throughput per worker")
        
        return recommendations
    
    def _record_scaling_metrics(self, result: ScalingTestResult, metrics_collector):
        """Record scaling test metrics."""
        try:
            # Record overall test metrics
            metrics_collector.record_scaling_test_result(
                linear_coefficient=result.linear_scaling_coefficient,
                scaling_efficiency=result.scaling_efficiency,
                is_linear=result.is_scaling_linear,
                test_duration=result.test_duration_seconds
            )
            
            # Record per-measurement metrics
            for measurement in result.measurements:
                metrics_collector.record_throughput_measurement(
                    worker_count=measurement.worker_count,
                    throughput_per_minute=measurement.throughput_per_minute,
                    throughput_per_worker=measurement.throughput_per_worker,
                    error_rate=measurement.error_rate
                )
                
        except Exception as e:
            logger.error(f"Failed to record scaling metrics: {e}")
    
    def get_scaling_history(self) -> List[ScalingTestResult]:
        """
        Get history of scaling test results.
        
        Returns:
            List of all scaling test results
        """
        return self.test_results.copy()
    
    def get_latest_scaling_result(self) -> Optional[ScalingTestResult]:
        """
        Get the most recent scaling test result.
        
        Returns:
            Latest ScalingTestResult or None if no tests run
        """
        return self.test_results[-1] if self.test_results else None
    
    def recommend_worker_count(self, target_throughput: float) -> Dict[str, Any]:
        """
        Recommend optimal worker count for target throughput.
        
        Args:
            target_throughput: Target throughput in tasks per minute
            
        Returns:
            Dictionary with recommendation details
        """
        if not self.test_results:
            return {
                "recommended_workers": 1,
                "confidence": "low",
                "reason": "No scaling test data available"
            }
        
        latest_result = self.test_results[-1]
        
        # Find measurement closest to target throughput
        best_measurement = None
        min_diff = float('inf')
        
        for measurement in latest_result.measurements:
            diff = abs(measurement.throughput_per_minute - target_throughput)
            if diff < min_diff:
                min_diff = diff
                best_measurement = measurement
        
        if best_measurement is None:
            return {
                "recommended_workers": 1,
                "confidence": "low",
                "reason": "No suitable measurement found"
            }
        
        # Calculate recommended worker count
        if best_measurement.throughput_per_worker > 0:
            recommended_workers = max(1, int(target_throughput / best_measurement.throughput_per_worker))
        else:
            recommended_workers = 1
        
        # Determine confidence based on scaling linearity
        if latest_result.is_scaling_linear:
            confidence = "high"
        elif latest_result.scaling_efficiency > 0.7:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "recommended_workers": recommended_workers,
            "confidence": confidence,
            "expected_throughput": recommended_workers * best_measurement.throughput_per_worker,
            "scaling_efficiency": latest_result.scaling_efficiency,
            "reason": f"Based on {best_measurement.throughput_per_worker:.1f} tasks/min/worker"
        }