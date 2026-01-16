"""
Performance Profiler for Gemini Context Caching.

This module provides utilities for profiling and optimizing cache performance:
- Memory usage tracking and analysis
- Performance bottleneck identification
- Optimization recommendations
- Real-time performance monitoring

Requirements: 11.1, 11.2, 11.3, 11.4
"""

import gc
import logging
import psutil
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Snapshot of performance metrics at a point in time."""
    timestamp: datetime
    memory_rss_mb: float
    memory_vms_mb: float
    cpu_percent: float
    tracemalloc_current_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None
    operation_name: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Complete performance profile for an operation."""
    operation_name: str
    start_snapshot: PerformanceSnapshot
    end_snapshot: PerformanceSnapshot
    duration_ms: float
    memory_delta_mb: float
    peak_memory_mb: float
    cpu_usage_avg: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def memory_efficiency_score(self) -> float:
        """Calculate memory efficiency score (0-100)."""
        if self.memory_delta_mb <= 0:
            return 100.0
        elif self.memory_delta_mb <= 1.0:
            return 90.0
        elif self.memory_delta_mb <= 5.0:
            return 70.0
        elif self.memory_delta_mb <= 10.0:
            return 50.0
        else:
            return 20.0
    
    @property
    def performance_grade(self) -> str:
        """Get overall performance grade."""
        memory_score = self.memory_efficiency_score
        
        # Time-based scoring (operation-specific)
        time_score = 100.0
        if "content_loading" in self.operation_name.lower():
            if self.duration_ms > 3000:
                time_score = 20.0
            elif self.duration_ms > 2000:
                time_score = 50.0
            elif self.duration_ms > 1000:
                time_score = 70.0
            else:
                time_score = 90.0
        elif "prompt_construction" in self.operation_name.lower():
            if self.duration_ms > 10:
                time_score = 20.0
            elif self.duration_ms > 5:
                time_score = 50.0
            elif self.duration_ms > 2:
                time_score = 70.0
            else:
                time_score = 90.0
        
        overall_score = (memory_score + time_score) / 2
        
        if overall_score >= 90:
            return "A"
        elif overall_score >= 80:
            return "B"
        elif overall_score >= 70:
            return "C"
        elif overall_score >= 60:
            return "D"
        else:
            return "F"


class PerformanceProfiler:
    """
    Performance profiler for cache operations.
    
    Provides detailed profiling of memory usage, CPU usage, and timing
    for cache operations to identify optimization opportunities.
    """
    
    def __init__(self, enable_tracemalloc: bool = True):
        """
        Initialize performance profiler.
        
        Args:
            enable_tracemalloc: Whether to enable detailed memory tracing
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.process = psutil.Process()
        self.profiles: List[PerformanceProfile] = []
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            logger.info("Memory tracing enabled")
    
    def take_snapshot(self, operation_name: Optional[str] = None) -> PerformanceSnapshot:
        """
        Take a performance snapshot.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Returns:
            Performance snapshot
        """
        # Get memory info
        memory_info = self.process.memory_info()
        memory_rss_mb = memory_info.rss / 1024 / 1024
        memory_vms_mb = memory_info.vms / 1024 / 1024
        
        # Get CPU usage
        cpu_percent = self.process.cpu_percent()
        
        # Get tracemalloc info if enabled
        tracemalloc_current_mb = None
        tracemalloc_peak_mb = None
        
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_current_mb = current / 1024 / 1024
            tracemalloc_peak_mb = peak / 1024 / 1024
        
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            memory_rss_mb=memory_rss_mb,
            memory_vms_mb=memory_vms_mb,
            cpu_percent=cpu_percent,
            tracemalloc_current_mb=tracemalloc_current_mb,
            tracemalloc_peak_mb=tracemalloc_peak_mb,
            operation_name=operation_name
        )
    
    @contextmanager
    def profile_operation(self, operation_name: str, **custom_metrics):
        """
        Context manager for profiling an operation.
        
        Args:
            operation_name: Name of the operation
            **custom_metrics: Additional metrics to track
            
        Usage:
            with profiler.profile_operation("content_loading", file_size_kb=100):
                # ... perform operation ...
        """
        # Take start snapshot
        start_snapshot = self.take_snapshot(operation_name)
        start_time = time.perf_counter()
        
        # Store active profile info
        profile_id = f"{operation_name}_{id(start_snapshot)}"
        self.active_profiles[profile_id] = {
            "start_snapshot": start_snapshot,
            "start_time": start_time,
            "custom_metrics": custom_metrics
        }
        
        try:
            yield profile_id
        finally:
            # Take end snapshot
            end_time = time.perf_counter()
            end_snapshot = self.take_snapshot(operation_name)
            
            # Calculate metrics
            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = end_snapshot.memory_rss_mb - start_snapshot.memory_rss_mb
            
            # Determine peak memory
            peak_memory_mb = max(
                start_snapshot.memory_rss_mb,
                end_snapshot.memory_rss_mb
            )
            
            if (start_snapshot.tracemalloc_peak_mb is not None and 
                end_snapshot.tracemalloc_peak_mb is not None):
                peak_memory_mb = max(peak_memory_mb, end_snapshot.tracemalloc_peak_mb)
            
            # Calculate average CPU usage (simplified)
            cpu_usage_avg = (start_snapshot.cpu_percent + end_snapshot.cpu_percent) / 2
            
            # Create profile
            profile = PerformanceProfile(
                operation_name=operation_name,
                start_snapshot=start_snapshot,
                end_snapshot=end_snapshot,
                duration_ms=duration_ms,
                memory_delta_mb=memory_delta_mb,
                peak_memory_mb=peak_memory_mb,
                cpu_usage_avg=cpu_usage_avg,
                custom_metrics=custom_metrics
            )
            
            self.profiles.append(profile)
            
            # Clean up active profile
            if profile_id in self.active_profiles:
                del self.active_profiles[profile_id]
            
            logger.debug(
                f"Profile completed: {operation_name} - "
                f"{duration_ms:.2f}ms, {memory_delta_mb:.2f}MB delta, "
                f"grade: {profile.performance_grade}"
            )
    
    def get_operation_profiles(self, operation_name: str) -> List[PerformanceProfile]:
        """
        Get all profiles for a specific operation.
        
        Args:
            operation_name: Name of the operation
            
        Returns:
            List of profiles for the operation
        """
        return [p for p in self.profiles if p.operation_name == operation_name]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get summary of all performance profiles.
        
        Returns:
            Dictionary with performance summary
        """
        if not self.profiles:
            return {"status": "no_profiles", "total_profiles": 0}
        
        # Group profiles by operation
        operations = {}
        for profile in self.profiles:
            if profile.operation_name not in operations:
                operations[profile.operation_name] = []
            operations[profile.operation_name].append(profile)
        
        # Calculate statistics for each operation
        operation_stats = {}
        for op_name, profiles in operations.items():
            durations = [p.duration_ms for p in profiles]
            memory_deltas = [p.memory_delta_mb for p in profiles]
            grades = [p.performance_grade for p in profiles]
            
            operation_stats[op_name] = {
                "profile_count": len(profiles),
                "duration_ms": {
                    "avg": sum(durations) / len(durations),
                    "min": min(durations),
                    "max": max(durations)
                },
                "memory_delta_mb": {
                    "avg": sum(memory_deltas) / len(memory_deltas),
                    "min": min(memory_deltas),
                    "max": max(memory_deltas)
                },
                "performance_grades": {
                    "A": grades.count("A"),
                    "B": grades.count("B"),
                    "C": grades.count("C"),
                    "D": grades.count("D"),
                    "F": grades.count("F")
                }
            }
        
        return {
            "status": "active",
            "total_profiles": len(self.profiles),
            "operations": operation_stats,
            "overall_memory_usage_mb": {
                "current": self.process.memory_info().rss / 1024 / 1024,
                "peak": max(p.peak_memory_mb for p in self.profiles)
            }
        }
    
    def identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks from profiles.
        
        Returns:
            List of identified bottlenecks with recommendations
        """
        bottlenecks = []
        
        if not self.profiles:
            return bottlenecks
        
        # Group by operation
        operations = {}
        for profile in self.profiles:
            if profile.operation_name not in operations:
                operations[profile.operation_name] = []
            operations[profile.operation_name].append(profile)
        
        for op_name, profiles in operations.items():
            # Check for slow operations
            slow_profiles = [p for p in profiles if p.performance_grade in ["D", "F"]]
            if slow_profiles:
                avg_duration = sum(p.duration_ms for p in slow_profiles) / len(slow_profiles)
                bottlenecks.append({
                    "type": "slow_operation",
                    "operation": op_name,
                    "severity": "high" if len(slow_profiles) > len(profiles) * 0.5 else "medium",
                    "description": f"{len(slow_profiles)} slow executions of {op_name}",
                    "avg_duration_ms": avg_duration,
                    "recommendation": self._get_performance_recommendation(op_name, "slow")
                })
            
            # Check for memory-intensive operations
            high_memory_profiles = [p for p in profiles if p.memory_delta_mb > 5.0]
            if high_memory_profiles:
                avg_memory = sum(p.memory_delta_mb for p in high_memory_profiles) / len(high_memory_profiles)
                bottlenecks.append({
                    "type": "high_memory_usage",
                    "operation": op_name,
                    "severity": "medium",
                    "description": f"{len(high_memory_profiles)} high-memory executions of {op_name}",
                    "avg_memory_mb": avg_memory,
                    "recommendation": self._get_performance_recommendation(op_name, "memory")
                })
            
            # Check for inconsistent performance
            durations = [p.duration_ms for p in profiles]
            if len(durations) > 5:
                import statistics
                std_dev = statistics.stdev(durations)
                mean_duration = statistics.mean(durations)
                cv = std_dev / mean_duration if mean_duration > 0 else 0
                
                if cv > 0.5:  # High coefficient of variation
                    bottlenecks.append({
                        "type": "inconsistent_performance",
                        "operation": op_name,
                        "severity": "low",
                        "description": f"Inconsistent performance in {op_name} (CV: {cv:.2f})",
                        "coefficient_of_variation": cv,
                        "recommendation": "Investigate causes of performance variability"
                    })
        
        return bottlenecks
    
    def _get_performance_recommendation(self, operation_name: str, issue_type: str) -> str:
        """Get performance recommendation for an operation and issue type."""
        recommendations = {
            ("content_loading", "slow"): "Consider caching loaded content, optimizing file I/O, or implementing lazy loading",
            ("content_loading", "memory"): "Implement content compression or streaming for large files",
            ("prompt_construction", "slow"): "Cache template objects, optimize string operations, or pre-compile templates",
            ("prompt_construction", "memory"): "Reuse template objects and optimize string concatenation",
            ("metrics_tracking", "slow"): "Optimize data structures or implement sampling for high-frequency metrics",
            ("metrics_tracking", "memory"): "Implement metric aggregation or use more compact storage formats"
        }
        
        key = (operation_name.lower(), issue_type)
        return recommendations.get(key, f"Optimize {operation_name} for {issue_type} performance")
    
    def generate_optimization_report(self) -> str:
        """
        Generate a detailed optimization report.
        
        Returns:
            Formatted optimization report
        """
        lines = [
            "=" * 80,
            "PERFORMANCE OPTIMIZATION REPORT",
            "=" * 80,
            f"Generated: {datetime.now().isoformat()}",
            f"Total Profiles: {len(self.profiles)}",
            ""
        ]
        
        # Performance summary
        summary = self.get_performance_summary()
        if summary["status"] == "active":
            lines.extend([
                "PERFORMANCE SUMMARY",
                "-" * 40
            ])
            
            for op_name, stats in summary["operations"].items():
                lines.extend([
                    f"{op_name.upper()}:",
                    f"  Profiles: {stats['profile_count']}",
                    f"  Avg Duration: {stats['duration_ms']['avg']:.2f}ms",
                    f"  Avg Memory Delta: {stats['memory_delta_mb']['avg']:.2f}MB",
                    f"  Grades: A={stats['performance_grades']['A']}, "
                    f"B={stats['performance_grades']['B']}, "
                    f"C={stats['performance_grades']['C']}, "
                    f"D={stats['performance_grades']['D']}, "
                    f"F={stats['performance_grades']['F']}",
                    ""
                ])
        
        # Bottlenecks
        bottlenecks = self.identify_performance_bottlenecks()
        if bottlenecks:
            lines.extend([
                "IDENTIFIED BOTTLENECKS",
                "-" * 40
            ])
            
            for bottleneck in bottlenecks:
                lines.extend([
                    f"[{bottleneck['severity'].upper()}] {bottleneck['type']} in {bottleneck['operation']}",
                    f"  Description: {bottleneck['description']}",
                    f"  Recommendation: {bottleneck['recommendation']}",
                    ""
                ])
        else:
            lines.extend([
                "IDENTIFIED BOTTLENECKS",
                "-" * 40,
                "No significant bottlenecks identified.",
                ""
            ])
        
        # Memory analysis
        if summary["status"] == "active":
            current_memory = summary["overall_memory_usage_mb"]["current"]
            peak_memory = summary["overall_memory_usage_mb"]["peak"]
            
            lines.extend([
                "MEMORY ANALYSIS",
                "-" * 40,
                f"Current Memory Usage: {current_memory:.2f} MB",
                f"Peak Memory Usage: {peak_memory:.2f} MB",
                f"Memory Efficiency: {'Good' if peak_memory < 15.0 else 'Needs Optimization'}",
                ""
            ])
        
        # Recommendations
        lines.extend([
            "OPTIMIZATION RECOMMENDATIONS",
            "-" * 40
        ])
        
        if not bottlenecks:
            lines.append("✓ No critical performance issues detected")
        else:
            high_priority = [b for b in bottlenecks if b["severity"] == "high"]
            medium_priority = [b for b in bottlenecks if b["severity"] == "medium"]
            
            if high_priority:
                lines.append("HIGH PRIORITY:")
                for bottleneck in high_priority:
                    lines.append(f"  • {bottleneck['recommendation']}")
            
            if medium_priority:
                lines.append("MEDIUM PRIORITY:")
                for bottleneck in medium_priority:
                    lines.append(f"  • {bottleneck['recommendation']}")
        
        lines.extend([
            "",
            "=" * 80
        ])
        
        return "\n".join(lines)
    
    def export_profiles_to_json(self, filepath: str):
        """
        Export profiles to JSON file for external analysis.
        
        Args:
            filepath: Path to export JSON file
        """
        import json
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_profiles": len(self.profiles),
            "profiles": []
        }
        
        for profile in self.profiles:
            profile_data = {
                "operation_name": profile.operation_name,
                "duration_ms": profile.duration_ms,
                "memory_delta_mb": profile.memory_delta_mb,
                "peak_memory_mb": profile.peak_memory_mb,
                "cpu_usage_avg": profile.cpu_usage_avg,
                "performance_grade": profile.performance_grade,
                "memory_efficiency_score": profile.memory_efficiency_score,
                "start_timestamp": profile.start_snapshot.timestamp.isoformat(),
                "end_timestamp": profile.end_snapshot.timestamp.isoformat(),
                "custom_metrics": profile.custom_metrics
            }
            export_data["profiles"].append(profile_data)
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(self.profiles)} profiles to {filepath}")
    
    def clear_profiles(self):
        """Clear all stored profiles."""
        self.profiles.clear()
        self.active_profiles.clear()
        logger.info("All profiles cleared")
    
    def __del__(self):
        """Cleanup when profiler is destroyed."""
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()


# Convenience functions for common profiling scenarios
def profile_content_loading(profiler: PerformanceProfiler, loader_func: Callable, **kwargs):
    """
    Profile content loading operation.
    
    Args:
        profiler: Performance profiler instance
        loader_func: Function that performs content loading
        **kwargs: Additional metrics to track
    """
    with profiler.profile_operation("content_loading", **kwargs):
        return loader_func()


def profile_prompt_construction(profiler: PerformanceProfiler, builder_func: Callable, **kwargs):
    """
    Profile prompt construction operation.
    
    Args:
        profiler: Performance profiler instance
        builder_func: Function that performs prompt construction
        **kwargs: Additional metrics to track
    """
    with profiler.profile_operation("prompt_construction", **kwargs):
        return builder_func()


def profile_cache_operation(profiler: PerformanceProfiler, operation_name: str, operation_func: Callable, **kwargs):
    """
    Profile any cache operation.
    
    Args:
        profiler: Performance profiler instance
        operation_name: Name of the operation
        operation_func: Function that performs the operation
        **kwargs: Additional metrics to track
    """
    with profiler.profile_operation(operation_name, **kwargs):
        return operation_func()


# Global profiler instance for convenience
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def enable_global_profiling():
    """Enable global performance profiling."""
    profiler = get_global_profiler()
    logger.info("Global performance profiling enabled")
    return profiler


def disable_global_profiling():
    """Disable global performance profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.clear_profiles()
        _global_profiler = None
    logger.info("Global performance profiling disabled")


def get_global_performance_report() -> str:
    """Get performance report from global profiler."""
    profiler = get_global_profiler()
    return profiler.generate_optimization_report()