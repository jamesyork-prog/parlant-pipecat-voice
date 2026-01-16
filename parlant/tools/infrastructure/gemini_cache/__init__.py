"""
Gemini Context Caching Infrastructure.

This module provides components for optimizing LLM costs through Gemini's implicit caching:
- ContentLoader: Loads and structures static content for prompt templates
- RequestBuilder: Constructs prompts optimized for implicit caching
- CacheMetricsTracker: Tracks cache effectiveness and cost savings
- CacheConfiguration: Configuration management with validation
- CacheAuditLogger: Comprehensive audit logging for troubleshooting

Implicit caching is automatic in Gemini 2.5+ models - content at the beginning of prompts
is automatically cached, providing 90% cost reduction on cached tokens.
"""

from .models import (
    StaticContent,
    CacheMetrics,
    CacheConfiguration,
    CostCalculation,
    ContentLoadingStatus,
)
from .config import (
    load_cache_config,
    validate_cache_config,
    get_validated_cache_config,
    DEFAULT_CACHE_CONFIG,
)
from .content_loader import ContentLoader
from .request_builder import RequestBuilder
from .metrics_tracker import CacheMetricsTracker
from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceAlert,
    PerformanceOptimizationResult,
    get_performance_optimizer,
    reset_performance_optimizer,
)
from .audit_logger import (
    CacheAuditLogger,
    AuditContext,
    AuditEventType,
    AuditSeverity,
    get_cache_audit_logger,
    reset_cache_audit_logger,
)

__all__ = [
    # Data Models
    "StaticContent",
    "CacheMetrics",
    "CacheConfiguration",
    "CostCalculation",
    "ContentLoadingStatus",
    # Configuration
    "load_cache_config",
    "validate_cache_config",
    "get_validated_cache_config",
    "DEFAULT_CACHE_CONFIG",
    # Components
    "ContentLoader",
    "RequestBuilder",
    "CacheMetricsTracker",
    # Performance Optimization
    "PerformanceOptimizer",
    "PerformanceAlert",
    "PerformanceOptimizationResult",
    "get_performance_optimizer",
    "reset_performance_optimizer",
    # Audit Logging
    "CacheAuditLogger",
    "AuditContext",
    "AuditEventType",
    "AuditSeverity",
    "get_cache_audit_logger",
    "reset_cache_audit_logger",
]
