"""
Data models for Gemini Context Caching.

This module defines the core data structures used throughout the caching system:
- StaticContent: Container for loaded static content
- CacheMetrics: Metrics for cache effectiveness tracking
- CacheConfiguration: System configuration with validation
- CostCalculation: Cost calculation results
- ContentLoadingStatus: Status tracking for content loading operations
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List
from enum import Enum


class ContentType(Enum):
    """Types of static content that can be loaded."""
    SYSTEM_IDENTITY = "system_identity"
    REFUND_POLICIES = "refund_policies"
    TOOL_DEFINITIONS = "tool_definitions"
    FEW_SHOT_EXAMPLES = "few_shot_examples"


class LoadingStatus(Enum):
    """Status of content loading operations."""
    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"
    FALLBACK = "fallback"


class ContentPriority(Enum):
    """Priority levels for content when size limits are exceeded."""
    CRITICAL = 1  # Must be included (policies)
    HIGH = 2      # Important (tools)
    MEDIUM = 3    # Helpful (examples)
    LOW = 4       # Optional (extended identity)


class ValidationStatus(Enum):
    """Status of content validation operations."""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    SIZE_EXCEEDED = "size_exceeded"
    STRUCTURE_INVALID = "structure_invalid"


@dataclass
class ContentValidationResult:
    """
    Result of content validation operations.
    
    Attributes:
        status: Validation status
        total_tokens: Total token count
        content_size_bytes: Content size in bytes
        structure_valid: Whether prompt structure is valid
        size_within_limits: Whether content fits within limits
        warnings: List of validation warnings
        errors: List of validation errors
        truncated_content: Content after truncation (if any)
        truncation_applied: Whether truncation was applied
    """
    status: ValidationStatus
    total_tokens: int = 0
    content_size_bytes: int = 0
    structure_valid: bool = True
    size_within_limits: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    truncated_content: Optional[str] = None
    truncation_applied: bool = False


@dataclass
class ContentLoadingStatus:
    """
    Status tracking for content loading operations.
    
    Attributes:
        content_type: Type of content being loaded
        status: Current loading status
        loaded_at: Timestamp when content was loaded
        error_message: Error message if loading failed
        file_path: Path to the content file
        token_count: Estimated token count for loaded content
    """
    content_type: ContentType
    status: LoadingStatus = LoadingStatus.NOT_LOADED
    loaded_at: Optional[datetime] = None
    error_message: Optional[str] = None
    file_path: Optional[str] = None
    token_count: int = 0


@dataclass
class StaticContent:
    """
    Container for loaded static content used in prompt templates.
    
    Static content is placed at the beginning of prompts to maximize
    Gemini's implicit caching effectiveness.
    
    Attributes:
        system_identity: System role and identity instructions
        refund_policies: Refund policy documents
        tool_definitions: JSON schema for available tools
        few_shot_examples: Training examples for the model
        total_tokens: Estimated total token count
        content_hash: Hash for change detection
        loaded_at: Timestamp when content was loaded
        loading_statuses: Status for each content type
        validation_result: Result of content validation
    """
    system_identity: str = ""
    refund_policies: str = ""
    tool_definitions: str = ""
    few_shot_examples: str = ""
    total_tokens: int = 0
    content_hash: str = ""
    loaded_at: datetime = field(default_factory=datetime.now)
    loading_statuses: Dict[ContentType, ContentLoadingStatus] = field(default_factory=dict)
    validation_result: Optional[ContentValidationResult] = None
    
    def is_complete(self) -> bool:
        """Check if all content types are loaded successfully."""
        return all(
            status.status == LoadingStatus.LOADED
            for status in self.loading_statuses.values()
        )
    
    def has_fallback_content(self) -> bool:
        """Check if any content is using fallback."""
        return any(
            status.status == LoadingStatus.FALLBACK
            for status in self.loading_statuses.values()
        )
    
    def get_failed_content_types(self) -> List[ContentType]:
        """Get list of content types that failed to load."""
        return [
            content_type
            for content_type, status in self.loading_statuses.items()
            if status.status == LoadingStatus.FAILED
        ]
    
    def get_content_by_priority(self) -> List[tuple[ContentType, str, ContentPriority]]:
        """
        Get content ordered by priority for truncation.
        
        Returns:
            List of (content_type, content, priority) tuples ordered by priority
        """
        content_priority_map = [
            (ContentType.REFUND_POLICIES, self.refund_policies, ContentPriority.CRITICAL),
            (ContentType.TOOL_DEFINITIONS, self.tool_definitions, ContentPriority.HIGH),
            (ContentType.FEW_SHOT_EXAMPLES, self.few_shot_examples, ContentPriority.MEDIUM),
            (ContentType.SYSTEM_IDENTITY, self.system_identity, ContentPriority.LOW),
        ]
        
        # Sort by priority (lower number = higher priority)
        return sorted(content_priority_map, key=lambda x: x[2].value)


@dataclass
class CacheMetrics:
    """
    Metrics for implicit cache effectiveness tracking.
    
    Tracks token counts, costs, and savings for each LLM request
    to measure the effectiveness of prompt structure optimization.
    
    Attributes:
        cached_tokens: Number of tokens cached by Gemini
        dynamic_tokens: Number of tokens not cached (full price)
        total_tokens: Total input tokens
        cost_with_cache: Actual cost with caching
        cost_without_cache: What cost would be without caching
        savings: Cost savings from caching
        cache_effectiveness: Percentage of tokens cached (0-100)
        timestamp: When this metric was recorded
        request_id: Optional request tracking ID
    """
    cached_tokens: int = 0
    dynamic_tokens: int = 0
    total_tokens: int = 0
    cost_with_cache: float = 0.0
    cost_without_cache: float = 0.0
    savings: float = 0.0
    cache_effectiveness: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: Optional[str] = None
    
    def savings_percentage(self) -> float:
        """Calculate percentage savings from caching."""
        if self.cost_without_cache == 0:
            return 0.0
        return (self.savings / self.cost_without_cache) * 100


@dataclass
class CostCalculation:
    """
    Detailed cost calculation for cache effectiveness analysis.
    
    Provides breakdown of costs for cached vs uncached tokens
    and the rates used for calculation.
    
    Attributes:
        cached_tokens: Number of cached tokens
        dynamic_tokens: Number of uncached tokens
        total_tokens: Total token count
        cached_content_cost: Cost for cached tokens
        dynamic_content_cost: Cost for uncached tokens
        total_cost_with_cache: Total cost with caching
        total_cost_without_cache: Total cost without caching
        savings: Cost savings
        cached_rate_per_1m: Rate used for cached tokens
        uncached_rate_per_1m: Rate used for uncached tokens
        discount_percentage: Discount percentage for cached tokens
    """
    cached_tokens: int
    dynamic_tokens: int
    total_tokens: int
    cached_content_cost: float
    dynamic_content_cost: float
    total_cost_with_cache: float
    total_cost_without_cache: float
    savings: float
    cached_rate_per_1m: float
    uncached_rate_per_1m: float
    discount_percentage: float
    
    def savings_percentage(self) -> float:
        """Calculate percentage savings from caching."""
        if self.total_cost_without_cache == 0:
            return 0.0
        return (self.savings / self.total_cost_without_cache) * 100


@dataclass
class CacheConfiguration:
    """
    Configuration for Gemini context caching system.
    
    Supports implicit caching optimization without Redis or manual
    cache token management. Configuration can be loaded from
    environment variables or JSON files.
    
    Attributes:
        enabled: Whether cache optimization is enabled
        monitor_cache_effectiveness: Whether to track cache metrics
        static_content_paths: Paths to static content files
        tool_definitions_path: Path to tool definitions directory
        few_shot_examples_path: Path to few-shot examples file
        gemini_api_key: API key for Gemini (from environment)
        input_cost_per_1m: Standard input cost per 1M tokens
        cached_input_cost_per_1m: Cached input cost per 1M tokens
        output_cost_per_1m: Output cost per 1M tokens
        max_static_content_tokens: Maximum tokens for static content
        prompt_construction_timeout_ms: Timeout for prompt construction
        content_loading_timeout_s: Timeout for content loading at startup
        cache_effectiveness_alert_threshold: Alert when below this %
        hot_reload_enabled: Whether to support config hot reload
    """
    # Feature Flags
    enabled: bool = True
    monitor_cache_effectiveness: bool = True
    
    # Content Paths
    static_content_paths: Dict[str, str] = field(default_factory=lambda: {
        "refund_policy": "parlant/context/processed/refund_policy_condensed.md",
        "decision_chart": "parlant/context/processed/refund_scenario_decision_chart.md",
        "ai_vs_human": "parlant/context/processed/ai_vs_human_refund_scenarios.md",
    })
    tool_definitions_path: str = "parlant/tools/"
    few_shot_examples_path: str = "parlant/context/few_shot_examples.json"
    system_identity_path: str = ""  # Empty means use default template
    
    # API Configuration
    gemini_api_key: str = ""
    
    # Pricing Configuration (Gemini 2.5 Flash rates)
    # 90% discount for cached tokens on Gemini 2.5
    input_cost_per_1m: float = 0.30
    cached_input_cost_per_1m: float = 0.03  # 90% discount
    output_cost_per_1m: float = 2.50
    
    # Performance Limits
    max_static_content_tokens: int = 32000
    prompt_construction_timeout_ms: int = 10
    content_loading_timeout_s: int = 3
    
    # Monitoring Configuration
    cache_effectiveness_alert_threshold: float = 70.0  # Alert if below 70%
    
    # Hot Reload
    hot_reload_enabled: bool = True
    
    def get_discount_percentage(self) -> float:
        """Calculate the discount percentage for cached tokens."""
        if self.input_cost_per_1m == 0:
            return 0.0
        return (1 - (self.cached_input_cost_per_1m / self.input_cost_per_1m)) * 100
    
    def validate(self) -> List[str]:
        """
        Validate configuration and return list of errors.
        
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        # Validate pricing
        if self.input_cost_per_1m < 0:
            errors.append("input_cost_per_1m must be non-negative")
        if self.cached_input_cost_per_1m < 0:
            errors.append("cached_input_cost_per_1m must be non-negative")
        if self.output_cost_per_1m < 0:
            errors.append("output_cost_per_1m must be non-negative")
        if self.cached_input_cost_per_1m > self.input_cost_per_1m:
            errors.append("cached_input_cost_per_1m should not exceed input_cost_per_1m")
        
        # Validate limits
        if self.max_static_content_tokens <= 0:
            errors.append("max_static_content_tokens must be positive")
        if self.prompt_construction_timeout_ms <= 0:
            errors.append("prompt_construction_timeout_ms must be positive")
        if self.content_loading_timeout_s <= 0:
            errors.append("content_loading_timeout_s must be positive")
        
        # Validate alert threshold
        if not (0 <= self.cache_effectiveness_alert_threshold <= 100):
            errors.append("cache_effectiveness_alert_threshold must be between 0 and 100")
        
        return errors
