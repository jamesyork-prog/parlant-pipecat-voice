"""
Retry strategy configuration for async webhook orchestration.

This module provides retry strategies for different types of failures:
- Exponential backoff for LLM rate limits and server errors
- Linear backoff for network timeouts
- Retry count tracking and maximum retry limits
"""

import time
import random
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of failures that can occur during processing."""
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_SERVER_ERROR = "llm_server_error"
    NETWORK_TIMEOUT = "network_timeout"
    FRESHDESK_API_ERROR = "freshdesk_api_error"
    PERMANENT_FAILURE = "permanent_failure"
    UNKNOWN_ERROR = "unknown_error"


class RetryDecision(Enum):
    """Decision on whether to retry a failed operation."""
    RETRY = "retry"
    DEAD_LETTER = "dead_letter"
    ESCALATE = "escalate"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int
    base_delay: float  # Base delay in seconds
    max_delay: float   # Maximum delay in seconds
    backoff_multiplier: float  # For exponential backoff
    jitter: bool = True  # Add random jitter to prevent thundering herd


@dataclass
class RetryResult:
    """Result of retry decision calculation."""
    should_retry: bool
    delay_seconds: float
    decision: RetryDecision
    reason: str
    retry_count: int
    max_retries: int


class BackoffStrategy(ABC):
    """Abstract base class for backoff strategies."""
    
    @abstractmethod
    def calculate_delay(self, retry_count: int, config: RetryConfig) -> float:
        """Calculate delay for the given retry attempt."""
        pass


class ExponentialBackoffStrategy(BackoffStrategy):
    """
    Exponential backoff strategy for LLM rate limits and server errors.
    
    Implements requirements:
    - 4.1: Exponential backoff for LLM rate limits
    - 4.2: Exponential backoff for LLM server errors
    """
    
    def calculate_delay(self, retry_count: int, config: RetryConfig) -> float:
        """
        Calculate exponential backoff delay.
        
        Formula: base_delay * (backoff_multiplier ^ retry_count)
        
        Args:
            retry_count: Current retry attempt (0-based)
            config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        delay = config.base_delay * (config.backoff_multiplier ** retry_count)
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            # Add ±25% jitter to prevent thundering herd
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay


class LinearBackoffStrategy(BackoffStrategy):
    """
    Linear backoff strategy for network timeouts.
    
    Implements requirement:
    - 4.3: Linear backoff for network timeouts
    """
    
    def calculate_delay(self, retry_count: int, config: RetryConfig) -> float:
        """
        Calculate linear backoff delay.
        
        Formula: base_delay + (retry_count * backoff_multiplier)
        
        Args:
            retry_count: Current retry attempt (0-based)
            config: Retry configuration
            
        Returns:
            Delay in seconds
        """
        delay = config.base_delay + (retry_count * config.backoff_multiplier)
        delay = min(delay, config.max_delay)
        
        if config.jitter:
            # Add ±10% jitter for linear backoff
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative
        
        return delay


class RetryStrategy:
    """
    Retry strategy manager for different failure types.
    
    Implements requirements:
    - 4.1: LLM rate limit retry up to 5 times with exponential backoff
    - 4.2: LLM server error retry up to 3 times with exponential backoff
    - 4.3: Network timeout retry up to 5 times with linear backoff
    - 4.4: Move to dead letter queue after max retries for transient failures
    - 4.5: Immediate dead letter for permanent failures
    """
    
    def __init__(self):
        """Initialize retry strategy with default configurations."""
        self.strategies = {
            FailureType.LLM_RATE_LIMIT: ExponentialBackoffStrategy(),
            FailureType.LLM_SERVER_ERROR: ExponentialBackoffStrategy(),
            FailureType.NETWORK_TIMEOUT: LinearBackoffStrategy(),
            FailureType.FRESHDESK_API_ERROR: ExponentialBackoffStrategy(),
        }
        
        self.configs = {
            # LLM rate limit: 5 retries, exponential backoff starting at 1s
            FailureType.LLM_RATE_LIMIT: RetryConfig(
                max_retries=5,
                base_delay=1.0,
                max_delay=60.0,
                backoff_multiplier=2.0,
                jitter=True
            ),
            
            # LLM server error: 3 retries, exponential backoff starting at 2s
            FailureType.LLM_SERVER_ERROR: RetryConfig(
                max_retries=3,
                base_delay=2.0,
                max_delay=30.0,
                backoff_multiplier=2.0,
                jitter=True
            ),
            
            # Network timeout: 5 retries, linear backoff starting at 1s
            FailureType.NETWORK_TIMEOUT: RetryConfig(
                max_retries=5,
                base_delay=1.0,
                max_delay=15.0,
                backoff_multiplier=2.0,  # Linear increment
                jitter=True
            ),
            
            # Freshdesk API error: 3 retries, exponential backoff
            FailureType.FRESHDESK_API_ERROR: RetryConfig(
                max_retries=3,
                base_delay=1.0,
                max_delay=20.0,
                backoff_multiplier=2.0,
                jitter=True
            ),
        }
    
    def classify_failure(self, error: Exception, context: Dict[str, Any] = None) -> FailureType:
        """
        Classify failure type based on error and context.
        
        Args:
            error: Exception that occurred
            context: Additional context about the failure
            
        Returns:
            FailureType classification
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Check for LLM rate limits
        if any(indicator in error_str for indicator in [
            "rate limit", "429", "too many requests", "quota exceeded"
        ]):
            return FailureType.LLM_RATE_LIMIT
        
        # Check for LLM server errors
        if any(indicator in error_str for indicator in [
            "500", "502", "503", "504", "internal server error", 
            "bad gateway", "service unavailable", "gateway timeout"
        ]):
            return FailureType.LLM_SERVER_ERROR
        
        # Check for network timeouts
        if any(indicator in error_str for indicator in [
            "timeout", "connection timeout", "read timeout", 
            "connect timeout", "timed out"
        ]) or "timeout" in error_type:
            return FailureType.NETWORK_TIMEOUT
        
        # Check for Freshdesk API errors
        if context and context.get("service") == "freshdesk":
            if any(indicator in error_str for indicator in [
                "401", "403", "404", "422", "unauthorized", "forbidden"
            ]):
                return FailureType.PERMANENT_FAILURE
            else:
                return FailureType.FRESHDESK_API_ERROR
        
        # Check for permanent failures
        if any(indicator in error_str for indicator in [
            "invalid ticket", "ticket not found", "malformed data",
            "validation error", "schema error", "schema validation", 
            "400", "401", "403", "404", "bad request", "unauthorized", 
            "forbidden", "not found"
        ]):
            return FailureType.PERMANENT_FAILURE
        
        return FailureType.UNKNOWN_ERROR
    
    def should_retry(
        self, 
        failure_type: FailureType, 
        retry_count: int, 
        error: Optional[Exception] = None
    ) -> RetryResult:
        """
        Determine if operation should be retried and calculate delay.
        
        Args:
            failure_type: Type of failure that occurred
            retry_count: Current retry count (0-based)
            error: Optional exception for additional context
            
        Returns:
            RetryResult with retry decision and delay
        """
        # Permanent failures should never be retried
        if failure_type == FailureType.PERMANENT_FAILURE:
            return RetryResult(
                should_retry=False,
                delay_seconds=0,
                decision=RetryDecision.DEAD_LETTER,
                reason="Permanent failure - no retry",
                retry_count=retry_count,
                max_retries=0
            )
        
        # Get configuration for this failure type
        config = self.configs.get(failure_type)
        if not config:
            # Unknown error type - use conservative retry
            config = RetryConfig(
                max_retries=2,
                base_delay=5.0,
                max_delay=30.0,
                backoff_multiplier=2.0,
                jitter=True
            )
        
        # Check if we've exceeded max retries
        if retry_count >= config.max_retries:
            return RetryResult(
                should_retry=False,
                delay_seconds=0,
                decision=RetryDecision.DEAD_LETTER,
                reason=f"Max retries ({config.max_retries}) exceeded",
                retry_count=retry_count,
                max_retries=config.max_retries
            )
        
        # Calculate delay using appropriate strategy
        strategy = self.strategies.get(failure_type, ExponentialBackoffStrategy())
        delay = strategy.calculate_delay(retry_count, config)
        
        return RetryResult(
            should_retry=True,
            delay_seconds=delay,
            decision=RetryDecision.RETRY,
            reason=f"Retry {retry_count + 1}/{config.max_retries} after {delay:.1f}s",
            retry_count=retry_count,
            max_retries=config.max_retries
        )
    
    def get_retry_delays(self, failure_type: FailureType) -> List[float]:
        """
        Get all retry delays for a failure type (for testing/monitoring).
        
        Args:
            failure_type: Type of failure
            
        Returns:
            List of delays for each retry attempt
        """
        config = self.configs.get(failure_type)
        if not config:
            return []
        
        strategy = self.strategies.get(failure_type, ExponentialBackoffStrategy())
        delays = []
        
        for retry_count in range(config.max_retries):
            delay = strategy.calculate_delay(retry_count, config)
            delays.append(delay)
        
        return delays
    
    def update_config(self, failure_type: FailureType, config: RetryConfig):
        """
        Update retry configuration for a failure type.
        
        Args:
            failure_type: Type of failure to configure
            config: New retry configuration
        """
        self.configs[failure_type] = config
        logger.info(f"Updated retry config for {failure_type.value}: {config}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get retry strategy statistics.
        
        Returns:
            Dictionary with configuration and statistics
        """
        stats = {
            "failure_types": {},
            "strategies": list(self.strategies.keys())
        }
        
        for failure_type, config in self.configs.items():
            strategy = self.strategies.get(failure_type)
            delays = self.get_retry_delays(failure_type)
            
            stats["failure_types"][failure_type.value] = {
                "max_retries": config.max_retries,
                "base_delay": config.base_delay,
                "max_delay": config.max_delay,
                "backoff_multiplier": config.backoff_multiplier,
                "strategy": strategy.__class__.__name__,
                "retry_delays": delays,
                "total_max_delay": sum(delays)
            }
        
        return stats


# Global retry strategy instance
retry_strategy = RetryStrategy()


def classify_error_for_retry(error: Exception, context: Dict[str, Any] = None) -> FailureType:
    """
    Convenience function to classify error for retry.
    
    Args:
        error: Exception that occurred
        context: Additional context
        
    Returns:
        FailureType classification
    """
    return retry_strategy.classify_failure(error, context)


def should_retry_operation(
    error: Exception, 
    retry_count: int, 
    context: Dict[str, Any] = None
) -> RetryResult:
    """
    Convenience function to determine if operation should be retried.
    
    Args:
        error: Exception that occurred
        retry_count: Current retry count
        context: Additional context
        
    Returns:
        RetryResult with retry decision
    """
    failure_type = retry_strategy.classify_failure(error, context)
    return retry_strategy.should_retry(failure_type, retry_count, error)