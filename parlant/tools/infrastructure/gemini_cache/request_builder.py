"""
Request Builder for Gemini Context Caching.

This module constructs LLM requests optimized for implicit caching:
- Structures prompts with static content first, dynamic content last
- Extracts cache metrics from response metadata
- Calculates cost savings from caching

Requirements: 6.1-6.5, 7.1-7.5
"""

import logging
import uuid
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any

from .models import CacheMetrics, CacheConfiguration, CostCalculation
from .content_loader import ContentLoader
from .metrics_tracker import CacheMetricsTracker
from .audit_logger import get_cache_audit_logger, AuditContext

logger = logging.getLogger(__name__)


class RequestBuilder:
    """
    Constructs LLM requests optimized for implicit caching.
    
    Structures prompts with static content first, dynamic content last.
    Monitors cache effectiveness via response metadata.
    """
    
    def __init__(
        self,
        content_loader: ContentLoader,
        config: CacheConfiguration,
        metrics_tracker: Optional[CacheMetricsTracker] = None
    ):
        """
        Initialize request builder.
        
        Args:
            content_loader: Content loader for static content
            config: Cache configuration with pricing
            metrics_tracker: Optional metrics tracker for recording
        """
        self.content_loader = content_loader
        self.config = config
        self.metrics_tracker = metrics_tracker or CacheMetricsTracker()
    
    async def build_request(
        self,
        ticket_data: Dict[str, Any],
        booking_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build LLM request optimized for implicit caching.
        
        Args:
            ticket_data: Dynamic ticket information
            booking_data: Optional dynamic booking information
            additional_context: Optional additional context
            
        Returns:
            Dict containing:
                - request: The request dict for the API
                - request_id: Tracking ID
                - static_content_tokens: Token count for static content
                - dynamic_content_tokens: Token count for dynamic content
                - total_tokens: Total token count
            
        Performance: <10ms per request
        """
        start_time = time.perf_counter()
        
        # Generate tracking ID
        request_id = self._generate_request_id()
        
        with AuditContext("prompt_construction", self.config) as audit:
            # Get prompt template with static content first
            template = self.content_loader.get_prompt_template()
            
            # Format dynamic content
            dynamic_content = self._format_dynamic_content(
                ticket_data,
                booking_data,
                additional_context
            )
            
            # Construct final prompt (static first, dynamic last)
            final_prompt = template.format(dynamic_content=dynamic_content)
            
            # Calculate token estimates for audit logging
            static_content = self.content_loader.get_static_content()
            static_tokens = static_content.total_tokens if static_content else 0
            dynamic_tokens = self._estimate_tokens(dynamic_content)
            total_tokens = self._estimate_tokens(final_prompt)
            
            # Build request
            request = {
                "contents": [{
                    "parts": [{
                        "text": final_prompt
                    }]
                }],
                "generation_config": {
                    "temperature": 0.1,
                    "max_output_tokens": 1000
                }
            }
            
            # Log performance timing
            construction_time_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(
                f"Built implicit cache optimized request: {request_id} "
                f"(construction time: {construction_time_ms:.2f}ms)"
            )
            
            # Validate prompt structure
            prompt_structure_valid = self._validate_prompt_structure(final_prompt)
            
            # Audit log prompt construction
            audit.log_prompt_constructed(
                request_id,
                static_tokens,
                dynamic_tokens,
                total_tokens,
                construction_time_ms,
                prompt_structure_valid,
                {
                    "ticket_data_keys": list(ticket_data.keys()) if ticket_data else [],
                    "has_booking_data": booking_data is not None,
                    "has_additional_context": additional_context is not None,
                    "template_length": len(template),
                    "final_prompt_length": len(final_prompt)
                }
            )
            
            # Alert if construction time exceeds target
            if construction_time_ms > self.config.prompt_construction_timeout_ms:
                logger.warning(
                    f"Prompt construction time ({construction_time_ms:.2f}ms) "
                    f"exceeded target ({self.config.prompt_construction_timeout_ms}ms) "
                    f"for request {request_id}"
                )
            
            # Return dict with all the info the caller needs
            return {
                "request": request,
                "request_id": request_id,
                "static_content_tokens": static_tokens,
                "dynamic_content_tokens": dynamic_tokens,
                "total_tokens": total_tokens,
                "construction_time_ms": construction_time_ms,
                "prompt_structure_valid": prompt_structure_valid
            }
    
    def build_optimized_prompt(
        self,
        dynamic_content: str
    ) -> str:
        """
        Build optimized prompt with static content first.
        
        This is a simpler interface for when you just need the prompt string.
        
        Args:
            dynamic_content: Dynamic content to append
            
        Returns:
            Optimized prompt string
        """
        start_time = time.perf_counter()
        
        template = self.content_loader.get_prompt_template()
        prompt = template.format(dynamic_content=dynamic_content)
        
        # Log performance timing
        construction_time_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Built optimized prompt (construction time: {construction_time_ms:.2f}ms)")
        
        # Alert if construction time exceeds target
        if construction_time_ms > self.config.prompt_construction_timeout_ms:
            logger.warning(
                f"Prompt construction time ({construction_time_ms:.2f}ms) "
                f"exceeded target ({self.config.prompt_construction_timeout_ms}ms)"
            )
        
        return prompt
    
    def process_response(
        self,
        response: Dict[str, Any],
        request_id: str
    ) -> CacheMetrics:
        """
        Process response and extract cache metrics.
        
        Args:
            response: Gemini API response
            request_id: Request tracking ID
            
        Returns:
            Cache effectiveness metrics
        """
        start_time = time.perf_counter()
        
        # Extract cache metrics from response
        usage_metadata = response.get("usage_metadata", {})
        cached_tokens = usage_metadata.get("cached_content_token_count", 0)
        total_tokens = usage_metadata.get("prompt_token_count", 0)
        dynamic_tokens = total_tokens - cached_tokens
        
        # Calculate costs
        cost_calc = self._calculate_costs(cached_tokens, dynamic_tokens, total_tokens)
        
        # Calculate cache effectiveness
        cache_effectiveness = (cached_tokens / total_tokens * 100) if total_tokens > 0 else 0
        
        metrics = CacheMetrics(
            cached_tokens=cached_tokens,
            dynamic_tokens=dynamic_tokens,
            total_tokens=total_tokens,
            cost_with_cache=cost_calc.total_cost_with_cache,
            cost_without_cache=cost_calc.total_cost_without_cache,
            savings=cost_calc.savings,
            cache_effectiveness=cache_effectiveness,
            timestamp=datetime.now(),
            request_id=request_id,
        )
        
        # Track metrics
        if self.config.monitor_cache_effectiveness:
            self.metrics_tracker.record_request(metrics)
        
        # Calculate processing duration
        processing_duration_ms = (time.perf_counter() - start_time) * 1000
        
        # Audit log response processing
        audit_logger = get_cache_audit_logger(self.config)
        audit_logger.log_cache_response_processed(
            metrics,
            request_id,
            processing_duration_ms,
            {
                "usage_metadata": usage_metadata,
                "response_has_candidates": "candidates" in response,
                "response_finish_reason": response.get("candidates", [{}])[0].get("finishReason") if response.get("candidates") else None
            }
        )
        
        # Audit log detailed cost calculation
        audit_logger.log_cost_calculation(
            cached_tokens,
            dynamic_tokens,
            cost_calc.total_cost_with_cache,
            cost_calc.total_cost_without_cache,
            cost_calc.savings,
            cost_calc.discount_percentage,
            request_id,
            {
                "cached_rate_per_1m": cost_calc.cached_rate_per_1m,
                "uncached_rate_per_1m": cost_calc.uncached_rate_per_1m,
                "cached_content_cost": cost_calc.cached_content_cost,
                "dynamic_content_cost": cost_calc.dynamic_content_cost
            }
        )
        
        logger.info(
            f"Request {request_id}: {cached_tokens} cached tokens, "
            f"${cost_calc.savings:.4f} saved ({cache_effectiveness:.1f}% cached)"
        )
        
        return metrics
    
    def calculate_cost(
        self,
        cached_tokens: int,
        dynamic_tokens: int
    ) -> CostCalculation:
        """
        Calculate costs for given token counts.
        
        Args:
            cached_tokens: Number of cached tokens
            dynamic_tokens: Number of uncached tokens
            
        Returns:
            Detailed cost calculation
        """
        total_tokens = cached_tokens + dynamic_tokens
        return self._calculate_costs(cached_tokens, dynamic_tokens, total_tokens)
    
    def _format_dynamic_content(
        self,
        ticket_data: Dict[str, Any],
        booking_data: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None
    ) -> str:
        """Format dynamic content for prompt."""
        sections = []
        
        # Format ticket data
        if ticket_data:
            sections.append("## Ticket Information")
            for key, value in ticket_data.items():
                sections.append(f"- {key}: {value}")
        
        # Format booking data
        if booking_data:
            sections.append("\n## Booking Information")
            for key, value in booking_data.items():
                sections.append(f"- {key}: {value}")
        
        # Add additional context
        if additional_context:
            sections.append(f"\n## Additional Context\n{additional_context}")
        
        return "\n".join(sections)
    
    def _calculate_costs(
        self,
        cached_tokens: int,
        dynamic_tokens: int,
        total_tokens: int
    ) -> CostCalculation:
        """Calculate detailed costs."""
        # Calculate costs with caching
        cached_cost = cached_tokens * self.config.cached_input_cost_per_1m / 1_000_000
        dynamic_cost = dynamic_tokens * self.config.input_cost_per_1m / 1_000_000
        total_cost_with_cache = cached_cost + dynamic_cost
        
        # Calculate what cost would be without caching
        total_cost_without_cache = total_tokens * self.config.input_cost_per_1m / 1_000_000
        
        # Calculate savings
        savings = total_cost_without_cache - total_cost_with_cache
        
        return CostCalculation(
            cached_tokens=cached_tokens,
            dynamic_tokens=dynamic_tokens,
            total_tokens=total_tokens,
            cached_content_cost=cached_cost,
            dynamic_content_cost=dynamic_cost,
            total_cost_with_cache=total_cost_with_cache,
            total_cost_without_cache=total_cost_without_cache,
            savings=savings,
            cached_rate_per_1m=self.config.cached_input_cost_per_1m,
            uncached_rate_per_1m=self.config.input_cost_per_1m,
            discount_percentage=self.config.get_discount_percentage(),
        )
    
    def _generate_request_id(self) -> str:
        """Generate unique request tracking ID."""
        return f"req_{uuid.uuid4().hex[:12]}"
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses a simple heuristic: ~4 characters per token.
        For production, use tiktoken or the actual tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Rough estimate: 4 characters per token
        return len(text) // 4
    
    def _validate_prompt_structure(self, prompt: str) -> bool:
        """
        Validate that prompt follows optimal structure for implicit caching.
        
        Args:
            prompt: Prompt to validate
            
        Returns:
            True if structure is valid for caching
        """
        try:
            # Check for required sections in correct order
            required_sections = [
                "# SYSTEM IDENTITY & ROLE",
                "# REFUND POLICY DOCUMENTS", 
                "# TOOL DEFINITIONS",
                "# FEW-SHOT EXAMPLES",
                "# CURRENT REQUEST"
            ]
            
            last_position = -1
            for section in required_sections:
                position = prompt.find(section)
                if position == -1:
                    # Missing section - still valid but not optimal
                    continue
                if position < last_position:
                    # Sections out of order - not optimal for caching
                    return False
                last_position = position
            
            # Check that dynamic content comes last
            dynamic_marker = "{dynamic_content}"
            if dynamic_marker in prompt:
                dynamic_pos = prompt.rfind(dynamic_marker)
                content_after_dynamic = prompt[dynamic_pos + len(dynamic_marker):].strip()
                if content_after_dynamic:
                    # Content after dynamic marker reduces cache effectiveness
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating prompt structure: {e}")
            return False
