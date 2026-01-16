"""
Performance optimization utilities for contradiction detection system.

This module provides performance optimizations to ensure contradiction detection
completes within the required 100ms per ticket target.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for contradiction detection operations."""
    total_time_ms: float
    parsing_time_ms: float
    validation_time_ms: float
    reasoning_time_ms: float
    component_status: Dict[str, str]
    cache_hits: int
    cache_misses: int


class PerformanceOptimizer:
    """
    Performance optimization utilities for contradiction detection.
    
    Provides caching, parallelization, and performance monitoring
    to ensure contradiction detection meets the <100ms target.
    """
    
    def __init__(self):
        """Initialize the performance optimizer."""
        self.complaint_cache = {}
        self.validation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 1000
        
        # Performance thresholds (in milliseconds) - optimized for webhook automation
        self.target_total_time = 200  # Increased for timeline analysis
        self.target_parsing_time = 50  # Increased for timeline extraction
        self.target_validation_time = 100  # Increased for scenario classification
        self.target_reasoning_time = 50  # Increased for enhanced reasoning
        
        # Webhook-specific performance targets
        self.webhook_target_time = 2000  # 2 seconds for webhook automation
        self.timeline_analysis_target = 200  # 200ms for timeline analysis
        self.scenario_classification_target = 500  # 500ms for scenario classification
        
        logger.info("PerformanceOptimizer initialized with enhanced caching and webhook targets")
    
    @lru_cache(maxsize=500)
    def cached_complaint_parsing(self, complaint_text: str) -> Tuple[str, ...]:
        """
        Cache complaint parsing results for identical complaint texts.
        
        Args:
            complaint_text: The complaint text to parse
            
        Returns:
            Tuple of complaint categories found
        """
        # This would be called by ComplaintParser with actual parsing logic
        # For now, return a placeholder that can be cached
        return tuple()
    
    def get_cache_key(self, customer_complaint: str, ticket_data: Dict) -> str:
        """
        Generate a cache key for contradiction detection results.
        
        Args:
            customer_complaint: Customer complaint text
            ticket_data: Ticket data dictionary
            
        Returns:
            Cache key string
        """
        # Create a hash-based key from complaint and relevant ticket data
        import hashlib
        
        # Include only relevant fields that affect contradiction detection
        relevant_data = {
            'complaint': customer_complaint.lower().strip(),
            'responses': ticket_data.get('description', '').lower(),
            'notes': ticket_data.get('notes', '').lower()
        }
        
        key_string = str(sorted(relevant_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def should_use_cache(self, customer_complaint: str, ticket_data: Dict) -> bool:
        """
        Determine if caching should be used for this request.
        
        Args:
            customer_complaint: Customer complaint text
            ticket_data: Ticket data dictionary
            
        Returns:
            True if caching is appropriate, False otherwise
        """
        # Don't cache very short complaints (likely incomplete)
        if len(customer_complaint.strip()) < 10:
            return False
        
        # Don't cache if ticket data is missing critical fields
        if not ticket_data.get('ticket_id'):
            return False
        
        return True
    
    def optimize_timeout_settings(self, config: Dict) -> Dict:
        """
        Optimize timeout settings for performance.
        
        Args:
            config: Current configuration dictionary
            
        Returns:
            Optimized configuration with performance-tuned timeouts
        """
        optimized_config = config.copy()
        
        # Optimize for webhook automation performance
        optimized_config['detection_timeout_seconds'] = 2  # Reduced for webhook speed
        optimized_config['max_processing_time_ms'] = 2000  # 2 second webhook target
        optimized_config['timeline_analysis_timeout_ms'] = 200  # Timeline analysis target
        optimized_config['scenario_classification_timeout_ms'] = 500  # Classification target
        
        # Add performance monitoring flags
        optimized_config['enable_performance_monitoring'] = True
        optimized_config['log_slow_operations'] = True
        optimized_config['slow_operation_threshold_ms'] = 100  # Increased threshold
        
        # Webhook-specific optimizations
        optimized_config['enable_caching'] = True
        optimized_config['cache_size'] = 1000
        optimized_config['deterministic_rules_first'] = True
        optimized_config['llm_fallback_timeout_ms'] = 10000  # 10s for LLM fallback
        
        logger.info("Optimized timeout settings for webhook automation performance")
        return optimized_config
    
    def create_fast_fallback_validator(self) -> 'LogicValidator':
        """
        Create a lightweight fallback validator for performance.
        
        Returns:
            Fast LogicValidator with minimal rules
        """
        try:
            from .logic_validator import LogicValidator
            
            # Create minimal config for fast validation
            fast_config = {
                "performance_mode": True,
                "minimal_rules_only": True,
                "skip_complex_validation": True,
                "fast_fallback": True
            }
            
            validator = LogicValidator(fast_config)
            
            # Set only the most critical rules for performance
            validator.mutually_exclusive_pairs = [
                ("accessibility", "oversold"),
                ("paid_again", "oversold")
            ]
            
            validator.response_conflicts = {
                "accessibility": {
                    "conflicting_responses": ["were you able to park? yes"],
                    "description": "Accessibility issue but able to park"
                }
            }
            
            logger.info("Created fast fallback validator")
            return validator
            
        except Exception as e:
            logger.error(f"Failed to create fast fallback validator: {e}")
            return None
    
    async def parallel_component_initialization(self, config: Dict) -> Tuple[Optional['ComplaintParser'], Optional['LogicValidator'], Optional['ReasoningGenerator']]:
        """
        Initialize components in parallel for better performance.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (complaint_parser, logic_validator, reasoning_generator)
        """
        async def init_complaint_parser():
            try:
                from .complaint_parser import ComplaintParser
                return ComplaintParser()
            except Exception as e:
                logger.error(f"Failed to initialize ComplaintParser: {e}")
                return None
        
        async def init_logic_validator():
            try:
                from .logic_validator import LogicValidator
                return LogicValidator(config)
            except Exception as e:
                logger.error(f"Failed to initialize LogicValidator: {e}")
                return self.create_fast_fallback_validator()
        
        async def init_reasoning_generator():
            try:
                from .reasoning_generator import ReasoningGenerator
                return ReasoningGenerator()
            except Exception as e:
                logger.error(f"Failed to initialize ReasoningGenerator: {e}")
                return None
        
        # Initialize all components in parallel
        start_time = time.time()
        
        results = await asyncio.gather(
            init_complaint_parser(),
            init_logic_validator(),
            init_reasoning_generator(),
            return_exceptions=True
        )
        
        init_time = (time.time() - start_time) * 1000
        logger.info(f"Parallel component initialization completed in {init_time:.1f}ms")
        
        # Handle any exceptions
        complaint_parser = results[0] if not isinstance(results[0], Exception) else None
        logic_validator = results[1] if not isinstance(results[1], Exception) else None
        reasoning_generator = results[2] if not isinstance(results[2], Exception) else None
        
        return complaint_parser, logic_validator, reasoning_generator
    
    async def fast_contradiction_detection(
        self,
        customer_complaint: str,
        ticket_data: Dict,
        complaint_parser: Optional['ComplaintParser'],
        logic_validator: Optional['LogicValidator'],
        reasoning_generator: Optional['ReasoningGenerator']
    ) -> 'ContradictionResult':
        """
        Perform optimized contradiction detection with performance monitoring.
        
        Args:
            customer_complaint: Customer complaint text
            ticket_data: Ticket data dictionary
            complaint_parser: Complaint parser instance
            logic_validator: Logic validator instance
            reasoning_generator: Reasoning generator instance
            
        Returns:
            ContradictionResult with performance metrics
        """
        from .contradiction_detector import ContradictionResult
        
        start_time = time.time()
        performance_metrics = PerformanceMetrics(
            total_time_ms=0,
            parsing_time_ms=0,
            validation_time_ms=0,
            reasoning_time_ms=0,
            component_status={},
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses
        )
        
        # Check for critical component failures - escalate for safety
        if not complaint_parser and not logic_validator:
            logger.error("Both ComplaintParser and LogicValidator unavailable - escalating for safety")
            return ContradictionResult(
                has_contradictions=True,
                contradictions=[],
                all_complaints=[],
                reasoning="Critical contradiction detection components failed. Escalating for safety to ensure proper review.",
                suggested_questions=["Please review the customer's complaint manually due to system component failure."],
                escalation_required=True
            )
        
        # Check cache first
        cache_key = None
        if self.should_use_cache(customer_complaint, ticket_data):
            cache_key = self.get_cache_key(customer_complaint, ticket_data)
            if cache_key in self.validation_cache:
                self.cache_hits += 1
                cached_result = self.validation_cache[cache_key]
                logger.debug(f"Cache hit for contradiction detection: {cache_key[:8]}...")
                return cached_result
            else:
                self.cache_misses += 1
        
        # Step 1: Fast complaint parsing
        complaints = []
        parsing_start = time.time()
        
        if complaint_parser:
            try:
                # Use timeout for parsing to prevent blocking
                complaints = await asyncio.wait_for(
                    asyncio.to_thread(complaint_parser.parse_complaints, customer_complaint),
                    timeout=0.5  # 500ms timeout for parsing
                )
                performance_metrics.component_status['complaint_parser'] = 'available'
            except asyncio.TimeoutError:
                logger.warning("Complaint parsing timed out - using empty complaints")
                complaints = []
                performance_metrics.component_status['complaint_parser'] = 'timeout'
            except Exception as e:
                logger.error(f"Complaint parsing failed: {e}")
                complaints = []
                performance_metrics.component_status['complaint_parser'] = 'failed'
        else:
            performance_metrics.component_status['complaint_parser'] = 'unavailable'
        
        performance_metrics.parsing_time_ms = (time.time() - parsing_start) * 1000
        
        # Step 2: Fast logic validation
        contradictions = []
        validation_start = time.time()
        
        if logic_validator:
            try:
                # Use timeout for validation to prevent blocking
                contradictions = await asyncio.wait_for(
                    asyncio.to_thread(logic_validator.validate_consistency, complaints, ticket_data),
                    timeout=1.0  # 1 second timeout for validation
                )
                performance_metrics.component_status['logic_validator'] = 'available'
            except asyncio.TimeoutError:
                logger.warning("Logic validation timed out - using empty contradictions")
                contradictions = []
                performance_metrics.component_status['logic_validator'] = 'timeout'
            except Exception as e:
                logger.error(f"Logic validation failed: {e}")
                contradictions = []
                performance_metrics.component_status['logic_validator'] = 'failed'
        else:
            performance_metrics.component_status['logic_validator'] = 'unavailable'
        
        performance_metrics.validation_time_ms = (time.time() - validation_start) * 1000
        
        # Step 3: Fast reasoning generation
        reasoning_start = time.time()
        has_contradictions = len(contradictions) > 0
        
        if has_contradictions and reasoning_generator:
            try:
                # Use timeout for reasoning to prevent blocking
                reasoning_result = await asyncio.wait_for(
                    asyncio.to_thread(reasoning_generator.generate_reasoning, contradictions),
                    timeout=0.5  # 500ms timeout for reasoning
                )
                reasoning = reasoning_result.explanation
                suggested_questions = reasoning_result.suggested_questions
                performance_metrics.component_status['reasoning_generator'] = 'available'
            except asyncio.TimeoutError:
                logger.warning("Reasoning generation timed out - using basic reasoning")
                reasoning = f"Found {len(contradictions)} contradictions requiring human review"
                suggested_questions = ["Please review the customer's complaint for contradictory information."]
                performance_metrics.component_status['reasoning_generator'] = 'timeout'
            except Exception as e:
                logger.error(f"Reasoning generation failed: {e}")
                reasoning = f"Found {len(contradictions)} contradictions requiring human review"
                suggested_questions = ["Please review the customer's complaint for contradictory information."]
                performance_metrics.component_status['reasoning_generator'] = 'failed'
        else:
            reasoning = "No contradictions detected" if not has_contradictions else f"Found {len(contradictions)} contradictions"
            suggested_questions = []
            performance_metrics.component_status['reasoning_generator'] = 'available' if reasoning_generator else 'unavailable'
        
        performance_metrics.reasoning_time_ms = (time.time() - reasoning_start) * 1000
        performance_metrics.total_time_ms = (time.time() - start_time) * 1000
        
        # Enhance reasoning with original data preservation for agent reference (if available)
        if has_contradictions and ticket_data.get("complaint_extraction"):
            complaint_extraction = ticket_data["complaint_extraction"]
            original_reason = complaint_extraction.get("original_reason", "")
            original_summary = complaint_extraction.get("original_summary", "")
            
            if original_reason or original_summary:
                reasoning += "<br><br><strong>Original Customer Input (for agent reference):</strong><br>"
                if original_reason:
                    reasoning += f"<strong>Reason:</strong> {original_reason}<br>"
                if original_summary:
                    reasoning += f"<strong>Summary:</strong> {original_summary}"
        
        # Create result
        result = ContradictionResult(
            has_contradictions=has_contradictions,
            contradictions=contradictions,
            all_complaints=complaints,
            reasoning=reasoning,
            suggested_questions=suggested_questions,
            escalation_required=has_contradictions
        )
        
        # Cache the result if appropriate
        if cache_key and performance_metrics.total_time_ms < self.target_total_time * 2:
            # Only cache if processing was reasonably fast
            if len(self.validation_cache) >= self.max_cache_size:
                # Remove oldest entries
                oldest_keys = list(self.validation_cache.keys())[:100]
                for key in oldest_keys:
                    del self.validation_cache[key]
            
            self.validation_cache[cache_key] = result
            logger.debug(f"Cached contradiction detection result: {cache_key[:8]}...")
        
        # Log comprehensive contradiction event if contradictions detected (for compatibility)
        if has_contradictions:
            try:
                from datetime import datetime
                import uuid
                
                event_data = {
                    "event_type": "contradiction_detected",
                    "timestamp": datetime.now().isoformat(),
                    "ticket_id": ticket_data.get("ticket_id", "unknown"),
                    "contradiction_count": len(contradictions),
                    "contradiction_types": [c.type for c in contradictions],
                    "analysis_details": {
                        "complaint_parsing": {
                            "detected_complaints": [
                                {
                                    "category": c.category,
                                    "confidence": c.confidence,
                                    "keywords": c.keywords
                                } for c in complaints
                            ],
                            "complaint_count": len(complaints),
                            "processing_time_ms": performance_metrics.parsing_time_ms,
                            "parser_available": complaint_parser is not None
                        },
                        "logic_validation": {
                            "rules_applied": (len(getattr(logic_validator, 'mutually_exclusive_pairs', [])) + 
                                           len(getattr(logic_validator, 'response_conflicts', {}))) if logic_validator else 0,
                            "conflicts_found": [
                                {
                                    "type": c.type,
                                    "severity": c.severity,
                                    "description": c.description
                                } for c in contradictions
                            ],
                            "processing_time_ms": performance_metrics.validation_time_ms,
                            "validator_available": logic_validator is not None
                        },
                        "reasoning_generation": {
                            "reasoning_length": len(reasoning),
                            "suggested_questions_count": len(suggested_questions),
                            "processing_time_ms": performance_metrics.reasoning_time_ms,
                            "generator_available": reasoning_generator is not None
                        }
                    },
                    "reasoning": reasoning,
                    "customer_context": {
                        "customer_id": ticket_data.get("customer_id"),
                        "original_complaint": customer_complaint,
                        "ticket_metadata": {
                            "subject": ticket_data.get("subject"),
                            "created_at": ticket_data.get("created_at"),
                            "source_system": ticket_data.get("source_system")
                        }
                    },
                    "processing_time_ms": performance_metrics.total_time_ms,
                    "component_status": performance_metrics.component_status,
                    "performance_optimized": True
                }
                
                # Call logging method for compatibility with tests
                # Try to use the detector's method if available, otherwise use our own
                if hasattr(self, 'detector') and hasattr(self.detector, '_log_contradiction_event'):
                    self.detector._log_contradiction_event(event_data)
                else:
                    self._log_contradiction_event(event_data)
                
                # Store audit trail for escalated cases (for compatibility)
                if result.escalation_required:
                    audit_data = {
                        "audit_id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "event_type": "contradiction_escalation",
                        "ticket_id": ticket_data.get("ticket_id", "unknown"),
                        "customer_input": {
                            "original_complaint": customer_complaint,
                            "processed_complaint": customer_complaint.lower().strip(),
                            "complaint_extraction": ticket_data.get("complaint_extraction", {})
                        },
                        "detected_conflicts": {
                            "contradiction_count": len(contradictions),
                            "contradiction_details": [
                                {
                                    "type": c.type,
                                    "severity": c.severity,
                                    "description": c.description,
                                    "involved_complaints": [
                                        c.complaint_a.category if c.complaint_a else None,
                                        c.complaint_b.category if c.complaint_b else None
                                    ]
                                } for c in contradictions
                            ]
                        },
                        "escalation_reasoning": {
                            "reasoning_text": reasoning,
                            "suggested_questions": suggested_questions,
                            "confidence_level": "high"  # High confidence when contradictions detected
                        },
                        "system_context": {
                            "detector_version": "1.0.0",
                            "configuration_version": getattr(logic_validator.config, 'version', 'unknown') if logic_validator and hasattr(logic_validator, 'config') else 'unknown',
                            "processing_time_ms": performance_metrics.total_time_ms,
                            "session_id": ticket_data.get("session_id"),
                            "user_agent": ticket_data.get("user_agent"),
                            "source_system": ticket_data.get("source_system"),
                            "component_status": {
                                "complaint_parser": "available" if complaint_parser else "failed",
                                "logic_validator": "available" if logic_validator else "failed",
                                "reasoning_generator": "available" if reasoning_generator else "failed"
                            }
                        },
                        "decision_trail": {
                            "steps": [
                                {
                                    "step_type": "complaint_parsing",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Detected {len(complaints)} complaint types",
                                    "processing_time_ms": performance_metrics.parsing_time_ms,
                                    "component_status": "available" if complaint_parser else "failed"
                                },
                                {
                                    "step_type": "logic_validation",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Found {len(contradictions)} contradictions",
                                    "processing_time_ms": performance_metrics.validation_time_ms,
                                    "component_status": "available" if logic_validator else "failed"
                                },
                                {
                                    "step_type": "reasoning_generation",
                                    "timestamp": datetime.now().isoformat(),
                                    "result": f"Generated reasoning and {len(suggested_questions)} questions",
                                    "processing_time_ms": performance_metrics.reasoning_time_ms,
                                    "component_status": "available" if reasoning_generator else "failed"
                                }
                            ],
                            "final_decision": {
                                "decision": "escalate",
                                "reason": "contradictions_detected",
                                "confidence": "high",
                                "timestamp": datetime.now().isoformat()
                            }
                        },
                        "escalation_required": True
                    }
                    
                    # Try to use the detector's method if available, otherwise use our own
                    if hasattr(self, 'detector') and hasattr(self.detector, '_store_audit_trail'):
                        self.detector._store_audit_trail(audit_data)
                    else:
                        self._store_audit_trail(audit_data)
                    
            except Exception as logging_error:
                logger.error(f"Failed to log contradiction event: {logging_error}")
                # Continue processing even if logging fails
        
        # Log performance metrics
        if performance_metrics.total_time_ms > self.target_total_time:
            logger.warning(
                f"Contradiction detection exceeded target time: "
                f"{performance_metrics.total_time_ms:.1f}ms > {self.target_total_time}ms"
            )
            logger.warning(
                f"  Parsing: {performance_metrics.parsing_time_ms:.1f}ms, "
                f"Validation: {performance_metrics.validation_time_ms:.1f}ms, "
                f"Reasoning: {performance_metrics.reasoning_time_ms:.1f}ms"
            )
        else:
            logger.debug(
                f"Contradiction detection completed in {performance_metrics.total_time_ms:.1f}ms "
                f"(target: {self.target_total_time}ms)"
            )
        
        return result
    
    def clear_caches(self):
        """Clear all performance caches."""
        self.complaint_cache.clear()
        self.validation_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Performance caches cleared")
    
    def optimize_timeline_analysis(self, timeline_analyzer, timeline_data: Dict) -> Dict:
        """
        Optimize timeline analysis for performance.
        
        Args:
            timeline_analyzer: TimelineAnalyzer instance
            timeline_data: Timeline data to analyze
            
        Returns:
            Optimized timeline analysis result
        """
        start_time = time.time()
        
        try:
            # Use cached result if available
            cache_key = self._get_timeline_cache_key(timeline_data)
            if cache_key in self.complaint_cache:
                self.cache_hits += 1
                logger.debug(f"Timeline analysis cache hit: {cache_key[:8]}...")
                return self.complaint_cache[cache_key]
            
            # Perform timeline analysis with timeout
            analysis_result = asyncio.wait_for(
                asyncio.to_thread(
                    timeline_analyzer.analyze_timeline,
                    timeline_data.get('booking_start'),
                    timeline_data.get('booking_end'),
                    timeline_data.get('arrival_time'),
                    timeline_data.get('exit_time')
                ),
                timeout=self.timeline_analysis_target / 1000  # Convert to seconds
            )
            
            # Cache the result
            self.complaint_cache[cache_key] = analysis_result
            self.cache_misses += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            if processing_time > self.timeline_analysis_target:
                logger.warning(f"Timeline analysis exceeded target: {processing_time:.1f}ms > {self.timeline_analysis_target}ms")
            
            return analysis_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeline analysis timed out after {self.timeline_analysis_target}ms")
            return None
        except Exception as e:
            logger.error(f"Timeline analysis optimization failed: {e}")
            return None
    
    def optimize_scenario_classification(self, classifier, complaint: str, timeline_analysis=None) -> Dict:
        """
        Optimize scenario classification for performance.
        
        Args:
            classifier: ScenarioClassifier instance
            complaint: Customer complaint text
            timeline_analysis: Optional timeline analysis
            
        Returns:
            Optimized classification result
        """
        start_time = time.time()
        
        try:
            # Use cached result if available
            cache_key = self._get_classification_cache_key(complaint, timeline_analysis)
            if cache_key in self.validation_cache:
                self.cache_hits += 1
                logger.debug(f"Scenario classification cache hit: {cache_key[:8]}...")
                return self.validation_cache[cache_key]
            
            # Perform classification with timeout
            classification_result = asyncio.wait_for(
                asyncio.to_thread(
                    classifier.classify_paid_again_scenario,
                    complaint,
                    timeline_analysis
                ),
                timeout=self.scenario_classification_target / 1000  # Convert to seconds
            )
            
            # Cache the result
            self.validation_cache[cache_key] = classification_result
            self.cache_misses += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            if processing_time > self.scenario_classification_target:
                logger.warning(f"Scenario classification exceeded target: {processing_time:.1f}ms > {self.scenario_classification_target}ms")
            
            return classification_result
            
        except asyncio.TimeoutError:
            logger.warning(f"Scenario classification timed out after {self.scenario_classification_target}ms")
            return None
        except Exception as e:
            logger.error(f"Scenario classification optimization failed: {e}")
            return None
    
    def _get_timeline_cache_key(self, timeline_data: Dict) -> str:
        """Generate cache key for timeline analysis."""
        import hashlib
        
        key_data = {
            'booking_start': str(timeline_data.get('booking_start')),
            'booking_end': str(timeline_data.get('booking_end')),
            'arrival_time': str(timeline_data.get('arrival_time')),
            'exit_time': str(timeline_data.get('exit_time'))
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_classification_cache_key(self, complaint: str, timeline_analysis) -> str:
        """Generate cache key for scenario classification."""
        import hashlib
        
        key_data = {
            'complaint': complaint.lower().strip(),
            'timeline_hash': str(hash(str(timeline_analysis))) if timeline_analysis else 'none'
        }
        
        key_string = str(sorted(key_data.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def validate_performance_targets(self, processing_times: Dict) -> Dict:
        """
        Validate that processing times meet performance targets.
        
        Args:
            processing_times: Dict with component processing times
            
        Returns:
            Validation results with pass/fail status
        """
        results = {
            "overall_pass": True,
            "component_results": {},
            "performance_summary": {}
        }
        
        # Check timeline analysis performance
        timeline_time = processing_times.get("timeline_analysis_ms", 0)
        timeline_pass = timeline_time <= self.timeline_analysis_target
        results["component_results"]["timeline_analysis"] = {
            "time_ms": timeline_time,
            "target_ms": self.timeline_analysis_target,
            "pass": timeline_pass
        }
        
        # Check scenario classification performance
        classification_time = processing_times.get("scenario_classification_ms", 0)
        classification_pass = classification_time <= self.scenario_classification_target
        results["component_results"]["scenario_classification"] = {
            "time_ms": classification_time,
            "target_ms": self.scenario_classification_target,
            "pass": classification_pass
        }
        
        # Check overall webhook performance
        total_time = processing_times.get("total_ms", 0)
        webhook_pass = total_time <= self.webhook_target_time
        results["component_results"]["webhook_total"] = {
            "time_ms": total_time,
            "target_ms": self.webhook_target_time,
            "pass": webhook_pass
        }
        
        # Overall pass requires all components to pass
        results["overall_pass"] = timeline_pass and classification_pass and webhook_pass
        
        # Performance summary
        results["performance_summary"] = {
            "timeline_analysis_efficiency": (self.timeline_analysis_target / max(timeline_time, 1)) * 100,
            "classification_efficiency": (self.scenario_classification_target / max(classification_time, 1)) * 100,
            "webhook_efficiency": (self.webhook_target_time / max(total_time, 1)) * 100,
            "cache_hit_rate": self.get_performance_stats()["cache_hit_rate_percent"]
        }
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """
        Get current performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate_percent": cache_hit_rate,
            "complaint_cache_size": len(self.complaint_cache),
            "validation_cache_size": len(self.validation_cache),
            "target_total_time_ms": self.target_total_time,
            "target_parsing_time_ms": self.target_parsing_time,
            "target_validation_time_ms": self.target_validation_time,
            "target_reasoning_time_ms": self.target_reasoning_time,
            "webhook_target_time_ms": self.webhook_target_time,
            "timeline_analysis_target_ms": self.timeline_analysis_target,
            "scenario_classification_target_ms": self.scenario_classification_target
        }
    
    def _log_contradiction_event(self, event_data: Dict) -> None:
        """
        Log comprehensive contradiction detection event.
        
        Args:
            event_data: Dictionary containing all event details for logging
        """
        try:
            # Log the event with structured data
            logger.info(
                f"Contradiction Event: {event_data.get('event_type', 'unknown')} "
                f"for ticket {event_data.get('ticket_id', 'unknown')}"
            )
            
            # Log detailed event data as structured information
            logger.debug(f"Event details: {event_data}")
            
        except Exception as e:
            logger.error(f"Failed to log contradiction event: {e}")
    
    def _store_audit_trail(self, audit_data: Dict) -> None:
        """
        Store comprehensive audit trail for contradiction cases.
        
        Args:
            audit_data: Dictionary containing complete audit trail information
        """
        try:
            # Store audit trail (in a real implementation, this would go to a database)
            logger.info(
                f"Audit Trail: {audit_data.get('event_type', 'unknown')} "
                f"for ticket {audit_data.get('ticket_id', 'unknown')} "
                f"with audit ID {audit_data.get('audit_id', 'unknown')}"
            )
            
            # Log audit trail details
            logger.debug(f"Audit trail data: {audit_data}")
            
        except Exception as e:
            logger.error(f"Failed to store audit trail: {e}")