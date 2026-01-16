"""
WebhookModeManager component for handling webhook vs chat mode differences.

This module provides the WebhookModeManager class that manages processing mode
differences, confidence thresholds, and escalation logic for webhook automation
versus interactive chat processing.
"""

import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass
from .classification_config import ClassificationConfigManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessingModeConfig:
    """Configuration for processing mode settings."""
    mode: str  # "webhook_automation" or "interactive_chat"
    confidence_threshold: float
    escalation_enabled: bool
    llm_fallback_enabled: bool
    timeout_seconds: int


class WebhookModeManager:
    """
    Manages webhook-specific processing logic with higher confidence thresholds.
    
    This class handles the differences between webhook automation (autonomous decisions)
    and interactive chat mode (human-assisted decisions), applying stricter confidence
    thresholds and escalation logic for webhook processing.
    """
    
    def __init__(self, config: Optional[Dict] = None, config_manager: Optional[ClassificationConfigManager] = None):
        """
        Initialize WebhookModeManager with configuration.
        
        Args:
            config: Configuration dict with mode settings (legacy support)
            config_manager: ClassificationConfigManager instance for configurable thresholds
        """
        if config is None:
            config = {}
        
        # Initialize configuration manager
        if config_manager is None:
            config_path = config.get("config_path")
            self.config_manager = ClassificationConfigManager(config_path)
        else:
            self.config_manager = config_manager
        
        # Load confidence thresholds from configuration
        # Priority: passed config > configuration file > defaults
        try:
            classification_config = self.config_manager.load_config()
            file_webhook_threshold = classification_config.confidence_thresholds.webhook_automation
            file_chat_threshold = classification_config.confidence_thresholds.interactive_chat
            logger.info("Loaded confidence thresholds from configuration file")
        except Exception as e:
            logger.warning(f"Failed to load confidence thresholds from config: {e}. Using defaults.")
            file_webhook_threshold = 0.85
            file_chat_threshold = 0.70
        
        # Allow passed config to override file config
        self.webhook_confidence_threshold = config.get("webhook_confidence_threshold", file_webhook_threshold)
        self.chat_confidence_threshold = config.get("chat_confidence_threshold", file_chat_threshold)
        
        # Processing mode settings
        self.webhook_timeout_seconds = config.get("webhook_timeout_seconds", 30)
        self.chat_timeout_seconds = config.get("chat_timeout_seconds", 60)
        
        # Escalation settings
        self.webhook_escalation_enabled = config.get("webhook_escalation_enabled", True)
        self.chat_llm_fallback_enabled = config.get("chat_llm_fallback_enabled", True)
        
        # Current processing mode (default to chat for backward compatibility)
        self.current_mode = config.get("default_mode", "interactive_chat")
        
        logger.info(f"WebhookModeManager initialized: webhook_threshold={self.webhook_confidence_threshold}, "
                   f"chat_threshold={self.chat_confidence_threshold}, default_mode={self.current_mode}")
    
    def set_processing_mode(self, mode: str) -> None:
        """
        Set the current processing mode.
        
        Args:
            mode: Processing mode ("webhook_automation" or "interactive_chat")
        """
        if mode not in ["webhook_automation", "interactive_chat"]:
            raise ValueError(f"Invalid processing mode: {mode}. Must be 'webhook_automation' or 'interactive_chat'")
        
        self.current_mode = mode
        logger.info(f"Processing mode set to: {mode}")
    
    def get_processing_config(self, mode: Optional[str] = None) -> ProcessingModeConfig:
        """
        Get processing configuration for the specified mode.
        
        Args:
            mode: Processing mode (uses current mode if None)
            
        Returns:
            ProcessingModeConfig with mode-specific settings
        """
        if mode is None:
            mode = self.current_mode
        
        if mode == "webhook_automation":
            return ProcessingModeConfig(
                mode=mode,
                confidence_threshold=self.webhook_confidence_threshold,
                escalation_enabled=self.webhook_escalation_enabled,
                llm_fallback_enabled=False,  # Webhook mode prefers escalation over LLM
                timeout_seconds=self.webhook_timeout_seconds
            )
        else:  # interactive_chat
            return ProcessingModeConfig(
                mode=mode,
                confidence_threshold=self.chat_confidence_threshold,
                escalation_enabled=True,  # Chat mode can escalate too
                llm_fallback_enabled=self.chat_llm_fallback_enabled,
                timeout_seconds=self.chat_timeout_seconds
            )
    
    def should_escalate(
        self,
        confidence: float,
        processing_mode: Optional[str] = None,
        scenario_type: Optional[str] = None,
        additional_factors: Optional[Dict] = None
    ) -> bool:
        """
        Determine if case should be escalated based on confidence and mode.
        
        Args:
            confidence: Classification confidence (0.0 to 1.0)
            processing_mode: Processing mode (uses current mode if None)
            scenario_type: Type of scenario being classified
            additional_factors: Additional factors that may influence escalation
            
        Returns:
            True if escalation required, False if autonomous decision acceptable
        """
        if processing_mode is None:
            processing_mode = self.current_mode
        
        config = self.get_processing_config(processing_mode)
        
        # Basic confidence threshold check
        if confidence < config.confidence_threshold:
            logger.info(f"Escalating due to low confidence: {confidence} < {config.confidence_threshold} "
                       f"(mode: {processing_mode})")
            return True
        
        # Webhook mode has additional escalation criteria
        if processing_mode == "webhook_automation":
            # Always escalate contradictory statements in webhook mode
            if scenario_type == "contradictory_statements":
                logger.info("Escalating contradictory statements in webhook mode")
                return True
            
            # Escalate uncertain scenarios in webhook mode (no LLM fallback)
            if scenario_type == "uncertain_requires_analysis":
                logger.info("Escalating uncertain scenario in webhook mode (no LLM fallback)")
                return True
            
            # Check for additional webhook-specific escalation factors
            if additional_factors:
                # Escalate if multiple scenarios could apply
                if additional_factors.get("multiple_scenarios_detected"):
                    logger.info("Escalating due to multiple scenario detection in webhook mode")
                    return True
                
                # Escalate if timeline data is inconsistent
                if additional_factors.get("timeline_inconsistent"):
                    logger.info("Escalating due to timeline inconsistency in webhook mode")
                    return True
                
                # Escalate if booking amount exceeds automation limit
                if additional_factors.get("amount_exceeds_limit"):
                    logger.info("Escalating due to amount limit in webhook mode")
                    return True
        
        # No escalation required
        logger.debug(f"No escalation required: confidence={confidence}, mode={processing_mode}")
        return False
    
    def get_escalation_reason(
        self,
        confidence: float,
        processing_mode: Optional[str] = None,
        scenario_type: Optional[str] = None,
        additional_factors: Optional[Dict] = None
    ) -> str:
        """
        Get detailed reason for escalation.
        
        Args:
            confidence: Classification confidence
            processing_mode: Processing mode
            scenario_type: Type of scenario
            additional_factors: Additional escalation factors
            
        Returns:
            Human-readable escalation reason
        """
        if processing_mode is None:
            processing_mode = self.current_mode
        
        config = self.get_processing_config(processing_mode)
        reasons = []
        
        # Confidence threshold reason
        if confidence < config.confidence_threshold:
            reasons.append(
                f"Classification confidence ({confidence:.2f}) below {processing_mode} "
                f"threshold ({config.confidence_threshold:.2f})"
            )
        
        # Scenario-specific reasons
        if scenario_type == "contradictory_statements":
            reasons.append("Customer statements contain logical contradictions requiring human review")
        
        if scenario_type == "uncertain_requires_analysis" and processing_mode == "webhook_automation":
            reasons.append("Uncertain scenario requires human analysis (LLM fallback disabled in webhook mode)")
        
        # Additional factor reasons
        if additional_factors:
            if additional_factors.get("multiple_scenarios_detected"):
                reasons.append("Multiple scenario patterns detected - requires human disambiguation")
            
            if additional_factors.get("timeline_inconsistent"):
                reasons.append("Timeline data contains inconsistencies requiring human verification")
            
            if additional_factors.get("amount_exceeds_limit"):
                amount = additional_factors.get("booking_amount", "unknown")
                reasons.append(f"Refund amount (${amount}) exceeds automation limit ($50)")
        
        if not reasons:
            reasons.append("Escalation criteria met but specific reason not identified")
        
        return "; ".join(reasons)
    
    def validate_scenario_consistency(
        self,
        primary_scenario: str,
        alternative_scenarios: list,
        confidence_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate consistency between multiple scenario classifications.
        
        Args:
            primary_scenario: The highest confidence scenario
            alternative_scenarios: List of other possible scenarios
            confidence_scores: Dict mapping scenario names to confidence scores
            
        Returns:
            Dict with consistency validation results
        """
        validation_result = {
            "is_consistent": True,
            "conflicts_detected": [],
            "confidence_gap": 0.0,
            "requires_escalation": False,
            "reasoning": ""
        }
        
        if not alternative_scenarios:
            validation_result["reasoning"] = "Single scenario detected - no conflicts possible"
            return validation_result
        
        primary_confidence = confidence_scores.get(primary_scenario, 0.0)
        
        # Check for competing high-confidence scenarios
        competing_scenarios = []
        for scenario in alternative_scenarios:
            alt_confidence = confidence_scores.get(scenario, 0.0)
            confidence_gap = primary_confidence - alt_confidence
            
            # If alternative scenario has confidence within 0.2 of primary, it's competing
            if confidence_gap < 0.2 and alt_confidence > 0.6:
                competing_scenarios.append({
                    "scenario": scenario,
                    "confidence": alt_confidence,
                    "gap": confidence_gap
                })
        
        if competing_scenarios:
            validation_result["is_consistent"] = False
            validation_result["conflicts_detected"] = competing_scenarios
            validation_result["confidence_gap"] = min(c["gap"] for c in competing_scenarios)
            
            # In webhook mode, escalate competing scenarios
            if self.current_mode == "webhook_automation":
                validation_result["requires_escalation"] = True
                validation_result["reasoning"] = (
                    f"Multiple scenarios have similar confidence: {primary_scenario} "
                    f"({primary_confidence:.2f}) vs competing scenarios. "
                    f"Webhook mode requires clear classification for autonomous decisions."
                )
            else:
                validation_result["reasoning"] = (
                    f"Multiple scenarios detected with similar confidence. "
                    f"LLM analysis may be needed to disambiguate."
                )
        else:
            validation_result["reasoning"] = f"Clear primary scenario: {primary_scenario} ({primary_confidence:.2f})"
        
        return validation_result
    
    def get_confidence_threshold(self, processing_mode: Optional[str] = None) -> float:
        """
        Get confidence threshold for the specified processing mode.
        
        Args:
            processing_mode: Processing mode (uses current mode if None)
            
        Returns:
            Confidence threshold for the mode
        """
        if processing_mode is None:
            processing_mode = self.current_mode
        
        config = self.get_processing_config(processing_mode)
        return config.confidence_threshold
    
    def is_webhook_mode(self) -> bool:
        """
        Check if currently in webhook automation mode.
        
        Returns:
            True if in webhook mode, False otherwise
        """
        return self.current_mode == "webhook_automation"
    
    def is_chat_mode(self) -> bool:
        """
        Check if currently in interactive chat mode.
        
        Returns:
            True if in chat mode, False otherwise
        """
        return self.current_mode == "interactive_chat"
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of current processing configuration.
        
        Returns:
            Dict with processing mode summary
        """
        config = self.get_processing_config()
        
        return {
            "current_mode": self.current_mode,
            "confidence_threshold": config.confidence_threshold,
            "escalation_enabled": config.escalation_enabled,
            "llm_fallback_enabled": config.llm_fallback_enabled,
            "timeout_seconds": config.timeout_seconds,
            "webhook_threshold": self.webhook_confidence_threshold,
            "chat_threshold": self.chat_confidence_threshold
        }