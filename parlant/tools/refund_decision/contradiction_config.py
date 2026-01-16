"""
Configuration system for contradiction detection rules.

This module provides configuration management for defining contradiction patterns,
supporting external configuration files, and validating rule consistency.
"""

import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import os

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class ContradictionRule:
    """Represents a single contradiction rule."""
    id: str
    name: str
    type: str  # "mutually_exclusive", "response_mismatch", "logical_conflict"
    complaint_a: str
    complaint_b: Optional[str] = None
    conflicting_responses: Optional[List[str]] = None
    severity: str = "medium"  # "high", "medium", "low"
    description: str = ""
    enabled: bool = True
    priority: int = 100  # Lower numbers = higher priority


@dataclass
class ContradictionConfig:
    """Configuration container for all contradiction rules."""
    version: str
    rules: List[ContradictionRule]
    global_settings: Dict[str, Any]


class ContradictionConfigManager:
    """
    Manages configuration for contradiction detection rules.
    
    Supports loading from external configuration files, validating rule consistency,
    and preventing circular dependencies.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to external configuration file
        """
        self.config_path = config_path
        self.config: Optional[ContradictionConfig] = None
        self._default_config = self._create_default_config()
        logger.info(f"ContradictionConfigManager initialized with config_path: {config_path}")
    
    def load_config(self) -> ContradictionConfig:
        """
        Load contradiction rules configuration.
        
        Attempts to load from external file first, falls back to default configuration
        if file loading fails or no path is provided. Implements comprehensive error
        handling with graceful degradation to basic patterns.
        
        Returns:
            ContradictionConfig with loaded rules
        
        Raises:
            ValueError: If configuration validation fails after all fallback attempts
        """
        logger.info("Loading contradiction rules configuration")
        
        config = None
        config_source = "unknown"
        
        # Try to load from external file first
        if self.config_path and os.path.exists(self.config_path):
            try:
                config = self._load_from_file(self.config_path)
                config_source = f"external_file:{self.config_path}"
                logger.info(f"Loaded configuration from file: {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                logger.info("Falling back to default configuration due to file loading error")
        elif self.config_path:
            logger.warning(f"Configuration file not found: {self.config_path}")
            logger.info("Falling back to default configuration due to missing file")
        
        # Fall back to default configuration if file loading failed
        if config is None:
            try:
                config = self._default_config
                config_source = "default_configuration"
                logger.info("Using default configuration")
            except Exception as e:
                logger.error(f"Failed to create default configuration: {e}")
                logger.info("Falling back to minimal basic configuration")
                config = self._create_minimal_config()
                config_source = "minimal_fallback"
        
        # Validate configuration with error handling
        try:
            self._validate_config(config)
            logger.info(f"Configuration validation passed (source: {config_source})")
        except Exception as validation_error:
            logger.error(f"Configuration validation failed: {validation_error}")
            
            # Try to create a minimal working configuration
            try:
                logger.info("Attempting to create minimal working configuration")
                config = self._create_minimal_config()
                self._validate_config(config)
                config_source = "minimal_validated_fallback"
                logger.warning("Using minimal configuration due to validation failure")
            except Exception as minimal_error:
                logger.error(f"Even minimal configuration failed: {minimal_error}")
                # Last resort: create absolutely basic config
                config = self._create_emergency_config()
                config_source = "emergency_fallback"
                logger.error("Using emergency fallback configuration - functionality will be limited")
        
        # Store configuration with metadata about its source
        self.config = config
        self.config.source = config_source  # Add source tracking
        
        logger.info(f"Configuration loaded successfully from: {config_source}")
        return config
    
    def _create_minimal_config(self) -> ContradictionConfig:
        """
        Create minimal contradiction rules configuration with only essential rules.
        
        Returns:
            Minimal ContradictionConfig with basic rules
        """
        minimal_rules = [
            # Only the most critical mutually exclusive rules
            ContradictionRule(
                id="accessibility_vs_oversold_minimal",
                name="Accessibility vs Oversold Conflict",
                type="mutually_exclusive",
                complaint_a="accessibility",
                complaint_b="oversold",
                severity="high",
                description="Customer claims accessibility issues but also reports facility was full",
                priority=10
            ),
            
            # Only the most critical response mismatch rule
            ContradictionRule(
                id="accessibility_parking_success_minimal",
                name="Accessibility vs Parking Success",
                type="response_mismatch",
                complaint_a="accessibility",
                conflicting_responses=["were you able to park? yes", "were you able to park?yes"],
                severity="high",
                description="Customer claims accessibility issues but confirms they were able to park",
                priority=20
            )
        ]
        
        minimal_settings = {
            "enable_contradiction_detection": True,
            "escalation_threshold": "high",  # Only escalate high-severity issues
            "max_rules_per_type": 10,
            "validation_strict_mode": False  # Relaxed validation for minimal config
        }
        
        return ContradictionConfig(
            version="1.0-minimal",
            rules=minimal_rules,
            global_settings=minimal_settings
        )
    
    def _create_emergency_config(self) -> ContradictionConfig:
        """
        Create emergency fallback configuration with no rules (detection disabled).
        
        Returns:
            Emergency ContradictionConfig that disables detection
        """
        emergency_settings = {
            "enable_contradiction_detection": False,  # Disable detection entirely
            "escalation_threshold": "high",
            "max_rules_per_type": 0,
            "validation_strict_mode": False
        }
        
        return ContradictionConfig(
            version="1.0-emergency",
            rules=[],  # No rules - detection will be disabled
            global_settings=emergency_settings
        )
    
    def save_config(self, config: ContradictionConfig, path: Optional[str] = None) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Optional path to save to (uses self.config_path if not provided)
        
        Raises:
            ValueError: If configuration validation fails
            IOError: If file cannot be written
        """
        # Validate before saving
        self._validate_config(config)
        
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No save path provided and no config_path set")
        
        try:
            # Convert to dictionary format for JSON serialization
            config_dict = {
                "version": config.version,
                "global_settings": config.global_settings,
                "rules": [asdict(rule) for rule in config.rules]
            }
            
            # Ensure directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file with pretty formatting
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise IOError(f"Cannot save configuration: {e}")
    
    def get_rules_by_type(self, rule_type: str) -> List[ContradictionRule]:
        """
        Get all enabled rules of a specific type.
        
        Args:
            rule_type: Type of rules to retrieve
        
        Returns:
            List of enabled rules of the specified type, sorted by priority
        """
        if not self.config:
            self.load_config()
        
        rules = [
            rule for rule in self.config.rules 
            if rule.type == rule_type and rule.enabled
        ]
        
        # Sort by priority (lower numbers first)
        rules.sort(key=lambda r: r.priority)
        
        return rules
    
    def get_mutually_exclusive_pairs(self) -> List[Tuple[str, str]]:
        """
        Get all mutually exclusive complaint pairs from configuration.
        
        Returns:
            List of tuples representing mutually exclusive complaint pairs
        """
        rules = self.get_rules_by_type("mutually_exclusive")
        pairs = []
        
        for rule in rules:
            if rule.complaint_a and rule.complaint_b:
                pairs.append((rule.complaint_a, rule.complaint_b))
        
        return pairs
    
    def get_response_conflicts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get response conflict rules organized by complaint type.
        
        Returns:
            Dictionary mapping complaint types to their conflicting responses
        """
        rules = self.get_rules_by_type("response_mismatch")
        conflicts = {}
        
        for rule in rules:
            if rule.complaint_a and rule.conflicting_responses:
                conflicts[rule.complaint_a] = {
                    "conflicting_responses": rule.conflicting_responses,
                    "description": rule.description,
                    "severity": rule.severity
                }
        
        return conflicts
    
    def add_rule(self, rule: ContradictionRule) -> None:
        """
        Add a new contradiction rule to the configuration.
        
        Args:
            rule: Rule to add
        
        Raises:
            ValueError: If rule validation fails or creates circular dependencies
        """
        if not self.config:
            self.load_config()
        
        # Check for duplicate rule IDs
        existing_ids = {r.id for r in self.config.rules}
        if rule.id in existing_ids:
            raise ValueError(f"Rule ID '{rule.id}' already exists")
        
        # Add rule temporarily for validation
        temp_config = ContradictionConfig(
            version=self.config.version,
            rules=self.config.rules + [rule],
            global_settings=self.config.global_settings
        )
        
        # Validate the updated configuration
        self._validate_config(temp_config)
        
        # If validation passes, add the rule
        self.config.rules.append(rule)
        logger.info(f"Added new contradiction rule: {rule.id}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a contradiction rule by ID.
        
        Args:
            rule_id: ID of rule to remove
        
        Returns:
            True if rule was removed, False if not found
        """
        if not self.config:
            self.load_config()
        
        original_count = len(self.config.rules)
        self.config.rules = [r for r in self.config.rules if r.id != rule_id]
        
        removed = len(self.config.rules) < original_count
        if removed:
            logger.info(f"Removed contradiction rule: {rule_id}")
        else:
            logger.warning(f"Rule not found for removal: {rule_id}")
        
        return removed
    
    def _load_from_file(self, file_path: str) -> ContradictionConfig:
        """
        Load configuration from JSON file.
        
        Args:
            file_path: Path to configuration file
        
        Returns:
            Loaded ContradictionConfig
        
        Raises:
            ValueError: If file format is invalid
            IOError: If file cannot be read
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert rule dictionaries back to ContradictionRule objects
            rules = []
            for rule_data in data.get("rules", []):
                rule = ContradictionRule(**rule_data)
                rules.append(rule)
            
            config = ContradictionConfig(
                version=data.get("version", "1.0"),
                rules=rules,
                global_settings=data.get("global_settings", {})
            )
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise IOError(f"Cannot read configuration file: {e}")
    
    def _create_default_config(self) -> ContradictionConfig:
        """
        Create default contradiction rules configuration.
        
        Returns:
            Default ContradictionConfig with built-in rules
        """
        default_rules = [
            # Mutually exclusive complaint pairs
            ContradictionRule(
                id="accessibility_vs_oversold",
                name="Accessibility vs Oversold Conflict",
                type="mutually_exclusive",
                complaint_a="accessibility",
                complaint_b="oversold",
                severity="high",
                description="Customer claims accessibility issues but also reports facility was full",
                priority=10
            ),
            ContradictionRule(
                id="paid_again_vs_oversold",
                name="Paid Again vs Oversold Conflict", 
                type="mutually_exclusive",
                complaint_a="paid_again",
                complaint_b="oversold",
                severity="high",
                description="Customer claims double billing but also reports no space available",
                priority=20
            ),
            ContradictionRule(
                id="vehicle_restriction_vs_oversold",
                name="Vehicle Restriction vs Oversold Conflict",
                type="mutually_exclusive", 
                complaint_a="vehicle_restriction",
                complaint_b="oversold",
                severity="medium",
                description="Customer claims vehicle was rejected but also reports facility was full",
                priority=30
            ),
            
            # Response mismatch rules
            ContradictionRule(
                id="accessibility_parking_success",
                name="Accessibility vs Parking Success",
                type="response_mismatch",
                complaint_a="accessibility",
                conflicting_responses=["were you able to park? yes", "were you able to park?yes"],
                severity="high",
                description="Customer claims accessibility issues but confirms they were able to park",
                priority=40
            ),
            ContradictionRule(
                id="oversold_parking_success",
                name="Oversold vs Parking Success",
                type="response_mismatch",
                complaint_a="oversold",
                conflicting_responses=["were you able to park? yes", "were you able to park?yes"],
                severity="high", 
                description="Customer claims facility was full but confirms they were able to park",
                priority=50
            ),
            ContradictionRule(
                id="location_finding_parking_success",
                name="Location Finding vs Parking Success",
                type="response_mismatch",
                complaint_a="location_finding",
                conflicting_responses=["were you able to park? yes", "were you able to park?yes"],
                severity="medium",
                description="Customer claims they couldn't find location but confirms they were able to park",
                priority=60
            ),
            
            # Logical conflict rules (complex scenarios)
            ContradictionRule(
                id="accessibility_paid_again_conflict",
                name="Road Closure + Double Billing Conflict",
                type="logical_conflict",
                complaint_a="accessibility",
                complaint_b="paid_again",
                severity="high",
                description="Customer claims road closure prevented access but also reports being charged multiple times",
                priority=70
            ),
            ContradictionRule(
                id="vehicle_restriction_paid_again_conflict", 
                name="Vehicle Restriction + Double Billing Conflict",
                type="logical_conflict",
                complaint_a="vehicle_restriction",
                complaint_b="paid_again",
                severity="medium",
                description="Customer claims vehicle was rejected but also reports being charged multiple times",
                priority=80
            )
        ]
        
        global_settings = {
            "enable_contradiction_detection": True,
            "escalation_threshold": "medium",
            "max_rules_per_type": 50,
            "validation_strict_mode": True
        }
        
        return ContradictionConfig(
            version="1.0",
            rules=default_rules,
            global_settings=global_settings
        )
    
    def _validate_config(self, config: ContradictionConfig) -> None:
        """
        Validate configuration for consistency and circular dependencies.
        
        Args:
            config: Configuration to validate
        
        Raises:
            ValueError: If validation fails
        """
        logger.debug("Validating contradiction rules configuration")
        
        # Check for duplicate rule IDs
        rule_ids = [rule.id for rule in config.rules]
        if len(rule_ids) != len(set(rule_ids)):
            duplicates = [rid for rid in rule_ids if rule_ids.count(rid) > 1]
            raise ValueError(f"Duplicate rule IDs found: {duplicates}")
        
        # Validate individual rules
        for rule in config.rules:
            self._validate_rule(rule)
        
        # Check for circular dependencies in mutually exclusive rules
        self._check_circular_dependencies(config.rules)
        
        # Validate global settings
        self._validate_global_settings(config.global_settings)
        
        logger.debug("Configuration validation passed")
    
    def _validate_rule(self, rule: ContradictionRule) -> None:
        """
        Validate a single contradiction rule.
        
        Args:
            rule: Rule to validate
        
        Raises:
            ValueError: If rule is invalid
        """
        # Check required fields
        if not rule.id or not rule.name or not rule.type:
            raise ValueError(f"Rule missing required fields: {rule.id}")
        
        # Validate rule type
        valid_types = ["mutually_exclusive", "response_mismatch", "logical_conflict"]
        if rule.type not in valid_types:
            raise ValueError(f"Invalid rule type '{rule.type}' for rule {rule.id}")
        
        # Validate severity
        valid_severities = ["high", "medium", "low"]
        if rule.severity not in valid_severities:
            raise ValueError(f"Invalid severity '{rule.severity}' for rule {rule.id}")
        
        # Type-specific validation
        if rule.type == "mutually_exclusive":
            if not rule.complaint_a or not rule.complaint_b:
                raise ValueError(f"Mutually exclusive rule {rule.id} requires both complaint_a and complaint_b")
            if rule.complaint_a == rule.complaint_b:
                raise ValueError(f"Mutually exclusive rule {rule.id} has identical complaint types")
        
        elif rule.type == "response_mismatch":
            if not rule.complaint_a or not rule.conflicting_responses:
                raise ValueError(f"Response mismatch rule {rule.id} requires complaint_a and conflicting_responses")
            if not isinstance(rule.conflicting_responses, list) or len(rule.conflicting_responses) == 0:
                raise ValueError(f"Response mismatch rule {rule.id} requires non-empty conflicting_responses list")
        
        elif rule.type == "logical_conflict":
            if not rule.complaint_a:
                raise ValueError(f"Logical conflict rule {rule.id} requires at least complaint_a")
            # complaint_b is optional for logical conflicts
        
        # Validate priority
        if not isinstance(rule.priority, int) or rule.priority < 0:
            raise ValueError(f"Rule {rule.id} priority must be a non-negative integer")
    
    def _check_circular_dependencies(self, rules: List[ContradictionRule]) -> None:
        """
        Check for circular dependencies in mutually exclusive rules.
        
        Args:
            rules: List of rules to check
        
        Raises:
            ValueError: If circular dependencies are found
        """
        # Build graph of mutually exclusive relationships
        exclusive_pairs = set()
        
        for rule in rules:
            if rule.type == "mutually_exclusive" and rule.enabled:
                if rule.complaint_a and rule.complaint_b:
                    # Add both directions for undirected relationship
                    exclusive_pairs.add((rule.complaint_a, rule.complaint_b))
                    exclusive_pairs.add((rule.complaint_b, rule.complaint_a))
        
        # Check for transitivity violations that could create logical inconsistencies
        # If A excludes B and B excludes C, then A should exclude C (or be explicitly allowed)
        complaint_types = set()
        for a, b in exclusive_pairs:
            complaint_types.add(a)
            complaint_types.add(b)
        
        # For now, we just check for direct circular references (A->B, B->A is expected)
        # More complex transitivity checking could be added if needed
        
        # Check for self-references (should not happen with validation above)
        for rule in rules:
            if (rule.type == "mutually_exclusive" and rule.enabled and 
                rule.complaint_a and rule.complaint_b and 
                rule.complaint_a == rule.complaint_b):
                raise ValueError(f"Circular dependency: rule {rule.id} references same complaint type")
    
    def _validate_global_settings(self, settings: Dict[str, Any]) -> None:
        """
        Validate global configuration settings.
        
        Args:
            settings: Global settings to validate
        
        Raises:
            ValueError: If settings are invalid
        """
        # Validate escalation threshold
        if "escalation_threshold" in settings:
            valid_thresholds = ["high", "medium", "low"]
            if settings["escalation_threshold"] not in valid_thresholds:
                raise ValueError(f"Invalid escalation_threshold: {settings['escalation_threshold']}")
        
        # Validate max rules per type
        if "max_rules_per_type" in settings:
            max_rules = settings["max_rules_per_type"]
            if not isinstance(max_rules, int) or max_rules <= 0:
                raise ValueError("max_rules_per_type must be a positive integer")
        
        # Validate boolean settings
        boolean_settings = ["enable_contradiction_detection", "validation_strict_mode"]
        for setting in boolean_settings:
            if setting in settings and not isinstance(settings[setting], bool):
                raise ValueError(f"{setting} must be a boolean value")