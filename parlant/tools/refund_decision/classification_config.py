"""
Configurable classification rules system for "Paid Again" scenarios.

This module provides a configuration system for overstay tolerance thresholds,
confidence thresholds, and extensible rule patterns that can be modified
without code changes.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import os

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceThresholds:
    """Configuration for confidence thresholds by processing mode."""
    webhook_automation: float = 0.85
    interactive_chat: float = 0.70
    
    def validate(self) -> List[str]:
        """Validate threshold values."""
        errors = []
        webhook_valid = 0.0 <= self.webhook_automation <= 1.0
        chat_valid = 0.0 <= self.interactive_chat <= 1.0
        
        if not webhook_valid:
            errors.append(f"webhook_automation threshold must be 0.0-1.0, got {self.webhook_automation}")
        if not chat_valid:
            errors.append(f"interactive_chat threshold must be 0.0-1.0, got {self.interactive_chat}")
        
        # Only check threshold comparison if both values are in valid range
        if webhook_valid and chat_valid and self.webhook_automation < self.interactive_chat:
            errors.append("webhook_automation threshold should be >= interactive_chat threshold")
        return errors


@dataclass
class ToleranceThresholds:
    """Configuration for timeline analysis tolerance thresholds."""
    overstay_tolerance_minutes: int = 15
    early_arrival_tolerance_minutes: int = 30
    
    def validate(self) -> List[str]:
        """Validate tolerance values."""
        errors = []
        if self.overstay_tolerance_minutes < 0:
            errors.append(f"overstay_tolerance_minutes must be >= 0, got {self.overstay_tolerance_minutes}")
        if self.early_arrival_tolerance_minutes < 0:
            errors.append(f"early_arrival_tolerance_minutes must be >= 0, got {self.early_arrival_tolerance_minutes}")
        if self.overstay_tolerance_minutes > 60:
            errors.append(f"overstay_tolerance_minutes should be <= 60 minutes, got {self.overstay_tolerance_minutes}")
        if self.early_arrival_tolerance_minutes > 120:
            errors.append(f"early_arrival_tolerance_minutes should be <= 120 minutes, got {self.early_arrival_tolerance_minutes}")
        return errors


@dataclass
class ScenarioRule:
    """Configuration for a single scenario classification rule."""
    id: str
    name: str
    scenario_type: str  # "accidental_double_booking", "early_arrival_overstay", etc.
    patterns: List[str]  # Regex patterns to match
    confidence_base: float  # Base confidence when pattern matches
    confidence_boost: float = 0.0  # Additional confidence for multiple matches
    priority: int = 100  # Lower numbers = higher priority
    enabled: bool = True
    description: str = ""
    
    def validate(self) -> List[str]:
        """Validate rule configuration."""
        errors = []
        if not self.id or not self.id.strip():
            errors.append("Rule id cannot be empty")
        if not self.name or not self.name.strip():
            errors.append("Rule name cannot be empty")
        if not self.scenario_type or not self.scenario_type.strip():
            errors.append("Rule scenario_type cannot be empty")
        if not self.patterns:
            errors.append("Rule must have at least one pattern")
        if not 0.0 <= self.confidence_base <= 1.0:
            errors.append(f"confidence_base must be 0.0-1.0, got {self.confidence_base}")
        if not 0.0 <= self.confidence_boost <= 1.0:
            errors.append(f"confidence_boost must be 0.0-1.0, got {self.confidence_boost}")
        if self.priority < 0:
            errors.append(f"priority must be >= 0, got {self.priority}")
        return errors


@dataclass
class ClassificationConfig:
    """Complete configuration for the classification system."""
    version: str = "1.0"
    confidence_thresholds: ConfidenceThresholds = field(default_factory=ConfidenceThresholds)
    tolerance_thresholds: ToleranceThresholds = field(default_factory=ToleranceThresholds)
    scenario_rules: List[ScenarioRule] = field(default_factory=list)
    global_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default global settings if not provided."""
        if not self.global_settings:
            self.global_settings = {
                "enable_rule_logging": True,
                "max_rules_per_scenario": 20,
                "validation_strict_mode": True,
                "rule_application_timeout_seconds": 5
            }
    
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        errors = []
        
        # Validate thresholds
        errors.extend(self.confidence_thresholds.validate())
        errors.extend(self.tolerance_thresholds.validate())
        
        # Validate scenario rules
        rule_ids = set()
        for rule in self.scenario_rules:
            rule_errors = rule.validate()
            errors.extend([f"Rule {rule.id}: {error}" for error in rule_errors])
            
            # Check for duplicate IDs
            if rule.id in rule_ids:
                errors.append(f"Duplicate rule ID: {rule.id}")
            rule_ids.add(rule.id)
        
        # Validate global settings
        if "max_rules_per_scenario" in self.global_settings:
            max_rules = self.global_settings["max_rules_per_scenario"]
            if not isinstance(max_rules, int) or max_rules <= 0:
                errors.append(f"max_rules_per_scenario must be positive integer, got {max_rules}")
        
        return errors
    
    def get_rules_by_scenario(self, scenario_type: str) -> List[ScenarioRule]:
        """Get all enabled rules for a specific scenario type, sorted by priority."""
        rules = [
            rule for rule in self.scenario_rules 
            if rule.enabled and rule.scenario_type == scenario_type
        ]
        return sorted(rules, key=lambda r: r.priority)
    
    def get_all_enabled_rules(self) -> List[ScenarioRule]:
        """Get all enabled rules sorted by priority."""
        rules = [rule for rule in self.scenario_rules if rule.enabled]
        return sorted(rules, key=lambda r: r.priority)


class ClassificationConfigManager:
    """
    Manager for loading, validating, and updating classification configuration.
    
    This class handles configuration file I/O, validation, and provides
    a clean interface for accessing configuration data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file (uses default if None)
        """
        if config_path is None:
            # Default to config file in same directory as this module
            config_dir = Path(__file__).parent
            config_path = config_dir / "classification_config.json"
        
        self.config_path = Path(config_path)
        self._config: Optional[ClassificationConfig] = None
        self._last_modified: Optional[float] = None
        
        logger.info(f"ClassificationConfigManager initialized with config path: {self.config_path}")
    
    def load_config(self, force_reload: bool = False) -> ClassificationConfig:
        """
        Load configuration from file with caching and validation.
        
        Args:
            force_reload: Force reload even if cached version exists
            
        Returns:
            Validated ClassificationConfig instance
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file doesn't exist and no defaults available
        """
        # Check if we need to reload
        if not force_reload and self._config is not None:
            if self.config_path.exists():
                current_modified = self.config_path.stat().st_mtime
                if current_modified == self._last_modified:
                    logger.debug("Using cached configuration")
                    return self._config
            else:
                # File doesn't exist but we have cached config
                logger.debug("Config file missing but using cached configuration")
                return self._config
        
        # Load from file or create defaults
        if self.config_path.exists():
            logger.info(f"Loading configuration from {self.config_path}")
            config_data = self._load_config_file()
            self._last_modified = self.config_path.stat().st_mtime
        else:
            logger.info("Config file not found, creating default configuration")
            config_data = self._create_default_config_data()
            self._last_modified = None
        
        # Parse and validate configuration
        self._config = self._parse_config_data(config_data)
        validation_errors = self._config.validate()
        
        if validation_errors:
            error_msg = f"Configuration validation failed: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Configuration loaded successfully: {len(self._config.scenario_rules)} rules, "
                   f"webhook_threshold={self._config.confidence_thresholds.webhook_automation}")
        
        return self._config
    
    def save_config(self, config: ClassificationConfig) -> None:
        """
        Save configuration to file with validation.
        
        Args:
            config: Configuration to save
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Validate before saving
        validation_errors = config.validate()
        if validation_errors:
            error_msg = f"Cannot save invalid configuration: {'; '.join(validation_errors)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert to dict and save
        config_data = self._config_to_dict(config)
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(self.config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        
        # Update cache
        self._config = config
        self._last_modified = self.config_path.stat().st_mtime
        
        logger.info(f"Configuration saved to {self.config_path}")
    
    def get_config(self) -> ClassificationConfig:
        """
        Get current configuration, loading if necessary.
        
        Returns:
            Current ClassificationConfig instance
        """
        if self._config is None:
            return self.load_config()
        return self._config
    
    def update_confidence_thresholds(
        self, 
        webhook_threshold: Optional[float] = None,
        chat_threshold: Optional[float] = None
    ) -> None:
        """
        Update confidence thresholds and save configuration.
        
        Args:
            webhook_threshold: New webhook automation threshold
            chat_threshold: New interactive chat threshold
        """
        config = self.get_config()
        
        if webhook_threshold is not None:
            config.confidence_thresholds.webhook_automation = webhook_threshold
        if chat_threshold is not None:
            config.confidence_thresholds.interactive_chat = chat_threshold
        
        self.save_config(config)
        logger.info(f"Updated confidence thresholds: webhook={config.confidence_thresholds.webhook_automation}, "
                   f"chat={config.confidence_thresholds.interactive_chat}")
    
    def update_tolerance_thresholds(
        self,
        overstay_tolerance: Optional[int] = None,
        early_arrival_tolerance: Optional[int] = None
    ) -> None:
        """
        Update tolerance thresholds and save configuration.
        
        Args:
            overstay_tolerance: New overstay tolerance in minutes
            early_arrival_tolerance: New early arrival tolerance in minutes
        """
        config = self.get_config()
        
        if overstay_tolerance is not None:
            config.tolerance_thresholds.overstay_tolerance_minutes = overstay_tolerance
        if early_arrival_tolerance is not None:
            config.tolerance_thresholds.early_arrival_tolerance_minutes = early_arrival_tolerance
        
        self.save_config(config)
        logger.info(f"Updated tolerance thresholds: overstay={config.tolerance_thresholds.overstay_tolerance_minutes}min, "
                   f"early_arrival={config.tolerance_thresholds.early_arrival_tolerance_minutes}min")
    
    def add_scenario_rule(self, rule: ScenarioRule) -> None:
        """
        Add a new scenario rule and save configuration.
        
        Args:
            rule: ScenarioRule to add
            
        Raises:
            ValueError: If rule is invalid or ID already exists
        """
        config = self.get_config()
        
        # Check for duplicate ID
        existing_ids = {r.id for r in config.scenario_rules}
        if rule.id in existing_ids:
            raise ValueError(f"Rule ID '{rule.id}' already exists")
        
        # Validate rule
        rule_errors = rule.validate()
        if rule_errors:
            raise ValueError(f"Invalid rule: {'; '.join(rule_errors)}")
        
        # Add rule and save
        config.scenario_rules.append(rule)
        self.save_config(config)
        
        logger.info(f"Added scenario rule: {rule.id} ({rule.scenario_type})")
    
    def remove_scenario_rule(self, rule_id: str) -> bool:
        """
        Remove a scenario rule by ID and save configuration.
        
        Args:
            rule_id: ID of rule to remove
            
        Returns:
            True if rule was found and removed, False otherwise
        """
        config = self.get_config()
        
        original_count = len(config.scenario_rules)
        config.scenario_rules = [r for r in config.scenario_rules if r.id != rule_id]
        
        if len(config.scenario_rules) < original_count:
            self.save_config(config)
            logger.info(f"Removed scenario rule: {rule_id}")
            return True
        else:
            logger.warning(f"Rule ID not found: {rule_id}")
            return False
    
    def _load_config_file(self) -> Dict[str, Any]:
        """Load configuration data from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file {self.config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading config file {self.config_path}: {e}")
    
    def _create_default_config_data(self) -> Dict[str, Any]:
        """Create default configuration data."""
        return {
            "version": "1.0",
            "confidence_thresholds": {
                "webhook_automation": 0.85,
                "interactive_chat": 0.70
            },
            "tolerance_thresholds": {
                "overstay_tolerance_minutes": 15,
                "early_arrival_tolerance_minutes": 30
            },
            "global_settings": {
                "enable_rule_logging": True,
                "max_rules_per_scenario": 20,
                "validation_strict_mode": True,
                "rule_application_timeout_seconds": 5
            },
            "scenario_rules": [
                {
                    "id": "double_booking_explicit",
                    "name": "Explicit Double Booking Language",
                    "scenario_type": "accidental_double_booking",
                    "patterns": [
                        r"booked.*twice",
                        r"double.*booking",
                        r"accidentally.*booked.*again"
                    ],
                    "confidence_base": 0.9,
                    "confidence_boost": 0.05,
                    "priority": 10,
                    "enabled": True,
                    "description": "Customer explicitly mentions booking twice or double booking"
                },
                {
                    "id": "entry_failure_qr_code",
                    "name": "QR Code Entry Failure",
                    "scenario_type": "entry_failure_complete_unused",
                    "patterns": [
                        r"qr.*code.*(?:didn't|wouldn't|not).*work",
                        r"qr.*(?:rejected|failed|invalid)",
                        r"scanner.*(?:rejected|broken|not.*working)"
                    ],
                    "confidence_base": 0.85,
                    "confidence_boost": 0.1,
                    "priority": 20,
                    "enabled": True,
                    "description": "QR code or scanner technical failure at entry"
                },
                {
                    "id": "exit_complication_success_then_failure",
                    "name": "Successful Entry with Exit Problems",
                    "scenario_type": "exit_complication_poor_experience",
                    "patterns": [
                        r"got.*in.*(?:fine|ok).*but.*(?:exit|leaving).*(?:problem|issue)",
                        r"entered.*garage.*but.*had.*to.*pay.*to.*(?:exit|leave)",
                        r"parking.*worked.*but.*departure.*(?:issue|problem)"
                    ],
                    "confidence_base": 0.9,
                    "confidence_boost": 0.05,
                    "priority": 30,
                    "enabled": True,
                    "description": "Customer successfully parked but had exit complications"
                }
            ]
        }
    
    def _parse_config_data(self, config_data: Dict[str, Any]) -> ClassificationConfig:
        """Parse configuration data into ClassificationConfig object."""
        # Parse confidence thresholds
        confidence_data = config_data.get("confidence_thresholds", {})
        confidence_thresholds = ConfidenceThresholds(
            webhook_automation=confidence_data.get("webhook_automation", 0.85),
            interactive_chat=confidence_data.get("interactive_chat", 0.70)
        )
        
        # Parse tolerance thresholds
        tolerance_data = config_data.get("tolerance_thresholds", {})
        tolerance_thresholds = ToleranceThresholds(
            overstay_tolerance_minutes=tolerance_data.get("overstay_tolerance_minutes", 15),
            early_arrival_tolerance_minutes=tolerance_data.get("early_arrival_tolerance_minutes", 30)
        )
        
        # Parse scenario rules
        scenario_rules = []
        for rule_data in config_data.get("scenario_rules", []):
            rule = ScenarioRule(
                id=rule_data["id"],
                name=rule_data["name"],
                scenario_type=rule_data["scenario_type"],
                patterns=rule_data["patterns"],
                confidence_base=rule_data["confidence_base"],
                confidence_boost=rule_data.get("confidence_boost", 0.0),
                priority=rule_data.get("priority", 100),
                enabled=rule_data.get("enabled", True),
                description=rule_data.get("description", "")
            )
            scenario_rules.append(rule)
        
        return ClassificationConfig(
            version=config_data.get("version", "1.0"),
            confidence_thresholds=confidence_thresholds,
            tolerance_thresholds=tolerance_thresholds,
            scenario_rules=scenario_rules,
            global_settings=config_data.get("global_settings", {})
        )
    
    def _config_to_dict(self, config: ClassificationConfig) -> Dict[str, Any]:
        """Convert ClassificationConfig to dictionary for JSON serialization."""
        return {
            "version": config.version,
            "confidence_thresholds": {
                "webhook_automation": config.confidence_thresholds.webhook_automation,
                "interactive_chat": config.confidence_thresholds.interactive_chat
            },
            "tolerance_thresholds": {
                "overstay_tolerance_minutes": config.tolerance_thresholds.overstay_tolerance_minutes,
                "early_arrival_tolerance_minutes": config.tolerance_thresholds.early_arrival_tolerance_minutes
            },
            "global_settings": config.global_settings,
            "scenario_rules": [
                {
                    "id": rule.id,
                    "name": rule.name,
                    "scenario_type": rule.scenario_type,
                    "patterns": rule.patterns,
                    "confidence_base": rule.confidence_base,
                    "confidence_boost": rule.confidence_boost,
                    "priority": rule.priority,
                    "enabled": rule.enabled,
                    "description": rule.description
                }
                for rule in config.scenario_rules
            ]
        }