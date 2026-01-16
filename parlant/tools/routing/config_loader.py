"""
Configuration loading and validation for the routing system.

This module handles loading router configuration from JSON files,
validating the configuration, and providing hot reload capabilities.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set, List, Any
from dataclasses import dataclass
from datetime import datetime
import pytz

logger = logging.getLogger(__name__)


@dataclass
class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    field: str
    message: str
    
    def __str__(self):
        return f"Configuration validation error in '{self.field}': {self.message}"


@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[ConfigValidationError]
    warnings: List[str]
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class RouterConfigLoader:
    """
    Loads and validates router configuration with hot reload support.
    
    Features:
    - JSON configuration file loading
    - Configuration validation with detailed error reporting
    - Hot reload support with file modification detection
    - Fallback to safe defaults on invalid configuration
    - Configuration change notifications
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        if config_path is None:
            # Default to router_config.json in the same directory as this module
            config_path = Path(__file__).parent / "router_config.json"
        
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self.last_modified: Optional[float] = None
        self.last_load_time: Optional[float] = None
        self.validation_result: Optional[ConfigValidationResult] = None
        
        logger.info(f"RouterConfigLoader initialized with config path: {self.config_path}")
    
    def load_config(self, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load configuration from file with validation.
        
        Args:
            force_reload: If True, reload even if file hasn't changed
            
        Returns:
            Configuration dictionary (may be defaults if file invalid)
            
        Raises:
            FileNotFoundError: If config file doesn't exist and no defaults available
        """
        try:
            # Check if file exists
            if not self.config_path.exists():
                logger.warning(f"Configuration file not found: {self.config_path}")
                logger.info("Using safe default configuration")
                return self._get_safe_defaults()
            
            # Check if reload is needed
            current_modified = self.config_path.stat().st_mtime
            
            if not force_reload and self.last_modified == current_modified:
                # File hasn't changed, return cached config
                return self.config
            
            # Load configuration from file
            logger.info(f"Loading configuration from {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            # Validate configuration
            validation_result = self._validate_config(raw_config)
            self.validation_result = validation_result
            
            if not validation_result.is_valid:
                # Configuration is invalid, log errors and use defaults
                logger.error("Configuration validation failed:")
                for error in validation_result.errors:
                    logger.error(f"  - {error}")
                
                logger.warning("Using safe default configuration due to validation errors")
                self.config = self._get_safe_defaults()
            else:
                # Configuration is valid
                self.config = raw_config
                
                # Log any warnings
                for warning in validation_result.warnings:
                    logger.warning(f"Configuration warning: {warning}")
                
                logger.info("Configuration loaded and validated successfully")
            
            # Update tracking variables
            self.last_modified = current_modified
            self.last_load_time = time.time()
            
            return self.config
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            logger.warning("Using safe default configuration due to JSON parsing error")
            return self._get_safe_defaults()
        
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            logger.warning("Using safe default configuration due to unexpected error")
            return self._get_safe_defaults()
    
    def _validate_config(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Validate configuration structure and values.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            ConfigValidationResult with validation status and errors
        """
        errors = []
        warnings = []
        
        # Validate MOR gate configuration
        if "mor_gate" in config:
            mor_config = config["mor_gate"]
            
            # Validate third_party_vendors
            if "third_party_vendors" in mor_config:
                vendors = mor_config["third_party_vendors"]
                if not isinstance(vendors, list):
                    errors.append(ConfigValidationError(
                        "mor_gate.third_party_vendors",
                        "Must be a list of vendor names"
                    ))
                elif len(vendors) == 0:
                    errors.append(ConfigValidationError(
                        "mor_gate.third_party_vendors", 
                        "Cannot be empty - at least one third-party vendor required"
                    ))
                elif not all(isinstance(v, str) and v.strip() for v in vendors):
                    errors.append(ConfigValidationError(
                        "mor_gate.third_party_vendors",
                        "All vendor names must be non-empty strings"
                    ))
            
            # Validate parkwhiz_partners
            if "parkwhiz_partners" in mor_config:
                partners = mor_config["parkwhiz_partners"]
                if not isinstance(partners, list):
                    errors.append(ConfigValidationError(
                        "mor_gate.parkwhiz_partners",
                        "Must be a list of partner names"
                    ))
                elif len(partners) == 0:
                    errors.append(ConfigValidationError(
                        "mor_gate.parkwhiz_partners",
                        "Cannot be empty - at least 'parkwhiz' required"
                    ))
                elif "parkwhiz" not in [p.lower() for p in partners]:
                    warnings.append("'parkwhiz' not found in parkwhiz_partners - may cause issues")
            
            # Validate special_partners
            if "special_partners" in mor_config:
                special = mor_config["special_partners"]
                if not isinstance(special, dict):
                    errors.append(ConfigValidationError(
                        "mor_gate.special_partners",
                        "Must be a dictionary mapping partner names to flags"
                    ))
                elif not all(isinstance(k, str) and isinstance(v, str) for k, v in special.items()):
                    errors.append(ConfigValidationError(
                        "mor_gate.special_partners",
                        "All keys and values must be strings"
                    ))
        
        # Validate Product Type gate configuration
        if "product_type_gate" in config:
            pt_config = config["product_type_gate"]
            
            # Validate on_demand_threshold_hours
            if "on_demand_threshold_hours" in pt_config:
                threshold = pt_config["on_demand_threshold_hours"]
                if not isinstance(threshold, (int, float)):
                    errors.append(ConfigValidationError(
                        "product_type_gate.on_demand_threshold_hours",
                        "Must be a number"
                    ))
                elif threshold <= 0:
                    errors.append(ConfigValidationError(
                        "product_type_gate.on_demand_threshold_hours",
                        "Must be greater than 0"
                    ))
                elif threshold > 168:  # More than a week seems unreasonable
                    warnings.append(f"on_demand_threshold_hours is very high ({threshold}h) - verify this is correct")
        
        # Validate router_settings
        if "router_settings" in config:
            settings = config["router_settings"]
            
            if "config_check_interval_seconds" in settings:
                interval = settings["config_check_interval_seconds"]
                if not isinstance(interval, (int, float)):
                    errors.append(ConfigValidationError(
                        "router_settings.config_check_interval_seconds",
                        "Must be a number"
                    ))
                elif interval < 1:
                    errors.append(ConfigValidationError(
                        "router_settings.config_check_interval_seconds",
                        "Must be at least 1 second"
                    ))
        
        # Validate Pre-Arrival gate configuration
        if "pre_arrival_gate" in config:
            pa_config = config["pre_arrival_gate"]
            
            # Validate target processing times
            if "target_processing_ms" in pa_config:
                target = pa_config["target_processing_ms"]
                if not isinstance(target, (int, float)):
                    errors.append(ConfigValidationError(
                        "pre_arrival_gate.target_processing_ms",
                        "Must be a number"
                    ))
                elif target <= 0:
                    errors.append(ConfigValidationError(
                        "pre_arrival_gate.target_processing_ms",
                        "Must be greater than 0"
                    ))
                elif target > 10000:  # More than 10 seconds seems unreasonable
                    warnings.append(f"target_processing_ms is very high ({target}ms) - verify this is correct")
            
            if "optimized_path_ms" in pa_config:
                optimized = pa_config["optimized_path_ms"]
                if not isinstance(optimized, (int, float)):
                    errors.append(ConfigValidationError(
                        "pre_arrival_gate.optimized_path_ms",
                        "Must be a number"
                    ))
                elif optimized <= 0:
                    errors.append(ConfigValidationError(
                        "pre_arrival_gate.optimized_path_ms",
                        "Must be greater than 0"
                    ))
            
            # Validate date parser configuration
            if "date_parser" in pa_config:
                dp_config = pa_config["date_parser"]
                
                if "date_formats" in dp_config:
                    formats = dp_config["date_formats"]
                    if not isinstance(formats, list):
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.date_parser.date_formats",
                            "Must be a list of date format strings"
                        ))
                    elif len(formats) == 0:
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.date_parser.date_formats",
                            "Cannot be empty - at least one date format required"
                        ))
                    else:
                        # Validate each date format string
                        for i, fmt in enumerate(formats):
                            if not isinstance(fmt, str):
                                errors.append(ConfigValidationError(
                                    f"pre_arrival_gate.date_parser.date_formats[{i}]",
                                    "Date format must be a string"
                                ))
                            else:
                                # Test the format string with a known date
                                try:
                                    test_date = datetime(2025, 12, 17, 19, 45, 0)
                                    formatted = test_date.strftime(fmt)
                                    parsed = datetime.strptime(formatted, fmt)
                                    if parsed != test_date:
                                        warnings.append(f"Date format '{fmt}' may not round-trip correctly")
                                except ValueError as e:
                                    errors.append(ConfigValidationError(
                                        f"pre_arrival_gate.date_parser.date_formats[{i}]",
                                        f"Invalid date format string '{fmt}': {e}"
                                    ))
            
            # Validate timezone handler configuration
            if "timezone_handler" in pa_config:
                tz_config = pa_config["timezone_handler"]
                
                if "city_to_timezone_mapping" in tz_config:
                    mapping = tz_config["city_to_timezone_mapping"]
                    if not isinstance(mapping, dict):
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.timezone_handler.city_to_timezone_mapping",
                            "Must be a dictionary mapping city names to timezone identifiers"
                        ))
                    else:
                        # Validate each timezone identifier
                        for city, tz_id in mapping.items():
                            if not isinstance(city, str) or not city.strip():
                                errors.append(ConfigValidationError(
                                    f"pre_arrival_gate.timezone_handler.city_to_timezone_mapping['{city}']",
                                    "City name must be a non-empty string"
                                ))
                            
                            if not isinstance(tz_id, str) or not tz_id.strip():
                                errors.append(ConfigValidationError(
                                    f"pre_arrival_gate.timezone_handler.city_to_timezone_mapping['{city}']",
                                    "Timezone identifier must be a non-empty string"
                                ))
                            else:
                                # Validate timezone identifier
                                try:
                                    pytz.timezone(tz_id)
                                except pytz.UnknownTimeZoneError:
                                    errors.append(ConfigValidationError(
                                        f"pre_arrival_gate.timezone_handler.city_to_timezone_mapping['{city}']",
                                        f"Invalid timezone identifier '{tz_id}' - must be a valid IANA timezone"
                                    ))
            
            # Validate comparator configuration
            if "comparator" in pa_config:
                comp_config = pa_config["comparator"]
                
                if "max_future_months" in comp_config:
                    months = comp_config["max_future_months"]
                    if not isinstance(months, (int, float)):
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.comparator.max_future_months",
                            "Must be a number"
                        ))
                    elif months <= 0:
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.comparator.max_future_months",
                            "Must be greater than 0"
                        ))
                    elif months > 60:  # More than 5 years seems unreasonable
                        warnings.append(f"max_future_months is very high ({months}) - verify this is correct")
                
                if "max_past_days" in comp_config:
                    days = comp_config["max_past_days"]
                    if not isinstance(days, (int, float)):
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.comparator.max_past_days",
                            "Must be a number"
                        ))
                    elif days <= 0:
                        errors.append(ConfigValidationError(
                            "pre_arrival_gate.comparator.max_past_days",
                            "Must be greater than 0"
                        ))
                    elif days > 365:  # More than a year seems unreasonable
                        warnings.append(f"max_past_days is very high ({days}) - verify this is correct")
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _get_safe_defaults(self) -> Dict[str, Any]:
        """
        Get safe default configuration when file is missing or invalid.
        
        Returns:
            Dictionary with safe default configuration values
        """
        return {
            "mor_gate": {
                "third_party_vendors": [
                    "axs", "seatgeek", "stubhub", "ticketmaster",
                    "ticketnetwork", "gametime", "vividseats"
                ],
                "parkwhiz_partners": [
                    "parkwhiz", "google", "waze", "apple"
                ],
                "special_partners": {
                    "groupon": "groupon_credit_only"
                }
            },
            "product_type_gate": {
                "on_demand_threshold_hours": 4,
                "season_package_detection": "future_implementation"
            },
            "pre_arrival_gate": {
                "target_processing_ms": 200,
                "optimized_path_ms": 100,
                "date_parser": {
                    "date_formats": [
                        "%A %b %d, %Y, %I:%M %p",  # "Wednesday Dec 17, 2025, 07:45 PM"
                        "%Y-%m-%dT%H:%M:%S",       # "2025-12-17T19:45:00"
                        "%m/%d/%Y %I:%M %p",       # "12/17/2025 7:45 PM"
                        "%b %d, %Y %I:%M:%S %p",   # "Dec 17, 2025 7:45:00 PM"
                        "%Y-%m-%d %H:%M:%S"        # "2025-12-17 19:45:00"
                    ]
                },
                "timezone_handler": {
                    "city_to_timezone_mapping": {
                        "new york, ny": "America/New_York",
                        "new york": "America/New_York",
                        "manhattan, ny": "America/New_York",
                        "brooklyn, ny": "America/New_York",
                        "los angeles, ca": "America/Los_Angeles",
                        "los angeles": "America/Los_Angeles",
                        "san francisco, ca": "America/Los_Angeles",
                        "chicago, il": "America/Chicago",
                        "chicago": "America/Chicago",
                        "dallas, tx": "America/Chicago",
                        "houston, tx": "America/Chicago",
                        "denver, co": "America/Denver",
                        "phoenix, az": "America/Phoenix",
                        "seattle, wa": "America/Los_Angeles",
                        "miami, fl": "America/New_York",
                        "atlanta, ga": "America/New_York",
                        "boston, ma": "America/New_York",
                        "washington, dc": "America/New_York",
                        "philadelphia, pa": "America/New_York"
                    }
                },
                "comparator": {
                    "max_future_months": 12,
                    "max_past_days": 7
                }
            },
            "router_settings": {
                "enable_hot_reload": True,
                "config_check_interval_seconds": 30,
                "performance_monitoring": True,
                "enable_mor_product_type_gates": True,
                "enable_pre_arrival_gate": True
            }
        }
    
    def check_for_updates(self) -> bool:
        """
        Check if configuration file has been modified since last load.
        
        Returns:
            True if file has been modified, False otherwise
        """
        if not self.config_path.exists():
            return False
        
        current_modified = self.config_path.stat().st_mtime
        return current_modified != self.last_modified
    
    def get_validation_result(self) -> Optional[ConfigValidationResult]:
        """
        Get the result of the last configuration validation.
        
        Returns:
            ConfigValidationResult from last load, or None if never loaded
        """
        return self.validation_result
    
    def get_config_info(self) -> Dict[str, Any]:
        """
        Get information about the current configuration state.
        
        Returns:
            Dictionary with configuration metadata
        """
        return {
            "config_path": str(self.config_path),
            "file_exists": self.config_path.exists(),
            "last_modified": self.last_modified,
            "last_load_time": self.last_load_time,
            "is_valid": self.validation_result.is_valid if self.validation_result else None,
            "validation_errors": len(self.validation_result.errors) if self.validation_result else 0,
            "validation_warnings": len(self.validation_result.warnings) if self.validation_result else 0
        }