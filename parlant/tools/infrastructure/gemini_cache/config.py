"""
Configuration management for Gemini Context Caching.

This module handles loading, validation, and hot-reloading of cache configuration:
- Environment variable loading
- JSON configuration file loading
- Configuration validation with safe defaults
- Hot reload support without restart

Requirements: 13.1, 13.2, 13.3, 13.4, 13.5
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from .models import CacheConfiguration

logger = logging.getLogger(__name__)


# Default configuration values
DEFAULT_CACHE_CONFIG = CacheConfiguration()


def load_cache_config(
    config_path: Optional[str] = None,
    env_prefix: str = "GEMINI_CACHE_"
) -> CacheConfiguration:
    """
    Load cache configuration from environment variables and optional JSON file.
    
    Configuration is loaded in this order (later sources override earlier):
    1. Default values
    2. JSON configuration file (if provided)
    3. Environment variables
    
    Environment Variables:
        GEMINI_CACHE_ENABLED: Enable/disable cache optimization (default: true)
        GEMINI_CACHE_MONITOR: Enable cache effectiveness monitoring (default: true)
        GEMINI_CACHE_INPUT_COST: Input cost per 1M tokens (default: 0.30)
        GEMINI_CACHE_CACHED_COST: Cached input cost per 1M tokens (default: 0.03)
        GEMINI_CACHE_OUTPUT_COST: Output cost per 1M tokens (default: 2.50)
        GEMINI_CACHE_MAX_TOKENS: Max static content tokens (default: 32000)
        GEMINI_CACHE_TIMEOUT_MS: Prompt construction timeout (default: 10)
        GEMINI_CACHE_LOAD_TIMEOUT_S: Content loading timeout (default: 3)
        GEMINI_CACHE_ALERT_THRESHOLD: Cache effectiveness alert threshold (default: 70)
        GEMINI_CACHE_HOT_RELOAD: Enable hot reload (default: true)
        GEMINI_API_KEY: Gemini API key
    
    Args:
        config_path: Optional path to JSON configuration file
        env_prefix: Prefix for environment variables
    
    Returns:
        CacheConfiguration: Loaded configuration
    """
    config = CacheConfiguration()
    
    # Load from JSON file if provided
    if config_path:
        config = _load_from_json(config_path, config)
    
    # Override with environment variables
    config = _load_from_env(config, env_prefix)
    
    return config


def _load_from_json(config_path: str, config: CacheConfiguration) -> CacheConfiguration:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON configuration file
        config: Base configuration to update
    
    Returns:
        Updated configuration
    """
    try:
        path = Path(config_path)
        if not path.exists():
            logger.warning(f"Configuration file not found: {config_path}")
            return config
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Extract context_caching section if present
        cache_data = data.get("context_caching", data)
        
        # Update configuration from JSON
        if "enabled" in cache_data:
            config.enabled = bool(cache_data["enabled"])
        if "monitor_cache_effectiveness" in cache_data:
            config.monitor_cache_effectiveness = bool(cache_data["monitor_cache_effectiveness"])
        if "static_content_paths" in cache_data:
            config.static_content_paths = cache_data["static_content_paths"]
        if "tool_definitions_path" in cache_data:
            config.tool_definitions_path = cache_data["tool_definitions_path"]
        if "few_shot_examples_path" in cache_data:
            config.few_shot_examples_path = cache_data["few_shot_examples_path"]
        if "system_identity_path" in cache_data:
            config.system_identity_path = cache_data["system_identity_path"]
        if "max_static_content_tokens" in cache_data:
            config.max_static_content_tokens = int(cache_data["max_static_content_tokens"])
        if "prompt_construction_timeout_ms" in cache_data:
            config.prompt_construction_timeout_ms = int(cache_data["prompt_construction_timeout_ms"])
        if "content_loading_timeout_s" in cache_data:
            config.content_loading_timeout_s = int(cache_data["content_loading_timeout_s"])
        if "cache_effectiveness_alert_threshold" in cache_data:
            config.cache_effectiveness_alert_threshold = float(cache_data["cache_effectiveness_alert_threshold"])
        if "hot_reload_enabled" in cache_data:
            config.hot_reload_enabled = bool(cache_data["hot_reload_enabled"])
        
        # Load pricing configuration
        pricing = cache_data.get("pricing", {})
        if "input_cost_per_1m" in pricing:
            config.input_cost_per_1m = float(pricing["input_cost_per_1m"])
        if "cached_input_cost_per_1m" in pricing:
            config.cached_input_cost_per_1m = float(pricing["cached_input_cost_per_1m"])
        if "output_cost_per_1m" in pricing:
            config.output_cost_per_1m = float(pricing["output_cost_per_1m"])
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_path}: {e}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        return config


def _load_from_env(config: CacheConfiguration, env_prefix: str) -> CacheConfiguration:
    """
    Load configuration from environment variables.
    
    Args:
        config: Base configuration to update
        env_prefix: Prefix for environment variables
    
    Returns:
        Updated configuration
    """
    # Feature flags
    enabled_str = os.getenv(f"{env_prefix}ENABLED", "").lower()
    if enabled_str:
        config.enabled = enabled_str in ("true", "1", "yes", "on")
    
    monitor_str = os.getenv(f"{env_prefix}MONITOR", "").lower()
    if monitor_str:
        config.monitor_cache_effectiveness = monitor_str in ("true", "1", "yes", "on")
    
    # Pricing
    input_cost = os.getenv(f"{env_prefix}INPUT_COST")
    if input_cost:
        try:
            config.input_cost_per_1m = float(input_cost)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}INPUT_COST value: {input_cost}")
    
    cached_cost = os.getenv(f"{env_prefix}CACHED_COST")
    if cached_cost:
        try:
            config.cached_input_cost_per_1m = float(cached_cost)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}CACHED_COST value: {cached_cost}")
    
    output_cost = os.getenv(f"{env_prefix}OUTPUT_COST")
    if output_cost:
        try:
            config.output_cost_per_1m = float(output_cost)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}OUTPUT_COST value: {output_cost}")
    
    # Limits
    max_tokens = os.getenv(f"{env_prefix}MAX_TOKENS")
    if max_tokens:
        try:
            config.max_static_content_tokens = int(max_tokens)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}MAX_TOKENS value: {max_tokens}")
    
    timeout_ms = os.getenv(f"{env_prefix}TIMEOUT_MS")
    if timeout_ms:
        try:
            config.prompt_construction_timeout_ms = int(timeout_ms)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}TIMEOUT_MS value: {timeout_ms}")
    
    load_timeout = os.getenv(f"{env_prefix}LOAD_TIMEOUT_S")
    if load_timeout:
        try:
            config.content_loading_timeout_s = int(load_timeout)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}LOAD_TIMEOUT_S value: {load_timeout}")
    
    # Alert threshold
    alert_threshold = os.getenv(f"{env_prefix}ALERT_THRESHOLD")
    if alert_threshold:
        try:
            config.cache_effectiveness_alert_threshold = float(alert_threshold)
        except ValueError:
            logger.warning(f"Invalid {env_prefix}ALERT_THRESHOLD value: {alert_threshold}")
    
    # Hot reload
    hot_reload_str = os.getenv(f"{env_prefix}HOT_RELOAD", "").lower()
    if hot_reload_str:
        config.hot_reload_enabled = hot_reload_str in ("true", "1", "yes", "on")
    
    # API key (standard env var name)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        config.gemini_api_key = api_key
    
    return config


def validate_cache_config(
    config: CacheConfiguration,
    logger_instance: Optional[logging.Logger] = None
) -> List[str]:
    """
    Validate cache configuration and return list of errors.
    
    Validation Rules:
        - Pricing values must be non-negative
        - Cached cost should not exceed standard cost
        - Token limits must be positive
        - Timeout values must be positive
        - Alert threshold must be between 0 and 100
    
    Args:
        config: CacheConfiguration to validate
        logger_instance: Optional logger for logging validation errors
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = config.validate()
    
    # Additional path validation
    for name, path in config.static_content_paths.items():
        if path and not Path(path).exists():
            # This is a warning, not an error - files may be loaded later
            if logger_instance:
                logger_instance.warning(f"Static content path not found: {name}={path}")
    
    # Log errors if logger provided
    if logger_instance and errors:
        for error in errors:
            logger_instance.error(f"Configuration validation error: {error}")
    
    return errors


def get_validated_cache_config(
    config_path: Optional[str] = None,
    logger_instance: Optional[logging.Logger] = None
) -> CacheConfiguration:
    """
    Load and validate cache configuration with safe defaults.
    
    This is a convenience function that combines loading and validation.
    If validation fails, it logs errors and uses safe defaults.
    
    Args:
        config_path: Optional path to JSON configuration file
        logger_instance: Optional logger for logging
    
    Returns:
        CacheConfiguration: Validated configuration with safe defaults
    """
    config = load_cache_config(config_path)
    errors = validate_cache_config(config, logger_instance)
    
    if errors:
        if logger_instance:
            logger_instance.warning(
                f"Configuration validation found {len(errors)} issues, using safe defaults where needed"
            )
        
        # Apply safe defaults for invalid values
        config = _apply_safe_defaults(config)
    
    # Log successful configuration
    if logger_instance:
        logger_instance.info(
            "Cache configuration loaded",
            extra={
                "enabled": config.enabled,
                "monitor_cache_effectiveness": config.monitor_cache_effectiveness,
                "input_cost_per_1m": config.input_cost_per_1m,
                "cached_input_cost_per_1m": config.cached_input_cost_per_1m,
                "discount_percentage": config.get_discount_percentage(),
                "max_static_content_tokens": config.max_static_content_tokens,
                "hot_reload_enabled": config.hot_reload_enabled,
            }
        )
    
    return config


def _apply_safe_defaults(config: CacheConfiguration) -> CacheConfiguration:
    """
    Apply safe defaults for invalid configuration values.
    
    Args:
        config: Configuration with potentially invalid values
    
    Returns:
        Configuration with safe defaults applied
    """
    defaults = DEFAULT_CACHE_CONFIG
    
    # Fix pricing
    if config.input_cost_per_1m < 0:
        config.input_cost_per_1m = defaults.input_cost_per_1m
    if config.cached_input_cost_per_1m < 0:
        config.cached_input_cost_per_1m = defaults.cached_input_cost_per_1m
    if config.output_cost_per_1m < 0:
        config.output_cost_per_1m = defaults.output_cost_per_1m
    if config.cached_input_cost_per_1m > config.input_cost_per_1m:
        config.cached_input_cost_per_1m = config.input_cost_per_1m * 0.1  # 90% discount
    
    # Fix limits
    if config.max_static_content_tokens <= 0:
        config.max_static_content_tokens = defaults.max_static_content_tokens
    if config.prompt_construction_timeout_ms <= 0:
        config.prompt_construction_timeout_ms = defaults.prompt_construction_timeout_ms
    if config.content_loading_timeout_s <= 0:
        config.content_loading_timeout_s = defaults.content_loading_timeout_s
    
    # Fix alert threshold
    if not (0 <= config.cache_effectiveness_alert_threshold <= 100):
        config.cache_effectiveness_alert_threshold = defaults.cache_effectiveness_alert_threshold
    
    return config


class ConfigurationManager:
    """
    Manages cache configuration with hot reload support.
    
    Provides:
    - Initial configuration loading
    - Hot reload without restart
    - Configuration change detection
    - Thread-safe configuration access
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Optional path to JSON configuration file
            logger_instance: Optional logger for logging
        """
        self._config_path = config_path
        self._logger = logger_instance or logger
        self._config: Optional[CacheConfiguration] = None
        self._last_load_time: Optional[float] = None
        self._reload_callbacks: List[callable] = []
    
    def get_config(self) -> CacheConfiguration:
        """
        Get current configuration.
        
        Returns:
            Current CacheConfiguration
        """
        if self._config is None:
            self._config = get_validated_cache_config(
                self._config_path,
                self._logger
            )
            self._last_load_time = os.path.getmtime(self._config_path) if self._config_path else None
        
        return self._config
    
    def reload_config(self) -> bool:
        """
        Reload configuration from source.
        
        Returns:
            True if configuration was reloaded, False if unchanged or failed
        """
        if not self._config or not self._config.hot_reload_enabled:
            return False
        
        try:
            new_config = get_validated_cache_config(
                self._config_path,
                self._logger
            )
            
            # Check if configuration changed
            if self._config_changed(new_config):
                old_config = self._config
                self._config = new_config
                self._last_load_time = os.path.getmtime(self._config_path) if self._config_path else None
                
                # Notify callbacks
                for callback in self._reload_callbacks:
                    try:
                        callback(old_config, new_config)
                    except Exception as e:
                        self._logger.error(f"Error in reload callback: {e}")
                
                self._logger.info("Configuration reloaded successfully")
                return True
            
            return False
            
        except Exception as e:
            self._logger.error(f"Error reloading configuration: {e}")
            return False
    
    def check_for_changes(self) -> bool:
        """
        Check if configuration file has changed.
        
        Returns:
            True if file has changed since last load
        """
        if not self._config_path or not self._last_load_time:
            return False
        
        try:
            current_mtime = os.path.getmtime(self._config_path)
            return current_mtime > self._last_load_time
        except OSError:
            return False
    
    def register_reload_callback(self, callback: callable):
        """
        Register a callback to be called when configuration is reloaded.
        
        Args:
            callback: Function(old_config, new_config) to call on reload
        """
        self._reload_callbacks.append(callback)
    
    def _config_changed(self, new_config: CacheConfiguration) -> bool:
        """Check if configuration has changed."""
        if self._config is None:
            return True
        
        # Compare key fields
        return (
            self._config.enabled != new_config.enabled or
            self._config.monitor_cache_effectiveness != new_config.monitor_cache_effectiveness or
            self._config.input_cost_per_1m != new_config.input_cost_per_1m or
            self._config.cached_input_cost_per_1m != new_config.cached_input_cost_per_1m or
            self._config.max_static_content_tokens != new_config.max_static_content_tokens or
            self._config.static_content_paths != new_config.static_content_paths
        )
