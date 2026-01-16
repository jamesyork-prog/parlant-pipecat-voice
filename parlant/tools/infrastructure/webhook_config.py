"""
Webhook configuration management module.

This module centralizes all webhook-related configuration, including:
- Environment variable loading
- Configuration validation
- Default values
- Event filtering configuration
"""

import os
import logging
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class WebhookConfig:
    """
    Webhook configuration container.
    
    Attributes:
        enabled: Whether webhook processing is enabled
        secret: Secret key for signature verification
        port: Port number for webhook endpoint
        path: URL path for webhook endpoint
        events: List of event types to process
        log_level: Logging level for webhook operations
        rate_limit: Maximum requests per minute
        rate_limit_window: Time window for rate limiting in seconds
        deduplication_window: Time window for event deduplication in seconds
    """
    enabled: bool
    secret: Optional[str]
    port: int
    path: str
    events: List[str]
    log_level: str
    rate_limit: int
    rate_limit_window: int
    deduplication_window: int


def load_webhook_config() -> WebhookConfig:
    """
    Load webhook configuration from environment variables.
    
    Environment Variables:
        WEBHOOK_ENABLED: Enable/disable webhook processing (default: true)
        WEBHOOK_SECRET: Secret key for signature verification (required if enabled)
        WEBHOOK_PORT: Port number for webhook endpoint (default: 8801)
        WEBHOOK_PATH: URL path for webhook endpoint (default: /webhook/freshdesk)
        WEBHOOK_EVENTS: Comma-separated list of events to process (default: ticket_created,ticket_updated)
        WEBHOOK_LOG_LEVEL: Logging level (default: INFO)
        WEBHOOK_RATE_LIMIT: Max requests per minute (default: 100)
        WEBHOOK_RATE_LIMIT_WINDOW: Rate limit window in seconds (default: 60)
        WEBHOOK_DEDUPLICATION_WINDOW: Event deduplication window in seconds (default: 60)
    
    Returns:
        WebhookConfig: Configuration object with all settings
    """
    # Load enabled flag
    enabled_str = os.getenv("WEBHOOK_ENABLED", "true").lower()
    enabled = enabled_str in ("true", "1", "yes", "on")
    
    # Load secret (optional if webhooks disabled)
    secret = os.getenv("WEBHOOK_SECRET")
    
    # Load port with validation
    port_str = os.getenv("WEBHOOK_PORT", "8801")
    try:
        port = int(port_str)
    except ValueError:
        port = 8801
    
    # Load path
    path = os.getenv("WEBHOOK_PATH", "/webhook/freshdesk")
    
    # Load and parse event list
    events_str = os.getenv("WEBHOOK_EVENTS", "ticket_created,ticket_updated")
    events = [e.strip() for e in events_str.split(",") if e.strip()]
    
    # Load log level
    log_level = os.getenv("WEBHOOK_LOG_LEVEL", "INFO").upper()
    
    # Load rate limiting configuration
    rate_limit_str = os.getenv("WEBHOOK_RATE_LIMIT", "100")
    try:
        rate_limit = int(rate_limit_str)
    except ValueError:
        rate_limit = 100
    
    rate_limit_window_str = os.getenv("WEBHOOK_RATE_LIMIT_WINDOW", "60")
    try:
        rate_limit_window = int(rate_limit_window_str)
    except ValueError:
        rate_limit_window = 60
    
    # Load deduplication window
    dedup_window_str = os.getenv("WEBHOOK_DEDUPLICATION_WINDOW", "60")
    try:
        deduplication_window = int(dedup_window_str)
    except ValueError:
        deduplication_window = 60
    
    return WebhookConfig(
        enabled=enabled,
        secret=secret,
        port=port,
        path=path,
        events=events,
        log_level=log_level,
        rate_limit=rate_limit,
        rate_limit_window=rate_limit_window,
        deduplication_window=deduplication_window
    )


def validate_webhook_config(config: WebhookConfig, logger: Optional[logging.Logger] = None) -> List[str]:
    """
    Validate webhook configuration and return list of errors.
    
    Validation Rules:
        - If enabled, WEBHOOK_SECRET must be set
        - WEBHOOK_PORT must be a valid port number (1-65535)
        - WEBHOOK_PATH must start with /
        - WEBHOOK_EVENTS must not be empty
        - WEBHOOK_RATE_LIMIT must be positive
        - WEBHOOK_RATE_LIMIT_WINDOW must be positive
        - WEBHOOK_DEDUPLICATION_WINDOW must be positive
    
    Args:
        config: WebhookConfig object to validate
        logger: Optional logger for logging validation errors
    
    Returns:
        List of error messages (empty list if valid)
    """
    errors = []
    
    # Validate secret if webhooks are enabled
    if config.enabled and not config.secret:
        errors.append("WEBHOOK_SECRET must be set when WEBHOOK_ENABLED is true")
    
    # Validate port number
    if not (1 <= config.port <= 65535):
        errors.append(f"WEBHOOK_PORT must be between 1 and 65535 (got: {config.port})")
    
    # Validate path starts with /
    if not config.path.startswith("/"):
        errors.append(f"WEBHOOK_PATH must start with / (got: {config.path})")
    
    # Validate events list is not empty
    if not config.events:
        errors.append("WEBHOOK_EVENTS must not be empty")
    
    # Validate rate limit is positive
    if config.rate_limit <= 0:
        errors.append(f"WEBHOOK_RATE_LIMIT must be positive (got: {config.rate_limit})")
    
    # Validate rate limit window is positive
    if config.rate_limit_window <= 0:
        errors.append(f"WEBHOOK_RATE_LIMIT_WINDOW must be positive (got: {config.rate_limit_window})")
    
    # Validate deduplication window is positive
    if config.deduplication_window <= 0:
        errors.append(f"WEBHOOK_DEDUPLICATION_WINDOW must be positive (got: {config.deduplication_window})")
    
    # Log errors if logger provided
    if logger and errors:
        for error in errors:
            logger.error(f"Configuration validation error: {error}")
    
    return errors


def get_validated_config(logger: Optional[logging.Logger] = None) -> WebhookConfig:
    """
    Load and validate webhook configuration.
    
    This is a convenience function that combines loading and validation.
    If validation fails, it logs errors and uses safe defaults where possible.
    
    Args:
        logger: Optional logger for logging validation errors
    
    Returns:
        WebhookConfig: Validated configuration object
    
    Raises:
        ValueError: If critical validation errors prevent safe operation
    """
    config = load_webhook_config()
    errors = validate_webhook_config(config, logger)
    
    if errors:
        error_msg = f"Webhook configuration validation failed: {'; '.join(errors)}"
        if logger:
            logger.error(error_msg)
        
        # If webhooks are enabled but secret is missing, this is critical
        if config.enabled and not config.secret:
            raise ValueError("Cannot enable webhooks without WEBHOOK_SECRET")
        
        # For other errors, log warnings but continue with defaults
        if logger:
            logger.warning("Continuing with default configuration values where possible")
    
    # Log successful configuration
    if logger:
        logger.info(
            "Webhook configuration loaded",
            extra={
                "enabled": config.enabled,
                "port": config.port,
                "path": config.path,
                "events": config.events,
                "rate_limit": config.rate_limit,
                "has_secret": bool(config.secret)
            }
        )
    
    return config
