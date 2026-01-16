"""
Tests for webhook configuration management.

This module tests the webhook configuration loading, validation, and error handling.
"""

import pytest
import os
from unittest.mock import patch
from app_tools.tools.infrastructure.webhook_config import (
    load_webhook_config,
    validate_webhook_config,
    get_validated_config,
    WebhookConfig
)


class TestLoadWebhookConfig:
    """Tests for load_webhook_config function"""
    
    def test_load_default_config(self, monkeypatch):
        """Test loading configuration with all defaults"""
        # Clear all webhook env vars
        for key in list(os.environ.keys()):
            if key.startswith("WEBHOOK_"):
                monkeypatch.delenv(key, raising=False)
        
        config = load_webhook_config()
        
        assert config.enabled is True
        assert config.secret is None
        assert config.port == 8801
        assert config.path == "/webhook/freshdesk"
        assert config.events == ["ticket_created", "ticket_updated"]
        assert config.log_level == "INFO"
        assert config.rate_limit == 100
        assert config.rate_limit_window == 60
        assert config.deduplication_window == 60
    
    def test_load_custom_config(self, monkeypatch):
        """Test loading configuration with custom values"""
        monkeypatch.setenv("WEBHOOK_ENABLED", "true")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret-123")
        monkeypatch.setenv("WEBHOOK_PORT", "9000")
        monkeypatch.setenv("WEBHOOK_PATH", "/custom/webhook")
        monkeypatch.setenv("WEBHOOK_EVENTS", "ticket_created,note_added")
        monkeypatch.setenv("WEBHOOK_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("WEBHOOK_RATE_LIMIT", "200")
        monkeypatch.setenv("WEBHOOK_RATE_LIMIT_WINDOW", "120")
        monkeypatch.setenv("WEBHOOK_DEDUPLICATION_WINDOW", "90")
        
        config = load_webhook_config()
        
        assert config.enabled is True
        assert config.secret == "test-secret-123"
        assert config.port == 9000
        assert config.path == "/custom/webhook"
        assert config.events == ["ticket_created", "note_added"]
        assert config.log_level == "DEBUG"
        assert config.rate_limit == 200
        assert config.rate_limit_window == 120
        assert config.deduplication_window == 90
    
    def test_load_disabled_config(self, monkeypatch):
        """Test loading configuration with webhooks disabled"""
        monkeypatch.setenv("WEBHOOK_ENABLED", "false")
        
        config = load_webhook_config()
        
        assert config.enabled is False
    
    def test_enabled_variations(self, monkeypatch):
        """Test various ways to enable/disable webhooks"""
        # Test true values
        for value in ["true", "True", "TRUE", "1", "yes", "YES", "on", "ON"]:
            monkeypatch.setenv("WEBHOOK_ENABLED", value)
            config = load_webhook_config()
            assert config.enabled is True, f"Failed for value: {value}"
        
        # Test false values
        for value in ["false", "False", "FALSE", "0", "no", "NO", "off", "OFF"]:
            monkeypatch.setenv("WEBHOOK_ENABLED", value)
            config = load_webhook_config()
            assert config.enabled is False, f"Failed for value: {value}"
    
    def test_invalid_port_uses_default(self, monkeypatch):
        """Test that invalid port number falls back to default"""
        monkeypatch.setenv("WEBHOOK_PORT", "invalid")
        
        config = load_webhook_config()
        
        assert config.port == 8801
    
    def test_invalid_rate_limit_uses_default(self, monkeypatch):
        """Test that invalid rate limit falls back to default"""
        monkeypatch.setenv("WEBHOOK_RATE_LIMIT", "not-a-number")
        
        config = load_webhook_config()
        
        assert config.rate_limit == 100
    
    def test_empty_events_list(self, monkeypatch):
        """Test handling of empty events list"""
        monkeypatch.setenv("WEBHOOK_EVENTS", "")
        
        config = load_webhook_config()
        
        assert config.events == []
    
    def test_events_with_whitespace(self, monkeypatch):
        """Test that event list handles whitespace correctly"""
        monkeypatch.setenv("WEBHOOK_EVENTS", " ticket_created , ticket_updated , note_added ")
        
        config = load_webhook_config()
        
        assert config.events == ["ticket_created", "ticket_updated", "note_added"]


class TestValidateWebhookConfig:
    """Tests for validate_webhook_config function"""
    
    def test_valid_config(self):
        """Test validation of a valid configuration"""
        config = WebhookConfig(
            enabled=True,
            secret="test-secret",
            port=8801,
            path="/webhook/freshdesk",
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert errors == []
    
    def test_missing_secret_when_enabled(self):
        """Test validation fails when secret is missing but webhooks enabled"""
        config = WebhookConfig(
            enabled=True,
            secret=None,
            port=8801,
            path="/webhook/freshdesk",
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert len(errors) == 1
        assert "WEBHOOK_SECRET must be set" in errors[0]
    
    def test_missing_secret_when_disabled_is_ok(self):
        """Test validation passes when secret is missing but webhooks disabled"""
        config = WebhookConfig(
            enabled=False,
            secret=None,
            port=8801,
            path="/webhook/freshdesk",
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert errors == []
    
    def test_invalid_port_number(self):
        """Test validation fails for invalid port numbers"""
        # Port too low
        config = WebhookConfig(
            enabled=True,
            secret="test",
            port=0,
            path="/webhook/freshdesk",
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        errors = validate_webhook_config(config)
        assert any("WEBHOOK_PORT must be between" in e for e in errors)
        
        # Port too high
        config.port = 70000
        errors = validate_webhook_config(config)
        assert any("WEBHOOK_PORT must be between" in e for e in errors)
    
    def test_invalid_path(self):
        """Test validation fails when path doesn't start with /"""
        config = WebhookConfig(
            enabled=True,
            secret="test",
            port=8801,
            path="webhook/freshdesk",  # Missing leading /
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert any("WEBHOOK_PATH must start with /" in e for e in errors)
    
    def test_empty_events_list(self):
        """Test validation fails when events list is empty"""
        config = WebhookConfig(
            enabled=True,
            secret="test",
            port=8801,
            path="/webhook/freshdesk",
            events=[],
            log_level="INFO",
            rate_limit=100,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert any("WEBHOOK_EVENTS must not be empty" in e for e in errors)
    
    def test_negative_rate_limit(self):
        """Test validation fails for negative rate limit"""
        config = WebhookConfig(
            enabled=True,
            secret="test",
            port=8801,
            path="/webhook/freshdesk",
            events=["ticket_created"],
            log_level="INFO",
            rate_limit=-1,
            rate_limit_window=60,
            deduplication_window=60
        )
        
        errors = validate_webhook_config(config)
        
        assert any("WEBHOOK_RATE_LIMIT must be positive" in e for e in errors)
    
    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are all reported"""
        config = WebhookConfig(
            enabled=True,
            secret=None,  # Error 1
            port=0,  # Error 2
            path="no-slash",  # Error 3
            events=[],  # Error 4
            log_level="INFO",
            rate_limit=-1,  # Error 5
            rate_limit_window=0,  # Error 6
            deduplication_window=-10  # Error 7
        )
        
        errors = validate_webhook_config(config)
        
        assert len(errors) == 7


class TestGetValidatedConfig:
    """Tests for get_validated_config function"""
    
    def test_valid_config_loads_successfully(self, monkeypatch):
        """Test that valid configuration loads without errors"""
        monkeypatch.setenv("WEBHOOK_ENABLED", "true")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("WEBHOOK_PORT", "8801")
        
        config = get_validated_config()
        
        assert config.enabled is True
        assert config.secret == "test-secret"
    
    def test_missing_secret_raises_error(self, monkeypatch):
        """Test that missing secret when enabled raises ValueError"""
        monkeypatch.setenv("WEBHOOK_ENABLED", "true")
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        
        with pytest.raises(ValueError, match="Cannot enable webhooks without WEBHOOK_SECRET"):
            get_validated_config()
    
    def test_disabled_webhooks_with_no_secret_is_ok(self, monkeypatch):
        """Test that disabled webhooks don't require secret"""
        monkeypatch.setenv("WEBHOOK_ENABLED", "false")
        monkeypatch.delenv("WEBHOOK_SECRET", raising=False)
        
        config = get_validated_config()
        
        assert config.enabled is False
        assert config.secret is None
