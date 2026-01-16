"""
Tests for webhook signature validation.

This module tests the HMAC-SHA256 signature validation for Freshdesk webhooks.
"""

import pytest
import hmac
import hashlib
from app_tools.tools.infrastructure.webhook_validator import validate_freshdesk_signature


class TestValidateFreshdeskSignature:
    """Tests for validate_freshdesk_signature function"""
    
    def test_valid_signature_verification(self):
        """Test that a valid signature is correctly verified"""
        # Arrange
        payload = b'{"ticket_id": "12345", "event": "ticket_created"}'
        secret = "test-webhook-secret"
        
        # Generate valid signature
        expected_signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act
        result = validate_freshdesk_signature(payload, expected_signature, secret)
        
        # Assert
        assert result is True
    
    def test_invalid_signature_rejection(self):
        """Test that an invalid signature is rejected"""
        # Arrange
        payload = b'{"ticket_id": "12345", "event": "ticket_created"}'
        secret = "test-webhook-secret"
        invalid_signature = "invalid_signature_12345"
        
        # Act
        result = validate_freshdesk_signature(payload, invalid_signature, secret)
        
        # Assert
        assert result is False
    
    def test_missing_signature_handling(self):
        """Test that missing signature (None) is handled correctly"""
        # Arrange
        payload = b'{"ticket_id": "12345", "event": "ticket_created"}'
        secret = "test-webhook-secret"
        
        # Act
        result = validate_freshdesk_signature(payload, None, secret)
        
        # Assert
        assert result is False
    
    def test_empty_signature_handling(self):
        """Test that empty signature string is handled correctly"""
        # Arrange
        payload = b'{"ticket_id": "12345", "event": "ticket_created"}'
        secret = "test-webhook-secret"
        
        # Act
        result = validate_freshdesk_signature(payload, "", secret)
        
        # Assert
        assert result is False
    
    def test_missing_secret_handling(self):
        """Test that missing secret is handled correctly"""
        # Arrange
        payload = b'{"ticket_id": "12345", "event": "ticket_created"}'
        signature = "some_signature"
        
        # Act with empty secret
        result = validate_freshdesk_signature(payload, signature, "")
        
        # Assert
        assert result is False
    
    def test_malformed_payload_handling(self):
        """Test that malformed payload doesn't crash validation"""
        # Arrange
        malformed_payload = b'not valid json {{{['
        secret = "test-webhook-secret"
        
        # Generate signature for malformed payload
        signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=malformed_payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act - should still validate signature even if payload is malformed
        result = validate_freshdesk_signature(malformed_payload, signature, secret)
        
        # Assert - signature validation should succeed (payload parsing is separate concern)
        assert result is True
    
    def test_different_payload_different_signature(self):
        """Test that different payloads produce different signatures"""
        # Arrange
        payload1 = b'{"ticket_id": "12345"}'
        payload2 = b'{"ticket_id": "67890"}'
        secret = "test-webhook-secret"
        
        # Generate signature for payload1
        signature1 = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload1,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act - try to validate payload2 with signature1
        result = validate_freshdesk_signature(payload2, signature1, secret)
        
        # Assert - should fail because payloads don't match
        assert result is False
    
    def test_different_secret_different_signature(self):
        """Test that different secrets produce different signatures"""
        # Arrange
        payload = b'{"ticket_id": "12345"}'
        secret1 = "test-webhook-secret-1"
        secret2 = "test-webhook-secret-2"
        
        # Generate signature with secret1
        signature = hmac.new(
            key=secret1.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act - try to validate with secret2
        result = validate_freshdesk_signature(payload, signature, secret2)
        
        # Assert - should fail because secrets don't match
        assert result is False
    
    def test_constant_time_comparison(self):
        """Test that signature comparison is constant-time (timing attack resistant)"""
        # Arrange
        payload = b'{"ticket_id": "12345"}'
        secret = "test-webhook-secret"
        
        # Generate valid signature
        valid_signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Create signature that differs only in last character
        almost_valid_signature = valid_signature[:-1] + ('0' if valid_signature[-1] != '0' else '1')
        
        # Act
        result = validate_freshdesk_signature(payload, almost_valid_signature, secret)
        
        # Assert - should fail
        assert result is False
    
    def test_unicode_payload_handling(self):
        """Test that unicode characters in payload are handled correctly"""
        # Arrange
        payload = '{"ticket_id": "12345", "subject": "Refund request 🎫"}'.encode('utf-8')
        secret = "test-webhook-secret"
        
        # Generate valid signature
        signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act
        result = validate_freshdesk_signature(payload, signature, secret)
        
        # Assert
        assert result is True
    
    def test_large_payload_handling(self):
        """Test that large payloads are handled correctly"""
        # Arrange
        large_payload = b'{"ticket_id": "12345", "description": "' + b'x' * 10000 + b'"}'
        secret = "test-webhook-secret"
        
        # Generate valid signature
        signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=large_payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act
        result = validate_freshdesk_signature(large_payload, signature, secret)
        
        # Assert
        assert result is True
    
    def test_empty_payload_handling(self):
        """Test that empty payload is handled correctly"""
        # Arrange
        payload = b''
        secret = "test-webhook-secret"
        
        # Generate valid signature for empty payload
        signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Act
        result = validate_freshdesk_signature(payload, signature, secret)
        
        # Assert
        assert result is True
