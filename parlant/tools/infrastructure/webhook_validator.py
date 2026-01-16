"""
Webhook signature validation for Freshdesk webhooks.

This module provides HMAC-SHA256 signature validation to ensure
webhook requests are authentic and come from Freshdesk.
"""

import hmac
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def validate_freshdesk_signature(
    payload: bytes,
    signature: Optional[str],
    secret: str
) -> bool:
    """
    Validates Freshdesk webhook signature using HMAC-SHA256.
    
    Args:
        payload: Raw request body as bytes
        signature: X-Freshdesk-Signature header value (can be None)
        secret: Webhook secret from environment configuration
        
    Returns:
        True if signature is valid, False otherwise
        
    Example:
        >>> payload = b'{"ticket_id": "12345"}'
        >>> signature = "abc123..."
        >>> secret = "my-webhook-secret"
        >>> validate_freshdesk_signature(payload, signature, secret)
        True
    """
    # Handle missing signature
    if signature is None or signature == "":
        logger.warning("Missing webhook signature header")
        return False
    
    # Handle missing or empty secret
    if not secret:
        logger.error("Webhook secret not configured")
        return False
    
    try:
        # Compute expected signature using HMAC-SHA256
        expected_signature = hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()
        
        # Constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(signature, expected_signature)
        
        if not is_valid:
            logger.warning(
                "Invalid webhook signature",
                extra={
                    "received_signature": signature[:10] + "...",
                    "expected_signature": expected_signature[:10] + "..."
                }
            )
        
        return is_valid
        
    except Exception as e:
        logger.error(f"Error validating webhook signature: {e}")
        return False
