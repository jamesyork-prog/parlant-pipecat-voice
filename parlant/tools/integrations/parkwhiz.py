"""
ParkWhiz API Client

Handles authentication and communication with the ParkWhiz API using OAuth2.
Supports both sandbox and production environments.
"""

import os
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import httpx
from cachetools import TTLCache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)


# Configure logging
logger = logging.getLogger(__name__)


# Custom Exceptions
class ParkWhizError(Exception):
    """Base exception for ParkWhiz API errors"""
    pass


class ParkWhizAuthenticationError(ParkWhizError):
    """Raised when authentication fails"""
    pass


class ParkWhizNotFoundError(ParkWhizError):
    """Raised when a resource is not found (404)"""
    pass


class ParkWhizTimeoutError(ParkWhizError):
    """Raised when a request times out"""
    pass


class ParkWhizRateLimitError(ParkWhizError):
    """Raised when rate limit is exceeded"""
    pass


class ParkWhizValidationError(ParkWhizError):
    """Raised when request validation fails"""
    pass


class ParkWhizOAuth2Client:
    """
    OAuth2-based client for ParkWhiz API.
    
    Uses OAuth2 Client Credentials flow for authentication.
    Automatically manages token lifecycle (generation and refresh).
    
    CRITICAL LIMITATION: The ParkWhiz API does NOT support searching bookings
    by customer email. This client can only retrieve bookings by known booking ID.
    See planning/parkwhiz_api_limitations.md for details.
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        scope: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        """
        Initialize OAuth2 client for ParkWhiz API.
        
        Args:
            client_id: OAuth2 client ID (reads from env if not provided)
            client_secret: OAuth2 client secret (reads from env if not provided)
            scope: OAuth2 scope (reads from env if not provided, default: "internal")
            base_url: API base URL (reads from env if not provided)
            timeout: Request timeout in seconds (reads from env if not provided)
            max_retries: Maximum retry attempts (reads from env if not provided)
        
        Raises:
            ParkWhizAuthenticationError: If credentials are not configured
        """
        # Load configuration from environment
        self.client_id = client_id or os.getenv("PARKWHIZ_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("PARKWHIZ_CLIENT_SECRET")
        self.scope = scope or os.getenv("PARKWHIZ_SCOPE", "partner")
        
        # Determine base URL based on environment
        env = os.getenv("PARKWHIZ_ENV", "sandbox")
        if env == "production":
            default_url = os.getenv("PARKWHIZ_PRODUCTION_URL", "https://api.parkwhiz.com/v4")
        else:
            default_url = os.getenv("PARKWHIZ_SANDBOX_URL", "https://api-sandbox.parkwhiz.com/v4")
        
        self.base_url = base_url or default_url
        self.timeout = timeout or int(os.getenv("PARKWHIZ_TIMEOUT", "30"))  # Increased for sandbox
        self.max_retries = max_retries or int(os.getenv("PARKWHIZ_MAX_RETRIES", "3"))
        
        # Validate required credentials
        if not self.client_id or not self.client_secret:
            logger.critical(
                "ParkWhiz OAuth2 credentials not configured. "
                "Set PARKWHIZ_CLIENT_ID and PARKWHIZ_CLIENT_SECRET environment variables."
            )
            raise ParkWhizAuthenticationError(
                "ParkWhiz OAuth2 credentials not configured. "
                "Set PARKWHIZ_CLIENT_ID and PARKWHIZ_CLIENT_SECRET environment variables. "
                "Contact dev-admin@parkwhiz.com to obtain credentials."
            )
        
        # Token management
        self._token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        # Cache configuration
        cache_ttl = int(os.getenv("PARKWHIZ_CACHE_TTL", "120"))  # 2 minutes default
        self._cache = TTLCache(maxsize=100, ttl=cache_ttl)
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        
        logger.info(
            "ParkWhiz OAuth2 client initialized",
            extra={
                "base_url": self.base_url,
                "scope": self.scope,
                "timeout": self.timeout,
                "cache_ttl": cache_ttl,
            }
        )
    
    async def _ensure_valid_token(self):
        """
        Ensure we have a valid OAuth2 access token.
        
        Automatically refreshes token if it's expired or about to expire (within 24 hours).
        """
        if self._token and self._token_expires_at:
            # Check if token expires in next 24 hours
            if datetime.now() < self._token_expires_at - timedelta(hours=24):
                logger.debug("Using existing OAuth2 token")
                return
        
        # Token is missing or expiring soon - refresh it
        await self._refresh_token()
    
    async def _refresh_token(self):
        """
        Request new OAuth2 access token using Client Credentials flow.
        
        Per ParkWhiz API docs: https://developer.parkwhiz.com/v4/#authentication
        - Token endpoint: POST /v4/oauth/token (under /v4, not at root)
        - Grant type: client_credentials
        - Auth: client_id and client_secret in request body
        
        Raises:
            ParkWhizAuthenticationError: If token request fails
        """
        logger.info("Requesting new OAuth2 token")
        
        # ParkWhiz token endpoint is under /v4/oauth/token
        # Build the full URL including /v4
        token_base_url = self.base_url.rstrip('/')
        token_url = f"{token_base_url}/oauth/token"
        
        logger.info(f"OAuth2 token URL: {token_url}")
        
        try:
            # ParkWhiz requires credentials in request body (not Basic Auth)
            response = await self.client.post(
                token_url,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": self.scope,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                }
            )
            
            if response.status_code != 200:
                logger.critical(
                    f"OAuth2 token request failed: {response.status_code}",
                    extra={"status_code": response.status_code, "response": response.text}
                )
                raise ParkWhizAuthenticationError(
                    f"OAuth2 token request failed with status {response.status_code}: {response.text}"
                )
            
            data = response.json()
            self._token = data["access_token"]
            expires_in = data.get("expires_in", 31557600)  # Default: 1 year
            self._token_expires_at = datetime.now() + timedelta(seconds=expires_in)
            
            logger.info(
                f"OAuth2 token refreshed successfully, expires at {self._token_expires_at}",
                extra={"expires_at": self._token_expires_at.isoformat()}
            )
            
        except KeyError as e:
            logger.critical(f"Malformed OAuth2 token response: missing {e}")
            raise ParkWhizAuthenticationError(f"Malformed OAuth2 token response: missing {e}")
        
        except Exception as e:
            logger.critical(f"OAuth2 token refresh failed: {e}", exc_info=True)
            raise ParkWhizAuthenticationError(f"OAuth2 token refresh failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        reraise=True,
    )
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make authenticated OAuth2 request to ParkWhiz API.
        
        Args:
            method: HTTP method
            endpoint: API endpoint path
            params: Query parameters
            json_data: JSON request body
        
        Returns:
            JSON response as dictionary
        
        Raises:
            ParkWhizAuthenticationError: Authentication failed
            ParkWhizNotFoundError: Resource not found
            ParkWhizTimeoutError: Request timed out
            ParkWhizError: Other API errors
        """
        # Ensure we have a valid token
        await self._ensure_valid_token()
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Build full URL
        url = f"{self.base_url}{endpoint}"
        
        # Log request
        logger.info(
            f"ParkWhiz OAuth2 API request: {method} {endpoint}",
            extra={"method": method, "endpoint": endpoint, "params": params}
        )
        
        start_time = time.time()
        
        try:
            # Make request
            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_data,
            )
            
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"ParkWhiz OAuth2 API response: {response.status_code}",
                extra={
                    "status_code": response.status_code,
                    "duration_ms": int(duration * 1000),
                    "endpoint": endpoint,
                }
            )
            
            # Performance warning
            if duration > 3.0:
                logger.warning(
                    f"ParkWhiz API slow response: {duration:.2f}s",
                    extra={"duration_seconds": duration, "endpoint": endpoint}
                )
            
            # Handle error responses
            if response.status_code == 401:
                raise ParkWhizAuthenticationError(
                    f"OAuth2 authentication failed: {response.text}"
                )
            elif response.status_code == 404:
                raise ParkWhizNotFoundError(f"Resource not found: {endpoint}")
            elif response.status_code == 429:
                raise ParkWhizRateLimitError("Rate limit exceeded")
            elif response.status_code == 400:
                raise ParkWhizValidationError(f"Validation error: {response.text}")
            elif response.status_code >= 400:
                raise ParkWhizError(
                    f"API error {response.status_code}: {response.text}"
                )
            
            # Parse JSON response (handle empty responses for DELETE)
            if response.status_code == 204 or not response.content:
                # 204 No Content or empty response - success
                return {"success": True, "status_code": response.status_code}
            
            return response.json()
            
        except httpx.TimeoutException as e:
            # Log full error details with stack trace (Requirement 11.5)
            logger.error(
                f"ParkWhiz API timeout: {e}",
                extra={
                    "endpoint": endpoint,
                    "error_type": "timeout",
                    "method": method,
                },
                exc_info=True
            )
            raise ParkWhizTimeoutError(f"Request timed out: {endpoint}")
        
        except httpx.NetworkError as e:
            # Log full error details with stack trace (Requirement 11.5)
            logger.error(
                f"ParkWhiz API network error: {e}",
                extra={
                    "endpoint": endpoint,
                    "error_type": "network_error",
                    "method": method,
                },
                exc_info=True
            )
            raise ParkWhizError(f"Network error: {e}")
    
    async def get_booking_by_id(self, booking_id: str) -> Dict[str, Any]:
        """
        Get a single booking by its ID using OAuth2 authentication.
        
        This is the ONLY reliable way to retrieve booking details from ParkWhiz API.
        Email/customer search does not work (see parkwhiz_api_limitations.md).
        
        Args:
            booking_id: The booking ID to retrieve
        
        Returns:
            Booking dictionary with full details
        
        Raises:
            ParkWhizNotFoundError: Booking not found
            ParkWhizError: API error occurred
        """
        request_timestamp = datetime.now().isoformat()
        logger.info(
            f"ParkWhiz API request: get_booking_by_id {booking_id}",
            extra={
                "booking_id": booking_id,
                "timestamp": request_timestamp,
                "operation": "get_booking_by_id",
            }
        )
        
        start_time = time.time()
        
        try:
            response = await self._request("GET", f"/bookings/{booking_id}")
            
            processing_time = time.time() - start_time
            processing_time_ms = int(processing_time * 1000)
            
            logger.info(
                f"Retrieved booking {booking_id}",
                extra={
                    "booking_id": booking_id,
                    "processing_time_ms": processing_time_ms,
                    "response_status": "success",
                }
            )
            
            if processing_time > 3.0:
                logger.warning(
                    f"Slow API response for get_booking_by_id: {processing_time:.2f}s",
                    extra={
                        "booking_id": booking_id,
                        "processing_time_seconds": processing_time,
                        "processing_time_ms": processing_time_ms,
                        "threshold_seconds": 3.0,
                        "operation": "get_booking_by_id",
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(
                f"Failed to retrieve booking {booking_id}: {e}",
                extra={"booking_id": booking_id, "error": str(e)}
            )
            raise

    async def delete_booking(self, booking_id: str) -> Dict[str, Any]:
        """
        Delete (cancel/refund) a booking using OAuth2 authentication.
        
        Args:
            booking_id: Booking ID to delete
        
        Returns:
            Deletion confirmation with refund details
        
        Raises:
            ParkWhizNotFoundError: Booking not found
            ParkWhizError: API error occurred
        """
        # Log request with timestamp (Requirement 11.1)
        request_timestamp = datetime.now().isoformat()
        logger.info(
            f"ParkWhiz API request: delete_booking {booking_id}",
            extra={
                "booking_id": booking_id,
                "timestamp": request_timestamp,
                "operation": "delete_booking",
            }
        )
        
        # Track processing time
        start_time = time.time()
        
        # Make DELETE request (may return empty response)
        try:
            response = await self._request("DELETE", f"/bookings/{booking_id}")
        except Exception as e:
            # If it's a JSON decode error on empty response, that's actually success
            if "Expecting value" in str(e) or "JSONDecodeError" in str(e):
                logger.info(
                    f"DELETE request succeeded with empty response (204 No Content) for booking {booking_id}",
                    extra={"booking_id": booking_id}
                )
                response = {"success": True, "booking_id": booking_id, "status": "deleted"}
            else:
                raise
        
        # Calculate processing time
        processing_time = time.time() - start_time
        processing_time_ms = int(processing_time * 1000)
        
        # Extract refund amount from response
        refund_amount = None
        if isinstance(response, dict):
            refund_amount = (
                response.get("refund_amount") or
                response.get("amount") or
                response.get("price_paid")
            )
        
        # Log refund confirmation (Requirement 11.4)
        logger.info(
            f"Successfully deleted/refunded booking {booking_id}",
            extra={
                "booking_id": booking_id,
                "refund_amount": refund_amount,
                "processing_time_ms": processing_time_ms,
                "response_status": "success",
                "operation": "delete_booking",
            }
        )
        
        # Performance warning if response exceeds 3 seconds (Requirement 12.2)
        if processing_time > 3.0:
            logger.warning(
                f"Slow API response for delete_booking: {processing_time:.2f}s",
                extra={
                    "booking_id": booking_id,
                    "processing_time_seconds": processing_time,
                    "processing_time_ms": processing_time_ms,
                    "threshold_seconds": 3.0,
                    "operation": "delete_booking",
                }
            )
        
        return response
    
    async def close(self):
        """Close the HTTP client connection pool."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()



def validate_oauth2_credentials() -> bool:
    """
    Validate that ParkWhiz OAuth2 credentials are configured.
    
    This function should be called at application startup to ensure
    the ParkWhiz API client can authenticate properly.
    
    Returns:
        True if credentials are configured, False otherwise
    
    Raises:
        ParkWhizAuthenticationError: If credentials are missing
    """
    # Validate credentials
    client_id = os.getenv("PARKWHIZ_CLIENT_ID")
    client_secret = os.getenv("PARKWHIZ_CLIENT_SECRET")
    
    if not client_id or not client_secret:
        logger.critical(
            "ParkWhiz OAuth2 credentials not configured. "
            "Set PARKWHIZ_CLIENT_ID and PARKWHIZ_CLIENT_SECRET environment variables."
        )
        raise ParkWhizAuthenticationError(
            "ParkWhiz OAuth2 credentials not configured. "
            "Set PARKWHIZ_CLIENT_ID and PARKWHIZ_CLIENT_SECRET environment variables. "
            "Contact dev-admin@parkwhiz.com to obtain credentials."
        )
    
    logger.info(
        "ParkWhiz OAuth2 credentials validated successfully",
        extra={
            "client_id_configured": bool(client_id),
            "client_secret_configured": bool(client_secret),
        }
    )
    
    return True
