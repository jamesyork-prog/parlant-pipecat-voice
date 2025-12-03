"""
Tests for ParkWhiz OAuth2 API Client

Tests OAuth2 authentication, token management, and API operations.
All tests use pytest-httpx to mock external API calls.
"""

import pytest
import pytest_asyncio
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import httpx
from app_tools.tools.parkwhiz_client import (
    ParkWhizOAuth2Client,
    ParkWhizError,
    ParkWhizAuthenticationError,
    ParkWhizNotFoundError,
    ParkWhizTimeoutError,
    ParkWhizRateLimitError,
    ParkWhizValidationError,
    validate_oauth2_credentials,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_env_oauth2(monkeypatch):
    """Mock environment variables for OAuth2 authentication."""
    monkeypatch.setenv("PARKWHIZ_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PARKWHIZ_CLIENT_SECRET", "test_client_secret")
    monkeypatch.setenv("PARKWHIZ_SCOPE", "partner")
    monkeypatch.setenv("PARKWHIZ_ENV", "sandbox")
    monkeypatch.setenv("PARKWHIZ_SANDBOX_URL", "https://api-sandbox.parkwhiz.com/v4")
    monkeypatch.setenv("PARKWHIZ_TIMEOUT", "30")
    monkeypatch.setenv("PARKWHIZ_MAX_RETRIES", "3")
    monkeypatch.setenv("PARKWHIZ_CACHE_TTL", "120")


@pytest_asyncio.fixture
async def oauth2_client(mock_env_oauth2):
    """Create OAuth2 client instance for testing."""
    client = ParkWhizOAuth2Client()
    yield client
    await client.close()


@pytest.fixture
def mock_token_response():
    """Mock OAuth2 token response."""
    return {
        "access_token": "test_access_token_12345",
        "token_type": "Bearer",
        "expires_in": 31557600,  # 1 year
        "scope": "partner"
    }


@pytest.fixture
def mock_booking_response():
    """Mock booking API response."""
    return {
        "id": "12345",
        "purchase_time": "2024-01-15T10:30:00Z",
        "start_time": "2024-02-01T09:00:00Z",
        "end_time": "2024-02-01T17:00:00Z",
        "price_paid": 25.00,
        "purchaser": {
            "email": "customer@example.com",
            "name": "John Doe"
        },
        "location": {
            "name": "Downtown Parking Garage",
            "address": "123 Main St"
        }
    }


# ============================================================================
# CLIENT INITIALIZATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_oauth2_client_initialization_success(mock_env_oauth2):
    """Test successful OAuth2 client initialization with environment variables."""
    client = ParkWhizOAuth2Client()
    
    assert client.client_id == "test_client_id"
    assert client.client_secret == "test_client_secret"
    assert client.scope == "partner"
    assert client.base_url == "https://api-sandbox.parkwhiz.com/v4"
    assert client.timeout == 30
    assert client.max_retries == 3
    assert client._token is None
    assert client._token_expires_at is None
    
    await client.close()


@pytest.mark.asyncio
async def test_oauth2_client_initialization_missing_credentials(monkeypatch):
    """Test OAuth2 client initialization fails without credentials."""
    monkeypatch.delenv("PARKWHIZ_CLIENT_ID", raising=False)
    monkeypatch.delenv("PARKWHIZ_CLIENT_SECRET", raising=False)
    
    with pytest.raises(ParkWhizAuthenticationError) as exc_info:
        ParkWhizOAuth2Client()
    
    assert "credentials not configured" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_oauth2_client_production_url(monkeypatch):
    """Test OAuth2 client uses production URL when configured."""
    monkeypatch.setenv("PARKWHIZ_CLIENT_ID", "test_client_id")
    monkeypatch.setenv("PARKWHIZ_CLIENT_SECRET", "test_client_secret")
    monkeypatch.setenv("PARKWHIZ_ENV", "production")
    monkeypatch.setenv("PARKWHIZ_PRODUCTION_URL", "https://api.parkwhiz.com/v4")
    
    client = ParkWhizOAuth2Client()
    
    assert client.base_url == "https://api.parkwhiz.com/v4"
    
    await client.close()



# ============================================================================
# OAUTH2 TOKEN MANAGEMENT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_oauth2_token_refresh_success(oauth2_client, httpx_mock, mock_token_response):
    """Test successful OAuth2 token refresh."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Refresh token
    await oauth2_client._refresh_token()
    
    # Verify token was stored
    assert oauth2_client._token == "test_access_token_12345"
    assert oauth2_client._token_expires_at is not None
    assert oauth2_client._token_expires_at > datetime.now()


@pytest.mark.asyncio
async def test_oauth2_token_refresh_failure(oauth2_client, httpx_mock):
    """Test OAuth2 token refresh handles API errors."""
    # Mock failed token request
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        status_code=401,
        text="Invalid credentials"
    )
    
    # Attempt token refresh
    with pytest.raises(ParkWhizAuthenticationError) as exc_info:
        await oauth2_client._refresh_token()
    
    assert "401" in str(exc_info.value)


@pytest.mark.asyncio
async def test_oauth2_token_refresh_malformed_response(oauth2_client, httpx_mock):
    """Test OAuth2 token refresh handles malformed responses."""
    # Mock malformed response (missing access_token)
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json={"token_type": "Bearer"},  # Missing access_token
        status_code=200
    )
    
    # Attempt token refresh
    with pytest.raises(ParkWhizAuthenticationError) as exc_info:
        await oauth2_client._refresh_token()
    
    assert "malformed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_oauth2_ensure_valid_token_refreshes_when_missing(oauth2_client, httpx_mock, mock_token_response):
    """Test _ensure_valid_token refreshes token when missing."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Ensure token is None initially
    assert oauth2_client._token is None
    
    # Call ensure_valid_token
    await oauth2_client._ensure_valid_token()
    
    # Verify token was refreshed
    assert oauth2_client._token == "test_access_token_12345"


@pytest.mark.asyncio
async def test_oauth2_ensure_valid_token_refreshes_when_expiring_soon(oauth2_client, httpx_mock, mock_token_response):
    """Test _ensure_valid_token refreshes token when expiring within 24 hours."""
    # Set token that expires in 12 hours
    oauth2_client._token = "old_token"
    oauth2_client._token_expires_at = datetime.now() + timedelta(hours=12)
    
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Call ensure_valid_token
    await oauth2_client._ensure_valid_token()
    
    # Verify token was refreshed
    assert oauth2_client._token == "test_access_token_12345"


@pytest.mark.asyncio
async def test_oauth2_ensure_valid_token_keeps_valid_token(oauth2_client):
    """Test _ensure_valid_token keeps token when still valid."""
    # Set token that expires in 48 hours (more than 24 hour threshold)
    oauth2_client._token = "valid_token"
    oauth2_client._token_expires_at = datetime.now() + timedelta(hours=48)
    
    # Call ensure_valid_token (should not refresh)
    await oauth2_client._ensure_valid_token()
    
    # Verify token was not changed
    assert oauth2_client._token == "valid_token"



# ============================================================================
# GET_BOOKING_BY_ID TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_get_booking_by_id_success(oauth2_client, httpx_mock, mock_token_response, mock_booking_response):
    """Test successful booking retrieval by ID."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock booking endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="GET",
        json=mock_booking_response,
        status_code=200
    )
    
    # Get booking
    result = await oauth2_client.get_booking_by_id("12345")
    
    # Verify result
    assert result["id"] == "12345"
    assert result["purchaser"]["email"] == "customer@example.com"
    assert result["price_paid"] == 25.00


@pytest.mark.asyncio
async def test_get_booking_by_id_not_found(oauth2_client, httpx_mock, mock_token_response):
    """Test booking retrieval handles 404 not found."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock 404 response
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/99999",
        method="GET",
        status_code=404,
        text="Booking not found"
    )
    
    # Attempt to get booking
    with pytest.raises(ParkWhizNotFoundError):
        await oauth2_client.get_booking_by_id("99999")


@pytest.mark.asyncio
async def test_get_booking_by_id_timeout(oauth2_client, httpx_mock, mock_token_response):
    """Test booking retrieval handles timeout errors."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock timeout
    httpx_mock.add_exception(
        httpx.TimeoutException("Request timed out"),
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="GET"
    )
    
    # Attempt to get booking
    with pytest.raises(ParkWhizTimeoutError):
        await oauth2_client.get_booking_by_id("12345")



# ============================================================================
# DELETE_BOOKING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_delete_booking_success(oauth2_client, httpx_mock, mock_token_response):
    """Test successful booking deletion."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock delete endpoint (204 No Content)
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="DELETE",
        status_code=204
    )
    
    # Delete booking
    result = await oauth2_client.delete_booking("12345")
    
    # Verify result
    assert result["success"] is True
    assert result["status_code"] == 204


@pytest.mark.asyncio
async def test_delete_booking_with_refund_details(oauth2_client, httpx_mock, mock_token_response):
    """Test booking deletion returns refund details."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock delete endpoint with refund details
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="DELETE",
        json={
            "success": True,
            "booking_id": "12345",
            "refund_amount": 25.00,
            "status": "refunded"
        },
        status_code=200
    )
    
    # Delete booking
    result = await oauth2_client.delete_booking("12345")
    
    # Verify result
    assert result["success"] is True
    assert result["refund_amount"] == 25.00


@pytest.mark.asyncio
async def test_delete_booking_not_found(oauth2_client, httpx_mock, mock_token_response):
    """Test booking deletion handles 404 not found."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock 404 response
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/99999",
        method="DELETE",
        status_code=404,
        text="Booking not found"
    )
    
    # Attempt to delete booking
    with pytest.raises(ParkWhizNotFoundError):
        await oauth2_client.delete_booking("99999")



# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_request_handles_401_authentication_error(oauth2_client, httpx_mock, mock_token_response):
    """Test _request handles 401 authentication errors."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock 401 response
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="GET",
        status_code=401,
        text="Invalid token"
    )
    
    # Attempt request
    with pytest.raises(ParkWhizAuthenticationError):
        await oauth2_client.get_booking_by_id("12345")


@pytest.mark.asyncio
async def test_request_handles_429_rate_limit(oauth2_client, httpx_mock, mock_token_response):
    """Test _request handles 429 rate limit errors."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock 429 response
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/12345",
        method="GET",
        status_code=429,
        text="Rate limit exceeded"
    )
    
    # Attempt request
    with pytest.raises(ParkWhizRateLimitError):
        await oauth2_client.get_booking_by_id("12345")


@pytest.mark.asyncio
async def test_request_handles_400_validation_error(oauth2_client, httpx_mock, mock_token_response):
    """Test _request handles 400 validation errors."""
    # Mock token endpoint
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/oauth/token",
        method="POST",
        json=mock_token_response,
        status_code=200
    )
    
    # Mock 400 response
    httpx_mock.add_response(
        url="https://api-sandbox.parkwhiz.com/v4/bookings/invalid",
        method="GET",
        status_code=400,
        text="Invalid booking ID format"
    )
    
    # Attempt request
    with pytest.raises(ParkWhizValidationError):
        await oauth2_client.get_booking_by_id("invalid")


# ============================================================================
# CREDENTIAL VALIDATION TESTS
# ============================================================================

def test_validate_oauth2_credentials_success(mock_env_oauth2):
    """Test credential validation succeeds with valid credentials."""
    result = validate_oauth2_credentials()
    assert result is True


def test_validate_oauth2_credentials_missing(monkeypatch):
    """Test credential validation fails with missing credentials."""
    monkeypatch.delenv("PARKWHIZ_CLIENT_ID", raising=False)
    monkeypatch.delenv("PARKWHIZ_CLIENT_SECRET", raising=False)
    
    with pytest.raises(ParkWhizAuthenticationError) as exc_info:
        validate_oauth2_credentials()
    
    assert "credentials not configured" in str(exc_info.value).lower()


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_oauth2_client_context_manager(mock_env_oauth2):
    """Test OAuth2 client works as async context manager."""
    async with ParkWhizOAuth2Client() as client:
        assert client.client_id == "test_client_id"
        assert client.client is not None
    
    # Client should be closed after context exit
    assert client.client.is_closed
