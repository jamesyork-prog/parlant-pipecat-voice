
import pytest
from unittest.mock import Mock
from parlant.tools.parkwhiz_tools import get_customer_orders


MOCK_PARKWHIZ_API_KEY = "test_parkwhiz_key"


@pytest.fixture
def mock_env(monkeypatch):
    """Mock the ParkWhiz API key environment variable."""
    monkeypatch.setattr("parlant.tools.parkwhiz_tools.PARKWHIZ_API_KEY", MOCK_PARKWHIZ_API_KEY)


@pytest.mark.asyncio
async def test_get_customer_orders_success(mock_env, httpx_mock):
    """Test successfully fetching customer orders from ParkWhiz."""
    customer_email = "customer@example.com"
    
    # Mock ParkWhiz API response
    mock_response = {
        "orders": [
            {
                "id": "PW-123456",
                "quote_id": "Q-789",
                "start_time": "2025-11-15T10:00:00Z",
                "end_time": "2025-11-15T18:00:00Z",
                "location_name": "Downtown Parking Garage",
                "purchase_amount": 45.00,
                "currency": "USD",
                "status": "confirmed"
            }
        ],
        "total": 1
    }
    
    httpx_mock.add_response(
        url="https://api.parkwhiz.com/v4/orders/?email=customer%40example.com",
        json=mock_response,
        status_code=200,
    )
    
    context = Mock()
    context.inputs = {"customer_email": customer_email}
    result = await get_customer_orders(context)
    
    # Verify result
    assert result.data["orders"] is not None
    assert len(result.data["orders"]) >= 1
    assert result.metadata["summary"].startswith("Found")


@pytest.mark.asyncio
async def test_get_customer_orders_no_orders(mock_env, httpx_mock):
    """Test when customer has no orders."""
    customer_email = "newcustomer@example.com"
    
    mock_response = {"orders": [], "total": 0}
    
    httpx_mock.add_response(
        url="https://api.parkwhiz.com/v4/orders/?email=newcustomer%40example.com",
        json=mock_response,
        status_code=200,
    )
    
    context = Mock()
    context.inputs = {"customer_email": customer_email}
    result = await get_customer_orders(context)
    
    assert result.data["orders"] == []
    assert "No orders" in result.metadata["summary"]


@pytest.mark.asyncio
async def test_get_customer_orders_api_error(mock_env, httpx_mock):
    """Test handling of ParkWhiz API errors."""
    customer_email = "test@example.com"
    
    httpx_mock.add_response(
        url="https://api.parkwhiz.com/v4/orders/?email=test%40example.com",
        status_code=500,
        text="Internal Server Error",
    )
    
    context = Mock()
    context.inputs = {"customer_email": customer_email}
    result = await get_customer_orders(context)
    
    assert "error" in result.data
    assert "500" in result.data["error"]


@pytest.mark.asyncio
async def test_get_customer_orders_no_api_key(monkeypatch):
    """Test when API key is not configured."""
    monkeypatch.setattr("parlant.tools.parkwhiz_tools.PARKWHIZ_API_KEY", None)
    
    context = Mock()
    context.inputs = {"customer_email": "test@example.com"}
    result = await get_customer_orders(context)
    
    assert "error" in result.data
    assert "not configured" in result.data["error"]
