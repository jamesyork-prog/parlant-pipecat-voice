
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime
from parlant.tools.database_logger import (
    log_audit_trail,
    log_run_metrics,
    update_customer_context,
)


@pytest.fixture
def mock_db_connection():
    """Mock database connection and cursor."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


@pytest.mark.asyncio
async def test_log_audit_trail_success(mock_db_connection):
    """Test that log_audit_trail executes correct INSERT statement."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_123",
        "event_type": "ticket_created",
        "event_details": {"ticket_id": "1205974", "subject": "Test"},
        "status": "success",
        "error_message": None,
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await log_audit_trail(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains the INSERT statement for agent_audit_log
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "INSERT INTO agent_audit_log" in sql_call
    assert "run_id" in sql_call
    assert "event_type" in sql_call
    
    # Verify parameters were passed
    params = mock_cursor.execute.call_args[0][1]
    assert params["run_id"] == "run_123"
    assert params["event_type"] == "ticket_created"
    
    # Verify connection was committed and closed
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_conn.close.called
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.metadata["summary"] == "Audit trail logged for run_123"


@pytest.mark.asyncio
async def test_log_audit_trail_database_error(mock_db_connection):
    """Test that log_audit_trail handles database errors gracefully."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = Exception("Database connection failed")
    
    context = Mock()
    context.inputs = {
        "run_id": "run_456",
        "event_type": "test_event",
        "event_details": {},
        "status": "pending",
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await log_audit_trail(context)
    
    # Verify error handling
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
    assert result.metadata["summary"] == "Error logging audit trail for run_456"
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_conn.close.called


@pytest.mark.asyncio
async def test_log_run_metrics_success(mock_db_connection):
    """Test that log_run_metrics executes correct INSERT or UPDATE statement."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_789",
        "journey_name": "Refund Request",
        "start_time": "2025-11-07T10:00:00Z",
        "end_time": "2025-11-07T10:05:00Z",
        "duration_ms": 300000,
        "final_outcome": "approved",
        "ticket_id": "1205974",
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await log_run_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains INSERT/UPDATE for agent_run_metrics
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "agent_run_metrics" in sql_call
    assert "run_id" in sql_call
    
    # Verify connection was committed and closed
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_conn.close.called
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.metadata["summary"] == "Run metrics logged for run_789"


@pytest.mark.asyncio
async def test_log_run_metrics_partial_data(mock_db_connection):
    """Test that log_run_metrics handles partial data (start_time only)."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_partial",
        "journey_name": "Test Journey",
        "start_time": "2025-11-07T10:00:00Z",
        # No end_time, duration_ms, or final_outcome yet
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await log_run_metrics(context)
    
    # Should still succeed with partial data
    assert mock_cursor.execute.called
    assert result.data["status"] == "logged"


@pytest.mark.asyncio
async def test_update_customer_context_new_customer(mock_db_connection):
    """Test that update_customer_context creates new customer record."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "customer_id": "customer_001",
        "total_interactions": 1,
        "total_denials": 0,
        "custom_notes": {"preference": "email", "vip": False},
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await update_customer_context(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains INSERT with ON CONFLICT for customer_context
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "customer_context" in sql_call
    assert "customer_id" in sql_call
    
    # Verify connection was committed and closed
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_conn.close.called
    
    # Verify successful result
    assert result.data["status"] == "updated"
    assert result.metadata["summary"] == "Customer context updated for customer_001"


@pytest.mark.asyncio
async def test_update_customer_context_existing_customer(mock_db_connection):
    """Test that update_customer_context updates existing customer record."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "customer_id": "customer_002",
        "total_interactions": 5,
        "total_denials": 2,
        "custom_notes": {"last_issue": "refund_denied", "escalated": True},
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await update_customer_context(context)
    
    # Verify the operation succeeded
    assert mock_cursor.execute.called
    assert result.data["status"] == "updated"


@pytest.mark.asyncio
async def test_update_customer_context_increment_mode(mock_db_connection):
    """Test that update_customer_context can increment counters."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "customer_id": "customer_003",
        "increment_interactions": True,
        "increment_denials": True,
    }
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=mock_conn):
        result = await update_customer_context(context)
    
    # Verify the SQL handles incrementing
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "customer_context" in sql_call
    
    assert result.data["status"] == "updated"


@pytest.mark.asyncio
async def test_database_connection_failure():
    """Test that tools handle connection failure gracefully."""
    context = Mock()
    context.inputs = {"run_id": "test_run", "event_type": "test"}
    
    with patch("parlant.tools.database_logger.get_db_connection", return_value=None):
        result = await log_audit_trail(context)
    
    # Verify error is returned
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
