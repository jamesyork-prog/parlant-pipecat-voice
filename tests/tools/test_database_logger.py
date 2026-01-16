
import pytest
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime
import psycopg2
from app_tools.tools.infrastructure.database import (
    log_audit_trail,
    log_run_metrics,
    log_refund_transaction,
    log_performance_metrics,
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
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
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    assert mock_return.call_args[0][0] == mock_conn
    
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_audit_trail(context)
    
    # Verify error handling
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
    assert result.metadata["summary"] == "Error logging audit trail for run_456"
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_return.called


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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_run_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains INSERT/UPDATE for agent_run_metrics
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "agent_run_metrics" in sql_call
    assert "run_id" in sql_call
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection"):
        result = await log_run_metrics(context)
    
    # Should still succeed with partial data
    assert mock_cursor.execute.called
    assert result.data["status"] == "logged"


@pytest.mark.asyncio
async def test_log_run_metrics_with_new_performance_fields(mock_db_connection):
    """Test that log_run_metrics handles new performance tracking fields."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_perf_123",
        "journey_name": "Refund Request",
        "start_time": "2025-11-07T10:00:00Z",
        "end_time": "2025-11-07T10:05:00Z",
        "duration_ms": 300000,
        "final_outcome": "approved",
        "ticket_id": "1205974",
        # New performance fields
        "token_usage": 1500,
        "api_calls_count": 5,
        "cache_hits": 3,
        "cache_misses": 2,
        "error_count": 0,
        "retry_count": 1,
        "confidence_score": "high",
        "agent_name": "refund_agent",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_run_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains the new columns
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "token_usage" in sql_call
    assert "api_calls_count" in sql_call
    assert "cache_hits" in sql_call
    assert "cache_misses" in sql_call
    assert "error_count" in sql_call
    assert "retry_count" in sql_call
    assert "confidence_score" in sql_call
    assert "agent_name" in sql_call
    
    # Verify parameters were passed correctly
    params = mock_cursor.execute.call_args[0][1]
    assert params["token_usage"] == 1500
    assert params["api_calls_count"] == 5
    assert params["cache_hits"] == 3
    assert params["cache_misses"] == 2
    assert params["error_count"] == 0
    assert params["retry_count"] == 1
    assert params["confidence_score"] == "high"
    assert params["agent_name"] == "refund_agent"
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.metadata["summary"] == "Run metrics logged for run_perf_123"


@pytest.mark.asyncio
async def test_log_run_metrics_with_optional_performance_fields(mock_db_connection):
    """Test that log_run_metrics handles missing optional performance fields."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_optional_123",
        "journey_name": "Refund Request",
        "start_time": "2025-11-07T10:00:00Z",
        "ticket_id": "1205975",
        # Only some performance fields provided
        "token_usage": 800,
        "confidence_score": "medium",
        # Missing: api_calls_count, cache_hits, cache_misses, error_count, retry_count, agent_name
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection"):
        result = await log_run_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify parameters include None for missing fields
    params = mock_cursor.execute.call_args[0][1]
    assert params["token_usage"] == 800
    assert params["confidence_score"] == "medium"
    assert params["api_calls_count"] is None
    assert params["cache_hits"] is None
    assert params["cache_misses"] is None
    assert params["error_count"] is None
    assert params["retry_count"] is None
    assert params["agent_name"] is None
    
    # Should still succeed with partial data
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await update_customer_context(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains INSERT with ON CONFLICT for customer_context
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "customer_context" in sql_call
    assert "customer_id" in sql_call
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection"):
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
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection"):
        result = await update_customer_context(context)
    
    # Verify the SQL handles incrementing
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "customer_context" in sql_call
    
    assert result.data["status"] == "updated"


@pytest.mark.asyncio
async def test_log_refund_transaction_success(mock_db_connection):
    """Test that log_refund_transaction executes correct INSERT statement."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_refund_123",
        "ticket_id": "1205974",
        "booking_id": "booking_456",
        "refund_amount": 25.50,
        "refund_type": "duplicate",
        "refund_status": "completed",
        "parkwhiz_refund_id": "pw_refund_789",
        "error_message": None,
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_refund_transaction(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains the INSERT statement for refund_transactions
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "INSERT INTO refund_transactions" in sql_call
    assert "run_id" in sql_call
    assert "ticket_id" in sql_call
    assert "booking_id" in sql_call
    assert "refund_amount" in sql_call
    assert "refund_type" in sql_call
    assert "refund_status" in sql_call
    assert "parkwhiz_refund_id" in sql_call
    assert "error_message" in sql_call
    
    # Verify parameters were passed correctly
    params = mock_cursor.execute.call_args[0][1]
    assert params["run_id"] == "run_refund_123"
    assert params["ticket_id"] == "1205974"
    assert params["booking_id"] == "booking_456"
    assert params["refund_amount"] == 25.50
    assert params["refund_type"] == "duplicate"
    assert params["refund_status"] == "completed"
    assert params["parkwhiz_refund_id"] == "pw_refund_789"
    assert params["error_message"] is None
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    assert mock_return.call_args[0][0] == mock_conn
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.data["ticket_id"] == "1205974"
    assert result.data["refund_status"] == "completed"
    assert result.metadata["summary"] == "Refund transaction logged for ticket 1205974"


@pytest.mark.asyncio
async def test_log_refund_transaction_with_optional_fields(mock_db_connection):
    """Test that log_refund_transaction handles missing optional fields."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_refund_456",
        "ticket_id": "1205975",
        # Optional fields not provided: booking_id, refund_amount, refund_type, parkwhiz_refund_id, error_message
        # refund_status should default to "pending"
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_refund_transaction(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify parameters include None for missing optional fields
    params = mock_cursor.execute.call_args[0][1]
    assert params["run_id"] == "run_refund_456"
    assert params["ticket_id"] == "1205975"
    assert params["booking_id"] is None
    assert params["refund_amount"] is None
    assert params["refund_type"] is None
    assert params["refund_status"] == "pending"  # Default value
    assert params["parkwhiz_refund_id"] is None
    assert params["error_message"] is None
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.data["refund_status"] == "pending"


@pytest.mark.asyncio
async def test_log_refund_transaction_failed_refund(mock_db_connection):
    """Test that log_refund_transaction handles failed refunds with error messages."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_refund_789",
        "ticket_id": "1205976",
        "booking_id": "booking_999",
        "refund_amount": 50.00,
        "refund_type": "policy",
        "refund_status": "failed",
        "error_message": "ParkWhiz API timeout",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection"):
        result = await log_refund_transaction(context)
    
    # Verify parameters include error message
    params = mock_cursor.execute.call_args[0][1]
    assert params["refund_status"] == "failed"
    assert params["error_message"] == "ParkWhiz API timeout"
    
    # Verify successful logging even for failed refund
    assert result.data["status"] == "logged"
    assert result.data["refund_status"] == "failed"


@pytest.mark.asyncio
async def test_log_refund_transaction_missing_required_fields():
    """Test that log_refund_transaction validates required fields."""
    context = Mock()
    context.inputs = {
        # Missing run_id
        "ticket_id": "1205977",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection") as mock_get_conn:
        result = await log_refund_transaction(context)
    
    # Should not attempt database connection
    assert not mock_get_conn.called
    
    # Verify error is returned
    assert "error" in result.data
    assert "Missing required fields" in result.data["error"]
    assert "run_id and ticket_id are required" in result.data["error"]


@pytest.mark.asyncio
async def test_log_refund_transaction_database_connection_failure():
    """Test that log_refund_transaction handles connection failure gracefully."""
    context = Mock()
    context.inputs = {
        "run_id": "run_refund_fail",
        "ticket_id": "1205978",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=None):
        result = await log_refund_transaction(context)
    
    # Verify error is returned
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
    assert result.metadata["summary"] == "Error logging refund transaction for ticket 1205978"


@pytest.mark.asyncio
async def test_log_refund_transaction_integrity_error(mock_db_connection):
    """Test that log_refund_transaction handles foreign key constraint violations."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = psycopg2.IntegrityError("Foreign key constraint violation")
    
    context = Mock()
    context.inputs = {
        "run_id": "invalid_run_id",  # Non-existent run_id
        "ticket_id": "1205979",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_refund_transaction(context)
    
    # Verify rollback was called
    assert mock_conn.rollback.called
    
    # Verify error handling
    assert "error" in result.data
    assert "Database integrity error" in result.data["error"]
    assert result.metadata["summary"] == "Error: Invalid run_id or duplicate entry for ticket 1205979"
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_return.called


@pytest.mark.asyncio
async def test_log_refund_transaction_general_database_error(mock_db_connection):
    """Test that log_refund_transaction handles general database errors gracefully."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = Exception("Database write failed")
    
    context = Mock()
    context.inputs = {
        "run_id": "run_refund_error",
        "ticket_id": "1205980",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_refund_transaction(context)
    
    # Verify rollback was called
    assert mock_conn.rollback.called
    
    # Verify error handling
    assert "error" in result.data
    assert "Failed to log refund transaction" in result.data["error"]
    assert "Database write failed" in result.data["error"]
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_return.called


@pytest.mark.asyncio
async def test_log_performance_metrics_success(mock_db_connection):
    """Test that log_performance_metrics executes correct INSERT statement."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_perf_001",
        "agent_name": "refund_agent",
        # Timing parameters
        "total_duration_ms": 5000,
        "llm_duration_ms": 2000,
        "api_duration_ms": 2500,
        "database_duration_ms": 500,
        "freshdesk_api_duration_ms": 800,
        "parkwhiz_api_duration_ms": 1200,
        "lakera_api_duration_ms": 300,
        "gemini_api_duration_ms": 200,
        # Resource parameters
        "token_usage": 1500,
        "api_calls_count": 5,
        "cache_hits": 3,
        "cache_misses": 2,
        "api_timeout_count": 0,
        "api_retry_count": 1,
        # Quality parameters
        "confidence_score": "high",
        "error_count": 0,
        "retry_count": 1,
        "data_quality_score": 0.95,
        # Extraction quality
        "booking_extraction_method": "pattern",
        "booking_extraction_confidence": "high",
        "missing_fields": ["customer_phone"],
        # Financial
        "refund_amount": 25.50,
        # System metrics
        "queue_wait_time_ms": 100,
        "concurrent_runs_count": 3,
        "system_load_percent": 45.5,
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_performance_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify the SQL contains the INSERT statement for agent_performance_metrics
    sql_call = mock_cursor.execute.call_args[0][0]
    assert "INSERT INTO agent_performance_metrics" in sql_call
    assert "run_id" in sql_call
    assert "agent_name" in sql_call
    assert "total_duration_ms" in sql_call
    assert "llm_duration_ms" in sql_call
    assert "api_duration_ms" in sql_call
    assert "token_usage" in sql_call
    assert "cache_hits" in sql_call
    assert "confidence_score" in sql_call
    
    # Verify parameters were passed correctly
    params = mock_cursor.execute.call_args[0][1]
    assert params["run_id"] == "run_perf_001"
    assert params["agent_name"] == "refund_agent"
    assert params["total_duration_ms"] == 5000
    assert params["llm_duration_ms"] == 2000
    assert params["api_duration_ms"] == 2500
    assert params["token_usage"] == 1500
    assert params["api_calls_count"] == 5
    assert params["cache_hits"] == 3
    assert params["cache_misses"] == 2
    assert params["confidence_score"] == "high"
    assert params["error_count"] == 0
    assert params["retry_count"] == 1
    assert params["refund_amount"] == 25.50
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    assert mock_return.call_args[0][0] == mock_conn
    
    # Verify successful result
    assert result.data["status"] == "logged"
    assert result.data["run_id"] == "run_perf_001"
    assert result.metadata["summary"] == "Performance metrics logged for run run_perf_001"


@pytest.mark.asyncio
async def test_log_performance_metrics_with_optional_fields(mock_db_connection):
    """Test that log_performance_metrics handles missing optional fields."""
    mock_conn, mock_cursor = mock_db_connection
    
    context = Mock()
    context.inputs = {
        "run_id": "run_perf_002",
        "agent_name": "refund_agent",
        # Only some fields provided
        "total_duration_ms": 3000,
        "token_usage": 800,
        "confidence_score": "medium",
        # Missing: most timing, resource, and quality parameters
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_performance_metrics(context)
    
    # Verify cursor.execute was called
    assert mock_cursor.execute.called
    
    # Verify parameters include None for missing fields
    params = mock_cursor.execute.call_args[0][1]
    assert params["run_id"] == "run_perf_002"
    assert params["agent_name"] == "refund_agent"
    assert params["total_duration_ms"] == 3000
    assert params["token_usage"] == 800
    assert params["confidence_score"] == "medium"
    assert params["llm_duration_ms"] is None
    assert params["api_duration_ms"] is None
    assert params["cache_hits"] is None
    assert params["cache_misses"] is None
    assert params["error_count"] is None
    
    # Verify connection was committed and returned to pool
    assert mock_conn.commit.called
    assert mock_cursor.close.called
    assert mock_return.called
    
    # Verify successful result
    assert result.data["status"] == "logged"


@pytest.mark.asyncio
async def test_log_performance_metrics_missing_required_field():
    """Test that log_performance_metrics validates required field (run_id)."""
    context = Mock()
    context.inputs = {
        # Missing run_id
        "agent_name": "refund_agent",
        "total_duration_ms": 3000,
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection") as mock_get_conn:
        result = await log_performance_metrics(context)
    
    # Should not attempt database connection
    assert not mock_get_conn.called
    
    # Verify error is returned
    assert "error" in result.data
    assert "Missing required field" in result.data["error"]
    assert "run_id is required" in result.data["error"]


@pytest.mark.asyncio
async def test_log_performance_metrics_database_connection_failure():
    """Test that log_performance_metrics handles connection failure gracefully."""
    context = Mock()
    context.inputs = {
        "run_id": "run_perf_fail",
        "agent_name": "refund_agent",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=None):
        result = await log_performance_metrics(context)
    
    # Verify error is returned
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
    assert result.metadata["summary"] == "Error logging performance metrics for run run_perf_fail"


@pytest.mark.asyncio
async def test_log_performance_metrics_integrity_error(mock_db_connection):
    """Test that log_performance_metrics handles foreign key constraint violations."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = psycopg2.IntegrityError("Foreign key constraint violation")
    
    context = Mock()
    context.inputs = {
        "run_id": "invalid_run_id",  # Non-existent run_id
        "agent_name": "refund_agent",
        "total_duration_ms": 3000,
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_performance_metrics(context)
    
    # Verify rollback was called
    assert mock_conn.rollback.called
    
    # Verify error handling
    assert "error" in result.data
    assert "Database integrity error" in result.data["error"]
    assert result.metadata["summary"] == "Error: Invalid run_id or duplicate entry for run invalid_run_id"
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_return.called


@pytest.mark.asyncio
async def test_log_performance_metrics_general_database_error(mock_db_connection):
    """Test that log_performance_metrics handles general database errors gracefully."""
    mock_conn, mock_cursor = mock_db_connection
    mock_cursor.execute.side_effect = Exception("Database write failed")
    
    context = Mock()
    context.inputs = {
        "run_id": "run_perf_error",
        "agent_name": "refund_agent",
    }
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=mock_conn), \
         patch("app_tools.tools.database_logger.return_db_connection") as mock_return:
        result = await log_performance_metrics(context)
    
    # Verify rollback was called
    assert mock_conn.rollback.called
    
    # Verify error handling
    assert "error" in result.data
    assert "Failed to log performance metrics" in result.data["error"]
    assert "Database write failed" in result.data["error"]
    
    # Verify cleanup still happened
    assert mock_cursor.close.called
    assert mock_return.called


@pytest.mark.asyncio
async def test_database_connection_failure():
    """Test that tools handle connection failure gracefully."""
    context = Mock()
    context.inputs = {"run_id": "test_run", "event_type": "test"}
    
    with patch("app_tools.tools.database_logger.get_db_connection", return_value=None):
        result = await log_audit_trail(context)
    
    # Verify error is returned
    assert "error" in result.data
    assert "Database connection failed" in result.data["error"]
