
import os
import json
import psycopg2
import psycopg2.extras
from psycopg2 import pool
import parlant.sdk as p


# Global connection pool
db_pool = None


def init_connection_pool():
    """
    Initialize database connection pool.
    
    Creates a SimpleConnectionPool with:
    - minconn=1: Minimum number of connections to maintain
    - maxconn=10: Maximum number of concurrent connections
    
    The pool is initialized lazily on first use and reused for all subsequent connections.
    """
    global db_pool
    if db_pool is None:
        try:
            db_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                host=os.getenv("POSTGRES_HOST", "localhost"),
                database=os.getenv("POSTGRES_DB", "WhizDB"),
                user=os.getenv("POSTGRES_USER", "admin"),
                password=os.getenv("POSTGRES_PASSWORD", "whiz"),
                port=os.getenv("POSTGRES_PORT", "5432")
            )
            print("Database connection pool initialized successfully")
        except psycopg2.OperationalError as e:
            print(f"Error: Unable to initialize connection pool. {e}")
            db_pool = None


def get_db_connection():
    """
    Gets a connection from the connection pool.
    
    Initializes the pool on first use if not already initialized.
    Connections obtained from the pool should be returned using return_db_connection().
    
    Returns:
        psycopg2.connection: A connection object from the pool, or None if pool is unavailable.
    """
    global db_pool
    
    # Initialize pool if not already done
    if db_pool is None:
        init_connection_pool()
    
    # Return None if pool initialization failed
    if db_pool is None:
        return None
    
    try:
        conn = db_pool.getconn()
        return conn
    except psycopg2.pool.PoolError as e:
        print(f"Error: Unable to get connection from pool. {e}")
        return None


def return_db_connection(conn):
    """
    Returns a connection back to the connection pool.
    
    Args:
        conn (psycopg2.connection): The connection to return to the pool
    """
    global db_pool
    
    if db_pool and conn:
        try:
            db_pool.putconn(conn)
        except psycopg2.pool.PoolError as e:
            print(f"Error: Unable to return connection to pool. {e}")


@p.tool
async def log_audit_trail(context: p.ToolContext) -> p.ToolResult:
    """
    Logs a detailed audit trail entry for an agent journey run.
    
    Args:
        run_id (str): Unique identifier for the journey run
        event_type (str): Type of event (e.g., "ticket_created", "refund_approved")
        event_details (dict): JSON object with additional event information
        status (str): Status of the event (e.g., "success", "error", "pending")
        error_message (str, optional): Error message if status is "error"
    """
    run_id = context.inputs.get("run_id")
    event_type = context.inputs.get("event_type")
    event_details = context.inputs.get("event_details", {})
    status = context.inputs.get("status", "pending")
    error_message = context.inputs.get("error_message")
    
    conn = get_db_connection()
    if not conn:
        return p.ToolResult(
            {"error": "Database connection failed"},
            metadata={"summary": f"Error logging audit trail for {run_id}"}
        )
    
    cursor = conn.cursor()
    
    try:
        sql = """
            INSERT INTO agent_audit_log (run_id, event_type, event_details, status, error_message)
            VALUES (%(run_id)s, %(event_type)s, %(event_details)s, %(status)s, %(error_message)s)
        """
        
        cursor.execute(sql, {
            "run_id": run_id,
            "event_type": event_type,
            "event_details": json.dumps(event_details),
            "status": status,
            "error_message": error_message,
        })
        
        conn.commit()
        
        return p.ToolResult(
            {"status": "logged", "run_id": run_id},
            metadata={"summary": f"Audit trail logged for {run_id}"}
        )
        
    except Exception as e:
        return p.ToolResult(
            {"error": f"Failed to log audit trail: {str(e)}"},
            metadata={"summary": f"Error logging audit trail for {run_id}"}
        )
    finally:
        cursor.close()
        return_db_connection(conn)


@p.tool
async def log_run_metrics(context: p.ToolContext) -> p.ToolResult:
    """
    Logs or updates high-level metrics for a journey run.
    
    Args:
        run_id (str): Unique identifier for the journey run
        journey_name (str): Name of the journey
        start_time (str, optional): ISO 8601 timestamp of when the journey started
        end_time (str, optional): ISO 8601 timestamp of when the journey ended
        duration_ms (int, optional): Duration of the journey in milliseconds
        final_outcome (str, optional): Final outcome (e.g., "approved", "denied", "escalated")
        ticket_id (str, optional): Associated Freshdesk ticket ID
        token_usage (int, optional): LLM tokens consumed during the journey
        api_calls_count (int, optional): Number of external API calls made
        cache_hits (int, optional): Number of cache hits
        cache_misses (int, optional): Number of cache misses
        error_count (int, optional): Number of errors encountered
        retry_count (int, optional): Number of retry attempts
        confidence_score (str, optional): Confidence level (e.g., "high", "medium", "low")
        agent_name (str, optional): Name of the agent that processed the journey
    """
    run_id = context.inputs.get("run_id")
    journey_name = context.inputs.get("journey_name")
    start_time = context.inputs.get("start_time")
    end_time = context.inputs.get("end_time")
    duration_ms = context.inputs.get("duration_ms")
    final_outcome = context.inputs.get("final_outcome")
    ticket_id = context.inputs.get("ticket_id")
    token_usage = context.inputs.get("token_usage")
    api_calls_count = context.inputs.get("api_calls_count")
    cache_hits = context.inputs.get("cache_hits")
    cache_misses = context.inputs.get("cache_misses")
    error_count = context.inputs.get("error_count")
    retry_count = context.inputs.get("retry_count")
    confidence_score = context.inputs.get("confidence_score")
    agent_name = context.inputs.get("agent_name")
    
    conn = get_db_connection()
    if not conn:
        return p.ToolResult(
            {"error": "Database connection failed"},
            metadata={"summary": f"Error logging run metrics for {run_id}"}
        )
    
    cursor = conn.cursor()
    
    try:
        # Use INSERT ... ON CONFLICT UPDATE to handle both new and existing records
        sql = """
            INSERT INTO agent_run_metrics 
            (run_id, journey_name, start_time, end_time, duration_ms, final_outcome, ticket_id,
             token_usage, api_calls_count, cache_hits, cache_misses, error_count, retry_count,
             confidence_score, agent_name)
            VALUES (%(run_id)s, %(journey_name)s, %(start_time)s, %(end_time)s, %(duration_ms)s, 
                    %(final_outcome)s, %(ticket_id)s, %(token_usage)s, %(api_calls_count)s, 
                    %(cache_hits)s, %(cache_misses)s, %(error_count)s, %(retry_count)s,
                    %(confidence_score)s, %(agent_name)s)
            ON CONFLICT (run_id) DO UPDATE SET
                journey_name = COALESCE(EXCLUDED.journey_name, agent_run_metrics.journey_name),
                end_time = COALESCE(EXCLUDED.end_time, agent_run_metrics.end_time),
                duration_ms = COALESCE(EXCLUDED.duration_ms, agent_run_metrics.duration_ms),
                final_outcome = COALESCE(EXCLUDED.final_outcome, agent_run_metrics.final_outcome),
                ticket_id = COALESCE(EXCLUDED.ticket_id, agent_run_metrics.ticket_id),
                token_usage = COALESCE(EXCLUDED.token_usage, agent_run_metrics.token_usage),
                api_calls_count = COALESCE(EXCLUDED.api_calls_count, agent_run_metrics.api_calls_count),
                cache_hits = COALESCE(EXCLUDED.cache_hits, agent_run_metrics.cache_hits),
                cache_misses = COALESCE(EXCLUDED.cache_misses, agent_run_metrics.cache_misses),
                error_count = COALESCE(EXCLUDED.error_count, agent_run_metrics.error_count),
                retry_count = COALESCE(EXCLUDED.retry_count, agent_run_metrics.retry_count),
                confidence_score = COALESCE(EXCLUDED.confidence_score, agent_run_metrics.confidence_score),
                agent_name = COALESCE(EXCLUDED.agent_name, agent_run_metrics.agent_name)
        """
        
        cursor.execute(sql, {
            "run_id": run_id,
            "journey_name": journey_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "final_outcome": final_outcome,
            "ticket_id": ticket_id,
            "token_usage": token_usage,
            "api_calls_count": api_calls_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "error_count": error_count,
            "retry_count": retry_count,
            "confidence_score": confidence_score,
            "agent_name": agent_name,
        })
        
        conn.commit()
        
        return p.ToolResult(
            {"status": "logged", "run_id": run_id},
            metadata={"summary": f"Run metrics logged for {run_id}"}
        )
        
    except Exception as e:
        return p.ToolResult(
            {"error": f"Failed to log run metrics: {str(e)}"},
            metadata={"summary": f"Error logging run metrics for {run_id}"}
        )
    finally:
        cursor.close()
        return_db_connection(conn)


@p.tool
async def log_refund_transaction(context: p.ToolContext) -> p.ToolResult:
    """
    Logs a refund transaction to the database.
    
    Args:
        run_id (str): Associated journey run ID
        ticket_id (str): Freshdesk ticket ID
        booking_id (str, optional): ParkWhiz booking ID
        refund_amount (float, optional): Amount refunded
        refund_type (str, optional): Type of refund (e.g., "duplicate", "policy", "manual")
        refund_status (str, optional): Status of refund (e.g., "pending", "completed", "failed")
        parkwhiz_refund_id (str, optional): ParkWhiz confirmation ID
        error_message (str, optional): Error message if refund failed
    """
    run_id = context.inputs.get("run_id")
    ticket_id = context.inputs.get("ticket_id")
    booking_id = context.inputs.get("booking_id")
    refund_amount = context.inputs.get("refund_amount")
    refund_type = context.inputs.get("refund_type")
    refund_status = context.inputs.get("refund_status", "pending")
    parkwhiz_refund_id = context.inputs.get("parkwhiz_refund_id")
    error_message = context.inputs.get("error_message")
    
    # Validate required fields
    if not run_id or not ticket_id:
        return p.ToolResult(
            {"error": "Missing required fields: run_id and ticket_id are required"},
            metadata={"summary": "Error: Missing required fields for refund transaction"}
        )
    
    conn = get_db_connection()
    if not conn:
        return p.ToolResult(
            {"error": "Database connection failed"},
            metadata={"summary": f"Error logging refund transaction for ticket {ticket_id}"}
        )
    
    cursor = conn.cursor()
    
    try:
        sql = """
            INSERT INTO refund_transactions 
            (run_id, ticket_id, booking_id, refund_amount, refund_type, 
             refund_status, parkwhiz_refund_id, error_message)
            VALUES (%(run_id)s, %(ticket_id)s, %(booking_id)s, %(refund_amount)s, 
                    %(refund_type)s, %(refund_status)s, %(parkwhiz_refund_id)s, %(error_message)s)
        """
        
        cursor.execute(sql, {
            "run_id": run_id,
            "ticket_id": ticket_id,
            "booking_id": booking_id,
            "refund_amount": refund_amount,
            "refund_type": refund_type,
            "refund_status": refund_status,
            "parkwhiz_refund_id": parkwhiz_refund_id,
            "error_message": error_message,
        })
        
        conn.commit()
        
        return p.ToolResult(
            {"status": "logged", "ticket_id": ticket_id, "refund_status": refund_status},
            metadata={"summary": f"Refund transaction logged for ticket {ticket_id}"}
        )
        
    except psycopg2.IntegrityError as e:
        conn.rollback()
        return p.ToolResult(
            {"error": f"Database integrity error: {str(e)}"},
            metadata={"summary": f"Error: Invalid run_id or duplicate entry for ticket {ticket_id}"}
        )
    except Exception as e:
        conn.rollback()
        return p.ToolResult(
            {"error": f"Failed to log refund transaction: {str(e)}"},
            metadata={"summary": f"Error logging refund transaction for ticket {ticket_id}"}
        )
    finally:
        cursor.close()
        return_db_connection(conn)


@p.tool
async def log_performance_metrics(context: p.ToolContext) -> p.ToolResult:
    """
    Logs detailed performance metrics for a journey run.
    
    Args:
        run_id (str): Associated journey run ID
        agent_name (str, optional): Name of the agent that processed the journey
        total_duration_ms (int, optional): Total duration in milliseconds
        llm_duration_ms (int, optional): LLM processing duration in milliseconds
        api_duration_ms (int, optional): API calls duration in milliseconds
        database_duration_ms (int, optional): Database operations duration in milliseconds
        freshdesk_api_duration_ms (int, optional): Freshdesk API duration in milliseconds
        parkwhiz_api_duration_ms (int, optional): ParkWhiz API duration in milliseconds
        lakera_api_duration_ms (int, optional): Lakera API duration in milliseconds
        gemini_api_duration_ms (int, optional): Gemini API duration in milliseconds
        token_usage (int, optional): LLM tokens consumed
        api_calls_count (int, optional): Number of external API calls made
        cache_hits (int, optional): Number of cache hits
        cache_misses (int, optional): Number of cache misses
        api_timeout_count (int, optional): Number of API timeouts
        api_retry_count (int, optional): Number of API retries
        confidence_score (str, optional): Confidence level (e.g., "high", "medium", "low")
        error_count (int, optional): Number of errors encountered
        retry_count (int, optional): Number of retry attempts
        data_quality_score (float, optional): Data quality score (0.0 to 1.0)
        booking_extraction_method (str, optional): Method used for booking extraction
        booking_extraction_confidence (str, optional): Confidence in booking extraction
        missing_fields (list, optional): List of missing fields
        refund_amount (float, optional): Refund amount if applicable
        queue_wait_time_ms (int, optional): Time spent waiting in queue
        concurrent_runs_count (int, optional): Number of concurrent runs
        system_load_percent (float, optional): System load percentage
    """
    run_id = context.inputs.get("run_id")
    agent_name = context.inputs.get("agent_name")
    
    # Timing parameters
    total_duration_ms = context.inputs.get("total_duration_ms")
    llm_duration_ms = context.inputs.get("llm_duration_ms")
    api_duration_ms = context.inputs.get("api_duration_ms")
    database_duration_ms = context.inputs.get("database_duration_ms")
    freshdesk_api_duration_ms = context.inputs.get("freshdesk_api_duration_ms")
    parkwhiz_api_duration_ms = context.inputs.get("parkwhiz_api_duration_ms")
    lakera_api_duration_ms = context.inputs.get("lakera_api_duration_ms")
    gemini_api_duration_ms = context.inputs.get("gemini_api_duration_ms")
    
    # Resource parameters
    token_usage = context.inputs.get("token_usage")
    api_calls_count = context.inputs.get("api_calls_count")
    cache_hits = context.inputs.get("cache_hits")
    cache_misses = context.inputs.get("cache_misses")
    api_timeout_count = context.inputs.get("api_timeout_count")
    api_retry_count = context.inputs.get("api_retry_count")
    
    # Quality parameters
    confidence_score = context.inputs.get("confidence_score")
    error_count = context.inputs.get("error_count")
    retry_count = context.inputs.get("retry_count")
    data_quality_score = context.inputs.get("data_quality_score")
    
    # Extraction quality
    booking_extraction_method = context.inputs.get("booking_extraction_method")
    booking_extraction_confidence = context.inputs.get("booking_extraction_confidence")
    missing_fields = context.inputs.get("missing_fields")
    
    # Financial
    refund_amount = context.inputs.get("refund_amount")
    
    # System metrics
    queue_wait_time_ms = context.inputs.get("queue_wait_time_ms")
    concurrent_runs_count = context.inputs.get("concurrent_runs_count")
    system_load_percent = context.inputs.get("system_load_percent")
    
    # Validate required field
    if not run_id:
        return p.ToolResult(
            {"error": "Missing required field: run_id is required"},
            metadata={"summary": "Error: Missing required field for performance metrics"}
        )
    
    conn = get_db_connection()
    if not conn:
        return p.ToolResult(
            {"error": "Database connection failed"},
            metadata={"summary": f"Error logging performance metrics for run {run_id}"}
        )
    
    cursor = conn.cursor()
    
    try:
        sql = """
            INSERT INTO agent_performance_metrics 
            (run_id, agent_name, total_duration_ms, llm_duration_ms, api_duration_ms, 
             database_duration_ms, freshdesk_api_duration_ms, parkwhiz_api_duration_ms, 
             lakera_api_duration_ms, gemini_api_duration_ms, token_usage, api_calls_count, 
             cache_hits, cache_misses, api_timeout_count, api_retry_count, confidence_score, 
             error_count, retry_count, data_quality_score, booking_extraction_method, 
             booking_extraction_confidence, missing_fields, refund_amount, queue_wait_time_ms, 
             concurrent_runs_count, system_load_percent)
            VALUES (%(run_id)s, %(agent_name)s, %(total_duration_ms)s, %(llm_duration_ms)s, 
                    %(api_duration_ms)s, %(database_duration_ms)s, %(freshdesk_api_duration_ms)s, 
                    %(parkwhiz_api_duration_ms)s, %(lakera_api_duration_ms)s, %(gemini_api_duration_ms)s, 
                    %(token_usage)s, %(api_calls_count)s, %(cache_hits)s, %(cache_misses)s, 
                    %(api_timeout_count)s, %(api_retry_count)s, %(confidence_score)s, 
                    %(error_count)s, %(retry_count)s, %(data_quality_score)s, 
                    %(booking_extraction_method)s, %(booking_extraction_confidence)s, 
                    %(missing_fields)s, %(refund_amount)s, %(queue_wait_time_ms)s, 
                    %(concurrent_runs_count)s, %(system_load_percent)s)
        """
        
        cursor.execute(sql, {
            "run_id": run_id,
            "agent_name": agent_name,
            "total_duration_ms": total_duration_ms,
            "llm_duration_ms": llm_duration_ms,
            "api_duration_ms": api_duration_ms,
            "database_duration_ms": database_duration_ms,
            "freshdesk_api_duration_ms": freshdesk_api_duration_ms,
            "parkwhiz_api_duration_ms": parkwhiz_api_duration_ms,
            "lakera_api_duration_ms": lakera_api_duration_ms,
            "gemini_api_duration_ms": gemini_api_duration_ms,
            "token_usage": token_usage,
            "api_calls_count": api_calls_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "api_timeout_count": api_timeout_count,
            "api_retry_count": api_retry_count,
            "confidence_score": confidence_score,
            "error_count": error_count,
            "retry_count": retry_count,
            "data_quality_score": data_quality_score,
            "booking_extraction_method": booking_extraction_method,
            "booking_extraction_confidence": booking_extraction_confidence,
            "missing_fields": missing_fields,
            "refund_amount": refund_amount,
            "queue_wait_time_ms": queue_wait_time_ms,
            "concurrent_runs_count": concurrent_runs_count,
            "system_load_percent": system_load_percent,
        })
        
        conn.commit()
        
        return p.ToolResult(
            {"status": "logged", "run_id": run_id},
            metadata={"summary": f"Performance metrics logged for run {run_id}"}
        )
        
    except psycopg2.IntegrityError as e:
        conn.rollback()
        return p.ToolResult(
            {"error": f"Database integrity error: {str(e)}"},
            metadata={"summary": f"Error: Invalid run_id or duplicate entry for run {run_id}"}
        )
    except Exception as e:
        conn.rollback()
        return p.ToolResult(
            {"error": f"Failed to log performance metrics: {str(e)}"},
            metadata={"summary": f"Error logging performance metrics for run {run_id}"}
        )
    finally:
        cursor.close()
        return_db_connection(conn)


@p.tool
async def update_customer_context(context: p.ToolContext) -> p.ToolResult:
    """
    Updates or creates customer context information for persistent memory.
    
    Args:
        customer_id (str): Unique identifier for the customer (email, phone, or customer ID)
        total_interactions (int, optional): Total number of interactions
        total_denials (int, optional): Total number of refund denials
        custom_notes (dict, optional): JSON object with custom notes
        increment_interactions (bool, optional): If True, increment the interaction counter
        increment_denials (bool, optional): If True, increment the denials counter
    """
    customer_id = context.inputs.get("customer_id")
    total_interactions = context.inputs.get("total_interactions")
    total_denials = context.inputs.get("total_denials")
    custom_notes = context.inputs.get("custom_notes")
    increment_interactions = context.inputs.get("increment_interactions", False)
    increment_denials = context.inputs.get("increment_denials", False)
    
    conn = get_db_connection()
    if not conn:
        return p.ToolResult(
            {"error": "Database connection failed"},
            metadata={"summary": f"Error updating customer context for {customer_id}"}
        )
    
    cursor = conn.cursor()
    
    try:
        # Use INSERT ... ON CONFLICT UPDATE for upsert behavior
        if increment_interactions or increment_denials:
            # Increment mode
            sql = """
                INSERT INTO customer_context 
                (customer_id, last_interaction_date, total_interactions, total_denials, custom_notes)
                VALUES (%(customer_id)s, NOW(), 1, 0, %(custom_notes)s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    last_interaction_date = NOW(),
                    total_interactions = customer_context.total_interactions + %(inc_interactions)s,
                    total_denials = customer_context.total_denials + %(inc_denials)s,
                    custom_notes = COALESCE(EXCLUDED.custom_notes, customer_context.custom_notes)
            """
            cursor.execute(sql, {
                "customer_id": customer_id,
                "custom_notes": json.dumps(custom_notes) if custom_notes else None,
                "inc_interactions": 1 if increment_interactions else 0,
                "inc_denials": 1 if increment_denials else 0,
            })
        else:
            # Direct set mode
            sql = """
                INSERT INTO customer_context 
                (customer_id, last_interaction_date, total_interactions, total_denials, custom_notes)
                VALUES (%(customer_id)s, NOW(), %(total_interactions)s, %(total_denials)s, %(custom_notes)s)
                ON CONFLICT (customer_id) DO UPDATE SET
                    last_interaction_date = NOW(),
                    total_interactions = COALESCE(EXCLUDED.total_interactions, customer_context.total_interactions),
                    total_denials = COALESCE(EXCLUDED.total_denials, customer_context.total_denials),
                    custom_notes = COALESCE(EXCLUDED.custom_notes, customer_context.custom_notes)
            """
            cursor.execute(sql, {
                "customer_id": customer_id,
                "total_interactions": total_interactions,
                "total_denials": total_denials,
                "custom_notes": json.dumps(custom_notes) if custom_notes else None,
            })
        
        conn.commit()
        
        return p.ToolResult(
            {"status": "updated", "customer_id": customer_id},
            metadata={"summary": f"Customer context updated for {customer_id}"}
        )
        
    except Exception as e:
        return p.ToolResult(
            {"error": f"Failed to update customer context: {str(e)}"},
            metadata={"summary": f"Error updating customer context for {customer_id}"}
        )
    finally:
        cursor.close()
        return_db_connection(conn)
