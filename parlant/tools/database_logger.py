
import os
import json
import psycopg2
import psycopg2.extras
import parlant.sdk as p


def get_db_connection():
    """
    Establishes and returns a connection to the PostgreSQL database.
    
    Returns:
        psycopg2.connection: A connection object to the database, or None if connection fails.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "ranjayDB"),
            user=os.getenv("POSTGRES_USER", "ranjay.kumar"),
            password=os.getenv("POSTGRES_PASSWORD", "ranjay"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error: Unable to connect to the database. {e}")
        return None


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
        conn.close()


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
    """
    run_id = context.inputs.get("run_id")
    journey_name = context.inputs.get("journey_name")
    start_time = context.inputs.get("start_time")
    end_time = context.inputs.get("end_time")
    duration_ms = context.inputs.get("duration_ms")
    final_outcome = context.inputs.get("final_outcome")
    ticket_id = context.inputs.get("ticket_id")
    
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
            (run_id, journey_name, start_time, end_time, duration_ms, final_outcome, ticket_id)
            VALUES (%(run_id)s, %(journey_name)s, %(start_time)s, %(end_time)s, %(duration_ms)s, %(final_outcome)s, %(ticket_id)s)
            ON CONFLICT (run_id) DO UPDATE SET
                journey_name = COALESCE(EXCLUDED.journey_name, agent_run_metrics.journey_name),
                end_time = COALESCE(EXCLUDED.end_time, agent_run_metrics.end_time),
                duration_ms = COALESCE(EXCLUDED.duration_ms, agent_run_metrics.duration_ms),
                final_outcome = COALESCE(EXCLUDED.final_outcome, agent_run_metrics.final_outcome),
                ticket_id = COALESCE(EXCLUDED.ticket_id, agent_run_metrics.ticket_id)
        """
        
        cursor.execute(sql, {
            "run_id": run_id,
            "journey_name": journey_name,
            "start_time": start_time,
            "end_time": end_time,
            "duration_ms": duration_ms,
            "final_outcome": final_outcome,
            "ticket_id": ticket_id,
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
        conn.close()


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
        conn.close()
