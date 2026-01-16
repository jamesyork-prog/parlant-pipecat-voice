"""
LLM Audit Logger for tracking all LLM interactions.

This module provides database logging for all LLM prompts and responses,
enabling debugging, analysis, and compliance tracking.
"""

import os
import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Dict, Optional, Any
import psycopg2
from psycopg2.extras import Json

logger = logging.getLogger(__name__)


def make_json_serializable(obj: Any) -> Any:
    """
    Convert an object to a JSON-serializable format.
    
    Handles dataclasses, datetime objects, and nested structures.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    if is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass to dict, then recursively process
        return make_json_serializable(asdict(obj))
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # For any other object, try to convert to string
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


class LLMAuditLogger:
    """
    Logs all LLM interactions to PostgreSQL for audit and debugging.
    
    This class captures:
    - Full prompt text sent to the LLM
    - Complete response received
    - Decision details (decision, confidence, policy)
    - Processing time and errors
    - Metadata for additional context
    """
    
    def __init__(self):
        """Initialize database connection."""
        self.db_config = {
            'dbname': os.getenv('POSTGRES_DB', 'WhizDB'),
            'user': os.getenv('POSTGRES_USER', 'admin'),
            'password': os.getenv('POSTGRES_PASSWORD', 'whiz'),
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432')
        }
    
    def log_llm_interaction(
        self,
        ticket_id: str,
        model_name: str,
        prompt_text: str,
        response_text: Optional[str] = None,
        decision: Optional[str] = None,
        confidence: Optional[str] = None,
        policy_applied: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log an LLM interaction to the database.
        
        Args:
            ticket_id: Freshdesk ticket ID
            model_name: LLM model used (e.g., "gemini-2.0-flash-exp")
            prompt_text: Full prompt sent to the LLM
            response_text: Full response received from the LLM
            decision: Decision made (Approved/Denied/Needs Human Review)
            confidence: Confidence level (high/medium/low)
            policy_applied: Policy rule that was applied
            processing_time_ms: Time taken for LLM call
            error_message: Error message if call failed
            metadata: Additional context (booking_info, rule_result, etc.)
        
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Insert log entry - serialize metadata to handle dataclasses
            serializable_metadata = make_json_serializable(metadata) if metadata else None
            
            cursor.execute("""
                INSERT INTO llm_audit_log (
                    ticket_id,
                    model_name,
                    prompt_text,
                    response_text,
                    decision,
                    confidence,
                    policy_applied,
                    processing_time_ms,
                    error_message,
                    metadata
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                ticket_id,
                model_name,
                prompt_text,
                response_text,
                decision,
                confidence,
                policy_applied,
                processing_time_ms,
                error_message,
                Json(serializable_metadata) if serializable_metadata else None
            ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"LLM interaction logged for ticket {ticket_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log LLM interaction: {e}")
            return False
    
    def get_llm_logs_for_ticket(self, ticket_id: str) -> list:
        """
        Retrieve all LLM logs for a specific ticket.
        
        Args:
            ticket_id: Freshdesk ticket ID
        
        Returns:
            List of log entries (dicts)
        """
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    id,
                    timestamp,
                    ticket_id,
                    model_name,
                    prompt_text,
                    response_text,
                    decision,
                    confidence,
                    policy_applied,
                    processing_time_ms,
                    error_message,
                    metadata
                FROM llm_audit_log
                WHERE ticket_id = %s
                ORDER BY timestamp DESC
            """, (ticket_id,))
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve LLM logs: {e}")
            return []
