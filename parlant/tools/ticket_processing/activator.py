"""
Journey activation module for triggering Parlant journeys from webhooks.

This module handles the activation of Parlant journeys by directly
calling the appropriate tools.
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import the tool we want to trigger
try:
    from app_tools.tools.ticket_processing.ticket_automation_engine import process_ticket_end_to_end
    TOOL_AVAILABLE = True
except ImportError:
    logger.warning("process_ticket_end_to_end tool not available for direct import")
    TOOL_AVAILABLE = False

# Parlant API configuration
PARLANT_BASE_URL = os.getenv("PARLANT_BASE_URL", "http://localhost:8800")
PARLANT_TIMEOUT = 30  # seconds


async def activate_journey(
    ticket_id: str,
    journey_name: str,
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Activate a Parlant journey for a ticket by directly calling the tool.
    
    Args:
        ticket_id: The Freshdesk ticket ID
        journey_name: Name of the journey to activate
        payload: The webhook payload containing ticket details
        
    Returns:
        Dict containing activation result with keys:
        - success: bool
        - error: str (if failed)
    """
    try:
        logger.info(
            "Activating journey by calling tool directly",
            extra={
                "ticket_id": ticket_id,
                "journey_name": journey_name,
                "tool_available": TOOL_AVAILABLE
            }
        )
        
        if not TOOL_AVAILABLE:
            return {
                "success": False,
                "error": "Tool not available for direct import"
            }
        
        # Create a mock ToolContext for the tool
        # Since we're calling it directly, we need to provide the context
        from unittest.mock import Mock
        
        context = Mock()
        context.inputs = {
            "ticket_id": ticket_id,
            "webhook_payload": payload  # Pass the webhook payload
        }
        context.agent_id = "webhook_agent"
        context.customer_id = f"ticket_{ticket_id}"
        context.session_id = f"webhook_session_{ticket_id}"
        
        # Call the tool directly
        result = await process_ticket_end_to_end(context, ticket_id)
        
        logger.info(
            "Journey activated successfully via direct tool call",
            extra={
                "ticket_id": ticket_id,
                "journey_name": journey_name
            }
        )
        
        return {
            "success": True,
            "journey_name": journey_name,
            "result": result.data if hasattr(result, 'data') else str(result)
        }
                
    except Exception as e:
        error_msg = f"Journey activation failed: {str(e)}"
        logger.error(
            "Journey activation error",
            extra={
                "ticket_id": ticket_id,
                "error": error_msg,
                "exception_type": type(e).__name__
            }
        )
        return {
            "success": False,
            "error": error_msg
        }


async def trigger_manual_tool_execution(
    ticket_id: str,
    tool_name: str = "process_ticket_end_to_end"
) -> Dict[str, Any]:
    """
    Directly trigger a Parlant tool without going through a journey.
    
    This is a simpler alternative that directly calls the tool endpoint.
    
    Args:
        ticket_id: The Freshdesk ticket ID
        tool_name: Name of the tool to execute
        
    Returns:
        Dict containing execution result
    """
    try:
        logger.info(
            "Triggering manual tool execution",
            extra={
                "ticket_id": ticket_id,
                "tool_name": tool_name
            }
        )
        
        async with httpx.AsyncClient(timeout=PARLANT_TIMEOUT) as client:
            response = await client.post(
                f"{PARLANT_BASE_URL}/tools/{tool_name}/execute",
                json={
                    "ticket_id": ticket_id
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(
                    "Tool executed successfully",
                    extra={
                        "ticket_id": ticket_id,
                        "tool_name": tool_name
                    }
                )
                return {
                    "success": True,
                    "result": result
                }
            else:
                error_msg = f"Tool execution failed: {response.status_code}"
                logger.error(
                    "Tool execution failed",
                    extra={
                        "ticket_id": ticket_id,
                        "error": error_msg
                    }
                )
                return {
                    "success": False,
                    "error": error_msg
                }
                
    except Exception as e:
        error_msg = f"Tool execution error: {str(e)}"
        logger.error(
            "Tool execution error",
            extra={
                "ticket_id": ticket_id,
                "error": error_msg
            }
        )
        return {
            "success": False,
            "error": error_msg
        }
