"""
Journey routing logic for determining which journey to activate based on trigger source.

This module provides routing functionality to separate webhook-triggered automated
processing from chat-triggered interactive processing.
"""

import os
import logging
from typing import Literal

# Import structured logging
from app_tools.tools.infrastructure.logging import configure_structured_logging

# Import metrics tracking
from app_tools.tools.infrastructure.metrics import get_metrics_tracker

# Configure structured logging
logger = configure_structured_logging(
    level=os.getenv("WEBHOOK_LOG_LEVEL", "INFO"),
    component_name="journey_router"
)

# Get metrics tracker instance
metrics = get_metrics_tracker()

# Journey name constants
AUTOMATED_JOURNEY_NAME = "Automated Ticket Processing"
INTERACTIVE_JOURNEY_NAME = "Interactive Ticket Processing"


def route_to_journey(
    trigger_source: Literal["webhook", "chat", "unknown"],
    ticket_id: str
) -> str:
    """
    Determines which journey to activate based on the trigger source.
    
    This function implements the core routing logic that separates automated
    webhook-triggered processing from interactive chat-based processing.
    
    Args:
        trigger_source: The source that triggered the processing request.
                       - "webhook": Triggered by Freshdesk webhook event
                       - "chat": Triggered by user chat message
                       - "unknown": Ambiguous trigger source
        ticket_id: The Freshdesk ticket ID being processed
        
    Returns:
        str: The name of the journey to activate. Returns either:
             - "Automated Ticket Processing" for webhook triggers
             - "Interactive Ticket Processing" for chat or unknown triggers
             
    Examples:
        >>> route_to_journey("webhook", "12345")
        'Automated Ticket Processing'
        
        >>> route_to_journey("chat", "12345")
        'Interactive Ticket Processing'
        
        >>> route_to_journey("unknown", "12345")
        'Interactive Ticket Processing'
    """
    if trigger_source == "webhook":
        journey_name = AUTOMATED_JOURNEY_NAME
    else:
        # Default to interactive journey for chat or unknown triggers
        journey_name = INTERACTIVE_JOURNEY_NAME
    
    # Record journey activation
    metrics.record_journey_activation(journey_name)
    
    logger.info(
        "Journey routing decision",
        extra={
            "ticket_id": ticket_id,
            "trigger_source": trigger_source,
            "journey_name": journey_name
        }
    )
    
    return journey_name


def detect_trigger_source(
    from_webhook: bool = False,
    from_chat: bool = False
) -> Literal["webhook", "chat", "unknown"]:
    """
    Detects the trigger source based on context flags.
    
    This function helps identify where a processing request originated from,
    which is used to route to the appropriate journey.
    
    Args:
        from_webhook: True if the request came from a webhook endpoint
        from_chat: True if the request came from a chat message
        
    Returns:
        str: The detected trigger source:
             - "webhook" if from_webhook is True
             - "chat" if from_chat is True
             - "unknown" if neither or both are True (ambiguous)
             
    Examples:
        >>> detect_trigger_source(from_webhook=True)
        'webhook'
        
        >>> detect_trigger_source(from_chat=True)
        'chat'
        
        >>> detect_trigger_source()
        'unknown'
    """
    # Webhook takes precedence if explicitly set
    if from_webhook and not from_chat:
        trigger_source = "webhook"
        logger.debug(
            "Trigger source detected",
            extra={"trigger_source": trigger_source}
        )
        return trigger_source
    
    # Chat is detected if explicitly set and not webhook
    if from_chat and not from_webhook:
        trigger_source = "chat"
        logger.debug(
            "Trigger source detected",
            extra={"trigger_source": trigger_source}
        )
        return trigger_source
    
    # Ambiguous or no flags set - default to unknown
    trigger_source = "unknown"
    logger.warning(
        "Ambiguous trigger source detected",
        extra={
            "from_webhook": from_webhook,
            "from_chat": from_chat,
            "defaulting_to": trigger_source
        }
    )
    return trigger_source
