"""
Webhook-specific handler for pre-arrival processing.

This module provides the WebhookPreArrivalHandler class that orchestrates
pre-arrival processing in webhook context with strict timeout enforcement
and automatic ticket updates.

The handler wraps the core PreArrivalGate logic with webhook-specific concerns:
- Timeout enforcement (<8 seconds to stay within Freshdesk 10s limit)
- Automatic ticket updates for approvals
- Silent error handling (no user feedback)
- Webhook-specific logging and metrics

Performance: <8 seconds total including ticket updates
"""

import time
import logging
import asyncio
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from .pre_arrival_gate import PreArrivalGate
from ..models import PreArrivalDecision
from ...integrations.freshdesk import add_note_direct, update_ticket_direct

logger = logging.getLogger(__name__)


@dataclass
class WebhookPreArrivalResult:
    """Result of webhook pre-arrival processing."""
    success: bool
    decision: Optional[str] = None
    reasoning: Optional[str] = None
    ticket_updated: bool = False
    processing_time_ms: int = 0
    errors: List[str] = field(default_factory=list)


class WebhookPreArrivalHandler:
    """
    Webhook-specific handler for pre-arrival processing.
    
    Handles:
    - Strict timeout enforcement (<8 seconds)
    - Automatic ticket updates
    - Silent error handling
    - Webhook-specific logging
    """
    
    def __init__(self, pre_arrival_gate: PreArrivalGate):
        """
        Initialize webhook handler.
        
        Args:
            pre_arrival_gate: Core pre-arrival gate logic
        """
        self.pre_arrival_gate = pre_arrival_gate
        self.timeout_seconds = 8  # Stay within Freshdesk 10s timeout
        self.performance_warning_threshold = 5  # Warn if processing exceeds 5 seconds
    
    async def process_webhook_pre_arrival(
        self, 
        ticket_id: str, 
        ticket_data: Dict
    ) -> WebhookPreArrivalResult:
        """
        Process pre-arrival check for webhook-triggered ticket.
        
        Args:
            ticket_id: Freshdesk ticket ID
            ticket_data: Zapier webhook payload
            
        Returns:
            WebhookPreArrivalResult with processing outcome
        """
        start = time.perf_counter()
        
        try:
            logger.info(
                f"Starting webhook pre-arrival processing for ticket {ticket_id}",
                extra={
                    "ticket_id": ticket_id,
                    "timeout_seconds": self.timeout_seconds
                }
            )
            
            # Enforce timeout using asyncio
            result = await asyncio.wait_for(
                self._process_with_timeout(ticket_id, ticket_data),
                timeout=self.timeout_seconds
            )
            
            processing_ms = int((time.perf_counter() - start) * 1000)
            result.processing_time_ms = processing_ms
            
            # Log performance warning if approaching timeout
            if processing_ms > self.performance_warning_threshold * 1000:
                logger.warning(
                    "Webhook pre-arrival processing approaching timeout",
                    extra={
                        "ticket_id": ticket_id,
                        "processing_time_ms": processing_ms,
                        "timeout_threshold_ms": self.timeout_seconds * 1000,
                        "warning_threshold_ms": self.performance_warning_threshold * 1000
                    }
                )
            
            logger.info(
                f"Webhook pre-arrival processing completed for ticket {ticket_id}",
                extra={
                    "ticket_id": ticket_id,
                    "processing_time_ms": processing_ms,
                    "success": result.success,
                    "decision": result.decision,
                    "ticket_updated": result.ticket_updated
                }
            )
            
            return result
            
        except asyncio.TimeoutError:
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                "Webhook pre-arrival processing timed out",
                extra={
                    "ticket_id": ticket_id,
                    "processing_time_ms": processing_ms,
                    "timeout_seconds": self.timeout_seconds
                }
            )
            
            # Add timeout note to ticket
            await self._add_timeout_note(ticket_id)
            
            return WebhookPreArrivalResult(
                success=True,  # Return success to prevent Freshdesk retries
                processing_time_ms=processing_ms,
                errors=["Processing timed out - added note to ticket for manual review"]
            )
        
        except Exception as e:
            processing_ms = int((time.perf_counter() - start) * 1000)
            logger.error(
                f"Unexpected error in webhook pre-arrival processing: {str(e)}",
                extra={
                    "ticket_id": ticket_id,
                    "processing_time_ms": processing_ms,
                    "error": str(e)
                },
                exc_info=True
            )
            
            return WebhookPreArrivalResult(
                success=False,
                processing_time_ms=processing_ms,
                errors=[f"Unexpected error: {str(e)}"]
            )
    
    async def _process_with_timeout(
        self, 
        ticket_id: str, 
        ticket_data: Dict
    ) -> WebhookPreArrivalResult:
        """
        Internal processing with timeout protection.
        
        Args:
            ticket_id: Freshdesk ticket ID
            ticket_data: Zapier webhook payload
            
        Returns:
            WebhookPreArrivalResult with processing outcome
        """
        # Run pre-arrival check with webhook context in executor to allow timeout
        loop = asyncio.get_event_loop()
        decision = await loop.run_in_executor(
            None, 
            self.pre_arrival_gate.check_pre_arrival, 
            ticket_data, 
            "webhook"
        )
        
        if not decision.should_continue:
            # Pre-arrival approval - update ticket automatically
            try:
                await self._apply_pre_arrival_approval(ticket_id, decision)
                
                logger.info(
                    "Webhook pre-arrival approval applied",
                    extra={
                        "ticket_id": ticket_id,
                        "hours_before_booking": decision.hours_before_booking,
                        "timezone_used": decision.timezone_used,
                        "processing_time_ms": decision.processing_time_ms
                    }
                )
                
                return WebhookPreArrivalResult(
                    success=True,
                    decision=decision.decision,
                    reasoning=decision.reasoning,
                    ticket_updated=True,
                    processing_time_ms=decision.processing_time_ms
                )
                
            except Exception as e:
                logger.error(
                    "Failed to apply pre-arrival approval to ticket",
                    extra={
                        "ticket_id": ticket_id,
                        "error": str(e),
                        "decision": decision.decision
                    },
                    exc_info=True
                )
                
                return WebhookPreArrivalResult(
                    success=False,
                    errors=[f"Failed to update ticket: {str(e)}"]
                )
        else:
            # Not pre-arrival - continue processing
            logger.info(
                "Webhook pre-arrival check: not pre-arrival, continuing",
                extra={
                    "ticket_id": ticket_id,
                    "processing_time_ms": decision.processing_time_ms,
                    "warnings": decision.warnings
                }
            )
            
            return WebhookPreArrivalResult(
                success=True,
                processing_time_ms=decision.processing_time_ms
            )
    
    async def _apply_pre_arrival_approval(
        self, 
        ticket_id: str, 
        decision: PreArrivalDecision
    ):
        """
        Apply pre-arrival approval to Freshdesk ticket.
        
        This method:
        1. Adds an approval note with timing details and policy reference
        2. Updates ticket status to resolved
        3. Adds appropriate tags (pre-arrival-approved, automated-decision)
        
        Args:
            ticket_id: Freshdesk ticket ID
            decision: PreArrivalDecision with approval details
            
        Raises:
            Exception: If ticket update fails
        """
        # Format approval note with timing details
        note_content = self._format_approval_note(decision)
        
        # Add approval note (customer-visible)
        await add_note_direct(
            ticket_id=ticket_id,
            note=note_content
        )
        
        logger.info(
            f"Added pre-arrival approval note to ticket {ticket_id}",
            extra={
                "ticket_id": ticket_id,
                "hours_before_booking": decision.hours_before_booking,
                "timezone_used": decision.timezone_used
            }
        )
        
        # Update ticket status to resolved with tags
        await update_ticket_direct(
            ticket_id=ticket_id,
            updates={
                "status": 4,  # Freshdesk status code for "Resolved"
                "tags": ["pre-arrival-approved", "automated-decision"]
            }
        )
        
        logger.info(
            f"Updated ticket {ticket_id} status to resolved with pre-arrival tags",
            extra={
                "ticket_id": ticket_id,
                "status": "Resolved",
                "tags": ["pre-arrival-approved", "automated-decision"]
            }
        )
    
    def _format_approval_note(self, decision: PreArrivalDecision) -> str:
        """
        Format approval note with timing details and policy reference.
        
        Args:
            decision: PreArrivalDecision with approval details
            
        Returns:
            Formatted approval note content
        """
        note_content = f"""
Pre-Arrival Cancellation - APPROVED

{decision.reasoning}

Processing Details:
- Hours before booking: {decision.hours_before_booking:.1f}
- Timezone used: {decision.timezone_used}
- Processing time: {decision.processing_time_ms}ms
- Automated decision via webhook

This ticket has been automatically approved per ParkWhiz pre-arrival policy.
        """.strip()
        
        return note_content
    
    async def _add_timeout_note(self, ticket_id: str):
        """
        Add timeout note to ticket for manual review.
        
        Args:
            ticket_id: Freshdesk ticket ID
        """
        note_content = """
Pre-Arrival Processing Timeout

The automated pre-arrival check timed out during processing. This ticket requires manual review to determine if it qualifies for pre-arrival approval.

Please check:
1. Booking start time vs. current time
2. If current time < booking start time, approve per pre-arrival policy
3. If not pre-arrival, continue with standard refund evaluation

This timeout was logged for system monitoring and optimization.
        """.strip()
        
        try:
            await add_note_direct(
                ticket_id=ticket_id,
                note=note_content
            )
            
            logger.info(
                f"Added timeout note to ticket {ticket_id}",
                extra={
                    "ticket_id": ticket_id,
                    "note_type": "timeout"
                }
            )
        except Exception as e:
            logger.error(
                "Failed to add timeout note to ticket",
                extra={
                    "ticket_id": ticket_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    def get_handler_info(self) -> Dict:
        """
        Get information about the webhook handler configuration.
        
        Returns:
            Dictionary with handler configuration
        """
        return {
            "handler_type": "webhook_pre_arrival",
            "timeout_seconds": self.timeout_seconds,
            "performance_warning_threshold": self.performance_warning_threshold,
            "gate_info": self.pre_arrival_gate.get_gate_info()
        }
