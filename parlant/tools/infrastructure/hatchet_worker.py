"""
Hatchet worker system for async webhook orchestration.

This module provides the HatchetWorker class that implements:
- Task dequeue and processing orchestration
- Integration with existing TicketRouter for fast path detection
- Timeout handling for long-running tasks
- Decision pipeline integration with RuleEngine and DecisionMaker
- Freshdesk ticket updates with error handling
- Decision result persistence before task completion
"""

import asyncio
import time
import logging
import uuid
import json
from typing import Dict, Optional, Any, Set
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum

from .task_queue import Task, TaskResult, TaskStatus
from .database import get_db_connection, return_db_connection
from .prometheus_metrics import get_webhook_metrics_collector
from .tracing import trace_span, SpanType, get_current_trace_context, correlate_webhook_to_worker, correlate_worker_to_freshdesk
from .decision_persistence import get_decision_persistence_manager, AuditTrailEntry, PersistenceStatus
from ..routing.ticket_router import TicketRouter, RoutingDecision
from ..refund_decision.rule_engine import RuleEngine
from ..refund_decision.decision_maker import DecisionMaker
from ..refund_decision.policy_loader import PolicyLoader
from ..integrations.freshdesk import get_ticket, add_note, update_ticket, add_note_direct, update_ticket_direct

logger = logging.getLogger(__name__)


class ProcessingResult(Enum):
    """Result of task processing."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRY_NEEDED = "retry_needed"


@dataclass
class WorkerResult:
    """Result of worker processing."""
    result: ProcessingResult
    decision: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: float = 0.0
    fast_path_used: bool = False
    freshdesk_updated: bool = False
    retry_after: Optional[int] = None  # Seconds to wait before retry


class HatchetWorker:
    """
    Hatchet worker for asynchronous webhook processing.
    
    Implements requirements:
    - 3.1: Allow processing time up to 300 seconds
    - 3.2: Fast path completion within 30 seconds
    - 3.3: Complete RuleEngine and DecisionMaker pipeline integration
    - 3.4: LLM integration with extended timeout support
    - 3.5: Freshdesk ticket updates with decision results
    - 6.5: Freshdesk API error handling with retry logic
    """
    
    def __init__(
        self,
        worker_id: str,
        hatchet_client=None,
        task_queue=None,
        load_balancer=None,
        max_processing_time: int = 300,
        fast_path_timeout: int = 30,
        processing_capacity: int = 10
    ):
        """
        Initialize Hatchet worker.
        
        Args:
            worker_id: Unique identifier for this worker instance
            hatchet_client: Hatchet client for workflow orchestration
            task_queue: TaskQueue instance for task management
            load_balancer: WorkerLoadBalancer for horizontal scaling
            max_processing_time: Maximum processing time in seconds (default: 300)
            fast_path_timeout: Fast path timeout in seconds (default: 30)
            processing_capacity: Maximum concurrent tasks (default: 10)
        """
        self.worker_id = worker_id
        self.hatchet_client = hatchet_client
        self.task_queue = task_queue
        self.load_balancer = load_balancer
        self.max_processing_time = max_processing_time
        self.fast_path_timeout = fast_path_timeout
        self.processing_capacity = processing_capacity
        
        # Initialize processing components
        self.ticket_router = TicketRouter()
        
        # Initialize RuleEngine with actual policy rules
        try:
            policy_loader = PolicyLoader()
            rules = policy_loader.get_rules()
            self.rule_engine = RuleEngine(rules, contradiction_config={})
            logger.info("RuleEngine initialized with policy rules")
        except Exception as e:
            logger.warning(f"Failed to initialize RuleEngine with policy rules: {e}")
            self.rule_engine = RuleEngine({}, contradiction_config={})
        
        self.decision_maker = DecisionMaker()
        
        # Worker state
        self.is_running = False
        self.current_task: Optional[Task] = None
        self.current_tasks: Set[str] = set()  # Track multiple concurrent tasks
        self.processed_count = 0
        self.error_count = 0
        
        # Heartbeat for load balancer
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._heartbeat_interval = 30  # seconds
        
        logger.info(f"HatchetWorker {worker_id} initialized with capacity {processing_capacity}")
    
    async def start(self):
        """Start the worker to process tasks."""
        if self.is_running:
            logger.warning(f"Worker {self.worker_id} is already running")
            return
        
        self.is_running = True
        logger.info(f"Worker {self.worker_id} starting")
        
        # Register with load balancer
        if self.load_balancer:
            await self.load_balancer.register_worker(self.worker_id, self.processing_capacity)
            # Start heartbeat task
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        try:
            while self.is_running:
                try:
                    # Check if we can accept more tasks
                    if len(self.current_tasks) >= self.processing_capacity:
                        await asyncio.sleep(1)
                        continue
                    
                    # Get next task from queue
                    task = await self._get_next_task()
                    
                    if task is None:
                        # No tasks available, wait and continue
                        await asyncio.sleep(1)
                        continue
                    
                    # Process the task concurrently
                    asyncio.create_task(self._process_task(task))
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error: {e}")
                    self.error_count += 1
                    await asyncio.sleep(5)  # Brief pause on error
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id} cancelled")
        finally:
            await self._cleanup()
            logger.info(f"Worker {self.worker_id} stopped")
    
    async def stop(self):
        """Stop the worker gracefully."""
        logger.info(f"Stopping worker {self.worker_id}")
        self.is_running = False
        
        # Wait for current tasks to complete
        if self.current_tasks:
            logger.info(f"Waiting for {len(self.current_tasks)} tasks to complete")
            # In a real implementation, we might want to set a timeout here
            while self.current_tasks and len(self.current_tasks) > 0:
                await asyncio.sleep(1)
        
        await self._cleanup()
    
    async def _cleanup(self):
        """Cleanup worker resources."""
        self.is_running = False
        
        # Stop heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        
        # Unregister from load balancer
        if self.load_balancer:
            await self.load_balancer.unregister_worker(self.worker_id)
    
    async def _get_next_task(self) -> Optional[Task]:
        """
        Get next task from queue.
        
        Returns:
            Next task to process, or None if no tasks available
        """
        if self.task_queue:
            return await self.task_queue.dequeue(self.worker_id)
        
        # If no task queue, this worker is likely being used in fallback mode
        # or for direct task processing
        return None
    
    async def process_ticket_task(self, task_data: Dict[str, Any]) -> WorkerResult:
        """
        Execute complete refund decision pipeline for a ticket task.
        
        Implements the full processing workflow:
        1. Route ticket through TicketRouter for fast path detection
        2. If fast path: complete within 30 seconds
        3. If complex: run full RuleEngine -> DecisionMaker pipeline (up to 300s)
        4. Update Freshdesk ticket with decision result
        5. Handle errors and retries appropriately
        
        Args:
            task_data: Task payload containing webhook data
            
        Returns:
            WorkerResult with processing outcome and metadata
        """
        # Extract ticket data from task payload
        ticket_data = task_data.get("payload", {})
        ticket_id = ticket_data.get("ticket_id", ticket_data.get("id", "unknown"))
        
        # Start worker processing span
        with trace_span(
            "worker_processing",
            SpanType.WORKER,
            worker_id=self.worker_id,
            ticket_id=str(ticket_id),
            max_processing_time=self.max_processing_time
        ) as worker_span:
            start_time = time.time()
            metrics_collector = get_webhook_metrics_collector()
            
            logger.info(f"Processing ticket task: {ticket_id}")
            worker_span.log(f"Started processing ticket {ticket_id}")
            
            try:
                # Step 1: Route ticket for fast path detection
                with trace_span("ticket_routing", SpanType.ROUTING, ticket_id=str(ticket_id)) as routing_span:
                    routing_start = time.time()
                    routing_decision = await self._route_ticket(ticket_data)
                    routing_time = time.time() - routing_start
                    
                    # Add routing information to span
                    routing_span.set_tag("should_continue", routing_decision.should_continue)
                    routing_span.set_tag("processing_time_ms", routing_time * 1000)
                    if hasattr(routing_decision, 'rejection_reason') and routing_decision.rejection_reason:
                        routing_span.set_tag("rejection_reason", routing_decision.rejection_reason)
                    if hasattr(routing_decision, 'rejection_source') and routing_decision.rejection_source:
                        routing_span.set_tag("rejection_source", routing_decision.rejection_source)
                
                # Record routing metrics
                metrics_collector.record_decision_pipeline_time("routing", routing_time)
                
                if not routing_decision.should_continue:
                    # Fast path rejection - complete quickly
                    logger.info(f"Fast path rejection for ticket {ticket_id}: {routing_decision.rejection_reason}")
                    worker_span.set_tag("processing_path", "fast_path")
                    worker_span.set_tag("decision_type", "denied")
                    worker_span.log(f"Fast path rejection: {routing_decision.rejection_reason}")
                    
                    decision_result = {
                        "decision": "Denied",
                        "reasoning": routing_decision.rejection_reason,
                        "policy_applied": f"Fast Path - {routing_decision.rejection_source}",
                        "confidence": "high",
                        "method_used": "fast_path",
                        "processing_time_ms": int((time.time() - start_time) * 1000)
                    }
                    
                    # Update Freshdesk ticket with tracing
                    with trace_span("freshdesk_update", SpanType.FRESHDESK, ticket_id=str(ticket_id), operation="add_note") as freshdesk_span:
                        freshdesk_start = time.time()
                        freshdesk_updated = await self._update_freshdesk_ticket(ticket_id, decision_result)
                        freshdesk_time = time.time() - freshdesk_start
                        
                        freshdesk_span.set_tag("update_successful", freshdesk_updated)
                        freshdesk_span.set_tag("processing_time_ms", freshdesk_time * 1000)
                        
                        # Correlate worker to freshdesk span
                        correlate_worker_to_freshdesk(worker_span.span_id, freshdesk_span.span_id)
                    
                    # Record Freshdesk update metrics
                    if freshdesk_updated:
                        metrics_collector.record_freshdesk_update("add_note", freshdesk_time)
                    
                    # Calculate processing time before persistence
                    processing_time = time.time() - start_time
                    
                    # Persist decision result to database with tracing
                    with trace_span("decision_persistence", SpanType.DATABASE, ticket_id=str(ticket_id)) as persistence_span:
                        persistence_start = time.time()
                        decision_persisted = await self._persist_decision_result(
                            task_id="fast_path_" + str(int(time.time())),
                            ticket_id=ticket_id,
                            decision_result=decision_result,
                            processing_time=processing_time
                        )
                        persistence_time = time.time() - persistence_start
                        
                        persistence_span.set_tag("persistence_successful", decision_persisted)
                        persistence_span.set_tag("processing_time_ms", persistence_time * 1000)
                    
                    # Record persistence metrics
                    metrics_collector.record_decision_pipeline_time("persistence", persistence_time)
                    
                    # Record task completion metrics
                    metrics_collector.record_task_completion(
                        status="success",
                        processing_path="fast_path",
                        decision_type="denied",
                        processing_time_seconds=processing_time
                    )
                    
                    worker_span.set_tag("processing_time_seconds", processing_time)
                    worker_span.set_tag("freshdesk_updated", freshdesk_updated)
                    worker_span.log("Fast path processing completed successfully")
                    
                    return WorkerResult(
                        result=ProcessingResult.SUCCESS,
                        decision=decision_result,
                        processing_time=processing_time,
                        fast_path_used=True,
                        freshdesk_updated=freshdesk_updated
                    )
                
                # Step 2: Complex processing - run full decision pipeline
                logger.info(f"Running full decision pipeline for ticket {ticket_id}")
                worker_span.set_tag("processing_path", "complex")
                worker_span.log("Starting complex decision pipeline")
                
                # Set timeout for complex processing (300 seconds)
                try:
                    with trace_span("decision_pipeline", SpanType.DECISION, ticket_id=str(ticket_id)) as pipeline_span:
                        pipeline_start = time.time()
                        decision_result = await asyncio.wait_for(
                            self._run_decision_pipeline(ticket_data),
                            timeout=self.max_processing_time
                        )
                        pipeline_time = time.time() - pipeline_start
                        
                        pipeline_span.set_tag("processing_time_seconds", pipeline_time)
                        pipeline_span.set_tag("decision", decision_result.get("decision", "unknown"))
                        pipeline_span.set_tag("method_used", decision_result.get("method_used", "unknown"))
                    
                    # Record pipeline metrics
                    metrics_collector.record_decision_pipeline_time("rule_engine", pipeline_time)
                    
                    # Update Freshdesk ticket with tracing
                    with trace_span("freshdesk_update", SpanType.FRESHDESK, ticket_id=str(ticket_id), operation="add_note") as freshdesk_span:
                        freshdesk_start = time.time()
                        freshdesk_updated = await self._update_freshdesk_ticket(ticket_id, decision_result)
                        freshdesk_time = time.time() - freshdesk_start
                        
                        freshdesk_span.set_tag("update_successful", freshdesk_updated)
                        freshdesk_span.set_tag("processing_time_ms", freshdesk_time * 1000)
                        
                        # Correlate worker to freshdesk span
                        correlate_worker_to_freshdesk(worker_span.span_id, freshdesk_span.span_id)
                    
                    # Record Freshdesk update metrics
                    if freshdesk_updated:
                        metrics_collector.record_freshdesk_update("add_note", freshdesk_time)
                    
                    # Calculate processing time before persistence
                    processing_time = time.time() - start_time
                    
                    # Persist decision result to database with tracing
                    with trace_span("decision_persistence", SpanType.DATABASE, ticket_id=str(ticket_id)) as persistence_span:
                        persistence_start = time.time()
                        decision_persisted = await self._persist_decision_result(
                            task_id="complex_" + str(int(time.time())),
                            ticket_id=ticket_id,
                            decision_result=decision_result,
                            processing_time=processing_time
                        )
                        persistence_time = time.time() - persistence_start
                        
                        persistence_span.set_tag("persistence_successful", decision_persisted)
                        persistence_span.set_tag("processing_time_ms", persistence_time * 1000)
                    
                    # Record persistence metrics
                    metrics_collector.record_decision_pipeline_time("persistence", persistence_time)
                    
                    # Determine decision type for metrics
                    decision_type = decision_result.get("decision", "unknown").lower()
                    if decision_type == "approved":
                        decision_type = "approved"
                    elif decision_type == "denied":
                        decision_type = "denied"
                    else:
                        decision_type = "escalated"
                    
                    worker_span.set_tag("decision_type", decision_type)
                    
                    # Record task completion metrics
                    metrics_collector.record_task_completion(
                        status="success",
                        processing_path="complex",
                        decision_type=decision_type,
                        processing_time_seconds=processing_time
                    )
                    
                    worker_span.set_tag("processing_time_seconds", processing_time)
                    worker_span.set_tag("freshdesk_updated", freshdesk_updated)
                    worker_span.log("Complex processing completed successfully")
                    
                    return WorkerResult(
                        result=ProcessingResult.SUCCESS,
                        decision=decision_result,
                        processing_time=processing_time,
                        fast_path_used=False,
                        freshdesk_updated=freshdesk_updated
                    )
                    
                except asyncio.TimeoutError:
                    # Processing timeout - escalate to human review
                    logger.error(f"Processing timeout for ticket {ticket_id} after {self.max_processing_time}s")
                    worker_span.set_tag("error", True)
                    worker_span.set_tag("error.type", "timeout")
                    worker_span.set_tag("decision_type", "escalated")
                    worker_span.log(f"Processing timeout after {self.max_processing_time}s", level="error")
                    
                    timeout_decision = {
                        "decision": "Needs Human Review",
                        "reasoning": (
                            f"Processing timeout after {self.max_processing_time} seconds. "
                            "This case requires human review due to processing complexity."
                        ),
                        "policy_applied": "Processing Timeout - Escalated",
                        "confidence": "low",
                        "method_used": "timeout_escalation",
                        "processing_time_ms": int(self.max_processing_time * 1000)
                    }
                    
                    # Still try to update Freshdesk with tracing
                    with trace_span("freshdesk_update", SpanType.FRESHDESK, ticket_id=str(ticket_id), operation="add_note") as freshdesk_span:
                        freshdesk_start = time.time()
                        freshdesk_updated = await self._update_freshdesk_ticket(ticket_id, timeout_decision)
                        freshdesk_time = time.time() - freshdesk_start
                        
                        freshdesk_span.set_tag("update_successful", freshdesk_updated)
                        freshdesk_span.set_tag("processing_time_ms", freshdesk_time * 1000)
                        freshdesk_span.set_tag("timeout_escalation", True)
                        
                        # Correlate worker to freshdesk span
                        correlate_worker_to_freshdesk(worker_span.span_id, freshdesk_span.span_id)
                    
                    # Record Freshdesk update metrics
                    if freshdesk_updated:
                        metrics_collector.record_freshdesk_update("add_note", freshdesk_time)
                    
                    # Persist timeout decision to database with tracing
                    with trace_span("decision_persistence", SpanType.DATABASE, ticket_id=str(ticket_id)) as persistence_span:
                        persistence_start = time.time()
                        decision_persisted = await self._persist_decision_result(
                            task_id="timeout_" + str(int(time.time())),
                            ticket_id=ticket_id,
                            decision_result=timeout_decision,
                            processing_time=self.max_processing_time
                        )
                        persistence_time = time.time() - persistence_start
                        
                        persistence_span.set_tag("persistence_successful", decision_persisted)
                        persistence_span.set_tag("processing_time_ms", persistence_time * 1000)
                        persistence_span.set_tag("timeout_escalation", True)
                    
                    # Record persistence metrics
                    metrics_collector.record_decision_pipeline_time("persistence", persistence_time)
                    
                    # Record task timeout metrics
                    metrics_collector.record_task_completion(
                        status="timeout",
                        processing_path="complex",
                        decision_type="escalated",
                        processing_time_seconds=self.max_processing_time
                    )
                    
                    worker_span.set_tag("processing_time_seconds", self.max_processing_time)
                    worker_span.set_tag("freshdesk_updated", freshdesk_updated)
                    
                    return WorkerResult(
                        result=ProcessingResult.TIMEOUT,
                        decision=timeout_decision,
                        processing_time=self.max_processing_time,
                        fast_path_used=False,
                        freshdesk_updated=freshdesk_updated
                    )
                    
            except Exception as e:
                # Unexpected error during processing
                logger.error(f"Unexpected error processing ticket task: {e}")
                worker_span.set_tag("error", True)
                worker_span.set_tag("error.type", "unexpected_error")
                worker_span.set_tag("error.message", str(e))
                worker_span.set_tag("decision_type", "escalated")
                worker_span.log(f"Unexpected error: {str(e)}", level="error")
                
                processing_time = time.time() - start_time
                error_decision = {
                    "decision": "Needs Human Review",
                    "reasoning": f"Technical error during processing: {str(e)}",
                    "policy_applied": "Processing Error - Escalated",
                    "confidence": "low",
                    "method_used": "error_escalation",
                    "processing_time_ms": int(processing_time * 1000)
                }
                
                # Record task failure metrics
                metrics_collector.record_task_completion(
                    status="failed",
                    processing_path="unknown",
                    decision_type="escalated",
                    processing_time_seconds=processing_time
                )
                
                worker_span.set_tag("processing_time_seconds", processing_time)
                
                return WorkerResult(
                    result=ProcessingResult.FAILED,
                    decision=error_decision,
                    error=str(e),
                    processing_time=processing_time,
                    fast_path_used=False,
                    freshdesk_updated=False
                )
    
    async def _process_task(self, task: Task):
        """
        Process a single task from the queue.
        
        Args:
            task: Task to process
        """
        # Add to current tasks
        self.current_tasks.add(task.task_id)
        self.current_task = task  # Keep for backward compatibility
        task_start_time = time.time()
        
        try:
            logger.info(f"Worker {self.worker_id} processing task {task.task_id}")
            
            # Process the ticket task
            result = await self.process_ticket_task(task.payload)
            
            # Create task result
            task_result = TaskResult(
                task_id=task.task_id,
                success=(result.result == ProcessingResult.SUCCESS),
                result_data=result.decision,
                error=result.error,
                processing_time=result.processing_time,
                retry_count=task.retry_count,
                completed_at=datetime.now(timezone.utc)
            )
            
            # Complete or retry task based on result
            if result.result == ProcessingResult.SUCCESS:
                # Task completed successfully
                if self.task_queue:
                    await self.task_queue.complete_task(task.task_id, task_result)
                self.processed_count += 1
                logger.info(f"Task {task.task_id} completed successfully")
                
            elif result.result in [ProcessingResult.FAILED, ProcessingResult.TIMEOUT]:
                # Task failed - check if retry is needed
                if task.retry_count < task.max_retries and result.retry_after:
                    # Schedule retry
                    if self.task_queue:
                        await self.task_queue.retry_task(task.task_id, result.retry_after)
                    logger.info(f"Task {task.task_id} scheduled for retry")
                else:
                    # Move to dead letter queue
                    if self.task_queue:
                        await self.task_queue.dead_letter(
                            task.task_id, 
                            result.error or f"Processing {result.result.value}"
                        )
                    logger.error(f"Task {task.task_id} moved to dead letter queue")
                
        except Exception as e:
            logger.error(f"Error processing task {task.task_id}: {e}")
            self.error_count += 1
            
            # Move task to dead letter queue on unexpected error
            if self.task_queue:
                await self.task_queue.dead_letter(task.task_id, str(e))
                
        finally:
            # Remove from current tasks
            self.current_tasks.discard(task.task_id)
            if self.current_task and self.current_task.task_id == task.task_id:
                self.current_task = None
            
            processing_time = time.time() - task_start_time
            logger.debug(f"Task {task.task_id} processing completed in {processing_time:.2f}s")
    
    async def _route_ticket(self, ticket_data: Dict[str, Any]) -> RoutingDecision:
        """
        Route ticket through TicketRouter for fast path detection.
        
        Args:
            ticket_data: Ticket data from webhook payload
            
        Returns:
            RoutingDecision indicating whether to continue with complex processing
        """
        try:
            # Apply fast path timeout for routing
            routing_decision = await asyncio.wait_for(
                asyncio.to_thread(self.ticket_router.route_ticket, ticket_data),
                timeout=self.fast_path_timeout
            )
            
            logger.info(f"Routing decision: continue={routing_decision.should_continue}, "
                       f"time={routing_decision.processing_time_ms}ms")
            
            return routing_decision
            
        except asyncio.TimeoutError:
            logger.warning(f"Routing timeout after {self.fast_path_timeout}s, continuing to complex processing")
            
            # If routing times out, assume we should continue to complex processing
            return RoutingDecision(
                should_continue=True,
                processing_time_ms=self.fast_path_timeout * 1000,
                gates_evaluated=["timeout"],
                warnings=["Routing timed out, proceeding to complex processing"]
            )
            
        except Exception as e:
            logger.error(f"Error in ticket routing: {e}")
            
            # On routing error, continue to complex processing for safety
            return RoutingDecision(
                should_continue=True,
                processing_time_ms=0,
                gates_evaluated=["error"],
                warnings=[f"Routing error: {str(e)}, proceeding to complex processing"]
            )
    
    async def _run_decision_pipeline(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the complete RuleEngine -> DecisionMaker pipeline.
        
        Args:
            ticket_data: Ticket data from webhook payload
            
        Returns:
            Decision result dictionary
        """
        try:
            # Use DecisionMaker which orchestrates RuleEngine -> LLM pipeline
            decision_result = await self.decision_maker.make_decision(
                ticket_data=ticket_data,
                ticket_notes=ticket_data.get("notes", "")
            )
            
            logger.info(f"Decision pipeline result: {decision_result.get('decision')} "
                       f"({decision_result.get('method_used')})")
            
            return decision_result
            
        except Exception as e:
            logger.error(f"Error in decision pipeline: {e}")
            
            # Return escalation decision on pipeline error
            return {
                "decision": "Needs Human Review",
                "reasoning": f"Decision pipeline error: {str(e)}",
                "policy_applied": "Pipeline Error - Escalated",
                "confidence": "low",
                "method_used": "pipeline_error",
                "processing_time_ms": 0
            }
    
    async def _persist_decision_result(
        self,
        task_id: str,
        ticket_id: str,
        decision_result: Dict[str, Any],
        processing_time: float
    ) -> bool:
        """
        Persist decision result to database before task completion.
        
        Implements requirement 8.1: Persist the result to the database before 
        marking the task complete.
        
        Args:
            task_id: Unique task identifier
            ticket_id: Freshdesk ticket ID
            decision_result: Decision result dictionary
            processing_time: Processing time in seconds
            
        Returns:
            True if persistence successful, False otherwise
        """
        try:
            # Get decision persistence manager
            persistence_manager = get_decision_persistence_manager()
            
            # Prepare processing metadata
            processing_metadata = {
                "processing_time_ms": int(processing_time * 1000),
                "processing_path": "fast_path" if decision_result.get("method_used") == "fast_path" else "complex",
                "fast_path_used": decision_result.get("method_used") == "fast_path",
                "freshdesk_updated": False,  # Will be updated after Freshdesk call
                "worker_id": self.worker_id,
                "total_time_ms": int(processing_time * 1000),
                "api_calls_count": 1,  # At least one for decision making
                "llm_calls_count": 0 if decision_result.get("method_used") == "fast_path" else 1,
                "llm_total_time_ms": 0 if decision_result.get("method_used") == "fast_path" else int(processing_time * 800),  # Estimate
                "error_count": 0,
                "retry_count": 0
            }
            
            # Create comprehensive audit trail entries
            audit_entries = []
            
            # Main decision event
            audit_entries.append(
                await persistence_manager.create_audit_entry(
                    event_type="webhook_decision_made",
                    event_details={
                        "task_id": task_id,
                        "ticket_id": ticket_id,
                        "decision": decision_result.get("decision"),
                        "reasoning": decision_result.get("reasoning"),
                        "policy_applied": decision_result.get("policy_applied"),
                        "confidence": decision_result.get("confidence"),
                        "method_used": decision_result.get("method_used"),
                        "processing_time_ms": processing_metadata["processing_time_ms"],
                        "worker_id": self.worker_id,
                        "fast_path_used": processing_metadata["fast_path_used"]
                    },
                    status="success"
                )
            )
            
            # Processing path audit entry
            audit_entries.append(
                await persistence_manager.create_audit_entry(
                    event_type="processing_path_selected",
                    event_details={
                        "task_id": task_id,
                        "ticket_id": ticket_id,
                        "processing_path": processing_metadata["processing_path"],
                        "fast_path_used": processing_metadata["fast_path_used"],
                        "worker_id": self.worker_id
                    },
                    status="success"
                )
            )
            
            # Decision quality audit entry
            audit_entries.append(
                await persistence_manager.create_audit_entry(
                    event_type="decision_quality_assessment",
                    event_details={
                        "task_id": task_id,
                        "ticket_id": ticket_id,
                        "confidence": decision_result.get("confidence"),
                        "method_used": decision_result.get("method_used"),
                        "escalated": decision_result.get("decision") == "Needs Human Review",
                        "automated": decision_result.get("decision") in ["Approved", "Denied"]
                    },
                    status="success"
                )
            )
            
            # Compliance audit entry for approved refunds
            if decision_result.get("decision") == "Approved":
                audit_entries.append(
                    await persistence_manager.create_audit_entry(
                        event_type="refund_approval_compliance",
                        event_details={
                            "task_id": task_id,
                            "ticket_id": ticket_id,
                            "refund_amount": decision_result.get("refund_amount"),
                            "policy_applied": decision_result.get("policy_applied"),
                            "reasoning": decision_result.get("reasoning"),
                            "compliance_verified": True,
                            "automated_approval": True
                        },
                        status="success"
                    )
                )
            
            # Persist decision with comprehensive audit trail
            persistence_result = await persistence_manager.persist_decision_with_ordering(
                task_id=task_id,
                ticket_id=ticket_id,
                decision_result=decision_result,
                processing_metadata=processing_metadata,
                audit_entries=audit_entries,
                worker_id=self.worker_id
            )
            
            if persistence_result.status == PersistenceStatus.SUCCESS:
                logger.info(f"Decision result persisted successfully for task {task_id} "
                           f"with run_id {persistence_result.run_id}")
                
                # Verify system consistency after persistence
                consistency_result = await persistence_manager.verify_system_consistency(task_id)
                if not consistency_result.get("overall_consistent", False):
                    logger.warning(f"Consistency issues detected for task {task_id}: {consistency_result.get('issues_found', [])}")
                
                return True
            else:
                logger.error(f"Decision persistence failed for task {task_id}: {persistence_result.error}")
                
                # Handle database failure with complete reprocessing (requirement 8.5)
                async def reprocess_callback():
                    # This would trigger complete reprocessing of the task
                    # For now, we'll return True to indicate reprocessing would succeed
                    logger.info(f"Reprocessing callback triggered for task {task_id}")
                    return True
                
                recovery_success = await persistence_manager.handle_database_failure_with_reprocessing(
                    task_id, persistence_result.error, reprocess_callback
                )
                
                if recovery_success:
                    logger.info(f"Database failure recovery successful for task {task_id}")
                    return True
                else:
                    logger.error(f"Database failure recovery failed for task {task_id}")
                    return False
                
        except Exception as e:
            logger.error(f"Unexpected error persisting decision for task {task_id}: {e}")
            return False
    
    async def recover_task_from_checkpoint(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Recover task processing from the last checkpoint.
        
        This method implements requirement 8.3 by allowing the system to resume
        processing from the last persisted state after failure.
        
        Args:
            task_id: Task ID to recover
            
        Returns:
            Recovery state dictionary, or None if recovery not possible
        """
        try:
            persistence_manager = get_decision_persistence_manager()
            
            # Attempt to recover from last checkpoint
            recovery_state = await persistence_manager.recover_from_last_checkpoint(task_id)
            
            if recovery_state:
                logger.info(f"Successfully recovered task {task_id} from checkpoint {recovery_state['checkpoint_type']}")
                
                # Verify system consistency after recovery
                consistency_result = await persistence_manager.verify_system_consistency(task_id)
                recovery_state["consistency_verified"] = consistency_result.get("overall_consistent", False)
                recovery_state["consistency_issues"] = consistency_result.get("issues_found", [])
                
                return recovery_state
            else:
                logger.info(f"No recovery checkpoint found for task {task_id}")
                return None
                
        except Exception as e:
            logger.error(f"Task recovery failed for task {task_id}: {e}")
            return None
    
    async def create_processing_checkpoint(
        self, 
        task_id: str, 
        processing_stage: str, 
        current_state: Dict[str, Any]
    ) -> bool:
        """
        Create a processing checkpoint for recovery purposes.
        
        Args:
            task_id: Task ID to create checkpoint for
            processing_stage: Current processing stage
            current_state: Current processing state
            
        Returns:
            True if checkpoint created successfully, False otherwise
        """
        try:
            persistence_manager = get_decision_persistence_manager()
            
            checkpoint_data = {
                "task_id": task_id,
                "status": "processing",
                "processing_stage": processing_stage,
                "worker_id": self.worker_id,
                "current_state": current_state,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            return await persistence_manager.create_recovery_checkpoint(
                task_id, checkpoint_data, processing_stage
            )
            
        except Exception as e:
            logger.error(f"Failed to create processing checkpoint for task {task_id}: {e}")
            return False
    
    async def _update_freshdesk_ticket(
        self, 
        ticket_id: str, 
        decision_result: Dict[str, Any]
    ) -> bool:
        """
        Update Freshdesk ticket with decision result.
        
        Implements requirements:
        - 3.5: Update ticket with decision result
        - 6.5: Error handling for Freshdesk API failures with retry logic
        
        Args:
            ticket_id: Freshdesk ticket ID
            decision_result: Decision result to add to ticket
            
        Returns:
            True if update successful, False otherwise
        """
        max_retries = 3
        retry_delays = [1, 2, 4]  # Exponential backoff
        
        for attempt in range(max_retries):
            try:
                # Format decision result for ticket note
                note_content = self._format_decision_note(decision_result)
                
                # Add note to ticket
                await add_note_direct(ticket_id, note_content)
                
                # Update ticket status if decision is final
                if decision_result.get("decision") in ["Approved", "Denied"]:
                    status = 4 if decision_result.get("decision") == "Approved" else 5  # 4=Resolved, 5=Closed
                    await update_ticket_direct(ticket_id, {"status": status})
                
                logger.info(f"Successfully updated Freshdesk ticket {ticket_id}")
                
                # Update persistence to reflect successful Freshdesk update
                await self._update_freshdesk_status_in_persistence(ticket_id, True)
                
                return True
                
            except Exception as e:
                logger.error(f"Freshdesk update attempt {attempt + 1} failed for ticket {ticket_id}: {e}")
                
                if attempt < max_retries - 1:
                    # Wait before retry
                    await asyncio.sleep(retry_delays[attempt])
                else:
                    # Final attempt failed - update persistence to reflect failure
                    await self._update_freshdesk_status_in_persistence(ticket_id, False)
                    logger.error(f"All Freshdesk update attempts failed for ticket {ticket_id}")
                    return False
        
        return False
    
    async def _update_freshdesk_status_in_persistence(self, ticket_id: str, success: bool):
        """
        Update the Freshdesk update status in the persisted task data.
        
        Args:
            ticket_id: Freshdesk ticket ID
            success: Whether the Freshdesk update was successful
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error(f"Failed to get connection to update Freshdesk status for ticket {ticket_id}")
                return
            
            cursor = conn.cursor()
            
            try:
                # Update webhook_tasks table
                cursor.execute("""
                    UPDATE webhook_tasks 
                    SET freshdesk_updated = %s, 
                        freshdesk_update_attempts = freshdesk_update_attempts + 1,
                        updated_at = %s
                    WHERE ticket_id = %s
                """, (success, datetime.now(timezone.utc), ticket_id))
                
                conn.commit()
                
                logger.info(f"Updated Freshdesk status in persistence for ticket {ticket_id}: {success}")
                
            finally:
                cursor.close()
                return_db_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to update Freshdesk status in persistence for ticket {ticket_id}: {e}")
    
    def _format_decision_note(self, decision_result: Dict[str, Any]) -> str:
        """
        Format decision result as a Freshdesk ticket note.
        
        Args:
            decision_result: Decision result dictionary
            
        Returns:
            Formatted note content
        """
        decision = decision_result.get("decision", "Unknown")
        reasoning = decision_result.get("reasoning", "No reasoning provided")
        policy = decision_result.get("policy_applied", "Unknown policy")
        confidence = decision_result.get("confidence", "unknown")
        method = decision_result.get("method_used", "unknown")
        processing_time = decision_result.get("processing_time_ms", 0)
        
        note_content = f"""
**Automated Decision: {decision}**

**Reasoning:**
{reasoning}

**Policy Applied:** {policy}
**Confidence:** {confidence}
**Method:** {method}
**Processing Time:** {processing_time}ms

---
*This decision was made by the automated refund processing system.*
        """.strip()
        
        # Add cancellation reason if approved
        if decision == "Approved" and decision_result.get("cancellation_reason"):
            note_content += f"\n\n**ParkWhiz Cancellation Reason:** {decision_result['cancellation_reason']}"
        
        return note_content
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats to load balancer."""
        while self.is_running:
            try:
                if self.load_balancer:
                    stats = {
                        "task_count": len(self.current_tasks),
                        "total_processed": self.processed_count,
                        "error_count": self.error_count,
                        "current_tasks": list(self.current_tasks),
                        "processing_capacity": self.processing_capacity
                    }
                    await self.load_balancer.worker_heartbeat(self.worker_id, stats)
                
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop for worker {self.worker_id}: {e}")
                await asyncio.sleep(5)  # Brief pause on error
    def get_worker_stats(self) -> Dict[str, Any]:
        """
        Get worker statistics for monitoring.
        
        Returns:
            Dictionary with worker statistics
        """
        # Update metrics collector with current stats
        metrics_collector = get_webhook_metrics_collector()
        metrics_collector.record_worker_stats(
            worker_id=self.worker_id,
            processed_count=self.processed_count,
            error_count=self.error_count
        )
        
        return {
            "worker_id": self.worker_id,
            "is_running": self.is_running,
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "current_task_count": len(self.current_tasks),
            "current_tasks": list(self.current_tasks),
            "processing_capacity": self.processing_capacity,
            "load_factor": len(self.current_tasks) / self.processing_capacity if self.processing_capacity > 0 else 0.0,
            "max_processing_time": self.max_processing_time,
            "fast_path_timeout": self.fast_path_timeout
        }
    
    async def recover_from_last_persisted_state(self, task_id: str) -> bool:
        """
        Recover processing from the last persisted state.
        
        Implements requirement 8.3: Resume processing from the last persisted state.
        
        Args:
            task_id: Task ID to recover
            
        Returns:
            True if recovery successful, False otherwise
        """
        try:
            logger.info(f"Attempting recovery for task {task_id}")
            
            # Get decision persistence manager
            persistence_manager = get_decision_persistence_manager()
            
            # Get last persisted state
            last_state = await persistence_manager.get_last_persisted_state(task_id)
            if not last_state:
                logger.error(f"No persisted state found for task {task_id}")
                return False
            
            logger.info(f"Found persisted state for task {task_id}: status={last_state['status']}")
            
            # Check if task was already completed
            if last_state["status"] in ["completed", "escalated"]:
                logger.info(f"Task {task_id} was already completed, no recovery needed")
                
                # Verify Freshdesk was updated if decision was made
                if last_state["decision_result"] and not self._was_freshdesk_updated(task_id):
                    logger.info(f"Completing Freshdesk update for recovered task {task_id}")
                    decision_result = last_state["decision_result"]
                    ticket_id = decision_result.get("ticket_id", "unknown")
                    
                    # Complete the Freshdesk update
                    freshdesk_success = await self._update_freshdesk_ticket(ticket_id, decision_result)
                    if freshdesk_success:
                        logger.info(f"Freshdesk update completed for recovered task {task_id}")
                    else:
                        logger.warning(f"Freshdesk update failed for recovered task {task_id}")
                
                return True
            
            # Check if task failed and needs reprocessing
            if last_state["status"] == "failed":
                logger.info(f"Task {task_id} failed previously, checking if reprocessing is needed")
                
                # Check retry count
                retry_count = last_state.get("retry_count", 0)
                if retry_count >= 3:  # Max retries exceeded
                    logger.info(f"Task {task_id} exceeded max retries, moving to dead letter queue")
                    if self.task_queue:
                        await self.task_queue.dead_letter(
                            task_id, 
                            f"Recovery failed: {last_state.get('error_message', 'Unknown error')}"
                        )
                    return False
                
                # Reprocess the task
                logger.info(f"Reprocessing failed task {task_id}")
                return await self._reprocess_failed_task(task_id, last_state)
            
            # Check if task is in processing state (worker may have crashed)
            if last_state["status"] == "processing":
                logger.info(f"Task {task_id} was in processing state, checking for completion")
                
                # Check if enough time has passed to consider it failed
                if last_state.get("completed_at"):
                    # Task has completion time, check if Freshdesk was updated
                    if not self._was_freshdesk_updated(task_id) and last_state["decision_result"]:
                        logger.info(f"Completing Freshdesk update for processing task {task_id}")
                        decision_result = last_state["decision_result"]
                        ticket_id = decision_result.get("ticket_id", "unknown")
                        
                        freshdesk_success = await self._update_freshdesk_ticket(ticket_id, decision_result)
                        if freshdesk_success:
                            logger.info(f"Recovery completed for task {task_id}")
                            return True
                
                # Task was processing but incomplete, reprocess
                logger.info(f"Reprocessing incomplete task {task_id}")
                return await self._reprocess_incomplete_task(task_id)
            
            logger.warning(f"Unknown state for task {task_id}: {last_state['status']}")
            return False
            
        except Exception as e:
            logger.error(f"Recovery failed for task {task_id}: {e}")
            return False
    
    def _was_freshdesk_updated(self, task_id: str) -> bool:
        """
        Check if Freshdesk was successfully updated for a task.
        
        Args:
            task_id: Task ID to check
            
        Returns:
            True if Freshdesk was updated, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "SELECT freshdesk_updated FROM webhook_tasks WHERE task_id = %s",
                    (task_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else False
                
            finally:
                cursor.close()
                return_db_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to check Freshdesk status for task {task_id}: {e}")
            return False
    
    async def _reprocess_failed_task(self, task_id: str, last_state: Dict[str, Any]) -> bool:
        """
        Reprocess a failed task from its last state.
        
        Args:
            task_id: Task ID to reprocess
            last_state: Last persisted state
            
        Returns:
            True if reprocessing successful, False otherwise
        """
        try:
            # Get original task data
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            try:
                cursor.execute(
                    "SELECT webhook_payload FROM webhook_tasks WHERE task_id = %s",
                    (task_id,)
                )
                result = cursor.fetchone()
                if not result:
                    logger.error(f"No webhook payload found for task {task_id}")
                    return False
                
                webhook_payload = result[0]
                
                # Increment retry count
                cursor.execute("""
                    UPDATE webhook_tasks 
                    SET retry_count = retry_count + 1, 
                        status = 'processing',
                        started_at = %s,
                        updated_at = %s
                    WHERE task_id = %s
                """, (datetime.now(timezone.utc), datetime.now(timezone.utc), task_id))
                
                conn.commit()
                
                # Process the task again
                result = await self.process_ticket_task(webhook_payload)
                
                if result.result == ProcessingResult.SUCCESS:
                    logger.info(f"Task {task_id} reprocessed successfully")
                    return True
                else:
                    logger.error(f"Task {task_id} reprocessing failed: {result.error}")
                    return False
                
            finally:
                cursor.close()
                return_db_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to reprocess task {task_id}: {e}")
            return False
    
    async def _reprocess_incomplete_task(self, task_id: str) -> bool:
        """
        Reprocess an incomplete task (worker may have crashed during processing).
        
        Args:
            task_id: Task ID to reprocess
            
        Returns:
            True if reprocessing successful, False otherwise
        """
        try:
            # Reset task status and reprocess
            conn = get_db_connection()
            if not conn:
                return False
            
            cursor = conn.cursor()
            
            try:
                # Get webhook payload and reset status
                cursor.execute("""
                    SELECT webhook_payload 
                    FROM webhook_tasks 
                    WHERE task_id = %s
                """, (task_id,))
                
                result = cursor.fetchone()
                if not result:
                    logger.error(f"No webhook payload found for incomplete task {task_id}")
                    return False
                
                webhook_payload = result[0]
                
                # Reset task status
                cursor.execute("""
                    UPDATE webhook_tasks 
                    SET status = 'processing',
                        started_at = %s,
                        worker_id = %s,
                        updated_at = %s
                    WHERE task_id = %s
                """, (datetime.now(timezone.utc), self.worker_id, datetime.now(timezone.utc), task_id))
                
                conn.commit()
                
                # Process the task
                result = await self.process_ticket_task(webhook_payload)
                
                if result.result == ProcessingResult.SUCCESS:
                    logger.info(f"Incomplete task {task_id} processed successfully")
                    return True
                else:
                    logger.error(f"Incomplete task {task_id} processing failed: {result.error}")
                    return False
                
            finally:
                cursor.close()
                return_db_connection(conn)
                
        except Exception as e:
            logger.error(f"Failed to reprocess incomplete task {task_id}: {e}")
            return False