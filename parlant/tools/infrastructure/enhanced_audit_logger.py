"""
Enhanced audit logging for the Paid Again Classification System.

This module extends the existing audit logging infrastructure to include
timeline analysis results, contradiction detection events, confidence scores,
and processing mode metrics separation.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
import psycopg2
from psycopg2.extras import Json

from .database import get_db_connection, return_db_connection
from ..booking_analysis.timeline_analyzer import TimelineAnalysis
from ..refund_decision.scenario_classifier import ScenarioClassification

logger = logging.getLogger(__name__)


class EnhancedAuditLogger:
    """
    Enhanced audit logger for comprehensive decision logging.
    
    This class extends the existing audit logging to capture:
    - Timeline analysis results with calculated durations
    - Contradiction detection events with specific types
    - Confidence score logging for escalation analysis
    - Processing mode metrics separation (webhook vs chat)
    - Classification accuracy metrics and false positive/negative tracking
    """
    
    def __init__(self):
        """Initialize enhanced audit logger."""
        logger.info("EnhancedAuditLogger initialized")
    
    def log_timeline_analysis(
        self,
        run_id: str,
        ticket_id: str,
        timeline_analysis: TimelineAnalysis,
        processing_mode: str = "interactive_chat",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log timeline analysis results to audit system.
        
        Args:
            run_id: Journey run identifier
            ticket_id: Freshdesk ticket ID
            timeline_analysis: Timeline analysis results
            processing_mode: Processing mode (webhook_automation/interactive_chat)
            metadata: Additional context
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for timeline analysis logging")
                return False
            
            cursor = conn.cursor()
            
            # Prepare timeline analysis data
            timeline_data = {
                "booking_start": timeline_analysis.booking_start.isoformat() if timeline_analysis.booking_start else None,
                "booking_end": timeline_analysis.booking_end.isoformat() if timeline_analysis.booking_end else None,
                "arrival_time": timeline_analysis.arrival_time.isoformat() if timeline_analysis.arrival_time else None,
                "exit_time": timeline_analysis.exit_time.isoformat() if timeline_analysis.exit_time else None,
                "early_arrival_minutes": timeline_analysis.early_arrival_minutes,
                "overstay_minutes": timeline_analysis.overstay_minutes,
                "actual_parking_duration": timeline_analysis.actual_parking_duration,
                "booked_duration": timeline_analysis.booked_duration,
                "has_early_arrival": timeline_analysis.has_early_arrival,
                "has_overstay": timeline_analysis.has_overstay,
                "is_timeline_consistent": timeline_analysis.is_timeline_consistent,
                "timeline_issues": timeline_analysis.timeline_issues,
                "processing_mode": processing_mode
            }
            
            # Add metadata if provided
            if metadata:
                timeline_data.update(metadata)
            
            # Insert into audit log
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status
                ) VALUES (%s, %s, %s, %s)
            """, (
                run_id,
                "timeline_analysis_complete",
                Json(timeline_data),
                "success"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Timeline analysis logged for ticket {ticket_id} (run: {run_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log timeline analysis: {e}")
            if conn:
                return_db_connection(conn)
            return False
    
    def log_contradiction_detection(
        self,
        run_id: str,
        ticket_id: str,
        contradictions: List[str],
        contradiction_types: List[str],
        processing_mode: str = "interactive_chat",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log contradiction detection events with specific contradiction types.
        
        Args:
            run_id: Journey run identifier
            ticket_id: Freshdesk ticket ID
            contradictions: List of detected contradictions
            contradiction_types: List of contradiction types
            processing_mode: Processing mode (webhook_automation/interactive_chat)
            metadata: Additional context
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for contradiction logging")
                return False
            
            cursor = conn.cursor()
            
            # Prepare contradiction data
            contradiction_data = {
                "ticket_id": ticket_id,
                "contradictions": contradictions,
                "contradiction_types": contradiction_types,
                "contradiction_count": len(contradictions),
                "processing_mode": processing_mode,
                "detected_at": datetime.utcnow().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                contradiction_data.update(metadata)
            
            # Insert into audit log
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status
                ) VALUES (%s, %s, %s, %s)
            """, (
                run_id,
                "contradiction_detection",
                Json(contradiction_data),
                "success"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Contradiction detection logged for ticket {ticket_id}: {len(contradictions)} contradictions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log contradiction detection: {e}")
            if conn:
                return_db_connection(conn)
            return False
    
    def log_scenario_classification(
        self,
        run_id: str,
        ticket_id: str,
        classification: ScenarioClassification,
        processing_mode: str = "interactive_chat",
        confidence_threshold: float = 0.70,
        escalation_triggered: bool = False,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log scenario classification results with confidence scores.
        
        Args:
            run_id: Journey run identifier
            ticket_id: Freshdesk ticket ID
            classification: Scenario classification results
            processing_mode: Processing mode (webhook_automation/interactive_chat)
            confidence_threshold: Threshold used for decision
            escalation_triggered: Whether escalation was triggered
            metadata: Additional context
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for classification logging")
                return False
            
            cursor = conn.cursor()
            
            # Prepare classification data
            classification_data = {
                "ticket_id": ticket_id,
                "scenario_type": classification.scenario_type,
                "confidence": classification.confidence,
                "reasoning": classification.reasoning,
                "recommended_action": classification.recommended_action,
                "refund_eligible": classification.refund_eligible,
                "key_factors": classification.key_factors,
                "contradictions": classification.contradictions,
                "processing_mode": processing_mode,
                "confidence_threshold": confidence_threshold,
                "escalation_triggered": escalation_triggered,
                "classified_at": datetime.utcnow().isoformat()
            }
            
            # Add timeline evidence if present
            if classification.timeline_evidence:
                classification_data["timeline_evidence"] = {
                    "early_arrival_minutes": classification.timeline_evidence.early_arrival_minutes,
                    "overstay_minutes": classification.timeline_evidence.overstay_minutes,
                    "has_early_arrival": classification.timeline_evidence.has_early_arrival,
                    "has_overstay": classification.timeline_evidence.has_overstay,
                    "is_timeline_consistent": classification.timeline_evidence.is_timeline_consistent
                }
            
            # Add metadata if provided
            if metadata:
                classification_data.update(metadata)
            
            # Insert into audit log
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status
                ) VALUES (%s, %s, %s, %s)
            """, (
                run_id,
                "scenario_classification",
                Json(classification_data),
                "success"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Scenario classification logged for ticket {ticket_id}: {classification.scenario_type} "
                       f"(confidence: {classification.confidence:.2f})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log scenario classification: {e}")
            if conn:
                return_db_connection(conn)
            return False
    
    def log_processing_mode_metrics(
        self,
        run_id: str,
        ticket_id: str,
        processing_mode: str,
        processing_time_ms: int,
        decision: str,
        confidence: float,
        escalated: bool = False,
        llm_used: bool = False,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log processing mode metrics with separation between webhook and chat.
        
        Args:
            run_id: Journey run identifier
            ticket_id: Freshdesk ticket ID
            processing_mode: Processing mode (webhook_automation/interactive_chat)
            processing_time_ms: Total processing time in milliseconds
            decision: Final decision made
            confidence: Decision confidence score
            escalated: Whether case was escalated
            llm_used: Whether LLM was used in processing
            metadata: Additional metrics
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for processing mode metrics")
                return False
            
            cursor = conn.cursor()
            
            # Prepare processing mode metrics
            metrics_data = {
                "ticket_id": ticket_id,
                "processing_mode": processing_mode,
                "processing_time_ms": processing_time_ms,
                "decision": decision,
                "confidence": confidence,
                "escalated": escalated,
                "llm_used": llm_used,
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                metrics_data.update(metadata)
            
            # Insert into audit log
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status
                ) VALUES (%s, %s, %s, %s)
            """, (
                run_id,
                "processing_mode_metrics",
                Json(metrics_data),
                "success"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Processing mode metrics logged for ticket {ticket_id}: {processing_mode} "
                       f"({processing_time_ms}ms, {decision})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log processing mode metrics: {e}")
            if conn:
                return_db_connection(conn)
            return False
    
    def log_accuracy_metrics(
        self,
        run_id: str,
        ticket_id: str,
        predicted_decision: str,
        actual_decision: Optional[str] = None,
        human_override: bool = False,
        override_reason: Optional[str] = None,
        false_positive: bool = False,
        false_negative: bool = False,
        processing_mode: str = "interactive_chat",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Log classification accuracy metrics and false positive/negative tracking.
        
        Args:
            run_id: Journey run identifier
            ticket_id: Freshdesk ticket ID
            predicted_decision: AI-predicted decision
            actual_decision: Actual final decision (if different)
            human_override: Whether human overrode AI decision
            override_reason: Reason for human override
            false_positive: Whether this was a false positive
            false_negative: Whether this was a false negative
            processing_mode: Processing mode
            metadata: Additional accuracy data
            
        Returns:
            True if logged successfully, False otherwise
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for accuracy metrics")
                return False
            
            cursor = conn.cursor()
            
            # Prepare accuracy metrics
            accuracy_data = {
                "ticket_id": ticket_id,
                "predicted_decision": predicted_decision,
                "actual_decision": actual_decision,
                "human_override": human_override,
                "override_reason": override_reason,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "processing_mode": processing_mode,
                "accuracy_correct": predicted_decision == actual_decision if actual_decision else None,
                "logged_at": datetime.utcnow().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                accuracy_data.update(metadata)
            
            # Insert into audit log
            cursor.execute("""
                INSERT INTO agent_audit_log (
                    run_id, event_type, event_details, status
                ) VALUES (%s, %s, %s, %s)
            """, (
                run_id,
                "accuracy_metrics",
                Json(accuracy_data),
                "success"
            ))
            
            conn.commit()
            cursor.close()
            return_db_connection(conn)
            
            logger.info(f"Accuracy metrics logged for ticket {ticket_id}: predicted={predicted_decision}, "
                       f"actual={actual_decision}, override={human_override}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log accuracy metrics: {e}")
            if conn:
                return_db_connection(conn)
            return False
    
    def get_processing_mode_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        processing_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get summary of processing mode metrics for analysis.
        
        Args:
            start_date: Start date for analysis (defaults to last 30 days)
            end_date: End date for analysis (defaults to now)
            processing_mode: Specific mode to analyze (None for both)
            
        Returns:
            Dict with processing mode summary statistics
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for processing mode summary")
                return {}
            
            cursor = conn.cursor()
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = datetime.utcnow().replace(day=1)  # Start of current month
            
            # Build query conditions
            conditions = ["timestamp >= %s", "timestamp <= %s"]
            params = [start_date, end_date]
            
            if processing_mode:
                conditions.append("event_details->>'processing_mode' = %s")
                params.append(processing_mode)
            
            # Query processing mode metrics
            cursor.execute(f"""
                SELECT 
                    event_details->>'processing_mode' as mode,
                    COUNT(*) as total_cases,
                    AVG((event_details->>'processing_time_ms')::int) as avg_processing_time_ms,
                    COUNT(CASE WHEN event_details->>'escalated' = 'true' THEN 1 END) as escalated_count,
                    COUNT(CASE WHEN event_details->>'llm_used' = 'true' THEN 1 END) as llm_used_count,
                    AVG((event_details->>'confidence')::float) as avg_confidence,
                    COUNT(CASE WHEN event_details->>'decision' = 'Approved' THEN 1 END) as approved_count,
                    COUNT(CASE WHEN event_details->>'decision' = 'Denied' THEN 1 END) as denied_count,
                    COUNT(CASE WHEN event_details->>'decision' = 'Needs Human Review' THEN 1 END) as escalated_decision_count
                FROM agent_audit_log 
                WHERE event_type = 'processing_mode_metrics'
                AND {' AND '.join(conditions)}
                GROUP BY event_details->>'processing_mode'
                ORDER BY mode
            """, params)
            
            results = cursor.fetchall()
            cursor.close()
            return_db_connection(conn)
            
            # Format results
            summary = {
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "modes": {}
            }
            
            for row in results:
                mode = row[0] or "unknown"
                summary["modes"][mode] = {
                    "total_cases": row[1],
                    "avg_processing_time_ms": round(row[2], 2) if row[2] else 0,
                    "escalated_count": row[3],
                    "escalation_rate": round((row[3] / row[1]) * 100, 2) if row[1] > 0 else 0,
                    "llm_used_count": row[4],
                    "llm_usage_rate": round((row[4] / row[1]) * 100, 2) if row[1] > 0 else 0,
                    "avg_confidence": round(row[5], 3) if row[5] else 0,
                    "approved_count": row[6],
                    "denied_count": row[7],
                    "escalated_decision_count": row[8]
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get processing mode summary: {e}")
            if conn:
                return_db_connection(conn)
            return {}
    
    def get_accuracy_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        processing_mode: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate accuracy report with false positive/negative rates.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            processing_mode: Specific mode to analyze
            
        Returns:
            Dict with accuracy metrics and false positive/negative rates
        """
        try:
            conn = get_db_connection()
            if not conn:
                logger.error("Failed to get database connection for accuracy report")
                return {}
            
            cursor = conn.cursor()
            
            # Set default date range if not provided
            if not end_date:
                end_date = datetime.utcnow()
            if not start_date:
                start_date = datetime.utcnow().replace(day=1)  # Start of current month
            
            # Build query conditions
            conditions = ["timestamp >= %s", "timestamp <= %s"]
            params = [start_date, end_date]
            
            if processing_mode:
                conditions.append("event_details->>'processing_mode' = %s")
                params.append(processing_mode)
            
            # Query accuracy metrics
            cursor.execute(f"""
                SELECT 
                    event_details->>'processing_mode' as mode,
                    COUNT(*) as total_cases,
                    COUNT(CASE WHEN event_details->>'accuracy_correct' = 'true' THEN 1 END) as correct_predictions,
                    COUNT(CASE WHEN event_details->>'human_override' = 'true' THEN 1 END) as human_overrides,
                    COUNT(CASE WHEN event_details->>'false_positive' = 'true' THEN 1 END) as false_positives,
                    COUNT(CASE WHEN event_details->>'false_negative' = 'true' THEN 1 END) as false_negatives
                FROM agent_audit_log 
                WHERE event_type = 'accuracy_metrics'
                AND {' AND '.join(conditions)}
                GROUP BY event_details->>'processing_mode'
                ORDER BY mode
            """, params)
            
            results = cursor.fetchall()
            cursor.close()
            return_db_connection(conn)
            
            # Format results
            report = {
                "date_range": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat()
                },
                "accuracy_by_mode": {}
            }
            
            total_cases = 0
            total_correct = 0
            total_false_positives = 0
            total_false_negatives = 0
            
            for row in results:
                mode = row[0] or "unknown"
                cases = row[1]
                correct = row[2]
                overrides = row[3]
                false_pos = row[4]
                false_neg = row[5]
                
                total_cases += cases
                total_correct += correct
                total_false_positives += false_pos
                total_false_negatives += false_neg
                
                report["accuracy_by_mode"][mode] = {
                    "total_cases": cases,
                    "correct_predictions": correct,
                    "accuracy_rate": round((correct / cases) * 100, 2) if cases > 0 else 0,
                    "human_overrides": overrides,
                    "override_rate": round((overrides / cases) * 100, 2) if cases > 0 else 0,
                    "false_positives": false_pos,
                    "false_positive_rate": round((false_pos / cases) * 100, 2) if cases > 0 else 0,
                    "false_negatives": false_neg,
                    "false_negative_rate": round((false_neg / cases) * 100, 2) if cases > 0 else 0
                }
            
            # Add overall summary
            report["overall_summary"] = {
                "total_cases": total_cases,
                "overall_accuracy_rate": round((total_correct / total_cases) * 100, 2) if total_cases > 0 else 0,
                "overall_false_positive_rate": round((total_false_positives / total_cases) * 100, 2) if total_cases > 0 else 0,
                "overall_false_negative_rate": round((total_false_negatives / total_cases) * 100, 2) if total_cases > 0 else 0
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate accuracy report: {e}")
            if conn:
                return_db_connection(conn)
            return {}


# Global enhanced audit logger instance
_enhanced_audit_logger: Optional[EnhancedAuditLogger] = None


def get_enhanced_audit_logger() -> EnhancedAuditLogger:
    """
    Get the global enhanced audit logger instance.
    
    Returns:
        The global EnhancedAuditLogger instance
    """
    global _enhanced_audit_logger
    if _enhanced_audit_logger is None:
        _enhanced_audit_logger = EnhancedAuditLogger()
    return _enhanced_audit_logger