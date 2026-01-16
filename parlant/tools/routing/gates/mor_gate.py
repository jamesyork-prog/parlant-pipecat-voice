"""
MOR (Merchant of Record) Gate implementation.

This gate determines if a ticket should be instantly rejected based on
the merchant of record (third-party platforms vs ParkWhiz).
"""

import time
import logging
from typing import Dict, Set, Optional

from ..models import MORDecision, MORGateConfig, MORMetrics
from ..audit_logger import get_gate_audit_logger

logger = logging.getLogger(__name__)


class MORGate:
    """
    Gate 2: Merchant of Record classification.
    
    Instantly rejects third-party MOR purchases (AXS, SeatGeek, etc.)
    Flags special partners (Groupon credit-only policy)
    Passes ParkWhiz/Google/Waze/Apple to next gate
    
    Performance: <20ms per ticket
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize MOR gate with vendor configuration.
        
        Args:
            config: Optional configuration with third-party MOR lists
                   If not provided, uses default policy lists
        """
        self.config = config or {}
        self.gate_config = MORGateConfig()
        self.metrics = MORMetrics()
        self.audit_logger = get_gate_audit_logger()
        
        # Override defaults with provided config
        if "third_party_vendors" in self.config:
            self.gate_config.third_party_vendors = set(self.config["third_party_vendors"])
        if "parkwhiz_partners" in self.config:
            self.gate_config.parkwhiz_partners = set(self.config["parkwhiz_partners"])
        if "special_partners" in self.config:
            self.gate_config.special_partners = self.config["special_partners"]
        
        # Override default LLM cost if provided
        if "average_llm_cost_per_ticket" in self.config:
            self.metrics.average_llm_cost_per_ticket = self.config["average_llm_cost_per_ticket"]
    
    def check_mor(self, ticket_data: Dict) -> MORDecision:
        """
        Check merchant of record and determine refundability.
        
        Args:
            ticket_data: Zapier webhook payload with "Partner" field
            
        Returns:
            MORDecision with routing instruction and rejection reason if applicable
            
        Processing time: <20ms (set lookup and string operations)
        """
        start = time.perf_counter()
        
        # Extract and normalize Partner field
        partner = ticket_data.get("Partner", "") or ""
        if partner is None:
            partner = ""
        partner = partner.strip()
        
        # Handle missing/empty Partner field
        if not partner:
            logger.warning("Partner field missing - defaulting to ParkWhiz MOR")
            processing_time_ms = int((time.perf_counter() - start) * 1000)
            
            # Log data validation warning
            self.audit_logger.log_data_warning(
                ticket_id=ticket_data.get("id", "unknown"),
                warning_type="missing_field",
                field_name="Partner",
                field_value=None,
                warning_message="Partner field missing - defaulting to ParkWhiz MOR"
            )
            
            # Log MOR decision
            self.audit_logger.log_mor_decision(
                ticket_id=ticket_data.get("id", "unknown"),
                partner_value="",
                vendor_detected="unknown",
                policy_rule="default_parkwhiz_assumption",
                decision="passed_with_warning",
                processing_time_ms=processing_time_ms
            )
            
            self.metrics.add_pass_through("unknown", processing_time_ms)
            return MORDecision(
                should_continue=True,
                processing_time_ms=processing_time_ms
            )
        
        # Normalize for case-insensitive matching
        partner_lower = partner.lower()
        
        # Check for third-party MOR (instant rejection)
        for vendor in self.gate_config.third_party_vendors:
            if vendor in partner_lower:
                vendor_display = vendor.replace("_", " ").title()
                processing_time_ms = int((time.perf_counter() - start) * 1000)
                rejection_reason = self._format_third_party_rejection(vendor_display)
                
                # Log MOR rejection decision
                self.audit_logger.log_mor_decision(
                    ticket_id=ticket_data.get("id", "unknown"),
                    partner_value=partner,
                    vendor_detected=vendor_display,
                    policy_rule="third_party_mor_instant_rejection",
                    decision="rejected",
                    processing_time_ms=processing_time_ms,
                    rejection_reason=rejection_reason
                )
                
                # Track metrics
                self.metrics.add_rejection(vendor_display, processing_time_ms)
                
                return MORDecision(
                    should_continue=False,
                    vendor_name=vendor_display,
                    rejection_reason=rejection_reason,
                    processing_time_ms=processing_time_ms
                )
        
        # Check for special partners (Groupon credit-only)
        for special_vendor, flag in self.gate_config.special_partners.items():
            if special_vendor in partner_lower:
                processing_time_ms = int((time.perf_counter() - start) * 1000)
                
                # Log MOR decision with special flag
                self.audit_logger.log_mor_decision(
                    ticket_id=ticket_data.get("id", "unknown"),
                    partner_value=partner,
                    vendor_detected=special_vendor.title(),
                    policy_rule="groupon_credit_only_policy",
                    decision="passed_with_flag",
                    processing_time_ms=processing_time_ms,
                    special_flag=flag
                )
                
                self.metrics.add_pass_through("groupon", processing_time_ms)
                return MORDecision(
                    should_continue=True,
                    special_flag=flag,
                    processing_time_ms=processing_time_ms
                )
        
        # Check for ParkWhiz-processable MOR
        if any(p in partner_lower for p in self.gate_config.parkwhiz_partners):
            processing_time_ms = int((time.perf_counter() - start) * 1000)
            
            # Log MOR pass-through decision
            self.audit_logger.log_mor_decision(
                ticket_id=ticket_data.get("id", "unknown"),
                partner_value=partner,
                vendor_detected=partner,
                policy_rule="parkwhiz_mor_pass_through",
                decision="passed",
                processing_time_ms=processing_time_ms
            )
            
            self.metrics.add_pass_through("parkwhiz", processing_time_ms)
            return MORDecision(
                should_continue=True,
                processing_time_ms=processing_time_ms
            )
        
        # Unknown partner - log and continue with warning
        logger.warning(f"Unknown Partner value: '{partner}' - defaulting to ParkWhiz MOR")
        processing_time_ms = int((time.perf_counter() - start) * 1000)
        
        # Log data validation warning for unknown partner
        self.audit_logger.log_data_warning(
            ticket_id=ticket_data.get("id", "unknown"),
            warning_type="unknown_value",
            field_name="Partner",
            field_value=partner,
            warning_message=f"Unknown Partner value: '{partner}' - defaulting to ParkWhiz MOR"
        )
        
        # Log MOR decision with warning
        self.audit_logger.log_mor_decision(
            ticket_id=ticket_data.get("id", "unknown"),
            partner_value=partner,
            vendor_detected="unknown",
            policy_rule="default_parkwhiz_assumption",
            decision="passed_with_warning",
            processing_time_ms=processing_time_ms
        )
        
        self.metrics.add_pass_through("unknown", processing_time_ms)
        return MORDecision(
            should_continue=True,
            processing_time_ms=processing_time_ms
        )
    
    def get_metrics(self) -> MORMetrics:
        """Get current metrics for this gate."""
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset metrics to initial state."""
        self.metrics = MORMetrics()
        # Preserve LLM cost setting
        if "average_llm_cost_per_ticket" in self.config:
            self.metrics.average_llm_cost_per_ticket = self.config["average_llm_cost_per_ticket"]
    
    def _format_third_party_rejection(self, vendor_name: str) -> str:
        """
        Create customer-friendly rejection message for third-party MOR.
        
        Args:
            vendor_name: Display name of third-party vendor
            
        Returns:
            Formatted rejection reason suitable for ticket response
        """
        return (
            f"Non-refundable. This pass was purchased through {vendor_name}. "
            f"ParkWhiz cannot process refunds for passes purchased through third-party platforms. "
            f"Please contact {vendor_name} directly for refund assistance."
        )