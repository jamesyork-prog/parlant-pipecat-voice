"""
Ticket routing system with MOR and Product Type gates.

This module implements the TicketRouter class that processes tickets through
a series of gates to determine refundability before expensive LLM processing.
"""

import time
import logging
import threading
from typing import Dict, Optional, List
from dataclasses import dataclass

from .models import MORProductTypeResult, PreArrivalDecision
from .gates.mor_gate import MORGate
from .gates.product_type_gate import ProductTypeGate
from .gates.pre_arrival_gate import PreArrivalGate
from .config_loader import RouterConfigLoader


@dataclass
class RoutingDecision:
    """Final routing decision from the ticket router."""
    should_continue: bool  # True = continue to LLM processing, False = instant reject/approve
    rejection_reason: Optional[str] = None
    rejection_source: Optional[str] = None  # Which gate rejected
    approval_reason: Optional[str] = None  # Pre-arrival approval reasoning
    approval_source: Optional[str] = None  # Which gate approved (e.g., "pre_arrival_gate")
    special_flags: List[str] = None  # Special handling flags (e.g., "groupon_credit_only")
    processing_time_ms: int = 0
    gates_evaluated: List[str] = None
    warnings: List[str] = None
    # Pre-arrival specific fields
    hours_before_booking: Optional[float] = None  # Hours before booking start (for pre-arrival approvals)
    timezone_used: Optional[str] = None  # Timezone used for pre-arrival comparison
    
    def __post_init__(self):
        if self.special_flags is None:
            self.special_flags = []
        if self.gates_evaluated is None:
            self.gates_evaluated = []
        if self.warnings is None:
            self.warnings = []


class TicketRouter:
    """
    Main ticket routing system that processes tickets through multiple gates.
    
    Gate execution order:
    1. Already Refunded Check (existing logic - not implemented here)
    2. MOR (Merchant of Record) Check ← NEW
    3. User Flags Check (existing logic - not implemented here)  
    4. Product Type Check ← NEW
    5. Pre-Arrival Fast Path ← NEW
    6. Scenario Classification (future)
    
    Performance targets:
    - MOR gate: <20ms
    - Product Type gate: <50ms
    - Pre-Arrival gate: <200ms
    - Combined: <300ms
    
    Features:
    - Configuration-based vendor lists (no code changes to add vendors)
    - Hot reload support for configuration updates
    - Fallback to safe defaults on invalid configuration
    - Comprehensive validation and error handling
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ticket router with configuration loading.
        
        Args:
            config_path: Path to configuration file. If None, uses default location.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration loader
        self.config_loader = RouterConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        
        # Check if MOR/Product Type gates are enabled
        self._gates_enabled = self.config.get("router_settings", {}).get("enable_mor_product_type_gates", True)
        
        # Check if Pre-Arrival gate is enabled
        self._pre_arrival_enabled = self.config.get("router_settings", {}).get("enable_pre_arrival_gate", True)
        
        # Initialize gates with loaded configuration
        self._initialize_gates()
        
        # Hot reload support
        self._hot_reload_enabled = self.config.get("router_settings", {}).get("enable_hot_reload", True)
        self._config_check_interval = self.config.get("router_settings", {}).get("config_check_interval_seconds", 30)
        self._last_config_check = time.time()
        self._config_lock = threading.RLock()
        
        self.logger.info("TicketRouter initialized with configuration-based MOR and Product Type gates")
        
        # Log configuration status
        config_info = self.config_loader.get_config_info()
        self.logger.info(f"Configuration loaded from: {config_info['config_path']}")
        
        validation_result = self.config_loader.get_validation_result()
        if validation_result:
            if validation_result.is_valid:
                self.logger.info("Configuration validation: PASSED")
                if validation_result.warnings:
                    self.logger.info(f"Configuration warnings: {len(validation_result.warnings)}")
            else:
                self.logger.warning(f"Configuration validation: FAILED ({len(validation_result.errors)} errors)")
                self.logger.info("Using safe default configuration")
    
    def _initialize_gates(self):
        """Initialize gates with current configuration."""
        if self._gates_enabled:
            self.mor_gate = MORGate(self.config.get("mor_gate"))
            self.product_type_gate = ProductTypeGate(self.config.get("product_type_gate"))
        else:
            self.mor_gate = None
            self.product_type_gate = None
        
        if self._pre_arrival_enabled:
            self.pre_arrival_gate = PreArrivalGate(self.config.get("pre_arrival_gate"))
        else:
            self.pre_arrival_gate = None
    
    def _check_config_updates(self):
        """Check for configuration updates and reload if necessary."""
        if not self._hot_reload_enabled:
            return
        
        current_time = time.time()
        if current_time - self._last_config_check < self._config_check_interval:
            return
        
        self._last_config_check = current_time
        
        if self.config_loader.check_for_updates():
            self.logger.info("Configuration file updated - reloading")
            
            with self._config_lock:
                try:
                    # Reload configuration
                    new_config = self.config_loader.load_config(force_reload=True)
                    
                    # Reinitialize gates with new configuration
                    old_config = self.config
                    self.config = new_config
                    self._gates_enabled = self.config.get("router_settings", {}).get("enable_mor_product_type_gates", True)
                    self._pre_arrival_enabled = self.config.get("router_settings", {}).get("enable_pre_arrival_gate", True)
                    self._initialize_gates()
                    
                    self.logger.info("Configuration hot reload completed successfully")
                    
                    # Log what changed
                    self._log_config_changes(old_config, new_config)
                    
                except Exception as e:
                    self.logger.error(f"Configuration hot reload failed: {e}")
                    self.logger.warning("Continuing with previous configuration")
    
    def _log_config_changes(self, old_config: Dict, new_config: Dict):
        """Log what changed between configurations."""
        try:
            # Check MOR gate changes
            old_mor = old_config.get("mor_gate", {})
            new_mor = new_config.get("mor_gate", {})
            
            if old_mor.get("third_party_vendors") != new_mor.get("third_party_vendors"):
                self.logger.info("MOR gate third-party vendors list updated")
            
            if old_mor.get("parkwhiz_partners") != new_mor.get("parkwhiz_partners"):
                self.logger.info("MOR gate ParkWhiz partners list updated")
            
            if old_mor.get("special_partners") != new_mor.get("special_partners"):
                self.logger.info("MOR gate special partners updated")
            
            # Check Product Type gate changes
            old_pt = old_config.get("product_type_gate", {})
            new_pt = new_config.get("product_type_gate", {})
            
            if old_pt.get("on_demand_threshold_hours") != new_pt.get("on_demand_threshold_hours"):
                old_threshold = old_pt.get("on_demand_threshold_hours", 4)
                new_threshold = new_pt.get("on_demand_threshold_hours", 4)
                self.logger.info(f"Product Type gate on-demand threshold updated: {old_threshold}h → {new_threshold}h")
            
            # Check Pre-Arrival gate changes
            old_pa = old_config.get("pre_arrival_gate", {})
            new_pa = new_config.get("pre_arrival_gate", {})
            
            if old_pa.get("date_parser", {}).get("date_formats") != new_pa.get("date_parser", {}).get("date_formats"):
                self.logger.info("Pre-Arrival gate date formats updated")
            
            if old_pa.get("timezone_handler", {}).get("city_to_timezone_mapping") != new_pa.get("timezone_handler", {}).get("city_to_timezone_mapping"):
                self.logger.info("Pre-Arrival gate city-to-timezone mapping updated")
            
        except Exception as e:
            self.logger.warning(f"Error logging configuration changes: {e}")
    
    def route_ticket(self, ticket_data: Dict) -> RoutingDecision:
        """
        Route ticket through all gates to determine processing path.
        
        Args:
            ticket_data: Zapier webhook payload with ticket fields
            
        Returns:
            RoutingDecision with routing instruction and metadata
            
        Processing order:
        1. Check for configuration updates (if hot reload enabled)
        2. MOR Check (Gate 2) - instant reject third-party MOR
        3. Product Type Check (Gate 4) - instant reject on-demand/season packages
        4. Pre-Arrival Check (Gate 5) - instant approve pre-arrival cancellations
        5. Continue to next gates if all pass
        """
        # Check for configuration updates before processing
        self._check_config_updates()
        
        start_time = time.perf_counter()
        gates_evaluated = []
        warnings = []
        special_flags = []
        
        with self._config_lock:
            self.logger.info(
                "Starting ticket routing",
                extra={"ticket_id": ticket_data.get("id", "unknown")}
            )
            
            # Check if gates are disabled - if so, return continue immediately
            if not self._gates_enabled:
                total_time = int((time.perf_counter() - start_time) * 1000)
                self.logger.info(
                    "MOR/Product Type gates disabled - continuing to next gates",
                    extra={"ticket_id": ticket_data.get("id", "unknown")}
                )
                return RoutingDecision(
                    should_continue=True,
                    processing_time_ms=total_time,
                    gates_evaluated=["gates_disabled"],
                    warnings=["MOR/Product Type gates are disabled in configuration"]
                )
            
            # Gate 2: MOR (Merchant of Record) Check
            self.logger.debug("Executing MOR gate check")
            gates_evaluated.append("mor_gate")
            
            mor_result = self.mor_gate.check_mor(ticket_data)
            
            if not mor_result.should_continue:
                # MOR gate rejected - early exit
                total_time = int((time.perf_counter() - start_time) * 1000)
                
                self.logger.info(
                    "Ticket rejected by MOR gate",
                    extra={
                        "ticket_id": ticket_data.get("id", "unknown"),
                        "vendor": mor_result.vendor_name,
                        "processing_time_ms": total_time
                    }
                )
                
                return RoutingDecision(
                    should_continue=False,
                    rejection_reason=mor_result.rejection_reason,
                    rejection_source="mor_gate",
                    processing_time_ms=total_time,
                    gates_evaluated=gates_evaluated
                )
            
            # Handle special MOR flags (e.g., Groupon credit-only)
            if mor_result.special_flag:
                special_flags.append(mor_result.special_flag)
                self.logger.info(
                    "MOR gate set special flag",
                    extra={
                        "ticket_id": ticket_data.get("id", "unknown"),
                        "special_flag": mor_result.special_flag
                    }
                )
            
            # Gate 4: Product Type Check
            self.logger.debug("Executing Product Type gate check")
            gates_evaluated.append("product_type_gate")
            
            product_type_result = self.product_type_gate.check_product_type(ticket_data)
            
            if not product_type_result.should_continue:
                # Product Type gate rejected - early exit
                total_time = int((time.perf_counter() - start_time) * 1000)
                
                self.logger.info(
                    "Ticket rejected by Product Type gate",
                    extra={
                        "ticket_id": ticket_data.get("id", "unknown"),
                        "product_type": product_type_result.product_type,
                        "inference_method": product_type_result.inference_method,
                        "processing_time_ms": total_time
                    }
                )
                
                return RoutingDecision(
                    should_continue=False,
                    rejection_reason=product_type_result.rejection_reason,
                    rejection_source="product_type_gate",
                    processing_time_ms=total_time,
                    gates_evaluated=gates_evaluated,
                    special_flags=special_flags
                )
            
            # Gate 5: Pre-Arrival Check (if enabled)
            if self._pre_arrival_enabled and self.pre_arrival_gate:
                self.logger.debug("Executing Pre-Arrival gate check")
                gates_evaluated.append("pre_arrival_gate")
                
                pre_arrival_result = self.pre_arrival_gate.check_pre_arrival(ticket_data, context="webhook")
                
                # Add any warnings from pre-arrival gate
                if pre_arrival_result.warnings:
                    warnings.extend(pre_arrival_result.warnings)
                
                if not pre_arrival_result.should_continue:
                    # Pre-arrival gate approved - early exit with approval
                    total_time = int((time.perf_counter() - start_time) * 1000)
                    
                    self.logger.info(
                        "Ticket approved by Pre-Arrival gate",
                        extra={
                            "ticket_id": ticket_data.get("id", "unknown"),
                            "hours_before_booking": pre_arrival_result.hours_before_booking,
                            "timezone_used": pre_arrival_result.timezone_used,
                            "processing_time_ms": total_time
                        }
                    )
                    
                    return RoutingDecision(
                        should_continue=False,
                        approval_reason=pre_arrival_result.reasoning,
                        approval_source="pre_arrival_gate",
                        processing_time_ms=total_time,
                        gates_evaluated=gates_evaluated,
                        special_flags=special_flags,
                        warnings=warnings,
                        hours_before_booking=pre_arrival_result.hours_before_booking,
                        timezone_used=pre_arrival_result.timezone_used
                    )
                
                # Pre-arrival gate determined NOT pre-arrival - continue to next gates
                self.logger.debug(
                    "Pre-arrival gate: NOT pre-arrival, continuing to next gate",
                    extra={
                        "ticket_id": ticket_data.get("id", "unknown"),
                        "timezone_used": pre_arrival_result.timezone_used
                    }
                )
            
            # All gates passed - continue to next gates
            total_time = int((time.perf_counter() - start_time) * 1000)
            
            gate_summary = {
                "mor_vendor": mor_result.vendor_name,
                "product_type": product_type_result.product_type,
                "inference_method": product_type_result.inference_method
            }
            
            # Add pre-arrival info if gate was executed
            if "pre_arrival_gate" in gates_evaluated:
                gate_summary["pre_arrival_checked"] = True
                gate_summary["timezone_used"] = getattr(pre_arrival_result, 'timezone_used', 'unknown')
            
            self.logger.info(
                "Ticket passed all routing gates",
                extra={
                    "ticket_id": ticket_data.get("id", "unknown"),
                    "processing_time_ms": total_time,
                    **gate_summary
                }
            )
            
            return RoutingDecision(
                should_continue=True,
                processing_time_ms=total_time,
                gates_evaluated=gates_evaluated,
                special_flags=special_flags,
                warnings=warnings
            )
    
    def get_combined_result(self, ticket_data: Dict) -> MORProductTypeResult:
        """
        Get detailed combined result from both MOR and Product Type gates.
        
        This method provides more detailed information than route_ticket()
        and is useful for testing and debugging.
        
        Args:
            ticket_data: Zapier webhook payload
            
        Returns:
            MORProductTypeResult with detailed gate results
        """
        # Check for configuration updates before processing
        self._check_config_updates()
        
        start_time = time.perf_counter()
        
        with self._config_lock:
            # Check if gates are disabled - if so, execute both gates but return combined result
            if not self._gates_enabled:
                total_time = int((time.perf_counter() - start_time) * 1000)
                return MORProductTypeResult(
                    mor_passed=True,
                    product_type_passed=True,
                    should_continue=True,
                    total_processing_ms=total_time,
                    gates_evaluated=["gates_disabled"],
                    warnings=["MOR/Product Type gates are disabled in configuration"]
                )
            
            # Execute MOR gate
            mor_result = self.mor_gate.check_mor(ticket_data)
            
            # Execute Product Type gate (even if MOR rejected, for complete analysis)
            product_type_result = self.product_type_gate.check_product_type(ticket_data)
            
            total_time = int((time.perf_counter() - start_time) * 1000)
            
            # Determine overall decision
            should_continue = mor_result.should_continue and product_type_result.should_continue
            
            # Determine rejection source and reason
            rejection_reason = None
            rejection_source = None
            
            if not mor_result.should_continue:
                rejection_reason = mor_result.rejection_reason
                rejection_source = "mor_gate"
            elif not product_type_result.should_continue:
                rejection_reason = product_type_result.rejection_reason
                rejection_source = "product_type_gate"
            
            return MORProductTypeResult(
                mor_passed=mor_result.should_continue,
                product_type_passed=product_type_result.should_continue,
                mor_vendor=mor_result.vendor_name,
                mor_special_flag=mor_result.special_flag,
                product_type=product_type_result.product_type,
                product_type_inference=product_type_result.inference_method,
                should_continue=should_continue,
                rejection_reason=rejection_reason,
                rejection_source=rejection_source,
                mor_processing_ms=mor_result.processing_time_ms,
                product_type_processing_ms=product_type_result.processing_time_ms,
                total_processing_ms=total_time,
                gates_evaluated=["mor_gate", "product_type_gate"]
            )
    
    def get_config_info(self) -> Dict:
        """
        Get information about the current configuration state.
        
        Returns:
            Dictionary with configuration metadata and status
        """
        config_info = self.config_loader.get_config_info()
        validation_result = self.config_loader.get_validation_result()
        
        return {
            **config_info,
            "hot_reload_enabled": self._hot_reload_enabled,
            "config_check_interval_seconds": self._config_check_interval,
            "gates_enabled": self._gates_enabled,
            "pre_arrival_enabled": self._pre_arrival_enabled,
            "validation_status": {
                "is_valid": validation_result.is_valid if validation_result else None,
                "error_count": len(validation_result.errors) if validation_result else 0,
                "warning_count": len(validation_result.warnings) if validation_result else 0,
                "errors": [str(e) for e in validation_result.errors] if validation_result else [],
                "warnings": validation_result.warnings if validation_result else []
            }
        }
    
    def are_gates_enabled(self) -> bool:
        """
        Check if MOR/Product Type gates are enabled.
        
        Returns:
            True if gates are enabled, False if disabled
        """
        return self._gates_enabled
    
    def is_pre_arrival_enabled(self) -> bool:
        """
        Check if Pre-Arrival gate is enabled.
        
        Returns:
            True if pre-arrival gate is enabled, False if disabled
        """
        return self._pre_arrival_enabled
    
    def reload_config(self) -> bool:
        """
        Manually reload configuration from file.
        
        Returns:
            True if reload was successful, False otherwise
        """
        try:
            with self._config_lock:
                old_config = self.config
                self.config = self.config_loader.load_config(force_reload=True)
                self._gates_enabled = self.config.get("router_settings", {}).get("enable_mor_product_type_gates", True)
                self._pre_arrival_enabled = self.config.get("router_settings", {}).get("enable_pre_arrival_gate", True)
                self._initialize_gates()
                
                self.logger.info("Manual configuration reload completed successfully")
                self._log_config_changes(old_config, self.config)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Manual configuration reload failed: {e}")
            return False