"""
Data models for routing gates.

This module contains the data classes and models used by the routing gates
to represent decisions, configurations, and results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Set, Dict, List


@dataclass
class MORDecision:
    """Result of MOR (Merchant of Record) gate check."""
    should_continue: bool  # False = instant reject, True = continue
    vendor_name: Optional[str] = None  # For third-party rejections
    special_flag: Optional[str] = None  # "groupon_credit_only", etc.
    rejection_reason: Optional[str] = None
    processing_time_ms: int = 0


@dataclass
class MORGateConfig:
    """Configuration for MOR gate with vendor lists."""
    third_party_vendors: Set[str] = field(default_factory=lambda: {
        "axs", "seatgeek", "stubhub", "ticketmaster",
        "ticketnetwork", "gametime", "vividseats"
    })
    parkwhiz_partners: Set[str] = field(default_factory=lambda: {
        "parkwhiz", "google", "waze", "apple"
    })
    special_partners: Dict[str, str] = field(default_factory=lambda: {
        "groupon": "groupon_credit_only"
    })


@dataclass
class ProductTypeDecision:
    """Result of product type check."""
    should_continue: bool  # False = instant reject, True = continue
    product_type: str  # "event", "on-demand", "transient", "season_package"
    inference_method: str  # How product type was determined
    rejection_reason: Optional[str] = None
    processing_time_ms: int = 0


@dataclass
class ProductTypeGateConfig:
    """Configuration for Product Type gate with duration threshold."""
    on_demand_threshold_hours: int = 4
    season_package_detection: str = "future_implementation"  # Placeholder


@dataclass
class MORMetrics:
    """Metrics tracking for MOR gate decisions."""
    # Vendor rejection counts
    vendor_rejection_counts: Dict[str, int] = field(default_factory=dict)
    
    # Pass-through counts
    parkwhiz_pass_count: int = 0
    groupon_special_count: int = 0
    unknown_partner_count: int = 0
    
    # Performance metrics
    total_processed: int = 0
    total_processing_time_ms: int = 0
    rejection_count: int = 0
    
    # Cost savings calculation
    average_llm_cost_per_ticket: float = 0.12  # Default $0.12 per ticket
    
    def add_rejection(self, vendor_name: str, processing_time_ms: int) -> None:
        """Record a third-party vendor rejection."""
        self.vendor_rejection_counts[vendor_name] = self.vendor_rejection_counts.get(vendor_name, 0) + 1
        self.rejection_count += 1
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
    
    def add_pass_through(self, partner_type: str, processing_time_ms: int) -> None:
        """Record a pass-through decision."""
        if partner_type == "parkwhiz":
            self.parkwhiz_pass_count += 1
        elif partner_type == "groupon":
            self.groupon_special_count += 1
        elif partner_type == "unknown":
            self.unknown_partner_count += 1
        
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
    
    def get_cost_savings(self) -> float:
        """Calculate cost savings from rejected tickets."""
        return self.rejection_count * self.average_llm_cost_per_ticket
    
    def get_rejection_rate(self) -> float:
        """Calculate rejection rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.rejection_count / self.total_processed) * 100
    
    def get_average_processing_time(self) -> float:
        """Calculate average processing time in milliseconds."""
        if self.total_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_processed


@dataclass
class ProductTypeMetrics:
    """Metrics tracking for Product Type gate decisions."""
    # Product type distribution
    product_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # Rejection counts by type
    on_demand_rejections: int = 0
    season_package_rejections: int = 0
    
    # Pass-through counts
    event_pass_count: int = 0
    transient_pass_count: int = 0
    
    # Performance metrics
    total_processed: int = 0
    total_processing_time_ms: int = 0
    rejection_count: int = 0
    
    # Duration calculation metrics
    duration_calculation_failures: int = 0
    
    # Cost savings calculation
    average_llm_cost_per_ticket: float = 0.12  # Default $0.12 per ticket
    
    def add_rejection(self, product_type: str, processing_time_ms: int) -> None:
        """Record a product type rejection."""
        self.product_type_counts[product_type] = self.product_type_counts.get(product_type, 0) + 1
        
        if product_type == "on-demand":
            self.on_demand_rejections += 1
        elif product_type == "season_package":
            self.season_package_rejections += 1
        
        self.rejection_count += 1
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
    
    def add_pass_through(self, product_type: str, processing_time_ms: int) -> None:
        """Record a pass-through decision."""
        self.product_type_counts[product_type] = self.product_type_counts.get(product_type, 0) + 1
        
        if product_type == "event":
            self.event_pass_count += 1
        elif product_type == "transient":
            self.transient_pass_count += 1
        
        self.total_processed += 1
        self.total_processing_time_ms += processing_time_ms
    
    def add_duration_failure(self) -> None:
        """Record a duration calculation failure."""
        self.duration_calculation_failures += 1
    
    def get_cost_savings(self) -> float:
        """Calculate cost savings from rejected tickets."""
        return self.rejection_count * self.average_llm_cost_per_ticket
    
    def get_rejection_rate(self) -> float:
        """Calculate rejection rate as percentage."""
        if self.total_processed == 0:
            return 0.0
        return (self.rejection_count / self.total_processed) * 100
    
    def get_average_processing_time(self) -> float:
        """Calculate average processing time in milliseconds."""
        if self.total_processed == 0:
            return 0.0
        return self.total_processing_time_ms / self.total_processed


@dataclass
class MORProductTypeResult:
    """Combined result from MOR and Product Type gates."""
    
    # MOR gate results (required fields first)
    mor_passed: bool
    product_type_passed: bool
    
    # Optional fields with defaults
    mor_vendor: Optional[str] = None
    mor_special_flag: Optional[str] = None
    product_type: Optional[str] = None
    product_type_inference: Optional[str] = None
    should_continue: bool = False  # True only if BOTH gates pass
    rejection_reason: Optional[str] = None
    rejection_source: Optional[str] = None  # "mor_gate" or "product_type_gate"
    mor_processing_ms: int = 0
    product_type_processing_ms: int = 0
    total_processing_ms: int = 0
    gates_evaluated: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Pre-Arrival Enhancement Data Models

@dataclass
class PreArrivalDecision:
    """Final decision from pre-arrival gate."""
    should_continue: bool  # False = instant approval, True = continue to next gate
    decision: Optional[str] = None  # "Approved" if pre-arrival, None otherwise
    reasoning: Optional[str] = None
    hours_before_booking: Optional[float] = None
    timezone_used: Optional[str] = None
    processing_time_ms: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreArrivalResult:
    """Result of pre-arrival time comparison."""
    is_pre_arrival: bool
    hours_before_booking: Optional[float] = None
    current_time: Optional[datetime] = None
    booking_start_time: Optional[datetime] = None
    timezone_used: Optional[str] = None
    processing_time_ms: int = 0


@dataclass
class PreArrivalMetrics:
    """Metrics tracking for pre-arrival gate performance with context-aware tracking."""
    
    # Processing counts (overall)
    total_processed: int = 0
    pre_arrival_approved: int = 0
    not_pre_arrival: int = 0
    parsing_failures: int = 0
    timezone_failures: int = 0
    
    # Context-specific processing counts (webhook vs chat)
    webhook_processed: int = 0
    webhook_approved: int = 0
    webhook_not_pre_arrival: int = 0
    webhook_parsing_failures: int = 0
    webhook_timezone_failures: int = 0
    
    chat_processed: int = 0
    chat_approved: int = 0
    chat_not_pre_arrival: int = 0
    chat_parsing_failures: int = 0
    chat_timezone_failures: int = 0
    
    # Performance metrics (overall)
    avg_processing_ms: float = 0.0
    p95_processing_ms: float = 0.0
    max_processing_ms: float = 0.0
    
    # Context-specific performance metrics
    webhook_avg_processing_ms: float = 0.0
    webhook_p95_processing_ms: float = 0.0
    webhook_max_processing_ms: float = 0.0
    webhook_processing_times: List[float] = field(default_factory=list)
    
    chat_avg_processing_ms: float = 0.0
    chat_p95_processing_ms: float = 0.0
    chat_max_processing_ms: float = 0.0
    chat_processing_times: List[float] = field(default_factory=list)
    
    # Timezone inference metrics
    timezone_inference_used: int = 0
    utc_fallback_used: int = 0
    timezone_inference_success_rate: float = 0.0
    
    # Date parsing metrics
    primary_format_success: int = 0
    fallback_format_success: int = 0
    parsing_success_rate: float = 0.0
    
    # Business metrics (overall)
    hours_before_booking_avg: float = 0.0
    cost_savings_estimate: float = 0.0  # LLM calls avoided × cost per call
    
    # Context-specific business metrics
    webhook_cost_savings: float = 0.0
    chat_cost_savings: float = 0.0
    
    # Webhook-specific metrics
    webhook_timeout_count: int = 0  # Number of webhook timeouts
    webhook_timeout_rate: float = 0.0  # Percentage of webhooks that timed out
    
    def calculate_derived_metrics(self):
        """Calculate derived metrics from raw counts."""
        if self.total_processed > 0:
            self.parsing_success_rate = (self.total_processed - self.parsing_failures) / self.total_processed
            self.timezone_inference_success_rate = (
                self.timezone_inference_used / (self.timezone_inference_used + self.utc_fallback_used)
                if (self.timezone_inference_used + self.utc_fallback_used) > 0 else 0.0
            )
        
        # Calculate webhook timeout rate
        if self.webhook_processed > 0:
            self.webhook_timeout_rate = (self.webhook_timeout_count / self.webhook_processed) * 100
    
    def update_context_performance_metrics(self, context: Optional[str]):
        """Update performance metrics for specific context."""
        if context == "webhook" and self.webhook_processing_times:
            self.webhook_avg_processing_ms = sum(self.webhook_processing_times) / len(self.webhook_processing_times)
            sorted_times = sorted(self.webhook_processing_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.webhook_p95_processing_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            self.webhook_max_processing_ms = max(self.webhook_processing_times)
        
        elif context == "chat" and self.chat_processing_times:
            self.chat_avg_processing_ms = sum(self.chat_processing_times) / len(self.chat_processing_times)
            sorted_times = sorted(self.chat_processing_times)
            p95_index = int(len(sorted_times) * 0.95)
            self.chat_p95_processing_ms = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
            self.chat_max_processing_ms = max(self.chat_processing_times)



@dataclass
class PreArrivalAuditLog:
    """Comprehensive audit log entry for pre-arrival decisions."""
    
    # Request identification
    ticket_id: str
    timestamp: datetime
    
    # Input data
    raw_start_time: Optional[str] = None
    raw_timezone: Optional[str] = None
    raw_location: Optional[str] = None
    
    # Processing details
    parsed_start_time: Optional[datetime] = None
    timezone_used: Optional[str] = None
    timezone_source: Optional[str] = None  # "direct", "inferred", "utc_fallback"
    current_time: Optional[datetime] = None
    
    # Decision details
    is_pre_arrival: bool = False
    hours_before_booking: Optional[float] = None
    decision: Optional[str] = None
    reasoning: Optional[str] = None
    
    # Performance metrics
    parsing_time_ms: int = 0
    timezone_time_ms: int = 0
    comparison_time_ms: int = 0
    total_time_ms: int = 0
    
    # Quality indicators
    warnings: List[str] = field(default_factory=list)
    data_quality_issues: List[str] = field(default_factory=list)
    suspicious_patterns: List[str] = field(default_factory=list)


@dataclass
class PreArrivalGateConfig:
    """Configuration for pre-arrival gate with date formats and timezone mappings."""
    
    # Date format configuration with priority ordering
    date_formats: List[str] = field(default_factory=lambda: [
        "%A %b %d, %Y, %I:%M %p",  # "Wednesday Dec 17, 2025, 07:45 PM" (Zapier primary)
        "%Y-%m-%dT%H:%M:%S",       # "2025-12-17T19:45:00" (ISO 8601)
        "%m/%d/%Y %I:%M %p",       # "12/17/2025 7:45 PM" (US format)
        "%b %d, %Y %I:%M:%S %p",   # "Dec 17, 2025 7:45:00 PM" (alternative)
        "%Y-%m-%d %H:%M:%S"        # "2025-12-17 19:45:00" (24-hour format)
    ])
    
    # City-to-timezone mapping for inference fallback
    city_to_timezone_mapping: Dict[str, str] = field(default_factory=lambda: {
        "new york, ny": "America/New_York",
        "new york": "America/New_York",
        "manhattan, ny": "America/New_York",
        "brooklyn, ny": "America/New_York",
        "los angeles, ca": "America/Los_Angeles",
        "los angeles": "America/Los_Angeles",
        "san francisco, ca": "America/Los_Angeles",
        "chicago, il": "America/Chicago",
        "chicago": "America/Chicago",
        "dallas, tx": "America/Chicago",
        "houston, tx": "America/Chicago",
        "denver, co": "America/Denver",
        "phoenix, az": "America/Phoenix",
        "seattle, wa": "America/Los_Angeles",
        "miami, fl": "America/New_York",
        "atlanta, ga": "America/New_York",
        "boston, ma": "America/New_York",
        "washington, dc": "America/New_York",
        "philadelphia, pa": "America/New_York"
    })
    
    # Boundary condition configuration
    max_future_months: int = 12  # Flag bookings >12 months in future
    max_past_days: int = 7       # Flag bookings >7 days in past
    
    # Performance targets
    target_processing_ms: int = 200  # Target total processing time
    optimized_path_ms: int = 100     # Target for common cases