"""
LogicValidator component for checking consistency between complaints and responses.

This module provides the LogicValidator class that validates logical consistency
between complaint types and customer responses to detect contradictions.
"""

import logging
from typing import Dict, List, Optional
from .complaint_parser import ComplaintType
from .contradiction_config import ContradictionConfigManager
from dataclasses import dataclass


@dataclass
class Contradiction:
    """Represents a detected contradiction between complaint types or responses."""
    type: str  # "logical_conflict", "mutually_exclusive", "response_mismatch"
    complaint_a: ComplaintType
    complaint_b: Optional[ComplaintType] = None  # None for response mismatches
    customer_response: Optional[str] = None  # For response contradiction cases
    severity: str = "medium"  # "high", "medium", "low"
    description: str = ""  # Human-readable description of the conflict

# Configure logger
logger = logging.getLogger(__name__)


class LogicValidator:
    """
    Validates logical consistency between complaint types and customer responses.
    
    This class checks for contradictions between different complaint types and
    between complaints and customer responses to parking-related questions.
    Uses configurable contradiction rules for flexible rule management.
    """
    
    def __init__(self, contradiction_rules: Dict):
        """
        Initialize the LogicValidator with contradiction rules.
        
        Args:
            contradiction_rules: Dictionary containing contradiction detection rules
                                Can include 'config_path' key for external configuration
        """
        self.rules = contradiction_rules
        self.config_manager = None
        self.config = None
        self.initialization_successful = False
        
        # Initialize configuration manager with comprehensive error handling
        try:
            config_path = contradiction_rules.get('config_path')
            self.config_manager = ContradictionConfigManager(config_path)
            logger.info("ContradictionConfigManager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ContradictionConfigManager: {e}")
            self.config_manager = None
        
        # Load configuration with fallback handling
        if self.config_manager:
            try:
                self.config = self.config_manager.load_config()
                logger.info(f"Loaded {len(self.config.rules)} contradiction rules from configuration")
                
                # Extract rules for backward compatibility and performance
                self.mutually_exclusive_pairs = self.config_manager.get_mutually_exclusive_pairs()
                self.response_conflicts = self.config_manager.get_response_conflicts()
                self.initialization_successful = True
                
            except Exception as e:
                logger.error(f"Failed to load contradiction configuration: {e}")
                logger.info("Configuration manager available but config loading failed - using hardcoded fallback")
                self._setup_hardcoded_fallback()
        else:
            logger.warning("Configuration manager not available - using hardcoded fallback rules")
            self._setup_hardcoded_fallback()
        
        # Final validation of setup
        if not hasattr(self, 'mutually_exclusive_pairs') or not hasattr(self, 'response_conflicts'):
            logger.error("Critical error: Neither configuration nor fallback rules were set up properly")
            self._setup_emergency_fallback()
        
        logger.info(f"LogicValidator initialized (successful: {self.initialization_successful})")
    
    def _setup_hardcoded_fallback(self):
        """Set up hardcoded fallback rules when configuration fails."""
        try:
            self.mutually_exclusive_pairs = [
                ("accessibility", "oversold"),  # Can't access vs facility full
                ("paid_again", "oversold"),     # Paid twice vs no space available
                ("vehicle_restriction", "oversold"),  # Vehicle rejected vs facility full
            ]
            
            self.response_conflicts = {
                "accessibility": {
                    "conflicting_responses": ["were you able to park? yes", "were you able to park?yes"],
                    "description": "Customer claims accessibility issues but confirms they were able to park",
                    "severity": "high"
                },
                "oversold": {
                    "conflicting_responses": ["were you able to park? yes", "were you able to park?yes"],
                    "description": "Customer claims facility was full but confirms they were able to park",
                    "severity": "high"
                },
                "location_finding": {
                    "conflicting_responses": ["were you able to park? yes", "were you able to park?yes"],
                    "description": "Customer claims they couldn't find location but confirms they were able to park",
                    "severity": "medium"
                }
            }
            
            logger.info("Hardcoded fallback rules set up successfully")
        except Exception as e:
            logger.error(f"Failed to set up hardcoded fallback rules: {e}")
            self._setup_emergency_fallback()
    
    def _setup_emergency_fallback(self):
        """Set up minimal emergency fallback when all else fails."""
        try:
            # Absolute minimal rules to prevent complete failure
            self.mutually_exclusive_pairs = [
                ("accessibility", "oversold")  # Only the most critical rule
            ]
            
            self.response_conflicts = {
                "accessibility": {
                    "conflicting_responses": ["were you able to park? yes"],
                    "description": "Customer claims accessibility issues but confirms they were able to park",
                    "severity": "high"
                }
            }
            
            logger.warning("Emergency fallback rules set up - functionality will be limited")
        except Exception as e:
            logger.error(f"Even emergency fallback failed: {e}")
            # Set empty rules to prevent crashes
            self.mutually_exclusive_pairs = []
            self.response_conflicts = {}
    
    def validate_consistency(
        self, 
        complaints: List[ComplaintType], 
        ticket_data: Dict
    ) -> List[Contradiction]:
        """
        Validate logical consistency between complaints and responses.
        
        This method checks for contradictions between complaint types and
        between complaints and customer responses to identify conflicts
        that require human review. Includes comprehensive error handling
        and graceful degradation.
        
        Args:
            complaints: List of parsed complaint types
            ticket_data: Dictionary containing ticket information and responses
        
        Returns:
            List of detected contradictions with details
        """
        logger.info(f"Validating consistency for {len(complaints)} complaints")
        
        contradictions = []
        validation_errors = []
        
        # Validate inputs
        if complaints is None:
            logger.warning("Complaints list is None - using empty list")
            complaints = []
        
        if ticket_data is None:
            logger.warning("Ticket data is None - using empty dict")
            ticket_data = {}
        
        # Check for mutually exclusive complaint combinations with error handling
        try:
            exclusive_contradictions = self._check_mutually_exclusive_complaints(complaints)
            contradictions.extend(exclusive_contradictions)
            logger.debug(f"Found {len(exclusive_contradictions)} mutually exclusive contradictions")
        except Exception as e:
            logger.error(f"Error checking mutually exclusive complaints: {e}")
            validation_errors.append(f"mutually_exclusive_check: {e}")
            # Continue with other checks
        
        # Check for response contradictions with error handling
        try:
            response_contradictions = self._check_response_contradictions(complaints, ticket_data)
            contradictions.extend(response_contradictions)
            logger.debug(f"Found {len(response_contradictions)} response contradictions")
        except Exception as e:
            logger.error(f"Error checking response contradictions: {e}")
            validation_errors.append(f"response_contradiction_check: {e}")
            # Continue with other checks
        
        # Check for complex logical conflicts with error handling
        try:
            logical_contradictions = self._check_complex_logical_conflicts(complaints)
            contradictions.extend(logical_contradictions)
            logger.debug(f"Found {len(logical_contradictions)} logical contradictions")
        except Exception as e:
            logger.error(f"Error checking complex logical conflicts: {e}")
            validation_errors.append(f"logical_conflict_check: {e}")
            # Continue processing
        
        # Log validation status
        if validation_errors:
            logger.warning(f"Validation completed with {len(validation_errors)} errors: {validation_errors}")
        
        logger.info(f"Found {len(contradictions)} total contradictions")
        return contradictions
    
    def _check_mutually_exclusive_complaints(self, complaints: List[ComplaintType]) -> List[Contradiction]:
        """
        Check for mutually exclusive complaint type combinations.
        
        Args:
            complaints: List of parsed complaint types
        
        Returns:
            List of contradictions for mutually exclusive complaints
        """
        contradictions = []
        
        try:
            # Validate inputs
            if not complaints:
                logger.debug("No complaints to check for mutual exclusivity")
                return contradictions
            
            if not hasattr(self, 'mutually_exclusive_pairs') or not self.mutually_exclusive_pairs:
                logger.debug("No mutually exclusive pairs configured")
                return contradictions
            
            # Build complaint categories mapping with error handling
            complaint_categories = {}
            for complaint in complaints:
                try:
                    if hasattr(complaint, 'category') and complaint.category:
                        complaint_categories[complaint.category] = complaint
                except Exception as e:
                    logger.warning(f"Error processing complaint for mutual exclusivity check: {e}")
                    continue
            
            # Check each mutually exclusive pair
            for pair in self.mutually_exclusive_pairs:
                try:
                    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                        logger.warning(f"Invalid mutually exclusive pair format: {pair}")
                        continue
                    
                    category_a, category_b = pair
                    
                    if category_a in complaint_categories and category_b in complaint_categories:
                        complaint_a = complaint_categories[category_a]
                        complaint_b = complaint_categories[category_b]
                        
                        # Determine severity based on confidence levels with error handling
                        try:
                            confidence_a = getattr(complaint_a, 'confidence', 0.5)
                            confidence_b = getattr(complaint_b, 'confidence', 0.5)
                            avg_confidence = (confidence_a + confidence_b) / 2
                            severity = "high" if avg_confidence > 0.8 else "medium"
                        except Exception as e:
                            logger.warning(f"Error calculating confidence for severity: {e}")
                            severity = "medium"  # Default severity
                        
                        contradiction = Contradiction(
                            type="mutually_exclusive",
                            complaint_a=complaint_a,
                            complaint_b=complaint_b,
                            customer_response=None,
                            severity=severity,
                            description=f"Customer claims both {category_a} and {category_b} issues, which are mutually exclusive"
                        )
                        
                        contradictions.append(contradiction)
                        logger.info(f"Found mutually exclusive complaints: {category_a} vs {category_b}")
                
                except Exception as e:
                    logger.error(f"Error checking mutually exclusive pair {pair}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Critical error in mutually exclusive complaints check: {e}")
            # Return what we have so far
        
        return contradictions
    
    def _check_response_contradictions(
        self, 
        complaints: List[ComplaintType], 
        ticket_data: Dict
    ) -> List[Contradiction]:
        """
        Check for contradictions between complaints and customer responses.
        
        Args:
            complaints: List of parsed complaint types
            ticket_data: Dictionary containing ticket information and responses
        
        Returns:
            List of contradictions between complaints and responses
        """
        contradictions = []
        
        try:
            # Validate inputs
            if not complaints:
                logger.debug("No complaints to check for response contradictions")
                return contradictions
            
            if not hasattr(self, 'response_conflicts') or not self.response_conflicts:
                logger.debug("No response conflicts configured")
                return contradictions
            
            # Extract customer responses from ticket data with error handling
            try:
                ticket_description = ticket_data.get("description", "") if ticket_data else ""
                ticket_notes = ticket_data.get("notes", "") if ticket_data else ""
                
                # Safely convert to lowercase
                ticket_description = str(ticket_description).lower() if ticket_description else ""
                ticket_notes = str(ticket_notes).lower() if ticket_notes else ""
                
                full_text = f"{ticket_description} {ticket_notes}"
            except Exception as e:
                logger.warning(f"Error extracting ticket text for response contradiction check: {e}")
                full_text = ""
            
            # Check each complaint for response contradictions
            for complaint in complaints:
                try:
                    if not hasattr(complaint, 'category') or not complaint.category:
                        logger.warning("Complaint missing category - skipping response contradiction check")
                        continue
                    
                    category = complaint.category
                    
                    if category in self.response_conflicts:
                        try:
                            conflict_info = self.response_conflicts[category]
                            
                            # Validate conflict info structure
                            if not isinstance(conflict_info, dict):
                                logger.warning(f"Invalid conflict info for category {category}: {conflict_info}")
                                continue
                            
                            conflicting_responses = conflict_info.get("conflicting_responses", [])
                            description = conflict_info.get("description", f"Response contradiction for {category}")
                            severity = conflict_info.get("severity", "high")
                            
                            if not conflicting_responses:
                                logger.debug(f"No conflicting responses configured for category {category}")
                                continue
                            
                            # Check if any conflicting responses are present
                            for response in conflicting_responses:
                                try:
                                    if response and str(response).lower() in full_text:
                                        contradiction = Contradiction(
                                            type="response_mismatch",
                                            complaint_a=complaint,
                                            complaint_b=None,
                                            customer_response=response,
                                            severity=severity,
                                            description=description
                                        )
                                        
                                        contradictions.append(contradiction)
                                        logger.info(f"Found response contradiction for {category}: {response}")
                                        break  # Only report one response contradiction per complaint type
                                
                                except Exception as e:
                                    logger.warning(f"Error checking response '{response}' for category {category}: {e}")
                                    continue
                        
                        except Exception as e:
                            logger.error(f"Error processing conflict info for category {category}: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error checking complaint for response contradictions: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Critical error in response contradictions check: {e}")
            # Return what we have so far
        
        return contradictions
    
    def _check_complex_logical_conflicts(self, complaints: List[ComplaintType]) -> List[Contradiction]:
        """
        Check for complex logical conflicts between multiple complaint types.
        
        This method identifies more nuanced contradictions that involve
        multiple complaint types or complex logical relationships.
        Uses configurable rules when available, falls back to hardcoded rules.
        
        Args:
            complaints: List of parsed complaint types
        
        Returns:
            List of complex logical contradictions
        """
        contradictions = []
        complaint_categories = {c.category: c for c in complaints}
        
        # Use configurable logical conflict rules if available
        if self.config_manager and self.config:
            logical_conflict_rules = self.config_manager.get_rules_by_type("logical_conflict")
            
            for rule in logical_conflict_rules:
                complaint_a_cat = rule.complaint_a
                complaint_b_cat = rule.complaint_b
                
                # Check if both complaint types are present
                if (complaint_a_cat in complaint_categories and 
                    complaint_b_cat and complaint_b_cat in complaint_categories):
                    
                    complaint_a = complaint_categories[complaint_a_cat]
                    complaint_b = complaint_categories[complaint_b_cat]
                    
                    contradiction = Contradiction(
                        type="logical_conflict",
                        complaint_a=complaint_a,
                        complaint_b=complaint_b,
                        customer_response=None,
                        severity=rule.severity,
                        description=rule.description
                    )
                    
                    contradictions.append(contradiction)
                    logger.info(f"Found logical conflict from rule {rule.id}: {complaint_a_cat} + {complaint_b_cat}")
        
        else:
            # Fallback to hardcoded logical conflict rules
            # Check for "road closure + double billed" contradiction (ticket 1273112 scenario)
            if "accessibility" in complaint_categories and "paid_again" in complaint_categories:
                accessibility_complaint = complaint_categories["accessibility"]
                paid_again_complaint = complaint_categories["paid_again"]
                
                # This is a logical conflict: if road was closed, how did customer get charged twice?
                contradiction = Contradiction(
                    type="logical_conflict",
                    complaint_a=accessibility_complaint,
                    complaint_b=paid_again_complaint,
                    customer_response=None,
                    severity="high",
                    description="Customer claims road closure prevented access but also reports being charged multiple times"
                )
                
                contradictions.append(contradiction)
                logger.info("Found road closure + double billing contradiction")
            
            # Check for vehicle restriction + paid again contradiction
            if "vehicle_restriction" in complaint_categories and "paid_again" in complaint_categories:
                vehicle_complaint = complaint_categories["vehicle_restriction"]
                paid_again_complaint = complaint_categories["paid_again"]
                
                contradiction = Contradiction(
                    type="logical_conflict",
                    complaint_a=vehicle_complaint,
                    complaint_b=paid_again_complaint,
                    customer_response=None,
                    severity="medium",
                    description="Customer claims vehicle was rejected but also reports being charged multiple times"
                )
                
                contradictions.append(contradiction)
                logger.info("Found vehicle restriction + paid again contradiction")
        
        return contradictions
    
    def reload_configuration(self) -> bool:
        """
        Reload contradiction rules configuration from file.
        
        Returns:
            True if configuration was reloaded successfully, False otherwise
        """
        if not self.config_manager:
            logger.warning("No configuration manager available for reload")
            return False
        
        try:
            self.config = self.config_manager.load_config()
            
            # Update cached rules for performance
            self.mutually_exclusive_pairs = self.config_manager.get_mutually_exclusive_pairs()
            self.response_conflicts = self.config_manager.get_response_conflicts()
            
            logger.info(f"Reloaded {len(self.config.rules)} contradiction rules")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            return False
    
    def add_contradiction_rule(self, rule_id: str, rule_name: str, rule_type: str, 
                             complaint_a: str, complaint_b: Optional[str] = None,
                             conflicting_responses: Optional[List[str]] = None,
                             severity: str = "medium", description: str = "",
                             priority: int = 100) -> bool:
        """
        Add a new contradiction rule to the configuration.
        
        Args:
            rule_id: Unique identifier for the rule
            rule_name: Human-readable name for the rule
            rule_type: Type of rule ("mutually_exclusive", "response_mismatch", "logical_conflict")
            complaint_a: First complaint type
            complaint_b: Second complaint type (for mutually_exclusive and logical_conflict)
            conflicting_responses: List of conflicting responses (for response_mismatch)
            severity: Severity level ("high", "medium", "low")
            description: Human-readable description of the conflict
            priority: Priority for rule application (lower = higher priority)
        
        Returns:
            True if rule was added successfully, False otherwise
        """
        if not self.config_manager:
            logger.warning("No configuration manager available for adding rules")
            return False
        
        try:
            from .contradiction_config import ContradictionRule
            
            rule = ContradictionRule(
                id=rule_id,
                name=rule_name,
                type=rule_type,
                complaint_a=complaint_a,
                complaint_b=complaint_b,
                conflicting_responses=conflicting_responses,
                severity=severity,
                description=description,
                priority=priority
            )
            
            self.config_manager.add_rule(rule)
            
            # Update cached rules
            self.mutually_exclusive_pairs = self.config_manager.get_mutually_exclusive_pairs()
            self.response_conflicts = self.config_manager.get_response_conflicts()
            
            logger.info(f"Added new contradiction rule: {rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add contradiction rule {rule_id}: {e}")
            return False
    
    def remove_contradiction_rule(self, rule_id: str) -> bool:
        """
        Remove a contradiction rule from the configuration.
        
        Args:
            rule_id: ID of rule to remove
        
        Returns:
            True if rule was removed successfully, False otherwise
        """
        if not self.config_manager:
            logger.warning("No configuration manager available for removing rules")
            return False
        
        try:
            success = self.config_manager.remove_rule(rule_id)
            
            if success:
                # Update cached rules
                self.mutually_exclusive_pairs = self.config_manager.get_mutually_exclusive_pairs()
                self.response_conflicts = self.config_manager.get_response_conflicts()
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove contradiction rule {rule_id}: {e}")
            return False
    
    def get_configuration_info(self) -> Dict:
        """
        Get information about the current configuration.
        
        Returns:
            Dictionary with configuration details
        """
        if not self.config:
            return {
                "status": "no_configuration",
                "rules_count": 0,
                "config_source": "hardcoded_fallback"
            }
        
        rules_by_type = {}
        for rule in self.config.rules:
            if rule.type not in rules_by_type:
                rules_by_type[rule.type] = 0
            if rule.enabled:
                rules_by_type[rule.type] += 1
        
        return {
            "status": "loaded",
            "version": self.config.version,
            "total_rules": len(self.config.rules),
            "enabled_rules": len([r for r in self.config.rules if r.enabled]),
            "rules_by_type": rules_by_type,
            "config_source": "external_file" if self.config_manager.config_path else "default",
            "config_path": self.config_manager.config_path,
            "global_settings": self.config.global_settings
        }