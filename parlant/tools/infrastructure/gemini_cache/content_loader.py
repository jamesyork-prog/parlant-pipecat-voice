"""
Content Loader for Gemini Context Caching.

This module loads and structures static content for prompt templates:
- Refund policy documents
- Tool definitions
- Few-shot examples
- System identity

Content is structured with static content first to maximize implicit caching.

Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.5, 4.1-4.5
"""

import hashlib
import logging
import os
import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List, Set
import json

from .models import (
    StaticContent,
    CacheConfiguration,
    ContentType,
    LoadingStatus,
    ContentLoadingStatus,
    ContentValidationResult,
    ContentPriority,
    ValidationStatus,
)
from .audit_logger import get_cache_audit_logger, AuditContext

logger = logging.getLogger(__name__)


# Default system identity template
DEFAULT_SYSTEM_IDENTITY = """You are Whiz AI Agent, an intelligent refund automation system for ParkWhiz customer support.

ROLE & CAPABILITIES:
- Analyze parking pass refund requests against ParkWhiz policies
- Make decisions: Approved, Denied, or Escalate to Human
- Provide clear reasoning for all decisions
- Handle complex scenarios with nuanced policy interpretation

COMMUNICATION STYLE:
- Professional and empathetic tone
- Clear, concise explanations
- Customer-friendly language in rejection messages
- Technical precision in reasoning for internal use

CONSTRAINTS:
- Follow ParkWhiz refund policies strictly
- Escalate ambiguous cases rather than guess
- Never approve refunds outside policy guidelines
- Always provide reasoning for decisions"""


# Default few-shot examples
DEFAULT_FEW_SHOT_EXAMPLES = """## Example 1: Pre-arrival Cancellation (Approved)
Ticket: Customer requests refund for parking pass, event is tomorrow
Decision: APPROVED
Reasoning: Pre-arrival cancellation within policy window

## Example 2: Post-event No-show (Denied)
Ticket: Customer didn't use pass, event was 3 days ago
Decision: DENIED
Reasoning: Post-event requests are not eligible for refund per policy

## Example 3: Oversold Location (Escalate)
Ticket: Customer couldn't park, location was full
Decision: ESCALATE
Reasoning: Oversold situations require human review for compensation"""


class ContentLoader:
    """
    Loads static content for prompt template construction.
    
    Responsibilities:
    - Load and validate static content files
    - Structure content for optimal implicit caching
    - Monitor content changes for template updates
    - Provide content for prompt construction
    - Handle fallback mode when content is unavailable
    
    Performance: <3s startup, <10ms per template construction
    """
    
    def __init__(self, config: CacheConfiguration):
        """
        Initialize content loader with configuration.
        
        Args:
            config: Cache configuration with file paths
        """
        self.config = config
        self.static_content: Optional[StaticContent] = None
        self._prompt_template: Optional[str] = None
        self._last_content_hash: Optional[str] = None
        self._file_modification_times: Dict[str, float] = {}
        self._tool_schema_hash: Optional[str] = None
        self._change_detection_enabled = True
        
        # Fallback mode state
        self._fallback_mode = False
        self._fallback_reason: Optional[str] = None
        self._fallback_activated_at: Optional[datetime] = None
        self._restoration_attempts = 0
        self._last_restoration_attempt: Optional[datetime] = None
        self._restoration_interval_seconds = 60  # Try restoration every 60 seconds
    
    async def initialize_content(self) -> bool:
        """
        Initialize static content at system startup.
        
        Loads all static content files and structures them for prompt templates.
        If content loading fails, activates fallback mode with hardcoded content.
        
        Returns:
            True if content loaded successfully, False if fallback needed
            
        Performance: <3 seconds total
        """
        start_time = time.time()
        
        with AuditContext("content_loading", self.config) as audit:
            logger.info("Loading static content for implicit caching...")
            
            try:
                loading_statuses: Dict[ContentType, ContentLoadingStatus] = {}
                
                # Load individual content sections
                system_identity, identity_status = await self._load_system_identity()
                loading_statuses[ContentType.SYSTEM_IDENTITY] = identity_status
                
                refund_policies, policies_status = await self._load_refund_policies()
                loading_statuses[ContentType.REFUND_POLICIES] = policies_status
                
                tool_definitions, tools_status = await self._load_tool_definitions()
                loading_statuses[ContentType.TOOL_DEFINITIONS] = tools_status
                
                few_shot_examples, examples_status = await self._load_few_shot_examples()
                loading_statuses[ContentType.FEW_SHOT_EXAMPLES] = examples_status
                
                # Combine all content
                combined_content = (
                    system_identity + refund_policies + tool_definitions + few_shot_examples
                )
                
                # Calculate metrics
                total_tokens = self._estimate_tokens(combined_content)
                content_hash = self._calculate_content_hash(combined_content)
                
                self.static_content = StaticContent(
                    system_identity=system_identity,
                    refund_policies=refund_policies,
                    tool_definitions=tool_definitions,
                    few_shot_examples=few_shot_examples,
                    total_tokens=total_tokens,
                    content_hash=content_hash,
                    loaded_at=datetime.now(),
                    loading_statuses=loading_statuses,
                )
                
                # Validate and process content (including size management)
                self.static_content = self.validate_and_process_content(self.static_content)
                
                # Build prompt template
                self._prompt_template = self._build_prompt_template()
                self._last_content_hash = self.static_content.content_hash
                
                # Initialize file modification tracking
                await self._update_file_modification_times()
                
                # Check if we should deactivate fallback mode (content restored)
                if self._fallback_mode and self.static_content.is_complete():
                    self._deactivate_fallback_mode()
                
                # Calculate loading duration
                loading_duration_ms = (time.time() - start_time) * 1000
                
                # Log results
                success = self.static_content.is_complete()
                if success:
                    logger.info(f"Static content loaded successfully. Total tokens: {total_tokens}")
                    
                    # Audit log successful loading
                    audit.log_content_loading_success(
                        self.static_content,
                        loading_duration_ms,
                        {"initialization_type": "startup"}
                    )
                else:
                    failed = self.static_content.get_failed_content_types()
                    has_actual_fallback = any(
                        status.status == LoadingStatus.FALLBACK 
                        for status in self.static_content.loading_statuses.values()
                    )
                    logger.warning(
                        f"Static content loaded with fallbacks. Failed: {[f.value for f in failed]}. "
                        f"Total tokens: {total_tokens}"
                    )
                    
                    # Audit log partial loading with fallbacks
                    audit.log_content_loading_success(
                        self.static_content,
                        loading_duration_ms,
                        {
                            "initialization_type": "startup_with_fallbacks",
                            "failed_content_types": [f.value for f in failed],
                            "has_actual_fallback": has_actual_fallback
                        }
                    )
                    
                    # Activate fallback mode only if there are actual failures or fallbacks due to errors
                    if failed or has_actual_fallback:
                        reason_parts = []
                        if failed:
                            reason_parts.append(f"Failed types: {[f.value for f in failed]}")
                        if has_actual_fallback:
                            reason_parts.append("Some content using error fallback")
                        reason = "Content loading incomplete. " + ", ".join(reason_parts)
                        self._activate_fallback_mode(reason)
                
                return success
                
            except Exception as e:
                loading_duration_ms = (time.time() - start_time) * 1000
                logger.error(f"Critical error during content initialization: {e}")
                
                # Audit log the failure
                audit.log_content_loading_failure(
                    str(e),
                    "Critical error during content initialization",
                    ["activate_fallback_mode", "use_hardcoded_content"],
                    loading_duration_ms,
                    {"initialization_type": "startup_failed"}
                )
                
                # Activate fallback mode with hardcoded content
                self._activate_fallback_mode(f"Content initialization failed: {str(e)}")
                self.static_content = self._create_fallback_content()
                self._prompt_template = self._build_prompt_template()
                self._last_content_hash = self.static_content.content_hash
                
                logger.info(f"Fallback mode activated. Using hardcoded content. Total tokens: {self.static_content.total_tokens}")
                return False
    
    def get_static_content(self) -> Optional[StaticContent]:
        """Get current static content for prompt construction."""
        return self.static_content
    
    def get_prompt_template(self) -> str:
        """
        Get structured prompt template with static content first.
        
        Automatically attempts content restoration if in fallback mode
        and restoration is due.
        
        Returns:
            Formatted prompt template optimized for implicit caching
        """
        # Check if we should attempt restoration
        if self._fallback_mode and self._should_attempt_restoration():
            # Attempt restoration asynchronously (fire and forget)
            # This ensures prompt construction doesn't block
            asyncio.create_task(self.attempt_content_restoration())
        
        if self._prompt_template:
            return self._prompt_template
        
        return self._get_fallback_template()
    
    def get_content_hash(self) -> Optional[str]:
        """Get hash of current content for change detection."""
        return self._last_content_hash
    
    def check_content_changed(self) -> bool:
        """
        Check if static content files have changed.
        
        Returns:
            True if content has changed since last load
        """
        if not self.static_content or not self._change_detection_enabled:
            return True
        
        try:
            changed_files = []
            change_type = "none"
            
            # Check policy document modification times
            for file_path in self.config.static_content_paths.values():
                if self._check_file_modified(file_path):
                    logger.info(f"Content change detected in policy file: {file_path}")
                    changed_files.append(file_path)
                    change_type = "policy_files"
            
            # Check few-shot examples file
            if self._check_file_modified(self.config.few_shot_examples_path):
                logger.info(f"Content change detected in examples file: {self.config.few_shot_examples_path}")
                changed_files.append(self.config.few_shot_examples_path)
                change_type = "examples_file" if change_type == "none" else "multiple_files"
            
            # Check system identity file if configured
            if self.config.system_identity_path and self._check_file_modified(self.config.system_identity_path):
                logger.info(f"Content change detected in identity file: {self.config.system_identity_path}")
                changed_files.append(self.config.system_identity_path)
                change_type = "identity_file" if change_type == "none" else "multiple_files"
            
            # Check tool definitions (simplified - would need actual tool registry integration)
            if self._check_tool_definitions_changed():
                logger.info("Content change detected in tool definitions")
                changed_files.append(self.config.tool_definitions_path)
                change_type = "tool_definitions" if change_type == "none" else "multiple_files"
            
            # Audit log content change detection
            if changed_files:
                audit_logger = get_cache_audit_logger(self.config)
                audit_logger.log_content_change_detected(
                    changed_files,
                    change_type,
                    True,  # reload will be triggered
                    {"detection_method": "file_modification_time"}
                )
            
            return len(changed_files) > 0
            
        except Exception as e:
            logger.error(f"Error checking content changes: {e}")
            
            # Audit log the error
            audit_logger = get_cache_audit_logger(self.config)
            audit_logger.log_content_loading_failure(
                str(e),
                "Error during content change detection",
                ["assume_content_changed"],
                additional_details={"detection_method": "file_modification_time"}
            )
            
            # On error, assume content changed to trigger reload
            return True
    
    def _check_file_modified(self, file_path: str) -> bool:
        """
        Check if a file has been modified since last check.
        
        Args:
            file_path: Path to file to check
            
        Returns:
            True if file was modified
        """
        try:
            path = Path(file_path)
            if not path.exists():
                # File doesn't exist - consider it changed if we had it before
                return file_path in self._file_modification_times
            
            current_mtime = path.stat().st_mtime
            last_mtime = self._file_modification_times.get(file_path, 0)
            
            return current_mtime > last_mtime
            
        except Exception as e:
            logger.warning(f"Error checking file modification time for {file_path}: {e}")
            return True  # Assume changed on error
    
    def _check_tool_definitions_changed(self) -> bool:
        """
        Check if tool definitions have changed.
        
        This is a simplified implementation. A full implementation would
        integrate with the Parlant tool registry to detect schema changes.
        
        Returns:
            True if tool definitions changed
        """
        try:
            # For now, check if the tools directory has been modified
            tools_path = Path(self.config.tool_definitions_path)
            if tools_path.exists() and tools_path.is_dir():
                # Get the latest modification time of any Python file in tools directory
                latest_mtime = 0
                for py_file in tools_path.rglob("*.py"):
                    try:
                        mtime = py_file.stat().st_mtime
                        latest_mtime = max(latest_mtime, mtime)
                    except Exception:
                        continue
                
                last_tools_mtime = self._file_modification_times.get("__tools_directory__", 0)
                return latest_mtime > last_tools_mtime
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking tool definitions changes: {e}")
            return False
    
    async def _update_file_modification_times(self):
        """Update stored file modification times for change detection."""
        try:
            # Update policy document times
            for file_path in self.config.static_content_paths.values():
                path = Path(file_path)
                if path.exists():
                    self._file_modification_times[file_path] = path.stat().st_mtime
            
            # Update examples file time
            examples_path = Path(self.config.few_shot_examples_path)
            if examples_path.exists():
                self._file_modification_times[self.config.few_shot_examples_path] = examples_path.stat().st_mtime
            
            # Update system identity file time if configured
            if self.config.system_identity_path:
                identity_path = Path(self.config.system_identity_path)
                if identity_path.exists():
                    self._file_modification_times[self.config.system_identity_path] = identity_path.stat().st_mtime
            
            # Update tools directory time
            tools_path = Path(self.config.tool_definitions_path)
            if tools_path.exists() and tools_path.is_dir():
                latest_mtime = 0
                for py_file in tools_path.rglob("*.py"):
                    try:
                        mtime = py_file.stat().st_mtime
                        latest_mtime = max(latest_mtime, mtime)
                    except Exception:
                        continue
                self._file_modification_times["__tools_directory__"] = latest_mtime
            
        except Exception as e:
            logger.error(f"Error updating file modification times: {e}")
    
    async def reload_content(self) -> bool:
        """
        Reload static content from files.
        
        If in fallback mode, attempts content restoration.
        If content loading fails, maintains existing templates.
        
        Returns:
            True if content was reloaded successfully
        """
        old_hash = self._last_content_hash
        old_template = self._prompt_template
        old_static_content = self.static_content
        
        try:
            logger.info("Reloading static content due to detected changes...")
            
            # If in fallback mode, this is a restoration attempt
            if self._fallback_mode:
                logger.info("Attempting content restoration from fallback mode...")
            
            success = await self.initialize_content()
            
            # Consider reload successful if we have any content (including fallback)
            if self.static_content is not None:
                if self._last_content_hash != old_hash:
                    logger.info(
                        f"Content successfully reloaded. Hash changed from {old_hash} to {self._last_content_hash}"
                    )
                    self._log_content_change_details(old_template)
                else:
                    logger.info("Content reloaded but no changes detected in hash")
                return True
            else:
                logger.error("Content reload failed completely, maintaining existing templates")
                # Restore previous state on complete failure
                self._last_content_hash = old_hash
                self._prompt_template = old_template
                self.static_content = old_static_content
                return False
            
        except Exception as e:
            logger.error(f"Error during content reload: {e}")
            # Restore previous state on error
            self._last_content_hash = old_hash
            self._prompt_template = old_template
            self.static_content = old_static_content
            
            # If not already in fallback mode, activate it
            if not self._fallback_mode:
                self._activate_fallback_mode(f"Content reload failed: {str(e)}")
            
            return False
    
    def _log_content_change_details(self, old_template: Optional[str]):
        """
        Log details about what content changed.
        
        Args:
            old_template: Previous prompt template for comparison
        """
        try:
            if not self.static_content:
                return
            
            changes = []
            
            # Check which content types were reloaded
            for content_type, status in self.static_content.loading_statuses.items():
                if status.status == LoadingStatus.LOADED:
                    changes.append(f"{content_type.value}: loaded successfully")
                elif status.status == LoadingStatus.FALLBACK:
                    changes.append(f"{content_type.value}: using fallback")
                elif status.status == LoadingStatus.FAILED:
                    changes.append(f"{content_type.value}: failed to load")
            
            if changes:
                logger.info(f"Content change details: {', '.join(changes)}")
            
            # Log token count changes
            logger.info(f"New total token count: {self.static_content.total_tokens}")
            
        except Exception as e:
            logger.warning(f"Error logging content change details: {e}")
    
    async def trigger_content_update(self, reason: str = "manual") -> bool:
        """
        Trigger a content update with template refresh.
        
        Args:
            reason: Reason for the update (for logging)
            
        Returns:
            True if update was successful
        """
        logger.info(f"Triggering content update. Reason: {reason}")
        
        try:
            success = await self.reload_content()
            
            if success:
                logger.info(f"Content update completed successfully. Reason: {reason}")
            else:
                logger.error(f"Content update failed. Reason: {reason}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during content update (reason: {reason}): {e}")
            return False
    
    def disable_change_detection(self):
        """Disable automatic change detection (for testing)."""
        self._change_detection_enabled = False
        logger.debug("Content change detection disabled")
    
    def enable_change_detection(self):
        """Enable automatic change detection."""
        self._change_detection_enabled = True
        logger.debug("Content change detection enabled")
    
    def is_fallback_mode(self) -> bool:
        """Check if content loader is in fallback mode."""
        return self._fallback_mode
    
    def get_fallback_status(self) -> Dict:
        """
        Get detailed fallback mode status.
        
        Returns:
            Dictionary with fallback mode details
        """
        return {
            "fallback_mode": self._fallback_mode,
            "fallback_reason": self._fallback_reason,
            "fallback_activated_at": self._fallback_activated_at,
            "restoration_attempts": self._restoration_attempts,
            "last_restoration_attempt": self._last_restoration_attempt,
            "next_restoration_due": self._get_next_restoration_time(),
        }
    
    def _activate_fallback_mode(self, reason: str):
        """
        Activate fallback mode when content loading fails.
        
        Args:
            reason: Reason for activating fallback mode
        """
        if not self._fallback_mode:
            self._fallback_mode = True
            self._fallback_reason = reason
            self._fallback_activated_at = datetime.now()
            self._restoration_attempts = 0
            self._last_restoration_attempt = None
            
            logger.warning(f"Fallback mode activated. Reason: {reason}")
            logger.info("Operating with basic/hardcoded content until restoration")
            
            # Audit log fallback mode activation
            failed_content_types = []
            if self.static_content:
                failed_content_types = [ct.value for ct in self.static_content.get_failed_content_types()]
            
            audit_logger = get_cache_audit_logger(self.config)
            audit_logger.log_fallback_mode_activated(
                reason,
                failed_content_types,
                {"activated_at": self._fallback_activated_at.isoformat()}
            )
        else:
            logger.debug(f"Already in fallback mode. Additional reason: {reason}")
    
    def _deactivate_fallback_mode(self):
        """Deactivate fallback mode when content is restored."""
        if self._fallback_mode:
            self._fallback_mode = False
            duration = datetime.now() - self._fallback_activated_at if self._fallback_activated_at else None
            duration_minutes = duration.total_seconds() / 60 if duration else 0
            
            logger.info(f"Fallback mode deactivated. Duration: {duration}")
            logger.info(f"Content restored after {self._restoration_attempts} attempts")
            
            # Audit log fallback mode deactivation
            audit_logger = get_cache_audit_logger(self.config)
            audit_logger.log_fallback_mode_deactivated(
                duration_minutes,
                self._restoration_attempts,
                {
                    "deactivated_at": datetime.now().isoformat(),
                    "total_duration_seconds": duration.total_seconds() if duration else 0
                }
            )
            
            # Reset fallback state
            self._fallback_reason = None
            self._fallback_activated_at = None
            self._restoration_attempts = 0
            self._last_restoration_attempt = None
    
    def _should_attempt_restoration(self) -> bool:
        """
        Check if we should attempt content restoration.
        
        Returns:
            True if restoration should be attempted
        """
        if not self._fallback_mode:
            return False
        
        if self._last_restoration_attempt is None:
            return True
        
        time_since_last_attempt = datetime.now() - self._last_restoration_attempt
        return time_since_last_attempt.total_seconds() >= self._restoration_interval_seconds
    
    def _get_next_restoration_time(self) -> Optional[datetime]:
        """Get the next scheduled restoration attempt time."""
        if not self._fallback_mode or self._last_restoration_attempt is None:
            return None
        
        return self._last_restoration_attempt + timedelta(seconds=self._restoration_interval_seconds)
    
    async def attempt_content_restoration(self) -> bool:
        """
        Attempt to restore content from fallback mode.
        
        Returns:
            True if content was successfully restored
        """
        if not self._fallback_mode:
            logger.debug("Not in fallback mode, no restoration needed")
            return True
        
        if not self._should_attempt_restoration():
            logger.debug("Restoration attempt not due yet")
            return False
        
        self._restoration_attempts += 1
        self._last_restoration_attempt = datetime.now()
        
        logger.info(f"Attempting content restoration (attempt #{self._restoration_attempts})")
        
        try:
            # Try to reload content
            old_static_content = self.static_content
            success = await self.initialize_content()
            
            if success and self.static_content and self.static_content.is_complete():
                # Full restoration successful - fallback mode should be deactivated by initialize_content
                logger.info("Content fully restored from fallback mode")
                return True
            elif self.static_content and not self._fallback_mode:
                # Partial restoration but good enough to exit fallback mode
                logger.info("Content partially restored, exited fallback mode")
                return True
            else:
                # Still in fallback mode
                failed_types = self.static_content.get_failed_content_types() if self.static_content else []
                if failed_types:
                    logger.info(f"Partial content restoration. Still failed: {[t.value for t in failed_types]}")
                else:
                    logger.warning(f"Content restoration attempt #{self._restoration_attempts} failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during content restoration attempt #{self._restoration_attempts}: {e}")
            return False
    
    def _create_fallback_content(self) -> StaticContent:
        """
        Create fallback static content when files are unavailable.
        
        Returns:
            StaticContent with hardcoded fallback content
        """
        # Create fallback loading statuses
        loading_statuses = {}
        for content_type in ContentType:
            status = ContentLoadingStatus(
                content_type=content_type,
                status=LoadingStatus.FALLBACK,
                loaded_at=datetime.now(),
                error_message="Using fallback content due to file unavailability",
            )
            loading_statuses[content_type] = status
        
        # Use hardcoded fallback content
        fallback_policies = """## Refund Policies (Fallback)
Basic refund policy guidelines:
- Pre-arrival cancellations may be eligible for refund
- Post-event requests are generally not eligible
- Oversold situations require human review
- Complex cases should be escalated"""
        
        fallback_tools = """## Available Tools (Fallback)
Basic tool definitions:
- get_ticket: Retrieve ticket information
- add_note: Add note to ticket
- update_ticket: Update ticket status"""
        
        # Calculate token counts
        system_tokens = self._estimate_tokens(DEFAULT_SYSTEM_IDENTITY)
        policies_tokens = self._estimate_tokens(fallback_policies)
        tools_tokens = self._estimate_tokens(fallback_tools)
        examples_tokens = self._estimate_tokens(DEFAULT_FEW_SHOT_EXAMPLES)
        
        # Update loading statuses with token counts
        loading_statuses[ContentType.SYSTEM_IDENTITY].token_count = system_tokens
        loading_statuses[ContentType.REFUND_POLICIES].token_count = policies_tokens
        loading_statuses[ContentType.TOOL_DEFINITIONS].token_count = tools_tokens
        loading_statuses[ContentType.FEW_SHOT_EXAMPLES].token_count = examples_tokens
        
        combined_content = (
            DEFAULT_SYSTEM_IDENTITY + fallback_policies + fallback_tools + DEFAULT_FEW_SHOT_EXAMPLES
        )
        
        return StaticContent(
            system_identity=DEFAULT_SYSTEM_IDENTITY,
            refund_policies=fallback_policies,
            tool_definitions=fallback_tools,
            few_shot_examples=DEFAULT_FEW_SHOT_EXAMPLES,
            total_tokens=system_tokens + policies_tokens + tools_tokens + examples_tokens,
            content_hash=self._calculate_content_hash(combined_content),
            loaded_at=datetime.now(),
            loading_statuses=loading_statuses,
        )
    
    async def _load_system_identity(self) -> tuple[str, ContentLoadingStatus]:
        """Load system identity and role instructions."""
        status = ContentLoadingStatus(
            content_type=ContentType.SYSTEM_IDENTITY,
            file_path=self.config.system_identity_path,
        )
        
        try:
            if self.config.system_identity_path:
                path = Path(self.config.system_identity_path)
                if path.exists():
                    content = path.read_text(encoding='utf-8')
                    status.status = LoadingStatus.LOADED
                    status.loaded_at = datetime.now()
                    status.token_count = self._estimate_tokens(content)
                    logger.debug(f"Loaded system identity from {path}")
                    return content, status
            
            # Use default template - this is normal, not a fallback
            status.status = LoadingStatus.LOADED  # Changed from FALLBACK to LOADED
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(DEFAULT_SYSTEM_IDENTITY)
            logger.debug("Using default system identity template")
            return DEFAULT_SYSTEM_IDENTITY, status
            
        except Exception as e:
            logger.error(f"Error loading system identity: {e}")
            status.status = LoadingStatus.FALLBACK
            status.error_message = str(e)
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(DEFAULT_SYSTEM_IDENTITY)
            return DEFAULT_SYSTEM_IDENTITY, status
    
    async def _load_refund_policies(self) -> tuple[str, ContentLoadingStatus]:
        """Load refund policy documents."""
        status = ContentLoadingStatus(
            content_type=ContentType.REFUND_POLICIES,
        )
        
        try:
            policies_content = []
            loaded_files = []
            
            for name, path_str in self.config.static_content_paths.items():
                path = Path(path_str)
                if path.exists():
                    content = path.read_text(encoding='utf-8')
                    policies_content.append(f"## {name.replace('_', ' ').title()}\n{content}")
                    loaded_files.append(path_str)
                    logger.debug(f"Loaded policy document: {path}")
                else:
                    logger.warning(f"Policy document not found: {path}")
            
            if policies_content:
                combined = "\n\n".join(policies_content)
                status.status = LoadingStatus.LOADED
                status.loaded_at = datetime.now()
                status.file_path = ", ".join(loaded_files)
                status.token_count = self._estimate_tokens(combined)
                return combined, status
            
            # No policies found - use fallback
            fallback = "## Refund Policies\nNo policy documents loaded. Using default behavior."
            status.status = LoadingStatus.FALLBACK
            status.loaded_at = datetime.now()
            status.error_message = "No policy documents found"
            status.token_count = self._estimate_tokens(fallback)
            logger.warning("No policy documents found, using fallback")
            return fallback, status
            
        except Exception as e:
            logger.error(f"Error loading refund policies: {e}")
            fallback = "## Refund Policies\nError loading policies. Using default behavior."
            status.status = LoadingStatus.FAILED
            status.error_message = str(e)
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(fallback)
            return fallback, status
    
    async def _load_tool_definitions(self) -> tuple[str, ContentLoadingStatus]:
        """Load tool definitions from Parlant tools."""
        status = ContentLoadingStatus(
            content_type=ContentType.TOOL_DEFINITIONS,
            file_path=self.config.tool_definitions_path,
        )
        
        try:
            # For now, return a placeholder - actual implementation would
            # extract tool schemas from registered Parlant tools
            tool_defs = """## Available Tools

### get_ticket
Retrieves ticket information from Freshdesk.
Parameters:
- ticket_id (string): The Freshdesk ticket ID

### get_customer_orders
Retrieves customer booking information from ParkWhiz.
Parameters:
- customer_email (string): Customer email address

### add_note
Adds a private note to a Freshdesk ticket.
Parameters:
- ticket_id (string): The Freshdesk ticket ID
- note (string): Note content to add

### update_ticket
Updates ticket status and fields.
Parameters:
- ticket_id (string): The Freshdesk ticket ID
- status (string): New ticket status
- custom_fields (object): Custom field updates"""
            
            status.status = LoadingStatus.LOADED
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(tool_defs)
            logger.debug("Loaded tool definitions")
            return tool_defs, status
            
        except Exception as e:
            logger.error(f"Error loading tool definitions: {e}")
            fallback = "## Available Tools\nNo tool definitions available."
            status.status = LoadingStatus.FAILED
            status.error_message = str(e)
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(fallback)
            return fallback, status
    
    async def _load_few_shot_examples(self) -> tuple[str, ContentLoadingStatus]:
        """Load few-shot examples for training."""
        status = ContentLoadingStatus(
            content_type=ContentType.FEW_SHOT_EXAMPLES,
            file_path=self.config.few_shot_examples_path,
        )
        
        try:
            path = Path(self.config.few_shot_examples_path)
            if path.exists():
                if path.suffix == '.json':
                    with open(path, 'r', encoding='utf-8') as f:
                        examples_data = json.load(f)
                    content = self._format_examples_from_json(examples_data)
                else:
                    content = path.read_text(encoding='utf-8')
                
                status.status = LoadingStatus.LOADED
                status.loaded_at = datetime.now()
                status.token_count = self._estimate_tokens(content)
                logger.debug(f"Loaded few-shot examples from {path}")
                return content, status
            
            # Use default examples - this is normal, not a fallback
            status.status = LoadingStatus.LOADED  # Changed from FALLBACK to LOADED
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(DEFAULT_FEW_SHOT_EXAMPLES)
            logger.debug("Using default few-shot examples")
            return DEFAULT_FEW_SHOT_EXAMPLES, status
            
        except Exception as e:
            logger.error(f"Error loading few-shot examples: {e}")
            status.status = LoadingStatus.FALLBACK
            status.error_message = str(e)
            status.loaded_at = datetime.now()
            status.token_count = self._estimate_tokens(DEFAULT_FEW_SHOT_EXAMPLES)
            return DEFAULT_FEW_SHOT_EXAMPLES, status
    
    def _format_examples_from_json(self, examples_data: List[Dict]) -> str:
        """Format JSON examples into prompt-friendly text."""
        formatted = ["## Few-Shot Examples"]
        
        for i, example in enumerate(examples_data, 1):
            formatted.append(f"\n### Example {i}: {example.get('title', 'Untitled')}")
            formatted.append(f"Ticket: {example.get('ticket', 'N/A')}")
            formatted.append(f"Decision: {example.get('decision', 'N/A')}")
            formatted.append(f"Reasoning: {example.get('reasoning', 'N/A')}")
        
        return "\n".join(formatted)
    
    def _build_prompt_template(self) -> str:
        """Build the prompt template with static content first."""
        if not self.static_content:
            return self._get_fallback_template()
        
        return f"""# SYSTEM IDENTITY & ROLE
{self.static_content.system_identity}

# REFUND POLICY DOCUMENTS
{self.static_content.refund_policies}

# TOOL DEFINITIONS
{self.static_content.tool_definitions}

# FEW-SHOT EXAMPLES
{self.static_content.few_shot_examples}

# CURRENT REQUEST
{{dynamic_content}}"""
    
    def _get_fallback_template(self) -> str:
        """Get fallback template when content loading fails."""
        return f"""# SYSTEM IDENTITY & ROLE
{DEFAULT_SYSTEM_IDENTITY}

# FEW-SHOT EXAMPLES
{DEFAULT_FEW_SHOT_EXAMPLES}

# CURRENT REQUEST
{{dynamic_content}}"""
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Uses a simple heuristic: ~4 characters per token.
        For production, use tiktoken or the actual tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        if not text:
            return 0
        # Rough estimate: 4 characters per token
        return len(text) // 4
    
    def _calculate_content_hash(self, content: str) -> str:
        """
        Calculate SHA-256 hash of content for change detection.
        
        Args:
            content: Content to hash
            
        Returns:
            Hex digest of SHA-256 hash
        """
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def validate_content_structure(self, template: str) -> ContentValidationResult:
        """
        Validate prompt structure for implicit caching optimization.
        
        Checks that static content comes first and dynamic content last.
        
        Args:
            template: Prompt template to validate
            
        Returns:
            Validation result with structure analysis
        """
        result = ContentValidationResult(status=ValidationStatus.VALID)
        
        try:
            # Check for required sections
            required_sections = [
                "# SYSTEM IDENTITY & ROLE",
                "# REFUND POLICY DOCUMENTS", 
                "# TOOL DEFINITIONS",
                "# FEW-SHOT EXAMPLES",
                "# CURRENT REQUEST",
                "{dynamic_content}"
            ]
            
            missing_sections = []
            section_positions = {}
            
            for section in required_sections:
                pos = template.find(section)
                if pos == -1:
                    missing_sections.append(section)
                else:
                    section_positions[section] = pos
            
            if missing_sections:
                result.status = ValidationStatus.STRUCTURE_INVALID
                result.structure_valid = False
                result.errors.append(f"Missing required sections: {missing_sections}")
            
            # Check section ordering (static first, dynamic last)
            if len(section_positions) >= 2:
                positions = list(section_positions.values())
                if positions != sorted(positions):
                    result.status = ValidationStatus.STRUCTURE_INVALID
                    result.structure_valid = False
                    result.errors.append("Sections are not in correct order (static first, dynamic last)")
            
            # Check that dynamic content placeholder is at the end
            dynamic_pos = template.find("{dynamic_content}")
            if dynamic_pos != -1:
                content_after_dynamic = template[dynamic_pos + len("{dynamic_content}"):].strip()
                if content_after_dynamic:
                    result.status = ValidationStatus.WARNING
                    result.warnings.append("Content found after dynamic placeholder may reduce cache effectiveness")
            
            logger.debug(f"Content structure validation: {result.status.value}")
            
        except Exception as e:
            logger.error(f"Error validating content structure: {e}")
            result.status = ValidationStatus.INVALID
            result.structure_valid = False
            result.errors.append(f"Validation error: {str(e)}")
        
        return result
    
    def validate_content_size(self, static_content: StaticContent) -> ContentValidationResult:
        """
        Validate content size against Gemini limits.
        
        Args:
            static_content: Static content to validate
            
        Returns:
            Validation result with size analysis
        """
        result = ContentValidationResult(status=ValidationStatus.VALID)
        
        try:
            # Calculate total size
            total_tokens = static_content.total_tokens
            combined_content = (
                static_content.system_identity +
                static_content.refund_policies +
                static_content.tool_definitions +
                static_content.few_shot_examples
            )
            content_size_bytes = len(combined_content.encode('utf-8'))
            
            result.total_tokens = total_tokens
            result.content_size_bytes = content_size_bytes
            
            # Check against limits
            max_tokens = self.config.max_static_content_tokens
            
            if total_tokens > max_tokens:
                result.status = ValidationStatus.SIZE_EXCEEDED
                result.size_within_limits = False
                result.errors.append(
                    f"Content size ({total_tokens} tokens) exceeds limit ({max_tokens} tokens)"
                )
            elif total_tokens > max_tokens * 0.9:
                result.status = ValidationStatus.WARNING
                result.warnings.append(
                    f"Content size ({total_tokens} tokens) approaching limit ({max_tokens} tokens)"
                )
            
            # Log size information
            logger.info(
                f"Content size validation: {total_tokens} tokens, "
                f"{content_size_bytes} bytes, status: {result.status.value}"
            )
            
        except Exception as e:
            logger.error(f"Error validating content size: {e}")
            result.status = ValidationStatus.INVALID
            result.errors.append(f"Size validation error: {str(e)}")
        
        return result
    
    def truncate_content_with_priority(self, static_content: StaticContent) -> StaticContent:
        """
        Truncate content based on priority to fit within size limits.
        
        Priority order: policies > tools > examples > identity
        
        Args:
            static_content: Content to truncate
            
        Returns:
            Truncated static content
        """
        max_tokens = self.config.max_static_content_tokens
        
        if static_content.total_tokens <= max_tokens:
            return static_content
        
        logger.warning(
            f"Content size ({static_content.total_tokens} tokens) exceeds limit "
            f"({max_tokens} tokens). Applying intelligent truncation."
        )
        
        # Get content by priority
        content_by_priority = static_content.get_content_by_priority()
        
        # Start with empty content and add by priority until we hit the limit
        truncated_identity = ""
        truncated_policies = ""
        truncated_tools = ""
        truncated_examples = ""
        
        # Reserve tokens for the template structure itself (headers, etc.)
        # Be more conservative with overhead estimation
        structure_overhead = 300  # Increased from 200 to be more conservative
        available_tokens = max(max_tokens - structure_overhead, max_tokens // 3)  # More conservative minimum
        
        truncation_applied = False
        
        for content_type, content, priority in content_by_priority:
            content_tokens = self._estimate_tokens(content)
            
            if content_tokens <= available_tokens:
                # Content fits completely
                if content_type == ContentType.SYSTEM_IDENTITY:
                    truncated_identity = content
                elif content_type == ContentType.REFUND_POLICIES:
                    truncated_policies = content
                elif content_type == ContentType.TOOL_DEFINITIONS:
                    truncated_tools = content
                elif content_type == ContentType.FEW_SHOT_EXAMPLES:
                    truncated_examples = content
                
                available_tokens -= content_tokens
            else:
                # Need to truncate this content
                if available_tokens > 50:  # Only truncate if we have reasonable space
                    truncated_content = self._truncate_content_intelligently(
                        content, available_tokens, content_type
                    )
                    
                    if content_type == ContentType.SYSTEM_IDENTITY:
                        truncated_identity = truncated_content
                    elif content_type == ContentType.REFUND_POLICIES:
                        truncated_policies = truncated_content
                    elif content_type == ContentType.TOOL_DEFINITIONS:
                        truncated_tools = truncated_content
                    elif content_type == ContentType.FEW_SHOT_EXAMPLES:
                        truncated_examples = truncated_content
                    
                    available_tokens = 0
                    truncation_applied = True
                
                # No more space for additional content
                break
        
        # Create truncated static content and validate it fits
        combined_truncated = (
            truncated_identity + truncated_policies + truncated_tools + truncated_examples
        )
        
        # Build the actual template to get accurate token count including structure
        temp_static_content = StaticContent(
            system_identity=truncated_identity,
            refund_policies=truncated_policies,
            tool_definitions=truncated_tools,
            few_shot_examples=truncated_examples,
            total_tokens=0,  # Will be recalculated
            content_hash="",
            loaded_at=static_content.loaded_at,
            loading_statuses=static_content.loading_statuses.copy(),
        )
        
        # Build template to get accurate size including headers
        template = self._build_prompt_template_from_content(temp_static_content)
        actual_tokens = self._estimate_tokens(template)
        
        # If still over limit, apply more aggressive truncation
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        while actual_tokens > max_tokens and iteration < max_iterations:
            iteration += 1
            excess_tokens = actual_tokens - max_tokens
            
            # For very small limits, be extremely aggressive
            if max_tokens <= 50:
                # For tiny limits, just keep minimal structure
                truncated_identity = ""
                truncated_policies = ""
                truncated_tools = ""
                truncated_examples = ""
                
                # Recalculate with minimal content
                temp_static_content = StaticContent(
                    system_identity=truncated_identity,
                    refund_policies=truncated_policies,
                    tool_definitions=truncated_tools,
                    few_shot_examples=truncated_examples,
                    total_tokens=0,
                    content_hash="",
                    loaded_at=static_content.loaded_at,
                    loading_statuses=static_content.loading_statuses.copy(),
                )
                
                template = self._build_prompt_template_from_content(temp_static_content)
                actual_tokens = self._estimate_tokens(template)
                truncation_applied = True
                break
            
            # Be more aggressive - reduce by at least 20% more than needed
            safety_factor = 1.3
            target_reduction = int(excess_tokens * safety_factor)
            
            # Apply reduction to lowest priority content first
            if truncated_examples and self._estimate_tokens(truncated_examples) > 0:
                examples_tokens = self._estimate_tokens(truncated_examples)
                if examples_tokens > target_reduction:
                    new_examples_tokens = max(0, examples_tokens - target_reduction)
                    truncated_examples = self._truncate_content_intelligently(
                        truncated_examples, new_examples_tokens, ContentType.FEW_SHOT_EXAMPLES
                    )
                    target_reduction = 0
                else:
                    truncated_examples = ""
                    target_reduction -= examples_tokens
                truncation_applied = True
            
            # If still need to reduce, truncate identity
            if target_reduction > 0 and truncated_identity and self._estimate_tokens(truncated_identity) > 0:
                identity_tokens = self._estimate_tokens(truncated_identity)
                if identity_tokens > target_reduction:
                    new_identity_tokens = max(0, identity_tokens - target_reduction)
                    truncated_identity = self._truncate_content_intelligently(
                        truncated_identity, new_identity_tokens, ContentType.SYSTEM_IDENTITY
                    )
                    target_reduction = 0
                else:
                    truncated_identity = ""
                    target_reduction -= identity_tokens
                truncation_applied = True
            
            # If still need to reduce, truncate tools
            if target_reduction > 0 and truncated_tools and self._estimate_tokens(truncated_tools) > 0:
                tools_tokens = self._estimate_tokens(truncated_tools)
                if tools_tokens > target_reduction:
                    new_tools_tokens = max(0, tools_tokens - target_reduction)
                    truncated_tools = self._truncate_content_intelligently(
                        truncated_tools, new_tools_tokens, ContentType.TOOL_DEFINITIONS
                    )
                    target_reduction = 0
                else:
                    truncated_tools = ""
                    target_reduction -= tools_tokens
                truncation_applied = True
            
            # If still need to reduce, truncate policies (last resort)
            if target_reduction > 0 and truncated_policies and self._estimate_tokens(truncated_policies) > 0:
                policies_tokens = self._estimate_tokens(truncated_policies)
                if policies_tokens > target_reduction:
                    new_policies_tokens = max(0, policies_tokens - target_reduction)
                    truncated_policies = self._truncate_content_intelligently(
                        truncated_policies, new_policies_tokens, ContentType.REFUND_POLICIES
                    )
                else:
                    truncated_policies = ""
                truncation_applied = True
            
            # Recalculate with new content
            temp_static_content = StaticContent(
                system_identity=truncated_identity,
                refund_policies=truncated_policies,
                tool_definitions=truncated_tools,
                few_shot_examples=truncated_examples,
                total_tokens=0,
                content_hash="",
                loaded_at=static_content.loaded_at,
                loading_statuses=static_content.loading_statuses.copy(),
            )
            
            template = self._build_prompt_template_from_content(temp_static_content)
            actual_tokens = self._estimate_tokens(template)
        
        # Update loading statuses to reflect truncation
        updated_statuses = static_content.loading_statuses.copy()
        for content_type, status in updated_statuses.items():
            if content_type == ContentType.SYSTEM_IDENTITY and truncated_identity != static_content.system_identity:
                status.error_message = "Content truncated due to size limits"
            elif content_type == ContentType.REFUND_POLICIES and truncated_policies != static_content.refund_policies:
                status.error_message = "Content truncated due to size limits"
            elif content_type == ContentType.TOOL_DEFINITIONS and truncated_tools != static_content.tool_definitions:
                status.error_message = "Content truncated due to size limits"
            elif content_type == ContentType.FEW_SHOT_EXAMPLES and truncated_examples != static_content.few_shot_examples:
                status.error_message = "Content truncated due to size limits"
        
        # Final content creation
        combined_truncated = (
            truncated_identity + truncated_policies + truncated_tools + truncated_examples
        )
        
        truncated_static_content = StaticContent(
            system_identity=truncated_identity,
            refund_policies=truncated_policies,
            tool_definitions=truncated_tools,
            few_shot_examples=truncated_examples,
            total_tokens=actual_tokens,
            content_hash=self._calculate_content_hash(combined_truncated),
            loaded_at=static_content.loaded_at,
            loading_statuses=updated_statuses,
        )
        
        if truncation_applied:
            logger.warning(
                f"Content truncated from {static_content.total_tokens} to "
                f"{truncated_static_content.total_tokens} tokens (iterations: {iteration})"
            )
            
            # Alert administrators about truncation
            self._alert_administrators_about_truncation(
                static_content.total_tokens,
                truncated_static_content.total_tokens,
                max_tokens
            )
        
        return truncated_static_content
    
    def _truncate_content_intelligently(
        self, 
        content: str, 
        max_tokens: int, 
        content_type: ContentType
    ) -> str:
        """
        Truncate content while preserving integrity.
        
        Args:
            content: Content to truncate
            max_tokens: Maximum tokens allowed
            content_type: Type of content for intelligent truncation
            
        Returns:
            Truncated content
        """
        if max_tokens <= 0:
            return ""
        
        # Be very conservative with token estimation to ensure we stay under limits
        # Use a more conservative characters per token ratio
        chars_per_token = 2.5  # Even more conservative than 3
        max_chars = int(max_tokens * chars_per_token)
        
        if len(content) <= max_chars:
            return content
        
        # Truncate based on content type
        if content_type == ContentType.REFUND_POLICIES:
            # For policies, try to keep complete sections
            return self._truncate_by_sections(content, max_chars)
        elif content_type == ContentType.TOOL_DEFINITIONS:
            # For tools, try to keep complete tool definitions
            return self._truncate_by_tools(content, max_chars)
        elif content_type == ContentType.FEW_SHOT_EXAMPLES:
            # For examples, try to keep complete examples
            return self._truncate_by_examples(content, max_chars)
        else:
            # For identity, simple truncation with ellipsis
            if max_chars > 3:
                return content[:max_chars-3] + "..."
            else:
                return content[:max_chars]
    
    def _truncate_by_sections(self, content: str, max_chars: int) -> str:
        """Truncate policy content by complete sections."""
        lines = content.split('\n')
        truncated_lines = []
        current_chars = 0
        
        for line in lines:
            line_chars = len(line) + 1  # +1 for newline
            if current_chars + line_chars > max_chars:
                break
            truncated_lines.append(line)
            current_chars += line_chars
        
        result = '\n'.join(truncated_lines)
        if len(result) < len(content):
            result += "\n\n[Content truncated due to size limits]"
        
        return result
    
    def _truncate_by_tools(self, content: str, max_chars: int) -> str:
        """Truncate tool definitions by complete tools."""
        # Split by tool sections (assuming ### headers)
        sections = content.split('###')
        truncated_sections = [sections[0]]  # Keep header
        current_chars = len(sections[0])
        
        for section in sections[1:]:
            section_with_header = '###' + section
            section_chars = len(section_with_header)
            
            if current_chars + section_chars > max_chars:
                break
            
            truncated_sections.append(section)
            current_chars += section_chars
        
        result = '###'.join(truncated_sections)
        if len(result) < len(content):
            result += "\n\n[Tool definitions truncated due to size limits]"
        
        return result
    
    def _truncate_by_examples(self, content: str, max_chars: int) -> str:
        """Truncate examples by complete examples."""
        # Split by example sections (assuming ### headers)
        sections = content.split('###')
        truncated_sections = [sections[0]]  # Keep header
        current_chars = len(sections[0])
        
        for section in sections[1:]:
            section_with_header = '###' + section
            section_chars = len(section_with_header)
            
            if current_chars + section_chars > max_chars:
                break
            
            truncated_sections.append(section)
            current_chars += section_chars
        
        result = '###'.join(truncated_sections)
        if len(result) < len(content):
            result += "\n\n[Examples truncated due to size limits]"
        
        return result
    
    def _alert_administrators_about_truncation(
        self, 
        original_tokens: int, 
        truncated_tokens: int, 
        limit_tokens: int
    ):
        """
        Alert administrators about content truncation.
        
        Args:
            original_tokens: Original token count
            truncated_tokens: Truncated token count
            limit_tokens: Token limit
        """
        logger.warning(
            f"ADMINISTRATOR ALERT: Content truncation applied. "
            f"Original: {original_tokens} tokens, "
            f"Truncated: {truncated_tokens} tokens, "
            f"Limit: {limit_tokens} tokens. "
            f"Consider reviewing content size or increasing limits."
        )
        
        # In a production system, this would send alerts via email, Slack, etc.
        # For now, we just log at WARNING level which should be monitored
    
    def validate_and_process_content(self, static_content: StaticContent) -> StaticContent:
        """
        Validate content structure and size, applying truncation if needed.
        
        Args:
            static_content: Content to validate and process
            
        Returns:
            Processed static content with validation results
        """
        # First validate structure
        template = self._build_prompt_template_from_content(static_content)
        structure_result = self.validate_content_structure(template)
        
        # Then validate size
        size_result = self.validate_content_size(static_content)
        
        # Combine validation results
        combined_result = ContentValidationResult(
            status=ValidationStatus.VALID,
            total_tokens=size_result.total_tokens,
            content_size_bytes=size_result.content_size_bytes,
            structure_valid=structure_result.structure_valid,
            size_within_limits=size_result.size_within_limits,
            warnings=structure_result.warnings + size_result.warnings,
            errors=structure_result.errors + size_result.errors,
        )
        
        # Determine overall status
        if combined_result.errors:
            if any("size" in error.lower() for error in combined_result.errors):
                combined_result.status = ValidationStatus.SIZE_EXCEEDED
            elif any("structure" in error.lower() for error in combined_result.errors):
                combined_result.status = ValidationStatus.STRUCTURE_INVALID
            else:
                combined_result.status = ValidationStatus.INVALID
        elif combined_result.warnings:
            combined_result.status = ValidationStatus.WARNING
        
        # Apply truncation if size exceeded
        processed_content = static_content
        if combined_result.status == ValidationStatus.SIZE_EXCEEDED:
            processed_content = self.truncate_content_with_priority(static_content)
            combined_result.truncation_applied = True
            combined_result.truncated_content = self._build_prompt_template_from_content(processed_content)
            
            # Re-validate size after truncation and update the result
            post_truncation_size = self.validate_content_size(processed_content)
            combined_result.total_tokens = post_truncation_size.total_tokens
            combined_result.content_size_bytes = post_truncation_size.content_size_bytes
            combined_result.size_within_limits = post_truncation_size.size_within_limits
            
            # Update status based on post-truncation validation
            if post_truncation_size.size_within_limits:
                # After truncation, content fits within limits
                if post_truncation_size.total_tokens > self.config.max_static_content_tokens * 0.9:
                    # Still close to limit after truncation, keep as warning
                    combined_result.status = ValidationStatus.WARNING
                    if "Content truncated and approaching size limit" not in combined_result.warnings:
                        combined_result.warnings.append("Content truncated and approaching size limit")
                else:
                    # Content is now within safe limits after truncation
                    # If the final result is well within limits, we can consider it valid
                    # even though truncation was applied
                    combined_result.status = ValidationStatus.VALID
                    # Keep a note about truncation in warnings but don't make it an error
                    if "Content was truncated to fit within size limits" not in combined_result.warnings:
                        combined_result.warnings.append("Content was truncated to fit within size limits")
                # Clear size-related errors since truncation resolved them
                combined_result.errors = [
                    error for error in combined_result.errors 
                    if "size" not in error.lower()
                ]
            else:
                # Still exceeds limits even after truncation - this shouldn't happen with our algorithm
                combined_result.status = ValidationStatus.SIZE_EXCEEDED
        
        # Attach validation result to content
        processed_content.validation_result = combined_result
        
        return processed_content
    
    def _build_prompt_template_from_content(self, static_content: StaticContent) -> str:
        """Build prompt template from static content."""
        return f"""# SYSTEM IDENTITY & ROLE
{static_content.system_identity}

# REFUND POLICY DOCUMENTS
{static_content.refund_policies}

# TOOL DEFINITIONS
{static_content.tool_definitions}

# FEW-SHOT EXAMPLES
{static_content.few_shot_examples}

# CURRENT REQUEST
{{dynamic_content}}"""
