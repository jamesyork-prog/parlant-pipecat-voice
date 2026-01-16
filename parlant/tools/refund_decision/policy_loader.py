"""
PolicyLoader component for loading and caching refund policy documents.

This module provides a singleton PolicyLoader class that loads policy documents
from the parlant/context/processed/ directory and caches them in memory to avoid
repeated file reads.
"""

import json
import os
import logging
from typing import Dict, Optional

# Configure logger
logger = logging.getLogger(__name__)


class PolicyLoader:
    """
    Loads and caches refund policy documents from JSON and Markdown files.
    
    This class implements a singleton pattern to ensure policy documents are
    loaded only once and cached for subsequent access.
    """
    
    _instance: Optional['PolicyLoader'] = None
    _initialized: bool = False
    
    def __new__(cls, context_dir: str = "parlant/context/processed"):
        """Ensure only one instance of PolicyLoader exists (singleton pattern)."""
        if cls._instance is None:
            cls._instance = super(PolicyLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, context_dir: str = "parlant/context/processed"):
        """
        Initialize the PolicyLoader with the context directory.
        
        Args:
            context_dir: Path to the directory containing policy files
        """
        # Only initialize once (singleton pattern)
        if PolicyLoader._initialized:
            return
        
        # Determine the correct base path
        # If running in Docker, the path is /app/app_tools/context/processed
        # If running locally, the path is parlant/context/processed
        if os.path.exists("/app/app_tools/context/processed"):
            self.context_dir = "/app/app_tools/context/processed"
        elif os.path.exists("context/processed"):
            self.context_dir = "context/processed"
        else:
            self.context_dir = context_dir
        
        self.rules: Dict = {}
        self.guide: Dict = {}
        self.scenarios: str = ""
        self.decision_chart: str = ""
        
        # Load policies on initialization
        self.load_policies()
        PolicyLoader._initialized = True
    
    def load_policies(self) -> None:
        """
        Load all policy documents from the context directory into memory.
        
        Loads the following files:
        - refund_rules.json: Business rules for refund scenarios
        - refund_guide.json: Detailed refund processing guide
        - refund_scenario_decision_chart.md: Decision tree for refund scenarios
        - ai_vs_human_refund_scenarios.md: Guidance on AI vs human handling
        
        Raises:
            FileNotFoundError: If required policy files are missing
        """
        logger.info(f"Loading policy documents from: {self.context_dir}")
        
        # Load refund_rules.json
        rules_path = os.path.join(self.context_dir, "refund_rules.json")
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Replace curly quotes with straight quotes
                content = content.replace('"', '"').replace('"', '"')
                content = content.replace(''', "'").replace(''', "'")
                try:
                    self.rules = json.loads(content)
                    logger.info(f"Successfully loaded and parsed refund_rules.json")
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, store as raw text for LLM consumption
                    logger.warning(f"Failed to parse refund_rules.json as JSON: {e}. Storing as raw content.")
                    self.rules = {"raw_content": content, "parsed": False}
        except FileNotFoundError:
            logger.error(f"Required policy file not found: {rules_path}")
            raise FileNotFoundError(f"Required policy file not found: {rules_path}")
        
        # Load refund_guide.json
        guide_path = os.path.join(self.context_dir, "refund_guide.json")
        try:
            with open(guide_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Replace curly quotes with straight quotes
                content = content.replace('"', '"').replace('"', '"')
                content = content.replace(''', "'").replace(''', "'")
                try:
                    self.guide = json.loads(content)
                    logger.info(f"Successfully loaded and parsed refund_guide.json")
                except json.JSONDecodeError as e:
                    # If JSON parsing fails, store as raw text for LLM consumption
                    logger.warning(f"Failed to parse refund_guide.json as JSON: {e}. Storing as raw content.")
                    self.guide = {"raw_content": content, "parsed": False}
        except FileNotFoundError:
            logger.error(f"Required policy file not found: {guide_path}")
            raise FileNotFoundError(f"Required policy file not found: {guide_path}")
        
        # Load refund_scenario_decision_chart.md
        decision_chart_path = os.path.join(self.context_dir, "refund_scenario_decision_chart.md")
        try:
            with open(decision_chart_path, 'r', encoding='utf-8') as f:
                self.decision_chart = f.read()
                logger.info(f"Successfully loaded refund_scenario_decision_chart.md ({len(self.decision_chart)} chars)")
        except FileNotFoundError:
            logger.error(f"Required policy file not found: {decision_chart_path}")
            raise FileNotFoundError(f"Required policy file not found: {decision_chart_path}")
        
        # Load ai_vs_human_refund_scenarios.md
        scenarios_path = os.path.join(self.context_dir, "ai_vs_human_refund_scenarios.md")
        try:
            with open(scenarios_path, 'r', encoding='utf-8') as f:
                self.scenarios = f.read()
                logger.info(f"Successfully loaded ai_vs_human_refund_scenarios.md ({len(self.scenarios)} chars)")
        except FileNotFoundError:
            logger.error(f"Required policy file not found: {scenarios_path}")
            raise FileNotFoundError(f"Required policy file not found: {scenarios_path}")
        
        logger.info("All policy documents loaded successfully")
    
    def get_rules(self) -> Dict:
        """
        Return parsed refund rules from refund_rules.json.
        
        Returns:
            Dict containing the refund rules data structure
        """
        return self.rules
    
    def get_condensed_policy_text(self) -> str:
        """
        Return condensed policy text optimized for LLM context.
        
        Returns the streamlined policy document that contains only essential
        decision-making information without verbose examples and procedures.
        This is the RECOMMENDED method for LLM prompts.
        
        Returns:
            str: Condensed policy text (~5KB instead of ~90KB)
        """
        condensed_path = os.path.join(self.context_dir, "refund_policy_condensed.md")
        try:
            with open(condensed_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            # Fallback to full policy if condensed version doesn't exist
            return self.get_full_policy_text()
    
    def get_full_policy_text(self) -> str:
        """
        Return combined policy text for LLM context.
        
        WARNING: This returns ~90KB of text (~23K tokens). Use get_condensed_policy_text() 
        instead for better performance and lower token usage.
        
        Returns:
            str: Combined policy text from all documents
        """
        policy_text = []
        
        # Add refund guide
        policy_text.append("# Refund and Credits Guide")
        policy_text.append("")
        if isinstance(self.guide, dict):
            # Check if it's unparsed raw content
            if "raw_content" in self.guide and self.guide.get("parsed") == False:
                policy_text.append(self.guide["raw_content"])
            else:
                # Parsed JSON structure
                if "title" in self.guide:
                    policy_text.append(f"## {self.guide['title']}")
                    policy_text.append("")
                if "introduction" in self.guide:
                    policy_text.append(self.guide['introduction'])
                    policy_text.append("")
                if "sections" in self.guide:
                    for section in self.guide['sections']:
                        if "title" in section:
                            policy_text.append(f"### {section['title']}")
                            policy_text.append("")
                        if "content" in section:
                            policy_text.append(section['content'])
                            policy_text.append("")
        
        # Add refund rules
        policy_text.append("# Refund Rules and Scenarios")
        policy_text.append("")
        if isinstance(self.rules, dict) and "raw_content" in self.rules:
            # Unparsed raw content
            policy_text.append(self.rules["raw_content"])
        elif isinstance(self.rules, list):
            # Parsed JSON structure
            for rule in self.rules:
                if "Scenario" in rule:
                    policy_text.append(f"## {rule['Scenario']}")
                    policy_text.append("")
                if "Trigger/Condition" in rule and rule["Trigger/Condition"]:
                    policy_text.append(f"**Trigger/Condition:** {rule['Trigger/Condition']}")
                    policy_text.append("")
                if "Checks" in rule and rule["Checks"]:
                    policy_text.append(f"**Checks:** {rule['Checks']}")
                    policy_text.append("")
                if "Action" in rule and rule["Action"]:
                    policy_text.append(f"**Action:** {rule['Action']}")
                    policy_text.append("")
                if "Recognition Phrases/Keywords" in rule and rule["Recognition Phrases/Keywords"]:
                    policy_text.append(f"**Keywords:** {rule['Recognition Phrases/Keywords']}")
                    policy_text.append("")
                if "Refund Reason/Settings" in rule and rule["Refund Reason/Settings"]:
                    policy_text.append(f"**Refund Settings:** {rule['Refund Reason/Settings']}")
                    policy_text.append("")
        
        # Add decision chart
        policy_text.append("# Refund Scenario Decision Chart")
        policy_text.append("")
        policy_text.append(self.decision_chart)
        policy_text.append("")
        
        # Add AI vs Human scenarios
        policy_text.append("# AI vs Human Refund Scenarios")
        policy_text.append("")
        policy_text.append(self.scenarios)
        
        return "\n".join(policy_text)
