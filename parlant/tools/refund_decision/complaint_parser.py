"""
ComplaintParser component for extracting and categorizing complaint types from customer text.

This module provides the ComplaintParser class that parses customer complaint text
and extracts all complaint types with their associated keywords and confidence levels.
"""

import re
import logging
from typing import List
from dataclasses import dataclass


@dataclass
class ComplaintType:
    """Represents a categorized customer complaint."""
    category: str  # "accessibility", "paid_again", "oversold", etc.
    keywords: List[str]  # Keywords that triggered this classification
    confidence: float  # Confidence in this classification (0.0-1.0)
    source_text: str  # Original text that contained these keywords

# Configure logger
logger = logging.getLogger(__name__)


class ComplaintParser:
    """
    Extracts and categorizes complaint types from customer input.
    
    This class analyzes customer complaint text to identify and categorize
    different types of complaints (accessibility, paid_again, oversold, etc.)
    with confidence levels and source text tracking.
    """
    
    def __init__(self):
        """Initialize the ComplaintParser with complaint type definitions."""
        # Define complaint type patterns with keywords and confidence weights
        self.complaint_patterns = {
            "accessibility": {
                "keywords": [
                    "road closed", "street closed", "blocked", "police block",
                    "construction", "parade", "barricade", "can't access",
                    "couldn't access", "unable to access", "no access", "blocked off",
                    "road closure", "detour", "emergency", "road was closed"
                ],
                "high_confidence": ["road closed", "street closed", "road closure", "police block"],
                "exclusions": []  # Keywords that negate this complaint type
            },
            "paid_again": {
                "keywords": [
                    "paid again", "pay again", "made me pay", "charged again",
                    "paid twice", "charged twice", "double charge", "double billed",
                    "charged at gate", "paid onsite", "paid on-site", "had to pay"
                ],
                "high_confidence": ["paid again", "pay again", "charged twice", "double billed"],
                "exclusions": ["overstay", "over stay", "exceeded", "extra time", "additional"]
            },
            "oversold": {
                "keywords": [
                    "oversold", "no space", "no spots", "at capacity",
                    "garage full", "lot full", "sold out", "facility was full",
                    "parking was full", "lot was full", "full capacity"
                ],
                "high_confidence": ["oversold", "at capacity", "sold out"],
                "exclusions": ["were you able to park? yes", "were you able to park?yes"]
            },
            "vehicle_restriction": {
                "keywords": [
                    "didn't allow", "don't allow", "not allowed", "wouldn't allow",
                    "turned away", "rejected", "denied entry", "vehicle type",
                    "crossover", "suv", "truck", "van", "sedan", "vehicle restrictions"
                ],
                "high_confidence": ["turned away", "didn't allow", "vehicle restrictions"],
                "exclusions": ["facility was full", "at capacity", "no space"]
            },
            "operational_failure": {
                "keywords": [
                    "no attendant", "missing attendant", "attendant not there",
                    "facility closed", "location closed", "gate closed",
                    "gate wouldn't open", "couldn't get in", "no one there",
                    "missing amenity", "no ev charger", "no handicap access"
                ],
                "high_confidence": ["no attendant", "facility closed", "gate wouldn't open"],
                "exclusions": []
            },
            "location_finding": {
                "keywords": [
                    "could not find location", "couldn't find location", "unable to find location",
                    "can't find location", "could not locate", "couldn't locate",
                    "went round and round", "driving around", "drove around",
                    "no signs", "no signage", "poor signage", "bad signage"
                ],
                "high_confidence": ["could not find location", "couldn't find location", "no signage"],
                "exclusions": []
            }
        }
        logger.info("ComplaintParser initialized with complaint patterns")
    
    def parse_complaints(self, customer_complaint: str) -> List[ComplaintType]:
        """
        Parse customer complaint text and extract all complaint types.
        
        This method analyzes the customer's complaint text to identify all
        complaint types present, along with their confidence levels and
        the specific text that triggered each classification.
        
        Args:
            customer_complaint: Customer's complaint text (lowercase)
        
        Returns:
            List of ComplaintType objects with categories and keywords
        """
        if not customer_complaint:
            logger.debug("Empty customer complaint provided")
            return []
        
        complaint_text = customer_complaint.lower().strip()
        logger.debug(f"Parsing complaint text: {complaint_text[:100]}...")
        
        detected_complaints = []
        
        for category, pattern_info in self.complaint_patterns.items():
            keywords = pattern_info["keywords"]
            high_confidence_keywords = pattern_info["high_confidence"]
            exclusions = pattern_info["exclusions"]
            
            # Check for exclusion keywords first
            if any(exclusion in complaint_text for exclusion in exclusions):
                logger.debug(f"Skipping {category} due to exclusion keywords")
                continue
            
            # Special handling for accessibility + oversold context
            if category == "accessibility":
                # If complaint contains both "blocked" and facility full indicators,
                # treat this as oversold rather than accessibility
                has_blocked = "blocked" in complaint_text
                has_full_indicators = any(indicator in complaint_text for indicator in [
                    "facility was full", "garage full", "lot full", "parking was full",
                    "full capacity", "at capacity", "oversold", "sold out", "garage was full"
                ])
                
                if has_blocked and has_full_indicators:
                    logger.debug("Skipping accessibility classification - 'blocked' appears in context of facility being full")
                    continue
            
            # Find matching keywords
            matched_keywords = []
            matched_text_parts = []
            
            for keyword in keywords:
                if keyword in complaint_text:
                    matched_keywords.append(keyword)
                    # Extract the sentence or phrase containing the keyword
                    sentences = re.split(r'[.!?]', complaint_text)
                    for sentence in sentences:
                        if keyword in sentence:
                            matched_text_parts.append(sentence.strip())
                            break
            
            if matched_keywords:
                # Calculate confidence based on keyword types
                has_high_confidence = any(kw in high_confidence_keywords for kw in matched_keywords)
                confidence = 0.9 if has_high_confidence else 0.7
                
                # Combine matched text parts
                source_text = "; ".join(set(matched_text_parts))
                
                complaint = ComplaintType(
                    category=category,
                    keywords=matched_keywords,
                    confidence=confidence,
                    source_text=source_text
                )
                
                detected_complaints.append(complaint)
                logger.debug(f"Detected {category} complaint with confidence {confidence}")
        
        logger.info(f"Parsed {len(detected_complaints)} complaint types from customer text")
        return detected_complaints