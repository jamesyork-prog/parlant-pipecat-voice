"""
Vehicle classification tool for matching customer vehicles against location restrictions.

This module uses Gemini LLM to classify vehicles into standard categories and
compare them against location-specific vehicle restrictions to determine if
a customer was legitimately turned away.
"""

import os
import json
import logging
from typing import Dict, Optional, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


class VehicleClassifier:
    """
    Classifies vehicles and compares against location restrictions.
    
    Uses Gemini LLM to understand vehicle types and determine if they match
    location restrictions extracted from booking data.
    """
    
    # Standard vehicle categories
    VEHICLE_CATEGORIES = [
        "sedan",
        "compact_suv",      # Crossovers, small SUVs (RAV4, CRV, RDX, etc.)
        "midsize_suv",      # Medium SUVs (Explorer, Highlander, etc.)
        "large_suv",        # Full-size SUVs (Suburban, Escalade, Expedition, etc.)
        "pickup_truck",     # All pickup trucks
        "van",              # Minivans and cargo vans
        "tesla",            # Tesla vehicles (any model)
        "sports_car",
        "coupe",
        "hatchback",
        "wagon",
        "motorcycle",
        "other"
    ]
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the VehicleClassifier with Gemini model.
        
        Args:
            model_name: Gemini model to use (defaults to GEMINI_MODEL env var)
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    async def check_vehicle_restriction_mismatch(
        self,
        vehicle_make_model: str,
        location_restrictions: str,
        ticket_description: str
    ) -> Dict:
        """
        Check if a vehicle was incorrectly turned away based on location restrictions.
        
        Args:
            vehicle_make_model: Customer's vehicle (e.g., "Acura RDX", "Ford F-150")
            location_restrictions: Location's restriction text from Zapier note
            ticket_description: Customer's description of what happened
        
        Returns:
            Dict containing:
                - is_mismatch (bool): True if vehicle was incorrectly rejected
                - vehicle_category (str): Classified vehicle category
                - restricted_categories (List[str]): Categories restricted by location
                - reasoning (str): Explanation of the decision
                - confidence (str): "high" | "medium" | "low"
        """
        logger.info(f"Checking vehicle restriction mismatch for: {vehicle_make_model}")
        
        # Create structured prompt
        prompt = self._create_classification_prompt(
            vehicle_make_model,
            location_restrictions,
            ticket_description
        )
        
        try:
            # Call Gemini with structured output
            response = await self._call_gemini(prompt)
            
            logger.info(f"Vehicle classification result: {response.get('vehicle_category')}, "
                       f"Mismatch: {response.get('is_mismatch')}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error classifying vehicle: {type(e).__name__}: {e}")
            return {
                "is_mismatch": False,
                "vehicle_category": "unknown",
                "restricted_categories": [],
                "reasoning": f"Unable to classify vehicle due to error: {str(e)}",
                "confidence": "low",
                "error": str(e)
            }
    
    def _create_classification_prompt(
        self,
        vehicle_make_model: str,
        location_restrictions: str,
        ticket_description: str
    ) -> str:
        """Create a structured prompt for vehicle classification."""
        return f"""You are analyzing whether a customer was incorrectly turned away from a parking location due to vehicle restrictions.

**Customer's Vehicle:**
{vehicle_make_model}

**Location's Stated Restrictions:**
{location_restrictions}

**Customer's Description:**
{ticket_description}

**Your Task:**
1. Classify the customer's vehicle into ONE of these categories:
   - sedan
   - compact_suv (crossovers, small SUVs like RAV4, CRV, RDX, NX, Q3, X3)
   - midsize_suv (medium SUVs like Explorer, Highlander, Pilot, Grand Cherokee)
   - large_suv (full-size SUVs like Suburban, Escalade, Expedition, Tahoe, Yukon)
   - pickup_truck (all pickup trucks: F-150, Silverado, Ram, Tacoma, etc.)
   - van (minivans and cargo vans: Odyssey, Sienna, Pacifica, Transit, Sprinter)
   - tesla (any Tesla model)
   - sports_car
   - coupe
   - hatchback
   - wagon
   - motorcycle
   - other

2. Extract which vehicle categories are ACTUALLY restricted by the location from the restriction text.

3. Determine if there's a MISMATCH:
   - Set is_mismatch to TRUE if the customer's vehicle category is NOT in the restricted categories
   - Set is_mismatch to FALSE if the customer's vehicle category IS in the restricted categories

4. Provide clear reasoning explaining:
   - What category the vehicle belongs to
   - What categories are restricted
   - Whether the customer was correctly or incorrectly turned away

**Important Guidelines:**
- Be precise about SUV sizes: "crossover" and "compact SUV" are NOT the same as "large SUV"
- If restrictions say "large SUVs" but not "compact SUVs", a compact SUV should NOT be restricted
- If restrictions say "SUVs" without size qualifier, assume ALL SUVs are restricted
- Tesla is often listed separately from other vehicle types
- Use "high" confidence when the classification is clear and unambiguous
- Use "medium" confidence when there's some ambiguity in vehicle size
- Use "low" confidence when vehicle information is unclear or incomplete

Return your analysis as JSON."""
    
    async def _call_gemini(self, prompt: str) -> Dict:
        """Call Gemini API with structured output schema."""
        import asyncio
        
        response = await asyncio.wait_for(
            asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {
                            "vehicle_category": {
                                "type": "string",
                                "enum": self.VEHICLE_CATEGORIES
                            },
                            "restricted_categories": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "is_mismatch": {"type": "boolean"},
                            "reasoning": {"type": "string"},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"]
                            }
                        },
                        "required": ["vehicle_category", "restricted_categories", "is_mismatch", "reasoning", "confidence"]
                    }
                )
            ),
            timeout=10.0
        )
        
        return json.loads(response.text)
    
    def extract_vehicle_from_ticket(self, ticket_text: str) -> Optional[str]:
        """
        Extract vehicle make/model from ticket text.
        
        Args:
            ticket_text: Full ticket text including Zapier note
        
        Returns:
            Vehicle make/model string or None if not found
        """
        import re
        
        # Look for "Make and Model:" field in Zapier note
        # The format is: "Make and Model: Acura RDX Were you able to park? No"
        # We need to extract only the vehicle, not the following fields
        match = re.search(r'Make and Model:\s*([A-Za-z0-9\s\-]+?)(?:\s+Were you able|\s+Proof of|\n|$)', ticket_text, re.IGNORECASE)
        if match:
            vehicle = match.group(1).strip()
            if vehicle and vehicle.lower() not in ['n/a', 'na', 'none', '']:
                return vehicle
        
        return None
    
    def extract_location_restrictions(self, ticket_text: str) -> Optional[str]:
        """
        Extract location vehicle restrictions from Zapier note.
        
        Args:
            ticket_text: Full ticket text including Zapier note
        
        Returns:
            Location restriction text or None if not found
        """
        import re
        
        # Look for "Location Description:" section which often contains restrictions
        # Example: "This location cannot accept Tesla, large SUVs, pickup trucks, or Vans."
        match = re.search(
            r'Location Description:\s*(.+?)(?:Location Admin Notes:|$)',
            ticket_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if match:
            description = match.group(1).strip()
            # Check if it mentions vehicle restrictions
            if any(keyword in description.lower() for keyword in ['cannot accept', 'not accept', 'no ', 'restrictions']):
                return description
        
        # Fallback: look for any sentence mentioning vehicle restrictions
        restriction_match = re.search(
            r'([^.]*(?:cannot accept|not accept|no tesla|no suv|no truck|no van)[^.]*\.)',
            ticket_text,
            re.IGNORECASE
        )
        
        if restriction_match:
            return restriction_match.group(1).strip()
        
        return None
