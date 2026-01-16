"""
BookingExtractor component for extracting structured booking information from ticket text.

This module uses a hybrid approach: first attempting fast pattern-based extraction,
then falling back to Gemini LLM for complex or unstructured ticket notes.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from google import genai
from google.genai import types
from .patterns import PatternExtractor
from .timeline_analyzer import TimelineAnalyzer, TimelineExtractor
from ..integrations.parkwhiz import ParkWhizOAuth2Client, ParkWhizNotFoundError, ParkWhizError
from ..refund_decision.booking_mismatch_detector import BookingMismatchDetector

# Configure logger
logger = logging.getLogger(__name__)


class BookingExtractor:
    """
    Extracts structured booking information from ticket text using hybrid approach.
    
    This class first attempts fast pattern-based extraction using regex and HTML parsing.
    If pattern extraction fails or has low confidence, it falls back to Gemini LLM
    for more sophisticated analysis.
    """
    
    def __init__(self, model_name: Optional[str] = None, use_pattern_fallback: bool = True, timeline_config: Optional[Dict] = None):
        """
        Initialize the BookingExtractor with Gemini model, pattern extractor, and timeline analysis.
        
        Args:
            model_name: Gemini model to use (defaults to GEMINI_MODEL env var or gemini-2.5-flash)
            use_pattern_fallback: Whether to try pattern-based extraction first (default: True)
            timeline_config: Configuration for timeline analysis (overstay/early arrival tolerances)
        """
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required. "
                "Please add it to your .env file."
            )
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=api_key)
        
        # Use provided model or fall back to environment variable or default
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        
        # Initialize pattern extractor for performance optimization
        self.use_pattern_fallback = use_pattern_fallback
        self.pattern_extractor = PatternExtractor() if use_pattern_fallback else None
        
        # Initialize timeline analysis components
        self.timeline_analyzer = TimelineAnalyzer(timeline_config)
        self.timeline_extractor = TimelineExtractor()
        
        # Initialize booking mismatch detector
        self.booking_mismatch_detector = BookingMismatchDetector(timeline_config)
        
        logger.info("BookingExtractor initialized with timeline analysis and booking mismatch detection")
    
    async def extract_booking_info(self, ticket_notes: str) -> Dict:
        """
        Extract booking information from ticket text using hybrid approach.
        
        First attempts fast pattern-based extraction. If that fails or has low
        confidence, falls back to Gemini LLM for more sophisticated analysis.
        
        Args:
            ticket_notes: Raw text from ticket notes/conversations
            
        Returns:
            Dict containing:
                - booking_info: Dict with extracted fields (booking_id, amount, dates, etc.)
                - confidence: "high" | "medium" | "low"
                - found: bool indicating if booking info was found
                - extraction_method: "pattern" | "llm"
        """
        logger.info("Starting booking information extraction")
        
        if not ticket_notes or not ticket_notes.strip():
            logger.warning("Empty ticket notes provided for extraction")
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "none"
            }
        
        logger.debug(f"Ticket notes length: {len(ticket_notes)} characters")
        
        # Try pattern-based extraction first (performance optimization)
        if self.use_pattern_fallback and self.pattern_extractor:
            logger.info("Attempting pattern-based extraction")
            pattern_result = self._try_pattern_extraction(ticket_notes)
            
            # If pattern extraction succeeded with medium or high confidence, use it
            if pattern_result["found"] and pattern_result["confidence"] in ["medium", "high"]:
                logger.info(f"Pattern extraction succeeded with {pattern_result['confidence']} confidence. "
                           f"Found: {list(pattern_result['booking_info'].keys())}")
                
                # CRITICAL: Always verify booking with API if booking ID found
                if pattern_result.get("booking_info", {}).get("booking_id"):
                    logger.info("Pattern extraction found booking ID, running API verification")
                    pattern_result = await self._verify_booking_with_api(pattern_result)
                
                # Post-process: Check for pass usage indicators
                pass_used = self._check_pass_usage_indicators(ticket_notes, pattern_result.get("booking_info", {}))
                if pass_used is not None:
                    pattern_result["pass_used"] = pass_used
                    logger.info(f"Pass usage determination: {'USED' if pass_used else 'NOT USED'}")
                
                # Post-process: Extract and analyze timeline data
                pattern_result = self._extract_and_analyze_timeline(pattern_result, ticket_notes)
                
                return pattern_result
            
            # If pattern extraction found some data but low confidence, we'll still try LLM
            # but can use the pattern result as a fallback if LLM fails
            logger.info(f"Pattern extraction had {pattern_result['confidence']} confidence, falling back to LLM")
        
        # Fall back to LLM extraction
        logger.info("Using LLM extraction")
        result = await self._extract_with_llm(ticket_notes)
        
        # Post-process: Fix amount if "Booking Base Price" is explicitly mentioned
        result = self._fix_booking_amount(result, ticket_notes)
        
        # Post-process: Validate extracted dates for common errors
        result = self._validate_extracted_dates(result, ticket_notes)
        
        # Post-process: Verify booking ID with ParkWhiz API if found
        result = await self._verify_booking_with_api(result)
        
        # Post-process: Check for pass usage indicators
        pass_used = self._check_pass_usage_indicators(ticket_notes, result.get("booking_info", {}))
        if pass_used is not None:
            result["pass_used"] = pass_used
            logger.info(f"Pass usage determination: {'USED' if pass_used else 'NOT USED'}")
        
        # Post-process: Extract and analyze timeline data
        result = self._extract_and_analyze_timeline(result, ticket_notes)
        
        return result
    
    def _try_pattern_extraction(self, ticket_notes: str) -> Dict:
        """
        Attempt pattern-based extraction from ticket text.
        
        Args:
            ticket_notes: Raw text from ticket notes
            
        Returns:
            Dict with extraction results
        """
        try:
            # Check if content looks like HTML
            if '<' in ticket_notes and '>' in ticket_notes:
                logger.debug("Detected HTML content, using HTML extraction")
                return self.pattern_extractor.extract_from_html(ticket_notes)
            else:
                logger.debug("Detected plain text content, using text extraction")
                return self.pattern_extractor.extract_from_text(ticket_notes)
        except Exception as e:
            logger.error(f"Pattern extraction error: {type(e).__name__}: {e}")
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "pattern"
            }
    
    async def _extract_with_llm(self, ticket_notes: str) -> Dict:
        """
        Extract booking information using Gemini LLM with timeout and error handling.
        
        Args:
            ticket_notes: Raw text from ticket notes
            
        Returns:
            Dict with extraction results
        """
        import asyncio
        
        # Create structured prompt for extraction
        prompt = self._create_extraction_prompt(ticket_notes)
        
        logger.debug(f"Calling Gemini API with model: {self.model_name}")
        
        try:
            # Make Gemini API call with JSON schema and 10-second timeout
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistent extraction
                        response_mime_type="application/json",
                        response_schema={
                            "type": "object",
                            "properties": {
                                "booking_id": {"type": "string"},
                                "amount": {"type": "number"},
                                "reservation_date": {"type": "string"},
                                "event_date": {"type": "string"},
                                "location": {"type": "string"},
                                "booking_type": {"type": "string"},
                                "customer_email": {"type": "string"},
                                "cancellation_date": {"type": "string"},
                                "is_reseller": {"type": "boolean"},
                                "is_non_refundable": {"type": "boolean"},
                                "arrival_time": {"type": "string"},
                                "exit_time": {"type": "string"},
                                "booking_window_start": {"type": "string"},
                                "booking_window_end": {"type": "string"},
                                "found": {"type": "boolean"},
                                "multiple_bookings": {"type": "boolean"},
                                "booking_not_found": {"type": "boolean"}
                            },
                            "required": ["found"]
                        }
                    )
                ),
                timeout=10.0  # 10-second timeout
            )
            
            # Parse response
            result = json.loads(response.text)
            
            # Calculate confidence based on completeness
            confidence = self._calculate_confidence(result)
            
            # Extract booking info (remove metadata fields)
            booking_info = {
                k: v for k, v in result.items() 
                if k not in ["found", "multiple_bookings"] and v is not None
            }
            
            logger.info(f"LLM extraction completed. Found: {result.get('found', False)}, "
                       f"Confidence: {confidence}, Fields: {list(booking_info.keys())}")
            
            if result.get("multiple_bookings"):
                logger.warning("Multiple bookings detected in ticket")
            
            return {
                "booking_info": booking_info,
                "confidence": confidence,
                "found": result.get("found", False),
                "extraction_method": "llm",
                "multiple_bookings": result.get("multiple_bookings", False),
                "booking_not_found": result.get("booking_not_found", False)
            }
            
        except asyncio.TimeoutError:
            # Handle timeout - log and return empty result
            logger.error("LLM extraction timeout after 10 seconds")
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "llm",
                "error": "Timeout after 10 seconds"
            }
        
        except json.JSONDecodeError as e:
            # Handle JSON parsing errors
            logger.error(f"Error parsing LLM response: {e}")
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "llm",
                "error": f"JSON parsing error: {str(e)}"
            }
        
        except Exception as e:
            # Handle all other errors (API failures, network issues, etc.)
            error_type = type(e).__name__
            logger.error(f"Error extracting booking info ({error_type}): {e}")
            return {
                "booking_info": {},
                "confidence": "low",
                "found": False,
                "extraction_method": "llm",
                "error": f"{error_type}: {str(e)}"
            }
    
    def _create_extraction_prompt(self, ticket_notes: str) -> str:
        """
        Create a structured prompt for booking information extraction.
        
        Args:
            ticket_notes: Raw ticket text
            
        Returns:
            Formatted prompt string
        """
        return f"""Extract booking information from the following ticket notes. Look for:

1. **Booking ID**: Any reference number like "PW-12345", "509266779", "Booking #123", etc.
2. **Amount**: The ORIGINAL booking price - look for "Booking Base Price", "Booking Price", or similar. DO NOT use amounts mentioned in the customer's complaint (like "paid again $50") - those are additional charges, not the booking amount.
3. **Reservation Date**: When the booking was made
4. **Event Date**: When the parking was scheduled for (start date/time) - THIS IS THE MOST CRITICAL FIELD
5. **Location**: Parking facility name or address
6. **Booking Type**: "confirmed", "on-demand", "third-party", or "unknown"
7. **Customer Email**: Email address of the customer
8. **Cancellation Date**: When the cancellation was requested (if mentioned)
9. **Is Reseller**: Look for "Is user flagged as reseller?" - set to true if "Yes", false if "No"
10. **Is Non-Refundable**: Look for "Is user flagged as non-refundable?" - set to true if "Yes", false if "No"

**CRITICAL DATE EXTRACTION RULES:**
- Look for "Parking Pass Start Time" - this is the event_date
- Look for "Arrival:" followed by a date - this is also the event_date  
- Look for "Booking Created" - this is the reservation_date
- ALWAYS use 4-digit years (e.g., 2025, not 25)
- Convert dates to ISO format: YYYY-MM-DD (e.g., "Thursday Dec 04, 2025, 12:00 PM" → "2025-12-04")
- Pay careful attention to the year - if you see "2025", use 2025, NOT 2020
- If a date includes time (e.g., "12:00 PM"), ignore the time and extract only the date
- **CRITICAL**: COMPLETELY IGNORE billing dates, credit card dates, payment dates, or other non-parking dates
- Examples of dates to IGNORE: "Go Tickets billed 4/22/2026", "Bisa Ending on 7360", "charged on 3/15/2026"
- The event_date should ONLY be the date when parking was scheduled to START
- If you see multiple dates, prioritize "Arrival:" and "Parking Pass Start Time" over any billing/payment dates

**BOOKING ID VALIDATION:**
- If the booking ID appears to be invalid or the notes mention "Booking information not found", set a special flag
- Look for phrases like "Booking information not found for provided Booking Number"
- This indicates the customer may have provided wrong booking details

**Important Instructions:**
- If multiple bookings are mentioned, extract information for the PRIMARY booking being disputed
- Set "found" to true if you find at least a booking ID or event date
- Set "multiple_bookings" to true if more than one booking is referenced
- Set "booking_not_found" to true if the notes indicate the booking ID was not found in the system
- Use ISO format (YYYY-MM-DD) for dates - ALWAYS include the full 4-digit year
- If a field is not found, omit it from the response (don't include null values)
- For booking_type, infer from context: "confirmed" for advance bookings, "on-demand" for same-day, "third-party" if booked through another platform

**TIMELINE EXTRACTION (NEW):**
- Look for arrival times: "Arrival:", "Arrived:", "Entry Time:", "Check-in:", "Started parking:"
- Look for exit times: "Exit:", "Departure:", "Left:", "Check-out:", "Finished parking:"
- Look for booking window: "Booking Start:", "Reserved from:", "Pass valid from/until:"
- Extract these as separate fields if found, using the same date format rules as above

**Ticket Notes:**
{ticket_notes}

Extract the booking information as JSON. Include timeline fields if found:
- arrival_time: When customer actually arrived (if mentioned)
- exit_time: When customer actually left (if mentioned)
- booking_window_start: When booking period starts (if different from event_date)
- booking_window_end: When booking period ends (if mentioned)"""
    
    def _fix_booking_amount(self, result: Dict, ticket_notes: str) -> Dict:
        """
        Post-process extraction result to fix booking amount if needed.
        
        The LLM sometimes extracts amounts from customer complaints (e.g., "paid again $50")
        instead of the actual booking price. This method looks for explicit "Booking Base Price"
        or "Booking Price" mentions and uses those instead.
        
        Args:
            result: Extraction result from LLM
            ticket_notes: Original ticket notes
            
        Returns:
            Updated result with corrected amount
        """
        import re
        
        # Look for "Booking Base Price: $XXX" or "Booking Price: $XXX"
        price_patterns = [
            r'Booking Base Price:\s*\$?(\d+(?:\.\d{2})?)',
            r'Booking Price:\s*\$?(\d+(?:\.\d{2})?)',
            r'Base Price:\s*\$?(\d+(?:\.\d{2})?)',
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, ticket_notes, re.IGNORECASE)
            if match:
                base_price = float(match.group(1))
                current_amount = result.get("booking_info", {}).get("amount")
                
                # If we found a base price and it's different from what was extracted, use it
                if current_amount != base_price:
                    logger.info(f"Correcting booking amount: ${current_amount} → ${base_price} (found 'Booking Base Price' in notes)")
                    if "booking_info" not in result:
                        result["booking_info"] = {}
                    result["booking_info"]["amount"] = base_price
                break
        
        return result
    
    def _calculate_confidence(self, result: Dict) -> str:
        """
        Calculate confidence level based on completeness of extracted data.
        
        Args:
            result: Extracted booking information
            
        Returns:
            "high" | "medium" | "low"
        """
        if not result.get("found", False):
            return "low"
        
        # Count critical fields present
        critical_fields = ["booking_id", "event_date"]
        critical_count = sum(1 for field in critical_fields if result.get(field))
        
        # Count optional fields present
        optional_fields = ["amount", "reservation_date", "location", "booking_type", "customer_email"]
        optional_count = sum(1 for field in optional_fields if result.get(field))
        
        # High confidence: both critical fields + at least 3 optional fields
        if critical_count == 2 and optional_count >= 3:
            return "high"
        
        # Medium confidence: both critical fields (even without optional fields)
        elif critical_count == 2:
            return "medium"
        
        # Medium confidence: 1 critical field + at least 3 optional fields
        elif critical_count == 1 and optional_count >= 3:
            return "medium"
        
        # Low confidence: missing critical fields or very incomplete
        else:
            return "low"
    
    def _check_pass_usage_indicators(self, ticket_notes: str, booking_info: Dict) -> Optional[bool]:
        """
        Check for indicators in ticket notes that suggest whether the pass was used.
        
        This method looks for language patterns that indicate pass usage vs non-usage.
        In a full implementation, this would integrate with ParkWhiz Admin API.
        
        Args:
            ticket_notes: Raw text from ticket notes
            booking_info: Extracted booking information
            
        Returns:
            True if pass appears to have been used, False if not used, None if unclear
        """
        import re
        
        if not ticket_notes:
            return None
        
        # Convert to lowercase for pattern matching
        notes_lower = ticket_notes.lower()
        
        # Patterns that suggest the pass WAS used (customer got parking value)
        used_patterns = [
            r'successfully entered',
            r'got in.*but.*exit',
            r'parked.*but.*leaving',
            r'entered.*garage.*but',
            r'used.*pass.*to.*enter',
            r'scanned.*in.*but',
            r'got.*parking.*but',
            r'parked.*successfully.*but',
            r'entered.*fine.*but.*exit'
        ]
        
        # Patterns that suggest the pass was NOT used (customer got no parking value)
        not_used_patterns = [
            r'pass.*rejected',
            r'attendant.*said.*invalid',
            r'wouldn\'t.*accept.*pass',
            r'refused.*to.*honor',
            r'never.*got.*to.*use',
            r'couldn\'t.*use.*pass',
            r'pass.*didn\'t.*work.*at.*all',
            r'had.*to.*pay.*instead',
            r'forced.*to.*pay.*gate.*rate'
        ]
        
        # Check for "not used" patterns first (more definitive)
        for pattern in not_used_patterns:
            if re.search(pattern, notes_lower):
                logger.info(f"Pass usage indicator: NOT USED (pattern: {pattern})")
                return False
        
        # Check for "used" patterns
        for pattern in used_patterns:
            if re.search(pattern, notes_lower):
                logger.info(f"Pass usage indicator: USED (pattern: {pattern})")
                return True
        
        # Special case: "paid again" could mean either scenario
        if re.search(r'paid.*again', notes_lower):
            # Look for additional context
            if re.search(r'paid.*again.*at.*exit|paid.*again.*to.*leave|paid.*again.*getting.*out', notes_lower):
                logger.info("Pass usage indicator: USED (paid again at exit)")
                return True
            elif re.search(r'paid.*again.*to.*park|paid.*again.*instead|had.*to.*pay.*again', notes_lower):
                logger.info("Pass usage indicator: NOT USED (paid again to park)")
                return False
        
        # Check for "had to pay to park" - this usually means they didn't use their pass
        if re.search(r'had.*to.*pay.*to.*park|pay.*to.*park.*with.*card', notes_lower):
            logger.info("Pass usage indicator: NOT USED (had to pay to park)")
            return False
        
        logger.info("Pass usage indicator: UNCLEAR (no definitive patterns found)")
        return None
    
    def _validate_extracted_dates(self, result: Dict, ticket_notes: str) -> Dict:
        """
        Validate extracted dates and fix common LLM extraction errors.
        
        Common issues:
        - LLM extracts billing dates instead of event dates
        - LLM confuses multiple dates in the ticket
        - LLM extracts future dates that are clearly wrong
        
        Args:
            result: Extraction result from LLM
            ticket_notes: Original ticket notes
            
        Returns:
            Updated result with corrected dates
        """
        import re
        from datetime import datetime, timedelta
        
        booking_info = result.get("booking_info", {})
        event_date = booking_info.get("event_date")
        
        if not event_date:
            return result
        
        try:
            # Parse the extracted event date
            event_dt = datetime.fromisoformat(event_date)
            current_dt = datetime.now()
            
            # Check if the event date is more than 3 months in the future
            # This is likely a billing date, not an event date
            if event_dt > current_dt + timedelta(days=90):
                logger.warning(f"Extracted event date {event_date} is more than 3 months in the future - likely a billing date")
                
                # Try to find the actual arrival date in the ticket
                arrival_patterns = [
                    r'Arrival:\s*(\d{1,2}/\d{1,2}/\d{4})',  # Arrival: 12/13/2025
                    r'Arrival:\s*(\d{4}-\d{2}-\d{2})',      # Arrival: 2025-12-13
                    r'Parking Pass Start Time[:\s]*([^,\n]+)',  # Parking Pass Start Time: ...
                ]
                
                for pattern in arrival_patterns:
                    match = re.search(pattern, ticket_notes, re.IGNORECASE)
                    if match:
                        arrival_text = match.group(1).strip()
                        
                        # Try to parse the arrival date
                        try:
                            # Handle MM/DD/YYYY format
                            if '/' in arrival_text:
                                parts = arrival_text.split('/')
                                if len(parts) == 3:
                                    month, day, year = parts
                                    corrected_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                            else:
                                corrected_date = arrival_text
                            
                            # Validate the corrected date
                            corrected_dt = datetime.fromisoformat(corrected_date)
                            
                            # Only use if it's within reasonable range (past 1 year to future 1 year)
                            if current_dt - timedelta(days=365) <= corrected_dt <= current_dt + timedelta(days=365):
                                logger.info(f"Correcting event date: {event_date} → {corrected_date}")
                                booking_info["event_date"] = corrected_date
                                result["booking_info"] = booking_info
                                
                                # Also update confidence if we made a correction
                                if result.get("confidence") == "low":
                                    result["confidence"] = "medium"
                                    logger.info("Upgraded confidence to medium after date correction")
                                break
                        except (ValueError, IndexError):
                            continue
                
                # If we couldn't find a better date, log a warning but keep the original
                if booking_info.get("event_date") == event_date:
                    logger.warning(f"Could not find alternative event date, keeping extracted date: {event_date}")
            
            # Additional validation: Check for obviously wrong years (like 2020 instead of 2025)
            elif event_dt.year < current_dt.year - 1:
                logger.warning(f"Extracted event date {event_date} is more than 1 year in the past - may be wrong year")
                
                # Try to find a more recent date in the ticket
                for pattern in [r'Arrival:\s*(\d{1,2}/\d{1,2}/\d{4})', r'(\d{1,2}/\d{1,2}/2025)']:
                    match = re.search(pattern, ticket_notes, re.IGNORECASE)
                    if match:
                        arrival_text = match.group(1).strip()
                        try:
                            if '/' in arrival_text:
                                parts = arrival_text.split('/')
                                if len(parts) == 3:
                                    month, day, year = parts
                                    corrected_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                                    corrected_dt = datetime.fromisoformat(corrected_date)
                                    
                                    # Use if it's more reasonable than the extracted date
                                    if corrected_dt.year >= current_dt.year - 1:
                                        logger.info(f"Correcting old event date: {event_date} → {corrected_date}")
                                        booking_info["event_date"] = corrected_date
                                        result["booking_info"] = booking_info
                                        break
                        except (ValueError, IndexError):
                            continue
            
        except (ValueError, AttributeError) as e:
            logger.error(f"Date validation error: {e}")
        
        return result
    async def _verify_booking_with_api(self, result: Dict) -> Dict:
        """
        Verify booking ID with ParkWhiz API to check if it exists in their system.
        
        This is the critical step that was missing - actually checking if the booking
        ID provided by the customer exists in the ParkWhiz system.
        
        IMPORTANT: In sandbox/testing mode, this verification is disabled to allow
        processing of real tickets without live ParkWhiz data.
        
        Args:
            result: Extraction result from LLM or pattern matching
            
        Returns:
            Updated result with booking_not_found flag set appropriately
        """
        booking_info = result.get("booking_info", {})
        booking_id = booking_info.get("booking_id")
        
        # Skip verification if no booking ID was extracted
        if not booking_id:
            logger.info("No booking ID extracted, skipping ParkWhiz API verification")
            return result
        
        # Check if we're in sandbox/testing mode
        parkwhiz_env = os.getenv("PARKWHIZ_ENV", "sandbox")
        api_verification_enabled = os.getenv("PARKWHIZ_API_VERIFICATION_ENABLED", "false").lower() == "true"
        
        if parkwhiz_env == "sandbox" and not api_verification_enabled:
            logger.info(f"ParkWhiz API verification disabled in sandbox mode for booking {booking_id}")
            logger.info("To enable verification in sandbox, set PARKWHIZ_API_VERIFICATION_ENABLED=true")
            
            # In sandbox mode without verification, we simulate successful verification
            # but apply pattern-based mismatch detection using ticket data only
            logger.info("Applying pattern-based mismatch detection without API verification")
            
            # Try to detect obvious mismatches using only the ticket data
            mismatch_detected = self._detect_pattern_based_mismatches(result, booking_id)
            
            if mismatch_detected:
                logger.warning(f"Pattern-based mismatch detected for booking {booking_id}")
                return result
            
            # Mark as verified (sandbox simulation) and upgrade confidence
            booking_info["api_verified"] = False  # False because we didn't actually verify
            booking_info["sandbox_mode"] = True
            result["booking_info"] = booking_info
            result["booking_not_found"] = False
            
            # Don't upgrade confidence in sandbox mode since we can't verify
            logger.info("Sandbox mode: skipping confidence upgrade due to lack of real API verification")
            
            return result
        
        logger.info(f"Verifying booking ID {booking_id} with ParkWhiz API (production mode)")
        
        try:
            # Initialize ParkWhiz client
            parkwhiz_client = ParkWhizOAuth2Client()
            
            # Try to get the booking from ParkWhiz API
            api_booking = await parkwhiz_client.get_booking_by_id(booking_id)
            
            logger.info(f"Booking {booking_id} found in ParkWhiz system")
            
            # Booking exists - enhance our extracted data with API data
            if api_booking:
                # Update booking info with verified data from API
                if api_booking.get("amount") and not booking_info.get("amount"):
                    booking_info["amount"] = api_booking["amount"]
                    logger.info(f"Added amount from API: ${api_booking['amount']}")
                
                if api_booking.get("event_date") and not booking_info.get("event_date"):
                    booking_info["event_date"] = api_booking["event_date"]
                    logger.info(f"Added event date from API: {api_booking['event_date']}")
                
                if api_booking.get("location") and not booking_info.get("location"):
                    booking_info["location"] = api_booking["location"]
                    logger.info(f"Added location from API: {api_booking['location']}")
                
                # CRITICAL: Check for booking mismatches
                # This detects cases where customer provided wrong booking ID
                logger.info("Checking for booking mismatches between customer claims and API data")
                
                # Prepare customer data from extracted booking info
                customer_data = {
                    "name": booking_info.get("customer_name", ""),
                    "email": booking_info.get("customer_email", ""),
                    "booking_id": booking_id,
                    "event_date": booking_info.get("event_date", ""),
                    "arrival_date": booking_info.get("arrival_time", "")
                }
                
                # Prepare system data from API response
                system_data = {
                    "user_name": api_booking.get("user_name", ""),
                    "user_email": api_booking.get("user_email", ""),
                    "booking_created": api_booking.get("booking_created", ""),
                    "pass_start": api_booking.get("pass_start", api_booking.get("event_date", "")),
                    "pass_end": api_booking.get("pass_end", ""),
                    "event_name": api_booking.get("event_name", ""),
                    "location": api_booking.get("location", "")
                }
                
                # Detect mismatches
                mismatches = self.booking_mismatch_detector.detect_mismatches(
                    customer_data, system_data
                )
                
                # Check if mismatches warrant escalation
                should_escalate, escalation_reason = self.booking_mismatch_detector.should_escalate_for_mismatch(mismatches)
                
                if should_escalate:
                    logger.warning(f"Booking mismatch detected for booking {booking_id}: {escalation_reason}")
                    
                    # Format mismatch summary for human agents
                    mismatch_summary = self.booking_mismatch_detector.format_mismatch_summary(mismatches)
                    
                    # Mark as booking mismatch (different from booking_not_found)
                    result["booking_mismatch"] = True
                    result["booking_mismatch_details"] = {
                        "mismatches": [
                            {
                                "type": m.mismatch_type,
                                "severity": m.severity,
                                "description": m.description,
                                "customer_claim": m.customer_claim,
                                "system_data": m.system_data,
                                "confidence": m.confidence
                            } for m in mismatches
                        ],
                        "escalation_reason": escalation_reason,
                        "summary": mismatch_summary
                    }
                    
                    # Set confidence to high for mismatch detection (we're confident there's a problem)
                    result["confidence"] = "high"
                    logger.info("Set confidence to high due to booking mismatch detection")
                    
                    # Don't upgrade confidence - this is a problem case
                    booking_info["api_verified"] = True
                    result["booking_info"] = booking_info
                    result["booking_not_found"] = False
                    
                    return result
                
                # No mismatches detected - proceed normally
                logger.info("No booking mismatches detected, proceeding with normal verification")
                
                # Mark as verified and upgrade confidence
                booking_info["api_verified"] = True
                result["booking_info"] = booking_info
                result["booking_not_found"] = False
                
                # Upgrade confidence since we have API verification
                if result.get("confidence") == "low":
                    result["confidence"] = "medium"
                    logger.info("Upgraded confidence to medium due to API verification")
                elif result.get("confidence") == "medium":
                    result["confidence"] = "high"
                    logger.info("Upgraded confidence to high due to API verification")
            
        except ParkWhizNotFoundError:
            # Booking ID not found in ParkWhiz system - this is the key issue!
            logger.warning(f"Booking ID {booking_id} NOT FOUND in ParkWhiz system")
            
            result["booking_not_found"] = True
            booking_info["api_verified"] = False
            result["booking_info"] = booking_info
            
            # Update reasoning to be clear about what happened
            result["api_error"] = f"Booking ID {booking_id} was not found in the ParkWhiz system"
            
        except ParkWhizError as e:
            # API error - don't fail the extraction, but log the issue
            logger.error(f"ParkWhiz API error while verifying booking {booking_id}: {e}")
            
            # Don't set booking_not_found=True for API errors, only for actual "not found"
            result["api_error"] = f"Unable to verify booking due to API error: {str(e)}"
            booking_info["api_verified"] = False
            result["booking_info"] = booking_info
            
        except Exception as e:
            # Unexpected error - log but don't fail the extraction
            logger.error(f"Unexpected error verifying booking {booking_id}: {type(e).__name__}: {e}")
            
            result["api_error"] = f"Unexpected error during verification: {str(e)}"
            booking_info["api_verified"] = False
            result["booking_info"] = booking_info
        
        return result
    
    def _detect_pattern_based_mismatches(self, result: Dict, booking_id: str) -> bool:
        """
        Detect obvious booking mismatches using only ticket data patterns.
        
        This method compares customer claims against system data within the same ticket
        to detect obvious contradictions where customer claims recent dates
        but system shows 2012.
        
        Args:
            result: Extraction result containing booking info
            booking_id: The booking ID to analyze
            
        Returns:
            True if obvious mismatch detected, False otherwise
        """
        booking_info = result.get("booking_info", {})
        
        # Get system data (source of truth)
        system_event_date = booking_info.get("event_date")  # From "Parking Pass Start Time"
        system_user_email = booking_info.get("user_email")  # From "User Email"
        system_user_name = booking_info.get("user_name")    # From "User Name"
        
        # Get customer claims
        customer_claimed_date = booking_info.get("customer_claimed_date")  # From "Arrival" in request section
        customer_email = booking_info.get("customer_email")  # From "Email" in request section
        customer_name = booking_info.get("customer_name")    # From "Name" in request section
        
        mismatches = []
        
        # 1. Temporal mismatch detection
        if system_event_date and customer_claimed_date:
            try:
                from datetime import datetime
                system_dt = datetime.fromisoformat(system_event_date)
                customer_dt = datetime.fromisoformat(customer_claimed_date)
                
                # Calculate time difference in years
                time_diff_days = abs((customer_dt - system_dt).days)
                years_diff = time_diff_days / 365.25
                
                logger.info(f"Pattern-based mismatch check: customer claims {customer_claimed_date}, "
                           f"system shows {system_event_date}, difference: {years_diff:.1f} years")
                
                # If difference is more than 5 years, it's a critical mismatch
                if years_diff >= 5:
                    mismatches.append({
                        "type": "temporal",
                        "severity": "critical",
                        "description": f"Customer claims parking for {customer_dt.strftime('%B %d, %Y')} but system shows actual booking was for {system_dt.strftime('%B %d, %Y')} - a difference of {years_diff:.1f} years. This suggests the customer provided an incorrect booking ID.",
                        "customer_claim": f"Customer claims: {customer_claimed_date}",
                        "system_data": f"System shows: {system_event_date}",
                        "confidence": 0.95
                    })
                    logger.warning(f"Critical temporal mismatch detected: {years_diff:.1f} years difference")
                
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse dates for pattern-based mismatch detection: {e}")
        
        # 2. Email mismatch detection
        if system_user_email and customer_email:
            # Clean and normalize emails for comparison
            system_email_clean = system_user_email.strip().lower()
            customer_email_clean = customer_email.strip().lower()
            
            logger.debug(f"Email comparison: system='{system_email_clean}' customer='{customer_email_clean}'")
            logger.debug(f"Email lengths: system={len(system_email_clean)} customer={len(customer_email_clean)}")
            logger.debug(f"Email repr: system={repr(system_email_clean)} customer={repr(customer_email_clean)}")
            
            if system_email_clean != customer_email_clean:
                # Check if domains are different (more significant)
                system_domain = system_email_clean.split('@')[-1] if '@' in system_email_clean else ""
                customer_domain = customer_email_clean.split('@')[-1] if '@' in customer_email_clean else ""
                
                logger.debug(f"Domain comparison: system='{system_domain}' customer='{customer_domain}'")
                
                if system_domain != customer_domain:
                    mismatches.append({
                        "type": "email",
                        "severity": "high",
                        "description": f"Customer email ({customer_email_clean}) has different domain than booking owner ({system_email_clean}). This suggests the customer may have provided someone else's booking ID.",
                        "customer_claim": f"Customer email: {customer_email_clean}",
                        "system_data": f"Booking owner email: {system_email_clean}",
                        "confidence": 0.85
                    })
                    logger.warning(f"Email domain mismatch detected: {customer_email_clean} vs {system_email_clean}")
                else:
                    # Same domain, different username = medium severity
                    mismatches.append({
                        "type": "email",
                        "severity": "medium",
                        "description": f"Customer email ({customer_email_clean}) has different username than booking owner ({system_email_clean}). Same domain but different username may indicate shared account or typo.",
                        "customer_claim": f"Customer email: {customer_email_clean}",
                        "system_data": f"Booking owner email: {system_email_clean}",
                        "confidence": 0.70
                    })
                    logger.warning(f"Email username mismatch detected: {customer_email_clean} vs {system_email_clean}")
            else:
                logger.debug(f"Emails match: {customer_email_clean}")
        
        # 3. Name mismatch detection (basic check)
        if system_user_name and customer_name:
            # Simple check - if names are completely different (no common words)
            system_words = set(system_user_name.lower().split())
            customer_words = set(customer_name.lower().split())
            
            if not (system_words & customer_words):  # No common words
                mismatches.append({
                    "type": "customer",
                    "severity": "medium",
                    "description": f"Customer name ({customer_name}) appears completely different from booking owner ({system_user_name}). This may indicate incorrect booking information.",
                    "customer_claim": f"Customer name: {customer_name}",
                    "system_data": f"Booking owner: {system_user_name}",
                    "confidence": 0.70
                })
                logger.warning(f"Name mismatch detected: {customer_name} vs {system_user_name}")
        
        # If any mismatches detected, create mismatch result
        if mismatches:
            logger.warning(f"Pattern-based mismatch detected for booking {booking_id}: {len(mismatches)} mismatches")
            
            # Find highest severity
            severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            highest_severity = min(mismatches, key=lambda m: severity_order.get(m["severity"], 4))["severity"]
            
            # Create escalation reason
            if highest_severity == "critical":
                escalation_reason = f"Critical booking mismatch detected: {mismatches[0]['description'][:100]}..."
            else:
                escalation_reason = f"Multiple booking mismatches detected requiring human verification"
            
            # Format summary for agents - ULTRA clean format, no raw data
            summary = "**BOOKING MISMATCH DETECTED**\n\n"
            summary += "The booking ID exists in the system, but the customer's details don't match "
            summary += "the actual booking information. This suggests the customer may have "
            summary += "provided incorrect booking details.\n\n"
            
            # Add only the most critical issue in simple terms
            critical_mismatches = [m for m in mismatches if m["severity"] == "critical"]
            if critical_mismatches:
                mismatch = critical_mismatches[0]
                if mismatch["type"] == "temporal":
                    # Extract years from description for cleaner display
                    try:
                        import re
                        years_match = re.search(r'(\d+\.?\d*)\s+years?', mismatch['description'])
                        if years_match:
                            years = years_match.group(1)
                            summary += f"**Key Issue:** Customer claims recent parking date but system shows "
                            summary += f"booking is {years} years old. This suggests wrong booking ID.\n\n"
                    except:
                        summary += "**Key Issue:** Significant date discrepancy detected.\n\n"
            
            # Simple comparison - only show the essential differences, no raw data
            summary += "**Customer vs System:**\n"
            if customer_name and system_user_name and customer_name != system_user_name:
                summary += f"• Customer: {customer_name} → System: {system_user_name}\n"
            if customer_claimed_date and system_event_date:
                summary += f"• Requested: {customer_claimed_date} → Actual: {system_event_date}\n"
            
            summary += "\n**Action Required:** Contact customer to verify correct booking ID."
            
            result["booking_mismatch"] = True
            result["booking_mismatch_details"] = {
                "mismatches": mismatches,
                "escalation_reason": escalation_reason,
                "summary": summary
            }
            
            # Set confidence to high for mismatch detection
            result["confidence"] = "high"
            logger.info("Pattern-based mismatch detected, set confidence to high")
            return True
        
        # No obvious mismatches detected
        logger.info("No pattern-based mismatches detected")
        return False
    
    def _extract_and_analyze_timeline(self, result: Dict, ticket_notes: str) -> Dict:
        """
        Extract timeline data and perform timeline analysis on the booking information.
        
        This method integrates TimelineAnalyzer and TimelineExtractor to add timeline
        analysis capabilities to the existing booking extraction results.
        
        Args:
            result: Existing extraction result from LLM or pattern matching
            ticket_notes: Original ticket notes for timeline extraction
            
        Returns:
            Updated result with timeline analysis data
        """
        logger.info("Starting timeline extraction and analysis")
        
        try:
            # Extract timeline data from ticket text using deterministic patterns
            timeline_data = self.timeline_extractor.extract_timeline_data(ticket_notes)
            
            # Get booking window from existing extraction results
            booking_info = result.get("booking_info", {})
            booking_start = None
            booking_end = None
            
            # Try to parse booking window from extracted event_date
            if booking_info.get("event_date"):
                try:
                    event_date_str = booking_info["event_date"]
                    
                    # Parse event date (should be in ISO format from LLM extraction)
                    if isinstance(event_date_str, str):
                        booking_start = datetime.fromisoformat(event_date_str)
                        # Assume 2-hour default booking window if no end time specified
                        # This can be enhanced later with more sophisticated booking window detection
                        booking_end = booking_start + timedelta(hours=2)
                        logger.debug(f"Derived booking window from event_date: {booking_start} to {booking_end}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not parse event_date for booking window: {e}")
            
            # Merge timeline data from pattern extraction and LLM extraction
            # LLM extraction takes precedence if available
            llm_arrival = self._parse_llm_datetime(booking_info.get('arrival_time'))
            llm_exit = self._parse_llm_datetime(booking_info.get('exit_time'))
            llm_booking_start = self._parse_llm_datetime(booking_info.get('booking_window_start'))
            llm_booking_end = self._parse_llm_datetime(booking_info.get('booking_window_end'))
            
            # Use LLM data if available, otherwise fall back to pattern extraction or derived data
            final_booking_start = llm_booking_start or timeline_data.get('booking_start') or booking_start
            final_booking_end = llm_booking_end or timeline_data.get('booking_end') or booking_end
            arrival_time = llm_arrival or timeline_data.get('arrival_time')
            exit_time = llm_exit or timeline_data.get('exit_time')
            
            # Perform timeline analysis if we have sufficient data
            if any([final_booking_start, final_booking_end, arrival_time, exit_time]):
                timeline_analysis = self.timeline_analyzer.analyze_timeline(
                    booking_start=final_booking_start,
                    booking_end=final_booking_end,
                    arrival_time=arrival_time,
                    exit_time=exit_time
                )
                
                # Add timeline analysis to result
                result["timeline_analysis"] = {
                    "booking_start": final_booking_start.isoformat() if final_booking_start else None,
                    "booking_end": final_booking_end.isoformat() if final_booking_end else None,
                    "arrival_time": arrival_time.isoformat() if arrival_time else None,
                    "exit_time": exit_time.isoformat() if exit_time else None,
                    "early_arrival_minutes": timeline_analysis.early_arrival_minutes,
                    "overstay_minutes": timeline_analysis.overstay_minutes,
                    "actual_parking_duration": timeline_analysis.actual_parking_duration,
                    "booked_duration": timeline_analysis.booked_duration,
                    "has_early_arrival": timeline_analysis.has_early_arrival,
                    "has_overstay": timeline_analysis.has_overstay,
                    "is_timeline_consistent": timeline_analysis.is_timeline_consistent,
                    "timeline_issues": timeline_analysis.timeline_issues
                }
                
                logger.info(f"Timeline analysis complete: early_arrival={timeline_analysis.has_early_arrival}, "
                           f"overstay={timeline_analysis.has_overstay}, "
                           f"consistent={timeline_analysis.is_timeline_consistent}")
                
                # Enhance confidence based on timeline analysis
                if timeline_analysis.is_timeline_consistent and (arrival_time or exit_time):
                    # Upgrade confidence if we have timeline data and it's consistent
                    current_confidence = result.get("confidence", "low")
                    if current_confidence == "low" and timeline_analysis.is_timeline_consistent:
                        result["confidence"] = "medium"
                        logger.info("Upgraded confidence to medium due to consistent timeline analysis")
                
            else:
                logger.info("Insufficient timeline data for analysis")
                result["timeline_analysis"] = None
                
        except Exception as e:
            logger.error(f"Timeline extraction and analysis error: {type(e).__name__}: {e}")
            # Don't fail the entire extraction due to timeline analysis errors
            result["timeline_analysis"] = None
            result["timeline_error"] = f"Timeline analysis failed: {str(e)}"
        
        return result
    
    def _parse_llm_datetime(self, datetime_str: Optional[str]) -> Optional[datetime]:
        """
        Parse datetime string from LLM extraction.
        
        Args:
            datetime_str: Datetime string from LLM (should be ISO format)
            
        Returns:
            Parsed datetime or None if parsing fails
        """
        if not datetime_str or not isinstance(datetime_str, str):
            return None
        
        try:
            return datetime.fromisoformat(datetime_str)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse LLM datetime '{datetime_str}': {e}")
            return None