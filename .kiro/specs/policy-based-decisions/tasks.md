# Implementation Plan - Policy-Based Decision Making

- [x] 1. Create PolicyLoader component
  - Create `parlant/tools/policy_loader.py` with PolicyLoader class
  - Implement `load_policies()` to read JSON and MD files from `parlant/context/processed/`
  - Implement `get_rules()` to return parsed refund rules
  - Implement `get_full_policy_text()` to return combined policy text for LLM
  - Add caching to avoid repeated file reads
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

- [x] 2. Create BookingExtractor component
  - Create `parlant/tools/booking_extractor.py` with BookingExtractor class
  - Implement `extract_booking_info()` to make direct Gemini API calls
  - Create structured prompt for booking information extraction
  - Parse LLM response into BookingInfo data model
  - Calculate confidence based on completeness of extracted data
  - Handle cases with multiple bookings mentioned
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 2.1 Add regex/pattern-based extraction fallback (Performance Optimization)
  - Create `parlant/tools/booking_patterns.py` with regex patterns for common fields
  - Define BOOKING_ID_PATTERNS, DATE_PATTERNS, LOCATION_PATTERNS, USER_INFO_PATTERNS
  - Implement `extract_from_html()` method using BeautifulSoup for structured HTML
  - Implement `extract_from_text()` method using regex for plain text
  - Try pattern-based extraction first before calling LLM
  - Fall back to LLM extraction if pattern matching fails or has low confidence
  - This optimization reduces LLM API calls and improves extraction speed
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Create RuleEngine component
  - Create `parlant/tools/rule_engine.py` with RuleEngine class
  - Implement `apply_rules()` to apply deterministic business logic
  - Implement date calculation logic (days_before_event)
  - Implement rule priority logic (7+ days, after event, 3-7 days, etc.)
  - Return decision with confidence level
  - Handle edge cases by returning "Uncertain" to trigger LLM
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 4. Create LLMAnalyzer component
  - Create `parlant/tools/llm_analyzer.py` with LLMAnalyzer class
  - Implement `analyze_case()` to make Gemini API calls with policy context
  - Create detailed prompt template with policy text, ticket data, and booking info
  - Request structured JSON response from Gemini
  - Parse and validate LLM output
  - Handle LLM failures gracefully (fallback to escalation)
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 5. Create CancellationReasonMapper component
  - Create `parlant/tools/cancellation_reason_mapper.py` with CancellationReasonMapper class
  - Define list of valid ParkWhiz cancellation reasons
  - Implement `map_reason()` to map decision reasoning to ParkWhiz cancellation reason
  - Use keyword matching for common scenarios (oversold, duplicate, pre-arrival, etc.)
  - Default to "Other" when no clear match
  - _Requirements: 5.3, 5.4_

- [x] 6. Create DecisionMaker orchestrator
  - Create `parlant/tools/decision_maker.py` with DecisionMaker class
  - Implement `make_decision()` to orchestrate the hybrid approach
  - Integrate PolicyLoader, BookingExtractor, RuleEngine, LLMAnalyzer, and CancellationReasonMapper
  - Implement decision flow: extract → rules → LLM (if needed) → map cancellation reason (if Approved)
  - Track processing time and method used
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

- [x] 7. Update extract_booking_info_from_note tool ✅ **COMPLETE**
  - Modified `parlant/tools/journey_helpers.py::extract_booking_info_from_note`
  - Replaced placeholder logic with BookingExtractor integration
  - Returns structured booking information with confidence levels
  - Handles empty or invalid ticket notes gracefully
  - Uses pattern-based + LLM extraction with automatic fallback
  - Returns extraction method used (pattern/llm/none) for monitoring
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_ ✅

- [x] 8. Update triage_ticket tool ✅ **COMPLETE**
  - Modified `parlant/tools/journey_helpers.py::triage_ticket`
  - Replaced placeholder logic with DecisionMaker orchestration
  - Handles both pre-extracted and missing booking info
  - Returns structured decision with reasoning, confidence, and cancellation_reason (if Approved)
  - Includes method_used and processing_time_ms in response
  - Validates ticket data and escalates when insufficient
  - Graceful error handling with escalation to human review
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_ ✅

- [x] 9. Create document_decision tool (MVP) ✅ **COMPLETE**
  - Created new tool in `parlant/tools/journey_helpers.py::document_decision`
  - Formats private note with decision, reasoning, policy applied, confidence, and cancellation reason (if Approved)
  - Includes ParkWhiz cancellation reason in the note for Approved decisions
  - Calls add_note to add private note to Freshdesk
  - Calls update_ticket to add "Processed by Whiz Agent" tag
  - Does NOT call ParkWhiz API to process refunds (MVP scope)
  - Graceful error handling with partial success states
  - Clear disclaimer: "This decision was made by the Whiz Agent. Please review before processing the refund."
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_ ✅

- [x] 9.1 Add dependencies for pattern-based extraction (if implementing Task 2.1)
  - Add beautifulsoup4 to requirements.txt for HTML parsing
  - Add lxml to requirements.txt for fast HTML parsing
  - Update Docker container with new dependencies
  - _Requirements: 2.1, 2.2_

- [x] 10. Add error handling and fallbacks
  - Add try/catch for LLM API failures in BookingExtractor
  - Add try/catch for LLM API failures in LLMAnalyzer
  - Implement fallback to rule-based logic when LLM fails
  - Implement escalation to "Needs Human Review" when no fallback available
  - Add timeout handling (10 seconds) for LLM calls
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.3_

- [x] 11. Add logging and monitoring
  - Add logging for policy loading (startup)
  - Add logging for booking extraction (confidence, found/not found)
  - Add logging for rule application (which rules matched)
  - Add logging for LLM analysis (decision, confidence, processing time)
  - Add logging for errors and timeouts
  - _Requirements: 8.5, 9.5_

- [x] 12. Create unit tests
  - [x] 12.1 Create `tests/tools/test_policy_loader.py`
    - Test loading all policy files
    - Test parsing JSON structure
    - Test caching behavior
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_
  
  - [x] 12.2 Create `tests/tools/test_booking_extractor.py`
    - Test extraction with complete booking info
    - Test extraction with partial booking info
    - Test extraction with no booking info
    - Test extraction with multiple bookings
    - Mock Gemini API responses
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [x] 12.2.1 Create `tests/tools/test_booking_patterns.py` (if implementing Task 2.1)
    - Test regex pattern matching for booking IDs
    - Test HTML parsing with BeautifulSoup
    - Test text extraction with various formats
    - Test fallback to LLM when patterns fail
    - Create test fixtures with real ticket HTML/text
    - _Requirements: 2.1, 2.2, 2.3_
  
  - [x] 12.3 Create `tests/tools/test_cancellation_reason_mapper.py`
    - Test mapping for common scenarios (oversold, duplicate, pre-arrival)
    - Test default to "Other" for ambiguous cases
    - Test all valid ParkWhiz cancellation reasons
    - _Requirements: 5.3, 5.4_

  - [x] 12.4 Create `tests/tools/test_rule_engine.py`
    - Test 7+ days rule (approve)
    - Test <3 days on-demand rule (deny)
    - Test after event rule (deny)
    - Test edge cases (uncertain)
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_
  
  - [x] 12.5 Create `tests/tools/test_llm_analyzer.py`
    - Test LLM analysis with mocked responses
    - Test error handling
    - Test timeout handling
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 12.6 Create `tests/tools/test_decision_maker.py`
    - Test full decision flow
    - Test rule-only path (high confidence)
    - Test hybrid path (low confidence → LLM)
    - Test error handling and fallbacks
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 12.7 Create `tests/tools/test_document_decision.py`
    - Test note formatting
    - Test Freshdesk API calls (add_note_to_ticket, update_ticket)
    - Verify tag is added correctly
    - Verify no ParkWhiz API calls are made
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 13. Create integration tests
  - [x] 13.1 Test with real ticket 1206331
    - Verify booking extraction works
    - Verify decision is made correctly
    - Verify note is added to Freshdesk
    - _Requirements: All_
  
  - [x] 13.2 Test with synthetic edge cases
    - Test missing booking ID
    - Test missing event date
    - Test ambiguous booking type
    - Test multiple bookings in one ticket
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_
  
  - [x] 13.3 Test performance
    - Measure rule-based decision time (<2s)
    - Measure LLM-based decision time (<10s)
    - Verify caching improves performance
    - _Requirements: 9.1, 9.2, 9.4_

- [x] 14. Update documentation
  - Update README with new decision-making approach
  - Document policy file format and location
  - Document how to add new rules
  - Document confidence levels and escalation criteria
  - Document MVP scope: decisions are documented but refunds are not automatically processed
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_
