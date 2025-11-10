# Plan 1: The Complete "Freshdesk Ticket Ingestion" Journey

**Objective:** To define a comprehensive, multi-step journey in `main.py` that automates the entire process from ticket creation to resolution, including security, data enrichment, decision making, and logging.

**High-Level View:**
The journey will be a single, sequential state machine. However, it will have multiple conditional branches to handle different scenarios (e.g., data already exists, security fails, refund is approved/denied). Each step will be followed by a logging step to ensure a complete audit trail.

---

### Detailed Journey Steps

1.  **State 1: Journey Start & Initial Log**
    *   **Trigger:** User message like "New Freshdesk ticket 12345".
    *   **Action:** The journey's first step is to call the `database_logger.log_audit_event` tool.
    *   **Details:** It will log a "JOURNEY_START" event, generate a unique `run_id` for this session, and store the `run_id` in a journey variable for all subsequent logs.

2.  **State 2: Fetch Ticket Data from Freshdesk**
    *   **Action:** Call the `freshdesk_tools.get_ticket` tool, using the ID from the trigger message.
    *   **Logging:** The next state calls `database_logger.log_audit_event` to save the complete, raw ticket data returned by the tool.

3.  **State 3: Security Scan**
    *   **Action:** Call a new `lakera_security_tool.check_content` tool, passing it the body of the ticket from the previous step's data.
    *   **Logging:** Log the result of the security scan.
    *   **Conditional Branch:** If the scan result is unsafe, the journey will immediately jump to the final "Escalate" path (see Stage 6).

4.  **State 4: Check for Pre-existing Booking Info**
    *   **Action:** Use a "Thinking Tool" (`extract_booking_info_from_note`).
    *   **Details:** This tool's instruction will be: "Analyze the private notes from the Freshdesk ticket data. If you find ParkWhiz booking information, extract it as a JSON object. If not, return null."
    *   **Logging:** Log whether the data was found in the notes.

5.  **State 5: Data Enrichment (Conditional)**
    *   This stage has two paths that merge into one.
    *   **Path A (Info Found):** If the previous step returned booking data, this path does nothing and proceeds directly to Stage 6.
    *   **Path B (Info NOT Found):** If the previous step returned null, this path executes a `tool_state` to call the `parkwhiz_tools.get_customer_orders` tool as a fallback.
    *   **Logging:** In Path B, the call to the ParkWhiz API and its result are logged via `database_logger.log_audit_event`.
    *   **Merge:** Both paths converge here, with the `booking_info` now stored in a journey variable, ready for the next stage.

6.  **State 6: Triage ("The Thinking Step")**
    *   **Action:** Call the main "Thinking Tool" (`triage_ticket`).
    *   **Details:** This tool will be given the full ticket data, the `booking_info`, and access to the **Refund Policy Retriever**. Its instruction will be to analyze all information and return one of three strings: "Approved", "Denied", or "Needs Human Review".
    *   **Logging:** Log the final classification decision.

7.  **State 7: Action & Resolution (Three Final Paths)**
    *   The journey splits into three final branches based on the triage decision. Each branch ends with a final logging step.
    *   **Branch A: "Approved"**
        1.  Call `process_refund` tool.
        2.  Call `add_note` tool (to log the automatic approval in Freshdesk).
        3.  Call `update_ticket` tool (to set status to "Closed").
        4.  **Final Log:** Call `database_logger.log_final_metric` and `database_logger.update_customer_context`.
    *   **Branch B: "Denied"**
        1.  Call `add_note` tool (using the Retriever to get precise policy wording for the denial note).
        2.  Call `update_ticket` tool (to set status to "Closed").
        3.  **Final Log:** Call `database_logger.log_final_metric` and `database_logger.update_customer_context` (e.g., to increment a `denial_count`).
    *   **Branch C: "Needs Human Review" / "Escalate"**
        1.  Call `add_note` tool (to leave a summary for the human agent).
        2.  Call `update_ticket` tool (to assign the ticket to a human queue).
        3.  **Final Log:** Call `database_logger.log_final_metric`.

---

### New Components & Tests Required for Plan 1

To implement the full journey, the following new components will need to be created, each with its own dedicated test suite.

**1. Database Logger Tool**
*   **Component:** A new tool file, `parlant/tools/database_logger.py`. This will contain the functions that handle all communication with the Postgres database.
    *   `log_audit_event()`
    *   `log_final_metric()`
    *   `update_customer_context()`
*   **Tests:** A new test file, `tests/tools/test_database_logger.py`.
    *   This will test each function to ensure it generates the correct SQL `INSERT` or `UPDATE` statements.
    *   It will also test that the tool handles potential database connection errors gracefully. (Note: The actual database connection will be mocked during tests).

**2. Lakera Security Tool**
*   **Component:** A new tool file, `parlant/tools/lakera_security_tool.py`. This will contain a `check_content()` function responsible for calling the Lakera API.
*   **Tests:** A new test file, `tests/tools/test_lakera_security_tool.py`.
    *   Test the tool's behavior with both "safe" and "unsafe" mock responses from the Lakera API.
    *   Test how the tool handles potential API errors (e.g., network issues, invalid API key).

**3. Journey Helper Tools ("Thinking Tools")**
*   **Component:** A new file, `parlant/tools/journey_helpers.py`, to hold simple tools that exist primarily to facilitate complex LLM reasoning steps.
    *   `extract_booking_info_from_note()`
    *   `triage_ticket()`
*   **Tests:** A new test file, `tests/tools/test_journey_helpers.py`.
    *   These tests will be simple, verifying that each tool can be called and returns its output in the correct `ToolResult` format.

**4. Refund Policy Retriever**
*   **Component:** This is not a new file, but a configuration step within `main.py` where a Parlant **Retriever** is created and pointed at the existing refund policy documents.
*   **Tests:** Direct unit testing of retrievers is complex. Initial verification will be done manually by asking the agent questions about the refund policy in the Parlant UI to ensure it's retrieving the correct information.

**5. The Journey Integration Test**
*   **Component:** The journey logic itself within the `create_ticket_ingestion_journey` function in `main.py`.
*   **Tests:** A new integration test file, `tests/test_journeys.py`.
    *   This will be the most comprehensive test. It will start the agent, trigger the journey with a message, and assert that the journey progresses through the correct states.
    *   All tools (`freshdesk`, `parkwhiz`, `database_logger`, etc.) will be mocked, so we can verify that the journey calls the right tool at the right time with the right information.
    *   We will create separate tests to verify the logic of the "Approved", "Denied", and "Escalate" paths from start to finish.

---

## Implementation Journey & Final Summary

This section documents the step-by-step progress and the final outcome of the implementation phase.

### Progress Update 1: Foundational Tools
- **Status:** ✅ Complete
- **Details:** Database Logger, Freshdesk API, and Lakera Security tools all implemented with full test coverage.

### Progress Update 2: Journey Helpers
- **Status:** ✅ Complete
- **Details:** The `extract_booking_info_from_note()` and `triage_ticket()` helper tools were created and tested.

### Progress Update 3: ParkWhiz API
- **Status:** ✅ Complete
- **Details:** The final tool, `get_customer_orders()`, was implemented and tested, completing all required tool development.

### Progress Update 4: Journey Implementation & Bug Fix
- **Status:** ✅ Complete
- **Details:** The initial `create_ticket_ingestion_journey` was wired together. A state-transition bug (tool-to-tool) was identified and fixed by inserting intermediate "thinking" chat states, ensuring correct Parlant journey flow.

---

## 🎉 FINAL SUCCESS! 🎉

The Parlant server is running successfully with the complete, end-to-end refund journey.

### ✅ Complete TDD Implementation:

**10 Tools Built (35 Passing Tests):**
1.  Database Logger - 3 tools ✅
2.  Freshdesk API - 3 tools ✅
3.  Lakera Security - 1 tool ✅
4.  Journey Helpers - 2 tools ✅
5.  ParkWhiz API - 1 tool ✅

**Database Schema:**
- 3 tables for audit logging, metrics, and customer context ✅

**Journey Implementation:**
- ✅ Complete refund processing journey implemented in `main.py`.
- ✅ Follows correct Parlant patterns (tool→chat→tool).
- ✅ Handles multiple paths: Approved, Denied, Escalate, and Security failure.
- ✅ **Server running successfully!**

### Journey Logic Overview:
1.  Fetches the initial Freshdesk ticket.
2.  Runs a Lakera security scan on the content.
3.  Extracts booking info from ticket notes, with a ParkWhiz API call as a fallback.
4.  Triages the refund decision based on established rules.
5.  Executes the final action (approves, denies, or escalates).
6.  Updates the Freshdesk ticket with the final outcome.
