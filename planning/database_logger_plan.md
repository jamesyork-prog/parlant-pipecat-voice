# Plan 2: Building the Database Logger Foundation

**Objective:** To design the required database tables and implement the Python tool (`database_logger.py`) that our journey will use to write information to the PostgreSQL database.

---

### Part 1: Define the Database Schema

First, we need to tell the PostgreSQL server what tables to create. We will do this by adding `CREATE TABLE` statements to the `postgres/init.sql` file. This ensures the tables are ready every time the database container starts.

**Action:** Append the following SQL commands to the `postgres/init.sql` file.

1.  **The Audit Trail Table:** A detailed, step-by-step log of every action.
    ```sql
    CREATE TABLE agent_audit_log (
        log_id SERIAL PRIMARY KEY,
        run_id VARCHAR(255) NOT NULL,
        timestamp TIMESTAMPTZ DEFAULT NOW(),
        event_type VARCHAR(50),
        event_details JSONB,
        status VARCHAR(50),
        error_message TEXT
    );
    ```

2.  **The Metrics Table:** A high-level summary of each journey run for reporting.
    ```sql
    CREATE TABLE agent_run_metrics (
        metric_id SERIAL PRIMARY KEY,
        run_id VARCHAR(255) UNIQUE NOT NULL,
        journey_name VARCHAR(255),
        start_time TIMESTAMPTZ,
        end_time TIMESTAMPTZ,
        duration_ms INTEGER,
        final_outcome VARCHAR(50),
        ticket_id VARCHAR(255)
    );
    ```

3.  **The Context "Memory" Table:** A persistent store of key facts about customers.
    ```sql
    CREATE TABLE customer_context (
        customer_id VARCHAR(255) PRIMARY KEY,
        last_interaction_date TIMESTAMPTZ,
        total_interactions INTEGER DEFAULT 1,
        total_denials INTEGER DEFAULT 0,
        custom_notes JSONB
    );
    ```

---

### Part 2: Create the Python Logger Tool

Next, we will create the actual tool that the agent journey can call. This involves creating a new Python file with three distinct tool functions.

**Action:** Create a new file named `parlant/tools/database_logger.py` with the necessary functions to interact with the tables above. The tool will handle connecting to the database and executing the `INSERT` or `UPDATE` commands.

**Action:** Create the corresponding test file `tests/tools/test_database_logger.py`. This test file will mock the database connection and verify that each logger tool tries to execute the correct SQL command, ensuring the logic is sound without requiring a live database during tests.

---
## Implementation Summary

**Status: ✅ Complete**

Following the TDD approach from this plan, the database logging foundation has been successfully implemented.

### Step 1: Database Schema ✅
Added three tables and four indexes to `postgres/init.sql`:
*   `agent_audit_log` - Detailed step-by-step audit trail with `run_id`, `event_type`, `event_details` (JSONB), status, and error messages.
*   `agent_run_metrics` - High-level journey metrics with start/end times, duration, outcome, and `ticket_id`.
*   `customer_context` - Persistent customer memory with interaction counts, denial counts, and custom notes (JSONB).

### Step 2: Test Suite ✅
Created `tests/tools/test_database_logger.py` with 8 comprehensive tests covering:
*   Successful operations for all three tools.
*   Database error handling and connection failures.
*   Partial data handling and increment modes.

### Step 3: Implementation ✅
Created `parlant/tools/database_logger.py` with three Parlant tools:
1.  `log_audit_trail` - Logs detailed event entries.
2.  `log_run_metrics` - Upserts journey run metrics (handles both new and updates).
3.  `update_customer_context` - Upserts customer data with both direct-set and increment modes.

### Step 4: Verification ✅
*   All 8 tests are passing.
*   Database tables are created and verified.
*   The tools are now ready to be integrated into the main journey.