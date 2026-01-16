# Integration Tests

This directory contains integration tests that verify the complete ticket processing workflow.

## Test Scripts

### test_gemini_workflow.py
**Purpose:** Tests the complete ticket processing workflow using Parlant's API.

**What it tests:**
- Session creation
- Message sending ("Process ticket <ticket_id>")
- Tool execution monitoring
- Workflow completion verification

**Usage:**
```bash
docker-compose exec parlant python /app/app_tools/test_gemini_workflow.py
```

**Status:** Reference implementation for API-based testing

### test_ticket_direct.py
**Purpose:** Tests ticket processing by directly calling the `process_ticket_end_to_end` tool.

**What it tests:**
- Direct tool invocation
- All workflow steps (fetch, scan, extract, triage, update)
- Freshdesk integration (notes and tags)
- Decision-making logic

**Usage:**
```bash
docker-compose exec parlant python /app/app_tools/test_ticket_direct.py
```

**Status:** Primary integration test for workflow verification

### verify_ticket_updates.py
**Purpose:** Verifies that Freshdesk tickets are actually updated with notes and tags.

**What it tests:**
- Freshdesk API connectivity
- Note creation and privacy settings
- Tag application
- Actual ticket state in Freshdesk

**Usage:**
```bash
docker-compose exec parlant python /app/app_tools/verify_ticket_updates.py
```

**Status:** Verification tool for Freshdesk integration

## Running All Integration Tests

```bash
# Run all integration tests
for test in tests/integration/*.py; do
    docker-compose exec parlant python /app/app_tools/$(basename $test)
done
```

## Test Ticket

All tests use a configurable test ticket (set via `TEST_TICKET_ID` environment variable) which should contain:
- Complete booking information in description
- Test markers in subject ("TEST - IGNORE")
- Known structure for consistent testing

## Expected Results

✅ All workflow steps complete
✅ Private note added to Freshdesk
✅ Tags applied: `needs_human_review`, `automated_analysis`
✅ Decision made: "Needs Human Review" (due to booking extraction issue)

## Policy-Based Decision Tests

### test_policy_decision_real_ticket.py
**Purpose:** Tests the complete policy-based decision workflow with a real ticket.

**What it tests:**
- Booking information extraction from ticket notes
- Rule-based or LLM-based decision making
- Cancellation reason mapping for Approved decisions
- Decision quality (valid decision, reasoning, policy reference, confidence)
- Note formatting for Freshdesk documentation
- Performance within expected thresholds

**Usage:**
```bash
docker-compose exec parlant python /app/tests/integration/test_policy_decision_real_ticket.py
```

**Expected Results:**
- ✅ Booking information extracted (if present in ticket)
- ✅ Valid decision made (Approved/Denied/Needs Human Review)
- ✅ Decision includes reasoning and policy reference
- ✅ Confidence level assigned
- ✅ Cancellation reason provided for Approved decisions
- ✅ Processing time within limits (<2s for rules, <10s for LLM)

### test_policy_decision_edge_cases.py
**Purpose:** Tests decision-making with synthetic edge cases.

**What it tests:**
- Missing booking ID → Should escalate
- Missing event date → Should escalate
- Ambiguous booking type → Should handle gracefully
- Multiple bookings in one ticket → Should handle gracefully
- Clear rule cases → Should use rules and be fast
- Past events → Should deny or escalate

**Usage:**
```bash
docker-compose exec parlant python /app/tests/integration/test_policy_decision_edge_cases.py
```

**Expected Results:**
- ✅ All edge cases handled without crashes
- ✅ Appropriate escalation for missing critical data
- ✅ Graceful handling of ambiguous cases
- ✅ Valid decisions for all scenarios

### test_policy_decision_performance.py
**Purpose:** Tests decision-making performance and caching.

**What it tests:**
- Rule-based decision time (<2 seconds)
- LLM-based decision time (<10 seconds)
- Policy caching effectiveness
- Batch processing performance

**Usage:**
```bash
docker-compose exec parlant python /app/tests/integration/test_policy_decision_performance.py
```

**Expected Results:**
- ✅ Rule-based decisions complete within 2s
- ✅ LLM-based decisions complete within 10s
- ✅ Policy caching improves or maintains performance
- ✅ Batch processing maintains good average time

## Running All Integration Tests

### Using the Test Runner Script
```bash
# Run all policy-based decision tests
./tests/integration/run_all_policy_tests.sh
```

### Running Tests Individually
```bash
# Run all integration tests
docker-compose exec parlant python /app/tests/integration/test_policy_decision_real_ticket.py
docker-compose exec parlant python /app/tests/integration/test_policy_decision_edge_cases.py
docker-compose exec parlant python /app/tests/integration/test_policy_decision_performance.py
```

## Next Steps

The policy-based decision making system is now implemented and tested. These integration tests verify:
- ✅ Booking information extraction
- ✅ Policy-based decision making (Approved/Denied/Needs Human Review)
- ✅ Cancellation reason mapping
- ✅ Edge case handling
- ✅ Performance requirements
