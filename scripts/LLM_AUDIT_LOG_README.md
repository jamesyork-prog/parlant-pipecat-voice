# LLM Audit Log System

This system logs every LLM interaction to PostgreSQL for debugging, analysis, and compliance.

## What Gets Logged

Every time the LLM is called for a refund decision, the following is captured:

- **Full prompt text** - Everything sent to the LLM (policy, ticket data, booking info, instructions)
- **Full response text** - The complete JSON response from the LLM
- **Decision details** - Decision, confidence, policy applied
- **Processing time** - How long the LLM call took
- **Metadata** - Booking info, rule engine result, key factors
- **Errors** - Any errors that occurred during the LLM call

## Database Schema

```sql
CREATE TABLE llm_audit_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ticket_id VARCHAR(50),
    model_name VARCHAR(100),
    prompt_text TEXT,              -- Full prompt sent to LLM
    response_text TEXT,             -- Full response from LLM
    decision VARCHAR(50),           -- Approved/Denied/Needs Human Review
    confidence VARCHAR(20),         -- high/medium/low
    policy_applied TEXT,            -- Which policy rule was applied
    processing_time_ms INTEGER,     -- Time taken for LLM call
    error_message TEXT,             -- Error if call failed
    metadata JSONB                  -- Additional context
);
```

## How to Query LLM Logs

### Option 1: Use the Helper Script (Easiest)

```bash
# View all LLM data for a specific ticket
./scripts/query_llm_logs.sh <ticket_id>
```

This will show:
- Summary (decision, confidence, timing)
- Full prompt text
- Full response text
- Metadata (booking info, rule result)

### Option 2: Direct SQL Queries

**See recent LLM calls:**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT 
    id, 
    timestamp, 
    ticket_id, 
    model_name, 
    decision, 
    confidence, 
    processing_time_ms 
FROM llm_audit_log 
ORDER BY timestamp DESC 
LIMIT 10;
"
```

**Get full prompt for a ticket:**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT prompt_text 
FROM llm_audit_log 
WHERE ticket_id = '<ticket_id>' 
ORDER BY timestamp DESC 
LIMIT 1;
"
```

**Get full response for a ticket:**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT response_text 
FROM llm_audit_log 
WHERE ticket_id = '<ticket_id>' 
ORDER BY timestamp DESC 
LIMIT 1;
"
```

**Get metadata (booking info, rule result):**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT jsonb_pretty(metadata) 
FROM llm_audit_log 
WHERE ticket_id = '<ticket_id>' 
ORDER BY timestamp DESC 
LIMIT 1;
"
```

**Find all tickets that used LLM (not just rules):**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT 
    ticket_id, 
    decision, 
    confidence, 
    processing_time_ms,
    timestamp
FROM llm_audit_log 
ORDER BY timestamp DESC;
"
```

**Find slow LLM calls (>10 seconds):**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT 
    ticket_id, 
    processing_time_ms, 
    decision,
    timestamp
FROM llm_audit_log 
WHERE processing_time_ms > 10000
ORDER BY processing_time_ms DESC;
"
```

**Find LLM errors:**
```bash
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT 
    ticket_id, 
    error_message, 
    timestamp
FROM llm_audit_log 
WHERE error_message IS NOT NULL
ORDER BY timestamp DESC;
"
```

## When Does LLM Get Called?

The LLM is only called when the rule engine returns:
- `decision = "Uncertain"` OR
- `confidence = "low"`

Otherwise, the rule engine makes the decision directly without LLM.

## Troubleshooting

**Table doesn't exist?**
```bash
cat postgres/migrate_llm_audit.sql | docker-compose exec -T postgres psql -U admin -d WhizDB
```

**No logs appearing?**
- Check that the LLM is actually being called (rule engine might be handling it)
- Check for errors in the logs: `docker-compose logs parlant | grep -i error`
- Verify database connection: `docker-compose exec postgres psql -U admin -d WhizDB -c "\dt"`

## Example: Debugging a Decision

```bash
# 1. Run a test on a ticket
# (trigger via webhook or manual test)

# 2. Check if LLM was used
./scripts/query_llm_logs.sh <ticket_id>

# 3. If no results, the rule engine handled it directly
# Check the Freshdesk note to see "Method Used: rules"

# 4. If results exist, you'll see:
#    - Exact prompt sent to LLM
#    - Exact response received
#    - All context (booking info, policy, rule result)
```

## Privacy & Compliance

- Prompts may contain customer PII (names, emails, booking IDs)
- Store this data securely and comply with data retention policies
- Consider adding data retention/cleanup policies if needed

## Performance Monitoring

Use this data to:
- Track LLM response times
- Identify slow queries
- Monitor error rates
- Analyze which cases require LLM vs rules
- Optimize prompts based on actual usage
