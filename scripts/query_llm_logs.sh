#!/bin/bash
# Query LLM Audit Logs
# Usage: ./scripts/query_llm_logs.sh [ticket_id]

TICKET_ID=${1:-}

if [ -z "$TICKET_ID" ]; then
    echo "Usage: ./scripts/query_llm_logs.sh <ticket_id>"
    echo ""
    echo "Example: ./scripts/query_llm_logs.sh <ticket_id>"
    echo ""
    echo "Or to see all recent LLM calls:"
    echo "  docker-compose exec postgres psql -U admin -d WhizDB -c \"SELECT id, timestamp, ticket_id, model_name, decision, confidence, processing_time_ms FROM llm_audit_log ORDER BY timestamp DESC LIMIT 10;\""
    exit 1
fi

echo "==================================="
echo "LLM Logs for Ticket: $TICKET_ID"
echo "==================================="
echo ""

# Get the log entry
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT 
    id,
    timestamp,
    ticket_id,
    model_name,
    decision,
    confidence,
    policy_applied,
    processing_time_ms,
    error_message
FROM llm_audit_log 
WHERE ticket_id = '$TICKET_ID' 
ORDER BY timestamp DESC;
"

echo ""
echo "==================================="
echo "Full Prompt for Ticket: $TICKET_ID"
echo "==================================="
echo ""

# Get the full prompt
docker-compose exec postgres psql -U admin -d WhizDB -t -c "
SELECT prompt_text 
FROM llm_audit_log 
WHERE ticket_id = '$TICKET_ID' 
ORDER BY timestamp DESC 
LIMIT 1;
"

echo ""
echo "==================================="
echo "Full Response for Ticket: $TICKET_ID"
echo "==================================="
echo ""

# Get the full response
docker-compose exec postgres psql -U admin -d WhizDB -t -c "
SELECT response_text 
FROM llm_audit_log 
WHERE ticket_id = '$TICKET_ID' 
ORDER BY timestamp DESC 
LIMIT 1;
"

echo ""
echo "==================================="
echo "Metadata for Ticket: $TICKET_ID"
echo "==================================="
echo ""

# Get the metadata (booking info, rule result, etc.)
docker-compose exec postgres psql -U admin -d WhizDB -t -c "
SELECT jsonb_pretty(metadata) 
FROM llm_audit_log 
WHERE ticket_id = '$TICKET_ID' 
ORDER BY timestamp DESC 
LIMIT 1;
"
