# Integration Guide: Adding Metrics to Parlant Application

This guide explains how to integrate the MOR/Product Type router metrics with your existing Parlant application.

## Option 1: Standalone Metrics Exporter (Recommended)

The standalone metrics exporter runs as a separate container and collects metrics from the database. This is the recommended approach as it doesn't require changes to your main application.

### Setup

1. **Start the monitoring stack**:
   ```bash
   ./monitoring/setup-monitoring.sh
   ```

2. **Verify metrics collection**:
   ```bash
   ./monitoring/health-check.sh
   ```

3. **Access dashboards**:
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

## Option 2: Integrated Metrics Endpoint

If you prefer to expose metrics directly from your Parlant application, you can add a metrics endpoint.

### Step 1: Add Metrics Endpoint to Webhook Server

Add the following to your `webhook_server.py`:

```python
from parlant.tools.routing.metrics_endpoint import get_prometheus_metrics, get_metrics_json

# Add this route to your FastAPI/Flask app
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_prometheus_metrics(),
        media_type="text/plain"
    )

@app.get("/metrics/json")
async def metrics_json():
    """JSON metrics endpoint for debugging."""
    return get_metrics_json()
```

### Step 2: Update Docker Compose

Add the metrics port to your main `docker-compose.yml`:

```yaml
services:
  parlant:
    ports:
      - "8800:8800"
      - "8801:8801"
      - "8080:8080"  # Add metrics port
```

### Step 3: Update Prometheus Configuration

Update `monitoring/prometheus/prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'parlant-app'
    static_configs:
      - targets: ['parlant:8080']  # Use your app's metrics port
    scrape_interval: 30s
    metrics_path: /metrics
```

## Option 3: Database-Only Metrics

If you only want to track metrics in the database without Prometheus/Grafana:

### Step 1: Enable Audit Logging

Ensure your router is configured with audit logging:

```python
from parlant.tools.routing.audit_logger import get_audit_logger

# In your router initialization
audit_logger = get_audit_logger()
```

### Step 2: Query Metrics

Use SQL queries to analyze router performance:

```sql
-- Overall rejection rate
SELECT 
    COUNT(*) as total_processed,
    COUNT(*) FILTER (WHERE decision = 'reject') as rejections,
    (COUNT(*) FILTER (WHERE decision = 'reject')::float / COUNT(*)) * 100 as rejection_rate
FROM routing_audit_log 
WHERE created_at > NOW() - INTERVAL '24 hours';

-- MOR gate vendor breakdown
SELECT 
    metadata->>'vendor' as vendor,
    COUNT(*) as rejection_count
FROM routing_audit_log 
WHERE gate_name = 'mor_gate' 
AND decision = 'reject'
AND created_at > NOW() - INTERVAL '24 hours'
GROUP BY metadata->>'vendor'
ORDER BY rejection_count DESC;

-- Performance metrics
SELECT 
    gate_name,
    AVG(processing_time_ms) as avg_time,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY processing_time_ms) as p95_time
FROM routing_audit_log 
WHERE created_at > NOW() - INTERVAL '1 hour'
GROUP BY gate_name;
```

## Testing the Integration

### 1. Generate Test Data

Run some test tickets through your router to generate metrics:

```python
# Example test script
from parlant.tools.routing.ticket_router import TicketRouter

router = TicketRouter()

# Test MOR rejections
test_tickets = [
    {"Partner": "AXS", "ticket_id": "test_1"},
    {"Partner": "SeatGeek", "ticket_id": "test_2"},
    {"Partner": "ParkWhiz", "ticket_id": "test_3"},
]

for ticket in test_tickets:
    result = router.route_ticket(ticket)
    print(f"Ticket {ticket['ticket_id']}: {result.decision}")
```

### 2. Verify Metrics

Check that metrics are being collected:

```bash
# Check Prometheus metrics
curl http://localhost:8080/metrics | grep router_

# Check JSON metrics
curl http://localhost:8080/metrics/json | jq .

# Check database
docker-compose exec postgres psql -U admin -d WhizDB -c "
SELECT gate_name, decision, COUNT(*) 
FROM routing_audit_log 
GROUP BY gate_name, decision;
"
```

### 3. Validate Dashboard

1. Open Grafana: http://localhost:3000
2. Navigate to the MOR/Product Type Router dashboard
3. Verify that panels show data
4. Test different time ranges

## Troubleshooting

### No Metrics Data

1. **Check audit logging**:
   ```sql
   SELECT COUNT(*) FROM routing_audit_log;
   ```

2. **Verify router integration**:
   ```python
   from parlant.tools.routing.metrics_tracker import get_gate_metrics_tracker
   tracker = get_gate_metrics_tracker()
   print(tracker.get_dashboard_data())
   ```

3. **Check metrics endpoint**:
   ```bash
   curl http://localhost:8080/metrics
   ```

### Dashboard Not Loading

1. **Check Grafana logs**:
   ```bash
   docker-compose -f monitoring/docker-compose.monitoring.yml logs grafana
   ```

2. **Verify data source**:
   - Go to Grafana → Configuration → Data Sources
   - Test the Prometheus connection

3. **Check Prometheus targets**:
   - Go to Prometheus → Status → Targets
   - Verify all targets are "UP"

### Performance Issues

1. **Reduce scrape frequency**:
   ```yaml
   # In prometheus.yml
   scrape_interval: 60s  # Increase from 30s
   ```

2. **Limit metrics retention**:
   ```yaml
   # In docker-compose.monitoring.yml
   command:
     - '--storage.tsdb.retention.time=7d'
   ```

3. **Optimize queries**:
   - Use recording rules for complex calculations
   - Add database indexes on audit log tables

## Production Considerations

### Security

1. **Restrict metrics access**:
   ```yaml
   # Only expose metrics internally
   services:
     parlant:
       expose:
         - "8080"  # Don't use 'ports' for metrics
   ```

2. **Enable authentication**:
   ```python
   # Add authentication to metrics endpoint
   @app.get("/metrics")
   async def metrics(api_key: str = Header(...)):
       if api_key != os.getenv("METRICS_API_KEY"):
           raise HTTPException(401, "Invalid API key")
       return Response(get_prometheus_metrics(), media_type="text/plain")
   ```

### Scalability

1. **Use external Prometheus**:
   - Deploy Prometheus on dedicated infrastructure
   - Configure remote write for long-term storage

2. **Implement caching**:
   ```python
   # Cache metrics for 30 seconds
   from functools import lru_cache
   import time
   
   @lru_cache(maxsize=1)
   def get_cached_metrics(timestamp: int):
       return get_prometheus_metrics()
   
   @app.get("/metrics")
   async def metrics():
       # Cache for 30 seconds
       cache_key = int(time.time() // 30)
       return Response(get_cached_metrics(cache_key), media_type="text/plain")
   ```

### Monitoring

1. **Set up alerts**:
   - Configure alert manager for critical issues
   - Set up notification channels (email, Slack, PagerDuty)

2. **Monitor the monitoring**:
   - Alert if metrics exporter goes down
   - Monitor Prometheus/Grafana resource usage

## Next Steps

1. **Customize dashboards** based on your specific needs
2. **Set up alerting** for critical performance thresholds
3. **Integrate with existing monitoring** infrastructure
4. **Add business-specific metrics** as needed
5. **Set up automated reporting** for stakeholders

For more detailed information, see:
- [Monitoring README](README.md)
- [MOR/Product Type Router Spec](../.kiro/specs/mor-product-type-router/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)