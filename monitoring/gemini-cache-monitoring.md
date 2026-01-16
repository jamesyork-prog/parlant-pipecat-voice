# Gemini Context Caching Monitoring Setup

This document provides comprehensive monitoring, alerting, and operational procedures for the Gemini Context Caching system.

## Overview

The Gemini Context Caching monitoring system provides visibility into:

- **Cache Effectiveness**: Token cache rates, cost savings, and optimization opportunities
- **Performance Metrics**: Content loading times, prompt construction latency, and throughput
- **Operational Health**: Content loading failures, fallback mode activation, and system status
- **Cost Analytics**: Real-time cost savings, ROI tracking, and billing validation
- **Configuration Management**: Hot reload status, validation errors, and deployment health

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Cache System  │───▶│ Metrics         │───▶│   Prometheus    │
│   (Gemini)      │    │ Tracker         │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   PostgreSQL    │───▶│ Audit Logger    │             │
│   Database      │    │ & Metrics       │             │
└─────────────────┘    └─────────────────┘             │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Grafana       │◀───│   Dashboard     │
                       │   Dashboard     │    │   & Alerts      │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Setup Cache Monitoring

```bash
# Run the cache monitoring setup
./monitoring/setup-cache-monitoring.sh

# Or manually start services
docker-compose -f monitoring/docker-compose.cache-monitoring.yml up -d
```

### 2. Access Dashboards

- **Cache Dashboard**: http://localhost:3000/d/gemini-cache (admin/admin)
- **Prometheus**: http://localhost:9090
- **Cache Metrics**: http://localhost:8080/cache/metrics
- **Health Check**: http://localhost:8080/cache/health

### 3. Verify Setup

```bash
# Run health check
./monitoring/cache-health-check.sh

# Check metrics collection
curl http://localhost:8080/cache/metrics | grep cache_
```

## Dashboard Features

### Cache Effectiveness Overview
- **Cache Hit Rate**: Percentage of requests with cached tokens
- **Average Cache Effectiveness**: Percentage of tokens cached per request
- **Total Cost Savings**: Cumulative USD savings from implicit caching
- **Daily Savings Trend**: Cost savings over time

### Performance Monitoring
- **Content Loading Time**: P95 latency for static content loading (<3s target)
- **Prompt Construction Time**: P95 latency for prompt building (<10ms target)
- **Cache Response Processing**: Time to extract and calculate cache metrics
- **Memory Usage**: Static content memory footprint (<15MB target)

### Operational Health
- **Content Loading Status**: Success/failure rates for static content types
- **Fallback Mode Status**: Current mode and activation history
- **Configuration Status**: Hot reload events and validation errors
- **Alert Summary**: Active alerts and recent alert history

### Cost Analytics
- **Hourly Cost Savings**: Detailed cost breakdown by hour
- **Cache Effectiveness Distribution**: Histogram of cache effectiveness rates
- **Token Usage Breakdown**: Cached vs dynamic token distribution
- **ROI Tracking**: Break-even analysis and projected annual savings

## Metrics Reference

### Core Cache Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| `cache_effectiveness_percent` | Percentage of tokens cached per request | >80% |
| `cache_hit_rate_percent` | Percentage of requests with cached tokens | >95% |
| `cache_cost_savings_usd_total` | Cumulative cost savings | $148k/year |
| `cache_content_loading_time_ms_p95` | Content loading P95 latency | <3000ms |
| `cache_prompt_construction_time_ms_p95` | Prompt construction P95 latency | <10ms |

### Performance Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `cache_slow_content_loads_total` | Content loads >3s | >20% of loads |
| `cache_slow_prompt_constructions_total` | Prompt constructions >10ms | >10% of constructions |
| `cache_memory_usage_mb` | Static content memory usage | >15MB |
| `cache_fallback_mode_active` | Fallback mode status | 1 (active) |

### Business Metrics

| Metric | Description | Business Value |
|--------|-------------|----------------|
| `cache_daily_savings_usd` | Daily cost savings | ROI measurement |
| `cache_monthly_projection_usd` | Monthly savings projection | Budget planning |
| `cache_yearly_projection_usd` | Yearly savings projection | Business case validation |
| `cache_break_even_days` | Days to break even | Investment recovery |

## Alerting Rules

### Performance Alerts

- **CacheSlowContentLoading**: Triggers when content loading P95 > 3s for 5 minutes
- **CacheSlowPromptConstruction**: Triggers when prompt construction P95 > 10ms for 5 minutes
- **CacheHighMemoryUsage**: Triggers when memory usage > 15MB for 10 minutes

### Effectiveness Alerts

- **CacheLowEffectiveness**: Triggers when cache effectiveness < 70% for 10 minutes
- **CacheLowHitRate**: Triggers when cache hit rate < 80% for 15 minutes
- **CacheNoSavings**: Triggers when no cost savings for 30 minutes

### Operational Alerts

- **CacheFallbackModeActive**: Triggers immediately when fallback mode activates
- **CacheContentLoadingFailures**: Triggers when >20% of content loads fail
- **CacheConfigurationErrors**: Triggers on configuration validation failures

## Health Checks

### Content Loading Health

```bash
# Check content loading status
curl http://localhost:8080/cache/health/content

# Expected response:
{
  "status": "healthy",
  "static_content_loaded": true,
  "content_types": {
    "system_identity": "loaded",
    "refund_policies": "loaded", 
    "tool_definitions": "loaded",
    "few_shot_examples": "loaded"
  },
  "total_tokens": 5000,
  "loading_time_ms": 1250,
  "last_updated": "2025-01-09T10:30:00Z"
}
```

### Cache Effectiveness Health

```bash
# Check cache effectiveness
curl http://localhost:8080/cache/health/effectiveness

# Expected response:
{
  "status": "healthy",
  "cache_hit_rate": 95.2,
  "avg_cache_effectiveness": 82.1,
  "recent_requests": 150,
  "total_savings_usd": 45.67,
  "daily_savings_usd": 12.34,
  "alerts": []
}
```

### Performance Health

```bash
# Check performance metrics
curl http://localhost:8080/cache/health/performance

# Expected response:
{
  "status": "healthy",
  "content_loading": {
    "avg_time_ms": 1200,
    "p95_time_ms": 2800,
    "slow_loads": 2,
    "total_loads": 45
  },
  "prompt_construction": {
    "avg_time_ms": 6.5,
    "p95_time_ms": 9.2,
    "slow_constructions": 1,
    "total_constructions": 150
  },
  "memory_usage_mb": 12.3
}
```

## Configuration Validation

### Startup Validation

```bash
# Validate configuration at startup
curl http://localhost:8080/cache/config/validate

# Expected response:
{
  "status": "valid",
  "enabled": true,
  "monitor_cache_effectiveness": true,
  "pricing": {
    "input_cost_per_1m": 0.30,
    "cached_input_cost_per_1m": 0.03,
    "discount_percentage": 90.0
  },
  "limits": {
    "max_static_content_tokens": 32000,
    "prompt_construction_timeout_ms": 10,
    "content_loading_timeout_s": 3
  },
  "validation_errors": []
}
```

### Hot Reload Status

```bash
# Check hot reload capability
curl http://localhost:8080/cache/config/reload

# Expected response:
{
  "status": "reloaded",
  "config_changed": true,
  "reload_time": "2025-01-09T10:35:00Z",
  "changes": [
    "cache_effectiveness_alert_threshold: 70.0 -> 75.0"
  ]
}
```

## Deployment Checks

### Pre-Deployment Validation

```bash
#!/bin/bash
# pre-deployment-check.sh

echo "🔍 Running Gemini Cache pre-deployment checks..."

# Check configuration
echo "📋 Validating configuration..."
if ! curl -s http://localhost:8080/cache/config/validate | jq -e '.status == "valid"' > /dev/null; then
    echo "❌ Configuration validation failed"
    exit 1
fi

# Check content loading
echo "📁 Checking content loading..."
if ! curl -s http://localhost:8080/cache/health/content | jq -e '.status == "healthy"' > /dev/null; then
    echo "❌ Content loading health check failed"
    exit 1
fi

# Check performance
echo "⚡ Checking performance..."
if ! curl -s http://localhost:8080/cache/health/performance | jq -e '.status == "healthy"' > /dev/null; then
    echo "❌ Performance health check failed"
    exit 1
fi

echo "✅ All pre-deployment checks passed"
```

### Post-Deployment Validation

```bash
#!/bin/bash
# post-deployment-check.sh

echo "🚀 Running Gemini Cache post-deployment checks..."

# Wait for system to stabilize
sleep 30

# Check cache effectiveness
echo "💰 Checking cache effectiveness..."
EFFECTIVENESS=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.avg_cache_effectiveness')
if (( $(echo "$EFFECTIVENESS < 70" | bc -l) )); then
    echo "⚠️  Cache effectiveness below target: $EFFECTIVENESS%"
fi

# Check cost savings
echo "💵 Checking cost savings..."
SAVINGS=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.daily_savings_usd')
if (( $(echo "$SAVINGS < 1" | bc -l) )); then
    echo "⚠️  Daily savings below expected: $SAVINGS USD"
fi

# Check alerts
echo "🚨 Checking for alerts..."
ALERTS=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.alerts | length')
if [ "$ALERTS" -gt 0 ]; then
    echo "⚠️  $ALERTS active alerts detected"
fi

echo "✅ Post-deployment checks completed"
```

## Rollback Procedures

### Content Loading Rollback

```bash
#!/bin/bash
# rollback-content-loading.sh

echo "🔄 Rolling back content loading issues..."

# Check if fallback mode is active
FALLBACK=$(curl -s http://localhost:8080/cache/health/content | jq -r '.fallback_mode_active')

if [ "$FALLBACK" = "true" ]; then
    echo "⚠️  System is in fallback mode"
    
    # Force content reload
    echo "🔄 Forcing content reload..."
    curl -X POST http://localhost:8080/cache/content/reload
    
    # Wait and check
    sleep 10
    STATUS=$(curl -s http://localhost:8080/cache/health/content | jq -r '.status')
    
    if [ "$STATUS" = "healthy" ]; then
        echo "✅ Content loading restored"
    else
        echo "❌ Content loading still failing - manual intervention required"
        exit 1
    fi
else
    echo "✅ Content loading is healthy"
fi
```

### Configuration Rollback

```bash
#!/bin/bash
# rollback-configuration.sh

echo "🔄 Rolling back configuration changes..."

# Backup current config
cp /app/cache_config.json /app/cache_config.json.backup

# Restore previous config
if [ -f "/app/cache_config.json.previous" ]; then
    cp /app/cache_config.json.previous /app/cache_config.json
    echo "📋 Configuration restored from backup"
    
    # Trigger hot reload
    curl -X POST http://localhost:8080/cache/config/reload
    
    # Validate
    sleep 5
    if curl -s http://localhost:8080/cache/config/validate | jq -e '.status == "valid"' > /dev/null; then
        echo "✅ Configuration rollback successful"
    else
        echo "❌ Configuration rollback failed"
        exit 1
    fi
else
    echo "❌ No previous configuration backup found"
    exit 1
fi
```

### Performance Rollback

```bash
#!/bin/bash
# rollback-performance.sh

echo "🔄 Rolling back performance issues..."

# Check current performance
CONTENT_P95=$(curl -s http://localhost:8080/cache/health/performance | jq -r '.content_loading.p95_time_ms')
PROMPT_P95=$(curl -s http://localhost:8080/cache/health/performance | jq -r '.prompt_construction.p95_time_ms')

# If performance is degraded, disable cache monitoring temporarily
if (( $(echo "$CONTENT_P95 > 5000" | bc -l) )) || (( $(echo "$PROMPT_P95 > 20" | bc -l) )); then
    echo "⚠️  Performance degraded - temporarily disabling cache monitoring"
    
    # Update config to disable monitoring
    curl -X POST http://localhost:8080/cache/config/update \
        -H "Content-Type: application/json" \
        -d '{"monitor_cache_effectiveness": false}'
    
    echo "🔄 Cache monitoring disabled - system will use basic prompts"
    echo "📞 Contact operations team for performance investigation"
else
    echo "✅ Performance is within acceptable limits"
fi
```

## Operational Runbooks

### Daily Operations

1. **Morning Health Check**:
   ```bash
   ./monitoring/cache-health-check.sh
   ```

2. **Review Cost Savings**:
   - Check Grafana dashboard for daily savings
   - Validate against expected targets ($400-500/day at scale)
   - Alert if savings drop below 80% of target

3. **Performance Review**:
   - Check P95 latencies for content loading and prompt construction
   - Review any performance alerts from previous 24 hours
   - Investigate slow operations if >10% exceed targets

### Weekly Operations

1. **Cache Effectiveness Analysis**:
   - Review weekly cache effectiveness trends
   - Identify any degradation patterns
   - Optimize prompt structure if effectiveness drops

2. **Content Update Review**:
   - Check for any content loading failures
   - Review fallback mode activations
   - Update static content if needed

3. **Cost Validation**:
   - Compare calculated savings with actual Gemini billing
   - Adjust pricing configuration if rates change
   - Update ROI projections

### Monthly Operations

1. **Performance Optimization**:
   - Analyze performance trends and bottlenecks
   - Review memory usage patterns
   - Optimize content loading if needed

2. **Configuration Review**:
   - Review and update cache configuration
   - Test hot reload functionality
   - Update alert thresholds based on operational experience

3. **Capacity Planning**:
   - Project scaling requirements
   - Plan for increased request volume
   - Update monitoring infrastructure if needed

## Troubleshooting Guide

### Low Cache Effectiveness

**Symptoms**: Cache effectiveness < 70%, reduced cost savings

**Diagnosis**:
```bash
# Check prompt structure
curl http://localhost:8080/cache/debug/prompt-structure

# Check content loading
curl http://localhost:8080/cache/health/content

# Review recent requests
curl http://localhost:8080/cache/debug/recent-requests
```

**Resolution**:
1. Verify static content is loaded correctly
2. Check prompt structure (static first, dynamic last)
3. Review content changes that might affect caching
4. Restart content loader if needed

### Content Loading Failures

**Symptoms**: Fallback mode active, content loading errors

**Diagnosis**:
```bash
# Check content status
curl http://localhost:8080/cache/health/content

# Check file permissions and paths
ls -la /app/parlant/context/processed/

# Check audit logs
curl http://localhost:8080/cache/audit/content-loading
```

**Resolution**:
1. Verify file paths and permissions
2. Check disk space and file system health
3. Force content reload
4. Use rollback procedure if needed

### Performance Degradation

**Symptoms**: High latency, slow response times

**Diagnosis**:
```bash
# Check performance metrics
curl http://localhost:8080/cache/health/performance

# Check memory usage
curl http://localhost:8080/cache/debug/memory

# Check system resources
docker stats
```

**Resolution**:
1. Check system resource usage
2. Optimize content loading if needed
3. Temporarily disable monitoring if severe
4. Scale resources if necessary

## Integration with Existing Infrastructure

### Existing Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'gemini-cache'
    static_configs:
      - targets: ['your-app-host:8080']
    metrics_path: /cache/metrics
    scrape_interval: 30s
```

### Existing Grafana

1. Import dashboard: `monitoring/grafana/dashboards/gemini-cache-dashboard.json`
2. Configure data source to point to your Prometheus
3. Set up notification channels for alerts

### Existing Alerting

Import alert rules from `monitoring/prometheus/rules/cache_alerts.yml`

## Security Considerations

### Metrics Endpoint Security

```python
# Add authentication to metrics endpoints
@app.get("/cache/metrics")
async def cache_metrics(api_key: str = Header(...)):
    if api_key != os.getenv("CACHE_METRICS_API_KEY"):
        raise HTTPException(401, "Invalid API key")
    return Response(get_cache_metrics(), media_type="text/plain")
```

### Audit Log Security

- Ensure audit logs don't contain sensitive data
- Implement log rotation and retention policies
- Restrict access to audit endpoints

### Configuration Security

- Encrypt sensitive configuration values
- Use environment variables for secrets
- Implement configuration change auditing

## Support and Maintenance

### Regular Maintenance Tasks

1. **Daily**: Health checks, cost validation, performance review
2. **Weekly**: Cache effectiveness analysis, content updates
3. **Monthly**: Performance optimization, configuration review
4. **Quarterly**: Capacity planning, infrastructure updates

### Monitoring the Monitoring

Set up alerts for the monitoring system itself:

```yaml
# Add to cache_alerts.yml
- alert: CacheMonitoringDown
  expr: up{job="gemini-cache"} == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Gemini cache monitoring is down"
```

### Emergency Contacts

- **Operations Team**: For performance and availability issues
- **Development Team**: For cache effectiveness and configuration issues
- **Finance Team**: For cost validation and billing discrepancies

## References

- [Gemini Context Caching Spec](../.kiro/specs/gemini-context-caching/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Gemini API Documentation](https://ai.google.dev/docs)