# Gemini Context Caching Operations Runbook

This runbook provides step-by-step procedures for managing the Gemini Context Caching system in production.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Weekly Operations](#weekly-operations)
3. [Monthly Operations](#monthly-operations)
4. [Emergency Procedures](#emergency-procedures)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Rollback Procedures](#rollback-procedures)
7. [Performance Optimization](#performance-optimization)
8. [Configuration Management](#configuration-management)

## Daily Operations

### Morning Health Check

**Frequency**: Every morning at 9:00 AM
**Duration**: 5-10 minutes
**Owner**: Operations Team

#### Steps

1. **Run Health Check Script**
   ```bash
   ./monitoring/cache-health-check.sh
   ```

2. **Review Dashboard**
   - Open Grafana: http://localhost:3001/d/gemini-cache
   - Check overall status indicators (green = healthy)
   - Verify cache effectiveness > 80%
   - Confirm cost savings are accumulating

3. **Check Alerts**
   ```bash
   curl http://localhost:8080/cache/health/effectiveness | jq '.alerts'
   ```

4. **Validate Key Metrics**
   - Cache Hit Rate: Should be > 95%
   - Cache Effectiveness: Should be > 80%
   - Content Loading P95: Should be < 3000ms
   - Prompt Construction P95: Should be < 10ms
   - Memory Usage: Should be < 15MB

#### Expected Results

- All health checks pass
- No active alerts
- Cache effectiveness within target range
- Performance metrics within SLA

#### Escalation

If any checks fail:
1. Check the [Troubleshooting Guide](#troubleshooting-guide)
2. If unresolved within 30 minutes, escalate to Development Team
3. If cache effectiveness < 50%, follow [Emergency Procedures](#emergency-procedures)

### Cost Savings Validation

**Frequency**: Daily at 5:00 PM
**Duration**: 10-15 minutes
**Owner**: Operations Team

#### Steps

1. **Check Daily Savings**
   ```bash
   curl http://localhost:8080/cache/health/effectiveness | jq '.daily_savings_usd'
   ```

2. **Compare Against Targets**
   - Current scale (300 calls/day): Target $12-15/day
   - Production scale (1000 tickets/day): Target $40-50/day

3. **Review Cost Trends**
   - Open Grafana cost savings panel
   - Check for any unusual drops or spikes
   - Validate against expected request volume

4. **Document Results**
   - Record daily savings in operations log
   - Note any anomalies or concerns
   - Update monthly cost tracking spreadsheet

#### Escalation

If daily savings are < 80% of target:
1. Check cache effectiveness metrics
2. Review recent configuration changes
3. Escalate to Development Team if no obvious cause

### Performance Monitoring

**Frequency**: Twice daily (morning and evening)
**Duration**: 5 minutes
**Owner**: Operations Team

#### Steps

1. **Check Performance Metrics**
   ```bash
   curl http://localhost:8080/cache/health/performance
   ```

2. **Review Key Indicators**
   - Content Loading P95 < 3000ms
   - Prompt Construction P95 < 10ms
   - Memory Usage < 15MB
   - No fallback mode activation

3. **Check for Performance Alerts**
   - Review Grafana alerts panel
   - Check Prometheus alert manager (if configured)

#### Escalation

If performance degrades beyond targets:
1. Follow [Performance Optimization](#performance-optimization) procedures
2. If degradation persists > 1 hour, escalate to Development Team

## Weekly Operations

### Cache Effectiveness Analysis

**Frequency**: Every Monday at 10:00 AM
**Duration**: 30-45 minutes
**Owner**: Operations Team + Development Team

#### Steps

1. **Generate Weekly Report**
   ```bash
   curl http://localhost:8080/cache/dashboard | jq '.trends.hourly_metrics' > weekly_cache_report.json
   ```

2. **Analyze Trends**
   - Review cache effectiveness over the week
   - Identify any degradation patterns
   - Check correlation with request volume changes

3. **Review Content Updates**
   - Check for any content loading failures
   - Review fallback mode activations
   - Validate static content freshness

4. **Performance Analysis**
   - Review P95 latencies for content loading and prompt construction
   - Check memory usage trends
   - Identify any performance bottlenecks

5. **Cost Validation**
   - Compare calculated savings with actual Gemini billing
   - Update cost projections based on actual usage
   - Validate ROI calculations

#### Actions

Based on analysis results:
- **Cache effectiveness declining**: Review prompt structure optimization
- **Performance degrading**: Follow performance optimization procedures
- **Cost savings below target**: Investigate cache configuration issues
- **Content loading issues**: Review content management procedures

### Configuration Review

**Frequency**: Every Wednesday at 2:00 PM
**Duration**: 20-30 minutes
**Owner**: Development Team

#### Steps

1. **Validate Current Configuration**
   ```bash
   curl http://localhost:8080/cache/config/validate
   ```

2. **Review Configuration Changes**
   - Check git history for recent config changes
   - Validate all changes were properly tested
   - Ensure documentation is up to date

3. **Test Hot Reload Functionality**
   ```bash
   curl -X POST http://localhost:8080/cache/config/reload
   ```

4. **Update Configuration Documentation**
   - Document any new configuration parameters
   - Update operational procedures if needed
   - Review and update alert thresholds

#### Actions

- Fix any configuration validation errors
- Update documentation for any changes
- Plan configuration optimizations based on operational experience

### Content Management Review

**Frequency**: Every Friday at 11:00 AM
**Duration**: 15-20 minutes
**Owner**: Operations Team

#### Steps

1. **Check Content Loading Status**
   ```bash
   curl http://localhost:8080/cache/health/content
   ```

2. **Review Content Update History**
   - Check for any recent policy document updates
   - Validate tool definition changes
   - Review few-shot example modifications

3. **Test Content Reload**
   ```bash
   curl -X POST http://localhost:8080/cache/content/reload
   ```

4. **Validate Content Integrity**
   - Check content hash changes
   - Verify token counts are reasonable
   - Ensure no content corruption

#### Actions

- Update static content if needed
- Fix any content loading issues
- Plan content optimization based on cache effectiveness

## Monthly Operations

### Performance Optimization Review

**Frequency**: First Monday of each month
**Duration**: 2-3 hours
**Owner**: Development Team + Operations Team

#### Steps

1. **Comprehensive Performance Analysis**
   - Review monthly performance trends
   - Identify bottlenecks and optimization opportunities
   - Analyze memory usage patterns

2. **Cache Effectiveness Deep Dive**
   - Analyze prompt structure effectiveness
   - Review content optimization opportunities
   - Identify patterns in low-effectiveness requests

3. **Cost Analysis and ROI Validation**
   - Compare actual costs with projections
   - Update ROI calculations based on real data
   - Plan scaling cost projections

4. **Infrastructure Review**
   - Review monitoring infrastructure performance
   - Plan capacity increases if needed
   - Update alerting thresholds based on operational data

#### Deliverables

- Monthly performance report
- Updated cost projections
- Performance optimization recommendations
- Infrastructure scaling plan (if needed)

### Configuration and Security Review

**Frequency**: Second Monday of each month
**Duration**: 1-2 hours
**Owner**: Development Team + Security Team

#### Steps

1. **Security Configuration Review**
   - Review API key rotation schedule
   - Validate access controls on monitoring endpoints
   - Check audit log retention and security

2. **Configuration Optimization**
   - Review alert thresholds based on operational experience
   - Optimize configuration parameters based on performance data
   - Update default values based on lessons learned

3. **Documentation Updates**
   - Update operational procedures based on experience
   - Review and update troubleshooting guides
   - Update emergency contact information

#### Deliverables

- Security review report
- Updated configuration recommendations
- Updated operational documentation

## Emergency Procedures

### Cache Effectiveness Critical (< 50%)

**Severity**: Critical
**Response Time**: Immediate
**Owner**: On-call Engineer

#### Immediate Actions (0-15 minutes)

1. **Check System Status**
   ```bash
   ./monitoring/cache-health-check.sh
   ```

2. **Identify Root Cause**
   - Check for fallback mode activation
   - Review recent configuration changes
   - Check content loading failures

3. **Immediate Mitigation**
   ```bash
   # Force content reload
   curl -X POST http://localhost:8080/cache/content/reload
   
   # Check if effectiveness improves
   curl http://localhost:8080/cache/health/effectiveness
   ```

#### Short-term Actions (15-60 minutes)

1. **If Content Loading Failed**
   ```bash
   # Run content loading rollback
   ./monitoring/cache/runbooks/rollback-content-loading.sh
   ```

2. **If Configuration Issues**
   ```bash
   # Run configuration rollback
   ./monitoring/cache/runbooks/rollback-configuration.sh
   ```

3. **If Performance Issues**
   ```bash
   # Run performance rollback
   ./monitoring/cache/runbooks/rollback-performance.sh
   ```

#### Long-term Actions (1-4 hours)

1. **Root Cause Analysis**
   - Review logs for error patterns
   - Analyze recent changes
   - Identify systemic issues

2. **Permanent Fix**
   - Implement proper fix for root cause
   - Test fix in staging environment
   - Deploy fix with monitoring

3. **Post-Incident Review**
   - Document incident timeline
   - Identify process improvements
   - Update procedures to prevent recurrence

### Fallback Mode Activation

**Severity**: High
**Response Time**: 30 minutes
**Owner**: On-call Engineer

#### Immediate Actions

1. **Assess Impact**
   ```bash
   curl http://localhost:8080/cache/health/content
   ```

2. **Check Content Loading Status**
   - Identify which content types failed
   - Check file system and permissions
   - Review error logs

3. **Attempt Content Recovery**
   ```bash
   # Force content reload
   curl -X POST http://localhost:8080/cache/content/reload
   
   # Wait 30 seconds and check status
   sleep 30
   curl http://localhost:8080/cache/health/content
   ```

#### If Recovery Fails

1. **Manual Content Verification**
   ```bash
   # Check content files exist and are readable
   ls -la /app/parlant/context/processed/
   
   # Check file permissions
   stat /app/parlant/context/processed/*.md
   ```

2. **Restart Cache System**
   ```bash
   docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
   ```

3. **Escalate if Unresolved**
   - Contact Development Team
   - Provide detailed error logs
   - Document all attempted recovery steps

### Performance Degradation Critical

**Severity**: High
**Response Time**: 1 hour
**Owner**: On-call Engineer

#### Immediate Actions

1. **Check System Resources**
   ```bash
   docker stats
   free -h
   df -h
   ```

2. **Review Performance Metrics**
   ```bash
   curl http://localhost:8080/cache/health/performance
   ```

3. **Identify Bottleneck**
   - Content loading > 5000ms: File system issue
   - Prompt construction > 50ms: Memory/CPU issue
   - Memory usage > 20MB: Memory leak

#### Mitigation Steps

1. **For File System Issues**
   ```bash
   # Check disk space and I/O
   iostat -x 1 5
   
   # Check for file system errors
   dmesg | grep -i error
   ```

2. **For Memory Issues**
   ```bash
   # Restart cache system to clear memory
   docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
   ```

3. **For CPU Issues**
   ```bash
   # Check CPU usage patterns
   top -p $(pgrep -f cache_metrics_exporter)
   ```

#### Escalation

If performance doesn't improve within 1 hour:
1. Temporarily disable cache monitoring
2. Contact Development Team
3. Prepare for potential system restart

## Troubleshooting Guide

### Low Cache Effectiveness (< 80%)

#### Symptoms
- Cache effectiveness below 80%
- Reduced cost savings
- Alerts firing for low effectiveness

#### Diagnosis Steps

1. **Check Prompt Structure**
   ```bash
   curl http://localhost:8080/cache/debug/prompt-structure
   ```

2. **Verify Content Loading**
   ```bash
   curl http://localhost:8080/cache/health/content
   ```

3. **Review Recent Changes**
   - Check for content updates
   - Review configuration changes
   - Check for tool definition modifications

#### Resolution Steps

1. **Content Issues**
   ```bash
   # Reload content
   curl -X POST http://localhost:8080/cache/content/reload
   ```

2. **Configuration Issues**
   ```bash
   # Validate configuration
   curl http://localhost:8080/cache/config/validate
   
   # Reload if needed
   curl -X POST http://localhost:8080/cache/config/reload
   ```

3. **Prompt Structure Issues**
   - Review prompt template structure
   - Ensure static content comes first
   - Verify dynamic content placement

### Content Loading Failures

#### Symptoms
- Fallback mode active
- Content loading errors in logs
- Missing or corrupted content

#### Diagnosis Steps

1. **Check File System**
   ```bash
   ls -la /app/parlant/context/processed/
   stat /app/parlant/context/processed/*.md
   ```

2. **Check Permissions**
   ```bash
   # Verify read permissions
   test -r /app/parlant/context/processed/refund_policy_condensed.md && echo "OK" || echo "FAIL"
   ```

3. **Check Disk Space**
   ```bash
   df -h
   ```

#### Resolution Steps

1. **Fix Permissions**
   ```bash
   chmod 644 /app/parlant/context/processed/*.md
   ```

2. **Restore Missing Files**
   ```bash
   # Copy from backup if available
   cp /backup/parlant/context/processed/*.md /app/parlant/context/processed/
   ```

3. **Force Reload**
   ```bash
   curl -X POST http://localhost:8080/cache/content/reload
   ```

### Performance Issues

#### Symptoms
- High latency (> targets)
- Slow response times
- Performance alerts firing

#### Diagnosis Steps

1. **Check System Resources**
   ```bash
   docker stats
   top
   iostat -x 1 5
   ```

2. **Review Performance Metrics**
   ```bash
   curl http://localhost:8080/cache/health/performance
   ```

3. **Check for Memory Leaks**
   ```bash
   # Monitor memory usage over time
   watch -n 5 'curl -s http://localhost:8080/cache/health/performance | jq .memory_usage_mb'
   ```

#### Resolution Steps

1. **Restart Services**
   ```bash
   docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart
   ```

2. **Optimize Configuration**
   - Reduce content size if needed
   - Adjust timeout values
   - Optimize memory usage

3. **Scale Resources**
   - Increase container memory limits
   - Add CPU resources if needed

## Rollback Procedures

### Content Loading Rollback

**Use Case**: Content loading failures, corrupted content
**Script**: `./monitoring/cache/runbooks/rollback-content-loading.sh`

#### Manual Steps

1. **Check Fallback Status**
   ```bash
   curl http://localhost:8080/cache/health/content | jq '.fallback_mode_active'
   ```

2. **Force Content Reload**
   ```bash
   curl -X POST http://localhost:8080/cache/content/reload
   ```

3. **Verify Recovery**
   ```bash
   sleep 10
   curl http://localhost:8080/cache/health/content | jq '.status'
   ```

4. **If Still Failing**
   ```bash
   # Restore from backup
   cp /backup/parlant/context/processed/*.md /app/parlant/context/processed/
   curl -X POST http://localhost:8080/cache/content/reload
   ```

### Configuration Rollback

**Use Case**: Configuration errors, invalid settings
**Script**: `./monitoring/cache/runbooks/rollback-configuration.sh`

#### Manual Steps

1. **Backup Current Config**
   ```bash
   cp /app/cache_config.json /app/cache_config.json.backup
   ```

2. **Restore Previous Config**
   ```bash
   cp /app/cache_config.json.previous /app/cache_config.json
   ```

3. **Reload Configuration**
   ```bash
   curl -X POST http://localhost:8080/cache/config/reload
   ```

4. **Validate**
   ```bash
   curl http://localhost:8080/cache/config/validate
   ```

### Performance Rollback

**Use Case**: Performance degradation, resource issues
**Script**: `./monitoring/cache/runbooks/rollback-performance.sh`

#### Manual Steps

1. **Check Performance**
   ```bash
   curl http://localhost:8080/cache/health/performance
   ```

2. **Disable Monitoring Temporarily**
   ```bash
   curl -X POST http://localhost:8080/cache/config/update \
        -H "Content-Type: application/json" \
        -d '{"monitor_cache_effectiveness": false}'
   ```

3. **Restart Services**
   ```bash
   docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart
   ```

4. **Re-enable When Stable**
   ```bash
   curl -X POST http://localhost:8080/cache/config/update \
        -H "Content-Type: application/json" \
        -d '{"monitor_cache_effectiveness": true}'
   ```

## Performance Optimization

### Content Loading Optimization

#### Identify Issues
- Content loading P95 > 3000ms
- High rate of slow content loads
- Memory usage increasing

#### Optimization Steps

1. **Optimize File I/O**
   ```bash
   # Check I/O patterns
   iostat -x 1 10
   
   # Optimize file system if needed
   # Consider SSD storage for content files
   ```

2. **Implement Caching**
   - Cache loaded content in memory
   - Use content hash for change detection
   - Implement lazy loading for large files

3. **Parallel Loading**
   - Load content types in parallel
   - Use async I/O operations
   - Implement timeout handling

### Prompt Construction Optimization

#### Identify Issues
- Prompt construction P95 > 10ms
- High CPU usage during construction
- Memory allocation issues

#### Optimization Steps

1. **Template Optimization**
   - Pre-compile templates
   - Cache formatted templates
   - Minimize string operations

2. **Memory Management**
   - Reuse string buffers
   - Implement object pooling
   - Optimize garbage collection

3. **Algorithm Optimization**
   - Optimize content ordering logic
   - Use efficient data structures
   - Minimize computational complexity

### Memory Usage Optimization

#### Identify Issues
- Memory usage > 15MB
- Memory leaks detected
- Frequent garbage collection

#### Optimization Steps

1. **Memory Profiling**
   ```bash
   # Profile memory usage
   python -m memory_profiler cache_metrics_exporter.py
   ```

2. **Optimize Data Structures**
   - Use memory-efficient data types
   - Implement data compression
   - Remove unnecessary data retention

3. **Garbage Collection Tuning**
   - Optimize GC parameters
   - Implement manual memory management
   - Use weak references where appropriate

## Configuration Management

### Configuration Validation

#### Pre-Deployment Validation

```bash
#!/bin/bash
# validate-config.sh

echo "Validating cache configuration..."

# Check configuration syntax
if ! curl -s http://localhost:8080/cache/config/validate | jq -e '.status == "valid"' > /dev/null; then
    echo "❌ Configuration validation failed"
    exit 1
fi

# Check required parameters
CONFIG=$(curl -s http://localhost:8080/cache/config/validate)

# Validate pricing
INPUT_COST=$(echo $CONFIG | jq -r '.pricing.input_cost_per_1m')
CACHED_COST=$(echo $CONFIG | jq -r '.pricing.cached_input_cost_per_1m')

if (( $(echo "$CACHED_COST >= $INPUT_COST" | bc -l) )); then
    echo "❌ Cached cost should be less than input cost"
    exit 1
fi

# Validate limits
MAX_TOKENS=$(echo $CONFIG | jq -r '.limits.max_static_content_tokens')
if [ "$MAX_TOKENS" -le 0 ]; then
    echo "❌ Max tokens must be positive"
    exit 1
fi

echo "✅ Configuration validation passed"
```

#### Hot Reload Testing

```bash
#!/bin/bash
# test-hot-reload.sh

echo "Testing configuration hot reload..."

# Get current config
ORIGINAL_CONFIG=$(curl -s http://localhost:8080/cache/config/validate)

# Make a test change
curl -X POST http://localhost:8080/cache/config/update \
     -H "Content-Type: application/json" \
     -d '{"cache_effectiveness_alert_threshold": 75.0}'

# Reload configuration
curl -X POST http://localhost:8080/cache/config/reload

# Verify change
NEW_CONFIG=$(curl -s http://localhost:8080/cache/config/validate)
NEW_THRESHOLD=$(echo $NEW_CONFIG | jq -r '.cache_effectiveness_alert_threshold')

if [ "$NEW_THRESHOLD" = "75" ]; then
    echo "✅ Hot reload successful"
else
    echo "❌ Hot reload failed"
    exit 1
fi

# Restore original configuration
curl -X POST http://localhost:8080/cache/config/update \
     -H "Content-Type: application/json" \
     -d "$ORIGINAL_CONFIG"

echo "✅ Configuration restored"
```

### Environment-Specific Configuration

#### Development Environment

```json
{
  "context_caching": {
    "enabled": true,
    "monitor_cache_effectiveness": true,
    "cache_effectiveness_alert_threshold": 60.0,
    "hot_reload_enabled": true,
    "pricing": {
      "input_cost_per_1m": 0.30,
      "cached_input_cost_per_1m": 0.03
    }
  }
}
```

#### Staging Environment

```json
{
  "context_caching": {
    "enabled": true,
    "monitor_cache_effectiveness": true,
    "cache_effectiveness_alert_threshold": 70.0,
    "hot_reload_enabled": true,
    "pricing": {
      "input_cost_per_1m": 0.30,
      "cached_input_cost_per_1m": 0.03
    }
  }
}
```

#### Production Environment

```json
{
  "context_caching": {
    "enabled": true,
    "monitor_cache_effectiveness": true,
    "cache_effectiveness_alert_threshold": 80.0,
    "hot_reload_enabled": false,
    "pricing": {
      "input_cost_per_1m": 0.30,
      "cached_input_cost_per_1m": 0.03
    }
  }
}
```

## Contact Information

### Emergency Contacts

- **On-Call Engineer**: +1-555-0123 (24/7)
- **Operations Team Lead**: ops-lead@company.com
- **Development Team Lead**: dev-lead@company.com
- **Infrastructure Team**: infrastructure@company.com

### Escalation Matrix

1. **Level 1**: On-Call Engineer (immediate response)
2. **Level 2**: Operations Team Lead (30 minutes)
3. **Level 3**: Development Team Lead (1 hour)
4. **Level 4**: Engineering Manager (2 hours)

### Communication Channels

- **Slack**: #cache-operations
- **Email**: cache-alerts@company.com
- **Incident Management**: PagerDuty/ServiceNow
- **Documentation**: Confluence/Wiki

## Appendix

### Useful Commands Reference

```bash
# Health checks
./monitoring/cache-health-check.sh
curl http://localhost:8080/cache/health
curl http://localhost:8080/cache/health/content
curl http://localhost:8080/cache/health/effectiveness
curl http://localhost:8080/cache/health/performance

# Configuration management
curl http://localhost:8080/cache/config/validate
curl -X POST http://localhost:8080/cache/config/reload
curl -X POST http://localhost:8080/cache/content/reload

# Metrics and monitoring
curl http://localhost:8080/cache/metrics
curl http://localhost:8080/cache/metrics/json
curl http://localhost:8080/cache/dashboard

# Service management
docker-compose -f monitoring/docker-compose.cache-monitoring.yml ps
docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs
docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart
```

### Log Locations

- **Cache Metrics Exporter**: `docker logs cache-metrics-exporter`
- **Prometheus**: `docker logs cache-prometheus`
- **Grafana**: `docker logs cache-grafana`
- **Application Logs**: `/app/logs/cache.log`

### Monitoring URLs

- **Grafana Dashboard**: http://localhost:3001/d/gemini-cache
- **Prometheus**: http://localhost:9091
- **Cache Health**: http://localhost:8080/cache/health
- **Cache Metrics**: http://localhost:8080/cache/metrics

---

**Document Version**: 1.0
**Last Updated**: January 2025
**Next Review**: February 2025