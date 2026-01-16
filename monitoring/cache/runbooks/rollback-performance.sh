#!/bin/bash

# Gemini Context Caching Performance Rollback Script
# This script handles rollback procedures for performance issues

set -e

echo "🔄 Rolling back performance issues..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Performance thresholds
CONTENT_LOADING_THRESHOLD=3000  # 3 seconds in ms
PROMPT_CONSTRUCTION_THRESHOLD=10  # 10ms
MEMORY_THRESHOLD=15  # 15MB
CRITICAL_CONTENT_LOADING_THRESHOLD=5000  # 5 seconds
CRITICAL_PROMPT_CONSTRUCTION_THRESHOLD=50  # 50ms

# Check current performance
echo "📊 Checking current performance metrics..."
PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)

if [ -z "$PERFORMANCE" ]; then
    echo -e "${RED}❌ Unable to retrieve performance metrics${NC}"
    echo "🔧 Attempting service restart..."
    docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
    echo "⏳ Waiting for service to stabilize..."
    sleep 30
    PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)
fi

# Extract performance metrics
CONTENT_P95=$(echo "$PERFORMANCE" | jq -r '.content_loading.p95_time_ms // 0' 2>/dev/null)
PROMPT_P95=$(echo "$PERFORMANCE" | jq -r '.prompt_construction.p95_time_ms // 0' 2>/dev/null)
MEMORY_USAGE=$(echo "$PERFORMANCE" | jq -r '.memory_usage_mb // 0' 2>/dev/null)

echo "📈 Current performance metrics:"
echo "   • Content Loading P95: ${CONTENT_P95}ms (target: <${CONTENT_LOADING_THRESHOLD}ms)"
echo "   • Prompt Construction P95: ${PROMPT_P95}ms (target: <${PROMPT_CONSTRUCTION_THRESHOLD}ms)"
echo "   • Memory Usage: ${MEMORY_USAGE}MB (target: <${MEMORY_THRESHOLD}MB)"

# Determine severity and required actions
CRITICAL_PERFORMANCE=false
DEGRADED_PERFORMANCE=false

# Check for critical performance issues
if (( $(echo "$CONTENT_P95 > $CRITICAL_CONTENT_LOADING_THRESHOLD" | bc -l 2>/dev/null || echo "0") )) || \
   (( $(echo "$PROMPT_P95 > $CRITICAL_PROMPT_CONSTRUCTION_THRESHOLD" | bc -l 2>/dev/null || echo "0") )) || \
   (( $(echo "$MEMORY_USAGE > 20" | bc -l 2>/dev/null || echo "0") )); then
    CRITICAL_PERFORMANCE=true
    echo -e "${RED}🚨 Critical performance degradation detected${NC}"
elif (( $(echo "$CONTENT_P95 > $CONTENT_LOADING_THRESHOLD" | bc -l 2>/dev/null || echo "0") )) || \
     (( $(echo "$PROMPT_P95 > $PROMPT_CONSTRUCTION_THRESHOLD" | bc -l 2>/dev/null || echo "0") )) || \
     (( $(echo "$MEMORY_USAGE > $MEMORY_THRESHOLD" | bc -l 2>/dev/null || echo "0") )); then
    DEGRADED_PERFORMANCE=true
    echo -e "${YELLOW}⚠️  Performance degradation detected${NC}"
else
    echo -e "${GREEN}✅ Performance is within acceptable limits${NC}"
    echo "ℹ️  No rollback needed"
    exit 0
fi

# Check system resources
echo "🖥️  Checking system resources..."
echo "💾 Memory usage:"
free -h | head -2

echo "💿 Disk usage:"
df -h | grep -E "(Filesystem|/dev/)"

echo "⚡ CPU usage:"
top -bn1 | grep "Cpu(s)" | head -1

# Check Docker container resources
echo "🐳 Docker container resources:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -5

# Critical performance - immediate mitigation
if [ "$CRITICAL_PERFORMANCE" = true ]; then
    echo -e "${RED}🚨 Implementing critical performance mitigation${NC}"
    
    # Temporarily disable cache monitoring to reduce load
    echo "🔄 Temporarily disabling cache monitoring..."
    if curl -s -X POST http://localhost:8080/cache/config/update \
            -H "Content-Type: application/json" \
            -d '{"monitor_cache_effectiveness": false}' >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Cache monitoring disabled${NC}"
    else
        echo -e "${YELLOW}⚠️  Failed to disable cache monitoring via API${NC}"
    fi
    
    # Restart services to clear memory
    echo "🔄 Restarting cache services to clear memory..."
    docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
    
    echo "⏳ Waiting for services to stabilize..."
    sleep 45
    
    # Check if performance improved
    echo "📊 Checking performance after restart..."
    NEW_PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)
    NEW_CONTENT_P95=$(echo "$NEW_PERFORMANCE" | jq -r '.content_loading.p95_time_ms // 0' 2>/dev/null)
    NEW_PROMPT_P95=$(echo "$NEW_PERFORMANCE" | jq -r '.prompt_construction.p95_time_ms // 0' 2>/dev/null)
    NEW_MEMORY_USAGE=$(echo "$NEW_PERFORMANCE" | jq -r '.memory_usage_mb // 0' 2>/dev/null)
    
    echo "📈 Performance after restart:"
    echo "   • Content Loading P95: ${NEW_CONTENT_P95}ms"
    echo "   • Prompt Construction P95: ${NEW_PROMPT_P95}ms"
    echo "   • Memory Usage: ${NEW_MEMORY_USAGE}MB"
    
    # Check if critical issues resolved
    if (( $(echo "$NEW_CONTENT_P95 < $CRITICAL_CONTENT_LOADING_THRESHOLD" | bc -l 2>/dev/null || echo "1") )) && \
       (( $(echo "$NEW_PROMPT_P95 < $CRITICAL_PROMPT_CONSTRUCTION_THRESHOLD" | bc -l 2>/dev/null || echo "1") )) && \
       (( $(echo "$NEW_MEMORY_USAGE < 20" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✅ Critical performance issues resolved${NC}"
        
        # Re-enable monitoring if it was disabled
        echo "🔄 Re-enabling cache monitoring..."
        curl -s -X POST http://localhost:8080/cache/config/update \
             -H "Content-Type: application/json" \
             -d '{"monitor_cache_effectiveness": true}' >/dev/null 2>&1
        
        echo -e "${GREEN}🎉 Critical performance rollback completed${NC}"
        exit 0
    else
        echo -e "${RED}❌ Critical performance issues persist after restart${NC}"
    fi
fi

# Degraded performance - gradual mitigation
if [ "$DEGRADED_PERFORMANCE" = true ]; then
    echo -e "${YELLOW}⚠️  Implementing degraded performance mitigation${NC}"
    
    # Check for memory leaks
    echo "🔍 Checking for memory leaks..."
    INITIAL_MEMORY=$(echo "$MEMORY_USAGE")
    sleep 60
    CURRENT_PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)
    CURRENT_MEMORY=$(echo "$CURRENT_PERFORMANCE" | jq -r '.memory_usage_mb // 0' 2>/dev/null)
    
    MEMORY_INCREASE=$(echo "$CURRENT_MEMORY - $INITIAL_MEMORY" | bc -l 2>/dev/null || echo "0")
    
    if (( $(echo "$MEMORY_INCREASE > 1" | bc -l 2>/dev/null || echo "0") )); then
        echo -e "${RED}🚨 Potential memory leak detected (${MEMORY_INCREASE}MB increase in 1 minute)${NC}"
        
        # Restart to clear potential memory leak
        echo "🔄 Restarting services to address memory leak..."
        docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
        sleep 30
    else
        echo -e "${GREEN}✅ No memory leak detected${NC}"
    fi
    
    # Optimize configuration for performance
    echo "⚙️  Applying performance optimizations..."
    
    # Increase timeouts to reduce pressure
    PERF_CONFIG='{
        "prompt_construction_timeout_ms": 20,
        "content_loading_timeout_s": 5,
        "cache_effectiveness_alert_threshold": 60.0
    }'
    
    if curl -s -X POST http://localhost:8080/cache/config/update \
            -H "Content-Type: application/json" \
            -d "$PERF_CONFIG" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Performance optimizations applied${NC}"
    else
        echo -e "${YELLOW}⚠️  Failed to apply performance optimizations${NC}"
    fi
    
    # Wait and check performance again
    echo "⏳ Waiting for optimizations to take effect..."
    sleep 30
    
    FINAL_PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)
    FINAL_CONTENT_P95=$(echo "$FINAL_PERFORMANCE" | jq -r '.content_loading.p95_time_ms // 0' 2>/dev/null)
    FINAL_PROMPT_P95=$(echo "$FINAL_PERFORMANCE" | jq -r '.prompt_construction.p95_time_ms // 0' 2>/dev/null)
    FINAL_MEMORY_USAGE=$(echo "$FINAL_PERFORMANCE" | jq -r '.memory_usage_mb // 0' 2>/dev/null)
    
    echo "📈 Final performance metrics:"
    echo "   • Content Loading P95: ${FINAL_CONTENT_P95}ms"
    echo "   • Prompt Construction P95: ${FINAL_PROMPT_P95}ms"
    echo "   • Memory Usage: ${FINAL_MEMORY_USAGE}MB"
    
    # Check if performance is now acceptable
    if (( $(echo "$FINAL_CONTENT_P95 < $CONTENT_LOADING_THRESHOLD" | bc -l 2>/dev/null || echo "1") )) && \
       (( $(echo "$FINAL_PROMPT_P95 < $PROMPT_CONSTRUCTION_THRESHOLD" | bc -l 2>/dev/null || echo "1") )) && \
       (( $(echo "$FINAL_MEMORY_USAGE < $MEMORY_THRESHOLD" | bc -l 2>/dev/null || echo "1") )); then
        echo -e "${GREEN}✅ Performance issues resolved${NC}"
        echo -e "${GREEN}🎉 Performance rollback completed successfully${NC}"
        exit 0
    else
        echo -e "${YELLOW}⚠️  Performance partially improved but still degraded${NC}"
    fi
fi

# If we get here, performance issues persist
echo -e "${RED}❌ Performance rollback was not fully successful${NC}"
echo ""
echo "🚨 Performance issues persist - manual intervention required:"
echo ""
echo "🔍 Current status:"
echo "   • Content Loading P95: ${FINAL_CONTENT_P95:-$CONTENT_P95}ms (target: <${CONTENT_LOADING_THRESHOLD}ms)"
echo "   • Prompt Construction P95: ${FINAL_PROMPT_P95:-$PROMPT_P95}ms (target: <${PROMPT_CONSTRUCTION_THRESHOLD}ms)"
echo "   • Memory Usage: ${FINAL_MEMORY_USAGE:-$MEMORY_USAGE}MB (target: <${MEMORY_THRESHOLD}MB)"
echo ""
echo "🛠️  Recommended actions:"
echo "   1. Check system resources and scale if needed:"
echo "      • CPU: Consider increasing container CPU limits"
echo "      • Memory: Consider increasing container memory limits"
echo "      • Disk I/O: Check for disk performance issues"
echo ""
echo "   2. Review recent changes:"
echo "      • Check for recent configuration changes"
echo "      • Review recent content updates"
echo "      • Check for increased request volume"
echo ""
echo "   3. Consider temporary workarounds:"
echo "      • Disable cache monitoring: monitor_cache_effectiveness=false"
echo "      • Increase timeout values"
echo "      • Reduce content size if possible"
echo ""
echo "   4. Advanced diagnostics:"
echo "      • Profile memory usage: docker exec cache-metrics-exporter python -m memory_profiler"
echo "      • Check I/O patterns: iostat -x 1 10"
echo "      • Monitor CPU usage: top -p \$(pgrep -f cache_metrics_exporter)"
echo ""
echo "🔧 Emergency commands:"
echo "   • Disable monitoring: curl -X POST http://localhost:8080/cache/config/update -d '{\"monitor_cache_effectiveness\": false}'"
echo "   • Restart service: docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart"
echo "   • Check logs: docker logs cache-metrics-exporter"
echo "   • Check resources: docker stats --no-stream"
echo ""
echo "📞 Emergency contacts:"
echo "   • On-call Engineer: +1-555-0123"
echo "   • Operations Team: ops-team@company.com"
echo "   • Development Team: dev-team@company.com"
echo "   • Infrastructure Team: infrastructure@company.com"
echo ""
echo "⚠️  System is running with degraded performance"
echo "⚠️  Consider scaling resources or temporarily disabling cache monitoring"

exit 1