#!/bin/bash

# Gemini Context Caching Post-Deployment Validation Script
# This script validates the cache system after deployment

set -e

echo "🚀 Running Gemini Cache post-deployment validation..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS="passed"
FAILED_CHECKS=()
WARNINGS=()

# Function to run a check and track results
run_check() {
    local check_name="$1"
    local check_command="$2"
    local required="${3:-true}"
    local timeout="${4:-30}"
    
    echo -n "  $check_name... "
    
    # Run command with timeout
    if timeout "$timeout" bash -c "$check_command" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ PASS${NC}"
        return 0
    else
        if [ "$required" = "true" ]; then
            echo -e "${RED}❌ FAIL${NC}"
            FAILED_CHECKS+=("$check_name")
            OVERALL_STATUS="failed"
            return 1
        else
            echo -e "${YELLOW}⚠️  WARN${NC}"
            WARNINGS+=("$check_name")
            return 0
        fi
    fi
}

# Function to check HTTP endpoint
check_http_endpoint() {
    local url="$1"
    local expected_status="${2:-200}"
    
    response=$(curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null)
    [ "$response" -eq "$expected_status" ]
}

# Function to check service is running
check_service_running() {
    local service_name="$1"
    docker-compose -f monitoring/docker-compose.cache-monitoring.yml ps "$service_name" | grep -q "Up"
}

# Function to wait for service to be ready
wait_for_service() {
    local service_name="$1"
    local url="$2"
    local max_attempts=30
    local attempt=1
    
    echo -n "    Waiting for $service_name to be ready"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo -e " ${GREEN}✅${NC}"
            return 0
        fi
        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e " ${RED}❌ Timeout${NC}"
    return 1
}

echo ""
echo "🐳 Service Status Checks"
echo "========================"

# Check if services are running
run_check "Cache Prometheus container running" "check_service_running cache-prometheus"
run_check "Cache Grafana container running" "check_service_running cache-grafana"
run_check "Cache metrics exporter container running" "check_service_running cache-metrics-exporter"

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to be ready..."

wait_for_service "Cache Prometheus" "http://localhost:9091/-/healthy"
wait_for_service "Cache Grafana" "http://localhost:3001/api/health"
wait_for_service "Cache Metrics Exporter" "http://localhost:8080/cache/health"

echo ""
echo "🌐 Endpoint Health Checks"
echo "========================="

# Check service endpoints
run_check "Cache Prometheus health endpoint" "check_http_endpoint http://localhost:9091/-/healthy"
run_check "Cache Grafana health endpoint" "check_http_endpoint http://localhost:3001/api/health"
run_check "Cache metrics exporter health endpoint" "check_http_endpoint http://localhost:8080/cache/health"

# Check specific cache endpoints
run_check "Cache content health endpoint" "check_http_endpoint http://localhost:8080/cache/health/content"
run_check "Cache effectiveness health endpoint" "check_http_endpoint http://localhost:8080/cache/health/effectiveness"
run_check "Cache performance health endpoint" "check_http_endpoint http://localhost:8080/cache/health/performance"
run_check "Cache configuration validation endpoint" "check_http_endpoint http://localhost:8080/cache/config/validate"

# Check metrics endpoints
run_check "Prometheus metrics endpoint" "check_http_endpoint http://localhost:8080/cache/metrics"
run_check "JSON metrics endpoint" "check_http_endpoint http://localhost:8080/cache/metrics/json"

echo ""
echo "📊 Configuration Validation"
echo "==========================="

# Validate configuration
CONFIG_STATUS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
run_check "Configuration is valid" "[ \"$CONFIG_STATUS\" = \"valid\" ]"

if [ "$CONFIG_STATUS" = "valid" ]; then
    # Get configuration details
    CONFIG=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null)
    
    ENABLED=$(echo "$CONFIG" | jq -r '.enabled // false')
    MONITOR=$(echo "$CONFIG" | jq -r '.monitor_cache_effectiveness // false')
    INPUT_COST=$(echo "$CONFIG" | jq -r '.pricing.input_cost_per_1m // 0')
    CACHED_COST=$(echo "$CONFIG" | jq -r '.pricing.cached_input_cost_per_1m // 0')
    DISCOUNT=$(echo "$CONFIG" | jq -r '.pricing.discount_percentage // 0')
    
    echo "    📋 Configuration Summary:"
    echo "       • Cache enabled: $ENABLED"
    echo "       • Monitoring enabled: $MONITOR"
    echo "       • Input cost per 1M tokens: \$$INPUT_COST"
    echo "       • Cached cost per 1M tokens: \$$CACHED_COST"
    echo "       • Discount percentage: ${DISCOUNT}%"
    
    # Validate pricing makes sense
    run_check "Pricing configuration is reasonable" "[ \"$ENABLED\" = \"true\" ] && [ \"$MONITOR\" = \"true\" ] && (( \$(echo \"$DISCOUNT > 50\" | bc -l) ))"
fi

echo ""
echo "📈 Metrics Collection"
echo "===================="

# Check if metrics are being collected
run_check "Prometheus is collecting metrics" "curl -s http://localhost:9091/api/v1/query?query=up | jq -r '.data.result | length' | grep -q '[1-9]'"
run_check "Cache metrics are available" "curl -s http://localhost:8080/cache/metrics | grep -q 'cache_'"

# Check specific metrics
METRICS_OUTPUT=$(curl -s http://localhost:8080/cache/metrics 2>/dev/null)
run_check "Cache effectiveness metric exists" "echo '$METRICS_OUTPUT' | grep -q 'cache_effectiveness_percent'" false
run_check "Cache hit rate metric exists" "echo '$METRICS_OUTPUT' | grep -q 'cache_hit_rate_percent'" false
run_check "Cost savings metric exists" "echo '$METRICS_OUTPUT' | grep -q 'cache_cost_savings_usd_total'" false

echo ""
echo "🎛️  Dashboard Accessibility"
echo "=========================="

# Check Grafana dashboard
run_check "Grafana login page accessible" "check_http_endpoint http://localhost:3001/login"
run_check "Grafana API accessible" "check_http_endpoint http://localhost:3001/api/datasources"

# Try to access the specific dashboard (may require authentication)
run_check "Cache dashboard exists in Grafana" "curl -s http://localhost:3001/api/dashboards/uid/gemini-cache | jq -r '.dashboard.title' | grep -q 'Gemini'" false

echo ""
echo "🔍 Content Loading Status"
echo "========================="

# Check content loading status
CONTENT_STATUS=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
run_check "Content loading is healthy" "[ \"$CONTENT_STATUS\" = \"healthy\" ]" false

if [ "$CONTENT_STATUS" = "healthy" ]; then
    CONTENT_HEALTH=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null)
    
    STATIC_LOADED=$(echo "$CONTENT_HEALTH" | jq -r '.static_content_loaded // false')
    TOTAL_TOKENS=$(echo "$CONTENT_HEALTH" | jq -r '.total_tokens // 0')
    FALLBACK_MODE=$(echo "$CONTENT_HEALTH" | jq -r '.fallback_mode_active // false')
    
    echo "    📁 Content Status:"
    echo "       • Static content loaded: $STATIC_LOADED"
    echo "       • Total tokens: $TOTAL_TOKENS"
    echo "       • Fallback mode active: $FALLBACK_MODE"
    
    run_check "Static content is loaded" "[ \"$STATIC_LOADED\" = \"true\" ]" false
    run_check "Not in fallback mode" "[ \"$FALLBACK_MODE\" = \"false\" ]" false
    run_check "Reasonable token count" "(( \$(echo \"$TOTAL_TOKENS > 1000\" | bc -l) )) && (( \$(echo \"$TOTAL_TOKENS < 50000\" | bc -l) ))" false
fi

echo ""
echo "⚡ Performance Validation"
echo "========================"

# Check performance metrics
PERFORMANCE=$(curl -s http://localhost:8080/cache/health/performance 2>/dev/null)
PERF_STATUS=$(echo "$PERFORMANCE" | jq -r '.status // "unknown"' 2>/dev/null)

run_check "Performance health is good" "[ \"$PERF_STATUS\" = \"healthy\" ]" false

if [ -n "$PERFORMANCE" ] && [ "$PERFORMANCE" != "null" ]; then
    CONTENT_P95=$(echo "$PERFORMANCE" | jq -r '.content_loading.p95_time_ms // 0' 2>/dev/null)
    PROMPT_P95=$(echo "$PERFORMANCE" | jq -r '.prompt_construction.p95_time_ms // 0' 2>/dev/null)
    MEMORY_USAGE=$(echo "$PERFORMANCE" | jq -r '.memory_usage_mb // 0' 2>/dev/null)
    
    echo "    ⚡ Performance Metrics:"
    echo "       • Content Loading P95: ${CONTENT_P95}ms (target: <3000ms)"
    echo "       • Prompt Construction P95: ${PROMPT_P95}ms (target: <10ms)"
    echo "       • Memory Usage: ${MEMORY_USAGE}MB (target: <15MB)"
    
    # Validate performance targets (warnings only for post-deployment)
    run_check "Content loading performance acceptable" "(( \$(echo \"$CONTENT_P95 < 5000\" | bc -l) ))" false
    run_check "Prompt construction performance acceptable" "(( \$(echo \"$PROMPT_P95 < 50\" | bc -l) ))" false
    run_check "Memory usage acceptable" "(( \$(echo \"$MEMORY_USAGE < 25\" | bc -l) ))" false
fi

echo ""
echo "💰 Cache Effectiveness"
echo "======================"

# Check cache effectiveness (may not have data immediately after deployment)
EFFECTIVENESS=$(curl -s http://localhost:8080/cache/health/effectiveness 2>/dev/null)
EFFECTIVENESS_STATUS=$(echo "$EFFECTIVENESS" | jq -r '.status // "unknown"' 2>/dev/null)

if [ "$EFFECTIVENESS_STATUS" != "no_data" ] && [ "$EFFECTIVENESS_STATUS" != "unknown" ]; then
    CACHE_HIT_RATE=$(echo "$EFFECTIVENESS" | jq -r '.cache_hit_rate // 0' 2>/dev/null)
    AVG_EFFECTIVENESS=$(echo "$EFFECTIVENESS" | jq -r '.avg_cache_effectiveness // 0' 2>/dev/null)
    TOTAL_SAVINGS=$(echo "$EFFECTIVENESS" | jq -r '.total_savings_usd // 0' 2>/dev/null)
    
    echo "    💰 Cache Effectiveness:"
    echo "       • Cache Hit Rate: ${CACHE_HIT_RATE}%"
    echo "       • Average Effectiveness: ${AVG_EFFECTIVENESS}%"
    echo "       • Total Savings: \$${TOTAL_SAVINGS}"
    
    # These are warnings since we may not have enough data yet
    run_check "Cache hit rate is reasonable" "(( \$(echo \"$CACHE_HIT_RATE > 50\" | bc -l) ))" false
    run_check "Cache effectiveness is reasonable" "(( \$(echo \"$AVG_EFFECTIVENESS > 30\" | bc -l) ))" false
else
    echo "    ℹ️  No cache effectiveness data yet (expected for new deployment)"
fi

echo ""
echo "🚨 Alert System"
echo "==============="

# Check if alerts are configured
run_check "Prometheus alert rules loaded" "curl -s http://localhost:9091/api/v1/rules | jq -r '.data.groups | length' | grep -q '[1-9]'" false

# Check for any active alerts
ACTIVE_ALERTS=$(curl -s http://localhost:9091/api/v1/alerts 2>/dev/null | jq -r '.data.alerts | length' 2>/dev/null)
if [ -n "$ACTIVE_ALERTS" ] && [ "$ACTIVE_ALERTS" != "null" ]; then
    run_check "No critical alerts active" "[ \"$ACTIVE_ALERTS\" -eq 0 ]" false
    
    if [ "$ACTIVE_ALERTS" -gt 0 ]; then
        echo "    ⚠️  $ACTIVE_ALERTS active alerts detected"
    fi
fi

echo ""
echo "🔧 Operational Readiness"
echo "========================"

# Check if operational scripts are in place
run_check "Health check script exists" "[ -f monitoring/cache-health-check.sh ]"
run_check "Rollback scripts exist" "[ -f monitoring/cache/runbooks/rollback-content-loading.sh ] && [ -f monitoring/cache/runbooks/rollback-configuration.sh ] && [ -f monitoring/cache/runbooks/rollback-performance.sh ]"
run_check "Operations runbook exists" "[ -f monitoring/cache/runbooks/CACHE_OPERATIONS_RUNBOOK.md ]"

# Test basic operational commands
run_check "Configuration reload works" "curl -s -X POST http://localhost:8080/cache/config/reload | jq -r '.status' | grep -q 'reloaded\\|unchanged'"
run_check "Content reload works" "curl -s -X POST http://localhost:8080/cache/content/reload | jq -r '.status' | grep -q 'reloaded\\|error'" false

echo ""
echo "📊 Post-Deployment Summary"
echo "=========================="

if [ "$OVERALL_STATUS" = "passed" ]; then
    echo -e "${GREEN}✅ Post-deployment validation successful${NC}"
    echo ""
    echo "🎉 Cache monitoring system is operational!"
    echo ""
    echo "📊 Access Points:"
    echo "   • Grafana Dashboard: http://localhost:3001 (admin/admin)"
    echo "   • Prometheus: http://localhost:9091"
    echo "   • Cache Health: http://localhost:8080/cache/health"
    echo "   • Cache Metrics: http://localhost:8080/cache/metrics"
    echo ""
    echo "🔧 Next Steps:"
    echo "   1. Change default Grafana password"
    echo "   2. Import the Gemini Cache dashboard if not auto-provisioned"
    echo "   3. Set up alert notifications (email, Slack, etc.)"
    echo "   4. Schedule daily health checks"
    echo "   5. Review operational runbook: monitoring/cache/runbooks/CACHE_OPERATIONS_RUNBOOK.md"
    echo ""
    
    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warnings (non-critical):${NC}"
        for warning in "${WARNINGS[@]}"; do
            echo "   • $warning"
        done
        echo ""
        echo "These warnings are expected for a new deployment and should resolve as the system processes requests."
        echo ""
    fi
    
    echo "🏃‍♂️ Operational Commands:"
    echo "   • Daily health check: ./monitoring/cache-health-check.sh"
    echo "   • View logs: docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs"
    echo "   • Restart services: docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Post-deployment validation failed${NC}"
    echo ""
    echo "🚨 Failed checks:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "   • $check"
    done
    echo ""
    
    if [ ${#WARNINGS[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warnings:${NC}"
        for warning in "${WARNINGS[@]}"; do
            echo "   • $warning"
        done
        echo ""
    fi
    
    echo "🔧 Troubleshooting steps:"
    echo "   1. Check service logs:"
    echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs"
    echo ""
    echo "   2. Check service status:"
    echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml ps"
    echo ""
    echo "   3. Restart services:"
    echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart"
    echo ""
    echo "   4. Run health check:"
    echo "      ./monitoring/cache-health-check.sh"
    echo ""
    echo "   5. Check individual endpoints:"
    echo "      curl http://localhost:8080/cache/health"
    echo "      curl http://localhost:9091/-/healthy"
    echo "      curl http://localhost:3001/api/health"
    echo ""
    echo "📚 Documentation:"
    echo "   • Operations runbook: monitoring/cache/runbooks/CACHE_OPERATIONS_RUNBOOK.md"
    echo "   • Setup guide: monitoring/gemini-cache-monitoring.md"
    echo ""
    echo "📞 Support:"
    echo "   • Check logs and documentation first"
    echo "   • Contact operations team if issues persist"
    echo ""
    exit 1
fi