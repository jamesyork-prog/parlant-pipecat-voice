#!/bin/bash

# Gemini Context Caching Health Check Script
# This script performs comprehensive health checks for the cache monitoring system

set -e

echo "🔍 Running Gemini Context Caching Health Checks..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Health check results
OVERALL_STATUS="healthy"
ISSUES=()

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "  Checking $service_name... "
    
    if response=$(curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null); then
        if [ "$response" -eq "$expected_status" ]; then
            echo -e "${GREEN}✅ Healthy${NC}"
            return 0
        else
            echo -e "${RED}❌ Unhealthy (HTTP $response)${NC}"
            ISSUES+=("$service_name returned HTTP $response")
            OVERALL_STATUS="unhealthy"
            return 1
        fi
    else
        echo -e "${RED}❌ Unreachable${NC}"
        ISSUES+=("$service_name is unreachable")
        OVERALL_STATUS="unhealthy"
        return 1
    fi
}

# Function to check metric value
check_metric() {
    local metric_name=$1
    local url=$2
    local threshold=$3
    local comparison=$4  # "gt" for greater than, "lt" for less than
    
    echo -n "  Checking $metric_name... "
    
    if value=$(curl -s "$url" 2>/dev/null | jq -r ".$metric_name" 2>/dev/null); then
        if [ "$value" != "null" ] && [ "$value" != "" ]; then
            case $comparison in
                "gt")
                    if (( $(echo "$value > $threshold" | bc -l 2>/dev/null || echo "0") )); then
                        echo -e "${GREEN}✅ $value (target: >$threshold)${NC}"
                        return 0
                    else
                        echo -e "${YELLOW}⚠️  $value (target: >$threshold)${NC}"
                        ISSUES+=("$metric_name is $value (target: >$threshold)")
                        return 1
                    fi
                    ;;
                "lt")
                    if (( $(echo "$value < $threshold" | bc -l 2>/dev/null || echo "0") )); then
                        echo -e "${GREEN}✅ $value (target: <$threshold)${NC}"
                        return 0
                    else
                        echo -e "${YELLOW}⚠️  $value (target: <$threshold)${NC}"
                        ISSUES+=("$metric_name is $value (target: <$threshold)")
                        return 1
                    fi
                    ;;
            esac
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("$metric_name has no data")
            OVERALL_STATUS="unhealthy"
            return 1
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch $metric_name")
        OVERALL_STATUS="unhealthy"
        return 1
    fi
}

echo ""
echo "🏥 Service Health Checks"
echo "========================"

# Check core services
check_service "Cache Metrics Exporter" "http://localhost:8080/cache/health"
check_service "Cache Prometheus" "http://localhost:9091/-/healthy"
check_service "Cache Grafana" "http://localhost:3001/api/health"

echo ""
echo "📊 Cache Effectiveness Checks"
echo "============================="

# Check cache effectiveness metrics
if curl -s http://localhost:8080/cache/health/effectiveness >/dev/null 2>&1; then
    check_metric "cache_hit_rate" "http://localhost:8080/cache/health/effectiveness" "80" "gt"
    check_metric "avg_cache_effectiveness" "http://localhost:8080/cache/health/effectiveness" "70" "gt"
    
    # Check for active alerts
    echo -n "  Checking for alerts... "
    if alerts=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.alerts | length' 2>/dev/null); then
        if [ "$alerts" -eq 0 ]; then
            echo -e "${GREEN}✅ No alerts${NC}"
        else
            echo -e "${YELLOW}⚠️  $alerts active alerts${NC}"
            ISSUES+=("$alerts active cache alerts")
        fi
    else
        echo -e "${RED}❌ Failed to check alerts${NC}"
        ISSUES+=("Failed to check cache alerts")
        OVERALL_STATUS="unhealthy"
    fi
else
    echo -e "${RED}❌ Cache effectiveness endpoint unreachable${NC}"
    ISSUES+=("Cache effectiveness endpoint unreachable")
    OVERALL_STATUS="unhealthy"
fi

echo ""
echo "⚡ Performance Checks"
echo "===================="

# Check performance metrics
if curl -s http://localhost:8080/cache/health/performance >/dev/null 2>&1; then
    # Content loading performance
    echo -n "  Checking content loading P95... "
    if p95_content=$(curl -s http://localhost:8080/cache/health/performance | jq -r '.content_loading.p95_time_ms' 2>/dev/null); then
        if [ "$p95_content" != "null" ] && [ "$p95_content" != "" ]; then
            if (( $(echo "$p95_content < 3000" | bc -l 2>/dev/null || echo "0") )); then
                echo -e "${GREEN}✅ ${p95_content}ms (target: <3000ms)${NC}"
            else
                echo -e "${YELLOW}⚠️  ${p95_content}ms (target: <3000ms)${NC}"
                ISSUES+=("Content loading P95 is ${p95_content}ms (target: <3000ms)")
            fi
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("Content loading P95 has no data")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch content loading P95")
        OVERALL_STATUS="unhealthy"
    fi
    
    # Prompt construction performance
    echo -n "  Checking prompt construction P95... "
    if p95_prompt=$(curl -s http://localhost:8080/cache/health/performance | jq -r '.prompt_construction.p95_time_ms' 2>/dev/null); then
        if [ "$p95_prompt" != "null" ] && [ "$p95_prompt" != "" ]; then
            if (( $(echo "$p95_prompt < 10" | bc -l 2>/dev/null || echo "0") )); then
                echo -e "${GREEN}✅ ${p95_prompt}ms (target: <10ms)${NC}"
            else
                echo -e "${YELLOW}⚠️  ${p95_prompt}ms (target: <10ms)${NC}"
                ISSUES+=("Prompt construction P95 is ${p95_prompt}ms (target: <10ms)")
            fi
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("Prompt construction P95 has no data")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch prompt construction P95")
        OVERALL_STATUS="unhealthy"
    fi
    
    # Memory usage
    echo -n "  Checking memory usage... "
    if memory=$(curl -s http://localhost:8080/cache/health/performance | jq -r '.memory_usage_mb' 2>/dev/null); then
        if [ "$memory" != "null" ] && [ "$memory" != "" ]; then
            if (( $(echo "$memory < 15" | bc -l 2>/dev/null || echo "0") )); then
                echo -e "${GREEN}✅ ${memory}MB (target: <15MB)${NC}"
            else
                echo -e "${YELLOW}⚠️  ${memory}MB (target: <15MB)${NC}"
                ISSUES+=("Memory usage is ${memory}MB (target: <15MB)")
            fi
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("Memory usage has no data")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch memory usage")
        OVERALL_STATUS="unhealthy"
    fi
else
    echo -e "${RED}❌ Performance endpoint unreachable${NC}"
    ISSUES+=("Performance endpoint unreachable")
    OVERALL_STATUS="unhealthy"
fi

echo ""
echo "📁 Content Loading Checks"
echo "========================="

# Check content loading status
if curl -s http://localhost:8080/cache/health/content >/dev/null 2>&1; then
    echo -n "  Checking static content status... "
    if status=$(curl -s http://localhost:8080/cache/health/content | jq -r '.status' 2>/dev/null); then
        if [ "$status" = "healthy" ]; then
            echo -e "${GREEN}✅ Healthy${NC}"
        else
            echo -e "${RED}❌ $status${NC}"
            ISSUES+=("Static content status is $status")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to check${NC}"
        ISSUES+=("Failed to check static content status")
        OVERALL_STATUS="unhealthy"
    fi
    
    # Check fallback mode
    echo -n "  Checking fallback mode... "
    if fallback=$(curl -s http://localhost:8080/cache/health/content | jq -r '.fallback_mode_active // false' 2>/dev/null); then
        if [ "$fallback" = "false" ]; then
            echo -e "${GREEN}✅ Normal mode${NC}"
        else
            echo -e "${YELLOW}⚠️  Fallback mode active${NC}"
            ISSUES+=("System is in fallback mode")
        fi
    else
        echo -e "${RED}❌ Failed to check${NC}"
        ISSUES+=("Failed to check fallback mode")
        OVERALL_STATUS="unhealthy"
    fi
else
    echo -e "${RED}❌ Content endpoint unreachable${NC}"
    ISSUES+=("Content endpoint unreachable")
    OVERALL_STATUS="unhealthy"
fi

echo ""
echo "⚙️  Configuration Checks"
echo "========================"

# Check configuration status
if curl -s http://localhost:8080/cache/config/validate >/dev/null 2>&1; then
    echo -n "  Checking configuration validity... "
    if config_status=$(curl -s http://localhost:8080/cache/config/validate | jq -r '.status' 2>/dev/null); then
        if [ "$config_status" = "valid" ]; then
            echo -e "${GREEN}✅ Valid${NC}"
        else
            echo -e "${RED}❌ $config_status${NC}"
            ISSUES+=("Configuration status is $config_status")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to check${NC}"
        ISSUES+=("Failed to check configuration")
        OVERALL_STATUS="unhealthy"
    fi
    
    # Check for validation errors
    echo -n "  Checking validation errors... "
    if errors=$(curl -s http://localhost:8080/cache/config/validate | jq -r '.validation_errors | length' 2>/dev/null); then
        if [ "$errors" -eq 0 ]; then
            echo -e "${GREEN}✅ No errors${NC}"
        else
            echo -e "${YELLOW}⚠️  $errors validation errors${NC}"
            ISSUES+=("$errors configuration validation errors")
        fi
    else
        echo -e "${RED}❌ Failed to check${NC}"
        ISSUES+=("Failed to check validation errors")
        OVERALL_STATUS="unhealthy"
    fi
else
    echo -e "${RED}❌ Configuration endpoint unreachable${NC}"
    ISSUES+=("Configuration endpoint unreachable")
    OVERALL_STATUS="unhealthy"
fi

echo ""
echo "💰 Cost Savings Checks"
echo "======================"

# Check cost savings
if curl -s http://localhost:8080/cache/health/effectiveness >/dev/null 2>&1; then
    echo -n "  Checking daily savings... "
    if daily_savings=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.daily_savings_usd' 2>/dev/null); then
        if [ "$daily_savings" != "null" ] && [ "$daily_savings" != "" ]; then
            if (( $(echo "$daily_savings > 0" | bc -l 2>/dev/null || echo "0") )); then
                echo -e "${GREEN}✅ \$${daily_savings}${NC}"
            else
                echo -e "${YELLOW}⚠️  \$${daily_savings} (no savings detected)${NC}"
                ISSUES+=("No daily cost savings detected")
            fi
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("Daily savings has no data")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch daily savings")
        OVERALL_STATUS="unhealthy"
    fi
    
    echo -n "  Checking total savings... "
    if total_savings=$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.total_savings_usd' 2>/dev/null); then
        if [ "$total_savings" != "null" ] && [ "$total_savings" != "" ]; then
            echo -e "${GREEN}✅ \$${total_savings}${NC}"
        else
            echo -e "${RED}❌ No data${NC}"
            ISSUES+=("Total savings has no data")
            OVERALL_STATUS="unhealthy"
        fi
    else
        echo -e "${RED}❌ Failed to fetch${NC}"
        ISSUES+=("Failed to fetch total savings")
        OVERALL_STATUS="unhealthy"
    fi
fi

echo ""
echo "📈 Metrics Collection Checks"
echo "============================"

# Check Prometheus metrics
echo -n "  Checking Prometheus metrics collection... "
if curl -s http://localhost:8080/cache/metrics | grep -q "cache_" 2>/dev/null; then
    echo -e "${GREEN}✅ Metrics available${NC}"
else
    echo -e "${RED}❌ No cache metrics found${NC}"
    ISSUES+=("No cache metrics found in Prometheus endpoint")
    OVERALL_STATUS="unhealthy"
fi

# Check Prometheus targets
echo -n "  Checking Prometheus targets... "
if targets=$(curl -s http://localhost:9091/api/v1/targets 2>/dev/null | jq -r '.data.activeTargets[] | select(.labels.job=="cache-metrics") | .health' 2>/dev/null); then
    if [ "$targets" = "up" ]; then
        echo -e "${GREEN}✅ Target is up${NC}"
    else
        echo -e "${RED}❌ Target is $targets${NC}"
        ISSUES+=("Prometheus cache-metrics target is $targets")
        OVERALL_STATUS="unhealthy"
    fi
else
    echo -e "${YELLOW}⚠️  Failed to check targets${NC}"
    ISSUES+=("Failed to check Prometheus targets")
fi

echo ""
echo "📊 Dashboard Checks"
echo "=================="

# Check Grafana dashboard
echo -n "  Checking Grafana dashboard access... "
if curl -s http://localhost:3001/api/dashboards/uid/gemini-cache >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Dashboard accessible${NC}"
else
    echo -e "${YELLOW}⚠️  Dashboard not found (may need to be imported)${NC}"
fi

# Summary
echo ""
echo "📋 Health Check Summary"
echo "======================"

if [ "$OVERALL_STATUS" = "healthy" ]; then
    echo -e "${GREEN}✅ Overall Status: HEALTHY${NC}"
    echo ""
    echo "🎉 All cache monitoring systems are operating normally!"
    echo ""
    echo "📊 Quick Stats:"
    if curl -s http://localhost:8080/cache/health/effectiveness >/dev/null 2>&1; then
        echo "   • Cache Hit Rate: $(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.cache_hit_rate // "N/A"')%"
        echo "   • Cache Effectiveness: $(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.avg_cache_effectiveness // "N/A"')%"
        echo "   • Daily Savings: \$$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.daily_savings_usd // "N/A"')"
        echo "   • Total Savings: \$$(curl -s http://localhost:8080/cache/health/effectiveness | jq -r '.total_savings_usd // "N/A"')"
    fi
    exit 0
else
    echo -e "${RED}❌ Overall Status: UNHEALTHY${NC}"
    echo ""
    echo "🚨 Issues detected:"
    for issue in "${ISSUES[@]}"; do
        echo "   • $issue"
    done
    echo ""
    echo "🔧 Recommended actions:"
    echo "   1. Check service logs: docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs"
    echo "   2. Restart services: docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart"
    echo "   3. Check configuration: curl http://localhost:8080/cache/config/validate"
    echo "   4. Review monitoring documentation: monitoring/gemini-cache-monitoring.md"
    exit 1
fi