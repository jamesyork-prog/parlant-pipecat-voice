#!/bin/bash

# Health check script for MOR/Product Type Router monitoring
# This script validates that all monitoring components are working correctly

set -e

echo "🔍 MOR/Product Type Router Monitoring Health Check"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service health
check_service() {
    local service_name=$1
    local url=$2
    local expected_status=${3:-200}
    
    echo -n "Checking $service_name... "
    
    if response=$(curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null); then
        if [ "$response" -eq "$expected_status" ]; then
            echo -e "${GREEN}✅ OK${NC}"
            return 0
        else
            echo -e "${RED}❌ FAIL (HTTP $response)${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ UNREACHABLE${NC}"
        return 1
    fi
}

# Function to check metrics availability
check_metrics() {
    local metric_name=$1
    local prometheus_url="http://localhost:9090"
    
    echo -n "Checking metric '$metric_name'... "
    
    query_url="$prometheus_url/api/v1/query?query=$metric_name"
    
    if response=$(curl -s "$query_url" 2>/dev/null); then
        if echo "$response" | grep -q '"status":"success"'; then
            if echo "$response" | grep -q '"result":\[\]'; then
                echo -e "${YELLOW}⚠️  NO DATA${NC}"
                return 1
            else
                echo -e "${GREEN}✅ OK${NC}"
                return 0
            fi
        else
            echo -e "${RED}❌ QUERY FAILED${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ UNREACHABLE${NC}"
        return 1
    fi
}

# Function to check Docker containers
check_containers() {
    echo "📦 Checking Docker containers..."
    
    containers=("whiz-prometheus" "whiz-grafana" "whiz-router-metrics")
    all_running=true
    
    for container in "${containers[@]}"; do
        echo -n "Checking container '$container'... "
        
        if docker ps --format "table {{.Names}}" | grep -q "^$container$"; then
            status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "no-healthcheck")
            if [ "$status" = "healthy" ] || [ "$status" = "no-healthcheck" ]; then
                echo -e "${GREEN}✅ RUNNING${NC}"
            else
                echo -e "${YELLOW}⚠️  UNHEALTHY ($status)${NC}"
                all_running=false
            fi
        else
            echo -e "${RED}❌ NOT RUNNING${NC}"
            all_running=false
        fi
    done
    
    return $([ "$all_running" = true ] && echo 0 || echo 1)
}

# Main health check
main() {
    local exit_code=0
    
    echo "🐳 Docker Container Status"
    echo "-------------------------"
    check_containers || exit_code=1
    
    echo ""
    echo "🌐 Service Connectivity"
    echo "----------------------"
    check_service "Prometheus" "http://localhost:9090/-/healthy" || exit_code=1
    check_service "Grafana" "http://localhost:3000/api/health" || exit_code=1
    check_service "Router Metrics Exporter" "http://localhost:8080/metrics" || exit_code=1
    
    echo ""
    echo "📊 Core Metrics Availability"
    echo "----------------------------"
    check_metrics "router_total_processed_total" || exit_code=1
    check_metrics "router_total_rejections_total" || exit_code=1
    check_metrics "router_total_cost_savings_usd_total" || exit_code=1
    check_metrics "router_mor_gate_processing_time_ms_p95" || exit_code=1
    check_metrics "router_product_type_gate_processing_time_ms_p95" || exit_code=1
    
    echo ""
    echo "🚨 Alert Rules Status"
    echo "--------------------"
    
    # Check if alert rules are loaded
    echo -n "Checking alert rules... "
    if response=$(curl -s "http://localhost:9090/api/v1/rules" 2>/dev/null); then
        if echo "$response" | grep -q "MORGateHighLatency"; then
            echo -e "${GREEN}✅ LOADED${NC}"
        else
            echo -e "${YELLOW}⚠️  NOT FOUND${NC}"
            exit_code=1
        fi
    else
        echo -e "${RED}❌ UNREACHABLE${NC}"
        exit_code=1
    fi
    
    echo ""
    echo "📈 Dashboard Status"
    echo "------------------"
    
    # Check if dashboard is available
    echo -n "Checking MOR/Product Type dashboard... "
    if response=$(curl -s "http://localhost:3000/api/search?query=MOR" 2>/dev/null); then
        if echo "$response" | grep -q "MOR/Product Type Router"; then
            echo -e "${GREEN}✅ AVAILABLE${NC}"
        else
            echo -e "${YELLOW}⚠️  NOT FOUND${NC}"
            exit_code=1
        fi
    else
        echo -e "${RED}❌ UNREACHABLE${NC}"
        exit_code=1
    fi
    
    echo ""
    echo "📋 Summary"
    echo "=========="
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ All monitoring components are healthy!${NC}"
        echo ""
        echo "🔗 Access URLs:"
        echo "   • Grafana Dashboard: http://localhost:3000"
        echo "   • Prometheus: http://localhost:9090"
        echo "   • Router Metrics: http://localhost:8080/metrics"
    else
        echo -e "${RED}❌ Some monitoring components have issues.${NC}"
        echo ""
        echo "🔧 Troubleshooting:"
        echo "   • Check container logs: docker-compose -f monitoring/docker-compose.monitoring.yml logs"
        echo "   • Restart services: docker-compose -f monitoring/docker-compose.monitoring.yml restart"
        echo "   • Review configuration: monitoring/README.md"
    fi
    
    return $exit_code
}

# Run health check
main "$@"