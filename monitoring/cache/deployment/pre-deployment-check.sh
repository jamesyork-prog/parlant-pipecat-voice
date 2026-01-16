#!/bin/bash

# Gemini Context Caching Pre-Deployment Validation Script
# This script validates the cache system before deployment

set -e

echo "🔍 Running Gemini Cache pre-deployment checks..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track overall status
OVERALL_STATUS="passed"
FAILED_CHECKS=()

# Function to run a check and track results
run_check() {
    local check_name="$1"
    local check_command="$2"
    local required="${3:-true}"
    
    echo -n "  $check_name... "
    
    if eval "$check_command" >/dev/null 2>&1; then
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
            return 0
        fi
    fi
}

# Function to validate JSON
validate_json() {
    local file="$1"
    if [ -f "$file" ]; then
        jq . "$file" >/dev/null 2>&1
    else
        return 1
    fi
}

# Function to check service health
check_service_health() {
    local url="$1"
    local expected_status="${2:-200}"
    
    response=$(curl -s -w "%{http_code}" -o /dev/null "$url" 2>/dev/null)
    [ "$response" -eq "$expected_status" ]
}

echo ""
echo "📋 Configuration Validation"
echo "============================"

# Check configuration file exists and is valid JSON
run_check "Configuration file exists" "[ -f monitoring/cache/.env ]"
run_check "Docker Compose file exists" "[ -f monitoring/docker-compose.cache-monitoring.yml ]"
run_check "Prometheus config exists" "[ -f monitoring/cache/prometheus/prometheus.yml ]"
run_check "Grafana dashboard exists" "[ -f monitoring/cache/grafana/dashboards/gemini-cache-dashboard.json ]"

# Validate JSON files
run_check "Grafana dashboard JSON valid" "validate_json monitoring/cache/grafana/dashboards/gemini-cache-dashboard.json"
run_check "Alert rules YAML valid" "python -c 'import yaml; yaml.safe_load(open(\"monitoring/cache/prometheus/rules/cache_alerts.yml\"))'" false

echo ""
echo "🐳 Docker Environment"
echo "===================="

# Check Docker and Docker Compose
run_check "Docker is available" "command -v docker"
run_check "Docker Compose is available" "command -v docker-compose"
run_check "Docker daemon is running" "docker info"

# Check if required images are available or can be pulled
run_check "Prometheus image available" "docker image inspect prom/prometheus:latest || docker pull prom/prometheus:latest" false
run_check "Grafana image available" "docker image inspect grafana/grafana:latest || docker pull grafana/grafana:latest" false

echo ""
echo "📁 File System Checks"
echo "===================="

# Check required directories exist
run_check "Cache monitoring directory exists" "[ -d monitoring/cache ]"
run_check "Runbooks directory exists" "[ -d monitoring/cache/runbooks ]"
run_check "Deployment scripts directory exists" "[ -d monitoring/cache/deployment ]"

# Check file permissions
run_check "Runbook scripts are executable" "[ -x monitoring/cache/runbooks/rollback-content-loading.sh ]"
run_check "Health check script is executable" "[ -x monitoring/cache-health-check.sh ]" false
run_check "Setup script is executable" "[ -x monitoring/setup-cache-monitoring.sh ]" false

# Check content directories (these should exist for the cache system to work)
run_check "Content directory exists" "[ -d parlant/context/processed ]" false
run_check "Tools directory exists" "[ -d parlant/tools ]" false

echo ""
echo "🔧 Environment Variables"
echo "========================"

# Check required environment variables
if [ -f monitoring/cache/.env ]; then
    source monitoring/cache/.env
    
    run_check "Grafana admin password set" "[ -n \"$GRAFANA_ADMIN_PASSWORD\" ]"
    run_check "Database configuration set" "[ -n \"$POSTGRES_DB\" ] && [ -n \"$POSTGRES_USER\" ] && [ -n \"$POSTGRES_PASSWORD\" ]"
    run_check "Cache metrics port set" "[ -n \"$CACHE_METRICS_PORT\" ]" false
else
    echo -e "${YELLOW}⚠️  Environment file not found - using defaults${NC}"
fi

# Check if Gemini API key is available (optional but recommended)
run_check "Gemini API key available" "[ -n \"$GEMINI_API_KEY\" ]" false

echo ""
echo "🌐 Network Connectivity"
echo "======================="

# Check if required ports are available
run_check "Port 8080 available" "! netstat -tuln | grep -q ':8080 '" false
run_check "Port 9091 available" "! netstat -tuln | grep -q ':9091 '" false
run_check "Port 3001 available" "! netstat -tuln | grep -q ':3001 '" false

# Check external connectivity (if needed)
run_check "Internet connectivity" "ping -c 1 8.8.8.8" false
run_check "Docker Hub connectivity" "curl -s https://registry-1.docker.io/v2/ | grep -q 'Docker Registry'" false

echo ""
echo "💾 Storage Requirements"
echo "======================"

# Check disk space
AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
run_check "Sufficient disk space (>1GB)" "[ $AVAILABLE_SPACE -gt 1 ]"

# Check if we can create required directories
run_check "Can create monitoring directories" "mkdir -p /tmp/cache-test && rmdir /tmp/cache-test"

echo ""
echo "🔍 Security Checks"
echo "=================="

# Check file permissions are secure
run_check "Environment file has secure permissions" "[ \$(stat -c '%a' monitoring/cache/.env 2>/dev/null || echo '644') -le 644 ]" false
run_check "Configuration files not world-writable" "! find monitoring/cache -name '*.yml' -o -name '*.json' | xargs ls -l | grep -q 'rw-rw-rw-'" false

echo ""
echo "⚙️  Service Dependencies"
echo "======================="

# Check if main application database is accessible (if running)
run_check "Main database accessible" "docker exec postgres psql -U admin -d WhizDB -c 'SELECT 1' 2>/dev/null" false

# Check if main application is running
run_check "Main application running" "curl -s http://localhost:8800/health" false

echo ""
echo "📊 Pre-Deployment Summary"
echo "========================="

if [ "$OVERALL_STATUS" = "passed" ]; then
    echo -e "${GREEN}✅ All critical pre-deployment checks passed${NC}"
    echo ""
    echo "🚀 Ready for deployment!"
    echo ""
    echo "Next steps:"
    echo "  1. Run deployment: ./monitoring/setup-cache-monitoring.sh"
    echo "  2. Wait for services to start (2-3 minutes)"
    echo "  3. Run post-deployment validation: ./monitoring/cache/deployment/post-deployment-check.sh"
    echo "  4. Access Grafana dashboard: http://localhost:3001"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Pre-deployment checks failed${NC}"
    echo ""
    echo "🚨 Failed checks:"
    for check in "${FAILED_CHECKS[@]}"; do
        echo "   • $check"
    done
    echo ""
    echo "🔧 Required actions before deployment:"
    echo ""
    
    # Provide specific guidance based on failed checks
    for check in "${FAILED_CHECKS[@]}"; do
        case "$check" in
            "Configuration file exists")
                echo "   • Create monitoring/cache/.env file with required configuration"
                echo "     cp monitoring/cache/.env.example monitoring/cache/.env"
                ;;
            "Docker is available"|"Docker Compose is available")
                echo "   • Install Docker and Docker Compose"
                echo "     https://docs.docker.com/get-docker/"
                ;;
            "Docker daemon is running")
                echo "   • Start Docker daemon"
                echo "     sudo systemctl start docker"
                ;;
            "Sufficient disk space"*)
                echo "   • Free up disk space (need at least 1GB)"
                echo "     docker system prune -f"
                ;;
            *)
                echo "   • Fix: $check"
                ;;
        esac
    done
    
    echo ""
    echo "📚 Documentation:"
    echo "   • Setup guide: monitoring/gemini-cache-monitoring.md"
    echo "   • Operations runbook: monitoring/cache/runbooks/CACHE_OPERATIONS_RUNBOOK.md"
    echo ""
    exit 1
fi