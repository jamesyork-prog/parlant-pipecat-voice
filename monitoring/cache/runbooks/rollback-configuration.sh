#!/bin/bash

# Gemini Context Caching Configuration Rollback Script
# This script handles rollback procedures for configuration issues

set -e

echo "🔄 Rolling back configuration changes..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CONFIG_FILE="/app/cache_config.json"
BACKUP_FILE="/app/cache_config.json.backup"
PREVIOUS_FILE="/app/cache_config.json.previous"

# Check current configuration status
echo -n "Checking current configuration status... "
CONFIG_STATUS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)

if [ "$CONFIG_STATUS" = "valid" ]; then
    echo -e "${GREEN}✅ Configuration is currently valid${NC}"
    echo "ℹ️  No rollback needed"
    exit 0
elif [ "$CONFIG_STATUS" = "invalid" ]; then
    echo -e "${RED}❌ Configuration is invalid${NC}"
    
    # Get validation errors
    ERRORS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.validation_errors[]' 2>/dev/null)
    if [ -n "$ERRORS" ]; then
        echo "🚨 Validation errors:"
        echo "$ERRORS" | while read -r error; do
            echo "   • $error"
        done
    fi
else
    echo -e "${YELLOW}⚠️  Configuration status unknown${NC}"
fi

# Backup current configuration
echo "📦 Backing up current configuration..."
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}✅ Current configuration backed up${NC}"
else
    echo -e "${YELLOW}⚠️  No current configuration file found${NC}"
fi

# Check for previous configuration
if [ -f "$PREVIOUS_FILE" ]; then
    echo "📋 Previous configuration found, attempting restore..."
    
    # Restore previous configuration
    cp "$PREVIOUS_FILE" "$CONFIG_FILE"
    echo -e "${GREEN}✅ Configuration restored from backup${NC}"
    
    # Trigger hot reload
    echo "🔄 Triggering configuration reload..."
    if curl -s -X POST http://localhost:8080/cache/config/reload >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Configuration reload triggered${NC}"
    else
        echo -e "${RED}❌ Failed to trigger configuration reload${NC}"
        echo "🔧 Attempting service restart..."
        
        # Restart the service
        if docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter >/dev/null 2>&1; then
            echo -e "${GREEN}✅ Service restarted${NC}"
            echo "⏳ Waiting for service to stabilize..."
            sleep 30
        else
            echo -e "${RED}❌ Failed to restart service${NC}"
            exit 1
        fi
    fi
    
    # Validate restored configuration
    echo "🔍 Validating restored configuration..."
    sleep 5
    
    NEW_STATUS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
    
    if [ "$NEW_STATUS" = "valid" ]; then
        echo -e "${GREEN}✅ Configuration rollback successful${NC}"
        
        # Show configuration summary
        echo "📊 Configuration summary:"
        CONFIG_SUMMARY=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null)
        
        ENABLED=$(echo "$CONFIG_SUMMARY" | jq -r '.enabled // false')
        MONITOR=$(echo "$CONFIG_SUMMARY" | jq -r '.monitor_cache_effectiveness // false')
        INPUT_COST=$(echo "$CONFIG_SUMMARY" | jq -r '.pricing.input_cost_per_1m // 0')
        CACHED_COST=$(echo "$CONFIG_SUMMARY" | jq -r '.pricing.cached_input_cost_per_1m // 0')
        DISCOUNT=$(echo "$CONFIG_SUMMARY" | jq -r '.pricing.discount_percentage // 0')
        
        echo "   • Cache enabled: $ENABLED"
        echo "   • Monitoring enabled: $MONITOR"
        echo "   • Input cost per 1M tokens: \$$INPUT_COST"
        echo "   • Cached cost per 1M tokens: \$$CACHED_COST"
        echo "   • Discount percentage: ${DISCOUNT}%"
        
        echo -e "${GREEN}🎉 Configuration rollback completed successfully${NC}"
        exit 0
    else
        echo -e "${RED}❌ Configuration rollback failed - restored config is also invalid${NC}"
        
        # Get new validation errors
        NEW_ERRORS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.validation_errors[]' 2>/dev/null)
        if [ -n "$NEW_ERRORS" ]; then
            echo "🚨 Validation errors in restored configuration:"
            echo "$NEW_ERRORS" | while read -r error; do
                echo "   • $error"
            done
        fi
    fi
else
    echo -e "${YELLOW}⚠️  No previous configuration backup found${NC}"
fi

# Attempt to create a minimal valid configuration
echo "🛠️  Creating minimal valid configuration..."

cat > "$CONFIG_FILE" << 'EOF'
{
  "context_caching": {
    "enabled": true,
    "monitor_cache_effectiveness": true,
    "static_content_paths": {
      "refund_policy": "parlant/context/processed/refund_policy_condensed.md",
      "decision_chart": "parlant/context/processed/refund_scenario_decision_chart.md",
      "ai_vs_human": "parlant/context/processed/ai_vs_human_refund_scenarios.md"
    },
    "tool_definitions_path": "parlant/tools/",
    "few_shot_examples_path": "parlant/context/few_shot_examples.json",
    "max_static_content_tokens": 32000,
    "prompt_construction_timeout_ms": 10,
    "content_loading_timeout_s": 3,
    "cache_effectiveness_alert_threshold": 70.0,
    "hot_reload_enabled": true,
    "pricing": {
      "input_cost_per_1m": 0.30,
      "cached_input_cost_per_1m": 0.03,
      "output_cost_per_1m": 2.50
    }
  }
}
EOF

echo -e "${GREEN}✅ Minimal configuration created${NC}"

# Trigger reload with minimal configuration
echo "🔄 Reloading with minimal configuration..."
if curl -s -X POST http://localhost:8080/cache/config/reload >/dev/null 2>&1; then
    sleep 5
    
    MINIMAL_STATUS=$(curl -s http://localhost:8080/cache/config/validate 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
    
    if [ "$MINIMAL_STATUS" = "valid" ]; then
        echo -e "${GREEN}✅ Minimal configuration is valid${NC}"
        echo -e "${YELLOW}⚠️  System is running with default configuration${NC}"
        echo ""
        echo "📋 Next steps:"
        echo "   1. Review the minimal configuration in $CONFIG_FILE"
        echo "   2. Update configuration parameters as needed"
        echo "   3. Test configuration changes in staging first"
        echo "   4. Apply proper configuration once validated"
        echo ""
        echo -e "${GREEN}🎉 Emergency configuration rollback completed${NC}"
        exit 0
    else
        echo -e "${RED}❌ Even minimal configuration is invalid${NC}"
    fi
else
    echo -e "${RED}❌ Failed to reload minimal configuration${NC}"
fi

# If we get here, all rollback attempts failed
echo -e "${RED}❌ Configuration rollback failed${NC}"
echo ""
echo "🚨 Critical configuration issue - manual intervention required:"
echo ""
echo "🔍 Diagnostic steps:"
echo "   1. Check configuration file syntax:"
echo "      cat $CONFIG_FILE | jq ."
echo ""
echo "   2. Check service logs:"
echo "      docker logs cache-metrics-exporter"
echo ""
echo "   3. Validate configuration manually:"
echo "      curl http://localhost:8080/cache/config/validate | jq ."
echo ""
echo "   4. Check environment variables:"
echo "      env | grep GEMINI_CACHE"
echo ""
echo "🛠️  Recovery options:"
echo "   1. Restore from backup:"
echo "      cp $BACKUP_FILE $CONFIG_FILE"
echo "      curl -X POST http://localhost:8080/cache/config/reload"
echo ""
echo "   2. Use environment variables only:"
echo "      rm $CONFIG_FILE"
echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter"
echo ""
echo "   3. Restart with default configuration:"
echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml down"
echo "      docker-compose -f monitoring/docker-compose.cache-monitoring.yml up -d"
echo ""
echo "📞 Emergency contacts:"
echo "   • On-call Engineer: +1-555-0123"
echo "   • Operations Team: ops-team@company.com"
echo "   • Development Team: dev-team@company.com"
echo ""
echo "⚠️  System may be running with degraded functionality until configuration is fixed"

exit 1