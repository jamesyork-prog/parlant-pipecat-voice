#!/bin/bash

# Gemini Context Caching Content Loading Rollback Script
# This script handles rollback procedures for content loading issues

set -e

echo "🔄 Rolling back content loading issues..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if fallback mode is active
echo -n "Checking fallback mode status... "
FALLBACK=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.fallback_mode_active // false' 2>/dev/null)

if [ "$FALLBACK" = "true" ]; then
    echo -e "${YELLOW}⚠️  System is in fallback mode${NC}"
    
    # Force content reload
    echo "🔄 Forcing content reload..."
    if curl -s -X POST http://localhost:8080/cache/content/reload >/dev/null 2>&1; then
        echo -e "${GREEN}✅ Content reload request sent${NC}"
    else
        echo -e "${RED}❌ Failed to send content reload request${NC}"
        exit 1
    fi
    
    # Wait and check
    echo "⏳ Waiting 10 seconds for content reload..."
    sleep 10
    
    STATUS=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
    
    if [ "$STATUS" = "healthy" ]; then
        echo -e "${GREEN}✅ Content loading restored${NC}"
        
        # Verify all content types are loaded
        echo "🔍 Verifying content types..."
        CONTENT_TYPES=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.content_types // {}' 2>/dev/null)
        
        for content_type in system_identity refund_policies tool_definitions few_shot_examples; do
            status=$(echo "$CONTENT_TYPES" | jq -r ".$content_type // \"unknown\"" 2>/dev/null)
            if [ "$status" = "loaded" ]; then
                echo -e "  ${GREEN}✅ $content_type: loaded${NC}"
            else
                echo -e "  ${YELLOW}⚠️  $content_type: $status${NC}"
            fi
        done
        
        echo -e "${GREEN}🎉 Content loading rollback completed successfully${NC}"
        exit 0
    else
        echo -e "${RED}❌ Content loading still failing - status: $STATUS${NC}"
        echo "🔧 Attempting manual recovery..."
    fi
else
    echo -e "${GREEN}✅ Content loading is healthy${NC}"
    echo "ℹ️  No rollback needed"
    exit 0
fi

# Manual recovery steps
echo "🛠️  Starting manual content recovery..."

# Check if content files exist
echo "📁 Checking content file availability..."
CONTENT_DIR="/app/parlant/context/processed"
BACKUP_DIR="/backup/parlant/context/processed"

declare -A CONTENT_FILES=(
    ["refund_policy"]="refund_policy_condensed.md"
    ["decision_chart"]="refund_scenario_decision_chart.md"
    ["ai_vs_human"]="ai_vs_human_refund_scenarios.md"
)

MISSING_FILES=()
BACKUP_AVAILABLE=false

for name in "${!CONTENT_FILES[@]}"; do
    file="${CONTENT_FILES[$name]}"
    if [ -f "$CONTENT_DIR/$file" ]; then
        echo -e "  ${GREEN}✅ $file: exists${NC}"
    else
        echo -e "  ${RED}❌ $file: missing${NC}"
        MISSING_FILES+=("$file")
    fi
    
    # Check if backup exists
    if [ -f "$BACKUP_DIR/$file" ]; then
        BACKUP_AVAILABLE=true
    fi
done

# Restore missing files from backup if available
if [ ${#MISSING_FILES[@]} -gt 0 ] && [ "$BACKUP_AVAILABLE" = true ]; then
    echo "📦 Restoring missing files from backup..."
    
    for file in "${MISSING_FILES[@]}"; do
        if [ -f "$BACKUP_DIR/$file" ]; then
            echo "  📄 Restoring $file..."
            cp "$BACKUP_DIR/$file" "$CONTENT_DIR/$file"
            chmod 644 "$CONTENT_DIR/$file"
            echo -e "    ${GREEN}✅ $file restored${NC}"
        else
            echo -e "    ${RED}❌ Backup not found for $file${NC}"
        fi
    done
    
    # Try content reload again
    echo "🔄 Retrying content reload after restoration..."
    if curl -s -X POST http://localhost:8080/cache/content/reload >/dev/null 2>&1; then
        sleep 10
        STATUS=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
        
        if [ "$STATUS" = "healthy" ]; then
            echo -e "${GREEN}✅ Content loading restored after file restoration${NC}"
            exit 0
        fi
    fi
fi

# Check file permissions
echo "🔐 Checking file permissions..."
for name in "${!CONTENT_FILES[@]}"; do
    file="${CONTENT_FILES[$name]}"
    if [ -f "$CONTENT_DIR/$file" ]; then
        if [ -r "$CONTENT_DIR/$file" ]; then
            echo -e "  ${GREEN}✅ $file: readable${NC}"
        else
            echo -e "  ${RED}❌ $file: not readable${NC}"
            echo "    🔧 Fixing permissions..."
            chmod 644 "$CONTENT_DIR/$file"
            echo -e "    ${GREEN}✅ Permissions fixed${NC}"
        fi
    fi
done

# Check disk space
echo "💾 Checking disk space..."
DISK_USAGE=$(df -h "$CONTENT_DIR" | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 90 ]; then
    echo -e "  ${RED}❌ Disk usage is high: ${DISK_USAGE}%${NC}"
    echo "  ⚠️  Low disk space may be causing content loading issues"
else
    echo -e "  ${GREEN}✅ Disk usage is acceptable: ${DISK_USAGE}%${NC}"
fi

# Final attempt at content reload
echo "🔄 Final attempt at content reload..."
if curl -s -X POST http://localhost:8080/cache/content/reload >/dev/null 2>&1; then
    sleep 15
    STATUS=$(curl -s http://localhost:8080/cache/health/content 2>/dev/null | jq -r '.status // "unknown"' 2>/dev/null)
    
    if [ "$STATUS" = "healthy" ]; then
        echo -e "${GREEN}✅ Content loading finally restored${NC}"
        echo -e "${GREEN}🎉 Manual recovery completed successfully${NC}"
        exit 0
    fi
fi

# If we get here, manual recovery failed
echo -e "${RED}❌ Content loading rollback failed${NC}"
echo ""
echo "🚨 Manual intervention required:"
echo "   1. Check application logs for detailed error messages"
echo "   2. Verify content file integrity and format"
echo "   3. Check file system health and permissions"
echo "   4. Consider restarting the cache metrics exporter service"
echo "   5. Contact the development team if issues persist"
echo ""
echo "📞 Emergency contacts:"
echo "   • On-call Engineer: +1-555-0123"
echo "   • Operations Team: ops-team@company.com"
echo "   • Development Team: dev-team@company.com"
echo ""
echo "🔍 Diagnostic commands:"
echo "   • Check logs: docker logs cache-metrics-exporter"
echo "   • Check files: ls -la $CONTENT_DIR"
echo "   • Check health: curl http://localhost:8080/cache/health/content"
echo "   • Restart service: docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter"

exit 1