#!/bin/bash

# Gemini Context Caching Monitoring Setup Script
# This script sets up Grafana, Prometheus, and metrics collection for the cache system

set -e

echo "🚀 Setting up Gemini Context Caching Monitoring..."

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Run pre-deployment checks
echo "🔍 Running pre-deployment validation..."
if [ -f "monitoring/cache/deployment/pre-deployment-check.sh" ]; then
    if ! ./monitoring/cache/deployment/pre-deployment-check.sh; then
        echo "❌ Pre-deployment checks failed. Please fix issues before continuing."
        exit 1
    fi
else
    echo "⚠️  Pre-deployment check script not found, continuing anyway..."
fi

# Create necessary directories
echo "📁 Creating cache monitoring directories..."
mkdir -p monitoring/cache/prometheus/rules
mkdir -p monitoring/cache/grafana/provisioning/datasources
mkdir -p monitoring/cache/grafana/provisioning/dashboards
mkdir -p monitoring/cache/grafana/dashboards
mkdir -p monitoring/cache/exporter
mkdir -p monitoring/cache/runbooks
mkdir -p monitoring/cache/deployment

# Set permissions for Grafana
echo "🔐 Setting up permissions..."
sudo chown -R 472:472 monitoring/cache/grafana/ || echo "⚠️  Could not set Grafana permissions (may need to run as root)"

# Create cache monitoring .env file if it doesn't exist
if [ ! -f monitoring/cache/.env ]; then
    echo "📝 Creating cache monitoring .env file..."
    cat > monitoring/cache/.env << EOF
# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=admin

# Database Configuration (should match main application)
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=whiz

# Cache Monitoring Configuration
CACHE_METRICS_PORT=8080
CACHE_METRICS_INTERVAL=30
CACHE_METRICS_API_KEY=cache_metrics_secret_key

# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CACHE_ENABLED=true
GEMINI_CACHE_MONITOR=true
EOF
    echo "✅ Created cache monitoring .env file with default values"
    echo "⚠️  Please update the API keys and database credentials"
fi

# Create Docker Compose file for cache monitoring
echo "🐳 Creating cache monitoring Docker Compose configuration..."
cat > monitoring/docker-compose.cache-monitoring.yml << 'EOF'
version: '3.8'

services:
  # Prometheus for cache metrics collection
  cache-prometheus:
    image: prom/prometheus:latest
    container_name: cache-prometheus
    ports:
      - "9091:9090"
    volumes:
      - ./cache/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./cache/prometheus/rules:/etc/prometheus/rules
      - cache-prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    networks:
      - cache-monitoring

  # Grafana for cache dashboards
  cache-grafana:
    image: grafana/grafana:latest
    container_name: cache-grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-piechart-panel
    volumes:
      - ./cache/grafana/provisioning:/etc/grafana/provisioning
      - ./cache/grafana/dashboards:/var/lib/grafana/dashboards
      - cache-grafana-data:/var/lib/grafana
    networks:
      - cache-monitoring
    depends_on:
      - cache-prometheus

  # Cache metrics exporter
  cache-metrics-exporter:
    build:
      context: .
      dockerfile: monitoring/cache/exporter/Dockerfile
    container_name: cache-metrics-exporter
    ports:
      - "${CACHE_METRICS_PORT:-8080}:8080"
    environment:
      - POSTGRES_HOST=postgres
      - POSTGRES_DB=${POSTGRES_DB:-WhizDB}
      - POSTGRES_USER=${POSTGRES_USER:-admin}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-whiz}
      - METRICS_INTERVAL=${CACHE_METRICS_INTERVAL:-30}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_CACHE_ENABLED=${GEMINI_CACHE_ENABLED:-true}
      - GEMINI_CACHE_MONITOR=${GEMINI_CACHE_MONITOR:-true}
    volumes:
      - ./parlant/context:/app/context:ro
      - ./parlant/tools:/app/tools:ro
    networks:
      - cache-monitoring
      - default
    depends_on:
      - postgres
    restart: unless-stopped

volumes:
  cache-prometheus-data:
  cache-grafana-data:

networks:
  cache-monitoring:
    driver: bridge
  default:
    external: true
    name: parlant_default
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
mkdir -p monitoring/cache/prometheus
cat > monitoring/cache/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 30s
  evaluation_interval: 30s

rule_files:
  - "/etc/prometheus/rules/*.yml"

scrape_configs:
  - job_name: 'cache-metrics'
    static_configs:
      - targets: ['cache-metrics-exporter:8080']
    metrics_path: /cache/metrics
    scrape_interval: 30s
    scrape_timeout: 10s

  - job_name: 'cache-health'
    static_configs:
      - targets: ['cache-metrics-exporter:8080']
    metrics_path: /cache/health/metrics
    scrape_interval: 60s
    scrape_timeout: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: []
EOF

# Create Grafana data source configuration
echo "📈 Creating Grafana data source configuration..."
mkdir -p monitoring/cache/grafana/provisioning/datasources
cat > monitoring/cache/grafana/provisioning/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Cache Prometheus
    type: prometheus
    access: proxy
    url: http://cache-prometheus:9090
    isDefault: true
    editable: true
EOF

# Create Grafana dashboard provisioning
echo "📋 Creating Grafana dashboard provisioning..."
mkdir -p monitoring/cache/grafana/provisioning/dashboards
cat > monitoring/cache/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'cache-dashboards'
    orgId: 1
    folder: 'Cache Monitoring'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
EOF

# Create cache metrics exporter Dockerfile
echo "🐳 Creating cache metrics exporter Dockerfile..."
mkdir -p monitoring/cache/exporter
cat > monitoring/cache/exporter/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY monitoring/cache/exporter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY monitoring/cache/exporter/cache_metrics_exporter.py .
COPY parlant/tools/infrastructure/gemini_cache/ ./gemini_cache/

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/cache/health || exit 1' > /health-check.sh && \
    chmod +x /health-check.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD /health-check.sh

CMD ["python", "cache_metrics_exporter.py"]
EOF

# Create cache metrics exporter requirements
echo "📦 Creating cache metrics exporter requirements..."
cat > monitoring/cache/exporter/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
psycopg2-binary==2.9.9
prometheus-client==0.19.0
httpx==0.25.2
python-multipart==0.0.6
pydantic==2.5.0
python-dotenv==1.0.0
EOF

# Create alert rules
echo "🚨 Creating cache alert rules..."
cat > monitoring/cache/prometheus/rules/cache_alerts.yml << 'EOF'
groups:
  - name: cache_performance
    rules:
      - alert: CacheSlowContentLoading
        expr: cache_content_loading_time_ms_p95 > 3000
        for: 5m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache content loading is slow"
          description: "Content loading P95 latency is {{ $value }}ms (target: <3000ms)"

      - alert: CacheSlowPromptConstruction
        expr: cache_prompt_construction_time_ms_p95 > 10
        for: 5m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache prompt construction is slow"
          description: "Prompt construction P95 latency is {{ $value }}ms (target: <10ms)"

      - alert: CacheHighMemoryUsage
        expr: cache_memory_usage_mb > 15
        for: 10m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache memory usage is high"
          description: "Cache memory usage is {{ $value }}MB (target: <15MB)"

  - name: cache_effectiveness
    rules:
      - alert: CacheLowEffectiveness
        expr: cache_effectiveness_percent < 70
        for: 10m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache effectiveness is low"
          description: "Cache effectiveness is {{ $value }}% (target: >80%)"

      - alert: CacheLowHitRate
        expr: cache_hit_rate_percent < 80
        for: 15m
        labels:
          severity: warning
          component: cache
        annotations:
          summary: "Cache hit rate is low"
          description: "Cache hit rate is {{ $value }}% (target: >95%)"

      - alert: CacheNoSavings
        expr: increase(cache_cost_savings_usd_total[30m]) == 0
        for: 30m
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "No cache cost savings detected"
          description: "No cost savings in the last 30 minutes - cache may not be working"

  - name: cache_operational
    rules:
      - alert: CacheFallbackModeActive
        expr: cache_fallback_mode_active == 1
        for: 0m
        labels:
          severity: critical
          component: cache
        annotations:
          summary: "Cache fallback mode is active"
          description: "Cache system is in fallback mode due to content loading failures"

      - alert: CacheContentLoadingFailures
        expr: (cache_content_loading_failures_total / cache_content_loading_attempts_total) > 0.2
        for: 5m
        labels:
          severity: error
          component: cache
        annotations:
          summary: "High cache content loading failure rate"
          description: "{{ $value | humanizePercentage }} of content loads are failing"

      - alert: CacheConfigurationErrors
        expr: cache_configuration_errors_total > 0
        for: 0m
        labels:
          severity: error
          component: cache
        annotations:
          summary: "Cache configuration errors detected"
          description: "{{ $value }} configuration validation errors detected"

  - name: cache_monitoring
    rules:
      - alert: CacheMonitoringDown
        expr: up{job="cache-metrics"} == 0
        for: 2m
        labels:
          severity: critical
          component: monitoring
        annotations:
          summary: "Cache monitoring is down"
          description: "Cache metrics exporter is not responding"

      - alert: CacheHealthCheckFailing
        expr: up{job="cache-health"} == 0
        for: 5m
        labels:
          severity: warning
          component: monitoring
        annotations:
          summary: "Cache health checks are failing"
          description: "Cache health check endpoint is not responding"
EOF

# Start cache monitoring services
echo "🐳 Starting cache monitoring services..."
cd monitoring
docker-compose -f docker-compose.cache-monitoring.yml up -d
cd ..

# Wait for services to be ready
echo "⏳ Waiting for cache monitoring services to start..."
sleep 45

# Check service health
echo "🔍 Checking cache monitoring service health..."

# Check Prometheus
if curl -s http://localhost:9091/-/healthy > /dev/null; then
    echo "✅ Cache Prometheus is healthy"
else
    echo "❌ Cache Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3001/api/health > /dev/null; then
    echo "✅ Cache Grafana is healthy"
else
    echo "❌ Cache Grafana is not responding"
fi

# Check cache metrics exporter
if curl -s http://localhost:8080/cache/health > /dev/null; then
    echo "✅ Cache metrics exporter is healthy"
else
    echo "❌ Cache metrics exporter is not responding"
fi

echo ""
echo "🎉 Gemini Context Caching monitoring setup complete!"
echo ""
echo "📊 Access your cache dashboards:"
echo "   • Cache Grafana: http://localhost:3001 (admin/admin)"
echo "   • Cache Prometheus: http://localhost:9091"
echo "   • Cache Metrics: http://localhost:8080/cache/metrics"
echo "   • Cache Health: http://localhost:8080/cache/health"
echo ""

# Run post-deployment validation
echo "🔍 Running post-deployment validation..."
if [ -f "monitoring/cache/deployment/post-deployment-check.sh" ]; then
    if ./monitoring/cache/deployment/post-deployment-check.sh; then
        echo ""
        echo "🎉 All systems operational!"
    else
        echo ""
        echo "⚠️  Some post-deployment checks failed. Review the output above."
        echo "    The system may still be functional, but some features may not work optimally."
    fi
else
    echo "⚠️  Post-deployment check script not found"
fi

echo ""
echo "🔧 Next steps:"
echo "   1. Change the default Grafana password"
echo "   2. Import the Gemini Cache dashboard (should be auto-provisioned)"
echo "   3. Configure cache alerting (optional)"
echo "   4. Set up notification channels (optional)"
echo "   5. Update API keys in monitoring/cache/.env"
echo "   6. Schedule daily health checks: ./monitoring/cache-health-check.sh"
echo ""
echo "📚 Documentation:"
echo "   • Operations runbook: monitoring/cache/runbooks/CACHE_OPERATIONS_RUNBOOK.md"
echo "   • Full documentation: monitoring/gemini-cache-monitoring.md"
echo "   • Dashboard location: monitoring/cache/grafana/dashboards/"
echo "   • Alerts configuration: monitoring/cache/prometheus/rules/cache_alerts.yml"