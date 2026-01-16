#!/bin/bash

# MOR/Product Type Router Monitoring Setup Script
# This script sets up Grafana, Prometheus, and metrics collection for the router

set -e

echo "🚀 Setting up MOR/Product Type Router Monitoring..."

# Check if Docker and Docker Compose are available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating monitoring directories..."
mkdir -p monitoring/prometheus/rules
mkdir -p monitoring/grafana/provisioning/datasources
mkdir -p monitoring/grafana/provisioning/dashboards
mkdir -p monitoring/grafana/dashboards
mkdir -p monitoring/exporter

# Set permissions for Grafana
echo "🔐 Setting up permissions..."
sudo chown -R 472:472 monitoring/grafana/ || echo "⚠️  Could not set Grafana permissions (may need to run as root)"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=admin

# Database Configuration (should match main application)
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=whiz
EOF
    echo "✅ Created .env file with default values"
    echo "⚠️  Please update the database credentials to match your main application"
fi

# Start monitoring services
echo "🐳 Starting monitoring services..."
docker-compose -f monitoring/docker-compose.monitoring.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check Prometheus
if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus is healthy"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -s http://localhost:3000/api/health > /dev/null; then
    echo "✅ Grafana is healthy"
else
    echo "❌ Grafana is not responding"
fi

# Check metrics exporter
if curl -s http://localhost:8080/metrics > /dev/null; then
    echo "✅ Router metrics exporter is healthy"
else
    echo "❌ Router metrics exporter is not responding"
fi

echo ""
echo "🎉 Monitoring setup complete!"
echo ""
echo "📊 Access your dashboards:"
echo "   • Grafana: http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo "   • Router Metrics: http://localhost:8080/metrics"
echo ""
echo "🔧 Next steps:"
echo "   1. Change the default Grafana password"
echo "   2. Import the MOR/Product Type Router dashboard"
echo "   3. Configure alerting (optional)"
echo "   4. Set up notification channels (optional)"
echo ""
echo "📚 Dashboard location: monitoring/grafana/dashboards/mor-product-type-router.json"
echo "🚨 Alerts configuration: monitoring/prometheus/rules/router_alerts.yml"