# MOR/Product Type Router Monitoring

This directory contains the complete monitoring setup for the MOR (Merchant of Record) and Product Type routing gates, including Grafana dashboards, Prometheus metrics collection, and alerting rules.

## Overview

The monitoring system provides comprehensive visibility into:

- **Performance Metrics**: Processing latency, throughput, and performance degradation alerts
- **Business Metrics**: Cost savings, rejection rates, and ROI tracking
- **Operational Metrics**: Vendor rejection patterns, product type distribution, and data quality
- **System Health**: Service availability, error rates, and resource utilization

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Router App    │───▶│ Metrics         │───▶│   Prometheus    │
│   (Parlant)     │    │ Exporter        │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   PostgreSQL    │───▶│ Audit Logs      │             │
│   Database      │    │ & Metrics       │             │
└─────────────────┘    └─────────────────┘             │
                                                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Grafana       │◀───│   Dashboard     │
                       │   Dashboard     │    │   & Alerts      │
                       └─────────────────┘    └─────────────────┘
```

## Quick Start

### 1. Setup Monitoring

```bash
# Run the setup script
./monitoring/setup-monitoring.sh

# Or manually start services
docker-compose -f monitoring/docker-compose.monitoring.yml up -d
```

### 2. Access Dashboards

- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Raw Metrics**: http://localhost:8080/metrics

### 3. Import Dashboard

The MOR/Product Type Router dashboard is automatically provisioned, but you can also import it manually:

1. Open Grafana (http://localhost:3000)
2. Go to Dashboards → Import
3. Upload `monitoring/grafana/dashboards/mor-product-type-router.json`

## Dashboard Features

### Overview Section
- **Total Tickets Processed**: Real-time count of all tickets through the router
- **Overall Rejection Rate**: Percentage of tickets rejected by both gates
- **Total Cost Savings**: Cumulative USD savings from avoided LLM calls
- **Combined Gates P95 Latency**: 95th percentile processing time

### MOR Gate Metrics
- **Vendor Rejection Distribution** (Pie Chart): Breakdown by third-party vendors
- **Vendor Rejection Counts** (Bar Chart): Historical rejection counts
- **Processing Time Trends**: MOR gate latency over time
- **Unknown Partner Alerts**: Data quality monitoring

### Product Type Gate Metrics
- **Product Type Distribution** (Pie Chart): Event, transient, on-demand, season package
- **Rejection Breakdown**: On-demand vs season package rejections
- **Duration Calculation Failures**: Data quality monitoring
- **Processing Time Trends**: Product type gate latency over time

### Performance Monitoring
- **Processing Latency Histogram**: 95th percentile latency for all gates
- **Cost Savings Over Time**: Trend analysis of financial impact
- **Ticket Processing Rates**: Throughput monitoring

## Metrics Reference

### Core Metrics

| Metric | Description | Target |
|--------|-------------|---------|
| `router_mor_gate_processing_time_ms_p95` | MOR gate 95th percentile latency | <20ms |
| `router_product_type_gate_processing_time_ms_p95` | Product Type gate 95th percentile latency | <50ms |
| `router_combined_gates_processing_time_ms_p95` | Combined gates 95th percentile latency | <100ms |
| `router_total_rejections_total` | Total tickets rejected | 20-30% of processed |
| `router_total_cost_savings_usd_total` | Cumulative cost savings | $20-45/day target |

### Business Metrics

| Metric | Description | Business Value |
|--------|-------------|----------------|
| `router_mor_gate_vendor_rejections_total` | Rejections by vendor | Identifies top MOR sources |
| `router_product_type_gate_type_distribution_total` | Product type breakdown | Validates inference accuracy |
| `router_total_cost_savings_usd_total` | Cost savings | ROI measurement |

### Data Quality Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `router_mor_gate_unknown_partner_total` | Unknown Partner values | >5% of tickets |
| `router_product_type_gate_duration_failures_total` | Duration calc failures | >10% of tickets |

## Alerting Rules

### Performance Alerts

- **MORGateHighLatency**: Triggers when MOR gate P95 > 20ms for 5 minutes
- **ProductTypeGateHighLatency**: Triggers when Product Type gate P95 > 50ms for 5 minutes
- **CombinedGatesHighLatency**: Triggers when combined P95 > 100ms for 5 minutes

### Data Quality Alerts

- **HighUnknownPartnerRate**: Triggers when >5% of tickets have unknown Partner values
- **HighDurationCalculationFailureRate**: Triggers when >10% of tickets fail duration calculation

### Business Alerts

- **LowRejectionRate**: Triggers when rejection rate <15% for 30 minutes (indicates potential issues)

## Configuration

### Environment Variables

```bash
# Database connection (must match main application)
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=whiz
POSTGRES_HOST=postgres

# Monitoring configuration
GRAFANA_ADMIN_PASSWORD=your_secure_password
METRICS_PORT=8080
METRICS_INTERVAL=30
```

### Customizing Dashboards

1. **Edit Dashboard**: Modify `monitoring/grafana/dashboards/mor-product-type-router.json`
2. **Add Panels**: Use Grafana UI to add new visualizations
3. **Export Changes**: Export updated dashboard JSON from Grafana UI

### Customizing Alerts

1. **Edit Rules**: Modify `monitoring/prometheus/rules/router_alerts.yml`
2. **Reload Config**: `docker-compose exec prometheus kill -HUP 1`
3. **Test Alerts**: Use Prometheus UI to validate alert expressions

## Troubleshooting

### Common Issues

#### Metrics Not Appearing

```bash
# Check metrics exporter logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs router-metrics-exporter

# Check if metrics endpoint is accessible
curl http://localhost:8080/metrics

# Verify database connection
docker-compose -f monitoring/docker-compose.monitoring.yml exec router-metrics-exporter python -c "
import psycopg2
import os
conn = psycopg2.connect(
    host=os.getenv('POSTGRES_HOST', 'postgres'),
    database=os.getenv('POSTGRES_DB', 'WhizDB'),
    user=os.getenv('POSTGRES_USER', 'admin'),
    password=os.getenv('POSTGRES_PASSWORD')
)
print('Database connection successful')
"
```

#### Dashboard Not Loading

```bash
# Check Grafana logs
docker-compose -f monitoring/docker-compose.monitoring.yml logs grafana

# Verify Prometheus data source
curl http://localhost:3000/api/datasources

# Check Prometheus connectivity
curl http://localhost:9090/api/v1/query?query=up
```

#### Alerts Not Firing

```bash
# Check alert rules
curl http://localhost:9090/api/v1/rules

# Verify alert expressions
curl "http://localhost:9090/api/v1/query?query=router_mor_gate_processing_time_ms_p95"

# Check alert manager (if configured)
curl http://localhost:9093/api/v1/alerts
```

### Performance Tuning

#### High Memory Usage

```yaml
# In docker-compose.monitoring.yml, add memory limits
services:
  prometheus:
    mem_limit: 512m
  grafana:
    mem_limit: 256m
```

#### Slow Queries

```yaml
# Reduce metrics retention in prometheus.yml
global:
  scrape_interval: 30s  # Increase from 15s
  
# Reduce data retention
command:
  - '--storage.tsdb.retention.time=7d'  # Reduce from 30d
```

## Production Deployment

### Security Considerations

1. **Change Default Passwords**:
   ```bash
   # Update .env file
   GRAFANA_ADMIN_PASSWORD=your_secure_password
   ```

2. **Enable HTTPS**:
   ```yaml
   # Add reverse proxy with SSL termination
   services:
     nginx:
       image: nginx:alpine
       ports:
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/ssl
   ```

3. **Restrict Access**:
   ```yaml
   # Remove public port exposure
   services:
     grafana:
       # ports:
       #   - "3000:3000"  # Remove this line
       expose:
         - "3000"
   ```

### Backup and Recovery

```bash
# Backup Grafana dashboards
docker-compose exec grafana grafana-cli admin export-dashboard

# Backup Prometheus data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine tar czf /backup/prometheus-backup.tar.gz /data

# Restore Prometheus data
docker run --rm -v prometheus_data:/data -v $(pwd):/backup alpine tar xzf /backup/prometheus-backup.tar.gz -C /
```

### Scaling

For high-volume deployments:

1. **Use External Prometheus**: Deploy Prometheus on dedicated infrastructure
2. **Enable Remote Storage**: Configure Prometheus remote write to long-term storage
3. **Load Balance Grafana**: Deploy multiple Grafana instances behind a load balancer

## Integration with Existing Infrastructure

### Existing Prometheus

If you already have Prometheus running:

1. **Add Scrape Config**:
   ```yaml
   scrape_configs:
     - job_name: 'router-metrics'
       static_configs:
         - targets: ['your-app-host:8080']
   ```

2. **Import Alert Rules**:
   ```bash
   cp monitoring/prometheus/rules/router_alerts.yml /path/to/your/prometheus/rules/
   ```

### Existing Grafana

1. **Import Dashboard**: Use the JSON file in `monitoring/grafana/dashboards/`
2. **Configure Data Source**: Point to your existing Prometheus instance

## Support and Maintenance

### Regular Maintenance Tasks

1. **Weekly**: Review alert firing patterns and adjust thresholds
2. **Monthly**: Analyze cost savings trends and validate ROI projections
3. **Quarterly**: Review dashboard effectiveness and add new metrics as needed

### Monitoring the Monitoring

Set up alerts for the monitoring system itself:

```yaml
# Add to router_alerts.yml
- alert: RouterMetricsDown
  expr: up{job="router-metrics"} == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Router metrics exporter is down"
```

## Contributing

To add new metrics or improve dashboards:

1. **Add Metrics**: Update `monitoring/exporter/metrics_exporter.py`
2. **Update Dashboard**: Modify the JSON file or use Grafana UI
3. **Add Alerts**: Update `monitoring/prometheus/rules/router_alerts.yml`
4. **Test Changes**: Verify in development environment
5. **Document**: Update this README with new features

## References

- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [MOR/Product Type Router Spec](.kiro/specs/mor-product-type-router/)