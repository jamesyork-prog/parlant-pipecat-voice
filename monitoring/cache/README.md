# Gemini Context Caching Production Monitoring

This directory contains the complete production monitoring setup for the Gemini Context Caching system, including dashboards, alerting, health checks, and operational procedures.

## Quick Start

### 1. Setup Monitoring

```bash
# Run the setup script (includes pre/post deployment validation)
./monitoring/setup-cache-monitoring.sh
```

### 2. Access Dashboards

- **Grafana Dashboard**: http://localhost:3001 (admin/admin)
- **Prometheus**: http://localhost:9091
- **Cache Health**: http://localhost:8080/cache/health
- **Cache Metrics**: http://localhost:8080/cache/metrics

### 3. Daily Operations

```bash
# Run daily health check
./monitoring/cache-health-check.sh

# View service logs
docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs

# Restart services if needed
docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart
```

## Directory Structure

```
monitoring/cache/
├── README.md                           # This file
├── .env                               # Environment configuration
├── docker-compose.cache-monitoring.yml # Docker services
├── deployment/                        # Deployment validation
│   ├── pre-deployment-check.sh       # Pre-deployment validation
│   └── post-deployment-check.sh      # Post-deployment validation
├── exporter/                          # Metrics exporter
│   ├── cache_metrics_exporter.py     # Main metrics service
│   ├── requirements.txt              # Python dependencies
│   └── Dockerfile                    # Container definition
├── grafana/                           # Grafana configuration
│   ├── dashboards/                   # Dashboard definitions
│   │   └── gemini-cache-dashboard.json
│   └── provisioning/                 # Auto-provisioning config
│       ├── datasources/              # Data source config
│       └── dashboards/               # Dashboard provisioning
├── prometheus/                        # Prometheus configuration
│   ├── prometheus.yml                # Main config
│   └── rules/                        # Alert rules
│       └── cache_alerts.yml          # Cache-specific alerts
└── runbooks/                         # Operational procedures
    ├── CACHE_OPERATIONS_RUNBOOK.md   # Complete operations guide
    ├── rollback-content-loading.sh   # Content loading rollback
    ├── rollback-configuration.sh     # Configuration rollback
    └── rollback-performance.sh       # Performance rollback
```

## Key Features

### 🎛️ Monitoring Dashboard

The Grafana dashboard provides comprehensive visibility into:

- **Cache Effectiveness**: Hit rates, effectiveness percentages, cost savings
- **Performance Metrics**: Content loading times, prompt construction latency
- **Operational Health**: Fallback mode status, content loading failures
- **Cost Analytics**: Real-time savings, ROI tracking, billing validation

### 🚨 Alerting System

Automated alerts for:

- **Performance Issues**: Slow content loading (>3s), slow prompt construction (>10ms)
- **Effectiveness Problems**: Low cache effectiveness (<70%), low hit rates (<80%)
- **Operational Issues**: Fallback mode activation, content loading failures
- **System Health**: Service downtime, configuration errors

### 🔍 Health Checks

Comprehensive health monitoring:

- **Service Health**: All monitoring services operational
- **Content Status**: Static content loading and availability
- **Performance**: Latency and resource usage within targets
- **Configuration**: Valid configuration and hot reload capability

### 📋 Operational Procedures

Complete runbooks for:

- **Daily Operations**: Health checks, performance monitoring, cost validation
- **Emergency Procedures**: Critical issues, fallback mode, performance degradation
- **Rollback Procedures**: Content, configuration, and performance rollbacks
- **Troubleshooting**: Common issues and resolution steps

## Configuration

### Environment Variables

Key configuration in `monitoring/cache/.env`:

```bash
# Grafana Configuration
GRAFANA_ADMIN_PASSWORD=admin

# Database Configuration
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=whiz

# Cache Monitoring
CACHE_METRICS_PORT=8080
GEMINI_API_KEY=your_api_key_here
GEMINI_CACHE_ENABLED=true
GEMINI_CACHE_MONITOR=true
```

### Service Ports

- **Cache Grafana**: 3001 (HTTP)
- **Cache Prometheus**: 9091 (HTTP)
- **Cache Metrics Exporter**: 8080 (HTTP)

### Alert Thresholds

Default alert thresholds (configurable):

- Content Loading P95: >3000ms (warning), >5000ms (critical)
- Prompt Construction P95: >10ms (warning), >50ms (critical)
- Cache Effectiveness: <70% (warning), <50% (critical)
- Memory Usage: >15MB (warning), >20MB (critical)

## Deployment Validation

### Pre-Deployment Checks

Validates before deployment:

- Docker and Docker Compose availability
- Configuration file validity
- Required directories and permissions
- Network port availability
- Disk space requirements

### Post-Deployment Checks

Validates after deployment:

- All services running and healthy
- Endpoints responding correctly
- Configuration loaded and valid
- Metrics collection working
- Dashboard accessibility

## Operational Procedures

### Daily Operations

1. **Morning Health Check** (9:00 AM)
   ```bash
   ./monitoring/cache-health-check.sh
   ```

2. **Cost Savings Validation** (5:00 PM)
   ```bash
   curl http://localhost:8080/cache/health/effectiveness | jq '.daily_savings_usd'
   ```

3. **Performance Monitoring** (Morning & Evening)
   ```bash
   curl http://localhost:8080/cache/health/performance
   ```

### Emergency Procedures

#### Cache Effectiveness Critical (<50%)

```bash
# Check system status
./monitoring/cache-health-check.sh

# Force content reload
curl -X POST http://localhost:8080/cache/content/reload

# Run rollback if needed
./monitoring/cache/runbooks/rollback-content-loading.sh
```

#### Performance Degradation

```bash
# Check performance metrics
curl http://localhost:8080/cache/health/performance

# Run performance rollback
./monitoring/cache/runbooks/rollback-performance.sh
```

#### Configuration Issues

```bash
# Validate configuration
curl http://localhost:8080/cache/config/validate

# Run configuration rollback
./monitoring/cache/runbooks/rollback-configuration.sh
```

## Troubleshooting

### Common Issues

#### Services Not Starting

```bash
# Check Docker status
docker info

# Check service logs
docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs

# Restart services
docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart
```

#### Metrics Not Appearing

```bash
# Check metrics endpoint
curl http://localhost:8080/cache/metrics

# Check Prometheus targets
curl http://localhost:9091/api/v1/targets

# Verify database connection
docker-compose -f monitoring/docker-compose.cache-monitoring.yml exec cache-metrics-exporter python -c "import psycopg2; import os; conn = psycopg2.connect(host=os.getenv('POSTGRES_HOST', 'postgres'), database=os.getenv('POSTGRES_DB', 'WhizDB'), user=os.getenv('POSTGRES_USER', 'admin'), password=os.getenv('POSTGRES_PASSWORD')); print('OK')"
```

#### Dashboard Not Loading

```bash
# Check Grafana logs
docker-compose -f monitoring/docker-compose.cache-monitoring.yml logs cache-grafana

# Check data source
curl http://localhost:3001/api/datasources

# Verify Prometheus connectivity
curl http://localhost:9091/api/v1/query?query=up
```

### Performance Issues

#### High Memory Usage

```bash
# Check container memory
docker stats --no-stream

# Restart to clear memory
docker-compose -f monitoring/docker-compose.cache-monitoring.yml restart cache-metrics-exporter
```

#### Slow Response Times

```bash
# Check system resources
top
iostat -x 1 5

# Check for I/O bottlenecks
iotop
```

## Security Considerations

### Access Control

- Change default Grafana password immediately
- Restrict access to monitoring ports in production
- Use environment variables for sensitive configuration
- Enable HTTPS for production deployments

### Data Protection

- Ensure audit logs don't contain sensitive data
- Implement log rotation and retention policies
- Encrypt configuration files containing secrets
- Use secure API key management

## Scaling and Performance

### Resource Requirements

**Minimum Requirements**:
- CPU: 1 core
- Memory: 2GB
- Disk: 10GB
- Network: 100Mbps

**Recommended for Production**:
- CPU: 2 cores
- Memory: 4GB
- Disk: 50GB SSD
- Network: 1Gbps

### Scaling Guidelines

- **High Volume**: Use external Prometheus with remote storage
- **Multiple Environments**: Deploy separate monitoring stacks
- **Enterprise**: Integrate with existing monitoring infrastructure
- **Global**: Multi-region deployment with centralized dashboards

## Integration

### Existing Prometheus

Add to your `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: 'gemini-cache'
    static_configs:
      - targets: ['cache-metrics-exporter:8080']
    metrics_path: /cache/metrics
```

### Existing Grafana

1. Import dashboard: `monitoring/cache/grafana/dashboards/gemini-cache-dashboard.json`
2. Configure data source to point to your Prometheus
3. Set up notification channels for alerts

### Existing Alerting

Import alert rules from `monitoring/cache/prometheus/rules/cache_alerts.yml`

## Support and Maintenance

### Regular Maintenance

- **Daily**: Health checks, cost validation, performance review
- **Weekly**: Cache effectiveness analysis, content updates
- **Monthly**: Performance optimization, configuration review
- **Quarterly**: Capacity planning, infrastructure updates

### Emergency Contacts

- **On-Call Engineer**: +1-555-0123 (24/7)
- **Operations Team**: ops-team@company.com
- **Development Team**: dev-team@company.com
- **Infrastructure Team**: infrastructure@company.com

### Documentation

- **Operations Runbook**: `runbooks/CACHE_OPERATIONS_RUNBOOK.md`
- **Full Documentation**: `../gemini-cache-monitoring.md`
- **API Documentation**: Available at service endpoints
- **Alert Documentation**: `prometheus/rules/cache_alerts.yml`

## Contributing

### Adding New Metrics

1. Update `exporter/cache_metrics_exporter.py`
2. Add Prometheus metrics definitions
3. Update Grafana dashboard
4. Add alert rules if needed
5. Update documentation

### Modifying Dashboards

1. Edit dashboard in Grafana UI
2. Export updated JSON
3. Replace `grafana/dashboards/gemini-cache-dashboard.json`
4. Test in development environment
5. Deploy to production

### Adding Alerts

1. Edit `prometheus/rules/cache_alerts.yml`
2. Test alert expressions in Prometheus UI
3. Update documentation
4. Deploy and verify alerts fire correctly

## Version History

- **v1.0.0**: Initial production monitoring setup
  - Complete dashboard with cache effectiveness metrics
  - Comprehensive alerting for performance and operational issues
  - Full operational runbooks and rollback procedures
  - Automated deployment validation

## License

This monitoring setup is part of the Gemini Context Caching system and follows the same license terms as the main project.