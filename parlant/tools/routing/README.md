# MOR/Product Type Router

This directory contains the MOR (Merchant of Record) and Product Type routing gates that provide instant rejection for non-refundable tickets before expensive LLM processing.

## Quick Start

1. **Configuration**: Edit `router_config.json` to customize vendor lists and thresholds
2. **Validation**: Run `python validate_config.py` to check configuration
3. **Monitoring**: Check logs for gate performance and rejection metrics

## Files Overview

### Core Implementation
- `gates/mor_gate.py` - MOR gate implementation
- `gates/product_type_gate.py` - Product type gate implementation  
- `models.py` - Data models for gate decisions
- `ticket_router.py` - Main router with gate integration
- `config_loader.py` - Configuration loading and validation
- `metrics_tracker.py` - Performance and rejection metrics

### Configuration
- `router_config.json` - Main configuration file
- `config_examples/` - Example configurations for different scenarios
- `validate_config.py` - Configuration validation script

### Documentation
- `CONFIGURATION_GUIDE.md` - Complete configuration documentation
- `OPERATOR_GUIDE.md` - Operational procedures and workflows
- `TROUBLESHOOTING_GUIDE.md` - Problem diagnosis and resolution
- `ESCALATION_PROCEDURES.md` - Escalation paths and procedures
- `OPERATOR_DOCUMENTATION_INDEX.md` - Complete documentation index
- `README.md` - This file

## Configuration Examples

### Default Configuration
```json
{
  "mor_gate": {
    "third_party_vendors": ["axs", "seatgeek", "stubhub", ...],
    "parkwhiz_partners": ["parkwhiz", "google", "waze", "apple"],
    "special_partners": {"groupon": "groupon_credit_only"}
  },
  "product_type_gate": {
    "on_demand_threshold_hours": 4
  },
  "router_settings": {
    "enable_mor_product_type_gates": true,
    "enable_hot_reload": true
  }
}
```

### Common Operations

**Add new third-party vendor:**
```bash
# Edit router_config.json to add vendor to third_party_vendors list
# Changes apply automatically within 30 seconds
```

**Adjust on-demand threshold:**
```bash
# Change on_demand_threshold_hours value
# Higher = more conservative (fewer rejections)
# Lower = more aggressive (more rejections)
```

**Emergency disable:**
```bash
# Set enable_mor_product_type_gates: false
# All tickets bypass gates and continue to existing pipeline
```

## Performance Targets

- **MOR Gate**: <20ms per ticket
- **Product Type Gate**: <50ms per ticket
- **Combined**: <100ms total
- **Rejection Rate**: 20-30% of tickets (based on historical data)

## Monitoring

### Key Metrics
- Rejection counts by vendor (AXS: 45, SeatGeek: 32, etc.)
- Product type distribution (on-demand: 87, transient: 543, event: 289)
- Processing latency (95th percentile)
- Cost savings (rejected tickets × average LLM cost)

### Log Messages
```
INFO: MOR gate rejected ticket 12345: Partner=SeatGeek (vendor: SeatGeek)
INFO: Product type gate rejected ticket 12346: on-demand (duration: 2.5h)
WARNING: Unknown Partner value: 'NewVendor' - defaulting to ParkWhiz MOR
```

## Testing

### Validate Configuration
```bash
python validate_config.py
python validate_config.py config_examples/conservative_config.json
```

### Run Tests
```bash
# Run all routing tests
docker-compose exec parlant pytest tests/tools/test_mor_gate.py -v
docker-compose exec parlant pytest tests/tools/test_product_type_gate.py -v
docker-compose exec parlant pytest tests/tools/test_combined_gate_performance.py -v

# Run integration tests
docker-compose exec parlant pytest tests/integration/test_mor_product_type_router_integration.py -v
```

## Troubleshooting

### Configuration Issues
1. **Validation Errors**: Run `python validate_config.py` for detailed error messages
2. **JSON Syntax**: Use online JSON validator or `python -m json.tool router_config.json`
3. **Hot Reload Not Working**: Check file permissions and `enable_hot_reload` setting

### Performance Issues
1. **High Latency**: Check for very large vendor lists or frequent config reloads
2. **High Rejection Rate**: Review recent config changes and disputed tickets
3. **Low Rejection Rate**: Verify gates are enabled and vendor lists are complete

### Unknown Partners
1. **Monitor Logs**: Look for "Unknown Partner value" warnings
2. **Analyze Patterns**: Determine if new partners should be added to config
3. **Update Config**: Add partners to appropriate lists (third_party_vendors or parkwhiz_partners)

## Development

### Adding New Gates
1. Create gate class in `gates/` directory
2. Add gate integration to `ticket_router.py`
3. Update configuration schema in `config_loader.py`
4. Add tests in `tests/tools/`

### Configuration Changes
1. Update `router_config.json` schema
2. Add validation logic in `config_loader.py`
3. Update documentation in `CONFIGURATION_GUIDE.md`
4. Add example configurations in `config_examples/`

## Support

### For Operators
- **Start Here**: `OPERATOR_DOCUMENTATION_INDEX.md` - Complete documentation guide
- **Daily Operations**: `OPERATOR_GUIDE.md` - Workflows and procedures
- **Problem Solving**: `TROUBLESHOOTING_GUIDE.md` - Quick issue resolution
- **Need Help**: `ESCALATION_PROCEDURES.md` - When and how to escalate

### For Developers
- **Configuration**: `CONFIGURATION_GUIDE.md` - Technical configuration details
- **Performance Problems**: Check metrics and logs
- **Development Questions**: Review code comments and tests