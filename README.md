# Whiz AI Agent

An autonomous customer support agent for parking refund processing, combining intelligent decision-making with automated ticket handling.

## Overview

Whiz AI Agent is an AI-powered automation system designed to streamline parking customer support operations. Built on the Parlant conversational AI framework, it processes refund requests by analyzing ticket data, applying business rules, and making intelligent decisions based on configurable policies.

The system integrates with Freshdesk for ticket management and uses a hybrid approach combining deterministic rules with LLM-powered analysis to handle both straightforward and complex refund scenarios.

## Features

### Core Capabilities

- **Automated Refund Processing**: Complete ticket-to-decision workflow with policy-based decision making
- **Hybrid Decision Engine**: Combines rule-based logic (< 2s) with LLM-powered analysis (< 10s)
- **Freshdesk Integration**: Automated ticket ingestion, note creation, and status updates via webhook
- **Intelligent Booking Extraction**: Pattern matching + LLM fallback to extract structured booking data
- **Security Scanning**: Lakera API integration for content safety before processing
- **Comprehensive Audit Trail**: PostgreSQL logging of all decisions, actions, and metrics
- **Webhook Automation**: Real-time ticket processing triggered by Freshdesk events
- **Conversational Agent**: Interactive chat interface for manual ticket processing

### Decision Outcomes

The system produces three types of decisions:

- **Approved**: Clear policy support with refund amount and cancellation reason
- **Denied**: Policy violation with specific reasoning and customer-friendly explanation
- **Needs Human Review**: Missing data, ambiguous cases, or low confidence requiring agent review

### Intelligent Routing System

The system includes a multi-gate routing pipeline that processes tickets through deterministic checks before LLM analysis:

**Pre-Arrival Gate**
- Instantly approves cancellations made before the booking start time
- Pure time arithmetic with timezone-aware comparisons
- Processing time: < 200ms (no LLM required)
- Handles 10-15% of tickets automatically

**MOR (Merchant of Record) Gate**
- Identifies third-party vendor bookings requiring special handling
- Pattern-based vendor detection
- Routes to appropriate processing paths

**Product Type Gate**
- Classifies bookings by product type (on-demand, confirmed, etc.)
- Applies type-specific business rules

### Timeline Analysis

Deterministic timeline analysis for accurate scenario classification:

- **Early Arrival Detection**: Identifies when customers arrive before booking start time
- **Overstay Calculation**: Calculates exact overstay duration beyond booked time
- **Exit Complication Detection**: Distinguishes between poor experience and refund eligibility

### Contradiction Detection

Intelligent detection of conflicting customer information:

- **Mutually Exclusive Claims**: Detects incompatible complaint types
- **Logical Inconsistencies**: Identifies response mismatches
- **Processing Time**: < 2ms per ticket

### Booking Verification

Validates booking information against system records:

- **Temporal Mismatch Detection**: Identifies significant date discrepancies
- **Customer Identity Verification**: Compares ticket submitter with booking owner
- **Smart Escalation**: Only escalates when mismatches are significant

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose                            │
├─────────────────┬─────────────────┬─────────────────────────┤
│   Parlant       │   PostgreSQL    │   Voice (Future)        │
│   Port 8800     │   Port 5432     │   Port 7860             │
│   Port 8801     │                 │                         │
│   (Webhook)     │                 │                         │
└─────────────────┴─────────────────┴─────────────────────────┘
```

**Parlant Service (Port 8800, 8801)**
- AI agent backend with Parlant SDK
- Webhook server for Freshdesk integration
- Tool orchestration for external APIs
- Decision engine with hybrid rule/LLM processing

**PostgreSQL (Port 5432)**
- Audit logs and decision history
- Performance metrics
- Customer context storage

### Processing Flow

```
Webhook Event
  ↓
Signature Validation
  ↓
Routing Pipeline
  ├─ Pre-Arrival Gate → Instant Approval (if applicable)
  ├─ MOR Gate → Vendor-specific routing
  └─ Product Type Gate → Type-specific rules
  ↓
Security Scan (Lakera)
  ↓
Booking Extraction
  ↓
Contradiction Detection
  ↓
Rule Engine (< 2s)
  ├─ High Confidence → Decision
  └─ Low Confidence → LLM Analysis (< 10s)
  ↓
Document Decision
  ↓
Audit Log
```

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Gemini API key (or OpenAI API key)
- Freshdesk account with API access
- Lakera API key for security scanning

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd whiz-ai-agent
   ```

2. **Configure environment variables**
   ```bash
   cp examples/.env.example .env
   # Edit .env and add your API keys
   ```

3. **Start services**
   ```bash
   docker-compose up -d
   ```

4. **Verify services**
   ```bash
   docker-compose ps
   docker-compose logs -f parlant
   ```

5. **Access the application**
   - Parlant UI: http://localhost:8800
   - Webhook endpoint: http://localhost:8801/webhook/freshdesk

### Configuration

Required environment variables (see `examples/.env.example`):

```env
# LLM Provider
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
GEMINI_MODEL=gemini-2.5-flash

# Freshdesk Integration
FRESHDESK_DOMAIN=your-domain.freshdesk.com
FRESHDESK_API_KEY=your-api-key

# Security
LAKERA_API_KEY=your-api-key

# Webhook
WEBHOOK_SECRET=your-secret
WEBHOOK_ENABLED=true

# Database
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=your-password
```

### Configuration Files

The system requires configuration files that are not included in the repository for security reasons:

**Policy Documents** (`parlant/context/`)
- Refund policies and business rules
- Decision charts and scenarios
- See `parlant/context/README.md` for required format

**Router Configuration** (`parlant/tools/routing/router_config.json`)
- Vendor lists and thresholds
- Gate configurations
- See `parlant/tools/routing/router_config.json.example` for template

**Classification Rules** (`parlant/tools/refund_decision/classification_config.json`)
- Scenario classification patterns
- Confidence thresholds
- See `parlant/tools/refund_decision/classification_config.json.example` for template

## Usage

### Automated Processing (Webhook)

1. Configure Freshdesk webhook to point to your endpoint
2. Tickets are automatically processed when created/updated
3. Decisions are documented as private notes on the ticket

### Interactive Processing (Chat)

1. Open http://localhost:8800
2. Type: "Process ticket 12345"
3. System provides step-by-step feedback
4. Review the final decision and reasoning

## Testing

All tests must be run inside the Docker container:

```bash
# Run all tests
docker-compose exec parlant pytest

# Run specific test suite
docker-compose exec parlant pytest tests/tools/ -v

# Run integration tests
docker-compose exec parlant pytest tests/integration/ -v

# Run with coverage
docker-compose exec parlant pytest --cov=app_tools --cov-report=html
```

## Project Structure

```
├── parlant/                    # Main application
│   ├── main.py                 # Agent entry point
│   ├── webhook_server.py       # FastAPI webhook endpoint
│   ├── journey_router.py       # Request routing
│   ├── tools/                  # Parlant tools
│   │   ├── integrations/       # External API clients
│   │   ├── infrastructure/     # Core infrastructure
│   │   ├── refund_decision/    # Decision logic
│   │   ├── routing/            # Routing pipeline
│   │   ├── booking_analysis/   # Booking extraction
│   │   └── ticket_processing/  # Ticket workflows
│   ├── context/                # Policy documents (gitignored)
│   └── retrievers/             # Policy retrieval
├── tests/                      # Test suites
│   ├── tools/                  # Unit tests
│   ├── integration/            # Integration tests
│   └── performance/            # Performance tests
├── monitoring/                 # Prometheus/Grafana configs
├── postgres/                   # Database migrations
├── scripts/                    # Utility scripts
├── docs/                       # Documentation
│   ├── operations/             # Operational guides
│   ├── examples/               # Example code
│   └── archive/                # Archived reports
└── examples/                   # Example configurations
```

## Monitoring

### Health Checks

```bash
# Webhook health
curl http://localhost:8801/webhook/health

# Metrics endpoint
curl http://localhost:8801/webhook/metrics
```

### Logs

```bash
# All logs
docker-compose logs -f parlant

# Decision logs
docker-compose logs parlant | grep "Decision:"

# Webhook logs
docker-compose logs parlant | grep webhook
```

## Troubleshooting

### Common Issues

**Webhook not receiving events**
- Verify webhook URL is publicly accessible
- Check Freshdesk webhook configuration
- Verify WEBHOOK_SECRET matches

**Decision making issues**
- Check policy files exist in `parlant/context/`
- Verify configuration files are valid JSON
- Review logs for specific errors

**Database connection errors**
- Wait 10-20 seconds after startup for PostgreSQL to initialize
- Check credentials in `.env`

**Gemini API errors**
- 404: Update GEMINI_MODEL to `gemini-2.5-flash`
- 401: Verify GEMINI_API_KEY
- 429: Rate limit exceeded

### Debug Commands

```bash
# Check service status
docker-compose ps

# Restart services
docker-compose restart parlant

# Clear database and restart
docker-compose down -v
docker-compose up -d

# Run tests
docker-compose exec parlant pytest -v
```

## Roadmap

### Current Status

- ✅ Complete decision-making pipeline
- ✅ Intelligent routing with pre-arrival, MOR, and product type gates
- ✅ Webhook automation with < 5 second processing
- ✅ Contradiction and mismatch detection
- ✅ Freshdesk integration
- ✅ Interactive chat interface
- ✅ Comprehensive audit logging

### Future Enhancements

- Automated refund processing via booking API
- Voice interface (WebRTC/Pipecat)
- Advanced analytics dashboard
- Multi-booking support
- Partial refund calculations

## Technology Stack

- **Python 3.x**: Primary language
- **Parlant SDK (>= 3.0.0)**: AI agent framework
- **PostgreSQL 15**: Database
- **Docker Compose**: Container orchestration
- **FastAPI**: Webhook server
- **httpx**: Async HTTP client
- **pytest**: Testing framework

## License

Proprietary - All rights reserved.
