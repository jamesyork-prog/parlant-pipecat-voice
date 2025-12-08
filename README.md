# Whiz AI Agent

An autonomous customer support agent for ParkWhiz refund processing, combining intelligent decision-making with automated ticket handling.

## Overview

Whiz AI Agent is an AI-powered automation system designed to streamline ParkWhiz customer support operations. Built on the Parlant conversational AI framework, it processes refund requests by analyzing ticket data, applying business rules, and making intelligent decisions based on configurable policies. The system integrates with Freshdesk for ticket management and uses a hybrid approach combining deterministic rules with LLM-powered analysis to handle both straightforward and complex refund scenarios.

## Features

### Current Capabilities

- **Automated Refund Processing**: Complete ticket-to-decision workflow with policy-based decision making
- **Hybrid Decision Engine**: Combines rule-based logic (< 2s) with LLM-powered analysis (< 10s) for intelligent decisions
- **Freshdesk Integration**: Automated ticket ingestion, note creation, and status updates via webhook
- **Intelligent Booking Extraction**: Pattern matching + LLM fallback to extract structured booking data from ticket text
- **Security Scanning**: Lakera API integration for content safety before processing
- **Policy-Driven Decisions**: Configurable refund rules loaded from JSON/Markdown files
- **Comprehensive Audit Trail**: PostgreSQL logging of all decisions, actions, and metrics
- **Webhook Automation**: Real-time ticket processing triggered by Freshdesk events
- **Conversational Agent**: Interactive chat interface for manual ticket processing and agent assistance

### Decision Outcomes

The system produces three types of decisions:

- **Approved**: Clear policy support with refund amount and ParkWhiz cancellation reason
- **Denied**: Policy violation with specific reasoning and customer-friendly explanation
- **Needs Human Review**: Missing data, ambiguous cases, or low confidence requiring agent review

## Getting Started

### Prerequisites

- Docker & Docker Compose
- Gemini API key (or OpenAI API key)
- Freshdesk account with API access
- Lakera API key for security scanning
- PostgreSQL 15 (included in Docker Compose)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/whiz-ai-agent.git
   cd whiz-ai-agent
   ```

2. **Configure environment variables**
   ```bash
   cp examples/.env.example .env
   # Edit .env and add your API keys
   ```

   Required environment variables:
   ```env
   # LLM Provider
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=your-gemini-api-key
   GEMINI_MODEL=gemini-2.5-flash

   # Freshdesk Integration
   FRESHDESK_DOMAIN=your-domain.freshdesk.com
   FRESHDESK_API_KEY=your-freshdesk-api-key

   # Security
   LAKERA_API_KEY=your-lakera-api-key

   # Webhook Configuration
   WEBHOOK_SECRET=your-secure-random-secret
   WEBHOOK_ENABLED=true

   # Database
   POSTGRES_DB=WhizDB
   POSTGRES_USER=admin
   POSTGRES_PASSWORD=whiz
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Verify services are running**
   ```bash
   # Check service status
   docker-compose ps

   # View logs
   docker-compose logs -f parlant
   ```

5. **Access the application**
   - Parlant UI: http://localhost:8800
   - Webhook endpoint: http://localhost:8801/webhook/freshdesk
   - Database: localhost:5432

### First Use

#### Automated Ticket Processing (Webhook)

1. Configure Freshdesk webhook to point to your endpoint
2. Create a refund request ticket in Freshdesk
3. System automatically processes the ticket in the background
4. Review the decision in the ticket's private notes

#### Interactive Processing (Chat)

1. Open http://localhost:8800 in your browser
2. Type: "Process ticket 12345" (replace with actual ticket ID)
3. System provides step-by-step feedback during processing
4. Review the final decision and reasoning

## Usage

### Automated Ticket Processing

The system automatically processes tickets when triggered by Freshdesk webhooks:

```
Freshdesk Ticket Created/Updated
  ↓
Webhook POST → /webhook/freshdesk
  ↓
Security Scan (Lakera API)
  ↓
Extract Booking Info (Pattern + LLM)
  ↓
Apply Business Rules
  ↓
Make Decision (Rules or LLM)
  ↓
Document Decision in Freshdesk
  ↓
Tag Ticket "Processed by Whiz Agent"
```

**Processing Time:**
- Rule-based decisions: < 2 seconds
- LLM-based decisions: < 10 seconds
- End-to-end: < 15 seconds

**Decision Documentation:**

The system adds a private note to each ticket with:
- Decision outcome (Approved/Denied/Needs Human Review)
- Detailed reasoning and policy applied
- Confidence level and method used
- ParkWhiz cancellation reason (for approved refunds)
- Processing time and metadata

### Conversational Agent

Use the chat interface for manual processing and agent assistance:

**Process a specific ticket:**
```
Agent: "Process ticket 1206331"
System: [Fetches ticket, runs security scan, extracts booking info, makes decision]
System: "Decision: Approved - Cancellation made 8 days before event..."
```

**Get ticket information:**
```
Agent: "Get ticket 1206331"
System: [Returns ticket details, subject, description, status]
```

**Ask about policies:**
```
Agent: "What's the refund policy for cancellations 5 days before the event?"
System: [Retrieves policy documents and explains the rules]
```

### Testing

**IMPORTANT: All tests MUST be run inside the Docker container.**

```bash
# Run all tests
docker-compose exec parlant pytest

# Run specific test suite
docker-compose exec parlant pytest tests/tools/test_booking_extractor.py -v

# Run integration tests
docker-compose exec parlant pytest tests/integration/ -v

# Run with coverage
docker-compose exec parlant pytest --cov=app_tools --cov-report=html
```

## Roadmap

### Current Status: MVP Phase

The system is in the MVP (Minimum Viable Product) phase, focusing on decision-making accuracy and confidence building. The agent documents decisions in Freshdesk but does NOT automatically process refunds via ParkWhiz API.

**What Works Now:**
- ✅ Complete decision-making pipeline (extraction → rules → LLM → decision)
- ✅ Freshdesk integration (ticket retrieval, note creation, tagging)
- ✅ Webhook automation for real-time processing
- ✅ Interactive chat interface for manual processing
- ✅ Comprehensive logging and metrics

**What's NOT Included:**
- ❌ Automatic refund processing via ParkWhiz API
- ❌ Automatic booking cancellations
- ❌ Payment processing or refund transactions
- ❌ Partial refund calculations
- ❌ Multi-booking handling

### Future Enhancements

**Phase 1: Validation & Tuning (Current)**
- Review agent decisions against human judgment
- Measure accuracy, precision, and recall
- Identify policy gaps and edge cases
- Tune confidence thresholds

**Phase 2: ParkWhiz Integration (Current)**
- Implement ParkWhiz refund API calls
- Add booking cancellation logic
- Implement refund transaction processing
- Add rollback handling for failed refunds

**Phase 3: Voice Integration**
- WebRTC-based voice interface using Pipecat
- Speech-to-text (OpenAI Whisper)
- Text-to-speech (OpenAI TTS)
- Voice Activity Detection (VAD)
- Real-time voice conversations with the agent

**Phase 4: Advanced Features**
- Partial refund calculations
- Multi-booking support
- Automatic customer notifications
- Refund status tracking and reporting
- Advanced analytics and dashboards

**Phase 5: Full Automation**
- Remove human review requirement for high-confidence decisions
- Implement automatic processing for approved decisions
- Real-time monitoring and alerting
- Production deployment with full automation

## Architecture

### System Components

The Whiz AI Agent consists of three containerized services:

#### 1. Parlant (Port 8800)

The AI agent backend providing:

- **Conversational AI Framework**: Structured dialogue flows using Parlant SDK
- **Journey Management**: Automated and interactive processing journeys
- **Tool Orchestration**: Coordinates Freshdesk, ParkWhiz, Lakera, and database operations
- **Decision Engine**: Hybrid rule-based + LLM-powered decision making
- **Policy Management**: Loads and applies refund policies from configuration files
- **Webhook Server**: FastAPI endpoint for Freshdesk webhook integration (port 8801)
- **Session Management**: Stateful conversation handling

**Key Components:**
- `main.py`: Agent creation and journey definitions
- `webhook_server.py`: FastAPI webhook endpoint
- `journey_router.py`: Routes requests to appropriate journeys
- `tools/`: Parlant tools for external integrations
- `retrievers/`: Policy document retrieval
- `context/`: Refund policy configuration files

#### 2. PostgreSQL (Port 5432)

Database providing:

- **Audit Logs**: Complete history of all decisions and actions
- **Metrics Storage**: Performance and quality metrics
- **Customer Context**: Historical interaction data
- **Session State**: Conversation state persistence

**Schema:**
- `decision_audit`: Decision history with full context
- `metrics`: Performance and quality metrics
- `customer_data`: Customer information and preferences
- `session_state`: Active conversation state

#### 3. Voice (Port 7860) - Roadmap

Future voice interface providing:

- **WebRTC Audio Streaming**: Real-time voice communication
- **Speech-to-Text**: OpenAI Whisper integration
- **Text-to-Speech**: OpenAI TTS integration
- **Voice Activity Detection**: Silero VAD
- **Parlant Bridge**: Connects voice pipeline to AI agent

**Status:** Planned for Phase 3 (not currently active)

### Data Flow

#### Automated Processing Flow

```
Freshdesk Webhook
  ↓
Webhook Validator (OAuth2)
  ↓
Journey Router → Automated Processing Journey
  ↓
Security Scan (Lakera API)
  ↓
Booking Extraction (Pattern + LLM)
  ↓
Rule Engine (< 2s)
  ├─ High Confidence → Decision
  └─ Low Confidence → LLM Analyzer (< 10s) → Decision
  ↓
Cancellation Reason Mapper (if Approved)
  ↓
Document Decision (Freshdesk API)
  ↓
Audit Log (PostgreSQL)
  ↓
HTTP 200 Response
```

#### Interactive Processing Flow

```
Agent Chat Message
  ↓
Journey Router → Interactive Processing Journey
  ↓
Chat State: "Fetching ticket..."
  ↓
Chat State: "Running security scan..."
  ↓
Chat State: "Extracting booking info..."
  ↓
Chat State: "Making decision..."
  ↓
Chat State: "Decision: [Approved/Denied/Escalated]"
  ↓
Final Summary with Reasoning
```

### Integration Points

- **Freshdesk API**: Ticket retrieval, note creation, tag updates
- **Gemini API**: Booking extraction, complex case analysis
- **Lakera API**: Content security scanning
- **PostgreSQL**: Audit logging and metrics
- **ParkWhiz API**: Booking data retrieval (fallback), refund processing (future)

### Technology Stack

**Core Technologies:**
- Python 3.x
- Parlant SDK (>= 3.0.0): AI agent framework
- PostgreSQL 15: Database
- Docker Compose: Container orchestration

**Key Libraries:**
- `httpx`: Async HTTP client for API integrations
- `psycopg2-binary`: PostgreSQL adapter
- `python-dotenv`: Environment variable management
- `pytest`, `pytest-asyncio`, `pytest-httpx`: Testing framework
- `fastapi`: Webhook server
- `uvicorn`: ASGI server

## Configuration

### Environment Variables

All configuration is managed through environment variables in `.env`:

```env
# LLM Provider Configuration
LLM_PROVIDER=gemini  # or 'openai'
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

# Freshdesk Integration
FRESHDESK_DOMAIN=your-domain.freshdesk.com
FRESHDESK_API_KEY=your-freshdesk-api-key

# Security
LAKERA_API_KEY=your-lakera-api-key

# Webhook Configuration
WEBHOOK_SECRET=your-secure-random-secret
WEBHOOK_ENABLED=true
WEBHOOK_PORT=8801
WEBHOOK_PATH=/webhook/freshdesk
WEBHOOK_EVENTS=ticket_created,ticket_updated
WEBHOOK_RATE_LIMIT=100  # requests per minute

# Database
POSTGRES_DB=WhizDB
POSTGRES_USER=admin
POSTGRES_PASSWORD=whiz
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Parlant Configuration
PARLANT_BASE_URL=http://parlant:8800
```

### Policy Configuration

Refund policies are configured in `parlant/context/processed/`:

- `refund_rules.json`: Business rules for LLM context
- `refund_guide.json`: Policy guidance text
- `refund_scenario_decision_chart.md`: Decision tree logic
- `refund_policy_condensed.md`: Human-readable policy summary
- `ai_vs_human_refund_scenarios.md`: Escalation criteria

**Note:** Business rules are hardcoded in `parlant/tools/rule_engine.py`. JSON files provide context for LLM analysis only.

### Docker Configuration

Services are defined in `docker-compose.yml`:

```yaml
services:
  parlant:
    ports:
      - "8800:8800"  # Parlant UI
      - "8801:8801"  # Webhook endpoint
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - FRESHDESK_API_KEY=${FRESHDESK_API_KEY}
      # ... other env vars

  postgres:
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  voice:  # Optional - not active in MVP
    ports:
      - "7860:7860"
```

## Monitoring & Observability

### Logging

View logs for specific services:

```bash
# All Whiz Agent logs
docker-compose logs -f parlant

# Decision-making logs
docker-compose logs parlant | grep -E "BookingExtractor|RuleEngine|LLMAnalyzer|DecisionMaker"

# Webhook logs
docker-compose logs parlant | grep webhook

# Database logs
docker-compose logs -f postgres
```

### Metrics

The system tracks key metrics for each decision:

**Performance Metrics:**
- Processing time (ms)
- Method used (rules/llm/hybrid)
- Extraction method (pattern/llm)
- API calls made

**Quality Metrics:**
- Confidence level (high/medium/low)
- Decision distribution (Approved/Denied/Escalated)
- Escalation rate
- Rule match rate

**Error Metrics:**
- Extraction failures
- LLM timeouts
- Policy load errors
- API errors

### Health Checks

```bash
# Webhook endpoint health
curl http://localhost:8801/webhook/health

# Parlant service health
curl http://localhost:8800

# Database connection
docker-compose exec postgres psql -U admin -d WhizDB -c "SELECT 1;"
```

## Troubleshooting

### Common Issues

**Webhook not receiving events:**
- Verify webhook URL is publicly accessible
- Check Freshdesk webhook configuration
- Verify WEBHOOK_SECRET matches in both systems
- Review signature validation logs

**Decision making issues:**
- Check policy files exist in `parlant/context/processed/`
- Verify JSON files are valid
- Review RuleEngine logic in `parlant/tools/rule_engine.py`
- Restart service to reload policies

**Booking extraction failing:**
- Review ticket format - ensure booking info is in notes or description
- Check Gemini API logs
- Verify GEMINI_API_KEY is set correctly
- Test pattern extraction

**Database connection errors:**
- Wait 10-20 seconds after `docker-compose up` for PostgreSQL to initialize
- Check logs: `docker-compose logs postgres`
- Verify credentials in `.env`

**Gemini API errors:**
- **404 Model Not Found**: Update GEMINI_MODEL to `gemini-2.5-flash`
- **401 Authentication**: Verify GEMINI_API_KEY is correct
- **429 Rate Limit**: Free tier has 15 RPM limit, consider upgrading

### Debug Commands

```bash
# Run tests
docker-compose exec parlant pytest -v

# Check service status
docker-compose ps

# Restart services
docker-compose restart parlant

# Clear database and restart
docker-compose down -v
docker-compose up -d

# View recent decisions
docker-compose logs parlant | grep "Decision:" | tail -20
```

## Documentation

Detailed documentation is available in the `.kiro/specs/` directory:

- **webhook-automation/**: Webhook integration specification
- **policy-based-decisions/**: Decision-making system specification
- **repository-rename-restructure/**: Repository restructuring documentation

Additional resources:

- `planning/docs/`: Architecture clarifications
- `tests/*/README.md`: Test suite documentation
- `scripts/README.md`: Utility scripts documentation

## Development

### Running Tests

```bash
# Run all tests
docker-compose exec parlant pytest

# Run specific test file
docker-compose exec parlant pytest tests/tools/test_booking_extractor.py -v

# Run integration tests
docker-compose exec parlant pytest tests/integration/ -v

# Run with coverage
docker-compose exec parlant pytest --cov=app_tools --cov-report=html
```

### Adding New Rules

Business rules are hardcoded in `parlant/tools/rule_engine.py`:

```python
# In RuleEngine.apply_rules()
if booking_info.get("booking_type") == "monthly" and days_before_event >= 3:
    return {
        "decision": "Approved",
        "reasoning": "Monthly passes can be cancelled 3+ days in advance",
        "policy_rule": "monthly_cancellation_3_days",
        "confidence": "high"
    }
```

After updating rules:
1. Restart the service: `docker-compose restart parlant`
2. Run tests: `docker-compose exec parlant pytest tests/tools/test_rule_engine.py -v`

### Project Structure

```
.
├── README.md
├── docker-compose.yml
├── .env
├── parlant/                    # Main application code
│   ├── main.py                 # Agent creation and journey definitions
│   ├── webhook_server.py       # FastAPI webhook endpoint
│   ├── journey_router.py       # Journey routing logic
│   ├── tools/                  # Parlant tools
│   │   ├── freshdesk_tools.py
│   │   ├── booking_extractor.py
│   │   ├── rule_engine.py
│   │   ├── llm_analyzer.py
│   │   ├── decision_maker.py
│   │   └── ...
│   ├── retrievers/             # Policy document retrieval
│   └── context/                # Policy configuration files
├── tests/                      # Test suite
│   ├── tools/
│   ├── integration/
│   └── debug/
├── postgres/                   # Database initialization
├── voice/                      # Voice interface (future)
└── scripts/                    # Utility scripts
```

## License

MIT




