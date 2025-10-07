# Parlant-Pipecat Voice Assistant 🎙️

A voice-enabled customer support system combining Parlant's AI agent framework with Pipecat's real-time voice pipeline.

## Features

- 🗣️ **Voice Interface** - WebRTC-based voice chat with STT/TTS
- 🤖 **AI Agent** - Structured conversation journeys with Parlant
- 🗄️ **Database Integration** - PostgreSQL for product catalog
- 🐳 **Docker Compose** - One-command deployment
- 📞 **Refund Processing** - Complete refund eligibility workflow

## Architecture

```
Architecture Overview
Three main components:

Parlant (Port 8800) - The AI agent backend

Manages conversation journeys (structured dialogue flows)
Implements a refund request workflow with conditional logic
Connects to PostgreSQL for product data
Provides tools: check_refund_eligibility, process_refund, find_products


Pipecat (Port 7860) - The voice interface layer

Handles WebRTC audio streaming from browser
Speech-to-text using OpenAI Whisper
Text-to-speech using OpenAI TTS
Voice Activity Detection (VAD) using Silero
ParlantBridge: Custom processor that connects voice pipeline to Parlant


PostgreSQL (Port 5432) - Database

Product catalog with sample electronics/accessories
Order history structure
```

```
Browser (WebRTC) ←→ Pipecat (Voice I/O) ←→ Parlant (AI Logic) ←→ PostgreSQL
```

## Key Flow

```
User speaks → Pipecat (STT) → ParlantBridge → Parlant Agent → Response
                                                    ↓
                                              PostgreSQL
                                                    ↓
Response ← Pipecat (TTS) ← ParlantBridge ← Agent decides action
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/parlant-pipecat-voice.git
   cd parlant-pipecat-voice
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

3. **Start all services**
   ```bash
   docker-compose up -d
   ```

4. **Access the application**
   - Parlant UI: http://localhost:8800
   - Voice Client: http://localhost:7860/client
   - Database: localhost:5432

### First Use

1. Open http://localhost:7860/client in your browser
2. Click "Connect" and allow microphone access
3. Say "Hello" to start chatting with the assistant
4. Try: "I'd like to get a refund" or "Show me your products"

## Services

### Parlant (Port 8800)
The AI agent backend with:
- Refund journey with conditional logic
- Product search tool
- Database integration
- Session management

### Pipecat (Port 7860)
The voice interface providing:
- Real-time speech-to-text (OpenAI Whisper)
- Text-to-speech (OpenAI TTS)
- WebRTC audio streaming
- Event-driven integration with Parlant

### PostgreSQL (Port 5432)
Database containing:
- Product catalog
- Customer data
- Order history

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...

# Parlant Configuration
PARLANT_BASE_URL=http://parlant:8800

# Database Configuration
POSTGRES_DB=ranjayDB
POSTGRES_USER=ranjay.kumar
POSTGRES_PASSWORD=ranjay
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Voice Bridge Tuning
TURN_TIMEOUT_SECS=30
POLL_WAIT_SECS=10
MAX_EMPTY_POLLS=3
```

## Development

### Running Locally (without Docker)

1. **Start PostgreSQL**
   ```bash
   docker run -d -p 5432:5432 \
     -e POSTGRES_DB=ranjayDB \
     -e POSTGRES_USER=ranjay.kumar \
     -e POSTGRES_PASSWORD=ranjay \
     postgres:15
   ```

2. **Run Parlant**
   ```bash
   cd parlant
   pip install -r requirements.txt
   python main.py
   ```

3. **Run Pipecat**
   ```bash
   cd voice
   pip install -r requirements.txt
   python app.py
   ```

### Logs

View logs for specific services:
```bash
docker-compose logs -f parlant
docker-compose logs -f pipecat
docker-compose logs -f postgres
```

## Project Structure

```
.
├── README.md
├── docker-compose.yml
├── .env.example
├── .gitignore
├── parlant/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py
├── voice/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
└── postgres/
    └── init.sql
```

## Troubleshooting

### Voice not working
- Ensure microphone permissions are granted
- Check browser console for WebRTC errors
- Verify OPENAI_API_KEY is set correctly

### Database connection errors
- Wait 10-20 seconds after `docker-compose up` for PostgreSQL to initialize
- Check logs: `docker-compose logs postgres`

### Agent not responding
- Verify Parlant is running: `curl http://localhost:8800`
- Check agent_id.txt was created
- Review logs: `docker-compose logs parlant pipecat`

## License

MIT

## Contributing

Pull requests welcome! Please open an issue first to discuss changes.
```