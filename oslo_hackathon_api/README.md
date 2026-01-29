# Oslo Hackathon API

AI-powered API for the DEIG AI Hackathon in Oslo

## Features

- **AI Agent**: Natural language interface for language queries

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
# HANA Database
HANA_HOST=your_hana_host
HANA_PORT=your_hana_port
HANA_USER=your_username
HANA_PASSWORD=your_password

# SAP AI Core
AICORE_CLIENT_ID=your_client_id
AICORE_CLIENT_SECRET=your_client_secret
AICORE_AUTH_URL=your_auth_url
AICORE_BASE_URL=your_base_url
AICORE_RESOURCE_GROUP=your_resource_group
```

## Running the API

```bash
# Development mode with auto-reload
uvicorn api:app --reload

# Production mode
uvicorn api:app --host 0.0.0.0 --port 8000

# Or run directly
python api.py
```

## Access Points

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **WebSocket**: ws://localhost:8000/ws/agent

## API Endpoints

### REST Endpoints

- `POST /api/agent/query` - Ask the AI agent questions
- `GET /health` - Health check

### WebSocket Streaming

Connect to `ws://localhost:8000/ws/agent` to receive real-time agent updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agent');

## Project Structure

```
oslo_hackathon_api/
├── agent.py           # Core AI agent logic and tools
├── api.py            # FastAPI application
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Example Usage

### cURL

```bash
# Ask a question
curl -X POST "http://localhost:8000/api/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are you?}'
```

## Development

The API is built with:
- **FastAPI** - Modern, fast web framework
- **WebSockets** - Real-time bidirectional communication
- **Pydantic** - Data validation
- **SAP AI Core** - AI/LLM integration
- **SAP HANA** - Enterprise database

## License

MIT
