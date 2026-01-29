# Explosives Intelligence API

AI-powered API for explosives safety, compatibility analysis, and inventory management.

## Features

- **AI Agent**: Natural language interface for explosives queries
- **WebSocket Streaming**: Real-time streaming of agent's thought process
- **Material Compatibility**: Check compatibility between explosive materials
- **Inventory Management**: Track materials across storage locations
- **Smart SQL Generation**: Natural language to SQL queries
- **Interactive Swagger UI**: Full API documentation

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
- `GET /api/materials/{material_number}/compatibility` - Get compatibility info
- `GET /api/materials/{material_number}/inventory` - Get inventory details
- `POST /api/database/query` - Natural language SQL queries
- `GET /api/materials` - List all materials
- `GET /api/storage/summary` - Storage location summary
- `GET /health` - Health check

### WebSocket Streaming

Connect to `ws://localhost:8000/ws/agent` to receive real-time agent updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/agent');

ws.onopen = () => {
    ws.send(JSON.stringify({
        question: "Can I store materials from groups D and F together?",
        model: "gpt-5"
    }));
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data);
    
    // Event types: status, iteration, tool_call_start, 
    // tool_call_result, tool_call_error, final_answer
};
```

## Project Structure

```
explosives_api/
├── agent.py           # Core AI agent logic and tools
├── api.py            # FastAPI application
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Example Usage

### Python Client

```python
import requests

# Ask the agent a question
response = requests.post(
    "http://localhost:8000/api/agent/query",
    json={"question": "What is material 885600034?"}
)
print(response.json()["answer"])

# Check material compatibility
response = requests.get(
    "http://localhost:8000/api/materials/885600034/compatibility"
)
print(response.json())
```

### cURL

```bash
# Ask a question
curl -X POST "http://localhost:8000/api/agent/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Show materials in compatibility group D"}'

# Get material inventory
curl "http://localhost:8000/api/materials/885600034/inventory"
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
