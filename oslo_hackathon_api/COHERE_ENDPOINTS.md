# Cohere Endpoints Documentation

This document describes the new Cohere endpoints added to the Oslo Hackathon API.

## Overview

The API now includes three main endpoint groups:
1. **OpenAI Chat** - Original endpoint for GPT models
2. **Cohere Chat** - New endpoint for Cohere's command-a-reasoning model
3. **Cohere Stream** - New streaming endpoint for real-time responses

---

## Endpoints

### 1. OpenAI Chat: `POST /api/chat`

Original endpoint for chatting with OpenAI models (gpt-5, gpt-4o, gpt-35-turbo).

**Request:**
```json
{
  "message": "What is quantum computing?",
  "model": "gpt-5"
}
```

**Response:**
```json
{
  "message": "What is quantum computing?",
  "response": "Quantum computing is...",
  "model_used": "gpt-5",
  "timestamp": "2026-01-21T10:30:00"
}
```

---

### 2. Cohere Chat: `POST /api/cohere/chat`

Chat with Cohere's command-a-reasoning model with optional thinking mode.

**Features:**
- Advanced reasoning capabilities
- Optional thinking mode (shows reasoning process)
- Configurable frequency penalty
- Token usage metrics

**Request:**
```json
{
  "message": "Explain the theory of relativity",
  "model": "cohere--command-a-reasoning",
  "enable_thinking": false,
  "frequency_penalty": 0.8
}
```

**Response:**
```json
{
  "message": "Explain the theory of relativity",
  "response": "The theory of relativity...",
  "thinking": null,
  "model_used": "cohere--command-a-reasoning",
  "finish_reason": "COMPLETE",
  "usage": {
    "billed_units": {
      "input_tokens": 10,
      "output_tokens": 150
    },
    "tokens": {
      "input_tokens": 1400,
      "output_tokens": 300
    }
  },
  "timestamp": "2026-01-21T10:30:00"
}
```

**With Thinking Enabled:**
```json
{
  "message": "What is 2+2?",
  "enable_thinking": true,
  "frequency_penalty": 0.8
}
```

Response includes the `thinking` field with the model's reasoning process:
```json
{
  "thinking": "The user is asking about basic arithmetic. Let me calculate...",
  "response": "2 + 2 = 4"
}
```

**Performance Notes:**
- **Without thinking**: ~2-3 seconds
- **With thinking**: ~12-15 seconds

---

### 3. Cohere Stream: `POST /api/cohere/stream`

Stream responses from Cohere in real-time using Server-Sent Events (SSE).

**Query Parameters:**
- `message` (required): The message to send
- `enable_thinking` (optional, default: false): Enable thinking mode
- `frequency_penalty` (optional, default: 0.8): Frequency penalty (0.0-2.0)

**Example URL:**
```
POST /api/cohere/stream?message=Hello&enable_thinking=false&frequency_penalty=0.8
```

**Event Types:**

1. **Status Event:**
```json
{
  "type": "status",
  "message": "Connecting to Cohere...",
  "timestamp": "2026-01-21T10:30:00"
}
```

2. **Chunk Event:**
```json
{
  "type": "chunk",
  "content": "partial response data...",
  "timestamp": "2026-01-21T10:30:01"
}
```

3. **Complete Event:**
```json
{
  "type": "complete",
  "message": "Response complete",
  "timestamp": "2026-01-21T10:30:05"
}
```

4. **Error Event:**
```json
{
  "type": "error",
  "message": "Error description",
  "timestamp": "2026-01-21T10:30:01"
}
```

---

## Usage Examples

### Python Client - Regular Chat

```python
import requests

response = requests.post(
    'http://localhost:8000/api/cohere/chat',
    json={
        'message': 'What is machine learning?',
        'enable_thinking': False,
        'frequency_penalty': 0.8
    }
)

data = response.json()
print(data['response'])
```

### Python Client - Streaming

```python
import requests
import json

response = requests.post(
    'http://localhost:8000/api/cohere/stream',
    params={
        'message': 'Tell me a story',
        'enable_thinking': False
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        event = json.loads(line[6:])
        
        if event['type'] == 'chunk':
            print(event['content'], end='', flush=True)
        elif event['type'] == 'complete':
            print('\n--- Complete ---')
        elif event['type'] == 'error':
            print(f"\nError: {event['message']}")
```

### JavaScript Client - Streaming

```javascript
const eventSource = new EventSource(
    '/api/cohere/stream?message=Hello&enable_thinking=false'
);

eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'status':
            console.log(`Status: ${data.message}`);
            break;
        case 'chunk':
            console.log(data.content);
            break;
        case 'complete':
            console.log('Response complete!');
            eventSource.close();
            break;
        case 'error':
            console.error(`Error: ${data.message}`);
            eventSource.close();
            break;
    }
};

eventSource.onerror = (error) => {
    console.error('Connection error:', error);
    eventSource.close();
};
```

### cURL Examples

**Regular Chat:**
```bash
curl -X POST "http://localhost:8000/api/cohere/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is AI?",
    "enable_thinking": false,
    "frequency_penalty": 0.8
  }'
```

**Streaming:**
```bash
curl -X POST "http://localhost:8000/api/cohere/stream?message=Hello&enable_thinking=false" \
  --no-buffer
```

---

## Running the API

1. **Start the server:**
```bash
cd oslo_hackathon_api
uvicorn api:app --reload
```

2. **Access the documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

3. **View available endpoints:**
```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "Oslo AI Hackathon API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health",
  "endpoints": {
    "chat": "/api/chat",
    "cohere_chat": "/api/cohere/chat",
    "cohere_stream": "/api/cohere/stream"
  }
}
```

---

## Performance Comparison

| Endpoint | Thinking Mode | Avg Response Time | Use Case |
|----------|---------------|-------------------|----------|
| `/api/chat` | N/A | ~1-2s | Quick GPT responses |
| `/api/cohere/chat` | Disabled | ~2-3s | Fast Cohere responses |
| `/api/cohere/chat` | Enabled | ~12-15s | Detailed reasoning needed |
| `/api/cohere/stream` | Disabled | Starts in ~1s | Real-time streaming |
| `/api/cohere/stream` | Enabled | Starts in ~1s | Streaming with reasoning |

---

## Notes

- Token caching is implemented for Cohere authentication (tokens valid for ~60 minutes)
- Subsequent requests within the token validity period are faster (no auth overhead)
- Thinking mode provides detailed insight into the model's reasoning but significantly increases response time
- Streaming is recommended for better user experience with long responses
