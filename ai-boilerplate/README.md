# ai_core_boilerplate

Simple Python module for chatting with AI models through SAP AI Core, supporting OpenAI and Cohere providers.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your SAP AI Core credentials:

```bash
cp .env.example .env
```

## Usage

### CLI

```bash
# OpenAI (default)
python ai_chat.py "Hello, world!"

# OpenAI with streaming
python ai_chat.py "Count to 5" --stream

# Specify a model
python ai_chat.py "Hello" --model gpt-4o

# Cohere
python ai_chat.py "Hello" --provider cohere

# Cohere with thinking mode
python ai_chat.py "Explain quantum computing" --provider cohere --thinking
```

### As a module

```python
from ai_chat import chat_with_openai, stream_chat_with_openai, chat_with_cohere

# OpenAI
response = chat_with_openai("Hello!", model="gpt-5")

# OpenAI streaming
for chunk in stream_chat_with_openai("Tell me a story"):
    print(chunk, end="")

# Cohere
result = chat_with_cohere("Explain AI", enable_thinking=True)
print(result["response"])
print(result["thinking"])
```

## Testing

```bash
pytest tests/ -v
```

All tests use mocked API calls — no credentials required.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AICORE_CLIENT_ID` | OAuth client ID |
| `AICORE_CLIENT_SECRET` | OAuth client secret |
| `AICORE_AUTH_URL` | Token endpoint base URL |
| `AICORE_BASE_URL` | AI Core API base URL |
| `AICORE_RESOURCE_GROUP` | Resource group name |

