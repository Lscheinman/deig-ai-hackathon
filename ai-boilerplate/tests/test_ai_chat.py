"""
Unit tests for ai_chat module.
All external API calls are mocked — no credentials or network required.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta


# We need to mock gen_ai_hub before importing ai_chat, since _get_chat() imports it
import sys

mock_chat = MagicMock()
mock_openai = MagicMock()
mock_openai.chat = mock_chat
mock_native = MagicMock()
mock_native.openai = mock_openai
mock_proxy = MagicMock()
mock_proxy.native = mock_native
mock_gen_ai_hub = MagicMock()
mock_gen_ai_hub.proxy = mock_proxy

sys.modules['gen_ai_hub'] = mock_gen_ai_hub
sys.modules['gen_ai_hub.proxy'] = mock_proxy
sys.modules['gen_ai_hub.proxy.native'] = mock_native
sys.modules['gen_ai_hub.proxy.native.openai'] = mock_openai

import ai_chat


@pytest.fixture(autouse=True)
def reset_token_cache():
    """Reset the token cache before each test."""
    ai_chat._token_cache['access_token'] = None
    ai_chat._token_cache['token_expiry'] = None
    yield


# ─── Token Tests ────────────────────────────────────────────────────────────────


@patch("ai_chat.requests.post")
def test_get_or_refresh_token_success(mock_post):
    """Token request succeeds and returns access token."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "test-token-123",
        "expires_in": 3600
    }
    mock_post.return_value = mock_response

    token = ai_chat.get_or_refresh_token()

    assert token == "test-token-123"
    mock_post.assert_called_once()


@patch("ai_chat.requests.post")
def test_get_or_refresh_token_uses_cache(mock_post):
    """Second call uses cached token without making another HTTP request."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "access_token": "cached-token",
        "expires_in": 3600
    }
    mock_post.return_value = mock_response

    # First call fetches token
    token1 = ai_chat.get_or_refresh_token()
    # Second call should use cache
    token2 = ai_chat.get_or_refresh_token()

    assert token1 == "cached-token"
    assert token2 == "cached-token"
    # Only one HTTP call made
    assert mock_post.call_count == 1


@patch("ai_chat.requests.post")
def test_get_or_refresh_token_failure(mock_post):
    """Token request fails with non-200 status raises exception."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="Token request failed: 401"):
        ai_chat.get_or_refresh_token()


# ─── OpenAI Chat Tests ──────────────────────────────────────────────────────────


def test_chat_with_openai():
    """chat_with_openai returns the model's response text."""
    mock_response = MagicMock()
    mock_response.to_dict.return_value = {
        "choices": [{"message": {"content": "Hello from GPT!"}}]
    }
    mock_chat_client = MagicMock()
    mock_chat_client.completions.create.return_value = mock_response

    with patch("ai_chat._get_chat", return_value=mock_chat_client):
        result = ai_chat.chat_with_openai("Hi there")

    assert result == "Hello from GPT!"
    mock_chat_client.completions.create.assert_called_once_with(
        model_name="gpt-5",
        messages=[{"role": "user", "content": "Hi there"}]
    )


def test_stream_chat_with_openai():
    """stream_chat_with_openai yields content chunks."""
    chunk1 = MagicMock()
    chunk1.to_dict.return_value = {"choices": [{"delta": {"content": "Hello"}}]}
    chunk2 = MagicMock()
    chunk2.to_dict.return_value = {"choices": [{"delta": {"content": " world"}}]}
    chunk3 = MagicMock()
    chunk3.to_dict.return_value = {"choices": [{"delta": {}}]}

    mock_chat_client = MagicMock()
    mock_chat_client.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    with patch("ai_chat._get_chat", return_value=mock_chat_client):
        chunks = list(ai_chat.stream_chat_with_openai("Hi", model="gpt-4o"))

    assert chunks == ["Hello", " world"]
    mock_chat_client.completions.create.assert_called_once_with(
        model_name="gpt-4o",
        messages=[{"role": "user", "content": "Hi"}],
        stream=True
    )


# ─── Cohere Chat Tests ──────────────────────────────────────────────────────────


@patch("ai_chat.get_or_refresh_token")
@patch("ai_chat.requests.post")
def test_chat_with_cohere_success(mock_post, mock_token):
    """chat_with_cohere returns structured response dict on success."""
    mock_token.return_value = "cohere-token"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": [
                {"type": "text", "text": "Cohere says hi!"}
            ]
        },
        "finish_reason": "COMPLETE",
        "usage": {"input_tokens": 5, "output_tokens": 10}
    }
    mock_post.return_value = mock_response

    result = ai_chat.chat_with_cohere("Hello Cohere")

    assert result["response"] == "Cohere says hi!"
    assert result["thinking"] is None
    assert result["finish_reason"] == "COMPLETE"
    assert result["usage"] == {"input_tokens": 5, "output_tokens": 10}


@patch("ai_chat.get_or_refresh_token")
@patch("ai_chat.requests.post")
def test_chat_with_cohere_with_thinking(mock_post, mock_token):
    """chat_with_cohere returns thinking content when enabled."""
    mock_token.return_value = "cohere-token"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "message": {
            "content": [
                {"type": "thinking", "thinking": "Let me think about this..."},
                {"type": "text", "text": "Here is my answer."}
            ]
        },
        "finish_reason": "COMPLETE",
        "usage": {"input_tokens": 5, "output_tokens": 20}
    }
    mock_post.return_value = mock_response

    result = ai_chat.chat_with_cohere("Deep question", enable_thinking=True)

    assert result["response"] == "Here is my answer."
    assert result["thinking"] == "Let me think about this..."


@patch("ai_chat.get_or_refresh_token")
@patch("ai_chat.requests.post")
def test_chat_with_cohere_error(mock_post, mock_token):
    """chat_with_cohere raises exception on API error."""
    mock_token.return_value = "cohere-token"
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_post.return_value = mock_response

    with pytest.raises(Exception, match="Cohere API error: 500"):
        ai_chat.chat_with_cohere("This will fail")
