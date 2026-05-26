"""
AI Chat Agent Module
====================
Simple module for chatting with different AI models including OpenAI and Cohere.
"""

import os
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth

# Load environment variables
load_dotenv()

# AI Core Credentials — set only if present (avoids TypeError when .env is missing)
for _key in ("AICORE_CLIENT_ID", "AICORE_CLIENT_SECRET", "AICORE_AUTH_URL",
             "AICORE_BASE_URL", "AICORE_RESOURCE_GROUP"):
    _val = os.getenv(_key)
    if _val is not None:
        os.environ[_key] = _val


def _get_chat():
    """Lazy import of gen_ai_hub chat client (requires valid credentials)."""
    from gen_ai_hub.proxy.native.openai import chat
    return chat


# Global token cache for Cohere
_token_cache = {
    'access_token': None,
    'token_expiry': None
}


def get_or_refresh_token() -> Optional[str]:
    """
    Get existing token or request a new one if needed.
    
    Returns:
        Access token string or None if failed
    """
    global _token_cache
    
    # Check if token already exists and is still valid
    if _token_cache['access_token'] and _token_cache['token_expiry']:
        if datetime.now() < _token_cache['token_expiry']:
            return _token_cache['access_token']
    
    # Get environment variables for authentication
    AUTH_URL = os.getenv("AICORE_AUTH_URL")
    CLIENT_ID = os.getenv("AICORE_CLIENT_ID") 
    CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET")
    
    # Make the POST request to get OAuth token
    response = requests.post(
        f"{AUTH_URL}/oauth/token",
        headers={
            'content-type': 'application/x-www-form-urlencoded'
        },
        data={
            'grant_type': 'client_credentials',
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
    )
    
    # Check if request was successful
    if response.status_code == 200:
        token_data = response.json()
        new_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 3600)  # Default to 1 hour
        
        # Calculate expiry time with 1-minute buffer
        new_expiry = datetime.now() + timedelta(seconds=expires_in - 60)
        
        # Update cache
        _token_cache['access_token'] = new_token
        _token_cache['token_expiry'] = new_expiry
        
        return new_token
    else:
        raise Exception(f"Token request failed: {response.status_code} - {response.text}")


def chat_with_openai(message: str, model: str = "gpt-5") -> str:
    """
    Send a message to an AI model and get a response.
    
    Args:
        message: The user's message to send to the AI
        model: The AI model to use (e.g., 'gpt-5', 'gpt-4o', 'gpt-35-turbo')
    
    Returns:
        The AI model's response as a string
    """
    try:
        # Create a simple chat completion
        response = _get_chat().completions.create(
            model_name=model,
            messages=[
                {"role": "user", "content": message}
            ]
        )
        
        # Extract and return the response
        response_dict = response.to_dict()
        return response_dict["choices"][0]["message"]["content"]
    
    except Exception as e:
        raise Exception(f"Error communicating with AI model: {str(e)}")


def stream_chat_with_openai(message: str, model: str = "gpt-5"):
    """
    Stream a response from an AI model.
    
    Args:
        message: The user's message to send to the AI
        model: The AI model to use (e.g., 'gpt-5', 'gpt-4o', 'gpt-35-turbo')
    
    Yields:
        Text chunks as they arrive from the model
    """
    try:
        # Create a streaming chat completion
        response = _get_chat().completions.create(
            model_name=model,
            messages=[
                {"role": "user", "content": message}
            ],
            stream=True
        )
        
        # Stream the response chunks
        for chunk in response:
            chunk_dict = chunk.to_dict()
            choices = chunk_dict.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content
    
    except Exception as e:
        raise Exception(f"Error streaming from AI model: {str(e)}")


def chat_with_cohere(
    message: str, 
    model: str = "cohere--command-a-reasoning",
    enable_thinking: bool = False,
    frequency_penalty: float = 0.8
) -> Dict[str, Any]:
    """
    Send a message to Cohere model and get a response.
    
    Args:
        message: The user's message to send to the AI
        model: The Cohere model to use
        enable_thinking: Whether to enable thinking mode (slower but more detailed)
        frequency_penalty: Frequency penalty for response generation
    
    Returns:
        Dictionary with response text and metadata
    """
    try:
        # Get access token
        access_token = get_or_refresh_token()
        
        # Get environment variables
        BASE_URL = os.getenv("AICORE_BASE_URL")
        RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP")
        
        # Construct the deployment URL
        DEPLOYMENT_URL = f"{BASE_URL}/inference/deployments/d1cd5340a145e7a9"
        
        # Chat request payload
        chat_payload = {
            "model": model,
            "stream": False,
            "frequency_penalty": frequency_penalty,
            "thinking": {
                "type": "enabled" if enable_thinking else "disabled"
            },
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        
        # Make the chat request
        chat_response = requests.post(
            f"{DEPLOYMENT_URL}/chat",
            headers={
                'AI-Resource-Group': RESOURCE_GROUP,
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {access_token}'
            },
            json=chat_payload
        )
        
        # Check response
        if chat_response.status_code == 200:
            chat_result = chat_response.json()
            
            # Extract the text response
            response_text = None
            thinking_text = None
            
            for content in chat_result["message"]["content"]:
                if content.get("type") == "text":
                    response_text = content["text"]
                elif content.get("type") == "thinking":
                    thinking_text = content.get("thinking")
            
            return {
                "response": response_text,
                "thinking": thinking_text if enable_thinking else None,
                "finish_reason": chat_result.get("finish_reason"),
                "usage": chat_result.get("usage")
            }
        else:
            raise Exception(f"Cohere API error: {chat_response.status_code} - {chat_response.text}")
    
    except Exception as e:
        raise Exception(f"Error communicating with Cohere model: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with AI models via SAP AI Core")
    parser.add_argument("message", help="Message to send to the AI model")
    parser.add_argument(
        "--provider", choices=["openai", "cohere"], default="openai",
        help="AI provider to use (default: openai)"
    )
    parser.add_argument("--model", help="Model name (defaults: gpt-5 for openai, cohere--command-a-reasoning for cohere)")
    parser.add_argument("--stream", action="store_true", help="Stream response (OpenAI only)")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (Cohere only)")

    args = parser.parse_args()

    if args.provider == "openai":
        model = args.model or "gpt-5"
        if args.stream:
            for chunk in stream_chat_with_openai(args.message, model=model):
                print(chunk, end="", flush=True)
            print()
        else:
            result = chat_with_openai(args.message, model=model)
            print(result)
    elif args.provider == "cohere":
        model = args.model or "cohere--command-a-reasoning"
        result = chat_with_cohere(args.message, model=model, enable_thinking=args.thinking)
        if args.thinking and result.get("thinking"):
            print(f"[Thinking]\n{result['thinking']}\n")
        print(result["response"])