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

mapping = {
    "AICORE_CLIENT_ID": os.getenv("AICORE_CLIENT_ID"),
    "AICORE_CLIENT_SECRET": os.getenv("AICORE_CLIENT_SECRET"),
    "AICORE_AUTH_URL": os.getenv("AICORE_AUTH_URL"),
    "AICORE_BASE_URL": os.getenv("AICORE_BASE_URL"),
    "AICORE_RESOURCE_GROUP": os.getenv("AICORE_RESOURCE_GROUP"),
}
for k, v in mapping.items():
    os.environ[k] = str(v)

from gen_ai_hub.proxy.native.openai import OpenAI

openai_client = OpenAI(
    base_url=os.environ["AICORE_BASE_URL"]
    )

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
    try:
        response = openai_client.chat.completions.create(
            model=model,  # use 'model' key
            messages=[{"role": "user", "content": message}]
        )
        return response.to_dict()["choices"][0]["message"]["content"]
    except Exception as e:
        raise Exception(f"Error communicating with AI model: {str(e)}")


def stream_chat_with_openai(message: str, model: str = "gpt-5"):
    try:
        stream = openai_client.chat.completions.create(
            model=model,  # use 'model' key
            messages=[{"role": "user", "content": message}],
            stream=True
        )
        for chunk in stream:
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
