# commons/ai_core/llm_client.py
"""
LLM client abstraction for JONE agent.
Provides a simple async interface for text generation.
"""
from __future__ import annotations

import asyncio
import os
from typing import Optional

from loguru import logger

# Try to import the GenAI Hub wrapper
try:
    from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
    from gen_ai_hub.proxy.native.openai import chat
    _GENAI_AVAILABLE = True
except Exception as exc:
    logger.warning(
        "[init] gen_ai_hub not available; falling back to heuristic client. Error: %r",
        exc,
    )
    chat = None
    _GENAI_AVAILABLE = False

from commons.core.config import get_settings


def _ensure_genai_env():
    """Map settings to environment variables expected by gen_ai_hub SDK."""
    s = get_settings()
    mapping = {
        "AICORE_CLIENT_ID": s.AICORE_CLIENT_ID,
        "AICORE_CLIENT_SECRET": s.AICORE_CLIENT_SECRET,
        "AICORE_AUTH_URL": s.AICORE_AUTH_URL,
        "AICORE_BASE_URL": s.AICORE_BASE_URL,
        "AICORE_RESOURCE_GROUP": s.AICORE_RESOURCE_GROUP,
    }
    for k, v in mapping.items():
        if v and not os.getenv(k):
            os.environ[k] = str(v)


class LlmClient:
    """Abstract base class for LLM backends."""

    async def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> str:
        """Generate text response given system and user messages."""
        raise NotImplementedError("LlmClient.generate_text must be implemented")


class HeuristicLlmClient(LlmClient):
    """Offline heuristic fallback - returns fixed responses."""

    async def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> str:
        logger.warning("[LLM] Using heuristic fallback - no AI Core available")
        return '{"intent": "out_of_scope", "reply": "LLM backend unavailable", "confidence": 0.5}'


class AiCoreLlmClient(LlmClient):
    """LLM client using SAP AI Core / GenAI Hub."""

    def __init__(self):
        _ensure_genai_env()
        if _GENAI_AVAILABLE:
            try:
                get_proxy_client("gen-ai-hub")
            except Exception as e:
                logger.warning(f"[LLM] Could not init proxy client: {e}")

    async def generate_text(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 64,
    ) -> str:
        if not _GENAI_AVAILABLE or chat is None:
            raise RuntimeError("AI Core / GenAI Hub not available")

        messages = []
        if system:
            messages.append({"role": "system", "content": system.strip()})
        messages.append({"role": "user", "content": user.strip()})

        model = os.getenv("AICORE_CHAT_MODEL", "gpt-4o")

        def _call_backend():
            kwargs = {
                "model_name": model,
                "messages": messages,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            if max_tokens is not None:
                kwargs["max_completion_tokens"] = max_tokens

            resp = chat.completions.create(**kwargs)
            return resp.choices[0].message.content

        try:
            return await asyncio.to_thread(_call_backend)
        except Exception as exc:
            msg = str(exc)
            logger.error(f"[LLM] GenAI call failed: {msg}")

            # Retry without temperature if unsupported
            if "temperature" in msg.lower():
                def _retry():
                    resp = chat.completions.create(
                        model_name=model,
                        messages=messages,
                        max_completion_tokens=max_tokens,
                    )
                    return resp.choices[0].message.content
                try:
                    return await asyncio.to_thread(_retry)
                except Exception:
                    pass

            return '{"intent": "out_of_scope", "reply": "LLM backend error", "confidence": 0.0}'


# Singleton client
_default_client: Optional[LlmClient] = None


def get_llm_client() -> LlmClient:
    """Return the default LLM client."""
    global _default_client

    if _default_client is None:
        if _GENAI_AVAILABLE and chat is not None:
            logger.info("[LLM] Initializing AiCoreLlmClient (SAP AI Core)")
            _default_client = AiCoreLlmClient()
        else:
            logger.warning("[LLM] AI Core unavailable; using HeuristicLlmClient")
            _default_client = HeuristicLlmClient()

    return _default_client
