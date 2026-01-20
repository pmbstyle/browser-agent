"""OpenRouter streaming client for LLM API."""

import json
import os
from typing import AsyncGenerator, Dict, List, Optional, Any
from functools import lru_cache

import httpx


DEFAULT_MODEL = "anthropic/claude-sonnet-4"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"


class OpenRouterError(Exception):
    """Error from OpenRouter API."""
    pass


class OpenRouterClient:
    """Client for OpenRouter Chat Completions API with streaming."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0
    ):
        """Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY env var
            model: Model to use. If None, reads from OPENROUTER_MODEL env var or uses default
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    @lru_cache(maxsize=1)
    async def get_model_pricing(self, model_name: str) -> Dict[str, float]:
        """Get pricing for a specific model from OpenRouter API.

        Args:
            model_name: Model identifier (e.g., "anthropic/claude-sonnet-4")

        Returns:
            Dict with 'input' and 'output' pricing per 1M tokens
            Defaults to {'input': 0, 'output': 0} if model not found
        """
        try:
            # Fetch models list
            async with self.client.stream("GET", MODELS_URL) as response:
                if response.status_code == 200:
                    data = await response.aread()
                    models_data = json.loads(data.decode())

                    # Find the model in the list
                    models = models_data.get("data", [])
                    for model in models:
                        if model.get("id") == model_name:
                            pricing = model.get("pricing", {})
                            return {
                                "input": float(pricing.get("prompt", 0)),
                                "output": float(pricing.get("completion", 0))
                            }
        except Exception:
            pass

        # Return default pricing if not found
        return {"input": 0.0, "output": 0.0}
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion from OpenRouter.
        
        Args:
            messages: Chat messages in OpenAI format
            tools: Tool definitions for the model to call
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Yields:
            Chunks containing 'delta' with content or tool_calls
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/browser-agent/browser-agent",
            "X-Title": "Browser Agent CLI",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if tools:
            payload["tools"] = tools
        
        try:
            async with self.client.stream(
                "POST",
                API_URL,
                headers=headers,
                json=payload
            ) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise OpenRouterError(
                        f"OpenRouter API error {response.status_code}: {error_text.decode()}"
                    )
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break

                        try:
                            chunk = self._parse_sse_chunk(data)
                            if chunk:
                                yield chunk
                        except Exception:
                            # Silently skip malformed chunks
                            continue
        except httpx.TimeoutException:
            raise OpenRouterError(f"Request timed out after {self.timeout}s")
        except httpx.HTTPError as e:
            raise OpenRouterError(f"HTTP error: {e}")
    
    def _parse_sse_chunk(self, data: str) -> Optional[Dict[str, Any]]:
        """Parse an SSE chunk from OpenRouter.
        
        Args:
            data: Raw SSE data string
            
        Returns:
            Parsed chunk dict or None if not a valid chunk
        """
        try:
            chunk = json.loads(data)
            return chunk
        except json.JSONDecodeError:
            return None
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def format_message(role: str, content: str) -> Dict[str, str]:
    """Format a message for the API.
    
    Args:
        role: Message role (system, user, assistant, tool)
        content: Message content
        
    Returns:
        Formatted message dict
    """
    return {"role": role, "content": content}


def format_tool_message(
    tool_call_id: str,
    content: str
) -> Dict[str, str]:
    """Format a tool result message.
    
    Args:
        tool_call_id: ID of the tool call
        content: Tool result content
        
    Returns:
        Formatted tool message dict
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": content
    }
