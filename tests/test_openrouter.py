"""Tests for OpenRouter client with mocking."""

import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    format_message,
    format_tool_message
)


@pytest.fixture
def mock_api_key():
    """Fixture for mock API key."""
    return "test-api-key-12345"


@pytest.fixture
def client(mock_api_key):
    """Fixture for OpenRouter client."""
    return OpenRouterClient(api_key=mock_api_key)


class TestOpenRouterClient:
    """Test OpenRouter client."""
    
    def test_client_requires_api_key(self):
        """Test that client requires API key."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                OpenRouterClient()
    
    def test_client_uses_env_api_key(self):
        """Test that client uses environment API key."""
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "env-key"}):
            client = OpenRouterClient()
            assert client.api_key == "env-key"
    
    def test_client_uses_provided_api_key(self, mock_api_key):
        """Test that client uses provided API key."""
        client = OpenRouterClient(api_key=mock_api_key)
        assert client.api_key == mock_api_key
    
    def test_client_default_model(self, mock_api_key):
        """Test default model selection."""
        client = OpenRouterClient(api_key=mock_api_key)
        assert client.model == "anthropic/claude-sonnet-4"
    
    def test_client_custom_model(self, mock_api_key):
        """Test custom model selection."""
        client = OpenRouterClient(api_key=mock_api_key, model="custom-model")
        assert client.model == "custom-model"
    
    def test_client_model_from_env(self, mock_api_key):
        """Test model from environment variable."""
        with patch.dict("os.environ", {"OPENROUTER_MODEL": "env-model"}):
            client = OpenRouterClient(api_key=mock_api_key)
            assert client.model == "env-model"
    
    @pytest.mark.asyncio
    async def test_stream_chat_completion(self, client):
        """Test streaming chat completion."""
        # Mock HTTP response context manager
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        # Create mock stream
        async def mock_iter_lines():
            chunks = [
                'data: {"id": "test", "choices": [{"delta": {"content": "Hello"}}]}',
                'data: {"id": "test", "choices": [{"delta": {"content": " world"}}]}',
                'data: [DONE]'
            ]
            for chunk in chunks:
                yield chunk
        
        mock_response.aiter_lines = mock_iter_lines
        mock_response.aread = AsyncMock(return_value=b"")
        
        # Mock the stream method to return an async context manager
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None
        
        with patch.object(client.client, "stream", return_value=mock_stream_context):
            chunks = []
            async for chunk in client.stream_chat_completion([
                format_message("user", "Hello")
            ]):
                chunks.append(chunk)
            
            assert len(chunks) == 2
    
    @pytest.mark.asyncio
    async def test_stream_chat_completion_with_tools(self, client):
        """Test streaming with tool definitions."""
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        async def mock_iter_lines():
            chunks = [
                'data: {"id": "test", "choices": [{"delta": {"content": "Thinking..."}}]}',
                'data: [DONE]'
            ]
            for chunk in chunks:
                yield chunk
        
        mock_response.aiter_lines = mock_iter_lines
        mock_response.aread = AsyncMock(return_value=b"")
        
        # Mock the stream context manager
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None
        
        tools = [
            {"type": "function", "function": {"name": "test", "parameters": {}}}
        ]
        
        with patch.object(client.client, "stream", return_value=mock_stream_context):
            chunks = []
            async for chunk in client.stream_chat_completion(
                [format_message("user", "test")],
                tools=tools
            ):
                chunks.append(chunk)
            
            assert len(chunks) == 1
    
    @pytest.mark.asyncio
    async def test_stream_error_handling(self, client):
        """Test error handling in stream."""
        mock_response = AsyncMock()
        mock_response.status_code = 401
        mock_response.aread = AsyncMock(return_value=b"Unauthorized")
        
        # Mock stream context manager
        mock_stream_context = AsyncMock()
        mock_stream_context.__aenter__.return_value = mock_response
        mock_stream_context.__aexit__.return_value = None
        
        with patch.object(client.client, "stream", return_value=mock_stream_context):
            with pytest.raises(OpenRouterError, match="401"):
                async for _ in client.stream_chat_completion([format_message("user", "test")]):
                    pass
    
    @pytest.mark.asyncio
    async def test_stream_timeout(self, client):
        """Test timeout handling."""
        from httpx import TimeoutException
        
        with patch.object(client.client, "stream", side_effect=TimeoutException("Timeout")):
            with pytest.raises(OpenRouterError, match="timed out"):
                async for _ in client.stream_chat_completion([format_message("user", "test")]):
                    pass
    
    @pytest.mark.asyncio
    async def test_close_client(self, client):
        """Test closing the client."""
        await client.close()
        # Should not raise


class TestFormatMessage:
    """Test message formatting."""
    
    def test_format_user_message(self):
        """Test formatting user message."""
        result = format_message("user", "Hello")
        assert result == {"role": "user", "content": "Hello"}
    
    def test_format_system_message(self):
        """Test formatting system message."""
        result = format_message("system", "You are helpful")
        assert result == {"role": "system", "content": "You are helpful"}
    
    def test_format_assistant_message(self):
        """Test formatting assistant message."""
        result = format_message("assistant", "I can help")
        assert result == {"role": "assistant", "content": "I can help"}


class TestFormatToolMessage:
    """Test tool message formatting."""
    
    def test_format_tool_message(self):
        """Test formatting tool result message."""
        result = format_tool_message("call_123", "Tool output here")
        assert result == {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Tool output here"
        }
