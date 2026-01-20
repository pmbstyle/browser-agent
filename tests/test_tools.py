"""Tests for tool parsing and validation."""

import pytest
from src.tools import (
    ToolCall,
    parse_tool_calls,
    format_tool_result,
    truncate_output
)


class TestToolCall:
    """Test ToolCall class."""
    
    def test_tool_call_creation(self):
        """Test creating a tool call."""
        call = ToolCall("browser.open", {"url": "https://example.com"}, "call_123")
        assert call.name == "browser.open"
        assert call.args == {"url": "https://example.com"}
        assert call.call_id == "call_123"
    
    def test_tool_call_to_dict(self):
        """Test converting tool call to dictionary."""
        call = ToolCall("browser.click", {"ref": "e1"}, "call_456")
        result = call.to_dict()
        assert result["tool"] == "browser.click"
        assert result["args"] == {"ref": "e1"}
        assert result["id"] == "call_456"
    
    def test_tool_call_without_id(self):
        """Test tool call without call ID."""
        call = ToolCall("browser.fill", {"ref": "e2", "text": "test"})
        result = call.to_dict()
        assert "id" not in result
    
    def test_tool_call_repr(self):
        """Test string representation."""
        call = ToolCall("browser.open", {"url": "https://test.com"})
        repr_str = repr(call)
        assert "browser.open" in repr_str
        assert "url=" in repr_str


class TestParseToolCalls:
    """Test tool call parsing."""
    
    def test_parse_tool_call_from_delta(self):
        """Test parsing tool call from delta."""
        delta = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "browser.open",
                        "arguments": '{"url": "https://example.com"}'
                    }
                }
            ]
        }
        result = parse_tool_calls(delta)
        assert result is not None
        assert result.name == "browser.open"
        assert result.args == {"url": "https://example.com"}
        assert result.call_id == "call_123"
    
    def test_parse_tool_call_no_name(self):
        """Test parsing delta without tool name."""
        delta = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "arguments": '{"url": "https://example.com"}'
                    }
                }
            ]
        }
        result = parse_tool_calls(delta)
        assert result is None
    
    def test_parse_tool_call_no_tool_calls(self):
        """Test parsing delta without tool calls."""
        delta = {"content": "Hello"}
        result = parse_tool_calls(delta)
        assert result is None
    
    def test_parse_tool_call_invalid_json(self):
        """Test parsing delta with invalid JSON arguments."""
        delta = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "browser.open",
                        "arguments": "invalid json"
                    }
                }
            ]
        }
        result = parse_tool_calls(delta)
        assert result is None


class TestFormatToolResult:
    """Test tool result formatting."""
    
    def test_format_tool_result_success(self):
        """Test formatting successful tool result."""
        result = format_tool_result(
            "browser.open",
            ok=True,
            output="Page loaded",
            refs={"e1": {"type": "button", "name": "Submit"}}
        )
        assert result["tool_result"]["tool"] == "browser.open"
        assert result["tool_result"]["ok"] is True
        assert result["tool_result"]["output"] == "Page loaded"
        assert result["tool_result"]["refs"] is not None
    
    def test_format_tool_result_failure(self):
        """Test formatting failed tool result."""
        result = format_tool_result(
            "browser.click",
            ok=False,
            output="Element not found"
        )
        assert result["tool_result"]["ok"] is False
        assert result["tool_result"]["output"] == "Element not found"


class TestTruncateOutput:
    """Test output truncation."""
    
    def test_no_truncation(self):
        """Test output below limit."""
        output = "a" * 1000
        result = truncate_output(output, max_size=50000)
        assert result == output
    
    def test_truncation(self):
        """Test output above limit."""
        output = "a" * 100000
        result = truncate_output(output, max_size=50000)
        assert len(result) < len(output)
        assert "truncated" in result.lower()
        assert "100000" in result
    
    def test_truncation_custom_size(self):
        """Test truncation with custom size."""
        output = "a" * 1000
        result = truncate_output(output, max_size=100)
        assert len(result) > 100
        assert "truncated" in result.lower()
