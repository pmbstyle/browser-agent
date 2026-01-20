"""Tool definitions and schemas for browser automation."""

import json
from typing import Any, Dict, List, Optional


# Tool definitions for OpenAI-compatible API
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_open",
            "description": "Navigate to a URL and get the initial page snapshot",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to navigate to"
                    }
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_snapshot",
            "description": "Get the current page snapshot with interactive elements and refs. Use compact=True for large pages to reduce tokens, and depth=N to limit tree depth",
            "parameters": {
                "type": "object",
                "properties": {
                    "interactive": {
                        "type": "boolean",
                        "description": "If true, show only interactive elements",
                        "default": False
                    },
                    "compact": {
                        "type": "boolean",
                        "description": "Use compact output (omits verbose attributes) - recommended for large pages to reduce tokens",
                        "default": False
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Limit tree depth (e.g., 2, 3) - reduces output size significantly",
                        "default": None
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_click",
            "description": "Click an element by its reference (e.g., 'e1', 'e2')",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_fill",
            "description": "Fill a text input with content",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to fill into the input"
                    }
                },
                "required": ["ref", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_text",
            "description": "Get text content from an element",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_value",
            "description": "Get input value from an element",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_url",
            "description": "Get current page URL",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_get_title",
            "description": "Get current page title",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_hover",
            "description": "Hover over an element to reveal tooltips/dropdowns",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_scroll",
            "description": "Scroll the page or element into view",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "Scroll direction (up, down, left, right)",
                        "default": "down"
                    },
                    "amount": {
                        "type": "integer",
                        "description": "Pixels to scroll",
                        "default": 500
                    },
                    "ref": {
                        "type": "string",
                        "description": "Optional element reference - if provided, scrolls element into view instead of page"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_select",
            "description": "Select a dropdown option",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to select"
                    }
                },
                "required": ["ref", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_press",
            "description": "Press keyboard key (Enter, Escape, Tab, Control+a, etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "Key to press (Enter, Escape, Tab, Control+a, etc.)"
                    }
                },
                "required": ["key"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_type",
            "description": "Type text without clearing the input field (unlike fill which clears first)",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to type"
                    }
                },
                "required": ["ref", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_check",
            "description": "Check a checkbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_uncheck",
            "description": "Uncheck a checkbox",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Element reference from the snapshot (e.g., 'e1')"
                    }
                },
                "required": ["ref"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_wait",
            "description": "Wait for a condition (element, time, text, network idle, or URL)",
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "Optional element reference to wait for"
                    },
                    "milliseconds": {
                        "type": "integer",
                        "description": "Wait for specific milliseconds"
                    },
                    "text": {
                        "type": "string",
                        "description": "Wait for text to appear"
                    },
                    "networkidle": {
                        "type": "boolean",
                        "description": "Wait until network is idle"
                    },
                    "url": {
                        "type": "string",
                        "description": "Wait until URL matches (supports glob patterns)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_back",
            "description": "Go back in browser history",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_forward",
            "description": "Go forward in browser history",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_reload",
            "description": "Reload the current page",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_screenshot",
            "description": "Take screenshot for debugging (optional: save to file, capture full page)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Optional file path to save screenshot"
                    },
                    "full_page": {
                        "type": "boolean",
                        "description": "Capture full page instead of viewport",
                        "default": False
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_state_save",
            "description": "Save browser state to file for session persistence",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to save state"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_state_load",
            "description": "Load browser state from file to restore session",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to load state from"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_close",
            "description": "Close the browser session",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


class ToolCall:
    """Represents a tool call from the LLM."""
    
    def __init__(self, name: str, args: Dict[str, Any], call_id: Optional[str] = None):
        """Initialize tool call.
        
        Args:
            name: Tool name
            args: Tool arguments
            call_id: Tool call ID from the API
        """
        self.name = name
        self.args = args
        self.call_id = call_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Tool call dictionary
        """
        result = {
            "tool": self.name,
            "args": self.args
        }
        if self.call_id:
            result["id"] = self.call_id
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"ToolCall({self.name}, {args_str})"


def parse_tool_calls(delta: Dict[str, Any]) -> Optional[ToolCall]:
    """Parse tool calls from a streaming chunk.
    
    Args:
        delta: Delta from streaming response
        
    Returns:
        ToolCall if present, None otherwise
    """
    tool_calls = delta.get("tool_calls")
    if not tool_calls:
        return None
    
    # Get the first tool call (usually only one)
    call = tool_calls[0]
    function = call.get("function", {})
    
    name = function.get("name")
    args_str = function.get("arguments", "{}")
    
    if not name:
        return None
    
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError:
        # Arguments might be streamed in chunks
        return None
    
    return ToolCall(name, args, call.get("id"))


def format_tool_result(
    tool: str,
    ok: bool,
    output: str,
    **kwargs: Any
) -> Dict[str, Any]:
    """Format a tool result for returning to the LLM.
    
    Args:
        tool: Tool name
        ok: Whether the tool call succeeded
        output: Tool output
        **kwargs: Additional metadata
        
    Returns:
        Formatted tool result
    """
    result = {
        "tool_result": {
            "tool": tool,
            "ok": ok,
            "output": output
        }
    }
    if kwargs:
        result["tool_result"].update(kwargs)
    return result


def truncate_output(output: str, max_size: int = 50000) -> str:
    """Truncate output if too large.
    
    Args:
        output: Output string
        max_size: Maximum size in characters
        
    Returns:
        Truncated output with indicator if needed
    """
    if len(output) <= max_size:
        return output
    
    return (
        output[:max_size]
        + f"\n\n[Output truncated: {len(output)} total characters]"
    )
