"""Agent controller for LLM orchestration with tool calling."""

import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncIterator

from .openrouter import OpenRouterClient, format_message, format_tool_message
from .tools import (
    TOOLS,
    ToolCall,
    format_tool_result
)
from .agent_browser import AgentBrowserWrapper, AgentBrowserError
from .logger import SessionLogger


class AgentController:
    """Controller for the agent loop with LLM and tool execution."""

    def __init__(
        self,
        client: OpenRouterClient,
        browser: AgentBrowserWrapper,
        logger: SessionLogger,
        max_iterations: int = 1000,
        debug: bool = False
    ):
        """Initialize agent controller.

        Args:
            client: OpenRouter client
            browser: agent-browser wrapper
            logger: Session logger
            max_iterations: Maximum tool iterations per task
            debug: Enable debug mode
        """
        self.client = client
        self.browser = browser
        self.logger = logger
        self.max_iterations = max_iterations
        self.debug = debug

        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        self.system_prompt = self._build_system_prompt()

        # Loop detection: track recent tool call signatures
        self.recent_actions: List[str] = []

        # Token tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.query_prompt_tokens = 0
        self.query_completion_tokens = 0

        # Cached pricing (will be fetched from API if not in static dict)
        self._cached_pricing: Optional[Dict[str, float]] = None

        # Sliding window: limit conversation history size
        self.MAX_HISTORY_MESSAGES = 12  # Reduced from 15
        self.MAX_SUMMARIZED_HISTORY = 6  # Reduced from 8
        
        # Aggressive summarization for large outputs
        self.SNAPSHOT_RESULT_TOKEN_LIMIT = 1000  # Reduced from 2000

    def _summarize_snapshot_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize a snapshot result to reduce tokens while preserving key info.
        
        Args:
            result: Result dict from browser_snapshot tool
            
        Returns:
            Minimized result dict
        """
        output = result.get("output", "")
        
        if not isinstance(output, dict):
            # Not a dict, return basic summary
            return {
                "ok": result.get("ok", False),
                "output": f"[Snapshot summarized - {len(str(output))} characters]"
            }
        
        refs = output.get("refs", {})
        snapshot = output.get("snapshot", "")
        
        # Simplified approach: just keep summary, not the refs dict
        # This dramatically reduces token usage
        minimized = {
            "ok": result.get("ok", False),
            "output": f"[Snapshot summarized: {len(refs)} interactive elements, {len(snapshot)} chars of page content]"
        }
        
        return minimized

    def _summarize_tool_result(self, tool_name: str, result: Dict[str, Any]) -> str:
        """Summarize a tool result for conversation history.
        
        Args:
            tool_name: Name of the tool that was called
            result: Result dict from tool execution
            
        Returns:
            Brief summary string
        """
        if not result.get("ok"):
            return f"{tool_name} failed"
        
        output = result.get("output", "")
        
        # For different tool types, create appropriate summaries
        if tool_name == "browser_open":
            return f"Opened URL: {output[:100] if output else 'unknown'}"
        elif tool_name == "browser_snapshot":
            # Minimal summary - just key metrics
            if isinstance(output, dict):
                refs = output.get("refs", {})
                return f"Snapshot: {len(refs)} interactive elements"
            return "Snapshot taken"
        elif tool_name == "browser_screenshot":
            return "Captured screenshot"
        elif tool_name == "browser_click":
            return f"Clicked element"
        elif tool_name == "browser_fill":
            return f"Filled input field"
        elif tool_name == "browser_get_text":
            return f"Retrieved text: {output[:100]}..."
        elif tool_name == "browser_scroll":
            return f"Scrolled page"
        elif tool_name == "browser_back":
            return "Navigated back"
        elif tool_name == "browser_forward":
            return "Navigated forward"
        else:
            # Generic summary for other tools
            return f"{tool_name} executed"

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that can browse the web and perform actions.

You have access to browser automation tools through agent-browser. Use these tools to:

1. Navigate to websites and explore pages
2. Interact with elements (click buttons, fill forms, select dropdowns, check boxes, hover)
3. Extract information from pages
4. Complete user tasks by browsing and taking actions

## Available Browser Tools

### Navigation
- `browser_open` - Navigate to a URL
- `browser_back` - Go back in history
- `browser_forward` - Go forward in history
- `browser_reload` - Reload current page

### Page Analysis
- `browser_snapshot` - Get page snapshot with refs (supports `interactive`, `compact`, `depth` parameters)
  * Use `interactive=True` to see only interactive elements (buttons, inputs, links)
  * Use `compact=True` for large pages to reduce token usage (omits verbose attributes)
  * Use `depth=N` to limit tree depth and reduce output size (e.g., `depth=2` for shallow trees)

### Interactions
- `browser_click` - Click element by ref
- `browser_fill` - Fill input with text (clears field first)
- `browser_type` - Type text without clearing (appends to existing text)
- `browser_hover` - Hover over element (reveals tooltips/dropdowns)
- `browser_select` - Select dropdown option by value
- `browser_check` - Check checkbox
- `browser_uncheck` - Uncheck checkbox
- `browser_press` - Press keyboard key (Enter, Escape, Tab, Control+a, etc.)
- `browser_scroll` - Scroll page or element into view (direction + amount, optional ref)

### Information
- `browser_get_text` - Get text content from element
- `browser_get_value` - Get input value from element
- `browser_get_url` - Get current page URL
- `browser_get_title` - Get current page title

### Timing
- `browser_wait` - Wait for conditions:
  * `ref` - Wait for element to be ready
  * `milliseconds` - Wait specific time
  * `text` - Wait for text to appear
  * `networkidle` - Wait until network is idle
  * `url` - Wait until URL matches (supports glob patterns)

### Debugging & State
- `browser_screenshot` - Take screenshot (optional path, full_page flag)
- `browser_state_save` - Save browser state to file (for auth persistence)
- `browser_state_load` - Load browser state from file (restore session)
- `browser_close` - Close browser session

## How to Use Refs Efficiently

The snapshot output shows elements with refs like:
- # - button "Submit" [ref=e1]
- # - textbox "Email" [ref=e2]

Use refs from snapshots to interact:
- Click: `browser_click(ref="e1")`
- Fill: `browser_fill(ref="e2", text="value")`
- Hover: `browser_hover(ref="e1")` - reveals dropdowns/tooltips
- Select: `browser_select(ref="e2", value="option")` - for dropdowns
- Check: `browser_check(ref="e3")` - for checkboxes
- Uncheck: `browser_uncheck(ref="e3")` - for checkboxes

## Best Practices for Efficiency

### Token Reduction
1. **Use compact snapshots** - For large pages, use `browser_snapshot(compact=True, interactive=True)` to reduce token usage by 30-40%
2. **Limit depth** - Use `browser_snapshot(depth=2)` on complex pages to cut output by 20-30%
3. **Prefer get_url** - Use `browser_get_url()` to verify navigation instead of re-snapshotting
4. **Interactive-only default** - Use `browser_snapshot(interactive=True)` by default to ignore non-interactive elements
5. **Smart waiting** - Use `browser_wait(networkidle=True)` after navigation instead of manual delays

### Content Type Limitations
**IMPORTANT:** Browser snapshots cannot extract text content from:
- PDF files (.pdf)
- Binary downloads (images, executables, etc.)
- Some document viewers (depends on browser support)

When you encounter such content:
1. The snapshot will show visual elements but no readable text
2. Look for HTML versions or alternative accessible content
3. Don't assume page is empty - it's just not readable by snapshots
4. If a PDF is linked, mention that text extraction is not possible via browser tools

### Workflow Patterns
- Don't snapshot if page hasn't changed (use wait instead)
- Verify navigation with get_url before snapshotting
- Use hover to reveal hidden elements before clicking
- Use scroll to bring off-screen elements into view
- Use compact mode for large or complex pages
- Use state save/load for repeated authentication flows

### Communication
- Be concise and direct in your responses
- Summarize findings for the user efficiently
- If a page doesn't load or an action fails, try alternative approaches
- Stop when you have completed the user's task or cannot proceed further

IMPORTANT - TWO REQUIREMENTS:
1. ALWAYS provide a text response after tool calls. Never complete with only tools. After each sequence of tool calls, you MUST explain in text what you found or what you accomplished.
2. Your text response is required even if tools fail or if you couldn't find exactly what the user wanted. Explain what you attempted.

When you have enough information to answer the user, provide a clear, helpful response without making additional tool calls.

EXAMPLE OF WRONG BEHAVIOR (DO NOT DO THIS):
- Execute browser_open, snapshot, click, snapshot, fill, click...
- Stop without any text response
- User sees no answer

EXAMPLE OF CORRECT BEHAVIOR:
- Execute browser_open, snapshot, click, snapshot...
- Then write: I visited NASA website and found that the sky is blue because of Rayleigh scattering. Here is what I discovered...
"""

    def reset(self) -> None:
        """Reset conversation history and close browser."""
        self.messages = []
        self.recent_actions = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # Note: We don't close the browser here as it may be reused
        # Browser cleanup is handled by cleanup() method

    async def process_task(
        self,
        user_task: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Process a user task through the agent loop.

        Args:
            user_task: User's task description

        Yields:
            Streaming response chunks with type and content
        """
        # Log and add user message to conversation history
        self.logger.log_message("user", user_task)
        self.messages.append(format_message("user", user_task))
        
        # Reset per-query token counters
        self.query_prompt_tokens = 0
        self.query_completion_tokens = 0

        # Build messages for API call
        api_messages = [
            format_message("system", self.system_prompt),
            *self.messages
        ]

        iteration = 0
        accumulated_tool_calls: List[ToolCall] = []
        tool_call_args_buffers: List[str] = []
        assistant_accumulated_content = ""
        assistant_tool_calls: List[Dict[str, Any]] = []

        while iteration < self.max_iterations:
            iteration += 1

            # Signal start of new message
            yield {"type": "message_start"}

            # Reset state for new iteration
            assistant_accumulated_content = ""
            assistant_tool_calls = []
            accumulated_tool_calls = []
            tool_call_args_buffers = []

            if self.debug:
                yield {
                    "type": "debug",
                    "content": f"Iteration {iteration}/{self.max_iterations}"
                }

            try:
                # Stream response from LLM
                async for chunk in self.client.stream_chat_completion(
                    messages=api_messages,
                    tools=TOOLS
                ):
                    delta = chunk.get("choices", [{}])[0].get("delta", {})

                    # Track token usage from chunk
                    usage = chunk.get("usage")
                    if usage:
                        self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                        self.total_completion_tokens += usage.get("completion_tokens", 0)

                    # Handle content
                    content = delta.get("content", "")
                    if content:
                        assistant_accumulated_content += content
                        yield {
                            "type": "content",
                            "content": content
                        }

                    # Handle tool calls
                    tool_calls = delta.get("tool_calls", [])
                    if tool_calls:
                        for tool_delta in tool_calls:
                            function = tool_delta.get("function", {})
                            call_index = tool_delta.get("index", 0)

                            # New tool call
                            if function.get("name"):
                                # Note: Multiple tool calls in single response are accumulated
                                # and executed after streaming completes to maintain message order

                                # Reset args buffer for new tool call
                                tool_call_args_buffers.append("")

                                current_tool_call = ToolCall(
                                    name=function["name"],
                                    args={},
                                    call_id=tool_delta.get("id")
                                )
                                accumulated_tool_calls.append(current_tool_call)
                                # Track this tool call for assistant message
                                assistant_tool_calls.append({
                                    "id": tool_delta.get("id"),
                                    "type": "function",
                                    "function": {
                                        "name": function["name"],
                                        "arguments": ""
                                    }
                                })

                            # Accumulate arguments (may be streamed)
                            if function.get("arguments") and len(accumulated_tool_calls) > 0:
                                args_str = function["arguments"]
                                call_index = tool_delta.get("index", len(accumulated_tool_calls) - 1)
                                if call_index < len(tool_call_args_buffers):
                                    tool_call_args_buffers[call_index] += args_str

                                # Update the corresponding tool call's arguments
                                if call_index < len(assistant_tool_calls):
                                    assistant_tool_calls[call_index]["function"]["arguments"] = tool_call_args_buffers[call_index]

                                # Try to parse arguments
                                try:
                                    accumulated_tool_calls[call_index].args = json.loads(tool_call_args_buffers[call_index])
                                except json.JSONDecodeError:
                                    # Arguments incomplete, wait for more chunks
                                    pass

            except Exception as e:
                error_msg = f"Error communicating with LLM: {e}"
                self.logger.log_error(error_msg)
                yield {
                    "type": "error",
                    "content": error_msg
                }
                break

            # Append assistant message to conversation history FIRST
            # (must come before tool results per OpenAI API format)
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_accumulated_content
            }
            if assistant_tool_calls:
                assistant_message["tool_calls"] = assistant_tool_calls
            self.messages.append(assistant_message)
            api_messages.append(assistant_message)

            # Log complete assistant message after streaming is done
            self.logger.log_message("assistant", assistant_accumulated_content)

            # Signal end of current message
            yield {
                "type": "message_end",
                "is_final": not assistant_tool_calls  # Final if no more tool calls
            }

            # Execute all accumulated tool calls (AFTER assistant message)
            if accumulated_tool_calls:
                for tool_call in accumulated_tool_calls:
                    # Validate that tool arguments are complete (valid JSON)
                    if not tool_call.args:
                        yield {
                            "type": "error",
                            "content": f"Tool call '{tool_call.name}' had incomplete or invalid arguments"
                        }
                        continue

                    # Track action for loop detection
                    # Only track exact signature for repetition detection
                    action_sig = f"{tool_call.name}:{json.dumps(tool_call.args, sort_keys=True)}"
                    self.recent_actions.append(action_sig)
                    if len(self.recent_actions) > 10:
                        self.recent_actions.pop(0)

                    # DISABLED: Pattern-based loop detection (too aggressive)
                    # Normal navigation sequences (snapshot -> click -> snapshot -> click) 
                    # were being flagged as loops. Only detect exact repetition.
                    exact_loop = self.recent_actions.count(action_sig) >= 5
                    
                    if exact_loop:
                        yield {
                            "type": "loop_detected",
                            "action": tool_call.name,
                            "args": tool_call.args
                        }

                    async for result in self._execute_tool_call(
                        tool_call,
                        api_messages
                    ):
                        yield result

            # Check if we should continue (more tool calls expected)
            if not assistant_tool_calls:
                # No tool calls in this iteration, we're done
                break

            # Reset for next iteration
            accumulated_tool_calls = []
            tool_call_args_buffers = []

        if iteration >= self.max_iterations:
            warning = f"Reached maximum iterations ({self.max_iterations})"
            self.logger.log_error(warning)
            yield {
                "type": "warning",
                "content": warning
            }

        # Yield usage and cost summary
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        query_tokens = self.query_prompt_tokens + self.query_completion_tokens

        # Get pricing from OpenRouter API
        if self._cached_pricing is None:
            self._cached_pricing = await self.client.get_model_pricing(self.client.model)
        pricing = self._cached_pricing or {"input": 0, "output": 0}

        cost = (self.total_prompt_tokens * pricing["input"] + self.total_completion_tokens * pricing["output"]) / 1_000_000
        query_cost = (self.query_prompt_tokens * pricing["input"] + self.query_completion_tokens * pricing["output"]) / 1_000_000

        yield {
            "type": "usage",
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost,
            "query_prompt_tokens": self.query_prompt_tokens,
            "query_completion_tokens": self.query_completion_tokens,
            "query_total_tokens": query_tokens,
            "query_cost_usd": query_cost
        }

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        api_messages: List[Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """Execute a tool call and feed result back.

        Args:
            tool_call: Tool call to execute
            api_messages: API messages list to update

        Yields:
            Streaming chunks about tool execution
        """
        # Log tool call
        self.logger.log_tool_call(tool_call.name, tool_call.args)

        # Always emit tool call info for user visibility
        yield {
            "type": "tool_call",
            "tool": tool_call.name,
            "args": tool_call.args
        }

        if self.debug:
            yield {
                "type": "debug",
                "content": f"Tool call: {tool_call}"
            }

        # Execute tool
        result = await self._run_tool(tool_call)

        # Log result
        self.logger.log_tool_result(
            tool_call.name,
            result["ok"],
            result["output"],
            **{k: v for k, v in result.items() if k not in ("ok", "output")}
        )

        if self.debug:
            yield {
                "type": "debug",
                "content": f"Tool result: ok={result['ok']}"
            }

        # Feed result back to API messages - with summarization for snapshots
        if tool_call.name == "browser_screenshot" and result.get("ok"):
            # Create a minimized version for conversation history
            minimized_result = {
                "ok": result["ok"],
                "output": "[Screenshot captured - image not included in conversation history]"
            }
            tool_message = format_tool_message(
                tool_call.call_id or "unknown",
                json.dumps(minimized_result)
            )
        elif tool_call.name == "browser_snapshot" and result.get("ok"):
            # ALWAYS summarize snapshots - send summarized version to API
            # Raw snapshots go to log but minimized version goes to API
            minimized_result = self._summarize_snapshot_result(result)
            tool_message = format_tool_message(
                tool_call.call_id or "unknown",
                json.dumps(minimized_result)
            )
        else:
            tool_message = format_tool_message(
                tool_call.call_id or "unknown",
                json.dumps(result)
            )
        api_messages.append(tool_message)
        
        # Implement sliding window: summarize old tool results
        if len(self.messages) > self.MAX_HISTORY_MESSAGES:
            # We need to reduce history size. Strategy:
            # 1. Keep user/assistant messages (they contain important context)
            # 2. Summarize old tool results to reduce their size
            # 3. Only remove very old messages if we're still over limit after summarizing
            
            messages_to_summarize = len(self.messages) - self.MAX_SUMMARIZED_HISTORY
            
            for i in range(messages_to_summarize):
                msg = self.messages[i]
                if msg.get("role") == "tool":
                    try:
                        tool_content = json.loads(msg.get("content", "{}"))
                        tool_name = tool_content.get("tool", "unknown")
                        summary = self._summarize_tool_result(tool_name, tool_content)
                        # Replace full tool result with brief summary
                        msg["content"] = json.dumps({
                            "tool": tool_name,
                            "ok": tool_content.get("ok", True),
                            "output": summary
                        })
                    except (json.JSONDecodeError, KeyError):
                        msg["content"] = "[Previous tool result summarized]"
            
            # If still over limit after summarizing, remove oldest messages
            # (but keep at least MAX_SUMMARIZED_HISTORY)
            while len(self.messages) > self.MAX_SUMMARIZED_HISTORY:
                self.messages.pop(0)

    async def _run_tool(self, tool_call: ToolCall) -> Dict[str, Any]:
        """Run a tool call.

        Args:
            tool_call: Tool call to execute

        Returns:
            Tool result dict
        """
        try:
            # Extract method name from tool name (e.g., "browser_open" -> "open")
            method_name = tool_call.name.replace("browser_", "")
            method = getattr(self.browser, method_name, None)
            if not method:
                return {
                    "ok": False,
                    "output": f"Unknown tool: {tool_call.name}"
                }

            # Execute tool
            result = await method(**tool_call.args)
            return result

        except AgentBrowserError as e:
            return {
                "ok": False,
                "output": f"Browser error: {e}"
            }
        except Exception as e:
            return {
                "ok": False,
                "output": f"Tool error: {e}"
            }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.browser.cleanup()
        except Exception:
            pass
