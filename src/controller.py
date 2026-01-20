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

        # Cached pricing (will be fetched from API if not in static dict)
        self._cached_pricing: Optional[Dict[str, float]] = None

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM.

        Returns:
            System prompt string
        """
        return """You are a helpful AI assistant that can browse the web and perform actions.

You have access to browser automation tools through agent-browser. Use these tools to:

1. Navigate to websites and explore pages
2. Interact with elements (click buttons, fill forms)
3. Extract information from pages
4. Complete user tasks by browsing and taking actions

How to use browser tools:
- Always start with `browser_open` to navigate to a URL
- Use `browser_snapshot` after page changes to see updated elements
- Use element refs from snapshots (e.g., "e1", "e2") to click or fill elements
- Use `browser_get_text` to extract specific content
- End with `browser_close` when done

The snapshot output shows elements with refs like:
- # - button "Submit" [ref=e1]
- # - textbox "Email" [ref=e2]

To click: Use `browser_click` with ref="e1"
To fill: Use `browser_fill` with ref="e2" and text="value"

Best practices:
- Be concise and direct in your responses
- Use tools efficiently (don't repeatedly snapshot if page hasn't changed)
- Summarize findings for the user
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
                    action_sig = f"{tool_call.name}:{json.dumps(tool_call.args, sort_keys=True)}"
                    self.recent_actions.append(action_sig)

                    # Keep only last 10 actions for loop detection
                    if len(self.recent_actions) > 10:
                        self.recent_actions.pop(0)

                    # Check for loop: same action 3+ times in recent actions
                    if self.recent_actions.count(action_sig) >= 3:
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

        # Get pricing from OpenRouter API
        if self._cached_pricing is None:
            self._cached_pricing = await self.client.get_model_pricing(self.client.model)
        pricing = self._cached_pricing or {"input": 0, "output": 0}

        cost = (self.total_prompt_tokens * pricing["input"] + self.total_completion_tokens * pricing["output"]) / 1_000_000

        yield {
            "type": "usage",
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost
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

        # Feed result back to API messages
        tool_message = format_tool_message(
            tool_call.call_id or "unknown",
            json.dumps(result)
        )
        api_messages.append(tool_message)
        self.messages.append(tool_message)

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
