"""Interactive CLI interface using rich and prompt_toolkit."""

import asyncio
import os
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.completion import WordCompleter, Completer

from .openrouter import OpenRouterClient, OpenRouterError, DEFAULT_MODEL
from .agent_browser import AgentBrowserWrapper
from .controller import AgentController
from .logger import SessionLogger
from .platform_check import (
    check_shell_compatibility,
    check_agent_browser_installed,
    create_runs_directory,
    get_agent_browser_install_instructions
)


class ConditionalCompleter(Completer):
    """Completer that only shows completions when input is empty or starts with /."""
    
    def __init__(self, completer: Completer):
        """Initialize wrapper completer.
        
        Args:
            completer: The underlying completer to wrap
        """
        self.completer = completer
    
    def get_completions(self, document, complete_event):
        """Get completions only when input is empty or starts with /.
        
        Args:
            document: The document to complete
            complete_event: The complete event
            
        Yields:
            Completions from the wrapped completer
        """
        text = document.text_before_cursor.strip()
        # Only show completions when input is empty or starts with /
        if not text or text.startswith("/"):
            yield from self.completer.get_completions(document, complete_event)


class CLI:
    """Interactive CLI for browser agent."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        debug: bool = False
    ):
        """Initialize CLI.
        
        Args:
            api_key: OpenRouter API key
            model: Model to use
            debug: Enable debug mode
        """
        self.console = Console()
        self.debug = debug
        self._should_exit = False
        self._reset_requested = False
        
        # Setup prompt session with history
        self.prompt_session = PromptSession(history=InMemoryHistory())
        
        # Command completer
        commands = ["/help", "/exit", "/reset", "/debug"]
        self.completer = ConditionalCompleter(WordCompleter(commands))
        
        # Initialize components
        self._client: Optional[OpenRouterClient] = None
        self._controller: Optional[AgentController] = None
        self._logger: Optional[SessionLogger] = None
        
        # Load environment
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL")
    
    def print_welcome(self) -> None:
        """Print welcome message."""
        self.console.print("\n[bold cyan]Browser Agent CLI[/bold cyan]\n", justify="center")
        self.console.print("LLM-powered browser automation\n", justify="center")
        self.console.print("Type /help for commands, /exit to quit\n")
    
    def print_help(self) -> None:
        """Print help message."""
        help_text = f"""
[bold]Available Commands:[/bold]

  [cyan]/help[/cyan]       - Show this help message
  [cyan]/exit[/cyan]       - Exit the CLI
  [cyan]/reset[/cyan]      - Clear conversation history
  [cyan]/debug on[/cyan]   - Enable debug mode
  [cyan]/debug off[/cyan]  - Disable debug mode

[bold]Usage:[/bold]

  Simply type a task and press Enter. For example:
  - "Find the pricing page and summarize the tiers"
  - "Search for Python tutorials and list the top 3"
  - "Open example.com and tell me what's on the page"

[bold]Environment Variables:[/bold]

  [cyan]OPENROUTER_API_KEY[/cyan]  - Your OpenRouter API key (required)
  [cyan]OPENROUTER_MODEL[/cyan]    - Model to use (default: {DEFAULT_MODEL})
"""
        self.console.print(help_text)
    
    def print_error(self, message: str) -> None:
        """Print an error message.
        
        Args:
            message: Error message
        """
        self.console.print(f"[red]Error: {message}[/red]")
    
    def print_success(self, message: str) -> None:
        """Print a success message.
        
        Args:
            message: Success message
        """
        self.console.print(f"[green]{message}[/green]")
    
    def print_info(self, message: str) -> None:
        """Print an info message.
        
        Args:
            message: Info message
        """
        self.console.print(f"[blue]{message}[/blue]")
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met.
        
        Returns:
            True if all checks pass, False otherwise
        """
        # Check shell compatibility
        shell_error = check_shell_compatibility()
        if shell_error:
            self.print_error(shell_error)
            return False
        
        # Check API key
        if not self.api_key:
            self.print_error(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY environment variable or create a .env file."
            )
            return False
        
        # Check agent-browser
        installed, version_or_error = check_agent_browser_installed()
        if not installed:
            self.print_error(f"agent-browser not installed: {version_or_error}")
            self.console.print("\n" + get_agent_browser_install_instructions())
            return False
        
        self.print_success(f"agent-browser {version_or_error} detected")
        return True
    
    async def process_user_input(self, user_input: str) -> None:
        """Process user input.
        
        Args:
            user_input: User's input
        """
        # Handle commands
        if user_input.startswith("/"):
            await self._handle_command(user_input)
            return
        
        # Handle reset request
        if self._reset_requested:
            self._reset_requested = False
            self._controller.reset()

            # Create new logger and browser wrapper
            runs_dir = create_runs_directory()
            self._logger = SessionLogger(runs_dir)
            self._logger.open_browser_log()

            # Update controller with new browser wrapper
            from .agent_browser import AgentBrowserWrapper
            new_browser = AgentBrowserWrapper(logger=self._logger)
            self._controller.browser = new_browser
            self._controller.logger = self._logger
        
        # Ensure session is initialized
        if self._logger is None:
            runs_dir = create_runs_directory()
            self._logger = SessionLogger(runs_dir)
            self._logger.open_browser_log()
        
        # Process task
        await self._process_task(user_input)
    
    async def _handle_command(self, command: str) -> None:
        """Handle a slash command.
        
        Args:
            command: Command string
        """
        parts = command.split()
        cmd = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        if cmd == "/help":
            self.print_help()
        
        elif cmd == "/exit":
            self._should_exit = True
        
        elif cmd == "/reset":
            self._reset_requested = True
            self.print_info("Conversation will be reset on next message")
        
        elif cmd == "/debug":
            if args and args[0].lower() in ("on", "true", "1"):
                self.debug = True
                if self._controller:
                    self._controller.debug = True
                self.print_info("Debug mode enabled")
            elif args and args[0].lower() in ("off", "false", "0"):
                self.debug = False
                if self._controller:
                    self._controller.debug = False
                self.print_info("Debug mode disabled")
            else:
                status = "on" if self.debug else "off"
                self.print_info(f"Debug mode is {status}")
        
        else:
            self.print_error(f"Unknown command: {cmd}")
    
    async def _process_task(self, task: str) -> None:
        """Process a task through the controller.

        Args:
            task: User's task
        """
        # Ensure components are initialized
        if self._client is None:
            self._client = OpenRouterClient(api_key=self.api_key, model=self.model)

        if self._controller is None:
            self._controller = AgentController(
                client=self._client,
                browser=AgentBrowserWrapper(logger=self._logger),
                logger=self._logger,
                debug=self.debug
            )

        # Display user input
        self.console.print("\n[bold]You:[/bold] ", end="")
        self.console.print(task, style="white")

        # Variables for handling multiple message panels
        full_response = ""
        tool_log = ""
        is_final = False

        try:
            # Stream response
            async for chunk in self._controller.process_task(task):
                chunk_type = chunk.get("type")
                content = chunk.get("content", "")

                if chunk_type == "message_start":
                    # Start new message - reset accumulators
                    full_response = ""
                    tool_log = ""
                    is_final = False

                elif chunk_type == "content":
                    # Streaming text from assistant
                    full_response += content

                elif chunk_type == "debug" and self.debug:
                    # Debug info - includes tool calls when debug mode is on
                    tool_log += f"[DEBUG] {content}\n"

                elif chunk_type == "tool_call":
                    # Display tool execution progress
                    tool_name = chunk.get("tool", "unknown")
                    args = chunk.get("args", {})
                    self.console.print(f"[dim]→ {tool_name}({args})[/dim]")

                elif chunk_type == "warning":
                    # Warning
                    self.console.print(f"\n[yellow]Warning: {content}[/yellow]")

                elif chunk_type == "error":
                    # Error
                    self.console.print(f"\n[red]Error: {content}[/red]")

                elif chunk_type == "loop_detected":
                    # Loop detected - prompt user to continue or stop
                    action = chunk.get("action")
                    args = chunk.get("args", {})
                    self.console.print(f"\n[yellow]Loop detected: {action}({args})[/yellow]")
                    self.console.print("[yellow]Agent is repeating actions. Press Enter to continue or Ctrl+C to stop.[/yellow]")
                    try:
                        await self.prompt_session.prompt_async("")
                    except KeyboardInterrupt:
                        self.console.print("\n[yellow]Stopping execution.[/yellow]")
                        break

                elif chunk_type == "message_end":
                    # End of current message - display panel
                    is_final = chunk.get("is_final", False)
                    if is_final:
                        # Only show final panel if there's actual content
                        if full_response:
                            self.console.print(self._create_final_display(full_response))
                        else:
                            # Warn if final message has no content (model didn't respond)
                            self.console.print("\n[yellow]⚠ Model completed without providing a text response[/yellow]")
                    elif full_response:
                        self.console.print(self._create_display(full_response, tool_log))
                    elif tool_log:
                        # Only show tool logs if debug mode is on
                        if self.debug:
                            self.console.print(self._create_display("", tool_log))

                elif chunk_type == "usage":
                    # Token usage and cost
                    prompt = chunk.get("prompt_tokens", 0)
                    completion = chunk.get("completion_tokens", 0)
                    total = chunk.get("total_tokens", 0)
                    cost = chunk.get("cost_usd", 0)
                    self.console.print(
                        f"[dim]Tokens: {prompt:,} prompt + {completion:,} completion = {total:,} total | Cost: ${cost:.4f}[/dim]"
                    )

        except KeyboardInterrupt:
            self.console.print("\n[yellow]Run interrupted by user.[/yellow]")
        finally:
            # Show session path
            session_path = self._logger.get_session_path()
            self.console.print(f"\n[dim]Session saved to: {session_path}[/dim]")

    def _create_display(self, response: str, tool_log: str) -> Panel:
        """Create a display panel with response and tool logs.

        Args:
            response: Assistant's response
            tool_log: Tool execution logs

        Returns:
            Panel with formatted content
        """
        if tool_log:
            # Combine markdown response with text tool logs
            content = Text.assemble(
                response,
                "\n\n",
                Text("--- Tool Logs ---\n", style="dim"),
                Text(tool_log, style="dim")
            )
        else:
            # Use Markdown for response only
            content = Markdown(response)

        return Panel(
            content,
            title="Assistant",
            title_align="left",
            border_style="cyan"
        )

    def _create_final_display(self, response: str) -> Panel:
        """Create a final answer display panel with highlighted styling.

        Args:
            response: Assistant's final response

        Returns:
            Panel with formatted content
        """
        return Panel(
            Markdown(response),
            title="[bold]Final Answer[/bold]",
            title_align="left",
            border_style="green"
        )

    async def run(self) -> None:
        """Run the interactive CLI."""
        self.print_welcome()

        # Check prerequisites
        if not self.check_prerequisites():
            return

        # Main loop
        while not self._should_exit:
            try:
                user_input = await self.prompt_session.prompt_async(
                    "> ",
                    completer=self.completer
                )

                if user_input.strip():
                    await self.process_user_input(user_input.strip())

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
            except EOFError:
                break

        # Cleanup
        await self._cleanup()
        self.console.print("\n[green]Goodbye![/green]")

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        if self._logger:
            self._logger.close_browser_log()
        if self._controller:
            await self._controller.cleanup()
        if self._client:
            await self._client.close()


def run_cli(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    debug: bool = False
) -> None:
    """Run the CLI (entry point).

    Args:
        api_key: OpenRouter API key
        model: Model to use
        debug: Enable debug mode
    """
    cli = CLI(api_key=api_key, model=model, debug=debug)
    asyncio.run(cli.run())
