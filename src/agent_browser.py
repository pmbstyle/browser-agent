"""Subprocess wrapper for agent-browser CLI."""

import asyncio
import json
import re
from typing import Optional, Dict, Any, List

from .logger import SessionLogger
from .tools import truncate_output


class AgentBrowserError(Exception):
    """Error from agent-browser subprocess."""
    pass


class AgentBrowserWrapper:
    """Wrapper for executing agent-browser commands via subprocess."""
    
    def __init__(
        self,
        logger: Optional[SessionLogger] = None,
        timeout: float = 180.0
    ):
        """Initialize wrapper.
        
        Args:
            logger: Session logger for capturing output
            timeout: Default timeout in seconds for commands
        """
        self.logger = logger
        self.timeout = timeout
        self._browser_session = None
    
    async def open(self, url: str) -> Dict[str, Any]:
        """Open a URL and get initial snapshot.
        
        Args:
            url: URL to open
            
        Returns:
            Result with ok, output, and refs
        """
        cmd = ["agent-browser", "open", url]
        return await self._run_command(cmd)
    
    async def snapshot(self, interactive: bool = False, compact: bool = False, depth: Optional[int] = None) -> Dict[str, Any]:
        """Get page snapshot.

        Args:
            interactive: If true, show only interactive elements
            compact: If true, use compact output (omits verbose attributes)
            depth: Limit tree depth (e.g., 2, 3)

        Returns:
            Result with ok, output, and refs
        """
        cmd = ["agent-browser", "snapshot", "--json"]
        if interactive:
            cmd.append("-i")
        if compact:
            cmd.append("-c")
        if depth is not None:
            cmd.extend(["-d", str(depth)])
        return await self._run_command(cmd)
    
    async def click(self, ref: str) -> Dict[str, Any]:
        """Click an element by ref.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "click", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def fill(self, ref: str, text: str) -> Dict[str, Any]:
        """Fill an input with text.
        
        Args:
            ref: Element reference (e.g., 'e1')
            text: Text to fill
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "fill", f"@{ref}", text]
        return await self._run_command(cmd)
    
    async def get_text(self, ref: str) -> Dict[str, Any]:
        """Get text from an element.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "get", "text", f"@{ref}", "--json"]
        return await self._run_command(cmd)
    
    async def get_value(self, ref: str) -> Dict[str, Any]:
        """Get input value from an element.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "get", "value", f"@{ref}", "--json"]
        return await self._run_command(cmd)
    
    async def get_url(self) -> Dict[str, Any]:
        """Get current page URL.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "get", "url", "--json"]
        return await self._run_command(cmd)
    
    async def get_title(self) -> Dict[str, Any]:
        """Get current page title.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "get", "title", "--json"]
        return await self._run_command(cmd)
    
    async def hover(self, ref: str) -> Dict[str, Any]:
        """Hover over an element.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "hover", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def scroll(self, direction: str = "down", amount: int = 500) -> Dict[str, Any]:
        """Scroll the page.
        
        Args:
            direction: Scroll direction (up, down, left, right)
            amount: Pixels to scroll
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "scroll", direction, str(amount)]
        return await self._run_command(cmd)
    
    async def scroll_into_view(self, ref: str) -> Dict[str, Any]:
        """Scroll element into view.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "scrollintoview", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def select(self, ref: str, value: str) -> Dict[str, Any]:
        """Select dropdown option.
        
        Args:
            ref: Element reference (e.g., 'e1')
            value: Value to select
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "select", f"@{ref}", value]
        return await self._run_command(cmd)
    
    async def press(self, key: str) -> Dict[str, Any]:
        """Press keyboard key.
        
        Args:
            key: Key to press (Enter, Escape, Tab, Control+a, etc.)
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "press", key]
        return await self._run_command(cmd)
    
    async def type(self, ref: str, text: str) -> Dict[str, Any]:
        """Type text without clearing input.
        
        Args:
            ref: Element reference (e.g., 'e1')
            text: Text to type
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "type", f"@{ref}", text]
        return await self._run_command(cmd)
    
    async def check(self, ref: str) -> Dict[str, Any]:
        """Check checkbox.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "check", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def uncheck(self, ref: str) -> Dict[str, Any]:
        """Uncheck checkbox.
        
        Args:
            ref: Element reference (e.g., 'e1')
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "uncheck", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def wait(self, ref: Optional[str] = None, milliseconds: Optional[int] = None, text: Optional[str] = None, networkidle: bool = False, url: Optional[str] = None) -> Dict[str, Any]:
        """Wait for condition.
        
        Args:
            ref: Wait for element by ref
            milliseconds: Wait specific time
            text: Wait for text to appear
            networkidle: Wait for network to be idle
            url: Wait for URL to match (supports glob patterns)
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "wait", "--json"]
        if ref:
            cmd.append(f"@{ref}")
        elif milliseconds:
            cmd.append(str(milliseconds))
        elif text:
            cmd.extend(["--text", text])
        elif networkidle:
            cmd.extend(["--load", "networkidle"])
        elif url:
            cmd.extend(["--url", url])
        return await self._run_command(cmd)
    
    async def back(self) -> Dict[str, Any]:
        """Go back in history.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "back"]
        return await self._run_command(cmd)
    
    async def forward(self) -> Dict[str, Any]:
        """Go forward in history.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "forward"]
        return await self._run_command(cmd)
    
    async def reload(self) -> Dict[str, Any]:
        """Reload current page.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "reload"]
        return await self._run_command(cmd)
    
    async def screenshot(self, path: Optional[str] = None, full_page: bool = False) -> Dict[str, Any]:
        """Take screenshot.
        
        Args:
            path: Save to file path
            full_page: Capture full page
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "screenshot"]
        if path:
            cmd.append(path)
        if full_page:
            cmd.append("--full")
        return await self._run_command(cmd)
    
    async def state_save(self, path: str) -> Dict[str, Any]:
        """Save browser state to file.
        
        Args:
            path: File path to save state
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "state", "save", path]
        return await self._run_command(cmd)
    
    async def state_load(self, path: str) -> Dict[str, Any]:
        """Load browser state from file.
        
        Args:
            path: File path to load state from
            
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "state", "load", path]
        return await self._run_command(cmd)
    
    async def close(self) -> Dict[str, Any]:
        """Close browser session.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "close"]
        return await self._run_command(cmd)
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.close()
        except Exception:
            pass
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """Execute an agent-browser command.
        
        Args:
            cmd: Command and arguments list
            
        Returns:
            Result dict with ok, output, and optional refs
        """
        if self.logger:
            self.logger.log_browser_command(" ".join(cmd))
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            stdout_str = stdout.decode("utf-8", errors="replace")
            stderr_str = stderr.decode("utf-8", errors="replace")
            
            if self.logger:
                self.logger.log_browser_output(stdout_str)
                if stderr_str:
                    self.logger.log_browser_output(f"[stderr] {stderr_str}")
            
            # Check for error exit code
            if process.returncode != 0:
                error_output = stderr_str or stdout_str or f"Command failed with exit code {process.returncode}"
                return {
                    "ok": False,
                    "output": truncate_output(error_output)
                }
            
            # Try to parse JSON output
            output = stdout_str.strip()
            try:
                parsed = json.loads(output)
                # If it's a dict with expected structure, use it
                if isinstance(parsed, dict):
                    return {
                        "ok": True,
                        "output": truncate_output(parsed.get("output", output)),
                        **{k: v for k, v in parsed.items() if k != "output"}
                    }
            except json.JSONDecodeError:
                pass
            
            # Return raw output if not JSON
            return {
                "ok": True,
                "output": truncate_output(output)
            }
            
        except asyncio.TimeoutError:
            return {
                "ok": False,
                "output": f"Command timed out after {self.timeout}s"
            }
        except FileNotFoundError:
            return {
                "ok": False,
                "output": "agent-browser CLI not found. Install with: npm install -g @anthropic/agent-browser"
            }
        except Exception as e:
            return {
                "ok": False,
                "output": f"Command execution failed: {e}"
            }