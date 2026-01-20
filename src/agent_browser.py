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
    
    async def snapshot(self, interactive: bool = False) -> Dict[str, Any]:
        """Get page snapshot.

        Args:
            interactive: If true, show only interactive elements

        Returns:
            Result with ok, output, and refs
        """
        cmd = ["agent-browser", "snapshot"]
        if interactive:
            cmd.append("-i")
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
        cmd = ["agent-browser", "get", "text", f"@{ref}"]
        return await self._run_command(cmd)
    
    async def close(self) -> Dict[str, Any]:
        """Close browser session.
        
        Returns:
            Result with ok and output
        """
        cmd = ["agent-browser", "close"]
        return await self._run_command(cmd)
    
    async def _run_command(
        self,
        cmd: List[str]
    ) -> Dict[str, Any]:
        """Run an agent-browser command.
        
        Args:
            cmd: Command and arguments
            
        Returns:
            Result with ok, output, and optionally refs
        """
        process = None
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.timeout
            )
            
            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")
            
            # Log output
            if self.logger:
                if stdout_text:
                    for line in stdout_text.split("\n"):
                        if line:
                            self.logger.write_browser_log(line)
                if stderr_text:
                    for line in stderr_text.split("\n"):
                        if line:
                            self.logger.write_browser_log(f"[ERROR] {line}")
            
            # Parse output for refs if present
            refs = self._parse_refs(stdout_text)
            
            # Truncate output if needed
            output = truncate_output(stdout_text + stderr_text)
            
            return {
                "ok": process.returncode == 0,
                "output": output,
                "refs": refs if refs else None,
                "returncode": process.returncode
            }
            
        except asyncio.TimeoutError:
            if process:
                process.kill()
                await process.wait()
            raise AgentBrowserError(
                f"Command timed out after {self.timeout}s: {' '.join(cmd)}"
            )
        except Exception as e:
            raise AgentBrowserError(f"Command failed: {e}")
    
    def _parse_refs(self, output: str) -> Optional[Dict[str, Any]]:
        """Parse element references from snapshot output.
        
        Args:
            output: Command output
            
        Returns:
            Dict mapping ref IDs to element info, or None if not found
        """
        # Look for JSON output from snapshot -i --json
        if "refs" in output:
            try:
                # Try to extract JSON from the output
                json_match = re.search(r'\{[^{}]*"refs"[^{}]*\}', output)
                if json_match:
                    data = json.loads(json_match.group())
                    return data.get("refs")
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Look for ref patterns in snapshot output
        # Format: # - element [ref=e1]
        ref_pattern = r'\[ref=([^\]]+)\]'
        refs = {}
        for match in re.finditer(ref_pattern, output):
            ref_id = match.group(1)
            # Extract element type and name from context
            line_start = output.rfind("\n", 0, match.start()) + 1
            line_end = output.find("\n", match.start())
            if line_end == -1:
                line_end = len(output)
            line = output[line_start:line_end].strip()
            
            # Parse element description
            # Example: "# - button 'Submit' [ref=e1]"
            element_match = re.match(r'# - (\w+) (.+?) \[ref=', line)
            if element_match:
                refs[ref_id] = {
                    "type": element_match.group(1),
                    "name": element_match.group(2)
                }
        
        return refs if refs else None
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            await self.close()
        except Exception:
            pass
