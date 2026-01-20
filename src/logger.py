"""Session logging for browser-agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class SessionLogger:
    """Logger for session transcripts and agent-browser output."""
    
    def __init__(self, runs_dir: Optional[Path] = None):
        """Initialize session logger.
        
        Args:
            runs_dir: Directory for session logs. Defaults to "./runs/"
        """
        if runs_dir is None:
            runs_dir = Path("runs")
        
        self.session_dir = runs_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        self.session_file = self.session_dir / "session.jsonl"
        self.browser_log_file = self.session_dir / "agent-browser.log"
        
        self._browser_log_handle = None
        self._browser_log_path = None
    
    def log_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Log a message to the session file.
        
        Args:
            role: Message role (user, assistant, system, tool, tool_result)
            content: Message content
            **kwargs: Additional metadata
        """
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            **kwargs
        }
        
        with open(self.session_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    
    def log_tool_call(self, tool: str, args: dict) -> None:
        """Log a tool call.
        
        Args:
            tool: Tool name
            args: Tool arguments
        """
        self.log_message(
            role="tool",
            content=f"Calling tool: {tool}",
            tool=tool,
            args=args
        )
    
    def log_tool_result(
        self,
        tool: str,
        ok: bool,
        output: str,
        **kwargs: Any
    ) -> None:
        """Log a tool result.
        
        Args:
            tool: Tool name
            ok: Whether the tool call succeeded
            output: Tool output
            **kwargs: Additional metadata (e.g., refs, artifacts)
        """
        self.log_message(
            role="tool_result",
            content=output,
            tool=tool,
            ok=ok,
            **kwargs
        )
    
    def log_error(self, error: str, **kwargs: Any) -> None:
        """Log an error.
        
        Args:
            error: Error message
            **kwargs: Additional metadata
        """
        self.log_message(
            role="error",
            content=error,
            **kwargs
        )
    
    def open_browser_log(self) -> Path:
        """Open browser log file for writing.
        
        Returns:
            Path to browser log file
        """
        self._browser_log_path = self.browser_log_file
        self._browser_log_handle = open(self.browser_log_file, "a", encoding="utf-8")
        return self._browser_log_path
    
    def write_browser_log(self, line: str) -> None:
        """Write a line to the browser log.
        
        Args:
            line: Line to write
        """
        if self._browser_log_handle and not self._browser_log_handle.closed:
            self._browser_log_handle.write(line + "\n")
            self._browser_log_handle.flush()
    
    def log_browser_command(self, command: str) -> None:
        """Log a browser command to the browser log.
        
        Args:
            command: Command string being executed
        """
        timestamp = datetime.now().isoformat()
        self.write_browser_log(f"[{timestamp}] CMD: {command}")
    
    def log_browser_output(self, output: str) -> None:
        """Log browser output to the browser log.
        
        Args:
            output: Output from browser command
        """
        timestamp = datetime.now().isoformat()
        # Truncate very long output in log
        if len(output) > 5000:
            output = output[:5000] + f"\n... [truncated {len(output) - 5000} chars]"
        self.write_browser_log(f"[{timestamp}] OUT: {output}")
    
    def close_browser_log(self) -> None:
        """Close browser log file."""
        if self._browser_log_handle and not self._browser_log_handle.closed:
            self._browser_log_handle.close()
            self._browser_log_handle = None
    
    def get_session_path(self) -> Path:
        """Get the session directory path.
        
        Returns:
            Path to session directory
        """
        return self.session_dir
    
    def __del__(self):
        """Clean up on deletion."""
        self.close_browser_log()
