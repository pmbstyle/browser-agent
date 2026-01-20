"""Platform and shell detection for browser-agent."""

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Optional


def get_platform_info() -> Tuple[str, bool]:
    """Get platform name and whether we're in WSL.

    Returns:
        Tuple of (platform_name, is_wsl)
    """
    system = platform.system().lower()

    if system == "linux":
        is_wsl = _is_wsl()
        return "wsl" if is_wsl else "linux", is_wsl
    elif system == "windows":
        return "windows", False
    elif system == "darwin":
        return "macos", False
    else:
        return "unknown", False


def _is_wsl() -> bool:
    """Check if running under WSL."""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except (FileNotFoundError, IOError):
        return False


def check_shell_compatibility() -> Optional[str]:
    """Check if current shell is compatible.
    
    Returns:
        Error message if incompatible, None otherwise
    """
    platform_name, _ = get_platform_info()
    
    if platform_name == "windows":
        shell = get_current_shell()
        if shell in ("powershell", "cmd"):
            return (
                "Windows PowerShell/CMD detected. "
                "Please use WSL or Git Bash for full compatibility.\n\n"
                "To use WSL:\n"
                "  1. Install WSL: wsl --install\n"
                "  2. Open WSL terminal\n"
                "  3. Install Node.js: curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && sudo apt install -y nodejs\n"
                "  4. Install agent-browser: npm install -g agent-browser\n"
                "  5. Run this CLI from WSL\n\n"
                "To use Git Bash:\n"
                "  1. Install Git for Windows (includes Git Bash)\n"
                "  2. Run this CLI from Git Bash"
            )
    
    return None


def get_current_shell() -> str:
    """Detect current shell.

    Returns:
        Shell name: 'bash', 'zsh', 'fish', 'powershell', 'cmd', or 'unknown'
    """
    import os

    # Check shell environment variable (most reliable)
    shell_path = os.environ.get("SHELL", "")
    if shell_path:
        shell_name = os.path.basename(shell_path)
        if shell_name in ("bash", "zsh", "fish"):
            return shell_name

    # Check parent process name on Unix-like systems
    if platform.system().lower() != "windows":
        try:
            result = subprocess.run(
                ["ps", "-o", "comm=", "-p", str(os.getppid())],
                capture_output=True,
                text=True,
                timeout=1
            )
            parent_process = result.stdout.strip()
            if parent_process in ("bash", "zsh", "fish"):
                return parent_process
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError):
            pass

    # Windows-specific detection
    if platform.system().lower() == "windows":
        # Check for PowerShell or CMD
        try:
            result = subprocess.run(
                ["powershell", "-Command", "echo $PSVersionTable.PSVersion"],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                return "powershell"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        return "cmd"

    return "unknown"


def check_agent_browser_installed() -> Tuple[bool, Optional[str]]:
    """Check if agent-browser is installed.
    
    Returns:
        Tuple of (is_installed, version_or_error)
    """
    try:
        result = subprocess.run(
            ["agent-browser", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return True, version
        else:
            return False, result.stderr.strip() or result.stdout.strip()
    except FileNotFoundError:
        return False, "agent-browser not found in PATH"
    except subprocess.TimeoutExpired:
        return False, "agent-browser command timed out"
    except Exception as e:
        return False, str(e)


def get_agent_browser_install_instructions() -> str:
    """Get installation instructions for agent-browser.
    
    Returns:
        Installation instructions string
    """
    platform_name, _ = get_platform_info()
    
    instructions = [
        "Install agent-browser:\n",
    ]
    
    if platform_name in ("wsl", "linux"):
        instructions.extend([
            "  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
            "  sudo apt install -y nodejs",
            "  npm install -g agent-browser",
            "  agent-browser install  # Download Chromium browser",
        ])
    elif platform_name == "macos":
        instructions.extend([
            "  brew install node",
            "  npm install -g agent-browser",
            "  agent-browser install  # Download Chromium browser",
        ])
    elif platform_name == "windows":
        instructions.extend([
            "  # Using WSL (recommended):",
            "  wsl --install",
            "  # Then in WSL:",
            "  curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -",
            "  sudo apt install -y nodejs",
            "  npm install -g agent-browser",
            "  agent-browser install",
        ])
    else:
        instructions.extend([
            "  # Install Node.js from https://nodejs.org/",
            "  npm install -g agent-browser",
            "  agent-browser install  # Download Chromium browser",
        ])
    
    return "\n".join(instructions)


def create_runs_directory() -> Path:
    """Create runs directory for session logs.
    
    Returns:
        Path to runs directory
    """
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    return runs_dir
