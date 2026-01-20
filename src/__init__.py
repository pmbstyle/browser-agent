"""Browser Agent - LLM-powered CLI for browser automation."""

from .openrouter import OpenRouterClient, DEFAULT_MODEL
from .controller import AgentController
from .cli import run_cli

__version__ = "0.1.0"
__all__ = [
    "run_cli",
    "AgentController",
    "OpenRouterClient",
    "DEFAULT_MODEL",
    "__version__",
]
