"""Main entry point for browser-agent CLI."""

import sys
import argparse
import os

from dotenv import load_dotenv

from .cli import run_cli
from .openrouter import DEFAULT_MODEL


def main() -> int:
    """Main entry point.
    
    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        description="LLM-powered CLI for browser automation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model",
        help=f"Model to use (default: from OPENROUTER_MODEL env var or {DEFAULT_MODEL})",
        default=None
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (default: from OPENROUTER_API_KEY env var)",
        default=None
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="browser-agent 0.1.0"
    )
    
    args = parser.parse_args()
    
    # Load .env file if present
    load_dotenv()
    
    try:
        run_cli(
            api_key=args.api_key,
            model=args.model,
            debug=args.debug
        )
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
