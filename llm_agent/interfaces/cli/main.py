"""CLI entry point for the LLM agent.

Usage:
    python -m llm_agent.interfaces.cli.main --prompt "Hello, who are you?"
    python -m llm_agent.interfaces.cli.main --interactive
"""

from __future__ import annotations

import argparse
import sys

import httpx

from llm_agent.application.agent import SimpleAgent
from llm_agent.config import get_config
from llm_agent.infrastructure.qwen_client import QwenHttpClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-agent",
        description="Chat with the Qwen LLM from the command line.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prompt",
        metavar="TEXT",
        help="Single-shot prompt (prints response and exits).",
    )
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Start an interactive session (type 'exit' or 'quit' to stop).",
    )
    return parser


def run_single_shot(agent: SimpleAgent, prompt: str) -> None:
    """Run a single prompt and print the response."""
    try:
        print(agent.ask(prompt))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        sys.exit(1)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_interactive(agent: SimpleAgent) -> None:
    """Run an interactive chat loop until the user exits."""
    print("Interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            raw = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        # Sanitize surrogate characters that some terminals produce
        user_input = (
            raw.encode("utf-8", errors="surrogateescape")
            .decode("utf-8", errors="replace")
            .strip()
        )

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            reply = agent.ask(user_input)
            print(f"Agent: {reply}\n")
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
        except httpx.HTTPStatusError as exc:
            print(f"API error: {exc}", file=sys.stderr)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            print(f"Network error: {exc}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = get_config()
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)

    with QwenHttpClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        timeout=config["timeout"],
    ) as client:
        agent = SimpleAgent(llm_client=client)

        if args.prompt is not None:
            run_single_shot(agent, args.prompt)
        else:
            run_interactive(agent)


if __name__ == "__main__":
    main()
