"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_config() -> dict:
    """Return validated configuration dict.

    Raises:
        ValueError: If QWEN_API_KEY is not set.
    """
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "QWEN_API_KEY environment variable is required but not set. "
            "Copy .env.example to .env and fill in your API key."
        )

    return {
        "api_key": api_key,
        "base_url": os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        "model": os.environ.get("QWEN_MODEL", "qwen-plus"),
        "timeout": float(os.environ.get("QWEN_TIMEOUT_SECONDS", "30")),
    }
