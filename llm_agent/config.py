"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def get_config() -> dict:
    """Вернуть проверенный словарь конфигурации.

    Raises:
        ValueError: Если QWEN_API_KEY не задан.
    """
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "Переменная окружения QWEN_API_KEY обязательна, но не задана. "
            "Скопируйте .env.example в .env и укажите ваш API-ключ."
        )

    return {
        "api_key": api_key,
        "base_url": os.environ.get(
            "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        ),
        "model": os.environ.get("QWEN_MODEL", "qwen-plus"),
        "timeout": float(os.environ.get("QWEN_TIMEOUT_SECONDS", "30")),
    }
