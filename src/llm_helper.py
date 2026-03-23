"""Вспомогательные функции для создания LLM-функций.

Предоставляет _make_llm_fn() с сигнатурой (system: str, user: str) -> str,
совместимой с RAGPipeline и DialogManager.
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Callable


def make_llm_fn(timeout: float = 60.0) -> Callable[[str, str], str]:
    """Создать LLM-функцию с сигнатурой (system, user) -> str.

    Пробует провайдеры в порядке: mcp_server.llm_client → Qwen HTTP → OpenAI HTTP.

    Args:
        timeout: Таймаут HTTP запроса в секундах.

    Returns:
        Функция (system_prompt: str, user_prompt: str) -> str.

    Raises:
        RuntimeError: Если ни один провайдер недоступен.
    """
    # 1. Пробуем через mcp_server.llm_client (1-arg wrapper)
    try:
        from mcp_server.llm_client import create_llm_fn
        _one_arg = create_llm_fn(timeout=timeout)

        def _wrapped_mcp(system: str, user: str) -> str:
            combined = f"{system}\n\n{user}" if system else user
            return _one_arg(combined)

        return _wrapped_mcp
    except Exception:
        pass

    # 2. Пробуем Qwen напрямую
    qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if qwen_key:
        return _make_qwen_fn(qwen_key, timeout)

    # 3. Пробуем OpenAI-совместимый API
    openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_TOKEN")
    if openai_key:
        return _make_openai_fn(openai_key, timeout)

    raise RuntimeError(
        "Нет доступного LLM. Задайте QWEN_API_KEY или OPENAI_API_KEY в .env"
    )


def _make_qwen_fn(api_key: str, timeout: float) -> Callable[[str, str], str]:
    """Создать LLM-функцию через Qwen HTTP API."""
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    model = os.environ.get("QWEN_MODEL", "qwen-plus")

    def llm_fn(system: str, user: str) -> str:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1500,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return llm_fn


def _make_openai_fn(api_key: str, timeout: float) -> Callable[[str, str], str]:
    """Создать LLM-функцию через OpenAI-совместимый HTTP API."""
    base_url = os.environ.get("API_URL", "https://api.openai.com/v1")
    model = os.environ.get("API_MODEL", "gpt-4o-mini")

    def llm_fn(system: str, user: str) -> str:
        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": 1500,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return llm_fn
