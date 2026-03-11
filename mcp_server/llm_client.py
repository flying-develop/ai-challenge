"""LLM-клиент для генерации дневных сводок новостей.

Образовательная концепция: MCP-сервер может сам вызывать LLM.
Это позволяет серверу автономно обрабатывать данные без участия агента.

Реализация:
- Читает провайдер из .env (LLM_PROVIDER=qwen|openai|claude)
- Делает HTTP-запрос через urllib (инвариант stdlib, без httpx)
- Возвращает callable: (prompt: str) -> str

Поддерживаемые провайдеры:
- qwen (Alibaba Cloud / OpenAI-compatible API)
- openai (OpenAI API)
- claude (Anthropic API — отдельный формат запроса)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _make_openai_compatible_fn(
    api_key: str,
    base_url: str,
    model: str,
    timeout: float = 60.0,
) -> Callable[[str], str]:
    """Создать функцию для OpenAI-совместимого API (Qwen, OpenAI).

    Args:
        api_key: API-ключ
        base_url: Базовый URL (с trailing slash)
        model: Название модели
        timeout: Таймаут запроса в секундах

    Returns:
        Функция (prompt: str) -> str
    """
    # Нормализуем URL: убеждаемся, что кончается на /
    base = base_url.rstrip("/") + "/"
    endpoint = base + "chat/completions"

    def call_llm(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ошибка LLM API {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ошибка сети при вызове LLM: {exc}") from exc

        choices = result.get("choices", [])
        if not choices:
            raise RuntimeError(f"LLM вернул пустой ответ: {result}")
        return choices[0]["message"]["content"]

    return call_llm


def _make_claude_fn(
    api_key: str,
    model: str,
    timeout: float = 60.0,
) -> Callable[[str], str]:
    """Создать функцию для Anthropic Claude API.

    Args:
        api_key: Anthropic API key (sk-ant-...)
        model: Название модели (claude-haiku-4-5-20251001 и др.)
        timeout: Таймаут запроса в секундах

    Returns:
        Функция (prompt: str) -> str
    """
    endpoint = "https://api.anthropic.com/v1/messages"

    def call_llm(prompt: str) -> str:
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ошибка Claude API {exc.code}: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Ошибка сети при вызове Claude: {exc}") from exc

        content = result.get("content", [])
        if not content:
            raise RuntimeError(f"Claude вернул пустой ответ: {result}")
        # Берём текст из первого блока
        for block in content:
            if block.get("type") == "text":
                return block["text"]
        raise RuntimeError(f"Claude: не найден текстовый блок в ответе: {result}")

    return call_llm


def create_llm_fn(timeout: float = 60.0) -> Callable[[str], str]:
    """Создать функцию для вызова LLM на основе .env конфигурации.

    Образовательная концепция: MCP-сервер использует тот же провайдер,
    что настроен для основного агента. Единая конфигурация через .env.

    Приоритет определения провайдера:
    1. LLM_PROVIDER=qwen|openai|claude
    2. Наличие ключей: QWEN_API_KEY → qwen, OPENAI_API_KEY → openai,
       ANTHROPIC_API_KEY → claude

    Returns:
        Функция (prompt: str) -> str

    Raises:
        ValueError: если не удалось определить провайдера или найти API-ключ
    """
    provider = os.environ.get("LLM_PROVIDER", "").strip().lower()

    # Если провайдер не указан явно — определяем по наличию ключей
    if not provider:
        if os.environ.get("QWEN_API_KEY", "").strip():
            provider = "qwen"
        elif os.environ.get("OPENAI_API_KEY", "").strip():
            provider = "openai"
        elif os.environ.get("ANTHROPIC_API_KEY", "").strip():
            provider = "claude"
        else:
            raise ValueError(
                "Не удалось определить LLM-провайдер. "
                "Укажите LLM_PROVIDER в .env или один из ключей: "
                "QWEN_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
            )

    if provider == "qwen":
        api_key = os.environ.get("QWEN_API_KEY", "").strip()
        if not api_key:
            raise ValueError("QWEN_API_KEY не задан в .env")
        base_url = os.environ.get(
            "QWEN_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        model = os.environ.get("QWEN_MODEL", "qwen-plus")
        return _make_openai_compatible_fn(api_key, base_url, model, timeout)

    elif provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан в .env")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        return _make_openai_compatible_fn(api_key, base_url, model, timeout)

    elif provider == "claude":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY не задан в .env")
        model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        return _make_claude_fn(api_key, model, timeout)

    else:
        raise ValueError(
            f"Неизвестный провайдер '{provider}'. "
            f"Поддерживаемые: qwen, openai, claude"
        )
