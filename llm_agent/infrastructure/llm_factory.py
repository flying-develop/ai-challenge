"""Фабрика LLM-клиентов для четырёх провайдеров: Qwen, OpenAI, Claude, Ollama.

Пример использования:
    from llm_agent.infrastructure.llm_factory import build_client

    client = build_client("qwen")          # qwen-plus из QWEN_API_KEY
    client = build_client("openai", model="gpt-4o-mini")
    client = build_client("claude", model="claude-haiku-4-5-20251001")
    client = build_client("ollama")        # qwen2.5:3b локально без ключа

Конфигурация (переменные окружения):
    Qwen:
        QWEN_API_KEY   — обязателен
        QWEN_BASE_URL  — (по умолчанию dashscope-intl)
        QWEN_MODEL     — (по умолчанию qwen-plus)
        QWEN_TIMEOUT_SECONDS

    OpenAI:
        OPENAI_API_KEY — обязателен
        OPENAI_BASE_URL — опционально (Azure, Together, etc.)
        OPENAI_MODEL   — (по умолчанию gpt-4o-mini)

    Claude / Anthropic:
        ANTHROPIC_API_KEY — обязателен (или CLAUDE_SESSION_INGRESS_TOKEN_FILE)
        ANTHROPIC_MODEL   — (по умолчанию claude-haiku-4-5-20251001)

    Ollama (локально, без ключа):
        OLLAMA_BASE_URL — (по умолчанию http://localhost:11434)
        OLLAMA_MODEL    — (по умолчанию qwen2.5:3b)
"""

from __future__ import annotations

import os
from typing import Literal

from llm_agent.domain.protocols import LLMClientProtocol

Provider = Literal["qwen", "openai", "claude", "ollama"]

# Провайдеры, поддерживаемые фабрикой
SUPPORTED_PROVIDERS: list[str] = ["qwen", "openai", "claude", "ollama"]

# Модели по умолчанию для каждого провайдера
DEFAULT_MODELS: dict[str, str] = {
    "qwen": "qwen-plus",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5-20251001",
    "ollama": "qwen2.5:3b",
}

# Краткие описания провайдеров
PROVIDER_LABELS: dict[str, str] = {
    "qwen": "Qwen (Alibaba Cloud)",
    "openai": "OpenAI (GPT)",
    "claude": "Claude (Anthropic)",
    "ollama": "Ollama (локальная модель)",
}

# Переменные окружения с API-ключами (Ollama ключа не требует)
_KEY_ENV_VARS: dict[str, str] = {
    "qwen": "QWEN_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "ollama": "",
}


def get_available_providers() -> list[dict]:
    """Вернуть список провайдеров с информацией о доступности.

    Для облачных провайдеров проверяется наличие API-ключа.
    Для Ollama делается HTTP-запрос к /api/tags (таймаут 5с).

    Returns:
        Список словарей:
            - provider: str
            - label: str
            - available: bool    (есть ключ или Ollama отвечает)
            - key_var: str       (переменная окружения с ключом; "" для Ollama)
            - hint: str          (подсказка при недоступности)
            - default_model: str
    """
    result = []
    for provider in SUPPORTED_PROVIDERS:
        key_var = _KEY_ENV_VARS[provider]

        if provider == "ollama":
            # Ollama: проверяем HTTP-доступность вместо ключа
            available, hint = _check_ollama_available()
        else:
            available = bool(os.environ.get(key_var, "").strip())
            # Для Claude проверяем альтернативный способ (файл токена)
            if not available and provider == "claude":
                token_file = os.environ.get("CLAUDE_SESSION_INGRESS_TOKEN_FILE", "").strip()
                available = bool(token_file and os.path.exists(token_file))
            hint = f"Нужна переменная: {key_var}" if not available else ""

        result.append({
            "provider": provider,
            "label": PROVIDER_LABELS[provider],
            "available": available,
            "key_var": key_var,
            "hint": hint,
            "default_model": DEFAULT_MODELS[provider],
        })
    return result


def _check_ollama_available() -> tuple[bool, str]:
    """Быстрая проверка доступности Ollama (таймаут 5с).

    Returns:
        (available, hint) — флаг доступности и подсказка при ошибке.
    """
    import urllib.request
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODELS["ollama"])
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        if model in models or f"{model}:latest" in models:
            return True, ""
        return False, f"Модель не загружена. Выполните: ollama pull {model}"
    except Exception:
        return False, f"Сервер недоступен. Запустите: ollama serve"


def get_provider_info(provider: str) -> dict:
    """Вернуть информацию о конкретном провайдере."""
    provider = provider.lower().strip()
    _validate_provider(provider)
    infos = get_available_providers()
    return next(i for i in infos if i["provider"] == provider)


def build_client(
    provider: str,
    model: str | None = None,
    timeout: float = 30.0,
) -> LLMClientProtocol:
    """Создать LLM-клиент для указанного провайдера.

    Args:
        provider: Один из «qwen», «openai», «claude».
        model: Название модели. Если None — берётся из переменной окружения
               или DEFAULT_MODELS.
        timeout: Таймаут запроса в секундах (только для Qwen).

    Returns:
        LLMClientProtocol-совместимый клиент.

    Raises:
        ValueError: Если провайдер не поддерживается или API-ключ не найден.
    """
    provider = provider.lower().strip()
    _validate_provider(provider)

    if provider == "qwen":
        return _build_qwen(model=model, timeout=timeout)
    if provider == "openai":
        return _build_openai(model=model)
    if provider == "claude":
        return _build_claude(model=model)
    if provider == "ollama":
        return _build_ollama(model=model, timeout=timeout)

    raise ValueError(f"Неизвестный провайдер: {provider!r}")


# ---------------------------------------------------------------------------
# Внутренние конструкторы
# ---------------------------------------------------------------------------

def _validate_provider(provider: str) -> None:
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Неизвестный провайдер: {provider!r}. "
            f"Доступные: {', '.join(SUPPORTED_PROVIDERS)}"
        )


def _build_qwen(model: str | None, timeout: float) -> LLMClientProtocol:
    api_key = os.environ.get("QWEN_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "QWEN_API_KEY не задан. "
            "Добавьте в .env: QWEN_API_KEY=your-key"
        )
    model = model or os.environ.get("QWEN_MODEL", DEFAULT_MODELS["qwen"])
    base_url = os.environ.get(
        "QWEN_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )
    timeout = float(os.environ.get("QWEN_TIMEOUT_SECONDS", str(timeout)))

    from llm_agent.infrastructure.qwen_client import QwenHttpClient
    return QwenHttpClient(api_key=api_key, base_url=base_url, model=model, timeout=timeout)


def _build_openai(model: str | None) -> LLMClientProtocol:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY не задан. "
            "Добавьте в .env: OPENAI_API_KEY=sk-..."
        )
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_MODELS["openai"])
    base_url = os.environ.get("OPENAI_BASE_URL") or None

    from llm_agent.infrastructure.openai_client import OpenAIClient
    return OpenAIClient(api_key=api_key, model=model, base_url=base_url)


def _build_claude(model: str | None) -> LLMClientProtocol:
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODELS["claude"])

    from llm_agent.infrastructure.anthropic_client import AnthropicHttpClient
    # AnthropicHttpClient сам читает ключ из окружения/файла
    return AnthropicHttpClient(model=model)


def _build_ollama(model: str | None, timeout: float = 120.0) -> LLMClientProtocol:
    """Создать OllamaHttpClient.

    Не требует API-ключа. Проверяет доступность сервера и модели,
    выдаёт понятную ошибку если Ollama не запущена.
    """
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = model or os.environ.get("OLLAMA_MODEL", DEFAULT_MODELS["ollama"])

    from llm_agent.infrastructure.ollama_client import OllamaHttpClient
    client = OllamaHttpClient(model=model, base_url=base_url, timeout=timeout)

    # Проверяем доступность при создании клиента
    available, hint = _check_ollama_available()
    if not available:
        raise ValueError(
            f"Ollama недоступна: {hint}\n"
            f"  Сервер: {base_url}\n"
            f"  Модель: {model}"
        )
    return client


def current_provider_from_env() -> str:
    """Определить провайдер из переменной LLM_PROVIDER (или первый доступный).

    Порядок проверки:
    1. LLM_PROVIDER env var
    2. Первый провайдер с доступным ключом (qwen → openai → claude)
    """
    env_provider = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if env_provider in SUPPORTED_PROVIDERS:
        return env_provider

    for info in get_available_providers():
        if info["available"]:
            return info["provider"]

    return "qwen"  # fallback
