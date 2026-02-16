#!/usr/bin/env python3
"""CLI for OpenAI-compatible chat completion APIs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any
from urllib import error, request


@dataclass(frozen=True)
class Settings:
    api_url: str
    model: str
    token: str

    @classmethod
    def from_env(cls) -> "Settings":
        api_url = os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        token = os.getenv("LLM_API_TOKEN", "").strip()

        missing = [
            name
            for name, value in (
                ("LLM_MODEL", model),
                ("LLM_API_TOKEN", token),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Не заданы обязательные переменные окружения: " + ", ".join(missing)
            )

        return cls(api_url=api_url, model=model, token=token)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Отправляет prompt в OpenAI-совместимое API и печатает ответ"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Текст prompt. Если не передан, будет запрошен интерактивно.",
    )
    return parser.parse_args()


def request_completion(settings: Settings, prompt: str) -> str:
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }

    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=settings.api_url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.token}",
        },
    )

    try:
        with request.urlopen(http_request, timeout=60) as response:
            raw_data = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP ошибка {exc.code} при вызове API: {details}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ошибка подключения к API: {exc.reason}") from exc

    data = json.loads(raw_data)

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"Неожиданный формат ответа API: {json.dumps(data, ensure_ascii=False)}"
        ) from exc


def main() -> int:
    args = parse_args()

    prompt = args.prompt or input("Введите prompt: ").strip()
    if not prompt:
        print("Prompt не должен быть пустым", file=sys.stderr)
        return 1

    try:
        settings = Settings.from_env()
        answer = request_completion(settings, prompt)
    except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
