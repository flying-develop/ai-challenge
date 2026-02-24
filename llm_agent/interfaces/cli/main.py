"""Точка входа CLI для LLM-агента.

Использование:
    python -m llm_agent.interfaces.cli.main --prompt "Привет, кто ты?"
    python -m llm_agent.interfaces.cli.main --interactive
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator

import httpx

from llm_agent.application.agent import SimpleAgent
from llm_agent.config import get_config
from llm_agent.infrastructure.qwen_client import QwenHttpClient


@contextmanager
def spinner(text: str = "Думаю") -> Generator[None, None, None]:
    """Контекстный менеджер: показывает анимированный спиннер до завершения блока.

    Спиннер выводится в stderr, чтобы не мешать stdout-выводу ответа.
    Работает только если stderr подключён к терминалу (TTY).
    """
    if not sys.stderr.isatty():
        # Не терминал (например, pipe) — просто выполняем блок без анимации
        yield
        return

    stop_event = threading.Event()
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def _spin() -> None:
        while not stop_event.is_set():
            frame = next(frames)
            # \r возвращает курсор в начало строки, \033[K стирает до конца строки
            sys.stderr.write(f"\r\033[K{frame} {text}...")
            sys.stderr.flush()
            time.sleep(0.08)

    thread = threading.Thread(target=_spin, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join()
        # Стираем строку спиннера полностью
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-agent",
        description="Общение с Qwen LLM через командную строку.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prompt",
        metavar="TEXT",
        help="Одиночный запрос (выводит ответ и завершает работу).",
    )
    mode.add_argument(
        "--interactive",
        action="store_true",
        help="Запустить интерактивную сессию (введите 'exit' или 'quit' для выхода).",
    )
    return parser


def run_single_shot(agent: SimpleAgent, prompt: str) -> None:
    """Выполнить одиночный запрос и вывести ответ."""
    try:
        with spinner():
            result = agent.ask(prompt)
        print(result)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as exc:
        print(f"Ошибка API: {exc}", file=sys.stderr)
        sys.exit(1)
    except (httpx.TimeoutException, httpx.RequestError) as exc:
        print(f"Ошибка сети: {exc}", file=sys.stderr)
        sys.exit(1)


def run_interactive(agent: SimpleAgent) -> None:
    """Запустить интерактивный цикл чата до выхода пользователя."""
    print("Интерактивный режим. Введите 'exit'/'quit' для выхода, 'clear' для сброса истории.\n")
    while True:
        try:
            raw = input("Вы: ")
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

        # Очищаем суррогатные символы, которые могут создавать некоторые терминалы
        user_input = (
            raw.encode("utf-8", errors="surrogateescape")
            .decode("utf-8", errors="replace")
            .strip()
        )

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("До свидания!")
            break

        # Команда сброса истории диалога
        if user_input.lower() == "clear":
            agent.clear_history()
            print("История очищена.\n")
            continue

        try:
            with spinner():
                reply = agent.ask(user_input)
            print(f"Агент: {reply}\n")
        except ValueError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
        except httpx.HTTPStatusError as exc:
            print(f"Ошибка API: {exc}", file=sys.stderr)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            print(f"Ошибка сети: {exc}", file=sys.stderr)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = get_config()
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(1)

    # Читаем опциональный системный промпт из переменных окружения
    system_prompt = os.environ.get("QWEN_SYSTEM_PROMPT", "").strip() or None

    with QwenHttpClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        timeout=config["timeout"],
    ) as client:
        agent = SimpleAgent(llm_client=client, system_prompt=system_prompt)

        if args.prompt is not None:
            run_single_shot(agent, args.prompt)
        else:
            run_interactive(agent)


if __name__ == "__main__":
    main()
