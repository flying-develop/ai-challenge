"""Точка входа CLI для LLM-агента.

Использование:
    python -m llm_agent.interfaces.cli.main --prompt "Привет, кто ты?"
    python -m llm_agent.interfaces.cli.main --interactive
    python -m llm_agent.interfaces.cli.main --interactive --session work
    python -m llm_agent.interfaces.cli.main --interactive --new-session
    python -m llm_agent.interfaces.cli.main --list-sessions
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

import httpx

from llm_agent.application.agent import SimpleAgent
from llm_agent.config import get_config
from llm_agent.infrastructure.chat_history_repository import SQLiteChatHistoryRepository
from llm_agent.infrastructure.qwen_client import QwenHttpClient
from mcp_client.client import MCPClient
from mcp_client.config import MCPConfigParser, MCPServerConfig

# Путь к БД: можно переопределить через переменную окружения LLM_AGENT_DB_PATH
_DEFAULT_DB_PATH = Path.home() / ".llm-agent" / "history.db"


def _get_db_path() -> Path:
    env_val = os.environ.get("LLM_AGENT_DB_PATH", "").strip()
    return Path(env_val) if env_val else _DEFAULT_DB_PATH


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
    mode.add_argument(
        "--list-sessions",
        action="store_true",
        help="Показать список сохранённых сессий и выйти.",
    )

    parser.add_argument(
        "--session",
        metavar="NAME",
        default="default",
        help="Имя сессии для загрузки/сохранения истории (по умолчанию: 'default').",
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Начать новую сессию, очистив историю перед стартом.",
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


def _load_mcp_servers() -> dict[str, MCPServerConfig]:
    """Загрузить конфигурацию MCP-серверов. Возвращает dict name→config."""
    try:
        parser = MCPConfigParser()
        servers = parser.load()
        return {s.name: s for s in servers}
    except Exception:
        return {}


def _get_mcp_client(server_name: str, servers: dict[str, MCPServerConfig]) -> MCPClient | None:
    """Получить MCPClient для сервера по имени. None если не сконфигурирован."""
    config = servers.get(server_name)
    if config is None:
        return None
    return MCPClient(config)


def handle_mcp_call(args_str: str, servers: dict[str, MCPServerConfig]) -> None:
    """
    Обработчик команды /mcp call <server> <tool> [key=value ...].

    Примеры:
        /mcp call cbr_currencies get_exchange_rates
        /mcp call cbr_currencies get_currency_rate currency_code=USD
        /mcp call cbr_currencies convert_currency amount=100 currency_code=USD direction=to_rub
    """
    parts = args_str.strip().split()

    if len(parts) < 2:
        print("Использование: /mcp call <server> <tool> [key=value ...]")
        print("Пример: /mcp call cbr_currencies get_exchange_rates")
        return

    server_name = parts[0]
    tool_name = parts[1]
    raw_kwargs = parts[2:]

    # Парсинг key=value аргументов
    arguments: dict = {}
    for kv in raw_kwargs:
        if "=" not in kv:
            print(f"Ошибка: аргумент '{kv}' должен быть в формате key=value")
            return
        key, _, value = kv.partition("=")
        # Автоматическое приведение типов: числа → float/int
        try:
            arguments[key] = int(value)
        except ValueError:
            try:
                arguments[key] = float(value)
            except ValueError:
                arguments[key] = value

    client = _get_mcp_client(server_name, servers)
    if client is None:
        print(f"MCP-сервер '{server_name}' не сконфигурирован.")
        print(f"Доступные серверы: {', '.join(servers.keys()) or '(нет)'}")
        return

    print(f"📡 Вызов: {tool_name}({', '.join(f'{k}={v}' for k, v in arguments.items())})")
    try:
        result = client.call_tool(tool_name, arguments)
        print(result)
    except RuntimeError as exc:
        print(str(exc))


def handle_convert(args_str: str, servers: dict[str, MCPServerConfig]) -> None:
    """
    Обработчик команды /convert.

    Форматы:
        /convert 100 USD         — валюта → рубли
        /convert 10000 RUB USD   — рубли → валюта
    """
    parts = args_str.strip().split()

    if len(parts) == 2:
        try:
            amount = float(parts[0].replace(",", "."))
        except ValueError:
            print(f"Ошибка: '{parts[0]}' — не число. Укажите сумму цифрами.")
            return
        currency = parts[1].upper()
        direction = "to_rub"

    elif len(parts) == 3 and parts[1].upper() == "RUB":
        try:
            amount = float(parts[0].replace(",", "."))
        except ValueError:
            print(f"Ошибка: '{parts[0]}' — не число. Укажите сумму цифрами.")
            return
        currency = parts[2].upper()
        direction = "from_rub"

    else:
        print("Использование:")
        print("  /convert 100 USD        — доллары в рубли")
        print("  /convert 10000 RUB USD  — рубли в доллары")
        return

    client = _get_mcp_client("cbr_currencies", servers)
    if client is None:
        print("MCP-сервер 'cbr_currencies' не сконфигурирован.")
        return

    try:
        result = client.call_tool("convert_currency", {
            "amount": amount,
            "currency_code": currency,
            "direction": direction,
        })
        print(f"💱 {result}")
    except RuntimeError as exc:
        print(str(exc))


def run_interactive(agent: SimpleAgent, history_count: int) -> None:
    """Запустить интерактивный цикл чата до выхода пользователя."""
    # Загружаем конфигурацию MCP-серверов один раз при старте
    mcp_servers = _load_mcp_servers()

    if history_count > 0:
        print(
            f"Загружена история: {history_count} сообщений. "
            "Введите 'clear' для сброса истории.\n"
        )
    else:
        print(
            "Интерактивный режим. "
            "Введите 'exit'/'quit' для выхода, 'clear' для сброса истории.\n"
            "Команды: /mcp call <server> <tool> [key=value ...], /convert <сумма> <валюта>\n"
        )

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

        # Команда /mcp call <server> <tool> [key=value ...]
        if user_input.startswith("/mcp call "):
            handle_mcp_call(user_input[len("/mcp call "):], mcp_servers)
            print()
            continue

        # Команда /mcp list — список доступных серверов
        if user_input.strip() == "/mcp list":
            if mcp_servers:
                print("Сконфигурированные MCP-серверы:")
                for name, cfg in mcp_servers.items():
                    print(f"  • {name}: {cfg.description}")
            else:
                print("MCP-серверы не сконфигурированы.")
            print()
            continue

        # Команда /convert
        if user_input.startswith("/convert"):
            handle_convert(user_input[len("/convert"):].strip(), mcp_servers)
            print()
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

    db_path = _get_db_path()

    # --list-sessions не требует конфигурации API
    if args.list_sessions:
        with SQLiteChatHistoryRepository(db_path, session_id="default") as repo:
            sessions = repo.list_sessions()
        if not sessions:
            print("Нет сохранённых сессий.")
        else:
            print("Сохранённые сессии:")
            for s in sessions:
                print(f"  • {s}")
        return

    try:
        config = get_config()
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(1)

    # Читаем опциональный системный промпт из переменных окружения
    system_prompt = os.environ.get("QWEN_SYSTEM_PROMPT", "").strip() or None

    session_id = args.session

    with (
        SQLiteChatHistoryRepository(db_path, session_id=session_id) as repo,
        QwenHttpClient(
            api_key=config["api_key"],
            base_url=config["base_url"],
            model=config["model"],
            timeout=config["timeout"],
        ) as client,
    ):
        # --new-session: очищаем историю перед стартом
        if args.new_session:
            repo.clear()

        history_count = repo.message_count()
        agent = SimpleAgent(llm_client=client, system_prompt=system_prompt, history_repo=repo)

        if args.prompt is not None:
            run_single_shot(agent, args.prompt)
        else:
            run_interactive(agent, history_count)


if __name__ == "__main__":
    main()
