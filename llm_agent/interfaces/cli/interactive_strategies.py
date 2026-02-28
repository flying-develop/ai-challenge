"""Интерактивный CLI с поддержкой переключения стратегий управления контекстом.

Использование:
    python -m llm_agent.interfaces.cli.interactive_strategies

Команды в интерактивном режиме:
    /strategy          — показать текущую стратегию
    /switch <номер>    — переключить стратегию (1=Sliding Window, 2=Facts, 3=Branching)
    /stats             — показать статистику стратегии
    /facts             — показать извлечённые факты (для стратегии Facts)
    /checkpoint <id>   — сохранить checkpoint (для стратегии Branching)
    /branch <id> <cp>  — создать ветку от checkpoint-а
    /switch-branch <id>— переключиться на ветку
    /branches          — список веток
    /clear             — сбросить историю
    /help              — показать справку
    exit, quit         — выход
"""

from __future__ import annotations

import itertools
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Generator

import httpx

from llm_agent.application.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.config import get_config
from llm_agent.infrastructure.qwen_client import QwenHttpClient
from llm_agent.infrastructure.token_counter import TiktokenCounter


# ---------------------------------------------------------------------------
# Спиннер
# ---------------------------------------------------------------------------

@contextmanager
def spinner(text: str = "Думаю") -> Generator[None, None, None]:
    if not sys.stderr.isatty():
        yield
        return

    stop_event = threading.Event()
    frames = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def _spin() -> None:
        while not stop_event.is_set():
            frame = next(frames)
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
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Формирование стратегий
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты полезный ассистент. Давай краткие, точные ответы на русском языке. "
    "Помни всё, что было сказано в нашем разговоре."
)

WINDOW_SIZE = 10  # для Sliding Window и Facts


def make_strategies(llm_client=None):
    """Создать все три стратегии."""
    return {
        1: SlidingWindowStrategy(window_size=WINDOW_SIZE),
        2: StickyFactsStrategy(window_size=WINDOW_SIZE // 2, llm_client=llm_client),
        3: BranchingStrategy(),
    }


STRATEGY_NAMES = {
    1: "Sliding Window",
    2: "Sticky Facts / Key-Value Memory",
    3: "Branching (ветки диалога)",
}


# ---------------------------------------------------------------------------
# Справка
# ---------------------------------------------------------------------------

HELP_TEXT = """\
Команды:
  /strategy            — текущая стратегия
  /switch 1|2|3        — переключить стратегию
                         1 = Sliding Window
                         2 = Sticky Facts / Key-Value Memory
                         3 = Branching (ветки диалога)
  /stats               — статистика стратегии
  /facts               — факты (только для стратегии 2)
  /checkpoint <id>     — сохранить checkpoint (стратегия 3)
  /branch <id> <cp_id> — создать ветку от checkpoint-а (стратегия 3)
  /switch-branch <id>  — переключиться на ветку (стратегия 3)
  /branches            — список веток и checkpoint-ов (стратегия 3)
  /clear               — сбросить историю
  /help                — эта справка
  exit / quit          — выход
"""


# ---------------------------------------------------------------------------
# Обработка команд
# ---------------------------------------------------------------------------

def handle_command(
    cmd: str,
    agent: StrategyAgent,
    strategies: dict,
    current_strategy_num: int,
) -> tuple[int, bool]:
    """Обработать команду пользователя.

    Returns:
        (current_strategy_num, handled): номер текущей стратегии и True если была команда.
    """
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/help":
        print(HELP_TEXT)
        return current_strategy_num, True

    if command == "/strategy":
        print(f"Текущая стратегия: [{current_strategy_num}] {agent.strategy_name}")
        return current_strategy_num, True

    if command == "/switch":
        if len(parts) < 2 or parts[1] not in ("1", "2", "3"):
            print("Использование: /switch 1|2|3")
            for num, name in STRATEGY_NAMES.items():
                marker = " <-- текущая" if num == current_strategy_num else ""
                print(f"  {num} = {name}{marker}")
            return current_strategy_num, True
        new_num = int(parts[1])
        if new_num == current_strategy_num:
            print(f"Стратегия {STRATEGY_NAMES[new_num]} уже активна.")
            return current_strategy_num, True
        new_strategy = strategies[new_num]
        new_strategy.reset()
        change = agent.switch_strategy(new_strategy)
        current_strategy_num = new_num
        print(f"Стратегия переключена: {change}")
        print(f"История сброшена. Начинаем новый диалог.\n")
        return current_strategy_num, True

    if command == "/stats":
        stats = agent.get_stats()
        print("Статистика:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return current_strategy_num, True

    if command == "/facts":
        strategy = agent.strategy
        if not isinstance(strategy, StickyFactsStrategy):
            print("Команда /facts доступна только для стратегии Sticky Facts (2).")
            return current_strategy_num, True
        facts = strategy.facts
        if not facts:
            print("Фактов пока нет.")
        else:
            print("Извлечённые факты:")
            for k, v in facts.items():
                print(f"  {k}: {v}")
        return current_strategy_num, True

    if command == "/checkpoint":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /checkpoint доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 2:
            print("Использование: /checkpoint <id> [описание]")
            return current_strategy_num, True
        cp_id = parts[1]
        desc = " ".join(parts[2:]) if len(parts) > 2 else ""
        try:
            cp = strategy.save_checkpoint(cp_id, description=desc)
            print(f"Checkpoint '{cp_id}' сохранён (ход {cp.turn}, {len(cp.messages)} сообщений).")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/branch":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /branch доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 3:
            print("Использование: /branch <branch_id> <checkpoint_id> [описание]")
            return current_strategy_num, True
        branch_id = parts[1]
        cp_id = parts[2]
        desc = " ".join(parts[3:]) if len(parts) > 3 else ""
        try:
            branch = strategy.create_branch(branch_id, cp_id, description=desc)
            print(f"Ветка '{branch_id}' создана от checkpoint '{cp_id}'.")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/switch-branch":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /switch-branch доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        if len(parts) < 2:
            print(f"Использование: /switch-branch <id>")
            print(f"Доступные ветки: {', '.join(strategy.branches)}")
            return current_strategy_num, True
        branch_id = parts[1]
        try:
            strategy.switch_branch(branch_id)
            print(f"Переключено на ветку '{branch_id}'.")
            info = strategy.get_branch_info(branch_id)
            print(f"  Сообщений: {info.get('total_messages', info.get('messages_count', '?'))}")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/branches":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /branches доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        print(f"Текущая ветка: {strategy.current_branch_id}")
        print(f"\nCheckpoints:")
        if not strategy.checkpoints:
            print("  (нет)")
        else:
            for cp_id in strategy.checkpoints:
                cp = strategy._checkpoints[cp_id]
                print(f"  [{cp_id}] ход {cp.turn}, {len(cp.messages)} сообщ. {cp.description}")
        print(f"\nВетки:")
        for b_id in strategy.branches:
            info = strategy.get_branch_info(b_id)
            marker = " <-- текущая" if b_id == strategy.current_branch_id else ""
            desc = info.get("description", "")
            total = info.get("total_messages", info.get("messages_count", "?"))
            print(f"  [{b_id}] {total} сообщ. {desc}{marker}")
        return current_strategy_num, True

    if command == "/clear":
        agent.clear_history()
        print("История очищена.\n")
        return current_strategy_num, True

    return current_strategy_num, False


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def run_interactive() -> None:
    """Запустить интерактивный режим с переключением стратегий."""
    try:
        config = get_config()
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(1)

    system_prompt = os.environ.get("QWEN_SYSTEM_PROMPT", "").strip() or SYSTEM_PROMPT

    client = QwenHttpClient(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        timeout=config["timeout"],
    )

    token_counter = TiktokenCounter()

    strategies = make_strategies(llm_client=client)
    current_strategy_num = 1

    agent = StrategyAgent(
        llm_client=client,
        strategy=strategies[current_strategy_num],
        system_prompt=system_prompt,
        token_counter=token_counter,
    )

    print("=" * 60)
    print("  ИНТЕРАКТИВНЫЙ РЕЖИМ: Стратегии управления контекстом")
    print("=" * 60)
    print(f"\nАктивная стратегия: [{current_strategy_num}] {STRATEGY_NAMES[current_strategy_num]}")
    print("Введите /help для списка команд.\n")

    while True:
        try:
            branch_info = ""
            if isinstance(agent.strategy, BranchingStrategy):
                branch_info = f" [{agent.strategy.current_branch_id}]"
            prompt_prefix = f"[{current_strategy_num}]{branch_info}"
            raw = input(f"{prompt_prefix} Вы: ")
        except (KeyboardInterrupt, EOFError):
            print("\nДо свидания!")
            break

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

        # Обработка команд
        if user_input.startswith("/"):
            current_strategy_num, handled = handle_command(
                user_input, agent, strategies, current_strategy_num
            )
            if handled:
                continue

        # Обычное сообщение — отправляем в агент
        try:
            with spinner():
                reply = agent.ask(user_input)

            # Показываем ответ
            print(f"Агент: {reply}")

            # Показываем краткую статистику токенов
            usage = agent.last_token_usage
            if usage:
                print(
                    f"  [токены: prompt={usage.history_tokens}, "
                    f"response={usage.response_tokens}, "
                    f"total={usage.total_tokens}]"
                )
            print()

        except ValueError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
        except httpx.HTTPStatusError as exc:
            print(f"Ошибка API: {exc}", file=sys.stderr)
        except (httpx.TimeoutException, httpx.RequestError) as exc:
            print(f"Ошибка сети: {exc}", file=sys.stderr)

    client.close()


def main() -> None:
    run_interactive()


if __name__ == "__main__":
    main()
