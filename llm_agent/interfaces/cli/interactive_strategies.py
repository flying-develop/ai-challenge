"""Интерактивный CLI с поддержкой переключения стратегий и LLM-провайдеров.

Использование:
    python chat.py                         # авто-определение провайдера
    python chat.py --provider qwen         # выбрать провайдер
    python chat.py --provider openai --model gpt-4o-mini
    python chat.py --provider claude --model claude-haiku-4-5-20251001
    python chat.py --strategy 2            # сразу запустить с нужной стратегией

Или через модуль:
    python -m llm_agent.interfaces.cli.interactive_strategies --provider openai

Команды в интерактивном режиме:
    /provider              — показать текущего провайдера
    /provider <name>       — переключить провайдер (qwen / openai / claude)
    /model <name>          — сменить модель в рамках текущего провайдера
    /providers             — список доступных провайдеров

    /strategy              — показать текущую стратегию
    /switch <номер>        — переключить стратегию (1=Sliding Window, 2=Facts, 3=Branching)
    /stats                 — показать статистику агента

    /facts                 — показать извлечённые факты (для стратегии 2)
    /checkpoint <id>       — сохранить checkpoint (для стратегии 3)
    /branch <id> <cp_id>   — создать ветку от checkpoint-а
    /switch-branch <id>    — переключиться на ветку
    /branches              — список веток и checkpoint-ов

    /clear                 — сбросить историю
    /help                  — показать справку
    exit, quit             — выход
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

from llm_agent.application.context_strategies import (
    BranchingStrategy,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.infrastructure.llm_factory import (
    DEFAULT_MODELS,
    PROVIDER_LABELS,
    SUPPORTED_PROVIDERS,
    build_client,
    current_provider_from_env,
    get_available_providers,
)
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
# Параметры
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Ты полезный ассистент. Давай краткие, точные ответы на русском языке. "
    "Помни всё, что было сказано в нашем разговоре."
)

WINDOW_SIZE = 10
FACTS_WINDOW_SIZE = 6

STRATEGY_NAMES = {
    1: "Sliding Window",
    2: "Sticky Facts / Key-Value Memory",
    3: "Branching (ветки диалога)",
}


# ---------------------------------------------------------------------------
# Справка
# ---------------------------------------------------------------------------

HELP_TEXT = """\
═══ LLM-агент: стратегии контекста + переключение провайдеров ════════

ПРОВАЙДЕРЫ:
  /provider              — показать текущего провайдера и модель
  /provider <name>       — переключить: qwen | openai | claude
  /model <name>          — сменить модель (в рамках текущего провайдера)
  /providers             — список провайдеров с доступностью ключей

СТРАТЕГИИ:
  /strategy              — текущая стратегия
  /switch 1|2|3          — переключить стратегию
                           1 = Sliding Window
                           2 = Sticky Facts / Key-Value Memory
                           3 = Branching (ветки диалога)
  /stats                 — полная статистика агента

STICKY FACTS (стратегия 2):
  /facts                 — показать извлечённые факты

BRANCHING (стратегия 3):
  /checkpoint <id>       — сохранить checkpoint
  /branch <id> <cp_id>   — создать ветку от checkpoint-а
  /switch-branch <id>    — переключиться на ветку
  /branches              — список веток и checkpoint-ов

ОБЩИЕ:
  /clear                 — сбросить историю диалога
  /help                  — эта справка
  exit / quit            — выход
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
    """Обработать команду. Возвращает (strategy_num, was_handled)."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    # ---- Справка ----
    if command == "/help":
        print(HELP_TEXT)
        return current_strategy_num, True

    # ---- Провайдеры ----
    if command == "/providers":
        print("\n  Доступные провайдеры:")
        for info in get_available_providers():
            status = "✓" if info["available"] else "✗ нет ключа"
            marker = " ◄ текущий" if info["provider"] == agent.provider_name else ""
            print(
                f"    [{info['provider']:8}] {info['label']:<28} "
                f"{status}{marker}"
            )
            if not info["available"]:
                print(f"              Нужна переменная: {info['key_var']}")
        print()
        return current_strategy_num, True

    if command == "/provider":
        if len(parts) == 1:
            model_str = f"  модель: {agent.model_name}" if agent.model_name else ""
            print(f"Провайдер: {agent.provider_name}  ({PROVIDER_LABELS.get(agent.provider_name, '')}){model_str}")
            return current_strategy_num, True

        new_provider = parts[1].lower()
        if new_provider not in SUPPORTED_PROVIDERS:
            print(f"Неизвестный провайдер: {new_provider!r}")
            print(f"Доступные: {', '.join(SUPPORTED_PROVIDERS)}")
            return current_strategy_num, True

        model = parts[2] if len(parts) > 2 else None
        try:
            new_client = build_client(new_provider, model=model)
            model_used = model or DEFAULT_MODELS[new_provider]
            change = agent.switch_client(
                new_client,
                provider_name=new_provider,
                model_name=model_used,
            )
            print(f"Провайдер переключён: {change}")
            print(f"Модель: {model_used}")
            print(f"История сохранена — продолжаем диалог с новым провайдером.\n")
        except ValueError as exc:
            print(f"Ошибка: {exc}")
        return current_strategy_num, True

    if command == "/model":
        if len(parts) < 2:
            print(f"Использование: /model <название-модели>")
            print(f"Текущий провайдер: {agent.provider_name}, модель: {agent.model_name or '(по умолчанию)'}")
            return current_strategy_num, True
        model = parts[1]
        try:
            new_client = build_client(agent.provider_name, model=model)
            change = agent.switch_client(
                new_client,
                provider_name=agent.provider_name,
                model_name=model,
            )
            print(f"Модель переключена: {change}")
            print(f"История сохранена — продолжаем диалог.\n")
        except ValueError as exc:
            print(f"Ошибка: {exc}")
        return current_strategy_num, True

    # ---- Стратегии ----
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
            print(f"Стратегия «{STRATEGY_NAMES[new_num]}» уже активна.")
            return current_strategy_num, True
        new_strategy = strategies[new_num]
        new_strategy.reset()
        # Если переключаемся на StickyFacts — прокидываем текущий клиент
        if isinstance(new_strategy, StickyFactsStrategy):
            new_strategy._llm_client = agent._llm_client
        change = agent.switch_strategy(new_strategy)
        current_strategy_num = new_num
        print(f"Стратегия переключена: {change}")
        print(f"История сброшена. Начинаем новый диалог.\n")
        return current_strategy_num, True

    if command == "/stats":
        stats = agent.get_stats()
        print("Статистика:")
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")
        return current_strategy_num, True

    # ---- Sticky Facts ----
    if command == "/facts":
        strategy = agent.strategy
        if not isinstance(strategy, StickyFactsStrategy):
            print("Команда /facts доступна только для стратегии Sticky Facts (2).")
            return current_strategy_num, True
        facts = strategy.facts
        if not facts:
            print("Фактов пока нет.")
        else:
            print(f"Извлечённые факты ({len(facts)} шт.):")
            for k, v in facts.items():
                print(f"  {k}: {v}")
        return current_strategy_num, True

    # ---- Branching ----
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
        branch_id, cp_id = parts[1], parts[2]
        desc = " ".join(parts[3:]) if len(parts) > 3 else ""
        try:
            strategy.create_branch(branch_id, cp_id, description=desc)
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
        try:
            strategy.switch_branch(parts[1])
            info = strategy.get_branch_info(parts[1])
            total = info.get("total_messages", info.get("messages_count", "?"))
            print(f"Переключено на ветку '{parts[1]}' ({total} сообщений).")
        except ValueError as e:
            print(f"Ошибка: {e}")
        return current_strategy_num, True

    if command == "/branches":
        strategy = agent.strategy
        if not isinstance(strategy, BranchingStrategy):
            print("Команда /branches доступна только для стратегии Branching (3).")
            return current_strategy_num, True
        print(f"Текущая ветка: {strategy.current_branch_id}")
        print("\nCheckpoints:")
        if not strategy.checkpoints:
            print("  (нет)")
        else:
            for cp_id in strategy.checkpoints:
                cp = strategy._checkpoints[cp_id]
                print(f"  [{cp_id}] ход {cp.turn}, {len(cp.messages)} сообщ. {cp.description}")
        print("\nВетки:")
        for b_id in strategy.branches:
            info = strategy.get_branch_info(b_id)
            marker = " ◄ текущая" if b_id == strategy.current_branch_id else ""
            total = info.get("total_messages", info.get("messages_count", "?"))
            desc = info.get("description", "")
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

def run_interactive(provider: str, model: str | None) -> None:
    system_prompt = os.environ.get("QWEN_SYSTEM_PROMPT", "").strip() or SYSTEM_PROMPT

    # Строим клиент
    try:
        client = build_client(provider, model=model)
        model_name = model or DEFAULT_MODELS.get(provider, "")
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    token_counter = TiktokenCounter()

    # Стратегии (StickyFacts получает тот же клиент для извлечения фактов)
    strategies: dict = {
        1: SlidingWindowStrategy(window_size=WINDOW_SIZE),
        2: StickyFactsStrategy(window_size=FACTS_WINDOW_SIZE, llm_client=client),
        3: BranchingStrategy(),
    }
    current_strategy_num = 1

    agent = StrategyAgent(
        llm_client=client,
        strategy=strategies[current_strategy_num],
        system_prompt=system_prompt,
        token_counter=token_counter,
        provider_name=provider,
        model_name=model_name,
    )

    print("=" * 62)
    print("  LLM-агент: стратегии контекста + выбор провайдера")
    print("=" * 62)
    print(f"\n  Провайдер : {provider}  ({PROVIDER_LABELS.get(provider, '')})")
    print(f"  Модель    : {model_name}")
    print(f"  Стратегия : [{current_strategy_num}] {STRATEGY_NAMES[current_strategy_num]}")
    print(f"\n  Введите /help для списка команд.\n")

    while True:
        try:
            branch_info = ""
            if isinstance(agent.strategy, BranchingStrategy):
                branch_info = f" [{agent.strategy.current_branch_id}]"
            prefix = f"[{agent.provider_name}|{current_strategy_num}]{branch_info}"
            raw = input(f"{prefix} Вы: ")
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

        if user_input.startswith("/"):
            current_strategy_num, handled = handle_command(
                user_input, agent, strategies, current_strategy_num
            )
            if handled:
                continue

        try:
            with spinner():
                reply = agent.ask(user_input)

            print(f"Агент: {reply}")

            usage = agent.last_token_usage
            if usage:
                print(
                    f"  [tokens: prompt={usage.history_tokens}, "
                    f"response={usage.response_tokens}, "
                    f"total={usage.total_tokens}]"
                )
            print()

        except ValueError as exc:
            print(f"Ошибка: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"Ошибка API: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chat",
        description="LLM-агент с переключением стратегий контекста и провайдеров.",
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=None,
        metavar="NAME",
        help=f"LLM-провайдер: {' | '.join(SUPPORTED_PROVIDERS)} "
             f"(по умолчанию: авто из LLM_PROVIDER или первый с ключом).",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="NAME",
        help="Название модели (переопределяет переменную окружения).",
    )
    parser.add_argument(
        "--list-providers",
        action="store_true",
        help="Показать доступные провайдеры и выйти.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_providers:
        print("\nДоступные провайдеры:")
        for info in get_available_providers():
            status = "✓ доступен" if info["available"] else f"✗ нет {info['key_var']}"
            print(f"  {info['provider']:8} — {info['label']:<30} {status}")
            print(f"             модель по умолчанию: {info['default_model']}")
        print()
        return

    provider = args.provider or current_provider_from_env()
    run_interactive(provider=provider, model=args.model)


if __name__ == "__main__":
    main()
