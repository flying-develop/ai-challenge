"""Демонстрация трёхуровневой памяти (Memory Layers).

Скрипт БЕЗ интерактивного ввода показывает, как три слоя памяти
(short-term, working, long-term) влияют на ответы LLM.

Сценарий:
    1. Сохраняет в long_term факт: «100 лет квантовой механике»
    2. Сохраняет в working задачу: «Обоснование возможности кротовых нор»
    3. Ведёт диалог из 3-4 turn-ов на тему чёрных дыр
    4. После каждого turn-а выводит дамп всех слоёв памяти
    5. Показывает, как long_term и working влияют на ответы
    6. Промоутит факт из working в long_term
    7. Очищает working и показывает изменение ответов

Запуск:
    python demo_memory.py                        # авто-провайдер
    python demo_memory.py --provider qwen
    python demo_memory.py --provider openai --model gpt-4o-mini
    python demo_memory.py --list-providers
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import textwrap
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import SlidingWindowStrategy
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
from llm_agent.memory.manager import MemoryManager

# ===========================================================================
# Параметры
# ===========================================================================

WIDTH = 78

SYSTEM_PROMPT = (
    "Ты — эрудированный научный ассистент. "
    "Давай точные, содержательные ответы на русском языке. "
    "Обязательно используй контекст из своей памяти (если он есть) при ответе."
)

DIALOG_QUESTIONS = [
    "Что такое чёрные дыры и как они образуются?",
    "Как связаны чёрные дыры и квантовая механика?",
    "Можно ли использовать чёрные дыры для путешествий через кротовые норы?",
    "Подведи итог: какие ключевые факты ты знаешь из нашего разговора и из своей памяти?",
]

QUESTION_AFTER_CLEANUP = (
    "Какова моя текущая задача? Какие факты ты помнишь о моих интересах?"
)


# ===========================================================================
# Вспомогательный вывод
# ===========================================================================

def sep(char: str = "=", width: int = WIDTH) -> None:
    print(char * width)


def header(title: str) -> None:
    print()
    sep()
    print(f"  {title}")
    sep()


def show_q(text: str) -> None:
    prefix = "  Q: "
    indent = " " * len(prefix)
    for i, line in enumerate(textwrap.wrap(text, width=WIDTH - len(prefix))):
        print(f"{prefix if i == 0 else indent}{line}")


def show_a(text: str, max_chars: int = 400) -> None:
    short = text[:max_chars] + ("..." if len(text) > max_chars else "")
    prefix = "  A: "
    indent = " " * len(prefix)
    for i, line in enumerate(textwrap.wrap(short, width=WIDTH - len(prefix))):
        print(f"{prefix if i == 0 else indent}{line}")


def show_tokens(agent: StrategyAgent) -> None:
    usage = agent.last_token_usage
    if usage:
        print(
            f"  [tokens: prompt={usage.history_tokens}, "
            f"response={usage.response_tokens}, "
            f"total={usage.total_tokens}]"
        )


def dump_memory(mm: MemoryManager) -> None:
    """Вывести содержимое всех слоёв памяти."""
    print()
    print("  ┌─────────────────────────────────────────────────────────┐")
    print("  │                   ДАМП ПАМЯТИ (/memory)                │")
    print("  └─────────────────────────────────────────────────────────┘")

    # Short-term
    short = mm.get_short_term()
    print(f"\n  SHORT-TERM ({len(short)} записей):")
    if not short:
        print("    (пусто)")
    else:
        for e in short[-6:]:  # последние 6 для краткости
            content_short = e.content[:60] + ("..." if len(e.content) > 60 else "")
            print(f"    [{e.role}] {content_short}")
        if len(short) > 6:
            print(f"    ... и ещё {len(short) - 6} записей")

    # Working
    working = mm.get_working()
    print(f"\n  WORKING ({len(working)} записей):")
    if not working:
        print("    (пусто)")
    else:
        for e in working:
            print(f"    [id={e.id}] {e.key}: {e.value}")

    # Long-term
    long = mm.get_long_term()
    print(f"\n  LONG-TERM ({len(long)} записей):")
    if not long:
        print("    (пусто)")
    else:
        for e in long:
            tags = f"  tags={e.tags}" if e.tags else ""
            print(f"    [id={e.id}] {e.key}: {e.value}{tags}")

    print()
    sep("-")


# ===========================================================================
# Основной сценарий
# ===========================================================================

def run_demo(provider: str, model: str) -> None:
    """Прогнать демо-сценарий Memory Layers."""

    client = build_client(provider, model=model)
    token_counter = TiktokenCounter()

    # Используем временную БД, чтобы не мешать основной
    tmp_dir = tempfile.mkdtemp(prefix="llm_memory_demo_")
    db_path = os.path.join(tmp_dir, "demo_memory.db")

    with MemoryManager(db_path) as mm:
        strategy = SlidingWindowStrategy(window_size=10)
        agent = StrategyAgent(
            llm_client=client,
            strategy=strategy,
            system_prompt=SYSTEM_PROMPT,
            token_counter=token_counter,
            provider_name=provider,
            model_name=model,
            memory_manager=mm,
        )

        # ==============================================================
        # ШАГ 1: Наполняем долговременную память
        # ==============================================================
        header("ШАГ 1: Наполняем LONG-TERM память")
        lt_id = mm.add_to_long(
            key="тема_года",
            value="В 2025 году исполняется 100 лет квантовой механике — "
                  "юбилей, который стоит учитывать в научных обсуждениях.",
            tags=["наука", "квантовая_механика"],
        )
        print(f"  Добавлено в LONG-TERM [id={lt_id}]:")
        print(f"    тема_года: 100 лет квантовой механике")

        # ==============================================================
        # ШАГ 2: Наполняем рабочую память
        # ==============================================================
        header("ШАГ 2: Наполняем WORKING память")
        w1_id = mm.add_to_working(
            key="цель",
            value="Обоснование теоретической возможности кротовых нор "
                  "(wormholes) на основе ОТО и квантовой гравитации.",
        )
        w2_id = mm.add_to_working(
            key="контекст",
            value="Нужно связать чёрные дыры, квантовую механику и "
                  "кротовые норы в единую картину.",
        )
        print(f"  Добавлено в WORKING [id={w1_id}]:")
        print(f"    цель: Обоснование возможности кротовых нор")
        print(f"  Добавлено в WORKING [id={w2_id}]:")
        print(f"    контекст: Связь чёрных дыр, КМ и кротовых нор")

        dump_memory(mm)

        # ==============================================================
        # ШАГ 3: Диалог — 4 turn-а
        # ==============================================================
        header("ШАГ 3: Диалог о чёрных дырах (4 turn-а)")
        print("  Память (working + long-term) включена в system prompt.\n")

        for i, q in enumerate(DIALOG_QUESTIONS, 1):
            print(f"  ── Ход {i}/{len(DIALOG_QUESTIONS)} ──")
            show_q(q)
            try:
                answer = agent.ask(q)
            except Exception as exc:
                print(f"  ОШИБКА: {exc}")
                break
            show_a(answer)
            show_tokens(agent)
            dump_memory(mm)
            time.sleep(0.5)

        # ==============================================================
        # ШАГ 4: Promote — перемещаем факт из working в long-term
        # ==============================================================
        header("ШАГ 4: /promote working → long-term")
        print(f"  Перемещаем working#{w1_id} (цель) в долговременную память...")
        try:
            new_lt_id = mm.promote("working", w1_id)
            print(f"  Запись working#{w1_id} → long-term#{new_lt_id}")
        except ValueError as e:
            print(f"  Ошибка: {e}")

        dump_memory(mm)

        # ==============================================================
        # ШАГ 5: Очистка working и проверка
        # ==============================================================
        header("ШАГ 5: Очищаем WORKING и проверяем влияние на ответ")
        removed = mm.remove_from_working()
        print(f"  Удалено из WORKING: {removed} запись(ей).")
        print(f"  WORKING теперь пуста, но LONG-TERM по-прежнему содержит факты.\n")

        dump_memory(mm)

        print("  ── Вопрос после очистки working ──")
        show_q(QUESTION_AFTER_CLEANUP)
        try:
            answer = agent.ask(QUESTION_AFTER_CLEANUP)
        except Exception as exc:
            print(f"  ОШИБКА: {exc}")
            answer = str(exc)
        show_a(answer, max_chars=500)
        show_tokens(agent)

        # ==============================================================
        # ШАГ 6: Итоговая статистика
        # ==============================================================
        header("ИТОГОВАЯ СТАТИСТИКА")
        stats = agent.get_stats()
        for k, v in stats.items():
            if isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    print(f"    {kk}: {vv}")
            else:
                print(f"  {k}: {v}")

        dump_memory(mm)

        # ==============================================================
        # Выводы
        # ==============================================================
        header("ВЫВОДЫ")
        print("""
  1. LONG-TERM память (долговременная):
     - Факт «100 лет квантовой механике» был доступен LLM с первого turn-а
     - Не очищается при /clear — сохраняется навсегда
     - Подходит для: предпочтений пользователя, архитектурных решений

  2. WORKING память (рабочая):
     - Задача «обоснование кротовых нор» направляла ответы LLM
     - После очистки LLM перестал учитывать эту задачу
     - Подходит для: текущих целей, промежуточных результатов

  3. SHORT-TERM память (краткосрочная):
     - Автоматически пополнялась каждым turn-ом
     - Управляется стратегией (Sliding Window)
     - Очищается при /clear

  4. /promote — позволяет «закрепить» рабочий факт навсегда
""")
        sep()

    if hasattr(client, "close"):
        client.close()

    print(f"\n  Временная БД: {db_path}")
    print(f"  Для очистки: rm -rf {tmp_dir}\n")


# ===========================================================================
# Точка входа
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo_memory",
        description="Демонстрация трёхуровневой памяти (Memory Layers).",
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
    model = args.model or DEFAULT_MODELS[provider]

    print("\n" + "#" * WIDTH)
    print("  ДЕМОНСТРАЦИЯ: ТРЁХУРОВНЕВАЯ ПАМЯТЬ (Memory Layers)")
    print(f"  Провайдер: {provider}  ({PROVIDER_LABELS.get(provider, '')})")
    print(f"  Модель: {model}")
    print("#" * WIDTH)

    try:
        run_demo(provider=provider, model=model)
    except KeyboardInterrupt:
        print("\n\n  Прервано пользователем.")
    except Exception as exc:
        print(f"\n  КРИТИЧЕСКАЯ ОШИБКА: {exc}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
