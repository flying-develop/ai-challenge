#!/usr/bin/env python3
"""demo_invariants.py — демонстрация системы инвариантов.

Запуск: python demo_invariants.py

Скрипт демонстрирует:
  Сценарий 1 — Загрузка и отображение инвариантов
  Сценарий 2 — Конфликт со стеком (Redis вместо SQLite)
  Сценарий 3 — Конфликт с архитектурой (FastAPI вместо CLI)
  Сценарий 4 — Мягкий инвариант (библиотека rich)
  Сценарий 5 — Допустимый запрос (новая команда /stats)
  Сценарий 6 — Hot-reload (добавление нового инварианта)
"""

from __future__ import annotations

import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import SlidingWindowStrategy
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.core.invariant_loader import InvariantLoader
from llm_agent.infrastructure.llm_factory import build_client, current_provider_from_env
from llm_agent.infrastructure.token_counter import TiktokenCounter

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# Демо использует зафиксированные инварианты из demo_data/ (не config/).
# config/invariants/ — рабочие инварианты пользователя, демо их не трогает.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_DATA_DIR = os.path.join(_SCRIPT_DIR, "demo_data")

# Временный файл инварианта для демонстрации hot-reload
_DEMO_INVARIANT_FILE = os.path.join(DEMO_DATA_DIR, "invariants", "_demo_hotreload.md")

# ---------------------------------------------------------------------------
# Сценарии для демо
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "name": "Конфликт со стеком",
        "input": "Давай добавим Redis для кэширования контекста диалога",
        "expected": "отказ с указанием на инвариант SQLite-only",
    },
    {
        "name": "Конфликт с архитектурой",
        "input": "Перепишем CLI на FastAPI и сделаем веб-интерфейс с REST API",
        "expected": "отказ с указанием на инвариант CLI-only",
    },
    {
        "name": "Мягкий инвариант",
        "input": "Добавим библиотеку rich для красивого вывода таблиц в терминале",
        "expected": "предупреждение о предпочтении stdlib, но выполнение",
    },
    {
        "name": "Допустимый запрос",
        "input": "Добавим новую команду /stats для показа статистики диалогов из SQLite",
        "expected": "нормальное выполнение, инварианты не нарушены",
    },
]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def banner(text: str) -> None:
    print(f"\n{'=' * 66}")
    print(f"  {text}")
    print(f"{'=' * 66}\n")


def scenario_header(num: int, name: str, expected: str) -> None:
    print(f"\n{'─' * 66}")
    print(f"  Сценарий {num}: {name}")
    print(f"  Ожидается: {expected}")
    print(f"{'─' * 66}\n")


def ask_and_show(agent: StrategyAgent, prompt: str) -> str:
    """Задать вопрос агенту и показать ответ (первые 600 символов)."""
    print(f"  >> Пользователь: {prompt}")
    reply = agent.ask(prompt)
    preview = reply[:600] + ("..." if len(reply) > 600 else "")
    print(f"  << Агент: {preview}")
    print()
    return reply


def check_invariants(loader: InvariantLoader, text: str, agent: StrategyAgent) -> None:
    """Проверить текст через LLM на соответствие инвариантам."""
    from llm_agent.domain.models import ChatMessage

    inv_block = loader.build_prompt_block()
    check_prompt = (
        f"Проверь следующий запрос на соответствие инвариантам проекта.\n\n"
        f"{inv_block}\n\n"
        f"Запрос:\n{text}\n\n"
        f"Если есть нарушения обязательных инвариантов, укажи:\n"
        f"⛔ Конфликт с инвариантом: [категория] → [правило]\n"
        f"Причина ограничения: [объяснение]\n"
        f"Альтернативное решение: [предложение]\n\n"
        f"Если нарушений нет, напиши: '✅ Запрос не нарушает инварианты.'"
    )
    messages = [ChatMessage(role="user", content=check_prompt)]
    response = agent._llm_client.generate(messages)
    print(f"  Результат проверки:\n  {response.text.strip()}\n")


def cleanup_demo_file() -> None:
    if os.path.exists(_DEMO_INVARIANT_FILE):
        os.remove(_DEMO_INVARIANT_FILE)


# ---------------------------------------------------------------------------
# Основное демо
# ---------------------------------------------------------------------------


def main() -> None:
    banner("demo_invariants.py — система инвариантов")

    # Инициализация провайдера
    provider = current_provider_from_env()
    try:
        client = build_client(provider)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        print("Проверьте .env файл с ключами API.", file=sys.stderr)
        sys.exit(1)

    print(f"  Провайдер: {provider}")
    print(f"  Demo data: {DEMO_DATA_DIR}\n")

    # Загрузчик инвариантов — только demo_data/, config/ не трогаем
    loader = InvariantLoader(DEMO_DATA_DIR)
    cats = loader.categories
    total_req = sum(len(c.required) for c in cats)
    total_rec = sum(len(c.recommended) for c in cats)

    # -----------------------------------------------------------------------
    # Сценарий 0: Отображение инвариантов
    # -----------------------------------------------------------------------

    scenario_header(0, "Загрузка и отображение инвариантов", "все инварианты отображены")

    print(f"  Загружено: {len(cats)} категорий, "
          f"{total_req} обязательных, {total_rec} рекомендуемых\n")

    for cat in cats:
        print(f"  [{cat.name}] {cat.title}")
        for rule in cat.required:
            print(f"    ⛔ (обяз.) {rule}")
        for rule in cat.recommended:
            print(f"    ⚠  (рек.)  {rule}")
    print()

    print("  Блок для system prompt:")
    print("  " + "-" * 60)
    for line in loader.build_prompt_block().splitlines():
        print(f"  {line}")
    print("  " + "-" * 60)

    # Создаём агент с инвариантами
    token_counter = TiktokenCounter()
    strategy = SlidingWindowStrategy(window_size=8)
    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt="Ты полезный ассистент. Отвечай на русском языке.",
        token_counter=token_counter,
        provider_name=provider,
        invariant_loader=loader,
    )

    # -----------------------------------------------------------------------
    # Сценарии 1–4: запросы к агенту
    # -----------------------------------------------------------------------

    for i, scenario in enumerate(SCENARIOS, start=1):
        scenario_header(i, scenario["name"], scenario["expected"])

        # Показываем быструю проверку через прямой вызов check_invariants
        print("  [Прямая проверка без LLM-диалога]")
        check_invariants(loader, scenario["input"], agent)
        time.sleep(0.5)

        # Отправляем запрос агенту (инварианты в system prompt)
        print("  [Ответ агента (инварианты в system prompt)]")
        try:
            ask_and_show(agent, scenario["input"])
        except Exception as exc:
            print(f"  Ошибка: {exc}")

        time.sleep(0.3)

    # -----------------------------------------------------------------------
    # Сценарий 5: Hot-reload
    # -----------------------------------------------------------------------

    scenario_header(
        5,
        "Hot-reload",
        "новый инвариант подхвачен без перезапуска",
    )

    cleanup_demo_file()

    # Создаём новый MD-файл инварианта
    demo_invariant_content = """# Демо-инвариант (горячая перезагрузка)

## Обязательные

- Никаких emoji в именах переменных Python

## Рекомендуемые

- Использовать snake_case для всех идентификаторов
"""

    before_cats = len(loader.categories)
    before_req = sum(len(c.required) for c in loader.categories)

    print(f"  До hot-reload: {before_cats} категорий, {before_req} обязательных")
    print(f"\n  Добавляем файл: {_DEMO_INVARIANT_FILE}")
    print(f"  Содержимое:\n")
    for line in demo_invariant_content.strip().splitlines():
        print(f"    {line}")
    print()

    with open(_DEMO_INVARIANT_FILE, "w", encoding="utf-8") as f:
        f.write(demo_invariant_content)

    # Горячая перезагрузка
    reloaded = loader.reload()
    after_req = sum(len(c.required) for c in reloaded)

    print(f"  После hot-reload: {len(reloaded)} категорий, {after_req} обязательных")

    # Найдём новую категорию
    new_cat = next(
        (c for c in reloaded if c.name == "_demo_hotreload"), None
    )
    if new_cat:
        print(f"\n  Новая категория подхвачена: [{new_cat.name}] {new_cat.title}")
        for rule in new_cat.required:
            print(f"    ⛔ {rule}")
        for rule in new_cat.recommended:
            print(f"    ⚠  {rule}")
        print()
    else:
        print("  ОШИБКА: новая категория не найдена!")

    print("  Проверяем, что новый инвариант попал в system prompt...")
    inv_block = loader.build_prompt_block()
    if "emoji" in inv_block.lower() or "Никаких emoji" in inv_block:
        print("  ✅ Новый инвариант присутствует в блоке system prompt!")
    else:
        print("  ✗ Новый инвариант НЕ найден в блоке system prompt.")

    # Удаляем демо-файл
    cleanup_demo_file()
    print(f"\n  Демо-файл удалён: {_DEMO_INVARIANT_FILE}")

    # -----------------------------------------------------------------------
    # Итог
    # -----------------------------------------------------------------------

    banner("Демо завершено успешно!")
    print("  Что было показано:")
    print("  1. Загрузка инвариантов из config/invariants/*.md")
    print("  2. Парсинг MD без markdown-библиотек (regex + split)")
    print("  3. Формирование блока <INVARIANTS> для system prompt")
    print("  4. Конфликты с обязательными инвариантами → отказ LLM")
    print("  5. Мягкие инварианты → предупреждение, но выполнение")
    print("  6. Hot-reload без перезапуска агента")
    print()
    print("  Команды в интерактивном режиме (python chat.py):")
    print("    /invariants              — показать все инварианты")
    print("    /invariants reload       — перечитать файлы")
    print("    /invariants check <текст> — проверить через LLM")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем.")
        sys.exit(0)
    finally:
        # Гарантированная очистка демо-файла
        cleanup_demo_file()
