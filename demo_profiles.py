#!/usr/bin/env python3
"""demo_profiles.py — демонстрация системы профилей пользователя.

Запуск: python demo_profiles.py

Скрипт без интерактивного ввода демонстрирует:
  Шаг 1 — Создание трёх профилей через --describe (LLM генерирует prompt)
  Шаг 2 — Один вопрос × три профиля (сравнение стилей ответов)
  Шаг 3 — Проверка ограничений (emoji / язык)
  Шаг 4 — Переключение профиля на лету
  Шаг 5 — Итоговая таблица
"""

from __future__ import annotations

import os
import sys
import textwrap
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.context_strategies import SlidingWindowStrategy
from llm_agent.application.strategy_agent import StrategyAgent
from llm_agent.domain.models import ChatMessage
from llm_agent.infrastructure.llm_factory import build_client, current_provider_from_env
from llm_agent.infrastructure.token_counter import TiktokenCounter
from llm_agent.memory.manager import MemoryManager
from llm_agent.memory.profile_manager import ProfileManager

# ---------------------------------------------------------------------------
# Параметры
# ---------------------------------------------------------------------------

MEMORY_DB = os.path.join(os.path.expanduser("~"), ".llm-agent", "demo_profiles.db")

BASE_SYSTEM_PROMPT = (
    "Ты полезный ассистент. Отвечай на русском языке."
)

PROFILE_DESCRIBES = {
    "expert": (
        "Опытный senior-разработчик. Без emoji. Без вводных фраз и воды. "
        "Пропускай очевидные объяснения. Отвечай кратко и точно. "
        "Предпочитай код, а не описание словами."
    ),
    "student": (
        "Начинающий разработчик изучает Python после PHP. Объясняй на русском. "
        "Используй аналогии с реальной жизнью. Объясняй каждый новый термин. "
        "Показывай пошаговые примеры. Ограничивай каждый ответ 100-150 словами."
    ),
    "rubber_duck": (
        "Режим резиновой уточки для отладки мышления. Никогда не давай готовых решений. "
        "Только задавай уточняющие вопросы, которые помогают пользователю самому найти ответ."
    ),
}

PROFILE_DISPLAY_NAMES = {
    "expert": "Senior Developer",
    "student": "Начинающий разработчик",
    "rubber_duck": "Резиновая уточка",
}

QUESTION_1 = "В чём разница между list и tuple в Python?"
QUESTION_2 = "Конечно! 😊 Расскажи о декораторах в Python."
QUESTION_3 = "У меня баг в коде. Функция возвращает None вместо ожидаемого значения."


# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def hr(char: str = "═", width: int = 70) -> None:
    print(char * width)


def section(title: str) -> None:
    print()
    hr()
    print(f"  {title}")
    hr()
    print()


def wrap_text(text: str, width: int = 68, indent: str = "  ") -> str:
    lines = text.splitlines()
    wrapped: list[str] = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            for chunk in textwrap.wrap(line, width=width):
                wrapped.append(indent + chunk)
    return "\n".join(wrapped)


def generate_system_prompt(llm_client, description: str) -> str:
    """Сгенерировать system prompt через LLM."""
    messages = [
        ChatMessage(
            role="user",
            content=(
                "Create a system prompt for an AI assistant based on this description. "
                "Return only the system prompt text, no explanations, no quotes.\n\n"
                f"Description: {description}"
            ),
        )
    ]
    response = llm_client.generate(messages)
    return response.text.strip()


def ask_with_profile(
    agent: StrategyAgent,
    profile_manager: ProfileManager,
    profile_name: str,
    question: str,
) -> tuple[str, str]:
    """Активировать профиль, задать вопрос, вернуть (ответ, итоговый_system_prompt)."""
    profile_manager.set_active(profile_name)
    agent.clear_history()

    # Собираем итоговый system prompt для отображения
    ctx = agent.memory_manager.get_context_for_llm() if agent.memory_manager else {}
    effective_prompt = profile_manager.build_system_prompt(
        base_prompt=BASE_SYSTEM_PROMPT,
        long_term_text=ctx.get("long_term_text", ""),
        working_text=ctx.get("working_text", ""),
    )

    for attempt in range(3):
        try:
            reply = agent.ask(question)
            return reply, effective_prompt
        except Exception as exc:
            if attempt < 2:
                print(f"  [таймаут, повтор {attempt + 2}/3...]")
                time.sleep(3)
                agent.clear_history()
            else:
                raise exc
    raise RuntimeError("unreachable")


def detect_language(text: str) -> str:
    """Простое определение языка по доле кириллицы."""
    cyrillic = sum(1 for c in text if "\u0400" <= c <= "\u04ff")
    total = len([c for c in text if c.isalpha()])
    if total == 0:
        return "—"
    ratio = cyrillic / total
    if ratio > 0.5:
        return "RU"
    elif ratio > 0.1:
        return "RU/EN"
    return "EN"


def has_emoji(text: str) -> bool:
    """Проверить наличие emoji в тексте."""
    for char in text:
        cp = ord(char)
        if (
            0x1F600 <= cp <= 0x1F64F
            or 0x1F300 <= cp <= 0x1F5FF
            or 0x1F680 <= cp <= 0x1F6FF
            or 0x2600 <= cp <= 0x26FF
            or 0x2700 <= cp <= 0x27BF
            or 0x1F900 <= cp <= 0x1F9FF
        ):
            return True
    return False


def has_code_block(text: str) -> bool:
    return "```" in text or "    " in text


# ---------------------------------------------------------------------------
# Шаги демонстрации
# ---------------------------------------------------------------------------

def step1_create_profiles(profile_manager: ProfileManager, llm_client) -> None:
    """Шаг 1: Создать три профиля через --describe."""
    section("ШАГ 1: Создание профилей через --describe (LLM генерирует prompt)")

    for name, description in PROFILE_DESCRIBES.items():
        display_name = PROFILE_DISPLAY_NAMES[name]

        # Удалить существующий (если есть от предыдущего запуска)
        existing = profile_manager.get(name)
        if existing:
            if existing.is_active:
                profile_manager.deactivate_all()
            profile_manager.delete(name)

        print(f"  Создаю профиль '{name}' ({display_name})...")
        print(f"  Описание: {description}")
        system_prompt = generate_system_prompt(llm_client, description)
        profile_manager.create(name, display_name, system_prompt)
        print(f"\n  Сгенерированный system prompt:")
        print(wrap_text(system_prompt))
        print()
        hr("-")

    # Список профилей
    print("\n  /profiles — список всех профилей:")
    for p in profile_manager.list_all():
        print(f"    [{p.name}] {p.display_name}")
    print()


def step2_one_question_three_profiles(
    agent: StrategyAgent,
    profile_manager: ProfileManager,
) -> dict[str, dict]:
    """Шаг 2: Один вопрос × три профиля."""
    section(f"ШАГ 2: Один вопрос — три профиля\n  Вопрос: «{QUESTION_1}»")

    results: dict[str, dict] = {}

    comments = {
        "expert": (
            "→ Прямо, технически, без вступлений. Только факты и, возможно, код."
        ),
        "student": (
            "→ На русском, с объяснениями терминов, возможно с аналогиями."
        ),
        "rubber_duck": (
            "→ Никаких объяснений! Только уточняющие вопросы."
        ),
    }

    for name in ["expert", "student", "rubber_duck"]:
        print(f"  ── Профиль: {name} ({PROFILE_DISPLAY_NAMES[name]}) ──")
        reply, system_prompt = ask_with_profile(agent, profile_manager, name, QUESTION_1)

        print(f"\n  System prompt (итоговый):")
        # Показываем только блок профиля для краткости
        profile = profile_manager.get(name)
        print(wrap_text(profile.system_prompt if profile else ""))
        print(f"\n  Ответ агента:")
        print(wrap_text(reply))
        print(f"\n  {comments[name]}")
        print()

        results[name] = {
            "reply": reply,
            "system_prompt": system_prompt,
            "length": len(reply),
        }

        hr("-")

    return results


def step3_check_constraints(agent: StrategyAgent, profile_manager: ProfileManager) -> None:
    """Шаг 3: Проверка ограничений."""
    section(f"ШАГ 3: Проверка ограничений\n  Вопрос: «{QUESTION_2}»")

    for name, expectation in [
        ("expert", "нет emoji, краткий ответ с кодом"),
        ("student", "русский язык, аналогии, объяснения"),
    ]:
        print(f"  ── Профиль: {name} — ожидаем: {expectation} ──")
        reply, _ = ask_with_profile(agent, profile_manager, name, QUESTION_2)

        emoji_found = has_emoji(reply)
        lang = detect_language(reply)
        code_found = has_code_block(reply)

        print(f"\n  Ответ агента:")
        print(wrap_text(reply))
        print()
        print(f"  Анализ:")
        print(f"    Язык:   {lang}")
        print(f"    Emoji:  {'есть' if emoji_found else 'нет'}")
        print(f"    Код:    {'есть' if code_found else 'нет'}")

        if name == "expert":
            ok = not emoji_found
            print(f"    Ограничение 'без emoji': {'✓ соблюдено' if ok else '✗ нарушено'}")
        elif name == "student":
            ok = lang in ("RU", "RU/EN")
            print(f"    Ограничение 'русский язык': {'✓ соблюдено' if ok else '✗ нарушено'}")

        print()
        hr("-")


def step4_switch_on_the_fly(agent: StrategyAgent, profile_manager: ProfileManager) -> None:
    """Шаг 4: Переключение профиля на лету."""
    section("ШАГ 4: Переключение профиля на лету")

    print(f"  Вопрос с профилем 'expert':")
    print(f"  «{QUESTION_3}»\n")
    reply_expert, _ = ask_with_profile(agent, profile_manager, "expert", QUESTION_3)
    print(f"  [expert] Ответ:")
    print(wrap_text(reply_expert))
    print()

    print(f"  Переключаемся на 'rubber_duck'...")
    print(f"  Follow-up: «{QUESTION_3}»\n")
    reply_duck, _ = ask_with_profile(agent, profile_manager, "rubber_duck", QUESTION_3)
    print(f"  [rubber_duck] Ответ:")
    print(wrap_text(reply_duck))
    print()

    print("  Вывод: expert даёт готовый анализ/решение, rubber_duck — только вопросы.")
    print()


def step5_summary_table(results: dict[str, dict]) -> None:
    """Шаг 5: Итоговая таблица."""
    section("ШАГ 5: Итоговый отчёт")

    print(f"  {'Профиль':<15} {'Длина':>6}  {'Язык':>6}  {'Есть код':>9}  {'Emoji':>6}")
    hr("-", 60)
    for name, data in results.items():
        reply = data["reply"]
        lang = detect_language(reply)
        code = "да" if has_code_block(reply) else "нет"
        emoji = "да" if has_emoji(reply) else "нет"
        length = len(reply)
        print(f"  {name:<15} {length:>6}  {lang:>6}  {code:>9}  {emoji:>6}")
    print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    provider = current_provider_from_env()
    print(f"\nИспользуем провайдер: {provider}")

    try:
        client = build_client(provider, timeout=90.0)
    except ValueError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)

    token_counter = TiktokenCounter()
    memory_manager = MemoryManager(MEMORY_DB)
    profile_manager = ProfileManager(MEMORY_DB)

    strategy = SlidingWindowStrategy(window_size=10)
    agent = StrategyAgent(
        llm_client=client,
        strategy=strategy,
        system_prompt=BASE_SYSTEM_PROMPT,
        token_counter=token_counter,
        provider_name=provider,
        memory_manager=memory_manager,
        profile_manager=profile_manager,
    )

    try:
        # Шаг 1: Создать профили
        step1_create_profiles(profile_manager, client)

        # Шаг 2: Один вопрос × три профиля
        results = step2_one_question_three_profiles(agent, profile_manager)

        # Шаг 3: Проверка ограничений
        step3_check_constraints(agent, profile_manager)

        # Шаг 4: Переключение на лету
        step4_switch_on_the_fly(agent, profile_manager)

        # Шаг 5: Итоговая таблица
        step5_summary_table(results)

    finally:
        memory_manager.close()
        profile_manager.close()

    print("Демонстрация завершена.")
    print(f"Профили сохранены в: {MEMORY_DB}")
    print("Для работы в интерактивном режиме: python chat.py")


if __name__ == "__main__":
    main()
