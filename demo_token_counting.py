"""Демонстрация подсчёта токенов в агенте с реальными запросами к OpenAI.

Запуск:
    pip install openai tiktoken
    export OPENAI_API_KEY=sk-...
    python demo_token_counting.py

Модель: gpt-3.5-turbo (контекст 4096 токенов — наименьший среди GPT-моделей).

Три сценария:
    1. Короткий диалог    — несколько ходов, токенов мало
    2. Длинный диалог     — токены накапливаются с каждым ходом
    3. Превышение лимита  — агент обнаруживает переполнение:
        a) ContextLimitError  — строгий режим, агент останавливается
        b) авто-обрезка       — агент удаляет старые ходы и продолжает

Токены считаются двумя способами:
    - До запроса:  tiktoken (точный, локально)
    - После запроса: usage из ответа API (официальные данные OpenAI)
"""

from __future__ import annotations

import os
import textwrap

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.models import ContextLimitError, TokenUsage
from llm_agent.infrastructure.openai_client import CONTEXT_LIMITS, OpenAIClient
from llm_agent.infrastructure.token_counter import TiktokenCounter

# ---------------------------------------------------------------------------
# Параметры модели
# ---------------------------------------------------------------------------

MODEL = "gpt-3.5-turbo"
CONTEXT_LIMIT = CONTEXT_LIMITS[MODEL]   # 4096 токенов
OVERFLOW_LIMIT = 800                    # маленький лимит для демо сценария 3

WIDTH = 72


# ---------------------------------------------------------------------------
# Вспомогательный вывод
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def subheader(text: str) -> None:
    print(f"\n  --- {text} ---")


def show_message(role: str, text: str) -> None:
    prefix = "  👤 User : " if role == "user" else "  🤖 GPT  : "
    indent = " " * len(prefix)
    lines = textwrap.wrap(text, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")


def print_usage(usage: TokenUsage | None, turn: int) -> None:
    if usage is None:
        return
    bar_full = 30
    filled = int(bar_full * usage.context_usage_percent / 100) if usage.context_limit else 0
    bar = "[" + "#" * filled + "." * (bar_full - filled) + "]"
    near = "  ⚠ БЛИЗКО К ЛИМИТУ" if usage.is_near_limit else ""
    over = "  ✗ ПРЕВЫШЕН ЛИМИТ" if usage.would_exceed_limit else ""
    limit_str = f"/{usage.context_limit}" if usage.context_limit else ""

    print(f"\n  Ход {turn} — токены (OpenAI API):")
    print(f"    Запрос (только текст)  : {usage.request_tokens:>5} токенов  (tiktoken)")
    print(f"    Весь контекст (prompt) : {usage.history_tokens:>5}{limit_str} токенов")
    print(f"    Ответ модели           : {usage.response_tokens:>5} токенов")
    print(f"    Итого                  : {usage.total_tokens:>5} токенов")
    if usage.context_limit:
        print(f"    Заполненность          : {usage.context_usage_percent:>5.1f}%  {bar}{near}{over}")


def make_agent(
    context_limit: int = CONTEXT_LIMIT,
    auto_truncate: bool = False,
    system_prompt: str = "Ты полезный ассистент. Отвечай кратко и по делу.",
    max_tokens: int | None = None,
) -> SimpleAgent:
    return SimpleAgent(
        llm_client=OpenAIClient(model=MODEL, max_tokens=max_tokens),
        system_prompt=system_prompt,
        token_counter=TiktokenCounter(model=MODEL),
        context_limit=context_limit,
        auto_truncate=auto_truncate,
    )


# ---------------------------------------------------------------------------
# Сценарий 1: Короткий диалог
# ---------------------------------------------------------------------------

def scenario_short() -> None:
    header("СЦЕНАРИЙ 1: Короткий диалог (3 хода)")
    print(f"  Модель: {MODEL}  |  Лимит: {CONTEXT_LIMIT} токенов")

    agent = make_agent()
    turns = [
        "Привет! Как тебя зовут?",
        "Сколько будет 2 + 2?",
        "Спасибо, пока!",
    ]

    for i, q in enumerate(turns, 1):
        subheader(f"Ход {i}")
        show_message("user", q)
        answer = agent.ask(q)
        show_message("assistant", answer)
        print_usage(agent.last_token_usage, i)

    u = agent.last_token_usage
    print(f"\n  Итог: {len(agent.history)} сообщений | "
          f"~{u.context_usage_percent:.1f}% контекста использовано")


# ---------------------------------------------------------------------------
# Сценарий 2: Длинный диалог — токены накапливаются
# ---------------------------------------------------------------------------

def scenario_long() -> None:
    header("СЦЕНАРИЙ 2: Длинный диалог (7 ходов)")
    print("  История передаётся ЦЕЛИКОМ при каждом ходе → токены нарастают.")
    print(f"  Лимит: {CONTEXT_LIMIT} токенов")

    agent = make_agent(
        system_prompt=(
            "Ты эксперт по машинному обучению. "
            "Давай подробные ответы на русском языке."
        ),
    )

    questions = [
        "Что такое машинное обучение? Объясни суть в 2–3 предложениях.",
        "Чем нейронная сеть отличается от обычного ML-алгоритма?",
        "Расскажи про механизм внимания (attention) в трансформерах.",
        "Что такое переобучение и три способа с ним бороться?",
        "Объясни разницу между обучением с учителем и без учителя.",
        "Какова роль функции потерь при обучении нейросети?",
        "Что такое батч-нормализация и зачем она нужна?",
    ]

    for i, q in enumerate(questions, 1):
        subheader(f"Ход {i}")
        show_message("user", q)
        answer = agent.ask(q)
        show_message("assistant", answer[:160] + ("…" if len(answer) > 160 else ""))
        print_usage(agent.last_token_usage, i)

    u = agent.last_token_usage
    print(f"\n  Итог: {len(agent.history)} сообщений | "
          f"~{u.context_usage_percent:.1f}% контекста использовано")


# ---------------------------------------------------------------------------
# Сценарий 3: Превышение лимита
# ---------------------------------------------------------------------------

def scenario_overflow() -> None:
    header("СЦЕНАРИЙ 3: Превышение лимита контекста")
    print(f"  Лимит: {OVERFLOW_LIMIT} токенов (намеренно маленький для демо)")
    print(f"  Ответы намеренно объёмные (max_tokens=200) для быстрого заполнения.")

    questions = [
        "Опиши архитектуру трансформера с основными компонентами.",
        "Как работает механизм multi-head attention? Опиши подробно.",
        "Что такое positional encoding и зачем он нужен в трансформере?",
        "Как обучают большие языковые модели (LLM)? Опиши этапы.",
    ]

    # --- 3a: ContextLimitError ---
    subheader("3a) auto_truncate=False → ContextLimitError при переполнении")
    agent_strict = make_agent(
        context_limit=OVERFLOW_LIMIT,
        auto_truncate=False,
        system_prompt="Давай подробные развёрнутые ответы на русском языке.",
        max_tokens=200,
    )

    overflow_turn = None
    for i, q in enumerate(questions, 1):
        try:
            subheader(f"Ход {i}")
            show_message("user", q)
            answer = agent_strict.ask(q)
            show_message("assistant", answer[:130] + ("…" if len(answer) > 130 else ""))
            print_usage(agent_strict.last_token_usage, i)
        except ContextLimitError as exc:
            overflow_turn = i
            print(f"\n  ✗ ContextLimitError на ходу {i}:")
            print(f"    {exc}")
            agent_strict._history.pop()  # откат незавершённого сообщения
            break

    if overflow_turn:
        print(f"\n  → Агент остановился на ходу {overflow_turn}.")
        print(f"    Нужно: очистить историю (agent.clear_history()) или")
        print(f"    переключить auto_truncate=True (сценарий 3b).")

    # --- 3b: авто-обрезка ---
    subheader("3b) auto_truncate=True → агент обрезает старые ходы и продолжает")
    agent_auto = make_agent(
        context_limit=OVERFLOW_LIMIT,
        auto_truncate=True,
        system_prompt="Давай подробные развёрнутые ответы на русском языке.",
        max_tokens=200,
    )

    for i, q in enumerate(questions, 1):
        subheader(f"Ход {i}")
        show_message("user", q)
        answer = agent_auto.ask(q)
        show_message("assistant", answer[:130] + ("…" if len(answer) > 130 else ""))
        usage = agent_auto.last_token_usage
        print_usage(usage, i)
        hist_len = len(agent_auto.history)
        full_len = i * 2
        if hist_len < full_len:
            print(f"    ✂  Обрезка: {hist_len} сообщ. в истории "
                  f"(без обрезки было бы {full_len})")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("\n  ОШИБКА: переменная OPENAI_API_KEY не задана.")
        print("  Установите её: export OPENAI_API_KEY=sk-...")
        print("  Или создайте файл .env с содержимым: OPENAI_API_KEY=sk-...")
        return

    print("\n" + "#" * WIDTH)
    print("  ДЕМОНСТРАЦИЯ ПОДСЧЁТА ТОКЕНОВ В LLM-АГЕНТЕ")
    print(f"  Модель : {MODEL}  (контекст {CONTEXT_LIMIT} токенов)")
    print(f"  Счётчик: tiktoken (до запроса) + OpenAI usage (после)")
    print("#" * WIDTH)

    try:
        scenario_short()
        scenario_long()
        scenario_overflow()
    except Exception as exc:
        print(f"\n  ОШИБКА: {exc}")
        raise

    print("\n" + "=" * WIDTH)
    print("  ВЫВОДЫ:")
    print("  • Токены нарастают — история передаётся целиком при каждом ходе")
    print("  • Короткий диалог:    < 5% контекста (< 200 токенов)")
    print("  • Длинный диалог:    ~40-70% после 7 ходов (зависит от ответов)")
    print("  • Переполнение 3a:   ContextLimitError — явная остановка агента")
    print("  • Переполнение 3b:   auto_truncate — старые ходы удаляются")
    print("  • tiktoken точен для GPT-моделей — можно использовать для мониторинга")
    print("=" * WIDTH + "\n")


if __name__ == "__main__":
    main()
