"""Демонстрация подсчёта токенов в агенте.

Запуск:
    python demo_token_counting.py

Три сценария:
    1. Короткий диалог    — токенов мало, контекст практически не занят
    2. Длинный диалог     — токены накапливаются с каждым ходом
    3. Превышение лимита  — агент обнаруживает переполнение и сообщает об этом,
                            а затем повторяет запрос с авто-обрезкой истории

Модель: gpt-3.5-turbo (контекст 4096 токенов).
Для запуска без API-ключа используется MockLLMClient с реалистичными ответами.
"""

from __future__ import annotations

import os
import textwrap

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.models import ChatMessage, ContextLimitError, LLMResponse, TokenUsage
from llm_agent.infrastructure.token_counter import TiktokenCounter

# ---------------------------------------------------------------------------
# Вспомогательные инструменты вывода
# ---------------------------------------------------------------------------

WIDTH = 72

def header(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def subheader(text: str) -> None:
    print(f"\n  --- {text} ---")


def print_usage(usage: TokenUsage | None, turn: int) -> None:
    if usage is None:
        return
    bar_full = 30
    filled = int(bar_full * usage.context_usage_percent / 100) if usage.context_limit else 0
    bar = "[" + "#" * filled + "." * (bar_full - filled) + "]"

    limit_str = f"/{usage.context_limit}" if usage.context_limit else ""
    near = " ⚠ БЛИЗКО К ЛИМИТУ" if usage.is_near_limit else ""
    over = " ✗ ПРЕВЫШЕН ЛИМИТ" if usage.would_exceed_limit else ""

    print(f"\n  Ход {turn} — статистика токенов:")
    print(f"    Текущий запрос   : {usage.request_tokens:>6} токенов")
    print(f"    Весь контекст    : {usage.history_tokens:>6}{limit_str} токенов")
    print(f"    Ответ модели     : {usage.response_tokens:>6} токенов")
    print(f"    Итого (вход+вых) : {usage.total_tokens:>6} токенов")
    if usage.context_limit:
        print(f"    Заполненность    : {usage.context_usage_percent:>5.1f}%  {bar}{near}{over}")


def show_message(role: str, text: str) -> None:
    prefix = "  👤 User  : " if role == "user" else "  🤖 Agent : "
    indent = " " * len(prefix)
    lines = textwrap.wrap(text, width=WIDTH - len(prefix))
    for i, line in enumerate(lines):
        print(f"{prefix if i == 0 else indent}{line}")


# ---------------------------------------------------------------------------
# Mock-клиент (работает без API-ключа, симулирует реалистичные ответы)
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Симулирует LLM с реалистичными ответами для демонстрации."""

    # Заготовленные ответы для коротких и длинных диалогов
    _SHORT_ANSWERS = [
        "Привет! Чем могу помочь?",
        "Конечно, 2 + 2 = 4.",
        "Отлично, рад был помочь!",
    ]
    _LONG_ANSWERS = [
        "Машинное обучение — это раздел искусственного интеллекта, "
        "в котором алгоритмы обучаются на данных без явного программирования правил.",

        "Нейронная сеть состоит из слоёв нейронов: входного, скрытых и выходного. "
        "Каждый нейрон вычисляет взвешенную сумму входов и применяет функцию активации.",

        "Обучение с подкреплением — это парадигма, в которой агент взаимодействует "
        "со средой, получает награды и штрафы, и учится максимизировать долгосрочное вознаграждение.",

        "Трансформеры используют механизм внимания (attention), чтобы учитывать "
        "все части входной последовательности при генерации каждого токена выходной.",

        "Переобучение (overfitting) — это явление, когда модель заучивает обучающие данные, "
        "но плохо обобщается на новые. Противодействие: регуляризация, дропаут, ранняя остановка.",

        "Градиентный спуск — оптимизационный алгоритм, который итерационно двигается "
        "в направлении антиградиента функции потерь, постепенно уменьшая ошибку модели.",

        "Свёрточные нейросети (CNN) особенно эффективны для обработки изображений: "
        "свёрточные слои извлекают пространственные признаки, а пулинг сокращает размерность.",
    ]

    def __init__(self, mode: str = "short") -> None:
        self._mode = mode
        self._call_count = 0

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        pool = self._SHORT_ANSWERS if self._mode == "short" else self._LONG_ANSWERS
        text = pool[self._call_count % len(pool)]
        self._call_count += 1
        return LLMResponse(text=text, model="mock-gpt-3.5-turbo", usage={})


# ---------------------------------------------------------------------------
# Сценарий 1: Короткий диалог
# ---------------------------------------------------------------------------

def scenario_short() -> None:
    header("СЦЕНАРИЙ 1: Короткий диалог (3 хода)")

    counter = TiktokenCounter(model="gpt-3.5-turbo")
    agent = SimpleAgent(
        llm_client=MockLLMClient(mode="short"),
        system_prompt="Ты лаконичный помощник.",
        token_counter=counter,
        context_limit=4096,
    )

    turns = [
        "Привет!",
        "Сколько будет 2 + 2?",
        "Спасибо, пока!",
    ]

    for i, question in enumerate(turns, start=1):
        subheader(f"Ход {i}")
        show_message("user", question)
        answer = agent.ask(question)
        show_message("assistant", answer)
        print_usage(agent.last_token_usage, i)

    print(f"\n  Итог: {len(agent.history)} сообщений в истории")


# ---------------------------------------------------------------------------
# Сценарий 2: Длинный диалог — токены накапливаются
# ---------------------------------------------------------------------------

def scenario_long() -> None:
    header("СЦЕНАРИЙ 2: Длинный диалог (7 ходов)")
    print("  Токены накапливаются с каждым ходом — видно рост заполненности контекста.")

    counter = TiktokenCounter(model="gpt-3.5-turbo")
    agent = SimpleAgent(
        llm_client=MockLLMClient(mode="long"),
        system_prompt="Ты эксперт по машинному обучению. Давай подробные ответы.",
        token_counter=counter,
        context_limit=4096,
    )

    questions = [
        "Что такое машинное обучение?",
        "Как устроена нейронная сеть?",
        "Расскажи об обучении с подкреплением.",
        "Чем знаменит механизм трансформера?",
        "Что такое переобучение и как с ним бороться?",
        "Объясни градиентный спуск.",
        "Какова роль свёрточных сетей в компьютерном зрении?",
    ]

    for i, question in enumerate(questions, start=1):
        subheader(f"Ход {i}")
        show_message("user", question)
        answer = agent.ask(question)
        show_message("assistant", answer[:120] + ("..." if len(answer) > 120 else ""))
        print_usage(agent.last_token_usage, i)

    print(f"\n  Итог: {len(agent.history)} сообщений в истории")


# ---------------------------------------------------------------------------
# Сценарий 3: Превышение лимита
# ---------------------------------------------------------------------------

# Один «большой» ответ (~120 токенов) для быстрого заполнения контекста
_BIG_ANSWER = (
    "Квантовые компьютеры используют кубиты, которые благодаря суперпозиции и запутанности "
    "способны выполнять вычисления, недоступные классическим машинам. Алгоритм Шора позволяет "
    "факторизовать большие числа за полиномиальное время, что угрожает современной криптографии. "
    "Алгоритм Гровера даёт квадратичное ускорение поиска в неструктурированных данных. "
    "Декогеренция остаётся главным инженерным препятствием на пути к практическим квантовым вычислениям."
)


class BigResponseMockClient:
    """Возвращает одинаково большие ответы для быстрого заполнения контекста."""

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        return LLMResponse(text=_BIG_ANSWER, model="mock-gpt-3.5-turbo", usage={})


def scenario_overflow() -> None:
    header("СЦЕНАРИЙ 3: Превышение лимита контекста")
    print("  Используем маленький лимит (600 токенов) для наглядной демонстрации.")
    print("  Показываем поведение агента при переполнении:")
    print("    a) ContextLimitError (auto_truncate=False)")
    print("    b) Авто-обрезка истории (auto_truncate=True)")

    DEMO_LIMIT = 600  # искусственно маленький лимит
    counter = TiktokenCounter(model="gpt-3.5-turbo")
    questions = [
        "Что такое квантовые компьютеры?",
        "Как работает алгоритм Шора?",
        "Какова роль запутанности в квантовых вычислениях?",
        "Что мешает построить практический квантовый компьютер?",
    ]

    # --- 3a: без авто-обрезки ---
    subheader("3a) auto_truncate=False — агент выбрасывает ContextLimitError")
    agent_strict = SimpleAgent(
        llm_client=BigResponseMockClient(),
        system_prompt="Ты эксперт по квантовым технологиям.",
        token_counter=counter,
        context_limit=DEMO_LIMIT,
        auto_truncate=False,
    )

    for i, question in enumerate(questions, start=1):
        try:
            subheader(f"Ход {i}")
            show_message("user", question)
            answer = agent_strict.ask(question)
            show_message("assistant", answer[:100] + "...")
            print_usage(agent_strict.last_token_usage, i)
        except ContextLimitError as exc:
            print(f"\n  ✗ ContextLimitError на ходу {i}:")
            print(f"    {exc}")
            # Показываем сколько было токенов в момент ошибки
            print(f"    Токены текущего запроса : {counter.count_tokens(question)}")
            # Восстанавливаем state (удаляем незавершённое сообщение пользователя)
            agent_strict._history.pop()
            break

    # --- 3b: с авто-обрезкой ---
    subheader("3b) auto_truncate=True — агент обрезает старую историю")
    agent_auto = SimpleAgent(
        llm_client=BigResponseMockClient(),
        system_prompt="Ты эксперт по квантовым технологиям.",
        token_counter=counter,
        context_limit=DEMO_LIMIT,
        auto_truncate=True,
    )

    for i, question in enumerate(questions, start=1):
        subheader(f"Ход {i}")
        show_message("user", question)
        answer = agent_auto.ask(question)
        show_message("assistant", answer[:100] + "...")
        usage = agent_auto.last_token_usage
        print_usage(usage, i)
        if usage:
            hist_len = len(agent_auto.history)
            print(f"    Сообщений в истории     : {hist_len} "
                  f"(обрезано при необходимости)")


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "#" * WIDTH)
    print("  ДЕМОНСТРАЦИЯ ПОДСЧЁТА ТОКЕНОВ В LLM-АГЕНТЕ")
    print(f"  Модель: gpt-3.5-turbo  |  Счётчик: tiktoken")
    print("#" * WIDTH)

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if api_key:
        print("\n  OPENAI_API_KEY найден — для реальных запросов замените")
        print("  MockLLMClient на OpenAIHttpClient в коде демо.")
    else:
        print("\n  OPENAI_API_KEY не задан → используется MockLLMClient.")
        print("  Токены считаются реальным tiktoken, ответы — симулированы.")

    scenario_short()
    scenario_long()
    scenario_overflow()

    print("\n" + "=" * WIDTH)
    print("  ВЫВОД:")
    print("  • Токены растут с каждым ходом — история всегда передаётся целиком")
    print("  • ContextLimitError позволяет явно контролировать переполнение")
    print("  • auto_truncate удаляет старые ходы, сохраняя диалог живым")
    print("  • Короткий диалог: < 1% контекста; длинный: ~15–30%; overflow: ~100%+")
    print("=" * WIDTH + "\n")


if __name__ == "__main__":
    main()
