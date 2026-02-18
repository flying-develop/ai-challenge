"""
Сравнение четырёх стратегий промптинга на одной задаче.

Задача (парадокс дней рождения):
    В комнате находятся 23 человека. Какова вероятность (в процентах,
    округлённая до целого), что хотя бы двое из них родились в один день?
    Считать, что год невисокосный, а рождаемость равномерно распределена.

Четыре стратегии:
    1. Прямой ответ без дополнительных инструкций.
    2. «Решай пошагово».
    3. Сначала составить оптимальный промпт, затем использовать его.
    4. Группа экспертов: аналитик, математик, критик.
"""

import sys
import time
import textwrap

from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from os import environ

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = environ.get("API_URL", "https://api.openai.com/v1").removesuffix("/chat/completions")
API_MODEL = environ.get("API_MODEL", "gpt-4")
API_TOKEN = environ.get("API_TOKEN")

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]

# ─── Задача ───────────────────────────────────────────────────────────────────

TASK = (
    "В комнате находятся 23 человека. "
    "Какова вероятность (в процентах, округлённая до целого числа), "
    "что хотя бы двое из них родились в один и тот же день года? "
    "Считайте, что год невисокосный (365 дней) "
    "и рождаемость равномерно распределена по всем дням."
)

# ─── API helper ───────────────────────────────────────────────────────────────


def create_client() -> OpenAI:
    return OpenAI(base_url=API_URL, api_key=API_TOKEN)


def call_api(client: OpenAI, messages: list[dict], label: str) -> str:
    """
    Отправляет запрос в API (без стриминга) и возвращает текст ответа.
    Повторяет при RateLimitError с экспоненциальной задержкой.
    """
    print(f"  [отправляю запрос: {label}]", flush=True)

    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            last_exc = e
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF[attempt]
            print(f"  Rate limit (429). Повтор через {wait}с...", flush=True)
            time.sleep(wait)
        except (APIError, APIConnectionError):
            raise

    raise last_exc  # type: ignore[misc]


# ─── Четыре стратегии ─────────────────────────────────────────────────────────


def strategy_1_direct(client: OpenAI) -> str:
    """Прямой вопрос без системных инструкций."""
    messages = [{"role": "user", "content": TASK}]
    return call_api(client, messages, "прямой запрос")


def strategy_2_step_by_step(client: OpenAI) -> str:
    """Системная инструкция: решай пошагово."""
    messages = [
        {
            "role": "system",
            "content": "Решай задачу пошагово, объясняя каждый шаг.",
        },
        {"role": "user", "content": TASK},
    ]
    return call_api(client, messages, "пошаговое решение")


def strategy_3_meta_prompt(client: OpenAI) -> str:
    """
    Шаг A — просим модель составить оптимальный промпт.
    Шаг B — используем этот промпт для решения задачи.
    """
    # Шаг A
    meta_request = [
        {
            "role": "system",
            "content": (
                "Ты — эксперт по prompt engineering. "
                "Составь оптимальный системный промпт для языковой модели, "
                "чтобы та максимально точно и ясно решала вероятностные задачи. "
                "Верни только текст промпта, без пояснений."
            ),
        },
        {
            "role": "user",
            "content": (
                "Составь системный промпт для решения следующей задачи:\n\n"
                + TASK
            ),
        },
    ]
    generated_prompt = call_api(client, meta_request, "генерация промпта (шаг A)")

    # Шаг B
    solve_request = [
        {"role": "system", "content": generated_prompt},
        {"role": "user", "content": TASK},
    ]
    answer = call_api(client, solve_request, "решение по сгенерированному промпту (шаг B)")
    return f"[Сгенерированный промпт]\n{generated_prompt}\n\n[Ответ]\n{answer}"


def strategy_4_expert_panel(client: OpenAI) -> str:
    """
    Группа экспертов: аналитик, математик, критик.
    Каждый даёт своё решение/оценку.
    """
    experts = [
        (
            "аналитик",
            (
                "Ты — опытный аналитик данных. "
                "Отвечай чётко и структурировано: дай итоговый ответ "
                "и краткое обоснование на 2–3 предложения."
            ),
        ),
        (
            "математик",
            (
                "Ты — математик, специализирующийся на теории вероятностей. "
                "Реши задачу строго через формулы, приведи вывод и итог."
            ),
        ),
        (
            "критик",
            (
                "Ты — скептичный критик. "
                "Укажи типичные ошибки, которые допускают при решении этой задачи, "
                "а затем дай правильный ответ с пояснением."
            ),
        ),
    ]

    parts = []
    for role, system_prompt in experts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": TASK},
        ]
        answer = call_api(client, messages, f"эксперт: {role}")
        parts.append(f"[{role.upper()}]\n{answer}")

    return "\n\n".join(parts)


# ─── Вывод и сравнение ────────────────────────────────────────────────────────

SEPARATOR = "=" * 72


def section(title: str, body: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)
    # Wrap long lines for readability, but preserve existing newlines
    for paragraph in body.split("\n"):
        if paragraph.strip():
            for line in textwrap.wrap(paragraph, width=70) or [paragraph]:
                print(line)
        else:
            print()
    print()


def run_comparison() -> None:
    if not API_TOKEN:
        print("Ошибка: API_TOKEN не задан. Заполни .env (см. .env.example)")
        sys.exit(1)

    client = create_client()

    print(SEPARATOR)
    print("  СРАВНЕНИЕ ЧЕТЫРЁХ СТРАТЕГИЙ ПРОМПТИНГА")
    print(SEPARATOR)
    print(f"\nМодель : {API_MODEL}")
    print(f"API URL: {API_URL}")
    print(f"\nЗАДАЧА:\n{TASK}\n")
    print("Правильный ответ: ~50% (точно: ≈50.7%)")
    print()

    strategies = [
        ("СТРАТЕГИЯ 1 — Прямой вопрос", strategy_1_direct),
        ("СТРАТЕГИЯ 2 — Пошаговое решение", strategy_2_step_by_step),
        ("СТРАТЕГИЯ 3 — Мета-промптинг", strategy_3_meta_prompt),
        ("СТРАТЕГИЯ 4 — Группа экспертов", strategy_4_expert_panel),
    ]

    results: list[tuple[str, str]] = []

    for title, fn in strategies:
        print(f"Запускаю: {title} ...", flush=True)
        try:
            answer = fn(client)
        except (APIError, APIConnectionError, RateLimitError) as e:
            answer = f"[ОШИБКА]: {e}"
        results.append((title, answer))

    # ─── Вывод результатов ────────────────────────────────────────────────────

    for title, answer in results:
        section(title, answer)

    # ─── Итоговое сравнение ───────────────────────────────────────────────────

    print(SEPARATOR)
    print("  ИТОГОВОЕ СРАВНЕНИЕ")
    print(SEPARATOR)
    comparison_prompt_parts = [
        "Вот задача:\n" + TASK,
        "Правильный ответ: ≈50.7% (≈50%).",
        "Ниже — четыре ответа, полученных разными способами промптинга.",
        "",
    ]
    for title, answer in results:
        # Берём только первые 400 символов каждого ответа, чтобы не раздувать запрос
        preview = answer[:400].replace("\n", " ")
        comparison_prompt_parts.append(f"**{title}**:\n{preview}\n")

    comparison_prompt_parts += [
        "Оцени каждый из четырёх ответов по критериям:",
        "1. Точность итогового числа (совпадает ли с ≈50%).",
        "2. Полнота обоснования.",
        "3. Ясность объяснения.",
        "",
        "Выдай краткую таблицу (4 строки) и назови победителя.",
    ]

    comparison_messages = [
        {
            "role": "system",
            "content": "Ты — беспристрастный судья. Оценивай строго и кратко.",
        },
        {
            "role": "user",
            "content": "\n".join(comparison_prompt_parts),
        },
    ]

    print("\nЗапрашиваю итоговое сравнение у модели...", flush=True)
    try:
        verdict = call_api(client, comparison_messages, "итоговое сравнение")
    except (APIError, APIConnectionError, RateLimitError) as e:
        verdict = f"[ОШИБКА при получении сравнения]: {e}"

    section("ВЕРДИКТ МОДЕЛИ", verdict)


if __name__ == "__main__":
    run_comparison()
