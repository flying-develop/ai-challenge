#!/usr/bin/env python3
"""benchmark.py — Сравнение слабой, средней и сильной моделей.

Отправляет одинаковый запрос трём моделям и замеряет:
  - время ответа
  - количество токенов (входных / выходных)
  - стоимость запроса (в USD)

Выводит полные ответы и сводную таблицу с выводами.

Использование:
  python benchmark.py
  python benchmark.py "Свой вопрос для сравнения"
"""

import sys
import time

from openai import OpenAI, APIError, APIConnectionError
from os import environ

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_URL = environ.get("API_URL", "https://api.openai.com/v1").removesuffix("/chat/completions")
API_TOKEN = environ.get("API_TOKEN")

# Три модели: слабая → средняя → сильная
# Цены: $ за 1 000 токенов (актуальны на 2025 г.)
MODELS: list[dict] = [
    {
        "name": "gpt-3.5-turbo",
        "label": "Слабая",
        "price_input": 0.0005,
        "price_output": 0.0015,
        "hf_link": "https://huggingface.co/openai-community/gpt2",
    },
    {
        "name": "gpt-4o-mini",
        "label": "Средняя",
        "price_input": 0.00015,
        "price_output": 0.0006,
        "hf_link": "https://huggingface.co/collections/openai",
    },
    {
        "name": "gpt-4o",
        "label": "Сильная",
        "price_input": 0.005,
        "price_output": 0.015,
        "hf_link": "https://huggingface.co/collections/openai",
    },
]

DEFAULT_PROMPT = (
    "Объясни принцип квантовой запутанности простыми словами (2–3 предложения). "
    "Затем реши задачу: найди производную функции f(x) = x³ + 2x² − 5x + 3 "
    "и кратко объясни каждый шаг."
)

LINE = "─" * 62


def create_client() -> OpenAI:
    return OpenAI(base_url=API_URL, api_key=API_TOKEN)


def run_benchmark(client: OpenAI, model_info: dict, prompt: str) -> dict:
    """Отправляет запрос к модели и возвращает метрики."""
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model_info["name"],
        messages=messages,
        stream=False,
    )
    elapsed = time.perf_counter() - start

    usage = response.usage
    input_tokens = usage.prompt_tokens
    output_tokens = usage.completion_tokens
    total_tokens = usage.total_tokens

    cost_usd = (
        input_tokens / 1000 * model_info["price_input"]
        + output_tokens / 1000 * model_info["price_output"]
    )

    reply = response.choices[0].message.content or ""

    return {
        "model": model_info["name"],
        "label": model_info["label"],
        "hf_link": model_info.get("hf_link", ""),
        "reply": reply,
        "elapsed": elapsed,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def format_cost(cost_usd: float) -> str:
    if cost_usd == 0.0:
        return "н/д"
    return f"${cost_usd:.6f}"


def print_result(result: dict, index: int) -> None:
    """Выводит полный ответ и метрики одной модели."""
    print(f"\n{'=' * 62}")
    print(f"  {index}. {result['label']} модель — {result['model']}")
    print(f"{'=' * 62}")
    print(result["reply"])
    print(LINE)
    print(f"  Время ответа:    {result['elapsed']:.2f} с")
    print(f"  Токены вх / вых: {result['input_tokens']} / {result['output_tokens']}")
    print(f"  Токены итого:    {result['total_tokens']}")
    print(f"  Стоимость:       {format_cost(result['cost_usd'])}")


def print_comparison(results: list[dict]) -> None:
    """Выводит сводную таблицу сравнения и итоговый вывод."""
    if not results:
        return

    print(f"\n{'=' * 62}")
    print("  СРАВНЕНИЕ МОДЕЛЕЙ")
    print(f"{'=' * 62}")
    print(f"{'Модель':<18} {'Уровень':<10} {'Время':>7} {'Токены':>8} {'Стоимость':>12}")
    print(LINE)

    fastest = min(results, key=lambda r: r["elapsed"])
    cheapest = min(results, key=lambda r: r["cost_usd"])
    most_tokens = max(results, key=lambda r: r["total_tokens"])

    for r in results:
        marks = []
        if r is fastest:
            marks.append("⚡")
        if r is cheapest:
            marks.append("💰")
        mark_str = " ".join(marks)

        cost_str = format_cost(r["cost_usd"])
        print(
            f"{r['model']:<18} {r['label']:<10} {r['elapsed']:>6.2f}с "
            f"{r['total_tokens']:>8} {cost_str:>12}  {mark_str}"
        )

    print(LINE)

    slowest = max(results, key=lambda r: r["elapsed"])
    fastest_val = fastest["elapsed"]
    slowest_val = slowest["elapsed"]
    speed_ratio = slowest_val / fastest_val if fastest_val > 0 else 1.0

    most_expensive = max(results, key=lambda r: r["cost_usd"])
    cheapest_val = cheapest["cost_usd"]
    expensive_val = most_expensive["cost_usd"]
    cost_ratio = expensive_val / cheapest_val if cheapest_val > 0 else 0.0

    print("\nВыводы:")
    print(f"  Скорость:    самая быстрая — {fastest['label']} ({fastest['model']}, {fastest['elapsed']:.2f}с),")
    print(f"               самая медленная — {slowest['label']} ({slowest['model']}, {slowest['elapsed']:.2f}с).")
    print(f"               Разрыв: {speed_ratio:.1f}×.")

    if cheapest_val > 0:
        print(f"  Стоимость:   самая дешёвая — {cheapest['label']} ({cheapest['model']}, {format_cost(cheapest_val)}),")
        print(f"               самая дорогая — {most_expensive['label']} ({most_expensive['model']}, {format_cost(expensive_val)}).")
        print(f"               Разрыв: {cost_ratio:.1f}×.")
    else:
        print("  Стоимость:   данные о стоимости недоступны для используемого API.")

    print(f"  Токены:      больше всего токенов — {most_tokens['label']} ({most_tokens['total_tokens']} шт.).")

    print(
        "\n  Качество:    сильная модель, как правило, даёт более точные и\n"
        "               развёрнутые ответы; слабая — короче, но быстрее.\n"
        "               Для простых задач слабой модели достаточно."
    )

    print("\nСсылки:")
    print("  • Список моделей OpenAI:  https://platform.openai.com/docs/models")
    print("  • Цены OpenAI:            https://openai.com/api/pricing/")
    print("  • HuggingFace Open LLM:   https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard")
    for r in results:
        if r.get("hf_link"):
            print(f"  • {r['label']} ({r['model']}): {r['hf_link']}")


def main(prompt: str | None = None) -> None:
    if not API_TOKEN:
        print("Ошибка: API_TOKEN не задан. Заполни .env файл (см. .env.example)")
        sys.exit(1)

    benchmark_prompt = prompt or DEFAULT_PROMPT

    client = create_client()

    print(f"Benchmark: сравниваем {len(MODELS)} модели")
    print(f"API: {API_URL}")
    print(f"\nЗапрос:\n  {benchmark_prompt}\n")

    results: list[dict] = []

    for i, model_info in enumerate(MODELS, 1):
        label = model_info["label"]
        name = model_info["name"]
        print(f"[{i}/{len(MODELS)}] {label} модель ({name})...", end="", flush=True)

        try:
            result = run_benchmark(client, model_info, benchmark_prompt)
            results.append(result)
            print(f" готово за {result['elapsed']:.2f}с")
            print_result(result, i)
        except APIError as e:
            print(f"\n  Ошибка API для {name}: {e.message}")
        except APIConnectionError:
            print(f"\n  Не удалось подключиться к {API_URL}")
        except Exception as e:
            print(f"\n  Неожиданная ошибка для {name}: {e}")

    if results:
        print_comparison(results)


if __name__ == "__main__":
    user_prompt = sys.argv[1] if len(sys.argv) > 1 else None
    main(user_prompt)
