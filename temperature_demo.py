"""
Демонстрация влияния параметра temperature на ответы модели.

Один и тот же запрос отправляется трижды с разными значениями temperature:
  • 0.0  — детерминированный, точный
  • 0.7  — сбалансированный (умолчание)
  • 1.2  — креативный, разнообразный

Запуск:
    python temperature_demo.py
"""

import sys
import time
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

TEMPERATURES = [0.0, 0.7, 1.2]

# Запрос, который достаточно открытый, чтобы разница температур проявилась
PROMPT = "Придумай слоган для небольшой уютной кофейни. Одно предложение, кратко и образно."

SEPARATOR = "=" * 60


def call_api(client: OpenAI, prompt: str, temperature: float) -> str:
    """Отправляет запрос и возвращает полный ответ."""
    messages = [{"role": "user", "content": prompt}]
    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as e:
            last_error = e
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF[attempt]
            print(f"  Rate limit. Повтор через {wait}с...", flush=True)
            time.sleep(wait)
        except (APIError, APIConnectionError):
            raise

    raise last_error  # type: ignore[misc]


def describe_temperature(t: float) -> str:
    if t == 0.0:
        return "Точный / детерминированный"
    if t < 1.0:
        return "Сбалансированный"
    return "Креативный / разнообразный"


def print_results(results: list[tuple[float, str]]) -> None:
    print("\n")
    print(SEPARATOR)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ ТЕМПЕРАТУР")
    print(SEPARATOR)
    print(f"Запрос: «{PROMPT}»\n")

    for temp, reply in results:
        label = describe_temperature(temp)
        print(f"temperature = {temp}  [{label}]")
        print("-" * 40)
        print(reply.strip())
        print()

    print(SEPARATOR)
    print("ВЫВОДЫ")
    print(SEPARATOR)
    print("""
┌─────────────────┬────────────────────────────────────────────────────────────┐
│  Temperature    │  Когда использовать                                        │
├─────────────────┼────────────────────────────────────────────────────────────┤
│  0.0            │  Факты, математика, SQL, код, извлечение данных.           │
│                 │  Ответ стабилен при повторных запросах — удобно для        │
│                 │  тестирования и детерминированных пайплайнов.              │
├─────────────────┼────────────────────────────────────────────────────────────┤
│  0.7            │  Чат-боты, суммаризация, технические объяснения,           │
│                 │  деловая переписка. Хороший баланс точности и              │
│                 │  читаемости; значение по умолчанию в большинстве           │
│                 │  продуктовых сценариев.                                    │
├─────────────────┼────────────────────────────────────────────────────────────┤
│  1.2            │  Брейншторм, слоганы, художественные тексты, поэзия,       │
│                 │  нейминг. Модель рискует и экспериментирует — именно       │
│                 │  это нужно, когда важна оригинальность, а не              │
│                 │  воспроизводимость.                                        │
└─────────────────┴────────────────────────────────────────────────────────────┘

Точность   ████████████░░░░░░░░  убывает с ростом temperature
Креативность ░░░░░░░░░░░░████████  растёт с ростом temperature
Разнообразие ░░░░░░░░░░░░████████  растёт с ростом temperature
""")


def main() -> None:
    if not API_TOKEN:
        print("Ошибка: API_TOKEN не задан. Заполни .env файл.")
        sys.exit(1)

    client = OpenAI(base_url=API_URL, api_key=API_TOKEN)

    print(f"Модель: {API_MODEL}  |  URL: {API_URL}")
    print(f"Запрос: «{PROMPT}»")
    print(f"Температуры: {TEMPERATURES}\n")

    results: list[tuple[float, str]] = []

    for temp in TEMPERATURES:
        print(f"  Запрос с temperature={temp}...", end=" ", flush=True)
        try:
            reply = call_api(client, PROMPT, temperature=temp)
            results.append((temp, reply))
            print("OK")
        except APIError as e:
            print(f"Ошибка API: {e.message}")
            sys.exit(1)
        except APIConnectionError:
            print(f"Не удалось подключиться к {API_URL}")
            sys.exit(1)

    print_results(results)


if __name__ == "__main__":
    main()
