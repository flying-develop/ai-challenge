"""
День 26: Демонстрация работы с локальной LLM через Ollama.
Три запроса разной сложности к qwen2.5:3b.

Запуск:
    python demo_ollama.py

Предварительно убедитесь, что Ollama запущена:
    ollama serve
    ollama pull qwen2.5:3b
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")

QUERIES = [
    {
        "id": 1,
        "name": "Простой вопрос",
        "system": "",
        "prompt": "Что такое Python? Ответь в 2-3 предложениях.",
        "difficulty": "easy",
        "expected": ["python", "язык", "программирован"],
    },
    {
        "id": 2,
        "name": "Задача с контекстом",
        "system": "Ты — эксперт по сетям и маршрутизации.",
        "prompt": (
            "Объясни разницу между NAT и VPN простыми словами. "
            "Приведи пример, когда лучше использовать каждый."
        ),
        "difficulty": "medium",
        "expected": ["nat", "vpn"],
    },
    {
        "id": 3,
        "name": "Генерация кода",
        "system": "Ты — Python-разработчик. Пиши только код без объяснений.",
        "prompt": (
            "Напиши функцию на Python, которая принимает список чисел "
            "и возвращает словарь со статистикой: min, max, mean, median."
        ),
        "difficulty": "hard",
        "expected": ["def ", "return"],
    },
]

# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

WIDTH = 58


def _box_line(content: str, char: str = "║") -> str:
    """Строка внутри рамки шириной WIDTH."""
    # Убираем ANSI-коды при подсчёте длины (там нет, но на всякий случай)
    visible = content
    pad = WIDTH - 2 - len(visible)
    return f"{char} {visible}{' ' * max(0, pad)} {char}"


def print_query_box(
    query: dict,
    response: str,
    elapsed: float,
    ok: bool,
    total: int,
) -> None:
    """Вывести рамку с результатами одного запроса."""
    border_top = "╔" + "═" * (WIDTH - 2) + "╗"
    border_sep = "╠" + "═" * (WIDTH - 2) + "╣"
    border_bot = "╚" + "═" * (WIDTH - 2) + "╝"

    title = f"Запрос {query['id']}/{total}: {query['name']}"
    prompt_short = query["prompt"][:45] + "..." if len(query["prompt"]) > 45 else query["prompt"]
    system_str = query["system"][:30] if query["system"] else "(нет)"
    time_str = f"{elapsed:.1f}с"
    tokens_approx = f"~{max(1, len(response.split()))}"

    print(border_top)
    print(_box_line(title))
    print(border_sep)
    print(_box_line(f"Промпт: {prompt_short}"))
    print(_box_line(f"System: {system_str}"))
    print(_box_line(f"Сложность: {query['difficulty']}"))
    print(_box_line(f"Время: {time_str}  |  Слов ответа: {tokens_approx}"))
    print(border_sep)
    print(_box_line("Ответ:"))

    # Выводим ответ по строкам, обрезая длинные
    for line in response.splitlines()[:8]:
        chunks = [line[i:i + WIDTH - 4] for i in range(0, max(1, len(line)), WIDTH - 4)]
        for chunk in chunks[:2]:
            print(_box_line(f"  {chunk}"))
    if len(response.splitlines()) > 8:
        print(_box_line("  ..."))

    status_str = "✅" if ok else "❌"
    print(border_sep)
    print(_box_line(f"Содержит ключевые слова: {status_str}"))
    print(border_bot)
    print()


def check_keywords(text: str, keywords: list[str]) -> bool:
    """Проверить, что ответ содержит хотя бы половину ключевых слов."""
    text_lower = text.lower()
    found = sum(1 for kw in keywords if kw.lower() in text_lower)
    return found >= max(1, len(keywords) // 2)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------

def ollama_is_available() -> tuple[bool, str]:
    """Проверить доступность Ollama и загруженность модели.

    Returns:
        (ok, message) — флаг и описание статуса.
    """
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m["name"] for m in data.get("models", [])]
        if MODEL in models or f"{MODEL}:latest" in models:
            return True, f"Модель {MODEL} загружена"
        return False, f"Модель не найдена. Загрузите: ollama pull {MODEL}"
    except Exception as exc:
        return False, f"Сервер недоступен ({exc}). Запустите: ollama serve"


def ollama_chat(prompt: str, system: str = "") -> tuple[str, float, int, int]:
    """Отправить запрос в Ollama /api/chat.

    Returns:
        (response_text, elapsed_sec, prompt_tokens, completion_tokens)

    Raises:
        RuntimeError: При недоступности сервера.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{BASE_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Ошибка при запросе к Ollama: {exc}") from exc
    elapsed = time.perf_counter() - t0

    text = data["message"]["content"]
    prompt_tokens = data.get("prompt_eval_count", 0)
    completion_tokens = data.get("eval_count", 0)
    return text, elapsed, prompt_tokens, completion_tokens


# ---------------------------------------------------------------------------
# Итоговый отчёт
# ---------------------------------------------------------------------------

def print_report(results: list[dict]) -> None:
    """Вывести таблицу с результатами всех запросов."""
    col1, col2, col3, col4 = 20, 10, 9, 8
    total_w = col1 + col2 + col3 + col4 + 5  # 5 = разделители

    def row(name: str, time_s: str, tokens: str, status: str) -> str:
        return f"│ {name:<{col1}} │ {time_s:<{col2}} │ {tokens:<{col3}} │ {status:<{col4}} │"

    sep = f"├{'─' * (col1 + 2)}┼{'─' * (col2 + 2)}┼{'─' * (col3 + 2)}┼{'─' * (col4 + 2)}┤"
    top = f"┌{'─' * (total_w)}┐"
    mid = f"├{'─' * (col1 + 2)}┬{'─' * (col2 + 2)}┬{'─' * (col3 + 2)}┬{'─' * (col4 + 2)}┤"
    bot = f"└{'─' * (col1 + 2)}┴{'─' * (col2 + 2)}┴{'─' * (col3 + 2)}┴{'─' * (col4 + 2)}┘"

    title = "Ollama Local LLM Test Report"
    print(top)
    print(f"│ {title:<{total_w - 2}} │")
    print(mid)
    print(row("Запрос", "Время", "Токены~", "Статус"))
    print(sep)

    total_time = 0.0
    total_tokens = 0
    ok_count = 0

    for r in results:
        time_str = f"{r['elapsed']:.1f}s"
        tokens_str = f"~{r['prompt_tokens'] + r['completion_tokens']}"
        status_str = "✅" if r["ok"] else "❌"
        print(row(r["name"][:col1], time_str, tokens_str, status_str))
        total_time += r["elapsed"]
        total_tokens += r["prompt_tokens"] + r["completion_tokens"]
        if r["ok"]:
            ok_count += 1

    print(sep)
    total_status = f"{ok_count}/{len(results)} ✅" if ok_count == len(results) else f"{ok_count}/{len(results)} ❌"
    print(row("ИТОГО", f"{total_time:.1f}s", f"~{total_tokens}", total_status))
    print(bot)
    print()
    print(f"  Модель  : {MODEL}")
    print(f"  Сервер  : {BASE_URL}")
    if ok_count == len(results):
        print(f"  Все запросы выполнены успешно.")
    else:
        print(f"  Часть запросов не прошла проверку ключевых слов.")
    print()


# ---------------------------------------------------------------------------
# Основной цикл
# ---------------------------------------------------------------------------

def main() -> int:
    print()
    print("=" * 60)
    print("  День 26: Ollama Local LLM Demo")
    print("=" * 60)
    print()

    # Шаг 1 — проверка доступности
    print("Шаг 1 — Проверка доступности Ollama...")
    ok, msg = ollama_is_available()
    if ok:
        print(f"  ✅ Ollama доступна: {BASE_URL}")
        print(f"  ✅ {msg}")
    else:
        print(f"  ❌ Ollama недоступна: {BASE_URL}")
        print(f"  ❌ {msg}")
        return 1
    print()

    results: list[dict] = []

    for query in QUERIES:
        n = query["id"]
        total = len(QUERIES)
        print(f"Шаг {n + 1} — Запрос {n}/{total}: {query['name']}")
        print(f"  [{query['difficulty']}] {query['prompt'][:70]}...")
        print("  Ожидание ответа...")

        try:
            text, elapsed, prompt_tok, completion_tok = ollama_chat(
                query["prompt"],
                system=query["system"],
            )
        except RuntimeError as exc:
            print(f"  ❌ {exc}")
            results.append({
                "name": query["name"],
                "elapsed": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "ok": False,
            })
            continue

        ok_check = check_keywords(text, query["expected"])
        results.append({
            "name": query["name"],
            "elapsed": elapsed,
            "prompt_tokens": prompt_tok,
            "completion_tokens": completion_tok,
            "ok": ok_check,
        })

        print_query_box(query, text, elapsed, ok_check, total)

    # Итоговый отчёт
    print("=" * 60)
    print("  Итоговый отчёт")
    print("=" * 60)
    print()
    print_report(results)

    all_ok = all(r["ok"] for r in results)
    return 0 if all_ok else 2


if __name__ == "__main__":
    sys.exit(main())
