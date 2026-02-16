import sys
import json
import time
import threading

import requests
from os import environ

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — читаем env напрямую

API_URL = environ.get("API_URL", "https://api.openai.com/v1/chat/completions")
API_MODEL = environ.get("API_MODEL", "gpt-4")
API_TOKEN = environ.get("API_TOKEN")

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]

SPINNER_CHARS = ["|", "/", "-", "\\"]


class Spinner:
    """Крутилка в консоли пока ждём ответ от API."""

    def __init__(self):
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join()
        # стираем символ спиннера
        print("\r" + " " * 80 + "\r", end="", flush=True)

    def _spin(self):
        i = 0
        while not self._stop.is_set():
            char = SPINNER_CHARS[i % len(SPINNER_CHARS)]
            print(f"\rAI: {char}", end="", flush=True)
            i += 1
            self._stop.wait(0.15)


def request_with_retry(url: str, headers: dict, payload: dict) -> requests.Response:
    """POST-запрос с retry при 429 (rate limit)."""

    for attempt in range(MAX_RETRIES + 1):
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60,
        )

        if response.status_code != 429 or attempt == MAX_RETRIES:
            response.raise_for_status()
            return response

        wait = RETRY_BACKOFF[attempt]
        print(f"\rRate limit (429). Повтор через {wait}с...", end="", flush=True)
        time.sleep(wait)

    return response


def send_message(messages: list[dict]) -> str:
    """Отправляет сообщения в API и стримит ответ в консоль."""

    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": API_MODEL,
        "messages": messages,
        "stream": True,
    }

    spinner = Spinner()
    spinner.start()

    try:
        response = request_with_retry(API_URL, headers, payload)
    except Exception:
        spinner.stop()
        raise

    first_token = True
    full_reply = []

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data = line[len("data: "):]

        if data == "[DONE]":
            break

        chunk = json.loads(data)
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")

        if token:
            if first_token:
                spinner.stop()
                print("AI: ", end="", flush=True)
                first_token = False
            print(token, end="", flush=True)
            full_reply.append(token)

    if first_token:
        spinner.stop()
        print("AI: ", end="", flush=True)

    print()
    return "".join(full_reply)


def main() -> None:
    if not API_TOKEN:
        print("Ошибка: API_TOKEN не задан. Заполни .env файл (см. .env.example)")
        sys.exit(1)

    print(f"Чат с {API_MODEL} ({API_URL})")
    print("Введи сообщение. Для выхода: quit или Ctrl+C\n")

    messages: list[dict] = []

    while True:
        try:
            user_input = input("Ты: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nПока!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Пока!")
            break

        messages.append({"role": "user", "content": user_input})

        try:
            reply = send_message(messages)
        except requests.exceptions.HTTPError as e:
            print(f"Ошибка API: {e}")
            messages.pop()
            continue
        except requests.exceptions.ConnectionError:
            print(f"Не удалось подключиться к {API_URL}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
