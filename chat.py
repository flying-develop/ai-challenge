import sys
import time
import threading

from openai import OpenAI, APIError, APIConnectionError, RateLimitError
from os import environ

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv не установлен — читаем env напрямую

def normalize_base_url(url: str) -> str:
    """Приводит API_URL к base URL, который ожидает OpenAI клиент."""
    cleaned = url.strip().rstrip("/")

    if cleaned.endswith("/chat/completions"):
        cleaned = cleaned[: -len("/chat/completions")]

    return cleaned


API_URL = normalize_base_url(environ.get("API_URL", "https://api.openai.com/v1"))
API_MODEL = environ.get("API_MODEL", "gpt-4")
API_TOKEN = environ.get("API_TOKEN") or environ.get("OPENAI_API_KEY")

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]

SPINNER_CHARS = ["|", "/", "-", "\\"]


def create_client() -> OpenAI:
    return OpenAI(base_url=API_URL, api_key=API_TOKEN)


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


def send_message(client: OpenAI, messages: list[dict]) -> str:
    """Отправляет сообщения в API и стримит ответ в консоль."""

    spinner = Spinner()
    spinner.start()

    last_error = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(
                model=API_MODEL,
                messages=messages,
                stream=True,
            )
            break
        except RateLimitError as e:
            last_error = e
            if attempt == MAX_RETRIES:
                spinner.stop()
                raise
            wait = RETRY_BACKOFF[attempt]
            spinner.stop()
            print(f"\rRate limit (429). Повтор через {wait}с...", end="", flush=True)
            time.sleep(wait)
            spinner.start()
        except Exception:
            spinner.stop()
            raise

    first_token = True
    full_reply = []

    for chunk in stream:
        token = chunk.choices[0].delta.content or ""

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

    client = create_client()

    print(f"Чат с {API_MODEL} ({API_URL})")
    print("Введи сообщение. Для выхода: quit или Ctrl+C\n")

    messages: list[dict] = []

    while True:
        try:
            raw = input("Ты: ")
            # Фикс суррогатных символов (кривая кодировка терминала)
            user_input = raw.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace").strip()
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
            reply = send_message(client, messages)
        except APIError as e:
            print(f"Ошибка API: {e.message}")
            messages.pop()
            continue
        except APIConnectionError:
            print(f"Не удалось подключиться к {API_URL}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
