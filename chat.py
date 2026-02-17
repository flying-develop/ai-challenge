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

API_URL = environ.get("API_URL", "https://api.openai.com/v1").removesuffix("/chat/completions")
API_MODEL = environ.get("API_MODEL", "gpt-4")
API_TOKEN = environ.get("API_TOKEN")
API_MAX_TOKENS = int(environ.get("API_MAX_TOKENS", "1024"))

MAX_RETRIES = 4
RETRY_BACKOFF = [2, 4, 8, 16]

SPINNER_CHARS = ["|", "/", "-", "\\"]

STOP_MARKER = "<END>"

SYSTEM_PROMPTS = {
    "structured": (
        "Ты — полезный ассистент. Отвечай строго по следующей структуре:\n\n"
        "## Кратко\nОдно-два предложения с сутью ответа.\n\n"
        "## Детали\nОсновное содержание ответа.\n\n"
        "## Итог\nКраткий вывод или рекомендация.\n\n"
        "Ограничения:\n"
        "- Отвечай только на русском языке, если не указано иное.\n"
        "- Не отклоняйся от заданной структуры.\n"
        "- Не более 5 пунктов в каждой секции.\n"
    ),
    "brief": (
        "Ты — полезный ассистент.\n"
        "Отвечай кратко: максимум 3 предложения.\n"
        "Язык ответа — русский, если не указано иное.\n"
    ),
    "json": (
        "Ты — полезный ассистент. Отвечай строго в формате JSON:\n"
        '{"summary": "...", "details": "...", "conclusion": "..."}\n'
        "Ограничения:\n"
        "- Только валидный JSON, без markdown-обёрток.\n"
        "- Язык значений — русский, если не указано иное.\n"
    ),
}


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


def build_system_message(config: dict) -> dict | None:
    """Собирает system-сообщение на основе текущей конфигурации."""
    if not config["constraints_enabled"]:
        return None

    fmt = config.get("format", "structured")
    base = SYSTEM_PROMPTS.get(fmt, SYSTEM_PROMPTS["structured"])
    parts = [base]

    max_tokens = config.get("max_tokens")
    if max_tokens:
        parts.append(f"- Ответ должен укладываться примерно в {max_tokens} токенов.\n")

    if config.get("stop_enabled"):
        parts.append(
            f"- Завершай ответ маркером {STOP_MARKER} после полезного текста.\n"
        )

    return {"role": "system", "content": "".join(parts)}


def configure(config: dict) -> dict:
    """Интерактивный диалог настройки параметров ответа."""
    formats = {"1": "structured", "2": "brief", "3": "json"}
    fmt_labels = {
        "structured": "структурированный",
        "brief": "краткий",
        "json": "JSON",
    }

    while True:
        ce = "ВКЛ" if config["constraints_enabled"] else "ВЫКЛ"
        fl = fmt_labels.get(config["format"], config["format"])
        se = "вкл" if config["stop_enabled"] else "выкл"

        print(f"\n{'=' * 40}")
        print("  Настройки ответа")
        print(f"{'=' * 40}")
        print(f"  1. Формат:        {fl}")
        print(f"  2. Макс. токенов: {config['max_tokens']}")
        print(f"  3. Stop-маркер:   {STOP_MARKER} ({se})")
        print(f"  4. Ограничения:   {ce}")
        print(f"{'=' * 40}")
        print("  Enter — вернуться в чат")
        print()

        choice = input("Номер настройки: ").strip()

        if not choice:
            break

        if choice == "1":
            print("  1) структурированный  2) краткий  3) JSON")
            fc = input("  Выбор: ").strip()
            if fc in formats:
                config["format"] = formats[fc]
        elif choice == "2":
            val = input(f"  Макс. токенов [{config['max_tokens']}]: ").strip()
            if val.isdigit() and int(val) > 0:
                config["max_tokens"] = int(val)
        elif choice == "3":
            config["stop_enabled"] = not config["stop_enabled"]
            st = "включён" if config["stop_enabled"] else "выключен"
            print(f"  Stop-маркер {st}")
        elif choice == "4":
            config["constraints_enabled"] = not config["constraints_enabled"]
            st = "включены" if config["constraints_enabled"] else "выключены"
            print(f"  Ограничения {st}")

    return config


def send_message(
    client: OpenAI,
    messages: list[dict],
    max_tokens: int | None = None,
    stop: list[str] | None = None,
) -> str:
    """Отправляет сообщения в API и стримит ответ в консоль."""

    spinner = Spinner()
    spinner.start()

    last_error = None

    api_kwargs: dict = {
        "model": API_MODEL,
        "messages": messages,
        "stream": True,
    }
    if max_tokens is not None:
        api_kwargs["max_tokens"] = max_tokens
    if stop is not None:
        api_kwargs["stop"] = stop

    for attempt in range(MAX_RETRIES + 1):
        try:
            stream = client.chat.completions.create(**api_kwargs)
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

    result = "".join(full_reply)
    # Удаляем stop-маркер, если он проскочил в ответ
    if stop:
        for marker in stop:
            result = result.replace(marker, "").rstrip()
    return result


def main() -> None:
    if not API_TOKEN:
        print("Ошибка: API_TOKEN не задан. Заполни .env файл (см. .env.example)")
        sys.exit(1)

    client = create_client()

    print(f"Чат с {API_MODEL} ({API_URL})")
    print("Введи сообщение. Для выхода: quit или Ctrl+C")
    print("Команда /config — настройки формата ответа\n")

    config: dict = {
        "format": "structured",
        "max_tokens": API_MAX_TOKENS,
        "stop_enabled": True,
        "constraints_enabled": True,
    }

    history: list[dict] = []

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

        if user_input.lower() == "/config":
            config = configure(config)
            continue

        history.append({"role": "user", "content": user_input})

        # Собираем messages: system (опционально) + история диалога
        messages: list[dict] = []
        sys_msg = build_system_message(config)
        if sys_msg:
            messages.append(sys_msg)
        messages.extend(history)

        # Параметры API из конфигурации
        max_tokens = config["max_tokens"] if config["constraints_enabled"] else None
        stop = (
            [STOP_MARKER]
            if config["constraints_enabled"] and config["stop_enabled"]
            else None
        )

        try:
            reply = send_message(client, messages, max_tokens=max_tokens, stop=stop)
        except APIError as e:
            print(f"Ошибка API: {e.message}")
            history.pop()
            continue
        except APIConnectionError:
            print(f"Не удалось подключиться к {API_URL}")
            history.pop()
            continue

        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
