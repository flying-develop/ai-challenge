import sys
import json

import requests
from dotenv import load_dotenv
from os import environ

load_dotenv()

API_URL = environ.get("API_URL", "https://api.openai.com/v1/chat/completions")
API_MODEL = environ.get("API_MODEL", "gpt-4")
API_TOKEN = environ.get("API_TOKEN")


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

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    )
    response.raise_for_status()

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
            print(token, end="", flush=True)
            full_reply.append(token)

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

        print("AI: ", end="", flush=True)
        try:
            reply = send_message(messages)
        except requests.exceptions.HTTPError as e:
            print(f"\nОшибка API: {e}")
            messages.pop()
            continue
        except requests.exceptions.ConnectionError:
            print(f"\nНе удалось подключиться к {API_URL}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
