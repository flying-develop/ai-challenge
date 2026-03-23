"""MCP-сервер: отправка прогресса и результатов в Telegram.

Образовательная концепция: вывод оркестратора — это не только финальный
ответ, но и промежуточные уведомления. Telegram-сервер отделён, чтобы
легко заменить канал доставки (SMS, email, webhook) без изменения логики.

Если TELEGRAM_BOT_TOKEN не задан — вывод в stdout (для тестов).

Запуск:
    python -m mcp_server.telegram_server

Инструменты:
    send_progress — уведомление о прогрессе (этап пайплайна)
    send_result   — финальный результат (с разбивкой на части)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from mcp.server.fastmcp import FastMCP

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

mcp = FastMCP("Telegram Server")

# Лимит длины сообщения Telegram
_TG_LIMIT = 4096


# ---------------------------------------------------------------------------
# Внутренние функции
# ---------------------------------------------------------------------------

def _send_tg_message(chat_id: str, text: str, token: str) -> bool:
    """Отправить одно сообщение в Telegram Bot API."""
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("ok", False)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[telegram_server] HTTP {exc.code}: {body}")
        return False
    except Exception as exc:
        print(f"[telegram_server] Ошибка отправки: {exc}")
        return False


def _split_text(text: str, limit: int = _TG_LIMIT) -> list[str]:
    """Разбить текст на части не длиннее limit символов."""
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        # Ищем ближайший перенос строки назад
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return parts


# ---------------------------------------------------------------------------
# Инструменты
# ---------------------------------------------------------------------------

@mcp.tool()
def send_progress(chat_id: str, stage: str, message: str) -> str:
    """
    Отправить уведомление о прогрессе в Telegram.

    Если TELEGRAM_BOT_TOKEN не задан — выводит в stdout.

    Args:
        chat_id: ID чата или @username (например, @flying_dev)
        stage: Название этапа (для лога)
        message: Текст уведомления

    Returns:
        Статус отправки
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        print(f"[TG_PROGRESS|{stage}] {message}")
        return f"stdout: [{stage}] {message}"

    # Если chat_id не задан — берём из .env
    if not chat_id:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not chat_id:
        return "Ошибка: chat_id не задан"

    ok = _send_tg_message(chat_id, message, token)
    if ok:
        return f"Отправлено [{stage}]: {message[:50]}..."
    else:
        return f"Ошибка отправки [{stage}]"


@mcp.tool()
def poll_messages(offset: int = 0, timeout: int = 30) -> str:
    """
    Получить новые сообщения через Telegram Bot API long-polling.

    Args:
        offset:  ID последнего обработанного update + 1 (для подтверждения).
        timeout: Время ожидания новых сообщений в секундах (long-polling).

    Returns:
        JSON: [{"update_id": N, "chat_id": "...", "text": "...", "username": "..."}, ...]
        Пустой список если новых сообщений нет.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return json.dumps([])

    url = (
        f"https://api.telegram.org/bot{token}/getUpdates"
        f"?offset={offset}&timeout={timeout}&allowed_updates=%5B%22message%22%5D"
    )
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"[telegram_server] poll_messages ошибка: {exc}")
        return json.dumps([])

    if not data.get("ok"):
        return json.dumps([])

    messages = []
    for update in data.get("result", []):
        msg = update.get("message", {})
        if not msg:
            continue
        chat = msg.get("chat", {})
        from_user = msg.get("from", {})
        text = msg.get("text", "")
        if not text:
            continue
        messages.append({
            "update_id": update["update_id"],
            "chat_id": str(chat.get("id", "")),
            "text": text,
            "username": from_user.get("username", "")
                        or from_user.get("first_name", "unknown"),
        })

    return json.dumps(messages, ensure_ascii=False)


@mcp.tool()
def send_message(chat_id: str, text: str) -> str:
    """
    Отправить сообщение пользователю в Telegram.

    Разбивает длинные сообщения на части (лимит 4096 символов).
    Если TELEGRAM_BOT_TOKEN не задан — выводит в stdout.

    Args:
        chat_id: ID чата пользователя (числовой или @username)
        text:    Текст сообщения (Markdown)

    Returns:
        Статус отправки
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    parts = _split_text(text)

    if not token:
        for i, part in enumerate(parts, 1):
            print(f"[TG_MESSAGE part {i}/{len(parts)}]\n{part}")
        return f"stdout: {len(parts)} частей"

    if not chat_id:
        return "Ошибка: chat_id не задан"

    sent = 0
    for part in parts:
        if _send_tg_message(chat_id, part, token):
            sent += 1

    return f"Отправлено {sent}/{len(parts)} частей в {chat_id}"


@mcp.tool()
def send_result(chat_id: str, text: str) -> str:
    """
    Отправить финальный результат в Telegram.

    Разбивает длинные сообщения на части (лимит 4096 символов).
    Если TELEGRAM_BOT_TOKEN не задан — выводит в stdout.

    Args:
        chat_id: ID чата или @username
        text: Полный текст результата (Markdown)

    Returns:
        Статус отправки (сколько сообщений отправлено)
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()

    # Если chat_id не задан — берём из .env
    if not chat_id:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    parts = _split_text(text)

    if not token:
        for i, part in enumerate(parts, 1):
            print(f"[TG_RESULT part {i}/{len(parts)}]\n{part}")
        return f"stdout: {len(parts)} частей"

    if not chat_id:
        return "Ошибка: chat_id не задан"

    sent = 0
    for part in parts:
        if _send_tg_message(chat_id, part, token):
            sent += 1

    return f"Отправлено {sent}/{len(parts)} частей в {chat_id}"


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
