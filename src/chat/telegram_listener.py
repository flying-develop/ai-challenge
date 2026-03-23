"""TelegramListener: бесконечный цикл получения сообщений от пользователей.

Использует long-polling (getUpdates) для получения сообщений.
Каждый пользователь (chat_id) получает изолированный DialogManager
с собственной историей и task state.

RAG-пайплайн и LLM — общие (shared).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .session_store import SessionStore

_TG_LIMIT = 4096

# Приветственное сообщение
_WELCOME = """\
Привет! Я support-бот по документации.

Задайте вопрос, и я найду ответ с цитатами из документации.

Доступные команды:
/help — список команд
/state — текущий контекст задачи
/reset — начать заново
/sources — источники последнего ответа"""

_HELP = """\
*Доступные команды:*

/start — приветствие
/help — этот список
/state — текущий контекст задачи (что я о вас знаю)
/reset — сбросить историю и начать заново
/sources — источники последнего ответа

Задайте любой вопрос — я найду ответ по документации."""


class TelegramListener:
    """Слушает сообщения от пользователей Telegram-бота.

    Каждый пользователь (chat_id) получает изолированный контекст:
    свою историю и task state. RAG-индекс и LLM — общие.

    Args:
        dialog_manager_factory: Функция create(user_id) → DialogManager.
        session_store:          Хранилище метаданных сессий.
        log_callback:           Опциональный колбэк для логирования запросов.
    """

    def __init__(
        self,
        dialog_manager_factory: Callable,
        session_store: SessionStore,
        log_callback: Callable | None = None,
    ) -> None:
        self._factory = dialog_manager_factory
        self._sessions: dict[str, object] = {}  # chat_id → DialogManager
        self.store = session_store
        self._log_callback = log_callback
        self._last_offset = 0
        self._token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self._running = False

    def run(self) -> None:
        """Запустить бесконечный цикл long-polling.

        Бесконечный цикл:
            1. getUpdates(offset, timeout=30)
            2. Для каждого сообщения:
               a. Получить/создать DialogManager
               b. Обработать команду или вопрос
               c. Отправить ответ
            3. Обновить offset

        Прерывается по KeyboardInterrupt или вызову stop().
        """
        if not self._token:
            print("[TelegramListener] TELEGRAM_BOT_TOKEN не задан — работа невозможна")
            return

        self._running = True
        print("[TelegramListener] Запуск long-polling...")

        while self._running:
            try:
                updates = self._poll(timeout=30)
            except KeyboardInterrupt:
                print("[TelegramListener] Остановка по Ctrl+C")
                break
            except Exception as exc:
                print(f"[TelegramListener] Ошибка polling: {exc}")
                time.sleep(5)
                continue

            for update in updates:
                try:
                    self._handle_update(update)
                except Exception as exc:
                    print(f"[TelegramListener] Ошибка обработки update: {exc}")

                self._last_offset = update["update_id"] + 1

    def stop(self) -> None:
        """Остановить цикл polling."""
        self._running = False

    def _poll(self, timeout: int = 30) -> list[dict]:
        """Получить новые сообщения через getUpdates.

        Args:
            timeout: Время ожидания (long-polling).

        Returns:
            Список update-словарей: {update_id, chat_id, text, username}.
        """
        url = (
            f"https://api.telegram.org/bot{self._token}/getUpdates"
            f"?offset={self._last_offset}&timeout={timeout}"
            f"&allowed_updates=%5B%22message%22%5D"
        )
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception:
            return []

        if not data.get("ok"):
            return []

        messages = []
        for update in data.get("result", []):
            msg = update.get("message", {})
            if not msg or "text" not in msg:
                continue
            chat = msg.get("chat", {})
            from_user = msg.get("from", {})
            messages.append({
                "update_id": update["update_id"],
                "chat_id": str(chat.get("id", "")),
                "text": msg["text"],
                "username": (
                    from_user.get("username")
                    or from_user.get("first_name", "unknown")
                ),
            })

        return messages

    def _handle_update(self, update: dict) -> None:
        """Обработать одно сообщение от пользователя.

        Args:
            update: {update_id, chat_id, text, username}
        """
        chat_id = update["chat_id"]
        text = update["text"].strip()
        username = update["username"]

        # Обновляем метаданные сессии
        self.store.get_or_create(chat_id, username)
        self.store.touch(chat_id)
        self.store.increment(chat_id)

        # Получаем или создаём DialogManager
        dialog = self._get_or_create_dialog(chat_id)

        # Обрабатываем команды
        if text == "/start":
            self._send(chat_id, _WELCOME)
            return
        if text == "/help":
            self._send(chat_id, _HELP)
            return
        if text == "/state":
            state_text = dialog.format_task_state()
            self._send(chat_id, f"*Текущий контекст:*\n{state_text}")
            return
        if text == "/reset":
            dialog.reset()
            self._send(chat_id, "История сброшена. Начните новый вопрос.")
            return
        if text == "/sources":
            sources = dialog.get_last_sources()
            if sources:
                src_text = "\n".join(f"• {s}" for s in sources)
                self._send(chat_id, f"*Источники последнего ответа:*\n{src_text}")
            else:
                self._send(chat_id, "Нет данных об источниках последнего ответа.")
            return

        # Обычный вопрос → RAG
        print(f"[TG] {username} ({chat_id}): {text[:80]}")

        try:
            print(f"[TG]   → поиск в индексе...")
            rag_answer = dialog.process_message(text)

            confidence = getattr(rag_answer, "confidence", 0.0)
            n_sources = len(getattr(rag_answer, "sources", []))
            structured = getattr(rag_answer, "structured", None)
            n_struct_sources = len(structured.sources) if structured else 0
            print(f"[TG]   → найдено {n_sources} чанков, confidence={confidence:.2f}, "
                  f"structured_sources={n_struct_sources}")

            response_text = _format_rag_answer(rag_answer)

            # Сохраняем источники для /sources
            sources = dialog.get_last_sources()
            self.store.set_last_sources(chat_id, sources)

            print(f"[TG]   → отправка ответа ({len(response_text)} симв.)")
            self._send(chat_id, response_text)
            print(f"[TG]   ✓ готово")

            # Логируем если есть колбэк
            if self._log_callback:
                self._log_callback({
                    "chat_id": chat_id,
                    "username": username,
                    "question": text,
                    "confidence": confidence,
                })

        except Exception as exc:
            import traceback
            print(f"[TG] Ошибка обработки вопроса: {exc}")
            traceback.print_exc()
            self._send(chat_id, "Произошла ошибка при обработке вопроса. Попробуйте переформулировать.")

    def _get_or_create_dialog(self, chat_id: str) -> object:
        """Получить или создать DialogManager для пользователя.

        Args:
            chat_id: Telegram chat ID.

        Returns:
            DialogManager с изолированной историей и task state.
        """
        if chat_id not in self._sessions:
            self._sessions[chat_id] = self._factory.create(user_id=chat_id)
        return self._sessions[chat_id]

    def _send(self, chat_id: str, text: str) -> None:
        """Отправить сообщение пользователю (с разбивкой на части).

        Args:
            chat_id: Telegram chat ID.
            text:    Текст сообщения (Markdown).
        """
        parts = _split_text(text)
        for part in parts:
            _send_tg_message(chat_id, part, self._token)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _format_rag_answer(rag_answer) -> str:
    """Форматировать RAGAnswer в Markdown для Telegram.

    Args:
        rag_answer: RAGAnswer объект из pipeline.answer().

    Returns:
        Строка в Markdown-формате для отправки в Telegram.
    """
    structured = getattr(rag_answer, "structured", None)

    if structured:
        answer_text = structured.answer or rag_answer.answer
        sources = structured.sources or []
        quotes = getattr(structured, "verified_quotes", None) or []
    else:
        answer_text = rag_answer.answer
        sources = getattr(rag_answer, "sources", [])
        quotes = []

    parts = []

    # Ответ
    if answer_text:
        clean_answer = _escape_markdown(answer_text)
        parts.append(f"📋 *Ответ:*\n{clean_answer}")

    # Источники
    # structured.sources — список SourceRef (поля: file, section)
    # rag_answer.sources — список RetrievalResult (поля: source, section)
    if sources:
        src_lines = []
        for i, src in enumerate(sources[:5], 1):
            source_name = getattr(src, "file", None) or getattr(src, "source", str(src))
            section = getattr(src, "section", "")
            label = f"`{source_name}`"
            if section:
                label += f" — {_escape_markdown(section)}"
            src_lines.append(f"{i}\\. {label}")
        parts.append("📚 *Источники:*\n" + "\n".join(src_lines))

    # Цитаты
    if quotes:
        quote_lines = []
        for i, q in enumerate(quotes[:3], 1):
            text_q = getattr(q, "text", str(q))
            quote_lines.append(f"\\[{i}\\] _{_escape_markdown(text_q[:150])}_")
        parts.append("💬 *Цитаты:*\n" + "\n".join(quote_lines))

    return "\n\n".join(parts) if parts else "Не удалось найти ответ в документации."


def _escape_markdown(text: str) -> str:
    """Экранировать специальные символы Markdown для Telegram.

    Args:
        text: Исходный текст.

    Returns:
        Экранированный текст (только критичные символы).
    """
    # Экранируем только то что ломает Markdown в Telegram
    for char in ["[", "]", "(", ")", "~", ">", "#", "+", "-", "=", "|", "{", "}"]:
        text = text.replace(char, f"\\{char}")
    return text


def _split_text(text: str, limit: int = _TG_LIMIT) -> list[str]:
    """Разбить текст на части не длиннее limit символов.

    Args:
        text:  Исходный текст.
        limit: Максимальная длина части.

    Returns:
        Список частей.
    """
    if len(text) <= limit:
        return [text]
    parts = []
    while text:
        if len(text) <= limit:
            parts.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        parts.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return parts


def _send_tg_message(chat_id: str, text: str, token: str) -> bool:
    """Отправить одно сообщение через Telegram Bot API.

    Args:
        chat_id: ID чата.
        text:    Текст сообщения.
        token:   Bot API token.

    Returns:
        True если успешно.
    """
    import json as _json
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }
    data = _json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = _json.loads(resp.read().decode("utf-8"))
            return result.get("ok", False)
    except Exception as exc:
        print(f"[TelegramListener] Ошибка отправки: {exc}")
        return False
