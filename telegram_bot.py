#!/usr/bin/env python3
"""Telegram-бот: принимает сообщения и запускает исследовательский агент.

Использует python-telegram-bot v20+ (async).

Установка:
    pip install python-telegram-bot

Запуск:
    python telegram_bot.py

Конфигурация (.env):
    TELEGRAM_BOT_TOKEN=...     # обязательно — токен от @BotFather
    TELEGRAM_CHAT_ID=...       # опционально: белый список (отдельные id через запятую)
                               # если не задан — принимает от любого
    SEARCH_MODE=mock           # mock работает без ключей Яндекса (по умолчанию)
    # LLM-провайдер (один из):
    QWEN_API_KEY=...
    OPENAI_API_KEY=...
    ANTHROPIC_API_KEY=...

Поддерживаемые команды бота:
    /start          — приветствие и инструкция
    /help           — справка
    /status         — статус последней задачи
    /cancel         — сбросить зависший запрос
    Любой текст     — запустить двухпроходное исследование

Архитектура:
    PTB Application (async polling) → message handler
    → ThreadPoolExecutor (1 worker) → ResearchOrchestrator.run()

    Ключевой момент: MCPClient.call_tool() внутри использует asyncio.run(),
    что нельзя вызвать из уже запущенного event loop (PTB).
    Решение: запускаем синхронный оркестратор в отдельном потоке через
    loop.run_in_executor(executor, ...) — каждый поток имеет свой event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import (
        Application,
        CommandHandler,
        ContextTypes,
        MessageHandler,
        filters,
    )
    _PTB_AVAILABLE = True
except ImportError:
    _PTB_AVAILABLE = False


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("telegram_bot")

# Глушим лишние логи от httpx/telegram
for _noisy in ("httpx", "telegram.ext.Application", "telegram.ext._updater"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Состояние бота (shared между async-хендлерами)
# ---------------------------------------------------------------------------

class BotState:
    """Потокобезопасное состояние бота."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._current_task = ""
        self._current_chat_id: int | str = 0
        self.last_task_id = ""
        self.last_state = ""
        self.last_result = ""

    def acquire(self, task: str, chat_id: int | str) -> bool:
        """Занять слот. False если уже занят."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._current_task = task
            self._current_chat_id = chat_id
            return True

    def release(self) -> None:
        with self._lock:
            self._busy = False
            self._current_task = ""

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    @property
    def current_task(self) -> str:
        with self._lock:
            return self._current_task


_state = BotState()
# Один воркер: задачи выполняются последовательно
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research")


# ---------------------------------------------------------------------------
# Белый список chat_id
# ---------------------------------------------------------------------------

def _allowed_ids() -> set[str]:
    """Разобрать TELEGRAM_CHAT_ID в множество строк."""
    raw = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def _is_allowed(chat_id: int | str) -> bool:
    allowed = _allowed_ids()
    if not allowed:
        return True
    return str(chat_id) in allowed


# ---------------------------------------------------------------------------
# Тексты сообщений
# ---------------------------------------------------------------------------

WELCOME = (
    "👋 Привет\\! Я исследовательский агент\\.\n\n"
    "Напишите любой вопрос или задачу — я проведу двухпроходное "
    "исследование и пришлю подробный ответ\\.\n\n"
    "Пример: *Хочу сходить в кино в эти выходные*\n\n"
    "Команды:\n"
    "  /help — справка\n"
    "  /status — статус задачи"
)

HELP = (
    "🤖 *Исследовательский агент*\n\n"
    "Напишите любой вопрос\\. Агент выполнит:\n"
    "1\\. Формирование поисковых запросов\n"
    "2\\. Поиск и чтение страниц \\(проход 1\\)\n"
    "3\\. Анализ, определение пробелов\n"
    "4\\. Уточняющий поиск \\(проход 2\\)\n"
    "5\\. Синтез финального ответа\n\n"
    "Прогресс будет приходить сообщениями по мере выполнения\\.\n\n"
    "Команды:\n"
    "  /start — начало\n"
    "  /help — эта справка\n"
    "  /status — текущий этап\n"
    "  /cancel — сбросить состояние"
)


# ---------------------------------------------------------------------------
# Синхронный runner для оркестратора (запускается в executor-потоке)
# ---------------------------------------------------------------------------

def _run_orchestrator_sync(task: str, chat_id: str) -> str:
    """
    Запустить ResearchOrchestrator синхронно.

    Вызывается из ThreadPoolExecutor, поэтому asyncio.run() внутри
    MCPClient работает корректно (каждый поток создаёт свой event loop).
    """
    try:
        from mcp_client.client import MCPClient
        from mcp_client.config import MCPServerConfig
        from mcp_server.llm_client import create_llm_fn
        from orchestrator.research_orchestrator import ResearchOrchestrator

        def _mk(module: str, name: str) -> MCPClient:
            return MCPClient(MCPServerConfig(
                name=name, transport="stdio", description=name,
                command="python", args=["-m", module],
            ))

        clients = {
            "search":   _mk("mcp_server.search_server",   "search_server"),
            "scraper":  _mk("mcp_server.scraper_server",  "scraper_server"),
            "telegram": _mk("mcp_server.telegram_server", "telegram_server"),
            "journal":  _mk("mcp_server.journal_server",  "journal_server"),
        }
        llm_fn = create_llm_fn(timeout=90.0)
        orchestrator = ResearchOrchestrator(
            mcp_clients=clients,
            llm_fn=llm_fn,
            verbose=True,
        )
        _state.last_task_id = orchestrator.context.task_id

        result = orchestrator.run(task, chat_id=chat_id)

        _state.last_state = str(orchestrator.state.value)
        _state.last_result = result
        return result

    except Exception as exc:
        logger.exception("Ошибка оркестратора")
        return f"❌ Ошибка: {exc}"
    finally:
        _state.release()


# ---------------------------------------------------------------------------
# Хендлеры PTB (async)
# ---------------------------------------------------------------------------

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    await update.message.reply_text(WELCOME, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return
    await update.message.reply_text(HELP, parse_mode=ParseMode.MARKDOWN_V2)


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        return
    if _state.busy:
        await update.message.reply_text(
            f"⏳ Выполняется: *{_state.current_task[:80]}*\n"
            f"Этап: {_state.last_state or 'инициализация'}",
            parse_mode=ParseMode.MARKDOWN,
        )
    else:
        state = _state.last_state or "нет активной задачи"
        await update.message.reply_text(f"✅ Состояние: `{state}`")


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if not _is_allowed(chat_id):
        return
    if _state.busy:
        _state.release()
        await update.message.reply_text("🛑 Сброшено. Можно начать новое исследование.")
    else:
        await update.message.reply_text("Нет активной задачи.")


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработать текстовое сообщение → запустить исследование."""
    chat_id = update.effective_chat.id
    task = (update.message.text or "").strip()

    if not _is_allowed(chat_id):
        await update.message.reply_text("⛔ Доступ запрещён.")
        return

    if not task:
        return

    if not _state.acquire(task, chat_id):
        await update.message.reply_text(
            f"⏳ Уже выполняется: *{_state.current_task[:60]}*\n"
            "Дождитесь завершения или /cancel",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    await update.message.reply_text(
        f"🎬 Принял задачу:\n_{task}_\n\nНачинаю исследование\\.\\.\\.",
        parse_mode=ParseMode.MARKDOWN_V2,
    )

    # Запускаем синхронный оркестратор в пуле потоков.
    # run_in_executor не блокирует event loop PTB.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        _run_orchestrator_sync,
        task,
        str(chat_id),
    )

    # telegram_server MCP уже должен был отправить результат через send_result.
    # Если токен не задан или произошла ошибка — отправляем напрямую как fallback.
    if result.startswith("❌"):
        await update.message.reply_text(result)
    # Если всё ОК — результат уже доставлен через telegram_server


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def main() -> None:
    if not _PTB_AVAILABLE:
        print("❌ python-telegram-bot не установлен.")
        print("   Установите: pip install 'python-telegram-bot>=20.0'")
        sys.exit(1)

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("❌ TELEGRAM_BOT_TOKEN не задан в .env")
        print("   Создайте бота у @BotFather и добавьте токен в .env")
        sys.exit(1)

    try:
        from mcp_server.llm_client import create_llm_fn
        create_llm_fn()
        logger.info("LLM-провайдер: ✅")
    except ValueError as exc:
        print(f"❌ LLM недоступен: {exc}")
        print("   Задайте QWEN_API_KEY, OPENAI_API_KEY или ANTHROPIC_API_KEY в .env")
        sys.exit(1)

    allowed = _allowed_ids()
    search_mode = os.environ.get("SEARCH_MODE", "mock")

    logger.info("Запуск бота...")
    logger.info(f"  Белый список: {allowed or 'любой chat_id'}")
    logger.info(f"  SEARCH_MODE: {search_mode}")

    app = (
        Application.builder()
        .token(token)
        .build()
    )

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("help",   cmd_help))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.info("Бот запущен. Остановить: Ctrl+C")
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,   # пропустить старые сообщения при старте
    )


if __name__ == "__main__":
    main()
