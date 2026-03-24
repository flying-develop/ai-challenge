"""AdminCLI: административный CLI для управления support-ботом.

Запускается параллельно с TelegramListener.
Предназначен для администратора (не для пользователей бота).

Команды:
    /index path|url|github ... — индексация документации
    /index status              — статус текущего индекса
    /mode local|cloud|status   — переключение режима провайдеров
    /users                     — активные пользователи
    /logs [N]                  — последние N запросов
    /stats                     — статистика
    /quit                      — остановить бота
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from .session_store import SessionStore


class AdminCLI:
    """Административный CLI для управления support-ботом.

    Запускается в основном потоке, TelegramListener — в фоновом.

    Args:
        session_store:     Хранилище сессий пользователей.
        index_fn:          Функция индексации (source_type, source_arg) → None.
        status_fn:         Функция получения статуса индекса () → str.
        stop_fn:           Функция остановки TelegramListener.
        log_capacity:      Максимальное количество записей в логе.
    """

    def __init__(
        self,
        session_store: SessionStore,
        index_fn: Callable[[str, str], None],
        status_fn: Callable[[], str],
        stop_fn: Callable[[], None],
        log_capacity: int = 100,
        mode_switch_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.store = session_store
        self._index_fn = index_fn
        self._status_fn = status_fn
        self._stop_fn = stop_fn
        self._mode_switch_fn = mode_switch_fn
        self._logs: deque[dict] = deque(maxlen=log_capacity)
        self._start_time = time.time()

    def add_log(self, entry: dict) -> None:
        """Добавить запись в лог (вызывается из TelegramListener).

        Args:
            entry: {chat_id, username, question, confidence}
        """
        entry["ts"] = datetime.now().strftime("%H:%M:%S")
        self._logs.append(entry)

    def run(self) -> None:
        """Запустить интерактивный CLI в основном потоке.

        Читает команды из stdin до /quit.
        """
        current_mode = os.environ.get("LLM_MODE", "cloud").upper()
        print(f"🔧 Admin CLI. Команды: /index, /mode, /users, /logs, /stats, /quit")
        print(f"   Режим: {current_mode}")
        print("   /index path ./docs       — индексировать локальные .md файлы")
        print("   /index url https://...   — загрузить документацию по URL")
        print("   /index github https://... — клонировать и индексировать репо")
        print("   /index clear             — очистить индекс")
        print("   /mode local              — переключить на Ollama (локальный стек)")
        print("   /mode cloud              — переключить на облачные API")
        print("   /mode status             — показать текущий режим и провайдеры")
        print()

        while True:
            try:
                cmd = input("admin> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Admin] Завершение работы...")
                self._stop_fn()
                break

            if not cmd:
                continue

            if cmd.startswith("/index"):
                self._handle_index(cmd)
            elif cmd.startswith("/mode"):
                self._handle_mode(cmd)
            elif cmd == "/users":
                self._show_active_users()
            elif cmd.startswith("/logs"):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 20
                self._show_logs(n)
            elif cmd == "/stats":
                self._show_stats()
            elif cmd == "/help":
                self._show_help()
            elif cmd == "/quit":
                print("[Admin] Остановка бота...")
                self._stop_fn()
                break
            else:
                print(f"Неизвестная команда: {cmd}. Введите /help.")

    # ------------------------------------------------------------------
    # Обработчики команд
    # ------------------------------------------------------------------

    def _handle_mode(self, cmd: str) -> None:
        """Обработать команду /mode.

        Подкоманды:
            /mode local   — переключить на Ollama (локальный стек)
            /mode cloud   — переключить на облачные API
            /mode status  — показать текущий режим и провайдеры

        ВАЖНО: при смене режима эмбеддинги несовместимы — предлагает переиндексировать.

        Args:
            cmd: Строка команды, например "/mode local"
        """
        parts = cmd.split()
        if len(parts) < 2 or parts[1] not in ("local", "cloud", "status"):
            print("Использование: /mode local|cloud|status")
            return

        subcommand = parts[1]
        current_mode = os.environ.get("LLM_MODE", "cloud").lower()

        if subcommand == "status":
            self._show_mode_status()
            return

        if subcommand == current_mode:
            print(f"[Mode] Уже в режиме {current_mode.upper()}.")
            return

        # Предупреждение о несовместимости эмбеддингов
        print(f"[Mode] Переключение: {current_mode.upper()} → {subcommand.upper()}")
        print("[Mode] ВНИМАНИЕ: эмбеддинги разных моделей несовместимы!")
        print("[Mode] Нужна переиндексация для корректной работы поиска.")
        confirm = input("Переиндексировать после переключения? (y/n): ").strip().lower()

        # Устанавливаем новый режим
        os.environ["LLM_MODE"] = subcommand
        print(f"[Mode] Режим переключён на {subcommand.upper()}.")

        if self._mode_switch_fn:
            result = self._mode_switch_fn(subcommand)
            print(f"[Mode] {result}")

        if confirm == "y":
            print("[Mode] Запустите переиндексацию:")
            if subcommand == "local":
                embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
                print(f"  python rag_indexer/main.py index --docs ./podkop-wiki/content")
                print(f"  --db ./output/index_local.db --embedder ollama")
            else:
                print(f"  python rag_indexer/main.py index --docs ./podkop-wiki/content")
                print(f"  --db ./output/index.db --embedder qwen")
        else:
            print("[Mode] Переиндексация пропущена. Результаты поиска могут быть неточными.")

    def _show_mode_status(self) -> None:
        """Показать текущий режим и провайдеры."""
        mode = os.environ.get("LLM_MODE", "cloud").lower()
        print(f"\n{'─' * 50}")
        print(f"  Текущий режим: {mode.upper()}")
        print(f"{'─' * 50}")

        if mode == "local":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            llm_model = os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5:3b")
            embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
            rerank_model = os.environ.get("OLLAMA_RERANK_MODEL", "qwen2.5:3b")
            print(f"  LLM:        Ollama / {llm_model}")
            print(f"  Embeddings: Ollama / {embed_model}")
            print(f"  Reranker:   Ollama / {rerank_model}")
            print(f"  Ollama URL: {base_url}")
            print(f"  Интернет:   не нужен (кроме Telegram)")
            print(f"  Стоимость:  0 ₽")

            # Быстрая проверка доступности Ollama
            import urllib.request as _urllib
            try:
                _urllib.urlopen(f"{base_url}/api/tags", timeout=2).read()
                print(f"  Ollama:     ✅ доступен")
            except Exception:
                print(f"  Ollama:     ❌ недоступен (запустите: ollama serve)")
        else:
            qwen_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
            cohere_key = os.environ.get("COHERE_API_KEY")
            qwen_model = os.environ.get("QWEN_MODEL", "qwen-plus")
            print(f"  LLM:        Qwen / {qwen_model}")
            print(f"  Embeddings: DashScope / text-embedding-v3")
            print(f"  Reranker:   {'Cohere / rerank-v3.5' if cohere_key else 'ThresholdFilter'}")
            print(f"  API keys:   QWEN={'✅' if qwen_key else '❌'}  COHERE={'✅' if cohere_key else '❌'}")
            print(f"  Интернет:   нужен")

        print(f"{'─' * 50}\n")

    def _handle_index(self, cmd: str) -> None:
        """Обработать команду /index.

        Args:
            cmd: Строка команды, например "/index path ./docs"
        """
        parts = cmd.split(maxsplit=2)

        if len(parts) < 2:
            print("Использование: /index path|url|github <аргумент>")
            print("               /index status")
            return

        subcommand = parts[1]

        if subcommand == "status":
            print(self._status_fn())
            return

        if subcommand == "clear":
            confirm = input("Удалить весь индекс? (yes/no): ").strip().lower()
            if confirm == "yes":
                self._index_fn("clear", "")
                print("[Admin] Индекс очищен.")
            else:
                print("[Admin] Отменено.")
            return

        if len(parts) < 3:
            print(f"Использование: /index {subcommand} <путь или URL>")
            return

        source_arg = parts[2]

        if subcommand not in ("path", "url", "github"):
            print(f"Неизвестный тип источника: {subcommand}")
            print("Допустимые: path, url, github, status, clear")
            return

        print(f"[Admin] Запуск индексации: {subcommand} {source_arg}")
        print("  Это может занять несколько минут...")

        try:
            self._index_fn(subcommand, source_arg)
            print(f"[Admin] Индексация завершена: {subcommand} {source_arg}")
        except Exception as exc:
            print(f"[Admin] Ошибка индексации: {exc}")

    def _show_active_users(self) -> None:
        """Показать активных пользователей."""
        sessions = self.store.get_active_sessions()

        if not sessions:
            print("Нет активных пользователей")
            return

        print(f"\n{'chat_id':<15} {'username':<20} {'сообщений':<10} {'последнее'}")
        print("─" * 60)
        for s in sessions:
            last = datetime.fromtimestamp(s.last_seen).strftime("%H:%M:%S")
            print(f"{s.chat_id:<15} {s.username:<20} {s.message_count:<10} {last}")
        print()

    def _show_logs(self, n: int = 20) -> None:
        """Показать последние N запросов.

        Args:
            n: Количество записей для отображения.
        """
        logs = list(self._logs)[-n:]

        if not logs:
            print("Лог запросов пуст")
            return

        print(f"\nПоследние {len(logs)} запросов:")
        print(f"{'Время':<10} {'username':<15} {'confidence':<12} {'вопрос'}")
        print("─" * 80)
        for entry in logs:
            ts = entry.get("ts", "?")
            username = entry.get("username", "?")[:14]
            confidence = f"{entry.get('confidence', 0):.2f}"
            question = entry.get("question", "")[:45]
            print(f"{ts:<10} {username:<15} {confidence:<12} {question}")
        print()

    def _show_stats(self) -> None:
        """Показать статистику бота."""
        uptime = time.time() - self._start_time
        uptime_str = _format_duration(uptime)

        total_sessions = len(self.store)
        active_sessions = len(self.store.get_active_sessions())

        logs = list(self._logs)
        total_requests = len(logs)
        avg_confidence = (
            sum(e.get("confidence", 0) for e in logs) / total_requests
            if total_requests > 0 else 0.0
        )

        print(f"\n{'─' * 40}")
        print(f"  Support Bot Statistics")
        print(f"{'─' * 40}")
        print(f"  Uptime:             {uptime_str}")
        print(f"  Всего сессий:       {total_sessions}")
        print(f"  Активных сессий:    {active_sessions}")
        print(f"  Запросов в логе:    {total_requests}")
        print(f"  Средний confidence: {avg_confidence:.2f}")
        print(f"{'─' * 40}\n")

        # Статус индекса
        print(self._status_fn())

    def _show_help(self) -> None:
        """Показать справку по командам."""
        print("""
Административные команды:
  /index path ./docs           — индексировать локальные .md файлы
  /index url https://...       — загрузить документацию по URL
  /index github https://...    — клонировать репозиторий, индексировать .md
  /index status                — статус текущего индекса
  /index clear                 — очистить индекс (с подтверждением)

  /mode local                  — переключить на Ollama (локальный стек)
  /mode cloud                  — переключить на облачные API
  /mode status                 — показать текущий режим и провайдеры

  /users                       — активные пользователи
  /logs [N]                    — последние N запросов (default: 20)
  /stats                       — общая статистика

  /quit                        — остановить бота
""")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Форматировать длительность в человекочитаемый вид.

    Args:
        seconds: Длительность в секундах.

    Returns:
        Строка вида "2h 15m 30s".
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)
