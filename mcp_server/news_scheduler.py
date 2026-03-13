"""Планировщик новостного пайплайна.

Образовательная концепция:
- Простой цикл while True + time.sleep (без cron, без APScheduler)
- Вызов пайплайна через MCPClient (stdio) — образовательный паттерн
- Рабочие часы МСК (09:00-17:00) — пайплайн не запускается вне окна
- Конфигурация через .env

Запуск:
    python -m mcp_server.news_scheduler

Переменные окружения:
    NEWS_POLL_INTERVAL_MINUTES  — интервал в минутах (по умолчанию 60)
    NEWS_WORK_HOURS_START       — начало рабочего дня МСК (по умолчанию 9)
    NEWS_WORK_HOURS_END         — конец рабочего дня МСК (по умолчанию 17)

Примечание:
    Для вызова пайплайна используется MCPClient + stdio-транспорт.
    Это означает, что каждый вызов запускает news_server.py как подпроцесс,
    выполняет run_news_pipeline и завершает процесс.
    Альтернативный подход (прямой импорт) показан в комментарии ниже.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Московское время (UTC+3)
MSK = timezone(timedelta(hours=3))


def _get_env_int(key: str, default: int) -> int:
    """Прочитать целочисленное значение из .env."""
    try:
        return int(os.environ.get(key, default))
    except (ValueError, TypeError):
        return default


def _get_news_mcp_client():
    """Создать MCPClient для news_digest сервера.

    Использует конфигурацию из config/mcp-servers.md.
    Если конфигурация недоступна — создаёт MCPClient напрямую.

    Returns:
        MCPClient готовый к вызову run_news_pipeline.
    """
    # Добавляем корень проекта в sys.path для корректного импорта
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from mcp_client.client import MCPClient
    from mcp_client.config import MCPServerConfig

    # Пробуем загрузить конфигурацию из mcp-servers.md
    try:
        from mcp_client.config import MCPConfigParser
        parser = MCPConfigParser()
        servers = parser.load()
        for srv in servers:
            if srv.name == "news_digest":
                return MCPClient(srv)
    except Exception:
        pass

    # Fallback: создаём конфигурацию вручную
    config = MCPServerConfig(
        name="news_digest",
        transport="stdio",
        command="python",
        args=["-m", "mcp_server.news_server"],
        env={},
        description="News Pipeline MCP Server",
    )
    return MCPClient(config)


def _run_pipeline_via_mcp() -> str:
    """Запустить пайплайн через MCPClient (stdio).

    Образовательный паттерн: планировщик не импортирует news_server напрямую,
    а обращается к нему как к внешнему MCP-сервису.
    Это обеспечивает полную изоляцию и проверку MCP-интеграции.

    Returns:
        Результат выполнения run_news_pipeline.
    """
    client = _get_news_mcp_client()
    return client.call_tool("run_news_pipeline", {})


def _run_pipeline_direct() -> str:
    """Альтернатива: запустить пайплайн напрямую (без MCP overhead).

    Используется как fallback если MCPClient недоступен.
    Образовательная заметка: прямой импорт быстрее, но не проверяет
    MCP-интеграцию.

    Returns:
        Результат выполнения run_news_pipeline.
    """
    from mcp_server.news_server import run_news_pipeline
    return run_news_pipeline()


def main() -> None:
    """Основной цикл планировщика.

    Работает бесконечно (while True). Завершается только по Ctrl+C
    или сигналу SIGTERM.

    Логика:
    1. Проверить рабочие часы (start_hour <= now.hour < end_hour)
    2. Если рабочее время — запустить пайплайн
    3. Подождать interval минут
    4. Повторить
    """
    interval = _get_env_int("NEWS_POLL_INTERVAL_MINUTES", 60)
    start_hour = _get_env_int("NEWS_WORK_HOURS_START", 9)
    end_hour = _get_env_int("NEWS_WORK_HOURS_END", 17)

    print("📡 Планировщик новостей запущен")
    print(f"   Интервал: {interval} мин")
    print(f"   Рабочие часы: {start_hour:02d}:00–{end_hour:02d}:00 МСК")
    print(f"   Запуск: через MCPClient → news_server.run_news_pipeline")
    print()

    while True:
        now_msk = datetime.now(MSK)
        time_str = now_msk.strftime("%H:%M")

        if start_hour <= now_msk.hour < end_hour:
            print(f"[{time_str} МСК] Запуск пайплайна...")

            # Основной способ: через MCPClient
            try:
                result = _run_pipeline_via_mcp()
                print(f"[{time_str} МСК] ✅ Пайплайн завершён:")
                # Выводим первую строку результата
                first_line = result.split("\n")[0] if result else "(нет ответа)"
                print(f"   {first_line}")
            except RuntimeError as exc:
                # MCPClient вернул ошибку → пробуем прямой вызов
                print(f"[{time_str} МСК] ⚠️  MCPClient: {exc}")
                print(f"[{time_str} МСК] Пробуем прямой вызов...")
                try:
                    result = _run_pipeline_direct()
                    first_line = result.split("\n")[0] if result else "(нет ответа)"
                    print(f"[{time_str} МСК] ✅ {first_line}")
                except Exception as exc2:
                    print(f"[{time_str} МСК] ❌ Ошибка: {exc2}")
            except Exception as exc:
                print(f"[{time_str} МСК] ❌ Ошибка подключения: {exc}")
        else:
            print(f"[{time_str} МСК] Нерабочее время, пропуск")

        # Ожидание до следующего запуска
        sleep_seconds = interval * 60
        print(f"   Следующий запуск через {interval} мин (~{now_msk.strftime('%H:%M')})")
        time.sleep(sleep_seconds)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n📡 Планировщик остановлен.")
        sys.exit(0)
