#!/usr/bin/env python3
"""Демонстрация подключения к MCP-серверу markdownify.

Скрипт проверяет окружение, загружает конфигурацию, подключается к серверу,
получает список инструментов и демонстрирует обработку ошибок.

Запуск:
    python demo_mcp_connect.py

Предварительные требования:
    1. Node.js >= 18 установлен
    2. markdownify-mcp склонирован и собран:
       git clone https://github.com/zcaceres/markdownify-mcp.git
       cd markdownify-mcp && pnpm install && pnpm run build
    3. В .env задан MARKDOWNIFY_MCP_PATH=/path/to/markdownify-mcp
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Утилиты для вывода
# ---------------------------------------------------------------------------

def _ok(label: str, detail: str = "") -> tuple[str, str, str]:
    return label, "✅", detail


def _fail(label: str, detail: str = "") -> tuple[str, str, str]:
    return label, "❌", detail


def _print_table(rows: list[tuple[str, str, str]]) -> None:
    """Вывести итоговую таблицу результатов."""
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    col3 = max(len(r[2]) for r in rows)

    w1, w2, w3 = max(col1, 14), max(col2, 8), max(col3, 16)

    sep_top  = f"┌{'─' * (w1 + 2)}┬{'─' * (w2 + 2)}┬{'─' * (w3 + 2)}┐"
    sep_head = f"├{'─' * (w1 + 2)}┼{'─' * (w2 + 2)}┼{'─' * (w3 + 2)}┤"
    sep_bot  = f"└{'─' * (w1 + 2)}┴{'─' * (w2 + 2)}┴{'─' * (w3 + 2)}┘"

    def row_line(label: str, status: str, detail: str) -> str:
        return f"│ {label:<{w1}} │ {status:<{w2}} │ {detail:<{w3}} │"

    print()
    print(sep_top)
    print(row_line("Тест", "Статус", "Детали"))
    print(sep_head)
    for label, status, detail in rows:
        print(row_line(label, status, detail))
    print(sep_bot)
    print()


# ---------------------------------------------------------------------------
# Шаг 1 — Проверка окружения
# ---------------------------------------------------------------------------

def step1_check_env() -> tuple[bool, str, str, str]:
    """
    Проверяет: node, MARKDOWNIFY_MCP_PATH, dist/index.js.
    Возвращает (ok, node_ver, mcp_path, index_path).
    """
    print("=" * 60)
    print("Шаг 1 — Проверка окружения")
    print("=" * 60)

    # Node.js
    node_bin = shutil.which("node")
    if not node_bin:
        print("❌ Node.js не найден")
        print("   Установите Node.js: https://nodejs.org/")
        return False, "", "", ""

    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5
        )
        node_ver = result.stdout.strip()
    except Exception as exc:
        print(f"❌ Не удалось получить версию Node.js: {exc}")
        return False, "", "", ""

    print(f"✅ Node.js: {node_ver}")

    # MARKDOWNIFY_MCP_PATH
    mcp_path = os.environ.get("MARKDOWNIFY_MCP_PATH", "").strip()
    if not mcp_path:
        print("❌ MARKDOWNIFY_MCP_PATH не задан")
        print("   Добавьте в .env: MARKDOWNIFY_MCP_PATH=/path/to/markdownify-mcp")
        return False, node_ver, "", ""

    print(f"✅ MARKDOWNIFY_MCP_PATH: {mcp_path}")

    # dist/index.js
    index_js = Path(mcp_path) / "dist" / "index.js"
    if not index_js.exists():
        print(f"❌ dist/index.js не найден: {index_js}")
        print("   Соберите markdownify-mcp:")
        print(f"     cd {mcp_path} && pnpm install && pnpm run build")
        return False, node_ver, mcp_path, ""

    print(f"✅ dist/index.js: найден")
    print()
    return True, node_ver, mcp_path, str(index_js)


# ---------------------------------------------------------------------------
# Шаг 2 — Загрузка конфигурации
# ---------------------------------------------------------------------------

def step2_load_config() -> tuple[bool, list]:
    """Читает config/mcp-servers.md и парсит серверы."""
    print("=" * 60)
    print("Шаг 2 — Загрузка конфигурации")
    print("=" * 60)

    from mcp_client.config import MCPConfigParser

    try:
        parser = MCPConfigParser()
        servers = parser.load()
    except EnvironmentError as exc:
        print(f"❌ Ошибка конфигурации: {exc}")
        return False, []

    if not servers:
        print("❌ Серверы не найдены в config/mcp-servers.md")
        return False, []

    for s in servers:
        print(f"✅ Найден сервер \"{s.name}\", transport={s.transport}")

    print()
    return True, servers


# ---------------------------------------------------------------------------
# Шаг 3 — Подключение и tool discovery
# ---------------------------------------------------------------------------

def step3_connect(servers: list) -> tuple[bool, object | None, float]:
    """Подключается к первому серверу и получает список инструментов."""
    print("=" * 60)
    print("Шаг 3 — Подключение и tool discovery")
    print("=" * 60)

    from mcp_client.client import MCPClient

    config = servers[0]
    client = MCPClient(config)

    start = time.monotonic()
    try:
        tools = client.connect_and_list_tools()
        elapsed = time.monotonic() - start
    except RuntimeError as exc:
        elapsed = time.monotonic() - start
        print(exc)
        return False, None, elapsed

    print(f"✅ Соединение установлено за {elapsed:.1f}с")
    print(f"   Получено инструментов: {len(tools)}")
    print()
    return True, client, elapsed


# ---------------------------------------------------------------------------
# Шаг 4 — Вывод списка инструментов
# ---------------------------------------------------------------------------

def step4_show_tools(client) -> None:
    """Выводит все инструменты через get_tools_summary."""
    print("=" * 60)
    print("Шаг 4 — Список инструментов")
    print("=" * 60)
    print(client.get_tools_summary())


# ---------------------------------------------------------------------------
# Шаг 5 — Детали одного инструмента
# ---------------------------------------------------------------------------

def step5_show_schema(client) -> bool:
    """Показывает полную inputSchema для webpage-to-markdown."""
    print("=" * 60)
    print("Шаг 5 — Детали инструмента (inputSchema)")
    print("=" * 60)

    tools = client.tools
    target = next((t for t in tools if t["name"] == "webpage-to-markdown"), None)
    if not target:
        target = tools[0] if tools else None

    if not target:
        print("❌ Инструменты отсутствуют")
        return False

    print(f"  Инструмент: {target['name']}")
    print(f"  Описание: {target.get('description', '')}")
    print()
    print("  inputSchema:")
    schema_str = json.dumps(target.get("inputSchema", {}), indent=4, ensure_ascii=False)
    for line in schema_str.splitlines():
        print(f"    {line}")

    print()
    print("  💡 Образовательный комментарий:")
    print("  inputSchema — это JSON Schema, описывающая параметры инструмента.")
    print("  LLM использует эту схему, чтобы автоматически формировать")
    print("  правильные вызовы. Это ключевая идея MCP — tool discovery.")
    print()
    return True


# ---------------------------------------------------------------------------
# Шаг 6 — Обработка ошибок
# ---------------------------------------------------------------------------

def step6_error_handling(index_path: str) -> tuple[bool, bool]:
    """Проверяет обработку ошибок: неверная команда и неверный путь."""
    print("=" * 60)
    print("Шаг 6 — Обработка ошибок")
    print("=" * 60)

    from mcp_client.client import MCPClient
    from mcp_client.config import MCPServerConfig

    # Попытка с несуществующей командой
    bad_cmd_ok = False
    bad_config = MCPServerConfig(
        name="test-bad-cmd",
        transport="stdio",
        description="Тест несуществующей команды",
        command="fake-node-binary-xyz",
        args=[index_path],
    )
    client = MCPClient(bad_config)
    try:
        client.connect_and_list_tools()
        print("  ❌ Ожидалась ошибка для несуществующей команды")
    except RuntimeError as exc:
        print(f"  ✅ Ошибка поймана (bad command):")
        for line in str(exc).splitlines():
            print(f"     {line}")
        bad_cmd_ok = True
    print()

    # Попытка с несуществующим файлом
    bad_path_ok = False
    bad_path_config = MCPServerConfig(
        name="test-bad-path",
        transport="stdio",
        description="Тест несуществующего пути",
        command="node",
        args=["/tmp/nonexistent-mcp-server-xyz/dist/index.js"],
    )
    client2 = MCPClient(bad_path_config)
    try:
        client2.connect_and_list_tools()
        print("  ❌ Ожидалась ошибка для несуществующего файла")
    except RuntimeError as exc:
        print(f"  ✅ Ошибка поймана (bad path):")
        for line in str(exc).splitlines():
            print(f"     {line}")
        bad_path_ok = True

    print()
    return bad_cmd_ok, bad_path_ok


# ---------------------------------------------------------------------------
# Главный сценарий
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        MCP Connection Demo — markdownify-mcp         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    report: list[tuple[str, str, str]] = []

    # Шаг 1
    env_ok, node_ver, mcp_path, index_path = step1_check_env()
    report.append(_ok("Node.js", node_ver) if node_ver else _fail("Node.js", "не найден"))
    report.append(_ok("MARKDOWNIFY_MCP_PATH", mcp_path[:30] + "..." if len(mcp_path) > 30 else mcp_path) if mcp_path else _fail("MARKDOWNIFY_MCP_PATH", "не задан"))
    report.append(_ok("Server path", "dist/index.js") if index_path else _fail("Server path", "не найден"))

    if not env_ok:
        _print_table(report)
        sys.exit(1)

    # Шаг 2
    cfg_ok, servers = step2_load_config()
    report.append(_ok("Config load", f"{len(servers)} сервер(а)") if cfg_ok else _fail("Config load", "ошибка"))

    if not cfg_ok:
        _print_table(report)
        sys.exit(1)

    # Шаг 3
    conn_ok, client, elapsed = step3_connect(servers)
    report.append(_ok("Connect+Init", f"{elapsed:.1f}s") if conn_ok else _fail("Connect+Init", f"ошибка ({elapsed:.1f}s)"))

    if not conn_ok:
        _print_table(report)
        sys.exit(1)

    tools_count = len(client.tools)
    report.append(_ok("List tools", f"{tools_count} tools"))

    # Шаг 4
    step4_show_tools(client)

    # Шаг 5
    schema_ok = step5_show_schema(client)
    report.append(_ok("Tool schema", "JSON Schema OK") if schema_ok else _fail("Tool schema", "ошибка"))

    # Шаг 6
    bad_cmd_ok, bad_path_ok = step6_error_handling(index_path)
    report.append(_ok("Bad command", "Ошибка поймана") if bad_cmd_ok else _fail("Bad command", "не поймана"))
    report.append(_ok("Bad path", "Ошибка поймана") if bad_path_ok else _fail("Bad path", "не поймана"))

    # Итоговый отчёт
    print("=" * 60)
    print("MCP Connection Test Report")
    _print_table(report)

    all_ok = all(r[1] == "✅" for r in report)
    if all_ok:
        print("✅ Все тесты прошли успешно!")
    else:
        failed = [r[0] for r in report if r[1] == "❌"]
        print(f"⚠️  Не прошли: {', '.join(failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
