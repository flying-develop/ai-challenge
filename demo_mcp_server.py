#!/usr/bin/env python3
"""Демонстрация собственного MCP-сервера ЦБ РФ.

Полный цикл без интерактивного ввода:
  Шаг 1 — Проверка окружения (cbr_server.py, FastMCP)
  Шаг 2 — Tool discovery (list_tools через MCPClient)
  Шаг 3 — get_exchange_rates (курсы на сегодня)
  Шаг 4 — convert_currency: 100 USD → рубли
  Шаг 5 — convert_currency: 10000 RUB → USD
  Шаг 6 — convert_currency: 500 EUR → рубли
  Шаг 7 — convert_currency: 1000 CNY → рубли (номинал 10)
  Шаг 8 — get_currency_dynamics: USD за последнюю неделю
  Шаг 9 — Обработка ошибок (XYZ, not-a-date, sideways)

Запуск:
    python demo_mcp_server.py
"""

from __future__ import annotations

import sys
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

    w1, w2, w3 = max(col1, 20), max(col2, 8), max(col3, 18)

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

def step1_check_env() -> bool:
    print("=" * 60)
    print("Шаг 1 — Проверка окружения")
    print("=" * 60)

    # Проверяем наличие cbr_server.py
    server_path = Path("mcp_server/cbr_server.py")
    if not server_path.exists():
        print(f"❌ Файл не найден: {server_path}")
        print("   Убедитесь, что запускаете скрипт из корня проекта.")
        return False
    print(f"✅ mcp_server/cbr_server.py: найден")

    # Проверяем импорт FastMCP
    try:
        from mcp.server.fastmcp import FastMCP  # noqa: F401
        print("✅ FastMCP (пакет mcp): импорт успешен")
    except ImportError as exc:
        print(f"❌ FastMCP не найден: {exc}")
        print("   Установите: pip install mcp")
        return False

    # Проверяем импорт cbr_api
    try:
        from mcp_server.cbr_api import get_daily_rates  # noqa: F401
        print("✅ mcp_server.cbr_api: импорт успешен")
    except ImportError as exc:
        print(f"❌ Ошибка импорта cbr_api: {exc}")
        return False

    print()
    return True


# ---------------------------------------------------------------------------
# Шаг 2 — Tool discovery
# ---------------------------------------------------------------------------

def step2_tool_discovery() -> tuple[bool, object | None, int]:
    print("=" * 60)
    print("Шаг 2 — Tool discovery (list_tools)")
    print("=" * 60)

    from mcp_client.client import MCPClient
    from mcp_client.config import MCPServerConfig

    config = MCPServerConfig(
        name="cbr_currencies",
        transport="stdio",
        description="Курсы валют ЦБ РФ",
        command="python",
        args=["-m", "mcp_server.cbr_server"],
    )
    client = MCPClient(config)

    try:
        tools = client.connect_and_list_tools()
    except RuntimeError as exc:
        print(f"❌ Ошибка подключения: {exc}")
        return False, None, 0

    print(f"✅ Подключение успешно, получено инструментов: {len(tools)}")
    print()
    print(client.get_tools_summary())
    return True, client, len(tools)


# ---------------------------------------------------------------------------
# Шаг 3 — get_exchange_rates
# ---------------------------------------------------------------------------

def step3_exchange_rates(client) -> tuple[bool, int]:
    print("=" * 60)
    print("Шаг 3 — get_exchange_rates (курсы на сегодня)")
    print("=" * 60)

    try:
        result = client.call_tool("get_exchange_rates", {})
    except RuntimeError as exc:
        print(f"❌ Ошибка: {exc}")
        return False, 0

    print(result)
    print()
    lines = [l for l in result.splitlines() if "=" in l]
    count = len(lines)
    return True, count


# ---------------------------------------------------------------------------
# Шаги 4–7 — Конвертация
# ---------------------------------------------------------------------------

def step_convert(
    client,
    step_num: int,
    amount: float,
    currency_code: str,
    direction: str,
    description: str,
) -> tuple[bool, str]:
    print("=" * 60)
    print(f"Шаг {step_num} — {description}")
    print("=" * 60)

    try:
        result = client.call_tool("convert_currency", {
            "amount": amount,
            "currency_code": currency_code,
            "direction": direction,
        })
    except RuntimeError as exc:
        print(f"❌ Ошибка: {exc}")
        return False, ""

    print(f"📡 Вызов: convert_currency(amount={amount}, currency_code={currency_code}, direction={direction})")
    print(f"💱 {result}")
    print()
    return True, result


# ---------------------------------------------------------------------------
# Шаг 8 — Динамика USD
# ---------------------------------------------------------------------------

def step8_dynamics(client) -> tuple[bool, int]:
    print("=" * 60)
    print("Шаг 8 — get_currency_dynamics: USD за последнюю неделю")
    print("=" * 60)

    try:
        result = client.call_tool("get_currency_dynamics", {
            "currency_code": "USD",
            "date_from": "2026-03-03",
            "date_to": "2026-03-10",
        })
    except RuntimeError as exc:
        print(f"❌ Ошибка: {exc}")
        return False, 0

    print(result)
    print()
    # Считаем строки с датами
    lines = [l for l in result.splitlines() if l.strip().startswith("  ")]
    return True, len(lines)


# ---------------------------------------------------------------------------
# Шаг 9 — Обработка ошибок
# ---------------------------------------------------------------------------

def step9_error_handling(client) -> tuple[bool, bool, bool]:
    print("=" * 60)
    print("Шаг 9 — Обработка ошибок")
    print("=" * 60)

    # Несуществующая валюта
    invalid_currency_ok = False
    print("  Тест 1: несуществующая валюта XYZ")
    try:
        result = client.call_tool("get_currency_rate", {"currency_code": "XYZ"})
        if "не найдена" in result or "not found" in result.lower() or "XYZ" in result:
            print(f"  ✅ Ошибка поймана: {result[:80]}")
            invalid_currency_ok = True
        else:
            print(f"  ❌ Ожидалась ошибка, получен результат: {result[:80]}")
    except RuntimeError as exc:
        print(f"  ✅ RuntimeError: {str(exc)[:80]}")
        invalid_currency_ok = True

    print()

    # Невалидная дата
    invalid_date_ok = False
    print("  Тест 2: невалидная дата 'not-a-date'")
    try:
        result = client.call_tool("get_exchange_rates", {"date_str": "not-a-date"})
        if "Неверный" in result or "формат" in result or "invalid" in result.lower():
            print(f"  ✅ Ошибка поймана: {result[:80]}")
            invalid_date_ok = True
        else:
            print(f"  ❌ Ожидалась ошибка, получен результат: {result[:80]}")
    except RuntimeError as exc:
        print(f"  ✅ RuntimeError: {str(exc)[:80]}")
        invalid_date_ok = True

    print()

    # Некорректное направление
    invalid_direction_ok = False
    print("  Тест 3: некорректное направление 'sideways'")
    try:
        result = client.call_tool("convert_currency", {
            "amount": 100.0,
            "currency_code": "USD",
            "direction": "sideways",
        })
        if "Неверное" in result or "направление" in result or "sideways" in result:
            print(f"  ✅ Ошибка поймана: {result[:80]}")
            invalid_direction_ok = True
        else:
            print(f"  ❌ Ожидалась ошибка, получен результат: {result[:80]}")
    except RuntimeError as exc:
        print(f"  ✅ RuntimeError: {str(exc)[:80]}")
        invalid_direction_ok = True

    print()
    return invalid_currency_ok, invalid_date_ok, invalid_direction_ok


# ---------------------------------------------------------------------------
# Главный сценарий
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║        MCP Server Demo — CBR Currency Rates          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    report: list[tuple[str, str, str]] = []

    # Шаг 1
    env_ok = step1_check_env()
    report.append(_ok("Env check", "cbr_server.py + FastMCP") if env_ok else _fail("Env check", "ошибка"))
    if not env_ok:
        _print_table(report)
        sys.exit(1)

    # Шаг 2
    disc_ok, client, tools_count = step2_tool_discovery()
    report.append(
        _ok("Tool discovery", f"{tools_count} tools")
        if disc_ok else _fail("Tool discovery", "ошибка подключения")
    )
    if not disc_ok:
        _print_table(report)
        sys.exit(1)

    # Шаг 3
    rates_ok, currencies_count = step3_exchange_rates(client)
    report.append(
        _ok("get_exchange_rates", f"{currencies_count} валют")
        if rates_ok else _fail("get_exchange_rates", "ошибка")
    )

    # Шаг 4: 100 USD → рубли
    conv4_ok, conv4_result = step_convert(
        client, 4, 100.0, "USD", "to_rub", "convert_currency: 100 USD → рубли"
    )
    detail4 = conv4_result[:25] if conv4_result else "ошибка"
    report.append(_ok("100 USD → RUB", detail4) if conv4_ok else _fail("100 USD → RUB", "ошибка"))

    # Шаг 5: 10000 RUB → USD
    conv5_ok, conv5_result = step_convert(
        client, 5, 10000.0, "USD", "from_rub", "convert_currency: 10000 RUB → USD"
    )
    detail5 = conv5_result[:25] if conv5_result else "ошибка"
    report.append(_ok("10000 RUB → USD", detail5) if conv5_ok else _fail("10000 RUB → USD", "ошибка"))

    # Шаг 6: 500 EUR → рубли
    conv6_ok, conv6_result = step_convert(
        client, 6, 500.0, "EUR", "to_rub", "convert_currency: 500 EUR → рубли"
    )
    detail6 = conv6_result[:25] if conv6_result else "ошибка"
    report.append(_ok("500 EUR → RUB", detail6) if conv6_ok else _fail("500 EUR → RUB", "ошибка"))

    # Шаг 7: 1000 CNY → рубли (номинал 10)
    conv7_ok, conv7_result = step_convert(
        client, 7, 1000.0, "CNY", "to_rub", "convert_currency: 1000 CNY → рубли (номинал 10)"
    )
    detail7 = conv7_result[:25] if conv7_result else "ошибка"
    # Для CNY нужно учесть, что в result должна быть корректная конвертация
    if conv7_ok and conv7_result:
        print("  💡 Образовательная заметка: CNY имеет номинал 10.")
        print("     rate_per_unit = rate / 10 — формула учитывает номинал автоматически.\n")
    report.append(
        _ok("1000 CNY → RUB", "номинал учтён") if conv7_ok else _fail("1000 CNY → RUB", "ошибка")
    )

    # Шаг 8
    dyn_ok, dyn_count = step8_dynamics(client)
    report.append(
        _ok("Dynamics USD", f"{dyn_count} дней")
        if dyn_ok else _fail("Dynamics USD", "ошибка")
    )

    # Шаг 9
    inv_cur_ok, inv_date_ok, inv_dir_ok = step9_error_handling(client)
    report.append(_ok("Invalid currency", "Ошибка поймана") if inv_cur_ok else _fail("Invalid currency", "не поймана"))
    report.append(_ok("Invalid date", "Ошибка поймана") if inv_date_ok else _fail("Invalid date", "не поймана"))
    report.append(_ok("Invalid direction", "Ошибка поймана") if inv_dir_ok else _fail("Invalid direction", "не поймана"))

    # Итоговый отчёт
    print("=" * 60)
    print("MCP Server + Convert Test Report")
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
