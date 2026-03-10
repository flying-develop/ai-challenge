"""MCP-сервер: Курсы валют ЦБ РФ.

Образовательная концепция: этот модуль показывает как создать MCP-сервер,
который оборачивает внешнее API. Любой MCP-клиент сможет подключиться к нему
и обнаружить инструменты автоматически (tool discovery).

Ключевые паттерны:
- FastMCP("CBR Currency Server") — создаёт сервер с именем
- @mcp.tool() — регистрирует функцию как MCP-инструмент
- Типы аргументов + docstring → inputSchema (автоматически)
- mcp.run() запускает stdio-транспорт (stdin/stdout JSON-RPC)

Запуск:
    python -m mcp_server.cbr_server
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from mcp_server.cbr_api import get_daily_rates as _get_daily_rates
from mcp_server.cbr_api import get_currency_dynamics as _get_currency_dynamics

mcp = FastMCP("CBR Currency Server")


# ---------------------------------------------------------------------------
# Инструмент 1: Все курсы на дату
# ---------------------------------------------------------------------------

@mcp.tool()
def get_exchange_rates(date_str: str = "") -> str:
    """
    Получить курсы валют ЦБ РФ на указанную дату.

    Args:
        date_str: Дата в формате YYYY-MM-DD (необязательно, по умолчанию — сегодня)

    Returns:
        Таблица курсов валют: код, название, номинал, курс
    """
    try:
        rates = _get_daily_rates(date_str)
    except ValueError as exc:
        return str(exc)
    except ConnectionError as exc:
        return str(exc)

    if not rates:
        return "Данные о курсах не получены."

    # Берём дату из первой записи
    sample = next(iter(rates.values()))
    date_label = sample.get("date", date_str or "сегодня")

    lines = [f"Курсы валют ЦБ РФ на {date_label}:"]
    for char_code, info in sorted(rates.items()):
        nominal = info["nominal"]
        rate = info["rate"]
        name = info["name"]
        if nominal == 1:
            lines.append(f"  {char_code} ({name}): 1 = {rate:.4f} ₽")
        else:
            lines.append(f"  {char_code} ({name}): {nominal} = {rate:.4f} ₽")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 2: Курс конкретной валюты на дату
# ---------------------------------------------------------------------------

@mcp.tool()
def get_currency_rate(currency_code: str, date_str: str = "") -> str:
    """
    Получить курс конкретной валюты на дату.

    Args:
        currency_code: Буквенный код валюты (USD, EUR, CNY и т.д.)
        date_str: Дата в формате YYYY-MM-DD (необязательно, по умолчанию — сегодня)

    Returns:
        Курс указанной валюты к рублю
    """
    try:
        rates = _get_daily_rates(date_str)
    except ValueError as exc:
        return str(exc)
    except ConnectionError as exc:
        return str(exc)

    code = currency_code.upper().strip()
    if code not in rates:
        available = sorted(rates.keys())
        return (
            f"Валюта '{code}' не найдена. "
            f"Доступные: {', '.join(available[:10])}"
            f"{'...' if len(available) > 10 else ''}"
        )

    info = rates[code]
    nominal = info["nominal"]
    rate = info["rate"]
    name = info["name"]
    date_label = info.get("date", date_str or "сегодня")
    rate_per_unit = rate / nominal

    return (
        f"{code} ({name}) на {date_label}:\n"
        f"  Номинал: {nominal}\n"
        f"  Курс: {rate:.4f} ₽ за {nominal} {code}\n"
        f"  Курс за 1 {code}: {rate_per_unit:.4f} ₽"
    )


# ---------------------------------------------------------------------------
# Инструмент 3: Динамика курса за период
# ---------------------------------------------------------------------------

@mcp.tool()
def get_currency_dynamics(
    currency_code: str,
    date_from: str,
    date_to: str,
) -> str:
    """
    Получить динамику курса валюты за период.

    Args:
        currency_code: Буквенный код валюты (USD, EUR, CNY)
        date_from: Начало периода, YYYY-MM-DD
        date_to: Конец периода, YYYY-MM-DD

    Returns:
        Таблица курсов по дням за указанный период
    """
    try:
        records = _get_currency_dynamics(currency_code, date_from, date_to)
    except ValueError as exc:
        return str(exc)
    except ConnectionError as exc:
        return str(exc)

    code = currency_code.upper().strip()

    if not records:
        return (
            f"Нет данных по {code} за период {date_from} — {date_to}.\n"
            f"Проверьте даты: ЦБ публикует курсы по рабочим дням."
        )

    lines = [f"Динамика курса {code} с {date_from} по {date_to} ({len(records)} записей):"]
    for rec in records:
        nominal = rec["nominal"]
        rate = rec["rate"]
        date = rec["date"]
        rate_per_unit = rate / nominal
        if nominal == 1:
            lines.append(f"  {date}: {rate:.4f} ₽")
        else:
            lines.append(f"  {date}: {rate:.4f} ₽ за {nominal} {code} ({rate_per_unit:.4f} ₽/шт.)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Инструмент 4: Конвертация
# ---------------------------------------------------------------------------

@mcp.tool()
def convert_currency(
    amount: float,
    currency_code: str,
    direction: str = "to_rub",
) -> str:
    """
    Конвертировать сумму между валютой и рублями по курсу ЦБ РФ.

    Args:
        amount: Сумма для конвертации
        currency_code: Буквенный код валюты (USD, EUR, CNY и т.д.)
        direction: Направление конвертации:
            "to_rub" — из валюты в рубли (по умолчанию)
            "from_rub" — из рублей в валюту

    Returns:
        Результат конвертации с указанием курса
    """
    if direction not in ("to_rub", "from_rub"):
        return (
            f"Неверное направление '{direction}'. "
            f"Используйте: 'to_rub' (валюта → рубли) или 'from_rub' (рубли → валюта)"
        )

    try:
        rates = _get_daily_rates()
    except ConnectionError as exc:
        return str(exc)

    code = currency_code.upper().strip()
    if code not in rates:
        available = sorted(rates.keys())
        return (
            f"Валюта '{code}' не найдена. "
            f"Доступные: {', '.join(available[:10])}"
            f"{'...' if len(available) > 10 else ''}"
        )

    info = rates[code]
    nominal = info["nominal"]
    rate = info["rate"]
    name = info["name"]

    # rate_per_unit — курс за 1 единицу валюты
    # Например: CNY номинал 10, rate=122.45 → rate_per_unit=12.245
    rate_per_unit = rate / nominal

    if direction == "to_rub":
        result = amount * rate_per_unit
        amount_fmt = f"{amount:,.2f}".replace(",", " ")
        result_fmt = f"{result:,.2f}".replace(",", " ")
        return (
            f"{amount_fmt} {code} = {result_fmt} ₽ "
            f"(курс ЦБ: {rate_per_unit:.4f} за 1 {code})"
        )
    else:  # from_rub
        result = amount / rate_per_unit
        amount_fmt = f"{amount:,.2f}".replace(",", " ")
        result_fmt = f"{result:,.2f}".replace(",", " ")
        return (
            f"{amount_fmt} ₽ = {result_fmt} {code} "
            f"(курс ЦБ: {rate_per_unit:.4f} за 1 {code})"
        )


if __name__ == "__main__":
    mcp.run()  # запуск в stdio-режиме (по умолчанию)
