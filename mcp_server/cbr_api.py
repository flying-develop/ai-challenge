"""API ЦБ РФ: HTTP-запросы и парсинг XML.

Образовательная концепция: этот модуль инкапсулирует всю работу с внешним API
и может тестироваться независимо от MCP-сервера.

Правила стека (инварианты):
- HTTP: только urllib.request (stdlib)
- XML: только xml.etree.ElementTree (stdlib)
- Кодировка ответа ЦБ: windows-1251
- Десятичный разделитель в Value: запятая → float через replace(",", ".")
"""

from __future__ import annotations

import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# Базовый URL XML-сервисов ЦБ РФ
_CBR_BASE = "http://www.cbr.ru/scripts"

# Кэш маппинга CharCode → внутренний ID ЦБ (R01235 и т.д.)
# Заполняется при первом вызове get_daily_rates() и обновляется при каждом вызове.
_char_to_id: dict[str, str] = {}


def _fetch_xml(url: str) -> str:
    """Загрузить XML по URL, вернуть декодированную строку (windows-1251).

    Raises:
        ConnectionError: если не удалось подключиться к cbr.ru
    """
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            content = response.read()
        return content.decode("windows-1251")
    except urllib.error.URLError as exc:
        raise ConnectionError(f"Ошибка: не удалось подключиться к cbr.ru: {exc}") from exc
    except OSError as exc:
        raise ConnectionError(f"Ошибка: не удалось подключиться к cbr.ru: {exc}") from exc


def _cbr_date(date_str: str) -> str:
    """Преобразовать YYYY-MM-DD → DD/MM/YYYY для URL ЦБ РФ.

    Если пустая строка — вернуть пустую строку (сервер вернёт курсы на сегодня).

    Raises:
        ValueError: если формат даты неверный
    """
    if not date_str:
        return ""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.strftime("%d/%m/%Y")
    except ValueError:
        raise ValueError(f"Неверный формат даты '{date_str}'. Используйте YYYY-MM-DD")


def get_daily_rates(date_str: str = "") -> dict[str, dict]:
    """Получить курсы всех валют ЦБ РФ на указанную дату.

    Побочный эффект: обновляет кэш _char_to_id.

    Args:
        date_str: Дата YYYY-MM-DD, или "" для сегодняшних курсов

    Returns:
        dict: {CharCode: {"name": str, "nominal": int, "rate": float, "id": str, "date": str}}

    Raises:
        ConnectionError: нет сети
        ValueError: неверный формат даты
    """
    global _char_to_id

    cbr_date = _cbr_date(date_str)
    if cbr_date:
        url = f"{_CBR_BASE}/XML_daily.asp?date_req={cbr_date}"
    else:
        url = f"{_CBR_BASE}/XML_daily.asp"

    xml_str = _fetch_xml(url)
    root = ET.fromstring(xml_str)

    # Дата из атрибута корневого элемента: Date="10.03.2026"
    response_date = root.get("Date", "")

    rates: dict[str, dict] = {}
    for valute in root.findall("Valute"):
        char_code = valute.findtext("CharCode", "").strip()
        val_id = valute.get("ID", "").strip()
        name = valute.findtext("Name", "").strip()
        nominal_str = valute.findtext("Nominal", "1").strip()
        value_str = valute.findtext("Value", "0").strip()

        try:
            nominal = int(nominal_str)
            rate = float(value_str.replace(",", "."))
        except ValueError:
            continue

        if char_code:
            rates[char_code] = {
                "name": name,
                "nominal": nominal,
                "rate": rate,
                "id": val_id,
                "date": response_date,
            }
            _char_to_id[char_code] = val_id

    return rates


def get_currency_id(currency_code: str) -> str:
    """Получить внутренний ID валюты по буквенному коду (с кэшированием).

    Args:
        currency_code: Буквенный код (USD, EUR, CNY, ...)

    Returns:
        Внутренний ID ЦБ (R01235, R01239, ...)

    Raises:
        ConnectionError: нет сети
        ValueError: валюта не найдена
    """
    code = currency_code.upper().strip()
    if code not in _char_to_id:
        get_daily_rates()  # заполнить кэш
    if code not in _char_to_id:
        available = sorted(_char_to_id.keys())
        raise ValueError(
            f"Валюта '{code}' не найдена. "
            f"Доступные: {', '.join(available[:10])}{'...' if len(available) > 10 else ''}"
        )
    return _char_to_id[code]


def get_currency_dynamics(
    currency_code: str,
    date_from: str,
    date_to: str,
) -> list[dict]:
    """Получить динамику курса валюты за период.

    Args:
        currency_code: Буквенный код (USD, EUR, CNY, ...)
        date_from: Начало периода YYYY-MM-DD
        date_to: Конец периода YYYY-MM-DD

    Returns:
        list: [{"date": "DD.MM.YYYY", "nominal": int, "rate": float}]

    Raises:
        ConnectionError: нет сети
        ValueError: неверный формат даты или валюта не найдена
    """
    code = currency_code.upper().strip()
    val_id = get_currency_id(code)

    cbr_from = _cbr_date(date_from)
    cbr_to = _cbr_date(date_to)

    url = (
        f"{_CBR_BASE}/XML_dynamic.asp"
        f"?date_req1={cbr_from}&date_req2={cbr_to}&VAL_NM_RQ={val_id}"
    )
    xml_str = _fetch_xml(url)
    root = ET.fromstring(xml_str)

    records: list[dict] = []
    for record in root.findall("Record"):
        date = record.get("Date", "").strip()
        nominal_str = record.findtext("Nominal", "1").strip()
        value_str = record.findtext("Value", "0").strip()
        try:
            nominal = int(nominal_str)
            rate = float(value_str.replace(",", "."))
        except ValueError:
            continue
        records.append({"date": date, "nominal": nominal, "rate": rate})

    return records
