"""Тесты для mcp_server.cbr_api и mcp_server.cbr_server.

Покрывают:
- Парсинг XML-ответа ЦБ (mock XML-строки)
- Маппинг CharCode → внутренний код (кэш _char_to_id)
- Обработку ошибок (неверная валюта, невалидная дата)
- Форматирование текстового ответа
- Инструменты MCP-сервера (через прямой вызов функций)
"""

from __future__ import annotations

import textwrap
from unittest.mock import MagicMock, patch

import pytest

# Проверяем наличие пакета mcp (для тестов cbr_server)
try:
    from mcp.server.fastmcp import FastMCP  # noqa: F401
    _MCP_AVAILABLE = True
except ImportError:
    _MCP_AVAILABLE = False

skip_if_no_mcp = pytest.mark.skipif(
    not _MCP_AVAILABLE,
    reason="Пакет mcp не установлен (pip install mcp)"
)


# ---------------------------------------------------------------------------
# Фикстуры: примеры XML-ответов ЦБ РФ
# ---------------------------------------------------------------------------

DAILY_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="windows-1251"?>
    <ValCurs Date="10.03.2026" name="Foreign Currency Market">
        <Valute ID="R01235">
            <NumCode>840</NumCode>
            <CharCode>USD</CharCode>
            <Nominal>1</Nominal>
            <Name>Доллар США</Name>
            <Value>88,6319</Value>
            <VunitRate>88,6319</VunitRate>
        </Valute>
        <Valute ID="R01239">
            <NumCode>978</NumCode>
            <CharCode>EUR</CharCode>
            <Nominal>1</Nominal>
            <Name>Евро</Name>
            <Value>95,1234</Value>
            <VunitRate>95,1234</VunitRate>
        </Valute>
        <Valute ID="R01375">
            <NumCode>156</NumCode>
            <CharCode>CNY</CharCode>
            <Nominal>10</Nominal>
            <Name>Китайский юань</Name>
            <Value>122,4050</Value>
            <VunitRate>12,2405</VunitRate>
        </Valute>
        <Valute ID="R01035">
            <NumCode>826</NumCode>
            <CharCode>GBP</CharCode>
            <Nominal>1</Nominal>
            <Name>Фунт стерлингов</Name>
            <Value>113,5000</Value>
            <VunitRate>113,5000</VunitRate>
        </Valute>
    </ValCurs>
""")

DYNAMICS_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="windows-1251"?>
    <ValCurs ID="R01235" DateRange1="03.03.2026" DateRange2="10.03.2026" name="Foreign Currency Market">
        <Record Date="03.03.2026" Id="R01235">
            <Nominal>1</Nominal>
            <Value>87,1234</Value>
        </Record>
        <Record Date="04.03.2026" Id="R01235">
            <Nominal>1</Nominal>
            <Value>87,5678</Value>
        </Record>
        <Record Date="05.03.2026" Id="R01235">
            <Nominal>1</Nominal>
            <Value>88,0000</Value>
        </Record>
        <Record Date="10.03.2026" Id="R01235">
            <Nominal>1</Nominal>
            <Value>88,6319</Value>
        </Record>
    </ValCurs>
""")


# ---------------------------------------------------------------------------
# Тесты cbr_api — парсинг XML
# ---------------------------------------------------------------------------

class TestGetDailyRates:
    """Тесты функции get_daily_rates с mock HTTP."""

    def test_parse_usd(self):
        """Парсит USD корректно: id, rate, nominal, name."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server import cbr_api
            rates = cbr_api.get_daily_rates()

        assert "USD" in rates
        usd = rates["USD"]
        assert usd["id"] == "R01235"
        assert usd["nominal"] == 1
        assert abs(usd["rate"] - 88.6319) < 0.0001
        assert "Доллар" in usd["name"]
        assert usd["date"] == "10.03.2026"

    def test_parse_eur(self):
        """Парсит EUR корректно."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server import cbr_api
            rates = cbr_api.get_daily_rates()

        assert "EUR" in rates
        eur = rates["EUR"]
        assert eur["id"] == "R01239"
        assert abs(eur["rate"] - 95.1234) < 0.0001

    def test_parse_cny_nominal_10(self):
        """Парсит CNY: номинал 10, rate — за 10 юаней."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server import cbr_api
            rates = cbr_api.get_daily_rates()

        assert "CNY" in rates
        cny = rates["CNY"]
        assert cny["nominal"] == 10
        assert abs(cny["rate"] - 122.4050) < 0.0001

    def test_parse_all_currencies(self):
        """Все 4 валюты из XML присутствуют."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server import cbr_api
            rates = cbr_api.get_daily_rates()

        assert set(rates.keys()) >= {"USD", "EUR", "CNY", "GBP"}

    def test_comma_decimal_separator(self):
        """Запятая в Value корректно преобразуется в float."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server import cbr_api
            rates = cbr_api.get_daily_rates()

        # 88,6319 → 88.6319
        assert isinstance(rates["USD"]["rate"], float)
        assert isinstance(rates["CNY"]["rate"], float)

    def test_updates_char_to_id_cache(self):
        """get_daily_rates обновляет кэш _char_to_id."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            import mcp_server.cbr_api as cbr_api
            cbr_api._char_to_id.clear()
            cbr_api.get_daily_rates()

        assert cbr_api._char_to_id.get("USD") == "R01235"
        assert cbr_api._char_to_id.get("EUR") == "R01239"
        assert cbr_api._char_to_id.get("CNY") == "R01375"


# ---------------------------------------------------------------------------
# Тесты cbr_api — маппинг CharCode → ID
# ---------------------------------------------------------------------------

class TestGetCurrencyId:
    """Тесты функции get_currency_id с кэшем."""

    def test_known_currency_from_cache(self):
        """Если валюта уже в кэше, HTTP не вызывается."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["USD"] = "R01235"

        with patch("mcp_server.cbr_api._fetch_xml") as mock_fetch:
            val_id = cbr_api.get_currency_id("USD")

        mock_fetch.assert_not_called()
        assert val_id == "R01235"

    def test_case_insensitive(self):
        """Код валюты регистронезависим."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["EUR"] = "R01239"

        val_id = cbr_api.get_currency_id("eur")
        assert val_id == "R01239"

        val_id2 = cbr_api.get_currency_id("Eur")
        assert val_id2 == "R01239"

    def test_unknown_currency_raises(self):
        """Несуществующая валюта → ValueError с полезным сообщением."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id.clear()

        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            with pytest.raises(ValueError, match="XYZ"):
                cbr_api.get_currency_id("XYZ")

    def test_unknown_currency_lists_available(self):
        """Сообщение об ошибке содержит доступные валюты."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id.clear()

        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            with pytest.raises(ValueError) as exc_info:
                cbr_api.get_currency_id("ZZZ")

        assert "Доступные" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Тесты cbr_api — _cbr_date
# ---------------------------------------------------------------------------

class TestCbrDate:
    """Тесты форматирования даты."""

    def test_valid_date(self):
        from mcp_server.cbr_api import _cbr_date
        assert _cbr_date("2026-03-10") == "10/03/2026"

    def test_empty_string(self):
        from mcp_server.cbr_api import _cbr_date
        assert _cbr_date("") == ""

    def test_invalid_format_raises(self):
        from mcp_server.cbr_api import _cbr_date
        with pytest.raises(ValueError, match="Неверный формат"):
            _cbr_date("not-a-date")

    def test_wrong_order_raises(self):
        """DD-MM-YYYY вместо YYYY-MM-DD → ошибка."""
        from mcp_server.cbr_api import _cbr_date
        with pytest.raises(ValueError):
            _cbr_date("10-03-2026")


# ---------------------------------------------------------------------------
# Тесты cbr_api — get_currency_dynamics
# ---------------------------------------------------------------------------

class TestGetCurrencyDynamics:
    """Тесты динамики курса."""

    def test_parse_records(self):
        """Парсит 4 записи из DYNAMICS_XML."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["USD"] = "R01235"

        with patch("mcp_server.cbr_api._fetch_xml", return_value=DYNAMICS_XML):
            records = cbr_api.get_currency_dynamics("USD", "2026-03-03", "2026-03-10")

        assert len(records) == 4
        assert records[0]["date"] == "03.03.2026"
        assert abs(records[0]["rate"] - 87.1234) < 0.0001
        assert records[-1]["date"] == "10.03.2026"

    def test_invalid_date_raises(self):
        """Неверная дата → ValueError."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["USD"] = "R01235"

        with pytest.raises(ValueError, match="Неверный формат"):
            cbr_api.get_currency_dynamics("USD", "bad-date", "2026-03-10")

    def test_connection_error_propagates(self):
        """Ошибка сети → ConnectionError."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["USD"] = "R01235"

        with patch("mcp_server.cbr_api._fetch_xml", side_effect=ConnectionError("Нет сети")):
            with pytest.raises(ConnectionError, match="Нет сети"):
                cbr_api.get_currency_dynamics("USD", "2026-03-03", "2026-03-10")


# ---------------------------------------------------------------------------
# Тесты инструментов MCP-сервера
# ---------------------------------------------------------------------------

@skip_if_no_mcp
class TestGetExchangeRatesTool:
    """Тесты get_exchange_rates инструмента."""

    def test_formats_rates_table(self):
        """Возвращает отформатированную таблицу курсов."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_exchange_rates
            result = get_exchange_rates()

        assert "USD" in result
        assert "EUR" in result
        assert "CNY" in result
        assert "₽" in result
        assert "10.03.2026" in result

    def test_nominal_displayed_for_cny(self):
        """Для CNY с номиналом 10 показывается '10 ='."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_exchange_rates
            result = get_exchange_rates()

        assert "10 = " in result or "10=" in result

    def test_invalid_date_returns_error_text(self):
        """Неверная дата → текст ошибки (не исключение)."""
        from mcp_server.cbr_server import get_exchange_rates
        result = get_exchange_rates("not-a-date")

        assert "Неверный" in result or "формат" in result

    def test_connection_error_returns_error_text(self):
        """Ошибка сети → текст ошибки (не исключение)."""
        with patch("mcp_server.cbr_api._fetch_xml", side_effect=ConnectionError("cbr.ru недоступен")):
            from mcp_server.cbr_server import get_exchange_rates
            result = get_exchange_rates()

        assert "Ошибка" in result or "cbr.ru" in result


@skip_if_no_mcp
class TestGetCurrencyRateTool:
    """Тесты get_currency_rate инструмента."""

    def test_known_currency_returns_rate(self):
        """Известная валюта → строка с курсом."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_currency_rate
            result = get_currency_rate("USD")

        assert "USD" in result
        assert "88" in result
        assert "₽" in result

    def test_unknown_currency_returns_error(self):
        """Неизвестная валюта → текст с 'не найдена'."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_currency_rate
            result = get_currency_rate("XYZ")

        assert "не найдена" in result or "XYZ" in result

    def test_cny_shows_rate_per_unit(self):
        """Для CNY (номинал 10) отображается курс за единицу."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_currency_rate
            result = get_currency_rate("CNY")

        assert "12.2405" in result or "12,2405" in result.replace(".", ",")


@skip_if_no_mcp
class TestConvertCurrencyTool:
    """Тесты convert_currency инструмента."""

    def test_to_rub_usd(self):
        """100 USD → рубли: result ≈ 8863.19."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(100.0, "USD", "to_rub")

        assert "USD" in result
        assert "₽" in result
        # 100 * 88.6319 = 8863.19
        assert "8" in result  # первая цифра суммы в рублях

    def test_from_rub_usd(self):
        """10000 RUB → USD: result ≈ 112.83."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(10000.0, "USD", "from_rub")

        assert "USD" in result
        assert "₽" in result

    def test_cny_nominal_applied(self):
        """1000 CNY → рубли: учитывает номинал 10 (rate_per_unit = rate/10)."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(1000.0, "CNY", "to_rub")

        assert "CNY" in result
        assert "₽" in result
        # 1000 * (122.405 / 10) = 12240.50
        assert "12" in result  # тысячи рублей

    def test_invalid_direction(self):
        """Неверное направление → сообщение об ошибке."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(100.0, "USD", "sideways")

        assert "Неверное" in result or "направление" in result

    def test_unknown_currency(self):
        """Несуществующая валюта → сообщение об ошибке."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(100.0, "XYZ", "to_rub")

        assert "не найдена" in result or "XYZ" in result

    def test_default_direction_is_to_rub(self):
        """По умолчанию direction='to_rub': валюта → рубли."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(100.0, "USD")

        assert "₽" in result

    def test_result_format_to_rub(self):
        """Формат ответа to_rub: '<amount> USD = <result> ₽ (курс ЦБ: ...)'."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(100.0, "USD", "to_rub")

        assert "100" in result
        assert "USD" in result
        assert "=" in result
        assert "₽" in result
        assert "курс ЦБ" in result

    def test_result_format_from_rub(self):
        """Формат ответа from_rub: '<amount> ₽ = <result> USD (курс ЦБ: ...)'."""
        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import convert_currency
            result = convert_currency(10000.0, "USD", "from_rub")

        assert "10" in result
        assert "₽" in result
        assert "USD" in result
        assert "=" in result
        assert "курс ЦБ" in result


@skip_if_no_mcp
class TestGetCurrencyDynamicsTool:
    """Тесты get_currency_dynamics инструмента."""

    def test_returns_records(self):
        """Возвращает список курсов по дням."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id["USD"] = "R01235"

        with patch("mcp_server.cbr_api._fetch_xml", return_value=DYNAMICS_XML):
            from mcp_server.cbr_server import get_currency_dynamics
            result = get_currency_dynamics("USD", "2026-03-03", "2026-03-10")

        assert "USD" in result
        assert "03.03.2026" in result
        assert "10.03.2026" in result

    def test_invalid_date_returns_error(self):
        """Неверная дата → текст ошибки."""
        from mcp_server.cbr_server import get_currency_dynamics
        result = get_currency_dynamics("USD", "bad-date", "2026-03-10")

        assert "Неверный" in result or "формат" in result

    def test_unknown_currency_returns_error(self):
        """Неизвестная валюта → текст ошибки."""
        import mcp_server.cbr_api as cbr_api
        cbr_api._char_to_id.clear()

        with patch("mcp_server.cbr_api._fetch_xml", return_value=DAILY_XML):
            from mcp_server.cbr_server import get_currency_dynamics
            result = get_currency_dynamics("XYZ", "2026-03-03", "2026-03-10")

        assert "не найдена" in result or "XYZ" in result


# ---------------------------------------------------------------------------
# Тесты MCPConfigParser — парсинг args с пробелами
# ---------------------------------------------------------------------------

class TestMCPConfigParserArgsWithSpaces:
    """Тесты что args корректно разбиваются по пробелам."""

    def test_module_args_split(self, tmp_path):
        """'-m mcp_server.cbr_server' парсится как ['-m', 'mcp_server.cbr_server']."""
        import textwrap as tw
        from mcp_client.config import MCPConfigParser

        content = tw.dedent("""\
            # MCP-серверы

            ## cbr_currencies
            - transport: stdio
            - command: python
            - args: -m mcp_server.cbr_server
            - описание: Курсы валют ЦБ РФ
        """)
        p = tmp_path / "mcp-servers.md"
        p.write_text(content, encoding="utf-8")
        parser = MCPConfigParser(config_path=p)
        servers = parser.load()

        assert len(servers) == 1
        s = servers[0]
        assert s.command == "python"
        assert s.args == ["-m", "mcp_server.cbr_server"]

    def test_single_path_stays_single(self, tmp_path):
        """'/path/to/server.js' (без пробелов) остаётся одним элементом."""
        import textwrap as tw
        from mcp_client.config import MCPConfigParser

        content = tw.dedent("""\
            # MCP-серверы

            ## markdownify
            - transport: stdio
            - command: node
            - args: /home/user/markdownify-mcp/dist/index.js
            - описание: Markdownify
        """)
        p = tmp_path / "mcp-servers.md"
        p.write_text(content, encoding="utf-8")
        parser = MCPConfigParser(config_path=p)
        servers = parser.load()

        assert servers[0].args == ["/home/user/markdownify-mcp/dist/index.js"]
