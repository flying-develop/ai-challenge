"""Тесты для mcp_client.

Покрывают:
- Парсинг config/mcp-servers.md
- Валидацию MCPServerConfig
- Форматирование get_tools_summary (mock)
- Обработку ошибок (mock, без реального subprocess)
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_client.config import MCPConfigParser, MCPServerConfig
from mcp_client.client import MCPClient, _MCP_SDK_AVAILABLE

# Маркер: пропустить тест, если пакет mcp не установлен
skip_if_no_mcp = pytest.mark.skipif(
    not _MCP_SDK_AVAILABLE,
    reason="Пакет mcp не установлен (pip install mcp)"
)


# ---------------------------------------------------------------------------
# Тесты парсера конфигурации
# ---------------------------------------------------------------------------

class TestMCPConfigParser:
    """Тесты парсинга mcp-servers.md."""

    def _make_parser(self, content: str, tmp_path: Path) -> MCPConfigParser:
        p = tmp_path / "mcp-servers.md"
        p.write_text(content, encoding="utf-8")
        return MCPConfigParser(config_path=p)

    def test_parse_single_server(self, tmp_path):
        content = textwrap.dedent("""\
            # MCP-серверы

            ## markdownify
            - transport: stdio
            - command: node
            - args: /home/user/markdownify-mcp/dist/index.js
            - описание: Конвертация файлов в Markdown
        """)
        parser = self._make_parser(content, tmp_path)
        servers = parser.load()

        assert len(servers) == 1
        s = servers[0]
        assert s.name == "markdownify"
        assert s.transport == "stdio"
        assert s.command == "node"
        assert s.args == ["/home/user/markdownify-mcp/dist/index.js"]
        assert "Markdown" in s.description

    def test_parse_multiple_servers(self, tmp_path):
        content = textwrap.dedent("""\
            # MCP-серверы

            ## server-one
            - transport: stdio
            - command: node
            - args: /path/one.js
            - описание: Первый сервер

            ## server-two
            - transport: sse
            - url: http://localhost:8080
            - описание: Второй сервер
        """)
        parser = self._make_parser(content, tmp_path)
        servers = parser.load()

        assert len(servers) == 2
        assert servers[0].name == "server-one"
        assert servers[1].name == "server-two"
        assert servers[1].transport == "sse"

    def test_env_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MARKDOWNIFY_MCP_PATH", "/home/user/markdownify-mcp")

        content = textwrap.dedent("""\
            # MCP-серверы

            ## markdownify
            - transport: stdio
            - command: node
            - args: {MARKDOWNIFY_MCP_PATH}/dist/index.js
            - описание: Тест
        """)
        parser = self._make_parser(content, tmp_path)
        servers = parser.load()

        assert servers[0].args == ["/home/user/markdownify-mcp/dist/index.js"]

    def test_env_substitution_missing_var(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MARKDOWNIFY_MCP_PATH", raising=False)

        content = textwrap.dedent("""\
            # MCP-серверы

            ## markdownify
            - transport: stdio
            - command: node
            - args: {MARKDOWNIFY_MCP_PATH}/dist/index.js
            - описание: Тест
        """)
        parser = self._make_parser(content, tmp_path)

        with pytest.raises(EnvironmentError, match="MARKDOWNIFY_MCP_PATH"):
            parser.load()

    def test_empty_file(self, tmp_path):
        content = "# MCP-серверы\n\n(нет серверов)\n"
        parser = self._make_parser(content, tmp_path)
        servers = parser.load()
        assert servers == []

    def test_missing_file(self, tmp_path):
        parser = MCPConfigParser(config_path=tmp_path / "nonexistent.md")
        servers = parser.load()
        assert servers == []


# ---------------------------------------------------------------------------
# Тесты MCPServerConfig
# ---------------------------------------------------------------------------

class TestMCPServerConfig:
    """Тесты датакласса MCPServerConfig."""

    def test_defaults(self):
        cfg = MCPServerConfig(name="test", transport="stdio", description="desc")
        assert cfg.command is None
        assert cfg.args is None
        assert cfg.env is None
        assert cfg.url is None
        assert cfg.headers is None

    def test_stdio_config(self):
        cfg = MCPServerConfig(
            name="markdownify",
            transport="stdio",
            description="desc",
            command="node",
            args=["/path/to/server.js"],
        )
        assert cfg.command == "node"
        assert cfg.args == ["/path/to/server.js"]

    def test_http_config(self):
        cfg = MCPServerConfig(
            name="remote",
            transport="streamablehttp",
            description="desc",
            url="http://localhost:8080",
            headers={"Authorization": "Bearer token"},
        )
        assert cfg.url == "http://localhost:8080"
        assert cfg.headers == {"Authorization": "Bearer token"}


# ---------------------------------------------------------------------------
# Тесты MCPClient.get_tools_summary
# ---------------------------------------------------------------------------

class TestMCPClientSummary:
    """Тесты форматирования списка инструментов."""

    def _make_client_with_tools(self, tools: list[dict]) -> MCPClient:
        cfg = MCPServerConfig(
            name="test-server",
            transport="stdio",
            description="Тест",
            command="node",
            args=["/tmp/server.js"],
        )
        client = MCPClient(cfg)
        client._tools = tools
        return client

    def test_empty_tools(self):
        client = self._make_client_with_tools([])
        summary = client.get_tools_summary()
        assert "не загружены" in summary

    def test_single_tool_no_schema(self):
        tools = [{"name": "my-tool", "description": "Описание", "inputSchema": {}}]
        client = self._make_client_with_tools(tools)
        summary = client.get_tools_summary()
        assert "my-tool" in summary
        assert "Описание" in summary
        assert "test-server" in summary

    def test_tool_with_required_param(self):
        tools = [{
            "name": "pdf-to-markdown",
            "description": "Convert PDF",
            "inputSchema": {
                "properties": {
                    "source": {"type": "string", "description": "URL or path"},
                },
                "required": ["source"],
            },
        }]
        client = self._make_client_with_tools(tools)
        summary = client.get_tools_summary()
        assert "source" in summary
        assert "required" in summary
        assert "string" in summary

    def test_tool_with_optional_param(self):
        tools = [{
            "name": "webpage-to-markdown",
            "description": "Convert webpage",
            "inputSchema": {
                "properties": {
                    "url": {"type": "string", "description": "URL"},
                    "timeout": {"type": "number", "description": "Timeout"},
                },
                "required": ["url"],
            },
        }]
        client = self._make_client_with_tools(tools)
        summary = client.get_tools_summary()
        assert "url" in summary
        assert "required" in summary
        assert "timeout" in summary
        assert "optional" in summary

    def test_multiple_tools_count(self):
        tools = [
            {"name": f"tool-{i}", "description": f"Tool {i}", "inputSchema": {}}
            for i in range(5)
        ]
        client = self._make_client_with_tools(tools)
        summary = client.get_tools_summary()
        assert "5 шт." in summary

    def test_numbering(self):
        tools = [
            {"name": "alpha", "description": "", "inputSchema": {}},
            {"name": "beta", "description": "", "inputSchema": {}},
        ]
        client = self._make_client_with_tools(tools)
        summary = client.get_tools_summary()
        assert "1." in summary
        assert "2." in summary


# ---------------------------------------------------------------------------
# Тесты обработки ошибок (без реального subprocess)
# ---------------------------------------------------------------------------

class TestMCPClientErrors:
    """Тесты обработки ошибок с mock."""

    def _make_client(self, command: str = "node", args: list | None = None) -> MCPClient:
        cfg = MCPServerConfig(
            name="test",
            transport="stdio",
            description="desc",
            command=command,
            args=args or ["/tmp/fake.js"],
        )
        return MCPClient(cfg)

    @skip_if_no_mcp
    def test_unknown_transport_raises(self):
        cfg = MCPServerConfig(
            name="test",
            transport="unknown-transport",
            description="desc",
        )
        client = MCPClient(cfg)
        with pytest.raises(ValueError, match="пока не реализован"):
            client.connect_and_list_tools()

    @skip_if_no_mcp
    def test_missing_command_raises(self, tmp_path):
        """Несуществующая команда → RuntimeError с диагностикой."""
        cfg = MCPServerConfig(
            name="test",
            transport="stdio",
            description="desc",
            command="fake-node-that-does-not-exist-xyz",
            args=[str(tmp_path / "server.js")],
        )
        # Создаём файл, чтобы только команда была проблемой
        (tmp_path / "server.js").write_text("// fake")
        client = MCPClient(cfg)

        with pytest.raises(RuntimeError, match="не найдена"):
            client.connect_and_list_tools()

    @skip_if_no_mcp
    def test_missing_server_file_raises(self, tmp_path):
        """Несуществующий файл сервера → RuntimeError с диагностикой."""
        import shutil
        if not shutil.which("node"):
            pytest.skip("node не установлен")

        cfg = MCPServerConfig(
            name="test",
            transport="stdio",
            description="desc",
            command="node",
            args=[str(tmp_path / "nonexistent" / "server.js")],
        )
        client = MCPClient(cfg)

        with pytest.raises(RuntimeError, match="Файл не найден"):
            client.connect_and_list_tools()

    @skip_if_no_mcp
    @patch("mcp_client.client.stdio_client")
    def test_timeout_raises(self, mock_stdio, tmp_path):
        """Таймаут initialize → RuntimeError с диагностикой."""
        import asyncio

        # Создаём фиктивный файл чтобы пройти _validate_stdio_config
        fake_js = tmp_path / "server.js"
        fake_js.write_text("// fake")

        # Мокаем stdio_client как async context manager
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session.initialize = AsyncMock(side_effect=asyncio.TimeoutError())

        mock_streams = MagicMock()
        mock_streams.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        mock_streams.__aexit__ = AsyncMock(return_value=None)
        mock_stdio.return_value = mock_streams

        import shutil
        if not shutil.which("node"):
            pytest.skip("node не установлен")

        cfg = MCPServerConfig(
            name="markdownify",
            transport="stdio",
            description="desc",
            command="node",
            args=[str(fake_js)],
        )
        client = MCPClient(cfg)

        with patch("mcp_client.client.ClientSession") as mock_cs:
            mock_cs.return_value = mock_session
            with pytest.raises(RuntimeError, match="Таймаут"):
                client.connect_and_list_tools()

    def test_parse_tools_from_result(self):
        """_parse_tools корректно извлекает поля из MCP-ответа."""
        cfg = MCPServerConfig(
            name="test", transport="stdio", description="desc"
        )
        client = MCPClient(cfg)

        tool_mock = MagicMock()
        tool_mock.name = "youtube-to-markdown"
        tool_mock.description = "Convert YouTube"
        tool_mock.inputSchema = {"type": "object", "properties": {}}

        result_mock = MagicMock()
        result_mock.tools = [tool_mock]

        parsed = client._parse_tools(result_mock)

        assert len(parsed) == 1
        assert parsed[0]["name"] == "youtube-to-markdown"
        assert parsed[0]["description"] == "Convert YouTube"
        assert parsed[0]["inputSchema"] == {"type": "object", "properties": {}}
