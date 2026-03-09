"""MCP-клиент для подключения к сторонним MCP-серверам через stdio.

Образовательная концепция: MCP (Model Context Protocol) позволяет агенту
динамически обнаруживать внешние инструменты. Сервер — это отдельный процесс
(сторонний проект), который клиент запускает и спрашивает: "какие tools у тебя есть?"
Это называется tool discovery.

Архитектурное решение: asyncio используется ТОЛЬКО внутри MCPClient,
потому что официальный Python SDK `mcp` не имеет синхронного API.
За пределами этого класса проект остаётся полностью синхронным.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    _MCP_SDK_AVAILABLE = True
except ImportError:
    _MCP_SDK_AVAILABLE = False
    ClientSession = None  # type: ignore[assignment,misc]
    StdioServerParameters = None  # type: ignore[assignment,misc]
    stdio_client = None  # type: ignore[assignment]

from mcp_client.config import MCPServerConfig


# Таймаут инициализации MCP-сессии (секунды)
_INIT_TIMEOUT = 10


class MCPClient:
    """
    MCP-клиент для подключения к сторонним MCP-серверам.

    Образовательная концепция: MCP позволяет агенту динамически
    обнаруживать внешние инструменты. Сервер — это отдельный процесс
    (сторонний проект), который клиент запускает и спрашивает:
    "какие tools у тебя есть?" Это называется tool discovery.

    Подход "connect-and-query": каждый вызов connect_and_list_tools()
    запускает сервер, получает данные, останавливает. Просто и предсказуемо.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._tools: list[dict] = []

    def connect_and_list_tools(self) -> list[dict]:
        """
        Запускает MCP-сервер, получает список tools, останавливает.

        Возвращает список dict: {name, description, inputSchema}.
        Бросает RuntimeError с диагностическим сообщением при ошибках.
        """
        if not _MCP_SDK_AVAILABLE:
            raise RuntimeError(
                "❌ MCP: пакет mcp не установлен.\n"
                "   Установите: pip install mcp"
            )

        async def _query() -> list[dict]:
            if self.config.transport == "stdio":
                return await self._query_stdio()
            raise ValueError(
                f"Транспорт '{self.config.transport}' пока не реализован. "
                f"Сейчас поддерживается: stdio"
            )

        self._tools = asyncio.run(_query())
        return self._tools

    async def _query_stdio(self) -> list[dict]:
        """Выполнить запрос через stdio-транспорт."""
        self._validate_stdio_config()

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args or [],
            env=self.config.env,
        )

        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    try:
                        await asyncio.wait_for(session.initialize(), timeout=_INIT_TIMEOUT)
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"❌ MCP: Таймаут инициализации \"{self.config.name}\" ({_INIT_TIMEOUT}с)"
                        )
                    result = await session.list_tools()
                    return self._parse_tools(result)
        except RuntimeError:
            raise
        except FileNotFoundError as exc:
            path = str(exc).split("'")[1] if "'" in str(exc) else str(exc)
            if self.config.command and not shutil.which(self.config.command):
                raise RuntimeError(
                    f"❌ MCP: Команда \"{self.config.command}\" не найдена\n"
                    f"Установите Node.js: https://nodejs.org/"
                ) from exc
            raise RuntimeError(
                f"❌ MCP: Файл не найден: {path}\n"
                f"Установите markdownify-mcp:\n"
                f"  git clone https://github.com/zcaceres/markdownify-mcp.git\n"
                f"  cd markdownify-mcp && pnpm install && pnpm run build"
            ) from exc
        except Exception as exc:
            msg = str(exc)
            if "returncode" in msg or "Process" in msg:
                raise RuntimeError(
                    f"❌ MCP: Сервер \"{self.config.name}\" завершился с ошибкой\n"
                    f"Попробуйте запустить вручную: "
                    f"{self.config.command} {' '.join(self.config.args or [])}"
                ) from exc
            raise RuntimeError(f"❌ MCP: Ошибка подключения к \"{self.config.name}\": {exc}") from exc

    def _validate_stdio_config(self) -> None:
        """Проверить наличие команды и файла сервера."""
        if not self.config.command:
            raise RuntimeError(f"❌ MCP: Не задана команда для сервера \"{self.config.name}\"")

        # Проверяем, что команда найдена в PATH
        if not shutil.which(self.config.command):
            raise RuntimeError(
                f"❌ MCP: Команда \"{self.config.command}\" не найдена\n"
                f"Установите Node.js: https://nodejs.org/"
            )

        # Проверяем существование файла сервера (первый аргумент, если это путь)
        args = self.config.args or []
        if args and not args[0].startswith("-"):
            server_path = Path(args[0])
            if not server_path.exists():
                raise RuntimeError(
                    f"❌ MCP: Файл не найден: {server_path}\n"
                    f"Установите markdownify-mcp:\n"
                    f"  git clone https://github.com/zcaceres/markdownify-mcp.git\n"
                    f"  cd markdownify-mcp && pnpm install && pnpm run build"
                )

    def _parse_tools(self, result) -> list[dict]:
        """Преобразовать ответ MCP SDK в список словарей."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema if isinstance(t.inputSchema, dict) else (
                    t.inputSchema.model_dump() if hasattr(t.inputSchema, "model_dump") else {}
                ),
            }
            for t in result.tools
        ]

    def get_tools_summary(self) -> str:
        """Форматирует список инструментов для CLI."""
        if not self._tools:
            return "Инструменты не загружены. Вызовите connect_and_list_tools()."

        lines = [
            f"📋 Инструменты MCP-сервера \"{self.config.name}\" "
            f"({len(self._tools)} шт.):\n"
        ]
        for i, tool in enumerate(self._tools, 1):
            lines.append(f"  {i}. {tool['name']}")
            if tool.get("description"):
                lines.append(f"     {tool['description']}")
            schema = tool.get("inputSchema", {})
            props = schema.get("properties", {})
            required = schema.get("required", [])
            for pname, pinfo in props.items():
                req = "required" if pname in required else "optional"
                ptype = pinfo.get("type", "any")
                desc = pinfo.get("description", "")
                line = f"     - {pname}: {ptype} ({req})"
                if desc:
                    line += f" — {desc}"
                lines.append(line)
            lines.append("")
        return "\n".join(lines)

    @property
    def tools(self) -> list[dict]:
        """Кэшированный список инструментов (из последнего вызова connect_and_list_tools)."""
        return self._tools
