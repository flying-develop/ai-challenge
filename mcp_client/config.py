"""Конфигурация MCP-серверов.

Читает config/mcp-servers.md и возвращает список MCPServerConfig.
Поддерживает подстановку переменных окружения вида {VAR_NAME}.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class MCPServerConfig:
    """Конфигурация одного MCP-сервера."""

    name: str
    transport: str          # "stdio" | "streamablehttp" | "sse"
    description: str

    # Для stdio-серверов:
    command: str | None = None
    args: list[str] | None = None
    env: dict | None = None

    # Для HTTP-серверов (заготовка на будущее):
    url: str | None = None
    headers: dict | None = None


class MCPConfigParser:
    """Парсер config/mcp-servers.md в список MCPServerConfig.

    Формат файла:
        ## <name>
        - transport: stdio
        - command: node
        - args: /path/to/server.js
        - описание: Описание сервера
    """

    _VAR_RE = re.compile(r"\{([A-Z0-9_]+)\}")

    def __init__(self, config_path: Path | str | None = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "mcp-servers.md"
        self.config_path = Path(config_path)

    def load(self) -> list[MCPServerConfig]:
        """Загрузить и распарсить конфигурацию."""
        if not self.config_path.exists():
            return []

        text = self.config_path.read_text(encoding="utf-8")
        return self._parse(text)

    def _parse(self, text: str) -> list[MCPServerConfig]:
        servers: list[MCPServerConfig] = []

        # Разбиваем по секциям ##
        sections = re.split(r"^##\s+", text, flags=re.MULTILINE)

        for section in sections[1:]:  # первый элемент — заголовок файла
            lines = section.strip().splitlines()
            if not lines:
                continue

            name = lines[0].strip()
            fields: dict[str, str] = {}

            for line in lines[1:]:
                m = re.match(r"^-\s+([^:]+):\s*(.+)$", line.strip())
                if m:
                    key = m.group(1).strip().lower()
                    value = m.group(2).strip()
                    fields[key] = value

            transport = fields.get("transport", "stdio")
            command = self._substitute_env(fields.get("command"))
            args_raw = self._substitute_env(fields.get("args"))
            # Разбиваем аргументы по пробелам, чтобы поддерживать
            # как одиночные пути ("/path/to/server.js"), так и флаги
            # с именем модуля ("-m mcp_server.cbr_server").
            args = args_raw.split() if args_raw else None
            description = fields.get("описание", fields.get("description", ""))

            servers.append(MCPServerConfig(
                name=name,
                transport=transport,
                description=description,
                command=command,
                args=args,
            ))

        return servers

    def _substitute_env(self, value: str | None) -> str | None:
        """Подставить {VAR_NAME} → значение из os.environ."""
        if value is None:
            return None

        def replace(m: re.Match) -> str:
            var_name = m.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise EnvironmentError(
                    f"Переменная {var_name} не задана\n"
                    f"Добавьте в .env: {var_name}=/path/to/..."
                )
            return env_val

        return self._VAR_RE.sub(replace, value)
