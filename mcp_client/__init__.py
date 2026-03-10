"""MCP-клиент для подключения к сторонним MCP-серверам.

Использование:
    from mcp_client.config import MCPConfigParser, MCPServerConfig
    from mcp_client.client import MCPClient

    parser = MCPConfigParser()
    servers = parser.load()
    config = servers[0]

    client = MCPClient(config)
    tools = client.connect_and_list_tools()
    print(client.get_tools_summary())
"""

from mcp_client.config import MCPConfigParser, MCPServerConfig
from mcp_client.client import MCPClient

__all__ = ["MCPClient", "MCPConfigParser", "MCPServerConfig"]
