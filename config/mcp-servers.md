# MCP-серверы

Конфигурация MCP-серверов для подключения агента к внешним инструментам.
Переменные окружения подставляются в формате {VAR_NAME}.

## markdownify
- transport: stdio
- command: node
- args: {MARKDOWNIFY_MCP_PATH}/dist/index.js
- описание: Конвертация файлов и веб-контента в Markdown (PDF, DOCX, YouTube, веб-страницы)
- требует: node >= 18, pnpm
