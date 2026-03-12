# MCP-серверы

Конфигурация MCP-серверов для подключения агента к внешним инструментам.
Переменные окружения подставляются в формате {VAR_NAME}.

## markdownify
- transport: stdio
- command: node
- args: {MARKDOWNIFY_MCP_PATH}/dist/index.js
- описание: Конвертация файлов и веб-контента в Markdown (PDF, DOCX, YouTube, веб-страницы)
- требует: node >= 18, pnpm

## cbr_currencies
- transport: stdio
- command: python
- args: -m mcp_server.cbr_server
- описание: Курсы валют ЦБ РФ (собственный MCP-сервер)
- требует: доступ к интернету (cbr.ru)

## news_digest
- transport: stdio
- command: python
- args: -m mcp_server.news_server
- описание: Новостной пайплайн Lenta.ru: fetch_news → summarize_news → deliver_news → run_news_pipeline
- требует: доступ к интернету (lenta.ru), LLM-провайдер в .env, опционально TELEGRAM_BOT_TOKEN
