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

## search_server
- transport: stdio
- command: python
- args: -m mcp_server.search_server
- описание: Поиск ссылок по запросу (Yandex Cloud Search API или mock)
- требует: SEARCH_MODE=yandex_cloud требует YANDEX_CLOUD_FOLDER_ID + YANDEX_CLOUD_OAUTH_TOKEN; mock работает без ключей

## scraper_server
- transport: stdio
- command: python
- args: -m mcp_server.scraper_server
- описание: Загрузка веб-страниц и конвертация в чистый Markdown (urllib + stdlib)
- требует: доступ к интернету

## telegram_server
- transport: stdio
- command: python
- args: -m mcp_server.telegram_server
- описание: Отправка прогресса и результатов в Telegram (send_progress, send_result)
- требует: TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID; если не заданы — вывод в stdout

## journal_server
- transport: stdio
- command: python
- args: -m mcp_server.journal_server
- описание: Аудит-журнал этапов исследования в SQLite (log_stage, get_log)
- требует: ничего (SQLite из stdlib, БД создаётся автоматически)
