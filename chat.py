#!/usr/bin/env python3
"""Точка входа: LLM-агент с переключением стратегий и провайдеров.

Использование:
    python chat.py                              # авто-провайдер, стратегия 1
    python chat.py --provider qwen             # Qwen (Alibaba)
    python chat.py --provider openai           # OpenAI GPT
    python chat.py --provider claude           # Claude (Anthropic)
    python chat.py --provider openai --model gpt-4o
    python chat.py --list-providers            # показать доступные провайдеры

Команды внутри чата:
    /provider <name>       переключить провайдер (qwen | openai | claude)
    /model <name>          сменить модель (история сохраняется)
    /providers             список доступных провайдеров
    /switch 1|2|3          переключить стратегию контекста
    /facts                 факты (стратегия 2)
    /checkpoint, /branch   управление ветками (стратегия 3)
    /stats                 статистика токенов
    /convert 100 USD       конвертация валют (ЦБ РФ)
    /news                  пайплайн новостей Lenta.ru (RSS→LLM→Telegram)
    /news status           статус последнего запуска
    /news history [N]      последние N суммаризаций из БД
    /news fetch            только получить новости (шаг 1)
    /news summarize        получить + суммаризировать (шаги 1-2)
    /research "<запрос>"   двухпроходное исследование (8 этапов, 4 MCP-сервера)
    /research status       текущий этап и прогресс
    /research log          журнал последнего исследования
    /research last         финальный ответ последнего исследования
    /help                  полная справка

Конфигурация (.env):
    QWEN_API_KEY=...
    OPENAI_API_KEY=sk-...
    ANTHROPIC_API_KEY=sk-ant-...
    LLM_PROVIDER=qwen         # провайдер по умолчанию
"""

from llm_agent.interfaces.cli.interactive_strategies import main

if __name__ == "__main__":
    main()
