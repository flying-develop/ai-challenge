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
