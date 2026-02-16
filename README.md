# ai-challenge

Минимальный Python CLI для OpenAI-совместимого API (`/v1/chat/completions`).

## Настройка

1. Создай и заполни `.env`:

```bash
cp .env.example .env
```

2. Экспортируй переменные в окружение:

```bash
set -a
source .env
set +a
```

Обязательные переменные:
- `LLM_API_URL` — URL API (по умолчанию `https://api.openai.com/v1/chat/completions`)
- `LLM_MODEL` — имя модели
- `LLM_API_TOKEN` — токен доступа

## Запуск

Передай prompt аргументом:

```bash
python3 llm_cli.py "Привет! Расскажи шутку."
```

Или запусти без аргумента — prompt будет запрошен интерактивно:

```bash
python3 llm_cli.py
```
