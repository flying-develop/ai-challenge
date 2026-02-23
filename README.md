# LLM Agent

A CLI agent for chatting with the Qwen LLM API, built with `httpx` and a clean layered architecture.

## Requirements

- Python 3.11+

## Installation

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and set your values:

```bash
cp .env.example .env
```

```
QWEN_API_KEY=your-api-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-plus
QWEN_TIMEOUT_SECONDS=30
```

Only `QWEN_API_KEY` is required. The others have sensible defaults.

## Usage

**Single-shot mode:**

```bash
python -m llm_agent.interfaces.cli.main --prompt "What is the capital of France?"
```

Expected output:
```
The capital of France is Paris.
```

**Interactive mode:**

```bash
python -m llm_agent.interfaces.cli.main --interactive
```

```
Interactive mode. Type 'exit' or 'quit' to stop.

You: Hello, who are you?
Agent: I am Qwen, a large language model created by Alibaba Cloud.

You: exit
Goodbye!
```

## Running Tests

```bash
pytest
```

Expected output:
```
========================= 9 passed in 0.XXs =========================
```

## Architecture

```
llm_agent/
  config.py              # Environment config loader
  domain/
    models.py            # ChatMessage, LLMResponse dataclasses
    protocols.py         # LLMClientProtocol (structural typing)
  infrastructure/
    qwen_client.py       # httpx-based Qwen API adapter
  application/
    agent.py             # SimpleAgent orchestrator
  interfaces/
    cli/
      main.py            # argparse CLI entry point
tests/
  test_agent.py          # Unit tests for SimpleAgent
  test_qwen_client.py    # HTTP-mocked tests for QwenHttpClient
```

### Design Patterns

- **Dependency Injection**: `SimpleAgent` receives its LLM client via constructor — no hidden dependencies.
- **Adapter**: `QwenHttpClient` adapts the external Qwen HTTP API to the internal `LLMClientProtocol` interface.
- **Single Responsibility**: The CLI layer only handles I/O; it delegates all AI logic to the agent.

The Qwen endpoint URL is isolated in `config.py` (via `QWEN_BASE_URL`) and `qwen_client.py`. Any OpenAI-compatible API can be used by changing the URL without touching business logic.
