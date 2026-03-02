"""Тесты для слоя приложения (SimpleAgent)."""

from __future__ import annotations

import pytest

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.models import ChatMessage, LLMResponse


class MockLLMClient:
    """Минимальный mock, удовлетворяющий обновлённому LLMClientProtocol."""

    def __init__(self, text: str = "mock response") -> None:
        self._text = text
        self.last_messages: list[ChatMessage] | None = None
        self.call_count: int = 0

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        self.last_messages = messages
        self.call_count += 1
        return LLMResponse(text=self._text, model="mock-model", usage={})


# --- Базовые тесты ---

def test_agent_rejects_empty_prompt() -> None:
    agent = SimpleAgent(llm_client=MockLLMClient())
    with pytest.raises(ValueError, match="Запрос не должен быть пустым"):
        agent.ask("")


def test_agent_rejects_whitespace_prompt() -> None:
    agent = SimpleAgent(llm_client=MockLLMClient())
    with pytest.raises(ValueError, match="Запрос не должен быть пустым"):
        agent.ask("   ")


def test_agent_returns_text_from_client() -> None:
    mock_client = MockLLMClient(text="  Hello from mock!  ")
    agent = SimpleAgent(llm_client=mock_client)
    result = agent.ask("Скажи привет")
    # Проверяем, что ответ очищен от пробелов
    assert result == "Hello from mock!"


def test_agent_passes_prompt_as_user_message() -> None:
    """Агент оборачивает prompt в ChatMessage с role='user'."""
    mock_client = MockLLMClient()
    agent = SimpleAgent(llm_client=mock_client)
    agent.ask("Сколько будет 2 + 2?")
    assert mock_client.last_messages is not None
    user_messages = [m for m in mock_client.last_messages if m.role == "user"]
    assert len(user_messages) == 1
    assert user_messages[0].content == "Сколько будет 2 + 2?"


# --- Тесты накопления истории ---

def test_agent_accumulates_history_across_turns() -> None:
    """После трёх вызовов ask() третий запрос содержит все предыдущие ходы."""
    mock_client = MockLLMClient(text="ответ")
    agent = SimpleAgent(llm_client=mock_client)

    agent.ask("Первый вопрос")
    agent.ask("Второй вопрос")
    agent.ask("Третий вопрос")

    # Третий вызов должен содержать 5 сообщений: user, assistant, user, assistant, user
    assert mock_client.last_messages is not None
    assert len(mock_client.last_messages) == 5
    roles = [m.role for m in mock_client.last_messages]
    assert roles == ["user", "assistant", "user", "assistant", "user"]


def test_agent_history_contents_are_correct() -> None:
    """Сообщения в истории содержат точный текст запросов и ответов."""
    mock_client = MockLLMClient(text="ответ")
    agent = SimpleAgent(llm_client=mock_client)

    agent.ask("первый вопрос")
    agent.ask("второй вопрос")

    messages = mock_client.last_messages
    assert messages[0].content == "первый вопрос"
    assert messages[1].content == "ответ"   # ответ ассистента на первый ход
    assert messages[2].content == "второй вопрос"


# --- Тесты системного промпта ---

def test_agent_includes_system_prompt_first() -> None:
    """Системный промпт добавляется первым сообщением перед всей историей."""
    mock_client = MockLLMClient()
    agent = SimpleAgent(llm_client=mock_client, system_prompt="Ты полезный ассистент.")

    agent.ask("Привет")

    messages = mock_client.last_messages
    assert messages is not None
    assert messages[0].role == "system"
    assert messages[0].content == "Ты полезный ассистент."
    assert messages[1].role == "user"
    assert messages[1].content == "Привет"


def test_agent_system_prompt_present_in_every_turn() -> None:
    """Системный промпт присутствует на позиции 0 в каждом последующем вызове."""
    mock_client = MockLLMClient(text="ответ")
    agent = SimpleAgent(llm_client=mock_client, system_prompt="Будь кратким.")

    agent.ask("Первый ход")
    agent.ask("Второй ход")

    # Второй вызов: system + user + assistant + user = 4 сообщения
    assert mock_client.last_messages[0].role == "system"
    assert mock_client.last_messages[0].content == "Будь кратким."
    assert len(mock_client.last_messages) == 4


def test_agent_without_system_prompt_sends_no_system_message() -> None:
    """Без system_prompt список сообщений начинается с 'user'."""
    mock_client = MockLLMClient()
    agent = SimpleAgent(llm_client=mock_client)

    agent.ask("Привет")

    assert mock_client.last_messages[0].role == "user"


# --- Тесты сброса истории ---

def test_agent_clear_history_resets_conversation() -> None:
    """clear_history() очищает историю, следующий ask() начинается заново."""
    mock_client = MockLLMClient(text="ответ")
    agent = SimpleAgent(llm_client=mock_client)

    agent.ask("Первый")
    agent.ask("Второй")
    agent.clear_history()
    agent.ask("После сброса")

    # После сброса только одно новое сообщение пользователя
    assert len(mock_client.last_messages) == 1
    assert mock_client.last_messages[0].role == "user"
    assert mock_client.last_messages[0].content == "После сброса"


def test_agent_clear_history_preserves_system_prompt() -> None:
    """После clear_history() системный промпт по-прежнему добавляется."""
    mock_client = MockLLMClient(text="ответ")
    agent = SimpleAgent(llm_client=mock_client, system_prompt="Ты пират.")

    agent.ask("Первый")
    agent.clear_history()
    agent.ask("После сброса")

    # Должно быть: system + user (два сообщения, не три)
    assert len(mock_client.last_messages) == 2
    assert mock_client.last_messages[0].role == "system"
    assert mock_client.last_messages[0].content == "Ты пират."
    assert mock_client.last_messages[1].role == "user"


def test_agent_system_prompt_whitespace_is_stripped() -> None:
    """Системный промпт из одних пробелов обрабатывается как отсутствие промпта."""
    mock_client = MockLLMClient()
    agent = SimpleAgent(llm_client=mock_client, system_prompt="   ")

    agent.ask("Привет")

    # Первое сообщение должно быть от пользователя, а не системы
    assert mock_client.last_messages[0].role == "user"
