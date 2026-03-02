"""Тесты подсчёта токенов: TiktokenCounter, TokenUsage, ContextLimitError, SimpleAgent."""

from __future__ import annotations

import pytest

from llm_agent.application.agent import SimpleAgent
from llm_agent.domain.models import (
    ChatMessage,
    ContextLimitError,
    LLMResponse,
    TokenUsage,
)
from llm_agent.infrastructure.token_counter import TiktokenCounter


# ---------------------------------------------------------------------------
# Вспомогательные заглушки
# ---------------------------------------------------------------------------

class MockLLMClient:
    def __init__(self, text: str = "mock response") -> None:
        self._text = text
        self.last_messages: list[ChatMessage] | None = None

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        self.last_messages = messages
        return LLMResponse(text=self._text, model="mock", usage={})


class MockLLMClientWithUsage:
    """Возвращает usage из API (как реальный OpenAI/Qwen)."""

    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int) -> None:
        self._text = text
        self._usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        return LLMResponse(text=self._text, model="mock", usage=self._usage)


# ---------------------------------------------------------------------------
# Тесты TiktokenCounter
# ---------------------------------------------------------------------------

class TestTiktokenCounter:
    def setup_method(self) -> None:
        self.counter = TiktokenCounter(model="gpt-3.5-turbo")

    def test_count_tokens_simple(self) -> None:
        # "Hello" — один токен у tiktoken, 1–2 у аппроксиматора; главное > 0
        assert self.counter.count_tokens("Hello") >= 1

    def test_count_tokens_empty(self) -> None:
        assert self.counter.count_tokens("") == 0

    def test_count_tokens_increases_with_length(self) -> None:
        short = self.counter.count_tokens("Hi")
        long = self.counter.count_tokens("Hi " * 50)
        assert long > short

    def test_count_messages_tokens_minimum(self) -> None:
        # Пустой список: только 2 токена прайминга
        assert self.counter.count_messages_tokens([]) == 2

    def test_count_messages_tokens_single(self) -> None:
        msgs = [ChatMessage(role="user", content="Hello")]
        tokens = self.counter.count_messages_tokens(msgs)
        # 2 (прайминг) + 4 (overhead) + count("user") + count("Hello")
        expected = 2 + 4 + self.counter.count_tokens("user") + self.counter.count_tokens("Hello")
        assert tokens == expected

    def test_count_messages_tokens_grows_with_history(self) -> None:
        one_msg = [ChatMessage(role="user", content="Привет")]
        two_msgs = [
            ChatMessage(role="user", content="Привет"),
            ChatMessage(role="assistant", content="Здравствуйте!"),
        ]
        assert self.counter.count_messages_tokens(two_msgs) > self.counter.count_messages_tokens(one_msg)

    def test_unknown_model_falls_back_gracefully(self) -> None:
        counter = TiktokenCounter(model="unknown-model-xyz")
        # Не должно выбрасывать исключение
        assert counter.count_tokens("test") > 0


# ---------------------------------------------------------------------------
# Тесты TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_context_usage_percent_no_limit(self) -> None:
        usage = TokenUsage(history_tokens=500, context_limit=0)
        assert usage.context_usage_percent == 0.0

    def test_context_usage_percent_half(self) -> None:
        usage = TokenUsage(history_tokens=2048, context_limit=4096)
        assert usage.context_usage_percent == pytest.approx(50.0)

    def test_is_near_limit_false(self) -> None:
        usage = TokenUsage(history_tokens=100, context_limit=4096)
        assert not usage.is_near_limit

    def test_is_near_limit_true(self) -> None:
        usage = TokenUsage(history_tokens=3500, context_limit=4096)
        assert usage.is_near_limit

    def test_would_exceed_limit_false(self) -> None:
        usage = TokenUsage(history_tokens=100, context_limit=4096)
        assert not usage.would_exceed_limit

    def test_would_exceed_limit_true(self) -> None:
        usage = TokenUsage(history_tokens=5000, context_limit=4096)
        assert usage.would_exceed_limit

    def test_no_limit_never_exceeds(self) -> None:
        usage = TokenUsage(history_tokens=999999, context_limit=0)
        assert not usage.would_exceed_limit
        assert not usage.is_near_limit


# ---------------------------------------------------------------------------
# Тесты ContextLimitError
# ---------------------------------------------------------------------------

class TestContextLimitError:
    def test_attributes(self) -> None:
        err = ContextLimitError(tokens=500, limit=300)
        assert err.tokens == 500
        assert err.limit == 300

    def test_message_contains_info(self) -> None:
        err = ContextLimitError(tokens=500, limit=300)
        msg = str(err)
        assert "500" in msg
        assert "300" in msg

    def test_is_exception(self) -> None:
        with pytest.raises(ContextLimitError):
            raise ContextLimitError(tokens=100, limit=50)


# ---------------------------------------------------------------------------
# Тесты SimpleAgent с token_counter
# ---------------------------------------------------------------------------

class TestAgentTokenCounting:
    def test_last_token_usage_is_none_without_counter(self) -> None:
        agent = SimpleAgent(llm_client=MockLLMClient())
        agent.ask("Привет")
        assert agent.last_token_usage is None

    def test_last_token_usage_populated_with_counter(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(llm_client=MockLLMClient(), token_counter=counter)
        agent.ask("Привет")
        usage = agent.last_token_usage
        assert usage is not None
        assert usage.request_tokens > 0
        assert usage.history_tokens > 0
        assert usage.response_tokens > 0
        assert usage.total_tokens == usage.history_tokens + usage.response_tokens

    def test_history_tokens_grow_with_turns(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(llm_client=MockLLMClient(text="ответ"), token_counter=counter)

        agent.ask("Первый вопрос")
        tokens_turn1 = agent.last_token_usage.history_tokens

        agent.ask("Второй вопрос")
        tokens_turn2 = agent.last_token_usage.history_tokens

        assert tokens_turn2 > tokens_turn1

    def test_context_limit_stored_in_usage(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(
            llm_client=MockLLMClient(),
            token_counter=counter,
            context_limit=4096,
        )
        agent.ask("Тест")
        assert agent.last_token_usage.context_limit == 4096

    def test_raises_context_limit_error(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(
            llm_client=MockLLMClient(text="ответ"),
            token_counter=counter,
            context_limit=20,   # очень маленький лимит
            auto_truncate=False,
        )
        with pytest.raises(ContextLimitError) as exc_info:
            agent.ask("Это довольно длинный запрос, который превысит маленький лимит токенов")
        assert exc_info.value.limit == 20

    def test_auto_truncate_does_not_raise(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(
            llm_client=MockLLMClient(text="ответ"),
            token_counter=counter,
            context_limit=80,
            auto_truncate=True,
        )
        # Накапливаем историю
        for i in range(5):
            agent.ask(f"Вопрос номер {i}")
        # Должен работать без ошибок
        result = agent.ask("Финальный вопрос")
        assert result == "ответ"

    def test_auto_truncate_stays_within_limit(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(
            llm_client=MockLLMClient(text="ok"),
            token_counter=counter,
            context_limit=80,
            auto_truncate=True,
        )
        for i in range(8):
            agent.ask(f"Сообщение {i}")
        usage = agent.last_token_usage
        assert usage is not None
        assert usage.history_tokens <= 80

    def test_api_usage_overrides_local_count(self) -> None:
        """Если API вернул usage — он имеет приоритет над локальным счётчиком."""
        counter = TiktokenCounter()
        client = MockLLMClientWithUsage(
            text="ответ", prompt_tokens=999, completion_tokens=111
        )
        agent = SimpleAgent(llm_client=client, token_counter=counter)
        agent.ask("Тест")
        usage = agent.last_token_usage
        assert usage.history_tokens == 999
        assert usage.response_tokens == 111
        assert usage.total_tokens == 999 + 111

    def test_clear_history_resets_usage(self) -> None:
        counter = TiktokenCounter()
        agent = SimpleAgent(llm_client=MockLLMClient(), token_counter=counter)
        agent.ask("Привет")
        assert agent.last_token_usage is not None
        agent.clear_history()
        assert agent.last_token_usage is None


# ---------------------------------------------------------------------------
# Интеграционный тест: три сценария из демо
# ---------------------------------------------------------------------------

class TestThreeScenarios:
    """Проверяем логику трёх сценариев без реального LLM."""

    def _make_agent(self, limit: int, auto_truncate: bool) -> SimpleAgent:
        counter = TiktokenCounter()
        return SimpleAgent(
            llm_client=MockLLMClient(text="Подробный ответ на ваш вопрос."),
            token_counter=counter,
            context_limit=limit,
            auto_truncate=auto_truncate,
        )

    def test_short_dialog_low_usage(self) -> None:
        agent = self._make_agent(limit=4096, auto_truncate=False)
        for q in ["Привет!", "2 + 2?", "Пока!"]:
            agent.ask(q)
        assert agent.last_token_usage.context_usage_percent < 5

    def test_long_dialog_usage_grows(self) -> None:
        agent = self._make_agent(limit=4096, auto_truncate=False)
        usages = []
        for i in range(7):
            agent.ask(f"Вопрос {i}: расскажи подробно о теме {i}")
            usages.append(agent.last_token_usage.history_tokens)
        # Каждый следующий ход должен быть больше предыдущего
        assert all(usages[i] < usages[i + 1] for i in range(len(usages) - 1))

    def test_overflow_raises_error(self) -> None:
        agent = self._make_agent(limit=50, auto_truncate=False)
        with pytest.raises(ContextLimitError):
            for _ in range(10):
                agent.ask("Длинный вопрос, который заполнит контекст")

    def test_overflow_auto_truncate_survives(self) -> None:
        agent = self._make_agent(limit=100, auto_truncate=True)
        for i in range(15):
            agent.ask(f"Вопрос {i}")
        assert agent.last_token_usage.history_tokens <= 100
