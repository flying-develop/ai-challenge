"""Тесты для стратегий управления контекстом."""

from __future__ import annotations

import pytest

from llm_agent.application.context_strategies import (
    BranchingStrategy,
    ContextStrategyProtocol,
    FactsStore,
    SlidingWindowStrategy,
    StickyFactsStrategy,
)
from llm_agent.domain.models import ChatMessage, LLMResponse


# ---------------------------------------------------------------------------
# Mock LLM клиент
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Mock LLM клиент для тестов."""

    def __init__(self, text: str = "mock response") -> None:
        self._text = text
        self.call_count: int = 0
        self.last_messages: list[ChatMessage] | None = None

    def generate(self, messages: list[ChatMessage]) -> LLMResponse:
        self.call_count += 1
        self.last_messages = messages
        return LLMResponse(text=self._text, model="mock", usage={})


# ===========================================================================
# Тесты SlidingWindowStrategy
# ===========================================================================

class TestSlidingWindowStrategy:

    def test_rejects_small_window(self) -> None:
        with pytest.raises(ValueError):
            SlidingWindowStrategy(window_size=1)

    def test_name_includes_window_size(self) -> None:
        s = SlidingWindowStrategy(window_size=5)
        assert "5" in s.name

    def test_add_message(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        s.add_message(ChatMessage(role="user", content="hello"))
        assert len(s.messages) == 1

    def test_trim_to_window_size(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        for i in range(10):
            s.add_message(ChatMessage(role="user", content=f"msg-{i}"))
        assert len(s.messages) == 4
        # Должны остаться последние 4 сообщения
        assert s.messages[0].content == "msg-6"
        assert s.messages[-1].content == "msg-9"

    def test_build_messages_with_system_prompt(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        s.add_message(ChatMessage(role="user", content="hi"))
        msgs = s.build_messages(system_prompt="Be helpful.")
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be helpful."
        assert msgs[1].role == "user"
        assert msgs[1].content == "hi"

    def test_build_messages_without_system_prompt(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        s.add_message(ChatMessage(role="user", content="hi"))
        msgs = s.build_messages()
        assert len(msgs) == 1
        assert msgs[0].role == "user"

    def test_reset(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        s.add_message(ChatMessage(role="user", content="hi"))
        s.reset()
        assert len(s.messages) == 0

    def test_get_stats(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        for i in range(6):
            s.add_message(ChatMessage(role="user", content=f"msg-{i}"))
        stats = s.get_stats()
        assert stats["window_size"] == 4
        assert stats["current_messages"] == 4
        assert stats["total_added"] == 6
        assert stats["total_dropped"] == 2

    def test_implements_protocol(self) -> None:
        s = SlidingWindowStrategy(window_size=4)
        assert isinstance(s, ContextStrategyProtocol)


# ===========================================================================
# Тесты FactsStore
# ===========================================================================

class TestFactsStore:

    def test_empty(self) -> None:
        fs = FactsStore()
        assert fs.to_text() == "(пусто)"
        assert fs.facts == {}

    def test_parse_from_text(self) -> None:
        fs = FactsStore()
        fs.parse_from_text(
            "ЦЕЛЬ: Создать трекер привычек\n"
            "БЮДЖЕТ: 2 млн рублей\n"
            "ТЕХНОЛОГИИ: Flutter, Firebase"
        )
        assert len(fs.facts) == 3
        assert fs.facts["ЦЕЛЬ"] == "Создать трекер привычек"
        assert fs.facts["БЮДЖЕТ"] == "2 млн рублей"
        assert fs.facts["ТЕХНОЛОГИИ"] == "Flutter, Firebase"

    def test_parse_ignores_empty_lines(self) -> None:
        fs = FactsStore()
        fs.parse_from_text("\n\nЦЕЛЬ: test\n\n")
        assert len(fs.facts) == 1

    def test_parse_ignores_lines_without_colon(self) -> None:
        fs = FactsStore()
        fs.parse_from_text("no colon here\nЦЕЛЬ: test")
        assert len(fs.facts) == 1

    def test_to_text_roundtrip(self) -> None:
        fs = FactsStore()
        fs.facts = {"ЦЕЛЬ": "test", "БЮДЖЕТ": "100"}
        text = fs.to_text()
        assert "ЦЕЛЬ: test" in text
        assert "БЮДЖЕТ: 100" in text


# ===========================================================================
# Тесты StickyFactsStrategy
# ===========================================================================

class TestStickyFactsStrategy:

    def test_rejects_small_window(self) -> None:
        with pytest.raises(ValueError):
            StickyFactsStrategy(window_size=1)

    def test_name(self) -> None:
        s = StickyFactsStrategy(window_size=6)
        assert "6" in s.name
        assert "Facts" in s.name

    def test_add_message_and_trim(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        for i in range(8):
            s.add_message(ChatMessage(role="user", content=f"msg-{i}"))
        assert len(s.messages) == 4
        assert s.messages[0].content == "msg-4"

    def test_build_messages_includes_facts(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        s._facts.facts = {"ЦЕЛЬ": "test app"}
        s.add_message(ChatMessage(role="user", content="hi"))

        msgs = s.build_messages(system_prompt="Be helpful.")
        assert msgs[0].role == "system"
        assert msgs[0].content == "Be helpful."
        assert msgs[1].role == "system"
        assert "ЦЕЛЬ: test app" in msgs[1].content
        assert msgs[2].role == "user"

    def test_build_messages_no_facts_block_when_empty(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        s.add_message(ChatMessage(role="user", content="hi"))
        msgs = s.build_messages(system_prompt="Be helpful.")
        # Только system prompt + user message
        assert len(msgs) == 2

    def test_on_response_calls_llm(self) -> None:
        mock = MockLLMClient(text="ЦЕЛЬ: Тест\nТЕМА: Привычки")
        s = StickyFactsStrategy(window_size=4, llm_client=mock)
        user_msg = ChatMessage(role="user", content="Создаём трекер")
        assistant_msg = ChatMessage(role="assistant", content="Хорошо!")
        s.on_response(user_msg, assistant_msg)
        assert mock.call_count == 1
        assert "ЦЕЛЬ" in s.facts
        assert s.facts["ЦЕЛЬ"] == "Тест"

    def test_on_response_without_llm_does_nothing(self) -> None:
        s = StickyFactsStrategy(window_size=4, llm_client=None)
        s.on_response(
            ChatMessage(role="user", content="x"),
            ChatMessage(role="assistant", content="y"),
        )
        assert len(s.facts) == 0

    def test_reset(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        s._facts.facts = {"ЦЕЛЬ": "x"}
        s.add_message(ChatMessage(role="user", content="hi"))
        s.reset()
        assert len(s.messages) == 0
        assert len(s.facts) == 0

    def test_get_stats(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        s._facts.facts = {"A": "1", "B": "2"}
        stats = s.get_stats()
        assert stats["facts_count"] == 2
        assert stats["window_size"] == 4

    def test_implements_protocol(self) -> None:
        s = StickyFactsStrategy(window_size=4)
        assert isinstance(s, ContextStrategyProtocol)


# ===========================================================================
# Тесты BranchingStrategy
# ===========================================================================

class TestBranchingStrategy:

    def test_name_shows_branch(self) -> None:
        s = BranchingStrategy()
        assert "main" in s.name

    def test_add_message_to_main(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="hi"))
        assert len(s.messages) == 1
        assert s.current_branch_id == "main"

    def test_build_messages(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="hi"))
        msgs = s.build_messages(system_prompt="Be helpful.")
        assert msgs[0].role == "system"
        assert msgs[1].role == "user"

    def test_save_checkpoint(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="msg1"))
        s.add_message(ChatMessage(role="assistant", content="reply1"))
        cp = s.save_checkpoint("cp1", description="After first exchange")
        assert cp.checkpoint_id == "cp1"
        assert len(cp.messages) == 2
        assert cp.description == "After first exchange"

    def test_save_checkpoint_duplicate_raises(self) -> None:
        s = BranchingStrategy()
        s.save_checkpoint("cp1")
        with pytest.raises(ValueError, match="уже существует"):
            s.save_checkpoint("cp1")

    def test_create_branch(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="msg1"))
        s.save_checkpoint("cp1")
        branch = s.create_branch("b1", "cp1", description="Branch 1")
        assert branch.branch_id == "b1"
        assert branch.checkpoint_id == "cp1"

    def test_create_branch_nonexistent_checkpoint_raises(self) -> None:
        s = BranchingStrategy()
        with pytest.raises(ValueError, match="не найден"):
            s.create_branch("b1", "nonexistent")

    def test_create_branch_duplicate_raises(self) -> None:
        s = BranchingStrategy()
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        with pytest.raises(ValueError, match="уже существует"):
            s.create_branch("b1", "cp1")

    def test_create_branch_main_raises(self) -> None:
        s = BranchingStrategy()
        s.save_checkpoint("cp1")
        with pytest.raises(ValueError, match="уже существует"):
            s.create_branch("main", "cp1")

    def test_switch_branch(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="main-msg"))
        s.add_message(ChatMessage(role="assistant", content="main-reply"))
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")
        assert s.current_branch_id == "b1"
        # Сообщения = checkpoint + ветка (пока пустая)
        assert len(s.messages) == 2

    def test_add_message_to_branch(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="main-msg"))
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")
        s.add_message(ChatMessage(role="user", content="branch-msg"))
        assert len(s.messages) == 2  # cp(1) + branch(1)
        assert s.messages[-1].content == "branch-msg"

    def test_switch_back_to_main(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="main-msg"))
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        s.switch_branch("b1")
        s.add_message(ChatMessage(role="user", content="branch-msg"))
        s.switch_branch("main")
        assert s.current_branch_id == "main"
        assert len(s.messages) == 1  # только main-msg

    def test_switch_nonexistent_raises(self) -> None:
        s = BranchingStrategy()
        with pytest.raises(ValueError, match="не найдена"):
            s.switch_branch("nonexistent")

    def test_independent_branches(self) -> None:
        """Две ветки от одного checkpoint-а не влияют друг на друга."""
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="base"))
        s.save_checkpoint("cp1")

        s.create_branch("b1", "cp1")
        s.create_branch("b2", "cp1")

        # Добавляем в ветку b1
        s.switch_branch("b1")
        s.add_message(ChatMessage(role="user", content="b1-msg1"))
        s.add_message(ChatMessage(role="user", content="b1-msg2"))
        assert len(s.messages) == 3  # cp(1) + 2

        # Переключаемся на b2 — в ней ничего нового
        s.switch_branch("b2")
        assert len(s.messages) == 1  # только cp(1)

        # Добавляем в b2
        s.add_message(ChatMessage(role="user", content="b2-msg1"))
        assert len(s.messages) == 2  # cp(1) + 1

        # Проверяем b1 не изменилась
        s.switch_branch("b1")
        assert len(s.messages) == 3

    def test_branches_list(self) -> None:
        s = BranchingStrategy()
        assert s.branches == ["main"]
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        s.create_branch("b2", "cp1")
        assert "main" in s.branches
        assert "b1" in s.branches
        assert "b2" in s.branches

    def test_get_branch_info(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="msg"))
        info = s.get_branch_info("main")
        assert info["branch_id"] == "main"
        assert info["messages_count"] == 1

    def test_get_branch_info_nonexistent_raises(self) -> None:
        s = BranchingStrategy()
        with pytest.raises(ValueError):
            s.get_branch_info("nonexistent")

    def test_reset(self) -> None:
        s = BranchingStrategy()
        s.add_message(ChatMessage(role="user", content="msg"))
        s.save_checkpoint("cp1")
        s.create_branch("b1", "cp1")
        s.reset()
        assert len(s.messages) == 0
        assert s.checkpoints == []
        assert s.branches == ["main"]
        assert s.current_branch_id == "main"

    def test_implements_protocol(self) -> None:
        s = BranchingStrategy()
        assert isinstance(s, ContextStrategyProtocol)


# ===========================================================================
# Тесты StrategyAgent
# ===========================================================================

class TestStrategyAgent:

    def test_ask_returns_response(self) -> None:
        from llm_agent.application.strategy_agent import StrategyAgent

        mock = MockLLMClient(text="Hello!")
        strategy = SlidingWindowStrategy(window_size=10)
        agent = StrategyAgent(llm_client=mock, strategy=strategy)
        result = agent.ask("Hi")
        assert result == "Hello!"

    def test_ask_adds_messages_to_strategy(self) -> None:
        from llm_agent.application.strategy_agent import StrategyAgent

        mock = MockLLMClient(text="reply")
        strategy = SlidingWindowStrategy(window_size=10)
        agent = StrategyAgent(llm_client=mock, strategy=strategy)
        agent.ask("Hi")
        # user + assistant = 2 messages
        assert len(strategy.messages) == 2

    def test_switch_strategy(self) -> None:
        from llm_agent.application.strategy_agent import StrategyAgent

        mock = MockLLMClient(text="reply")
        s1 = SlidingWindowStrategy(window_size=10)
        s2 = BranchingStrategy()
        agent = StrategyAgent(llm_client=mock, strategy=s1)
        agent.ask("Hi")
        change = agent.switch_strategy(s2)
        assert "Branching" in change
        assert agent.strategy is s2

    def test_rejects_empty_prompt(self) -> None:
        from llm_agent.application.strategy_agent import StrategyAgent

        mock = MockLLMClient()
        agent = StrategyAgent(
            llm_client=mock,
            strategy=SlidingWindowStrategy(window_size=10),
        )
        with pytest.raises(ValueError):
            agent.ask("")

    def test_on_response_called(self) -> None:
        """Проверяем, что on_response вызывается для StickyFactsStrategy."""
        from llm_agent.application.strategy_agent import StrategyAgent

        facts_llm = MockLLMClient(text="ЦЕЛЬ: тест")
        main_llm = MockLLMClient(text="Ок!")
        strategy = StickyFactsStrategy(window_size=10, llm_client=facts_llm)
        agent = StrategyAgent(llm_client=main_llm, strategy=strategy)
        agent.ask("Создаём тест")
        # facts_llm должен быть вызван для обновления фактов
        assert facts_llm.call_count == 1
        assert "ЦЕЛЬ" in strategy.facts

    def test_clear_history(self) -> None:
        from llm_agent.application.strategy_agent import StrategyAgent

        mock = MockLLMClient(text="reply")
        strategy = SlidingWindowStrategy(window_size=10)
        agent = StrategyAgent(llm_client=mock, strategy=strategy)
        agent.ask("Hi")
        agent.clear_history()
        assert len(strategy.messages) == 0
        assert agent.turn == 0
