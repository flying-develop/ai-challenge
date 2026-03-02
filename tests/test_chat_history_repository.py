"""Тесты для SQLiteChatHistoryRepository."""

from __future__ import annotations

import pytest

from llm_agent.domain.models import ChatMessage
from llm_agent.infrastructure.chat_history_repository import SQLiteChatHistoryRepository


@pytest.fixture
def repo(tmp_path):
    """Репозиторий с временной БД (удаляется после теста)."""
    db_file = tmp_path / "test_history.db"
    with SQLiteChatHistoryRepository(db_file, session_id="test") as r:
        yield r


# --- Базовые операции ---


def test_repo_load_empty_on_new_session(repo):
    """Новая сессия не содержит сообщений."""
    assert repo.load() == []


def test_repo_append_and_load(repo):
    """Добавленное сообщение возвращается при load()."""
    msg = ChatMessage(role="user", content="Привет")
    repo.append(msg)
    loaded = repo.load()
    assert len(loaded) == 1
    assert loaded[0].role == "user"
    assert loaded[0].content == "Привет"


def test_repo_preserves_order(repo):
    """Сообщения возвращаются в порядке добавления."""
    messages = [
        ChatMessage(role="user", content="Первый"),
        ChatMessage(role="assistant", content="Ответ 1"),
        ChatMessage(role="user", content="Второй"),
        ChatMessage(role="assistant", content="Ответ 2"),
    ]
    for m in messages:
        repo.append(m)
    loaded = repo.load()
    assert [(m.role, m.content) for m in loaded] == [
        (m.role, m.content) for m in messages
    ]


def test_repo_clear_removes_all_messages(repo):
    """clear() удаляет все сообщения текущей сессии."""
    repo.append(ChatMessage(role="user", content="Привет"))
    repo.append(ChatMessage(role="assistant", content="Ответ"))
    repo.clear()
    assert repo.load() == []


def test_repo_clear_allows_new_messages_after(repo):
    """После clear() можно добавлять новые сообщения."""
    repo.append(ChatMessage(role="user", content="До очистки"))
    repo.clear()
    repo.append(ChatMessage(role="user", content="После очистки"))
    loaded = repo.load()
    assert len(loaded) == 1
    assert loaded[0].content == "После очистки"


# --- Сессии ---


def test_repo_sessions_are_isolated(tmp_path):
    """Два репозитория с разными session_id не влияют друг на друга."""
    db_file = tmp_path / "shared.db"
    with SQLiteChatHistoryRepository(db_file, session_id="session_a") as repo_a:
        repo_a.append(ChatMessage(role="user", content="Сообщение A"))

    with SQLiteChatHistoryRepository(db_file, session_id="session_b") as repo_b:
        assert repo_b.load() == []  # session_b пуста

    with SQLiteChatHistoryRepository(db_file, session_id="session_a") as repo_a2:
        loaded = repo_a2.load()
        assert len(loaded) == 1
        assert loaded[0].content == "Сообщение A"


def test_repo_list_sessions(tmp_path):
    """list_sessions() возвращает все созданные сессии."""
    db_file = tmp_path / "sessions.db"
    for session_id in ("alpha", "beta", "gamma"):
        with SQLiteChatHistoryRepository(db_file, session_id=session_id):
            pass
    with SQLiteChatHistoryRepository(db_file, session_id="alpha") as repo:
        sessions = repo.list_sessions()
    assert set(sessions) == {"alpha", "beta", "gamma"}


def test_repo_history_persists_across_instances(tmp_path):
    """История сохраняется между созданием новых экземпляров репозитория."""
    db_file = tmp_path / "persist.db"

    # Первый экземпляр — записываем
    with SQLiteChatHistoryRepository(db_file, session_id="conv") as r1:
        r1.append(ChatMessage(role="user", content="Вопрос"))
        r1.append(ChatMessage(role="assistant", content="Ответ"))

    # Второй экземпляр (имитирует перезапуск CLI) — читаем
    with SQLiteChatHistoryRepository(db_file, session_id="conv") as r2:
        loaded = r2.load()

    assert len(loaded) == 2
    assert loaded[0].role == "user"
    assert loaded[0].content == "Вопрос"
    assert loaded[1].role == "assistant"
    assert loaded[1].content == "Ответ"


# --- message_count ---


def test_repo_message_count(repo):
    """message_count() возвращает точное количество сообщений."""
    assert repo.message_count() == 0
    repo.append(ChatMessage(role="user", content="Привет"))
    assert repo.message_count() == 1
    repo.append(ChatMessage(role="assistant", content="Ответ"))
    assert repo.message_count() == 2
    repo.clear()
    assert repo.message_count() == 0


# --- Директория создаётся автоматически ---


def test_repo_creates_parent_directories(tmp_path):
    """SQLiteChatHistoryRepository создаёт вложенные директории при необходимости."""
    nested_db = tmp_path / "a" / "b" / "c" / "history.db"
    with SQLiteChatHistoryRepository(nested_db, session_id="x") as r:
        r.append(ChatMessage(role="user", content="test"))
        loaded = r.load()
    assert len(loaded) == 1
    assert nested_db.exists()
