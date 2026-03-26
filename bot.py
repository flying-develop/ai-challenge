"""Support Bot — точка входа.

Запускает:
    - TelegramListener в фоновом потоке (long-polling, отвечает на вопросы)
    - AdminCLI в основном потоке (управление индексом, мониторинг)

Конфигурация через .env:
    TELEGRAM_BOT_TOKEN   — токен бота
    TELEGRAM_CHAT_ID     — дефолтный chat_id (для отправки)
    QWEN_API_KEY         — API ключ для LLM и эмбеддингов
    DASHSCOPE_API_KEY    — API ключ для эмбеддингов (Qwen)
    CHAT_HISTORY_LIMIT   — размер sliding window (default: 20)
    CHAT_INTERFACE       — cli | telegram | both (default: both)
    LLM_MODE             — local | cloud (default: cloud)
    OLLAMA_BASE_URL      — URL сервера Ollama (default: http://localhost:11434)
    OLLAMA_LLM_MODEL     — модель LLM для Ollama (default: qwen2.5:3b)
    OLLAMA_EMBED_MODEL   — модель эмбеддингов для Ollama (default: nomic-embed-text)
    OLLAMA_RERANK_MODEL  — модель реранкера для Ollama (default: qwen2.5:3b)

Запуск:
    python bot.py
    LLM_MODE=local python bot.py    # полностью локальный режим
    python bot.py --no-telegram     # только AdminCLI (без Telegram)
    python bot.py --no-cli          # только Telegram (daemon)
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Добавляем корень проекта в путь
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rag_indexer.src.embedding.provider import EmbeddingProvider
from rag_indexer.src.storage.index_store import IndexStore
from rag_indexer.src.retrieval.retriever import HybridRetriever, VectorRetriever, BM25Retriever
from rag_indexer.src.retrieval.pipeline import RAGPipeline
from rag_indexer.src.retrieval.reranker import ThresholdFilter
from llm_agent.memory.manager import MemoryManager
from src.llm_helper import make_llm_fn

from src.chat.dialog_manager import DialogManager
from src.chat.telegram_listener import TelegramListener
from src.chat.admin_cli import AdminCLI
from src.chat.session_store import SessionStore
from src.indexer import IndexManager, _create_source


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

_DB_PATH = Path("./output/index.db")
_MEMORY_DB_PATH = Path("./output/memory_{user_id}.db")
_CHAT_INTERFACE = os.environ.get("CHAT_INTERFACE", "both")
_LLM_MODE = os.environ.get("LLM_MODE", "cloud").lower()


# ---------------------------------------------------------------------------
# Инициализация компонентов
# ---------------------------------------------------------------------------

def _build_llm_fn():
    """Построить функцию LLM из доступного провайдера.

    LLM_MODE=local  → Ollama (qwen2.5:3b, без API-ключей)
    LLM_MODE=cloud  → облачный провайдер (Qwen/OpenAI)
    """
    if _LLM_MODE == "local":
        from src.providers.stack_config import create_providers
        providers = create_providers("local")
        return providers["llm_fn"]
    return make_llm_fn(timeout=60.0)


def _build_rag_pipeline(llm_fn) -> RAGPipeline:
    """Построить RAG-пайплайн со структурированными ответами.

    LLM_MODE=local  → OllamaEmbedder + OllamaReranker (без API-ключей)
    LLM_MODE=cloud  → QwenEmbedder + ThresholdFilter

    Returns:
        RAGPipeline с Hybrid-поиском и structured responses.
    """
    if _LLM_MODE == "local":
        from src.providers.stack_config import create_providers
        providers = create_providers("local")
        embedding_provider = providers["embedder"]
        reranker = providers["reranker"]
        print(
            f"[Bot] LOCAL mode: embedder={providers['embed_model']}, "
            f"reranker={providers['rerank_model']}"
        )
    else:
        embedding_provider = EmbeddingProvider.create("qwen")
        reranker = ThresholdFilter(threshold=0.0)

    store = IndexStore(_DB_PATH)

    vector_retriever = VectorRetriever(store=store, embedder=embedding_provider)
    bm25_retriever = BM25Retriever(store=store)
    retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        llm_fn=llm_fn,
        reranker=reranker,
        use_structured=True,
    )
    return pipeline


def _build_dialog_factory(rag_pipeline, llm_fn):
    """Построить фабрику DialogManager (per-user).

    Каждый вызов create(user_id=...) создаёт новый DialogManager
    с изолированной памятью.

    Args:
        rag_pipeline: Общий RAG пайплайн (shared).
        llm_fn:       Общий LLM клиент (shared).

    Returns:
        Объект с методом create(user_id) → DialogManager.
    """
    class Factory:
        def create(self, user_id: str) -> DialogManager:
            memory_path = str(_MEMORY_DB_PATH).format(user_id=user_id)
            memory = MemoryManager(memory_path)
            return DialogManager(
                user_id=user_id,
                memory_manager=memory,
                rag_pipeline=rag_pipeline,
                llm_fn=llm_fn,
            )

    return Factory()


def _build_index_manager() -> IndexManager:
    """Создать IndexManager для AdminCLI.

    Использует тот же провайдер эмбеддингов, что и RAG-пайплайн.
    """
    if _LLM_MODE == "local":
        from src.providers.stack_config import create_providers
        providers = create_providers("local")
        embedding_provider = providers["embedder"]
    else:
        embedding_provider = EmbeddingProvider.create("qwen")
    return IndexManager(
        db_path=_DB_PATH,
        embedding_provider=embedding_provider,
    )


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить support-бот."""
    print("=" * 60)
    print("  Support Bot — День 27")
    print(f"  Режим: {'LOCAL (Ollama)' if _LLM_MODE == 'local' else 'CLOUD'}")
    print("=" * 60)

    # Проверяем индекс
    if not _DB_PATH.exists():
        print(f"\n[WARN] Индекс не найден: {_DB_PATH}")
        print("  Для индексации используйте AdminCLI:")
        print("  /index path ./podkop-wiki/content")
        print("  /index url https://laravel.com/docs/13.x")
        print("  /index github https://github.com/user/repo")
        print()

    # Инициализация компонентов
    print("[Bot] Инициализация LLM...")
    llm_fn = _build_llm_fn()

    print("[Bot] Инициализация RAG pipeline...")
    rag_pipeline = _build_rag_pipeline(llm_fn)

    print("[Bot] Инициализация сессий...")
    session_store = SessionStore()
    factory = _build_dialog_factory(rag_pipeline, llm_fn)

    # Лог-колбэк: пишем в AdminCLI
    admin_cli_ref = [None]  # mutable container

    def log_callback(entry: dict) -> None:
        if admin_cli_ref[0]:
            admin_cli_ref[0].add_log(entry)

    # IndexManager для AdminCLI
    index_manager = _build_index_manager()

    def index_fn(source_type: str, source_arg: str) -> None:
        if source_type == "clear":
            from rag_indexer.src.storage.index_store import IndexStore
            with IndexStore(_DB_PATH) as store:
                deleted = store.clear_all()
            print(f"[IndexManager] Удалено {deleted} чанков")
        else:
            index_manager.reindex(source_type, source_arg)

    def status_fn() -> str:
        return index_manager.status()

    # TelegramListener
    telegram_listener = TelegramListener(
        dialog_manager_factory=factory,
        session_store=session_store,
        log_callback=log_callback,
    )

    # AdminCLI
    admin_cli = AdminCLI(
        session_store=session_store,
        index_fn=index_fn,
        status_fn=status_fn,
        stop_fn=telegram_listener.stop,
    )
    admin_cli_ref[0] = admin_cli

    interface = _CHAT_INTERFACE.lower()

    if interface in ("telegram", "both"):
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        if not token:
            print("[Bot] TELEGRAM_BOT_TOKEN не задан — Telegram listener отключён")
        else:
            # Запускаем Telegram в фоновом потоке
            tg_thread = threading.Thread(
                target=telegram_listener.run,
                daemon=True,
                name="TelegramListener",
            )
            tg_thread.start()
            print(f"[Bot] TelegramListener запущен в потоке {tg_thread.name}")

    if interface in ("cli", "both"):
        print("[Bot] Запуск AdminCLI...")
        admin_cli.run()
    else:
        # Только Telegram — ждём завершения
        print("[Bot] Режим: только Telegram. Ctrl+C для остановки.")
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            telegram_listener.stop()

    print("[Bot] Завершение работы.")


if __name__ == "__main__":
    main()
