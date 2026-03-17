#!/usr/bin/env python3
"""Демонстрация пайплайна RAG-индексации (podkop-wiki).

Скрипт показывает все 5 компонентов пайплайна в действии:
  1. DocumentLoader  — загрузка и парсинг .md файлов
  2. ChunkingStrategy — 4 стратегии нарезки (сравнение)
  3. EmbeddingProvider — LocalRandomEmbedder (детерминированный, без API)
  4. IndexStore       — SQLite хранилище
  5. IndexingPipeline — полный пайплайн + compare_strategies()

Запуск:
    python demo_rag_indexer.py

Требует:
    pip install pyyaml numpy
    git clone https://github.com/flying-develop/podkop-wiki.git podkop-wiki
    (или задать переменную DEMO_DOCS_PATH)

Переменные окружения:
    DEMO_DOCS_PATH   — путь к документам (по умолчанию ./podkop-wiki/content)
    DEMO_DB_PATH     — путь к БД        (по умолчанию ./rag_indexer/output/demo.db)
    DASHSCOPE_API_KEY — если задан, использует QwenEmbedder вместо LocalRandom
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Настройка путей
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_RAG_DIR = _SCRIPT_DIR / "rag_indexer"

if str(_RAG_DIR) not in sys.path:
    sys.path.insert(0, str(_RAG_DIR))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Конфигурация
DOCS_PATH = Path(os.environ.get("DEMO_DOCS_PATH", _SCRIPT_DIR / "podkop-wiki" / "content"))
DB_PATH = Path(os.environ.get("DEMO_DB_PATH", _RAG_DIR / "output" / "demo.db"))


# ---------------------------------------------------------------------------
# Импорт модулей пайплайна
# ---------------------------------------------------------------------------

def _check_imports() -> bool:
    """Проверить наличие зависимостей."""
    missing = []
    try:
        import yaml  # noqa: F401
    except ImportError:
        missing.append("pyyaml")
    if missing:
        print(f"❌ Отсутствуют зависимости: {', '.join(missing)}")
        print(f"   Установите: pip install {' '.join(missing)}")
        return False
    return True


# ---------------------------------------------------------------------------
# Шаг 1: DocumentLoader
# ---------------------------------------------------------------------------

def demo_loader() -> list:
    """Демонстрация DocumentLoader."""
    print("\n" + "=" * 60)
    print("  ШАГ 1: DocumentLoader")
    print("=" * 60)

    from src.loader import DocumentLoader

    if not DOCS_PATH.exists():
        print(f"⚠️  Директория не найдена: {DOCS_PATH}")
        print("    Клонируйте wiki: git clone https://github.com/flying-develop/podkop-wiki.git podkop-wiki")
        print("    Используем синтетические документы для демонстрации...")
        return _create_synthetic_docs()

    loader = DocumentLoader(DOCS_PATH)
    t0 = time.time()
    docs = loader.load()
    elapsed = time.time() - t0

    print(f"\n  Загружено : {len(docs)} документов за {elapsed:.2f}с")
    print(f"  Источник  : {DOCS_PATH}")

    if docs:
        total_chars = sum(len(d.content) for d in docs)
        print(f"  Объём     : {total_chars:,} символов ({total_chars / 1024:.1f} KB)")
        print(f"\n  Примеры документов:")
        for doc in docs[:5]:
            sections_count = len(doc.sections)
            print(f"    [{doc.weight:3}] {doc.source}")
            print(f"          title={doc.title!r}  sections={sections_count}")

    return docs


def _create_synthetic_docs() -> list:
    """Создать синтетические документы для демонстрации без wiki."""
    from src.loader import Document

    docs = []
    templates = [
        {
            "title": "Установка podkop",
            "source": "install/index.md",
            "content": """## Установка

Для установки podkop выполните следующие шаги:

1. Скачайте пакет с официального сайта
2. Загрузите на роутер через SCP или веб-интерфейс
3. Установите командой: opkg install podkop_*.ipk

## Настройка после установки

После установки необходимо настроить конфигурационный файл /etc/config/podkop.

### Основные параметры

Откройте файл конфигурации и задайте следующие параметры:
- interface — сетевой интерфейс
- proxy_type — тип прокси (sing-box, xray)
- server — адрес сервера

## Проверка работы

Для проверки запустите: /etc/init.d/podkop status
""",
            "weight": 10,
        },
        {
            "title": "Конфигурация sing-box",
            "source": "config/singbox.md",
            "content": """## Конфигурация sing-box

Podkop поддерживает sing-box в качестве основного транспортного движка.

### Структура конфига

```json
{
  "inbounds": [...],
  "outbounds": [...],
  "route": {...}
}
```

### Настройка VPN

Для настройки VPN через sing-box укажите в конфиге outbound типа vless или vmess.

## Обновление конфига

Конфиг обновляется автоматически при изменении настроек через luci.
Ручное обновление: /etc/init.d/podkop restart
""",
            "weight": 20,
        },
        {
            "title": "Решение проблем",
            "source": "troubleshoot/index.md",
            "content": """## Решение проблем

### Podkop не запускается

Проверьте логи: logread | grep podkop

Убедитесь что:
1. Пакет установлен правильно
2. Конфиг не содержит ошибок JSON
3. Сервер доступен

### Медленная скорость

Возможные причины:
- Перегружен сервер — попробуйте другой
- MTU не оптимален — установите 1400 вместо стандартного
- Выберите протокол с меньшими накладными расходами

### Потеря соединения

Включите keepalive в настройках outbound:
- interval: 30s
- idle: 120s
""",
            "weight": 30,
        },
    ]

    for t in templates:
        from src.loader import _parse_sections
        content = t["content"]
        sections = _parse_sections(content)
        docs.append(Document(
            content=content,
            source=t["source"],
            file_path=t["source"],
            title=t["title"],
            weight=t["weight"],
            frontmatter={"title": t["title"], "weight": t["weight"]},
            sections=sections,
        ))

    print(f"  Создано {len(docs)} синтетических документов")
    return docs


# ---------------------------------------------------------------------------
# Шаг 2: Chunking Strategies
# ---------------------------------------------------------------------------

def demo_chunking(docs: list) -> None:
    """Демонстрация 4 стратегий нарезки."""
    print("\n" + "=" * 60)
    print("  ШАГ 2: Chunking Strategies (сравнение)")
    print("=" * 60)

    from src.chunking.strategies import STRATEGIES, estimate_tokens

    for name, strategy in STRATEGIES.items():
        t0 = time.time()
        chunks = strategy.chunk_all(docs)
        elapsed = time.time() - t0

        if not chunks:
            print(f"\n  {name}: нет чанков")
            continue

        token_counts = [c.token_count for c in chunks]
        avg_tok = sum(token_counts) / len(token_counts)
        min_tok = min(token_counts)
        max_tok = max(token_counts)
        total_tok = sum(token_counts)

        print(f"\n  [{name}]")
        print(f"    Чанков  : {len(chunks)}  (за {elapsed:.3f}с)")
        print(f"    Токенов : avg={avg_tok:.0f}  min={min_tok}  max={max_tok}  total={total_tok}")

        # Показать первый чанк
        first = chunks[0]
        preview = first.text[:120].replace("\n", " ")
        if len(first.text) > 120:
            preview += "..."
        print(f"    Первый  : [{first.section!r}] {preview}")


# ---------------------------------------------------------------------------
# Шаг 3: EmbeddingProvider
# ---------------------------------------------------------------------------

def demo_embedding() -> "EmbeddingProvider":
    """Демонстрация EmbeddingProvider."""
    print("\n" + "=" * 60)
    print("  ШАГ 3: EmbeddingProvider")
    print("=" * 60)

    from src.embedding.provider import EmbeddingProvider

    provider = EmbeddingProvider.create("qwen")  # Graceful fallback к LocalRandom

    print(f"\n  Провайдер : {provider.__class__.__name__}")
    print(f"  Модель    : {provider.model_name}")
    print(f"  Размерность: {provider.dimension}")

    test_texts = [
        "как установить podkop на роутер",
        "настройка sing-box для обхода блокировок",
        "проблемы с подключением VPN",
    ]

    print(f"\n  Генерация эмбеддингов ({len(test_texts)} текста)...")
    t0 = time.time()
    embeddings = provider.embed_texts(test_texts)
    elapsed = time.time() - t0

    import math
    from src._math import cosine_similarities
    for text, emb in zip(test_texts, embeddings):
        norm = math.sqrt(sum(x * x for x in emb))
        print(f"    [{norm:.3f}] {text[:50]!r}  → dim={len(emb)}")
    print(f"\n  Время: {elapsed:.3f}с")

    # Косинусное сходство между первыми двумя
    sims = cosine_similarities(embeddings[0], [embeddings[1]])
    print(f"  Similarity(0,1): {sims[0]:.4f}")

    return provider


# ---------------------------------------------------------------------------
# Шаг 4 + 5: Pipeline + IndexStore
# ---------------------------------------------------------------------------

def demo_pipeline(docs: list, provider) -> None:
    """Демонстрация полного пайплайна (IndexingPipeline + IndexStore)."""
    print("\n" + "=" * 60)
    print("  ШАГ 4+5: IndexingPipeline + IndexStore")
    print("=" * 60)

    from src.pipeline import IndexingPipeline
    from src.chunking.strategies import STRATEGIES

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Удалить старую демо-БД для чистого запуска
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"\n  Старая БД удалена: {DB_PATH}")

    pipeline = IndexingPipeline(
        docs_path=DOCS_PATH if DOCS_PATH.exists() else ".",
        db_path=DB_PATH,
        embedding_provider=provider,
        strategies=list(STRATEGIES.values()),
    )
    # Перезаписываем документы (т.к. папки может не быть)
    pipeline._documents = docs

    print(f"\n  БД: {DB_PATH}")
    print()

    t0 = time.time()
    pipeline.run()
    elapsed = time.time() - t0

    print(f"\n  Полное время индексации: {elapsed:.2f}с")


# ---------------------------------------------------------------------------
# Демонстрация поиска
# ---------------------------------------------------------------------------

def demo_search(provider) -> None:
    """Демонстрация векторного поиска."""
    print("\n" + "=" * 60)
    print("  ПОИСК: Тестовые запросы")
    print("=" * 60)

    from src.storage.index_store import IndexStore

    if not DB_PATH.exists():
        print("  ❌ Индекс не найден — сначала запустите пайплайн")
        return

    queries = [
        "как установить podkop",
        "настройка прокси сервера",
        "решение проблем подключения",
    ]

    with IndexStore(DB_PATH) as store:
        for query in queries:
            print(f"\n  Запрос: «{query}»")
            query_vec = provider.embed_texts([query])[0]
            results = store.search(query_vec, top_k=2)
            for res in results:
                c = res.chunk
                print(f"    [{res.rank}] score={res.score:.4f} | {c.source} | {c.section!r}")
                preview = c.text[:100].replace("\n", " ")
                print(f"        {preview}...")


# ---------------------------------------------------------------------------
# Сравнение стратегий
# ---------------------------------------------------------------------------

def demo_compare() -> None:
    """Вывести ASCII-таблицу сравнения стратегий."""
    print("\n" + "=" * 60)
    print("  СРАВНЕНИЕ СТРАТЕГИЙ")
    print("=" * 60)

    from src.pipeline import IndexingPipeline
    from src.embedding.provider import EmbeddingProvider

    provider = EmbeddingProvider.create("local")
    pipeline = IndexingPipeline(docs_path=".", db_path=DB_PATH, embedding_provider=provider)
    pipeline.compare_strategies()


# ---------------------------------------------------------------------------
# Главная функция
# ---------------------------------------------------------------------------

def main() -> None:
    """Запустить полную демонстрацию пайплайна."""
    print("╔" + "═" * 58 + "╗")
    print("║   RAG-ИНДЕКСАЦИЯ: Демонстрация пайплайна podkop-wiki   ║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n  Docs : {DOCS_PATH}")
    print(f"  DB   : {DB_PATH}")

    # Проверка зависимостей
    if not _check_imports():
        sys.exit(1)

    # Шаг 1: Загрузка
    docs = demo_loader()

    if not docs:
        print("\n❌ Нет документов для индексации")
        sys.exit(1)

    # Шаг 2: Chunking
    demo_chunking(docs)

    # Шаг 3: Embedding
    provider = demo_embedding()

    # Шаг 4+5: Pipeline
    demo_pipeline(docs, provider)

    # Поиск
    demo_search(provider)

    # Сравнение
    demo_compare()

    print("\n" + "=" * 60)
    print("  ✓ Демонстрация завершена!")
    print(f"  Индекс сохранён: {DB_PATH}")
    print()
    print("  Дальнейшие команды:")
    print(f"    python rag_indexer/main.py compare --db {DB_PATH}")
    print(f'    python rag_indexer/main.py search --db {DB_PATH} --query "установить podkop"')
    print("=" * 60)


if __name__ == "__main__":
    main()
