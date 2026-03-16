#!/usr/bin/env python3
"""CLI для пайплайна RAG-индексации документов.

Команды:
    index    — индексация документов (одна или все стратегии)
    compare  — ASCII-таблица сравнения стратегий
    search   — тестовый векторный поиск по индексу

Использование::

    # Индексация всеми стратегиями
    python main.py index --docs ./podkop-wiki/content --db ./output/index.db

    # Одна стратегия
    python main.py index --docs ./podkop-wiki/content --db ./output/index.db --strategy fixed_500

    # Сравнение (после индексации)
    python main.py compare --db ./output/index.db

    # Тестовый поиск
    python main.py search --db ./output/index.db --query "как установить podkop" --top_k 5

Зависимости:
    pyyaml>=6.0  — парсинг frontmatter
    numpy>=1.24  — векторные операции

Переменные окружения:
    DASHSCOPE_API_KEY — ключ DashScope (Qwen). Если не задан — LocalRandomEmbedder.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Убедиться, что пакет src доступен
sys.path.insert(0, str(Path(__file__).parent))

from src.chunking.strategies import STRATEGIES
from src.embedding.provider import EmbeddingProvider
from src.pipeline import IndexingPipeline
from src.storage.index_store import IndexStore


# ---------------------------------------------------------------------------
# Команды
# ---------------------------------------------------------------------------

def cmd_index(args: argparse.Namespace) -> None:
    """Запустить индексацию документов."""
    docs_path = Path(args.docs)
    db_path = Path(args.db)

    if not docs_path.exists():
        print(f"[ERROR] Директория не найдена: {docs_path}", file=sys.stderr)
        sys.exit(1)

    # Провайдер эмбеддингов (graceful fallback)
    provider = EmbeddingProvider.create("qwen", dimension=args.dimension)
    print(f"[Embedding] Провайдер: {provider.model_name} (dim={provider.dimension})")

    # Выбор стратегий
    strategies = list(STRATEGIES.values())

    pipeline = IndexingPipeline(
        docs_path=docs_path,
        db_path=db_path,
        embedding_provider=provider,
        strategies=strategies,
    )

    pipeline.run(strategy_name=args.strategy)

    # После индексации — сравнение
    if args.strategy is None:
        print("\n" + "=" * 60)
        pipeline.compare_strategies()


def cmd_compare(args: argparse.Namespace) -> None:
    """Вывести сравнение стратегий."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] БД не найдена: {db_path}", file=sys.stderr)
        print("Сначала запустите: python main.py index --docs <path> --db <path>")
        sys.exit(1)

    # Фиктивный провайдер (только для compare, не нужен)
    provider = EmbeddingProvider.create("local")
    pipeline = IndexingPipeline(
        docs_path=".",
        db_path=db_path,
        embedding_provider=provider,
    )
    pipeline.compare_strategies()


def cmd_search(args: argparse.Namespace) -> None:
    """Тестовый поиск по индексу."""
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"[ERROR] БД не найдена: {db_path}", file=sys.stderr)
        sys.exit(1)

    query = args.query
    top_k = args.top_k
    strategy = args.strategy

    print(f"\n[Search] Запрос: «{query}»")
    if strategy:
        print(f"[Search] Стратегия: {strategy}")
    print(f"[Search] Top-{top_k}")
    print()

    # Эмбеддинг запроса
    provider = EmbeddingProvider.create("qwen", dimension=args.dimension)
    print(f"[Search] Генерация эмбеддинга запроса ({provider.model_name})...", end=" ", flush=True)
    query_vectors = provider.embed_texts([query])
    query_vector = query_vectors[0]
    print("OK")

    # Поиск
    with IndexStore(db_path) as store:
        results = store.search(query_vector, strategy=strategy, top_k=top_k)

    if not results:
        print("[INFO] Результатов не найдено. Убедитесь, что индекс заполнен.")
        return

    print(f"\nНайдено {len(results)} результатов:\n")
    print("─" * 70)

    for res in results:
        c = res.chunk
        print(f"[{res.rank}] score={res.score:.4f} | {c.strategy} | {c.source}")
        print(f"    Раздел : {c.section or '(нет)'}")
        print(f"    Заголов: {c.doc_title}")
        print(f"    Токенов: {c.token_count}")
        # Превью текста
        preview = c.text[:300].replace("\n", " ")
        if len(c.text) > 300:
            preview += "..."
        print(f"    Текст  : {preview}")
        print("─" * 70)


# ---------------------------------------------------------------------------
# Парсер
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Построить argparse-парсер."""
    parser = argparse.ArgumentParser(
        prog="rag-indexer",
        description="Пайплайн индексации документов для RAG.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python main.py index --docs ./podkop-wiki/content --db ./output/index.db
  python main.py index --docs ./podkop-wiki/content --db ./output/index.db --strategy fixed_500
  python main.py compare --db ./output/index.db
  python main.py search --db ./output/index.db --query "как установить podkop" --top_k 5
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- index ---
    p_index = subparsers.add_parser("index", help="Индексировать документы")
    p_index.add_argument(
        "--docs",
        required=True,
        metavar="PATH",
        help="Путь к директории с .md файлами",
    )
    p_index.add_argument(
        "--db",
        required=True,
        metavar="PATH",
        help="Путь к SQLite-базе (создаётся автоматически)",
    )
    p_index.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default=None,
        metavar="NAME",
        help=f"Конкретная стратегия: {' | '.join(STRATEGIES)}. По умолчанию — все.",
    )
    p_index.add_argument(
        "--dimension",
        type=int,
        default=1024,
        help="Размерность эмбеддингов (по умолчанию 1024)",
    )

    # --- compare ---
    p_compare = subparsers.add_parser("compare", help="Сравнить стратегии (ASCII-таблица)")
    p_compare.add_argument("--db", required=True, metavar="PATH", help="Путь к SQLite-базе")

    # --- search ---
    p_search = subparsers.add_parser("search", help="Тестовый поиск по индексу")
    p_search.add_argument("--db", required=True, metavar="PATH", help="Путь к SQLite-базе")
    p_search.add_argument("--query", required=True, metavar="TEXT", help="Текст запроса")
    p_search.add_argument("--top_k", type=int, default=5, help="Количество результатов")
    p_search.add_argument(
        "--strategy",
        choices=list(STRATEGIES.keys()),
        default=None,
        help="Фильтр по стратегии",
    )
    p_search.add_argument("--dimension", type=int, default=1024, help="Размерность эмбеддингов")

    return parser


def main() -> None:
    """Точка входа CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "search":
        cmd_search(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
