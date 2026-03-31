from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_indexer.src.chunking.strategies import STRATEGIES
from rag_indexer.src.embedding.provider import EmbeddingProvider
from rag_indexer.src.pipeline import IndexingPipeline
from rag_indexer.src.storage.index_store import IndexStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="index_project_docs",
        description="Индексировать README + docs проекта в отдельный RAG-индекс.",
    )
    default_docs_dir = os.environ.get("DOCUMAKER_DOCS_PATH", "")
    parser.add_argument(
        "--docs-dir",
        required=not bool(default_docs_dir),
        default=default_docs_dir or None,
        help="Путь к docs проекта. Можно задать через DOCUMAKER_DOCS_PATH в .env.",
    )
    parser.add_argument(
        "--readme",
        help="Путь к README.md проекта. По умолчанию: соседний с docs-dir README.md.",
    )
    parser.add_argument(
        "--index",
        default="project_docs.db",
        help="Путь к SQLite-файлу project docs индекса.",
    )
    parser.add_argument(
        "--strategy",
        default="structural",
        choices=tuple(STRATEGIES.keys()),
        help="Стратегия чанкования.",
    )
    return parser


def detect_embedder_name() -> str:
    return "ollama" if os.environ.get("LLM_MODE", "cloud").strip().lower() == "local" else "qwen"


def create_embedder():
    provider_name = detect_embedder_name()
    if provider_name == "qwen":
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
        return EmbeddingProvider.create("qwen", api_key=api_key)
    return EmbeddingProvider.create("ollama")


def resolve_readme_path(docs_dir: Path, readme_arg: str | None) -> Path:
    if readme_arg:
        return Path(readme_arg).resolve()
    return (docs_dir.resolve().parent / "README.md").resolve()


def copy_project_docs(staging_dir: Path, readme_path: Path, docs_dir: Path) -> None:
    if readme_path.exists():
        shutil.copy2(readme_path, staging_dir / "README.md")

    docs_target = staging_dir / "docs"
    docs_target.mkdir(parents=True, exist_ok=True)
    for md_file in sorted(docs_dir.rglob("*.md")):
        relative = md_file.relative_to(docs_dir)
        target = docs_target / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(md_file, target)


def read_docs_meta(docs_dir: Path) -> dict:
    meta_path = docs_dir / "_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> int:
    args = build_parser().parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    readme_path = resolve_readme_path(docs_dir, args.readme)
    index_path = Path(args.index).resolve()

    if not docs_dir.exists():
        print(f"[index_project_docs] docs dir не найден: {docs_dir}", file=sys.stderr)
        return 1

    embedder = create_embedder()
    strategy = STRATEGIES[args.strategy]

    with IndexStore(index_path) as store:
        store.clear_all()

    with tempfile.TemporaryDirectory(prefix="project-docs-") as temp_dir:
        staging_dir = Path(temp_dir)
        copy_project_docs(staging_dir, readme_path, docs_dir)

        pipeline = IndexingPipeline(
            docs_path=staging_dir,
            db_path=index_path,
            embedding_provider=embedder,
            strategies=[strategy],
        )
        pipeline.run()

    docs_meta = read_docs_meta(docs_dir)
    project_name = docs_meta.get("project_name") or docs_dir.parent.name
    with IndexStore(index_path) as store:
        store.set_meta("project_name", str(project_name))
        store.set_meta("docs_dir", str(docs_dir))
        store.set_meta("readme_path", str(readme_path))
        store.set_meta("strategy", args.strategy)
        store.set_meta("llm_mode", os.environ.get("LLM_MODE", "cloud"))
        for key, value in docs_meta.items():
            store.set_meta(f"project_meta_{key}", str(value))

    print(f"[index_project_docs] docs={docs_dir}")
    print(f"[index_project_docs] readme={readme_path}")
    print(f"[index_project_docs] index={index_path}")
    print(f"[index_project_docs] strategy={args.strategy}")
    print(f"[index_project_docs] embedder={embedder.model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# test
