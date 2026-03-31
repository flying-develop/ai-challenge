#!/usr/bin/env python3
"""Day 32 — AI Code Review Pipeline.

Анализирует diff PR через RAG (project_docs.db) + Qwen API и возвращает
структурированное ревью: потенциальные баги, архитектурные проблемы, рекомендации.

Использование::

    # Из stdin
    git diff HEAD~1 | python day32_code_review.py

    # Из файла
    python day32_code_review.py --diff-file changes.diff

    # С кастомным RAG-индексом
    python day32_code_review.py --rag-db ./project_docs.db --diff-file pr.diff

    # JSON-вывод (для GitHub Action)
    python day32_code_review.py --format json --diff-file pr.diff

Переменные окружения:
    QWEN_API_KEY       — ключ DashScope / Qwen API (или DASHSCOPE_API_KEY)
    QWEN_MODEL         — модель (по умолчанию qwen-plus)
    QWEN_BASE_URL      — base URL OpenAI-compatible API
    RAG_DB_PATH        — путь к SQLite-индексу (по умолчанию project_docs.db)
    RAG_TOP_K          — количество RAG-результатов (по умолчанию 5)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen-plus"
DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_RAG_DB = "project_docs.db"
DEFAULT_TOP_K = 5
MAX_DIFF_CHARS = 12_000      # обрезаем diff если слишком большой
MAX_RAG_CHARS = 4_000        # контекст из RAG

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ReviewResult:
    bugs: list[str] = field(default_factory=list)
    architecture: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    summary: str = ""
    rag_sources: list[str] = field(default_factory=list)


@dataclass
class RagChunk:
    text: str
    source: str
    section: str
    doc_title: str
    score: float = 0.0


# ---------------------------------------------------------------------------
# RAG: BM25-like поиск по SQLite без эмбеддингов
# ---------------------------------------------------------------------------

def extract_keywords(diff: str, changed_files: list[str]) -> list[str]:
    """Извлечь ключевые слова из diff и изменённых файлов."""
    keywords: set[str] = set()

    # Имена файлов без расширения и пути
    for f in changed_files:
        stem = Path(f).stem
        if len(stem) > 3:
            keywords.add(stem.lower())

    # Добавленные строки: импорты, функции, классы
    added_lines = [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    text = "\n".join(added_lines[:200])  # первые 200 добавленных строк

    # Имена функций / классов / импортов
    patterns = [
        r"\bclass\s+(\w+)",
        r"\bdef\s+(\w+)",
        r"\bimport\s+(\w+)",
        r"\bfrom\s+(\w+)",
        r"\basync\s+def\s+(\w+)",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            word = m.group(1)
            if len(word) > 3 and not word.startswith("_"):
                keywords.add(word.lower())

    return list(keywords)[:20]


def rag_search(db_path: str, keywords: list[str], top_k: int = 5) -> list[RagChunk]:
    """Простой keyword-поиск по SQLite (BM25-like без эмбеддингов)."""
    if not os.path.exists(db_path) or not keywords:
        return []

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Подсчитываем «score» как количество ключевых слов в тексте чанка
        results: list[tuple[int, sqlite3.Row]] = []

        for row in conn.execute(
            "SELECT chunk_id, text, source, section, doc_title FROM chunks LIMIT 2000"
        ):
            text_lower = row["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                results.append((score, row))

        conn.close()

        # Сортируем по score desc, берём top_k
        results.sort(key=lambda x: x[0], reverse=True)
        chunks = []
        seen_sources: set[str] = set()

        for score, row in results[:top_k * 3]:
            # Дедупликация по source + section
            key = f"{row['source']}::{row['section']}"
            if key in seen_sources:
                continue
            seen_sources.add(key)
            chunks.append(RagChunk(
                text=row["text"],
                source=row["source"],
                section=row["section"] or "",
                doc_title=row["doc_title"] or "",
                score=score,
            ))
            if len(chunks) >= top_k:
                break

        return chunks

    except Exception as e:
        print(f"[RAG] Ошибка поиска: {e}", file=sys.stderr)
        return []


def build_rag_context(chunks: list[RagChunk], max_chars: int = MAX_RAG_CHARS) -> str:
    """Собрать контекст из RAG-чанков."""
    if not chunks:
        return ""

    parts = []
    total = 0
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] {chunk.doc_title or chunk.source} / {chunk.section or 'general'}"
        block = f"{header}\n{chunk.text}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Парсинг diff
# ---------------------------------------------------------------------------

def parse_changed_files(diff: str) -> list[str]:
    """Извлечь список изменённых файлов из diff."""
    files = []
    for line in diff.splitlines():
        if line.startswith("diff --git"):
            # diff --git a/path/file.py b/path/file.py
            parts = line.split()
            if len(parts) >= 4:
                files.append(parts[3][2:])  # убрать "b/"
    return files


def truncate_diff(diff: str, max_chars: int = MAX_DIFF_CHARS) -> str:
    """Обрезать diff если слишком большой, сохранить начало и конец."""
    if len(diff) <= max_chars:
        return diff
    half = max_chars // 2
    return (
        diff[:half]
        + f"\n\n... [DIFF TRUNCATED: {len(diff) - max_chars} chars omitted] ...\n\n"
        + diff[-half:]
    )


# ---------------------------------------------------------------------------
# Qwen API (OpenAI-compatible, через httpx)
# ---------------------------------------------------------------------------

def call_llm(
    system: str,
    user: str,
    model: str,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    max_tokens: int = 2000,
) -> str:
    """Вызов OpenAI-compatible API (Qwen/DashScope) через httpx."""
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx не установлен: pip install httpx")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert code reviewer. Your job is to analyze a git diff (pull request changes) and produce a structured review.

Focus on:
1. **Potential Bugs** — logic errors, off-by-one, null/None dereferences, uncaught exceptions, race conditions, incorrect async/await usage, wrong data types
2. **Architecture Issues** — violations of SOLID/DRY/KISS, tight coupling, missing abstractions, wrong layer placement, security vulnerabilities (SQL injection, XSS, command injection, hardcoded secrets)
3. **Recommendations** — code style, naming, missing tests, performance hints, documentation gaps

Rules:
- Be specific: cite file names and line numbers from the diff when possible
- Only comment on what you can see in the diff
- If RAG context is provided, use it to check consistency with existing patterns
- Skip trivial nitpicks (formatting, whitespace)
- Respond in the same language as the code comments (Russian if Russian, English if English)

Output format — strictly valid JSON:
{
  "summary": "1-2 sentence overall assessment",
  "bugs": ["specific bug description with location", ...],
  "architecture": ["specific architecture issue", ...],
  "recommendations": ["actionable recommendation", ...]
}

If a section has no issues, use an empty list [].
"""


def build_user_prompt(
    diff: str,
    changed_files: list[str],
    rag_context: str,
) -> str:
    files_str = "\n".join(f"  - {f}" for f in changed_files) if changed_files else "  (unknown)"
    rag_section = ""
    if rag_context:
        rag_section = f"\n\n## Project Documentation Context (RAG)\n\n{rag_context}\n"

    return f"""## Changed Files
{files_str}
{rag_section}
## Git Diff

```diff
{diff}
```

Please review this pull request and return your analysis as JSON."""


# ---------------------------------------------------------------------------
# Парсинг ответа LLM
# ---------------------------------------------------------------------------

def parse_review_response(text: str) -> ReviewResult:
    """Извлечь ReviewResult из ответа LLM (JSON или fallback)."""
    # Попытка найти JSON в ответе
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            return ReviewResult(
                summary=data.get("summary", ""),
                bugs=data.get("bugs", []),
                architecture=data.get("architecture", []),
                recommendations=data.get("recommendations", []),
            )
        except json.JSONDecodeError:
            pass

    # Fallback: весь текст в summary
    return ReviewResult(summary=text, bugs=[], architecture=[], recommendations=[])


# ---------------------------------------------------------------------------
# Форматирование вывода
# ---------------------------------------------------------------------------

def format_markdown(result: ReviewResult, rag_sources: list[str]) -> str:
    """Форматировать ревью в Markdown (для GitHub PR comment)."""
    lines = ["## 🤖 AI Code Review (Day 32)\n"]

    if result.summary:
        lines.append(f"**Summary:** {result.summary}\n")

    if result.bugs:
        lines.append("### 🐛 Potential Bugs\n")
        for bug in result.bugs:
            lines.append(f"- {bug}")
        lines.append("")

    if result.architecture:
        lines.append("### 🏗️ Architecture Issues\n")
        for issue in result.architecture:
            lines.append(f"- {issue}")
        lines.append("")

    if result.recommendations:
        lines.append("### 💡 Recommendations\n")
        for rec in result.recommendations:
            lines.append(f"- {rec}")
        lines.append("")

    if not result.bugs and not result.architecture and not result.recommendations:
        lines.append("✅ No significant issues found.\n")

    if rag_sources:
        lines.append("---")
        lines.append(f"*RAG context: {', '.join(rag_sources[:3])}*")

    return "\n".join(lines)


def format_json(result: ReviewResult, rag_sources: list[str]) -> str:
    """JSON-вывод для машинной обработки."""
    return json.dumps({
        "summary": result.summary,
        "bugs": result.bugs,
        "architecture": result.architecture,
        "recommendations": result.recommendations,
        "rag_sources": rag_sources,
    }, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="day32_code_review",
        description="AI Code Review: diff → RAG → Claude → structured review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              git diff HEAD~1 | python day32_code_review.py
              python day32_code_review.py --diff-file pr.diff --format markdown
              python day32_code_review.py --diff-file pr.diff --format json
        """),
    )
    parser.add_argument(
        "--diff-file", "-d",
        help="Путь к файлу с diff (по умолчанию stdin)",
    )
    parser.add_argument(
        "--files", "-f",
        help="Список изменённых файлов через запятую (опционально, парсится из diff)",
    )
    parser.add_argument(
        "--rag-db",
        default=os.environ.get("RAG_DB_PATH", DEFAULT_RAG_DB),
        help=f"Путь к SQLite RAG-индексу (по умолчанию: {DEFAULT_RAG_DB})",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=int(os.environ.get("RAG_TOP_K", DEFAULT_TOP_K)),
        help=f"Количество RAG-результатов (по умолчанию: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("QWEN_MODEL", DEFAULT_MODEL),
        help=f"Модель Qwen (по умолчанию: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("QWEN_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "both"],
        default="markdown",
        help="Формат вывода (по умолчанию: markdown)",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Отключить RAG (только diff)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Максимальное количество токенов в ответе",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # API key
    api_key = os.environ.get("QWEN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key:
        print(
            "[ERROR] QWEN_API_KEY или DASHSCOPE_API_KEY не задан.\n"
            "Установите переменную окружения или добавьте в .env",
            file=sys.stderr,
        )
        return 1

    # Читаем diff
    if args.diff_file:
        diff_path = Path(args.diff_file)
        if not diff_path.exists():
            print(f"[ERROR] Файл не найден: {diff_path}", file=sys.stderr)
            return 1
        diff = diff_path.read_text(encoding="utf-8", errors="replace")
    else:
        if sys.stdin.isatty():
            print("[ERROR] Нет diff. Передайте через stdin или --diff-file", file=sys.stderr)
            return 1
        diff = sys.stdin.read()

    if not diff.strip():
        print("[ERROR] Diff пустой", file=sys.stderr)
        return 1

    # Изменённые файлы
    changed_files: list[str] = []
    if args.files:
        changed_files = [f.strip() for f in args.files.split(",") if f.strip()]
    if not changed_files:
        changed_files = parse_changed_files(diff)

    print(f"[review] Файлов в PR: {len(changed_files)}", file=sys.stderr)
    if changed_files:
        for f in changed_files[:10]:
            print(f"  - {f}", file=sys.stderr)

    # Обрезаем diff
    diff_truncated = truncate_diff(diff)
    print(f"[review] Diff: {len(diff)} chars → {len(diff_truncated)} chars", file=sys.stderr)

    # RAG
    rag_context = ""
    rag_sources: list[str] = []

    if not args.no_rag:
        keywords = extract_keywords(diff_truncated, changed_files)
        print(f"[review] RAG keywords: {keywords[:10]}", file=sys.stderr)

        chunks = rag_search(args.rag_db, keywords, args.rag_top_k)
        print(f"[review] RAG chunks найдено: {len(chunks)}", file=sys.stderr)

        rag_context = build_rag_context(chunks)
        rag_sources = list({c.source for c in chunks})
    else:
        print("[review] RAG отключён (--no-rag)", file=sys.stderr)

    # Строим промпт
    user_prompt = build_user_prompt(diff_truncated, changed_files, rag_context)

    # Вызываем Qwen
    print(f"[review] Вызов {args.model} ({args.base_url})...", file=sys.stderr)
    try:
        raw_response = call_llm(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            model=args.model,
            api_key=api_key,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
        )
    except Exception as e:
        print(f"[ERROR] LLM API error: {e}", file=sys.stderr)
        return 1

    # Парсим ответ
    result = parse_review_response(raw_response)
    result.rag_sources = rag_sources

    # Выводим результат
    if args.format == "json":
        print(format_json(result, rag_sources))
    elif args.format == "both":
        print(format_markdown(result, rag_sources))
        print("\n---\n")
        print(format_json(result, rag_sources))
    else:
        print(format_markdown(result, rag_sources))

    # Summary в stderr для логов
    total_issues = len(result.bugs) + len(result.architecture) + len(result.recommendations)
    print(
        f"\n[review] Готово: {len(result.bugs)} bugs, "
        f"{len(result.architecture)} architecture, "
        f"{len(result.recommendations)} recommendations "
        f"({total_issues} total)",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
