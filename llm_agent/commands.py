"""Команды developer assistant поверх project docs и git MCP."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_agent.domain.models import ChatMessage
from mcp_client.client import MCPClient
from mcp_client.config import MCPConfigParser
from rag_indexer.src.embedding.provider import EmbeddingProvider
from rag_indexer.src.retrieval.retriever import BM25Retriever, HybridRetriever, VectorRetriever
from rag_indexer.src.storage.index_store import IndexStore


class HelpCommandHandler:
    """
    /help [вопрос]

    Режимы:
      /help                → общая справка о проекте
      /help что такое X    → RAG-запрос по документации
      /help git            → git MCP-инструменты + ответ
    """

    def __init__(
        self,
        llm_agent,
        project_root: str | Path,
        rag_db_path: str | Path | None = None,
        repo_path: str | Path | None = None,
        mcp_config_path: str | Path | None = None,
    ) -> None:
        self.llm = llm_agent
        self.project_root = Path(project_root).resolve()
        self.repo_path = str((Path(repo_path) if repo_path else self.project_root).resolve())
        self.db_path = Path(
            rag_db_path
            or os.environ.get("PROJECT_DOCS_DB_PATH")
            or self.project_root / "project_docs.db"
        ).resolve()
        self.mcp_config_path = Path(
            mcp_config_path or self.project_root / "config" / "mcp-servers.md"
        ).resolve()
        self._store: IndexStore | None = None
        self._retrievers: dict[str, object] = {}
        self._git_client: MCPClient | None = None

    def handle(self, query: str) -> str:
        cleaned = query.strip()
        if not cleaned:
            return self._overview()

        lowered = cleaned.lower()
        if any(token in lowered for token in ("git", "ветк", "branch", "commit", "diff", "измен")):
            return self._git_context(cleaned)
        return self._rag_answer(cleaned)

    def _overview(self) -> str:
        branch = self._call_git_tool("get_current_branch", {"repo_path": self.repo_path}) or "unknown"
        project_name = self._read_index_meta("project_name") or self.project_root.name
        components_count = self._read_index_meta("project_meta_components_found") or "?"
        description = self._read_readme_summary()
        components = self._pick_components_preview()

        lines = [
            f"🌿 Ветка: {branch}",
            f"📦 Проект: {project_name}",
            f"🧩 Компонентов: {components_count}",
        ]
        if description:
            lines.append(description)
        if components:
            lines.append("Компоненты:")
            for item in components:
                lines.append(f"  - {item}")
        lines.append("")
        lines.append("Задай вопрос: /help <что хочешь узнать>")
        lines.append("Источники: README.md, components.md")
        return "\n".join(lines)

    def _git_context(self, query: str) -> str:
        lowered = query.lower()
        payload_lines: list[str] = []

        if any(token in lowered for token in ("git", "ветк", "branch")):
            payload_lines.append(
                f"Текущая ветка:\n{self._call_git_tool('get_current_branch', {'repo_path': self.repo_path})}"
            )

        if any(token in lowered for token in ("commit", "коммит", "истор")):
            payload_lines.append(
                "Последние коммиты:\n"
                + self._to_pretty_text(
                    self._call_git_tool(
                        "get_recent_commits",
                        {"repo_path": self.repo_path, "limit": 5},
                    )
                )
            )

        if any(token in lowered for token in ("diff", "измен", "status", "файл")):
            payload_lines.append(
                "Изменённые файлы:\n"
                + self._to_pretty_text(
                    self._call_git_tool("list_changed_files", {"repo_path": self.repo_path})
                )
            )
            payload_lines.append(
                "Diff:\n"
                + self._call_git_tool(
                    "get_file_diff",
                    {"repo_path": self.repo_path, "file_path": ""},
                )
            )

        if not payload_lines:
            payload_lines.append(
                f"Текущая ветка:\n{self._call_git_tool('get_current_branch', {'repo_path': self.repo_path})}"
            )

        return self._run_llm(
            system_prompt=(
                "Ты — ассистент разработчика. "
                "Отвечай только на основе git-контекста. "
                "Если данных недостаточно, скажи об этом прямо."
            ),
            user_prompt=f"Запрос пользователя: {query}\n\nGit-контекст:\n{chr(10).join(payload_lines)}",
        )

    def _rag_answer(self, query: str) -> str:
        results = self._search_docs(query, top_k=5)
        if not results:
            return (
                "Не нашёл в документации. "
                "Доступные темы: компоненты, архитектура, API-интеграция, установка."
            )

        context_parts = []
        for idx, item in enumerate(results, start=1):
            header = f"[{idx}] {item.source} / {item.section or 'без раздела'}"
            context_parts.append(f"{header}\n{item.text}")

        project_name = self._read_index_meta("project_name") or self.project_root.name
        components_count = self._read_index_meta("project_meta_components_found") or "?"
        answer = self._run_llm(
            system_prompt=(
                f"Ты — ассистент разработчика проекта {project_name}.\n"
                f"Проект: компонентов {components_count}.\n\n"
                "Отвечай на вопросы о проекте используя ТОЛЬКО предоставленный контекст.\n"
                "Если информации нет в контексте — честно скажи 'не нашёл в документации'.\n"
                "Не придумывай детали реализации.\n"
                "В конце всегда добавляй строку вида [SOURCES] file1.md, file2.md."
            ),
            user_prompt=f"Контекст из документации:\n\n{chr(10).join(context_parts)}\n\nВопрос: {query}",
        ).strip()

        if "[SOURCES]" not in answer:
            answer = f"{answer}\n[SOURCES] {', '.join(self._unique_sources(results))}"
        return answer

    def _search_docs(self, query: str, top_k: int = 5):
        retriever = self._get_retriever("hybrid")
        if retriever is None:
            return []
        return retriever.search(query, top_k=top_k)

    def _get_retriever(self, mode: str):
        if not self.db_path.exists():
            return None

        store = self._get_store()
        if store.get_stats()["chunks"] == 0:
            return None

        if not self._retrievers:
            provider_name = "ollama" if os.environ.get("LLM_MODE", "cloud").strip().lower() == "local" else "qwen"
            if provider_name == "qwen":
                api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY", "")
                embedder = EmbeddingProvider.create("qwen", api_key=api_key)
            else:
                embedder = EmbeddingProvider.create("ollama")
            vector = VectorRetriever(store=store, embedder=embedder)
            bm25 = BM25Retriever(store=store)
            self._retrievers["vector"] = vector
            self._retrievers["bm25"] = bm25
            self._retrievers["hybrid"] = HybridRetriever(vector, bm25)

        return self._retrievers.get(mode)

    def _get_store(self) -> IndexStore:
        if self._store is None:
            self._store = IndexStore(self.db_path)
        return self._store

    def _read_index_meta(self, key: str) -> str | None:
        if not self.db_path.exists():
            return None
        return self._get_store().get_meta(key)

    def _get_git_client(self) -> MCPClient | None:
        if self._git_client is not None:
            return self._git_client

        try:
            parser = MCPConfigParser(config_path=self.mcp_config_path)
            for config in parser.load():
                if config.name == "git_server":
                    self._git_client = MCPClient(config)
                    return self._git_client
        except Exception:
            return None
        return None

    def _call_git_tool(self, tool_name: str, arguments: dict) -> str:
        try:
            client = self._get_git_client()
            if client is None:
                return self._fallback_git_tool(tool_name, arguments)
            return client.call_tool(tool_name, arguments)
        except Exception as exc:
            fallback = self._fallback_git_tool(tool_name, arguments)
            if fallback:
                return fallback
            return f"git MCP error: {exc}"

    def _run_llm(self, system_prompt: str, user_prompt: str) -> str:
        response = self.llm._llm_client.generate(
            [
                ChatMessage(role="system", content=system_prompt),
                ChatMessage(role="user", content=user_prompt),
            ]
        )
        return response.text.strip()

    def _fallback_git_tool(self, tool_name: str, arguments: dict) -> str:
        repo_path = arguments.get("repo_path", self.repo_path)
        try:
            if tool_name == "get_current_branch":
                result = subprocess.run(
                    ["git", "-C", repo_path, "rev-parse", "--abbrev-ref", "HEAD"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return result.stdout.strip() or "unknown (not a git repo)"
            if tool_name == "get_recent_commits":
                limit = str(arguments.get("limit", 5))
                result = subprocess.run(
                    [
                        "git",
                        "-C",
                        repo_path,
                        "log",
                        "--format=%H|%s|%an|%ad",
                        "--date=short",
                        "-n",
                        limit,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                commits = []
                for line in result.stdout.splitlines():
                    parts = line.split("|", 3)
                    if len(parts) == 4:
                        commits.append(
                            {
                                "hash": parts[0],
                                "message": parts[1],
                                "author": parts[2],
                                "date": parts[3],
                            }
                        )
                return json.dumps(commits, ensure_ascii=False)
            if tool_name == "list_changed_files":
                result = subprocess.run(
                    ["git", "-C", repo_path, "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                payload = {"staged": [], "unstaged": [], "untracked": []}
                for line in result.stdout.splitlines():
                    if len(line) < 4:
                        continue
                    x_status = line[0]
                    y_status = line[1]
                    path = line[3:].strip()
                    if x_status == "?" and y_status == "?":
                        payload["untracked"].append(path)
                    else:
                        if x_status != " ":
                            payload["staged"].append(path)
                        if y_status != " ":
                            payload["unstaged"].append(path)
                return json.dumps(payload, ensure_ascii=False)
            if tool_name == "get_file_diff":
                file_path = arguments.get("file_path", "")
                cmd = ["git", "-C", repo_path, "diff", "HEAD"]
                limit = 200
                if file_path:
                    cmd.extend(["--", file_path])
                    limit = 100
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return "\n".join(result.stdout.splitlines()[:limit]) or "No changes"
        except Exception:
            return ""
        return ""

    def _pick_description(self, results: list) -> str:
        for item in results:
            if item.source.endswith("README.md"):
                paragraphs = [
                    line.strip()
                    for line in item.text.splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                if paragraphs:
                    return paragraphs[0][:260]
        if results:
            return results[0].text[:260]
        return ""

    def _read_readme_summary(self) -> str:
        readme_path = self._read_index_meta("readme_path")
        if not readme_path:
            return ""
        path = Path(readme_path)
        if not path.exists():
            return ""
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("|")
        ]
        return lines[0][:260] if lines else ""

    def _pick_components_preview(self) -> list[str]:
        docs_dir = self._read_index_meta("docs_dir")
        if docs_dir:
            components_path = Path(docs_dir) / "components.md"
            if components_path.exists():
                preview = []
                for line in components_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    stripped = line.strip()
                    if stripped.startswith("## "):
                        preview.append(stripped.removeprefix("## ").strip())
                    elif stripped.startswith("| `"):
                        parts = stripped.split("`")
                        if len(parts) > 1:
                            preview.append(parts[1])
                    if len(preview) >= 5:
                        return preview
        preview: list[str] = []
        results = self._search_docs("components list component names", top_k=3)
        for item in results:
            for line in item.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("## "):
                    preview.append(stripped.removeprefix("## ").strip())
                elif stripped.startswith("| `"):
                    preview.append(stripped.split("`")[1])
                if len(preview) >= 5:
                    return preview
        return preview

    def _unique_sources(self, results: list) -> list[str]:
        seen: list[str] = []
        for item in results:
            if item.source not in seen:
                seen.append(item.source)
        return seen

    def _to_pretty_text(self, payload) -> str:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return payload
        return json.dumps(payload, ensure_ascii=False, indent=2)
