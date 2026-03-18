"""RAG prompt builder."""
from __future__ import annotations

from dataclasses import dataclass, field

from .retriever import RetrievalResult


@dataclass
class RAGContext:
    system_prompt: str
    user_prompt: str
    sources: list[RetrievalResult]
    query: str


class RAGQueryBuilder:
    _SYSTEM = (
        "Ты — ассистент по документации Podkop. "
        "Отвечай ТОЛЬКО на основе контекста. "
        "Если ответ не найден — скажи честно. "
        "Указывай источники. Отвечай на русском."
    )

    def build(
        self,
        query: str,
        results: list[RetrievalResult],
        max_tokens: int = 3000,
    ) -> RAGContext:
        context_parts = []
        total_chars = 0
        char_limit = max_tokens * 4  # rough chars-per-token estimate

        for i, r in enumerate(results, start=1):
            header = f"[{i}] {r.doc_title or r.source} / {r.section or 'нет раздела'} (score={r.score:.3f})"
            chunk_text = f"{header}\n{r.text}"
            if total_chars + len(chunk_text) > char_limit:
                break
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        context_str = "\n\n".join(context_parts)
        user_prompt = f"Контекст из документации:\n\n{context_str}\n\nВопрос: {query}"

        return RAGContext(
            system_prompt=self._SYSTEM,
            user_prompt=user_prompt,
            sources=results,
            query=query,
        )
