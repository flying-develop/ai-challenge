"""Query Rewriting для улучшения поиска в RAG.

QueryRewriter генерирует несколько вариантов пользовательского запроса
через LLM, чтобы расширить охват при поиске в документации.

Использование:
    from src.retrieval.query_rewrite import QueryRewriter

    rewriter = QueryRewriter(llm_fn=my_llm_fn)
    variants = rewriter.rewrite("Как установить podkop?")
    # → ["Как установить podkop?",
    #    "Инструкция по установке podkop на OpenWrt",
    #    "Podkop установка и настройка роутер",
    #    "Как поставить podkop пакет opkg"]
"""
from __future__ import annotations

from typing import Callable


class QueryRewriter:
    """Переписывает вопрос пользователя для улучшения поиска.

    Генерирует 2-3 варианта запроса через LLM. Поиск идёт по всем вариантам,
    результаты объединяются и дедуплицируются по chunk_id.
    Исходный вопрос всегда включается первым (страховка от неудачных переформулировок).
    """

    _SYSTEM = (
        "Ты — помощник для улучшения поисковых запросов по технической документации."
    )

    _REWRITE_PROMPT = (
        "Ты — помощник для улучшения поисковых запросов.\n"
        "Пользователь задал вопрос по документации проекта podkop\n"
        "(утилита обхода блокировок для OpenWrt-роутеров).\n\n"
        'Исходный вопрос: "{question}"\n\n'
        "Сгенерируй ровно 2 варианта поискового запроса,\n"
        "которые помогут найти релевантную информацию в документации.\n"
        "Каждый вариант — на отдельной строке, без нумерации.\n"
        "Варианты должны покрывать разные аспекты вопроса."
    )

    def __init__(self, llm_fn: Callable[[str, str], str]):
        """
        Args:
            llm_fn: Функция (system_prompt, user_prompt) -> ответ строкой.
                    Сигнатура совместима с RAGEvaluator.llm_fn и demo_rag_query.py.
        """
        self.llm_fn = llm_fn

    def rewrite(self, question: str) -> list[str]:
        """Вернуть исходный вопрос + до 3 переформулировок.

        Args:
            question: Исходный вопрос пользователя.

        Returns:
            [question, variant1, variant2, variant3] — всегда начинается с оригинала.
            При ошибке LLM возвращает [question] (поиск по исходному запросу).
        """
        prompt = self._REWRITE_PROMPT.format(question=question)
        try:
            response = self.llm_fn(self._SYSTEM, prompt)
            variants = [
                q.strip()
                for q in response.strip().split("\n")
                if q.strip()
            ]
            return [question] + variants[:2]
        except Exception:
            return [question]
