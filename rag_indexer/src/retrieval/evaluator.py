"""RAG quality evaluator with 10 control questions."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .retriever import RetrievalStrategy, RetrievalResult
from .rag_query import RAGQueryBuilder


EVAL_QUESTIONS = [
    {
        "id": 1,
        "question": "Как установить Podkop на OpenWrt?",
        "expected_source": "install/index.md",
        "keywords": ["установить", "openwrt", "podkop", "установка"],
    },
    {
        "id": 2,
        "question": "Какие требования для установки Podkop (версия ОС, место)?",
        "expected_source": "install/index.md",
        "keywords": ["требования", "версия", "место", "память", "openwrt"],
    },
    {
        "id": 3,
        "question": "Как настроить WireGuard туннель в Podkop?",
        "expected_source": "tunnels/wg_settings/index.md",
        "keywords": ["wireguard", "туннель", "настройка", "wg"],
    },
    {
        "id": 4,
        "question": "Как настроить AmneziaWG туннель?",
        "expected_source": "tunnels/awg_settings/index.md",
        "keywords": ["amneziawg", "awg", "туннель", "настройка"],
    },
    {
        "id": 5,
        "question": "Как совместить Podkop с AdGuard Home?",
        "expected_source": "adguard/index.md",
        "keywords": ["adguard", "adguard home", "совместить", "интеграция"],
    },
    {
        "id": 6,
        "question": "Как просматривать DNS-запросы в Podkop?",
        "expected_source": "dnsmasqlogs/index.md",
        "keywords": ["dns", "запросы", "логи", "dnsmasq"],
    },
    {
        "id": 7,
        "question": "В чём разница между sing-box, xray и mihomo?",
        "expected_source": "dev/singbox-vs-xray-vs-mihomo.md",
        "keywords": ["sing-box", "xray", "mihomo", "разница"],
    },
    {
        "id": 8,
        "question": "Что делать если заблокированные сайты не открываются?",
        "expected_source": "troubleshooting/index.md",
        "keywords": ["заблокированные", "сайты", "не открываются", "troubleshooting"],
    },
    {
        "id": 9,
        "question": "Как изменить DNS-протокол в Podkop (DoH/DoT/UDP)?",
        "expected_source": "dns/index.md",
        "keywords": ["dns", "doh", "dot", "протокол", "udp"],
    },
    {
        "id": 10,
        "question": "Как работать с корпоративным VPN через Podkop?",
        "expected_source": "workvpn/index.md",
        "keywords": ["vpn", "корпоративный", "корпоративным", "workvpn"],
    },
]


@dataclass
class EvalResult:
    question_id: int
    question: str
    retriever_name: str
    source_hit: bool
    keyword_hits: int
    keyword_total: int
    answer_preview: str
    elapsed_ms: float


class RAGEvaluator:
    def __init__(
        self,
        retrievers: list[RetrievalStrategy],
        llm_fn: Optional[Callable[[str, str], str]] = None,
        query_builder: Optional[RAGQueryBuilder] = None,
        top_k: int = 5,
    ):
        self._retrievers = retrievers
        self._llm_fn = llm_fn
        self._builder = query_builder or RAGQueryBuilder()
        self._top_k = top_k

    def run(self) -> list[EvalResult]:
        results = []
        for q in EVAL_QUESTIONS:
            for retriever in self._retrievers:
                t0 = time.monotonic()
                retrieved = retriever.search(q["question"], top_k=self._top_k)
                elapsed = (time.monotonic() - t0) * 1000

                sources = [r.source for r in retrieved]
                source_hit = any(q["expected_source"] in s for s in sources)

                combined_text = " ".join(r.text.lower() for r in retrieved)
                keyword_hits = sum(
                    1 for kw in q["keywords"] if kw.lower() in combined_text
                )

                answer_preview = ""
                if self._llm_fn and retrieved:
                    ctx = self._builder.build(q["question"], retrieved)
                    try:
                        answer = self._llm_fn(ctx.system_prompt, ctx.user_prompt)
                        answer_preview = answer[:300]
                    except Exception as e:
                        answer_preview = f"[ошибка LLM: {e}]"

                results.append(
                    EvalResult(
                        question_id=q["id"],
                        question=q["question"],
                        retriever_name=retriever.name,
                        source_hit=source_hit,
                        keyword_hits=keyword_hits,
                        keyword_total=len(q["keywords"]),
                        answer_preview=answer_preview,
                        elapsed_ms=elapsed,
                    )
                )
        return results

    def print_report(self, results: list[EvalResult]) -> None:
        retrievers = list(dict.fromkeys(r.retriever_name for r in results))
        print("\n" + "=" * 72)
        print("RAG EVALUATION REPORT")
        print("=" * 72)

        # Group by retriever
        for ret_name in retrievers:
            ret_results = [r for r in results if r.retriever_name == ret_name]
            hits = sum(1 for r in ret_results if r.source_hit)
            total_kw = sum(r.keyword_hits for r in ret_results)
            max_kw = sum(r.keyword_total for r in ret_results)
            avg_ms = sum(r.elapsed_ms for r in ret_results) / len(ret_results)

            print(f"\n[{ret_name.upper()}]  source_hits={hits}/{len(ret_results)}  "
                  f"keyword_hits={total_kw}/{max_kw}  avg_ms={avg_ms:.0f}")
            print("-" * 72)
            print(f"{'#':<3} {'Вопрос':<42} {'Hit':<5} {'KW':<6} {'ms':<7}")
            print("-" * 72)
            for r in ret_results:
                hit_str = "✓" if r.source_hit else "✗"
                kw_str = f"{r.keyword_hits}/{r.keyword_total}"
                q_short = r.question[:40] + ".." if len(r.question) > 40 else r.question
                print(f"{r.question_id:<3} {q_short:<42} {hit_str:<5} {kw_str:<6} {r.elapsed_ms:<7.0f}")

        print("=" * 72 + "\n")
