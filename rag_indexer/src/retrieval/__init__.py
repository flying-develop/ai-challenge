"""Модуль RAG-запросов: поиск, реранкинг, пайплайн, оценка, цитаты."""

from .retriever import RetrievalResult, RetrievalStrategy, VectorRetriever, BM25Retriever, HybridRetriever
from .reranker import Reranker, ThresholdFilter
from .query_rewrite import QueryRewriter
from .rag_query import RAGContext, RAGQueryBuilder
from .pipeline import RAGAnswer, RAGPipeline, PipelineEvalResult, PipelineEvaluator
from .evaluator import EvalResult, RAGEvaluator, EVAL_QUESTIONS, ANTI_QUESTIONS
from .confidence import ConfidenceScorer, ConfidenceLevel
from .structured_prompt import StructuredRAGPrompt, StructuredPrompt
from .response_parser import ResponseParser, StructuredResponse, SourceRef, Quote
from .formatter import format_structured_response, format_refusal, format_confidence_level

__all__ = [
    "RetrievalResult", "RetrievalStrategy", "VectorRetriever", "BM25Retriever", "HybridRetriever",
    "Reranker", "ThresholdFilter",
    "QueryRewriter",
    "RAGContext", "RAGQueryBuilder",
    "RAGAnswer", "RAGPipeline", "PipelineEvalResult", "PipelineEvaluator",
    "EvalResult", "RAGEvaluator", "EVAL_QUESTIONS", "ANTI_QUESTIONS",
    "ConfidenceScorer", "ConfidenceLevel",
    "StructuredRAGPrompt", "StructuredPrompt",
    "ResponseParser", "StructuredResponse", "SourceRef", "Quote",
    "format_structured_response", "format_refusal", "format_confidence_level",
]
