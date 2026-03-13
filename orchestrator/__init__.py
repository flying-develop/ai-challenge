"""Orchestration MCP: двухпроходный исследовательский оркестратор.

Координирует несколько MCP-серверов в длинном многошаговом флоу.
Агент (LLM) принимает решения, серверы выполняют работу.
"""

from orchestrator.research_states import ResearchState
from orchestrator.research_context import ResearchContext
from orchestrator.research_orchestrator import ResearchOrchestrator

__all__ = ["ResearchState", "ResearchContext", "ResearchOrchestrator"]
