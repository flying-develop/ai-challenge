"""Источники документации для индексации support-бота."""

from .source import DocumentSource, RawDocument
from .local_source import LocalMarkdownSource
from .url_source import URLDocSource
from .github_source import GitHubRepoSource

__all__ = [
    "DocumentSource",
    "RawDocument",
    "LocalMarkdownSource",
    "URLDocSource",
    "GitHubRepoSource",
]
