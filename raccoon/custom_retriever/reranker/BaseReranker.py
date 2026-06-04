from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from raccoon.types import Corpus, Metrics, Queries, Results


class BaseReranker(ABC):
    reranker_type = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.rerank_metrics: Metrics = {}

    @abstractmethod
    def _load_model(self) -> None:
        pass

    @abstractmethod
    def rerank(
        self,
        corpus: Corpus,
        queries: Queries,
        results: Results,
    ) -> tuple[Results, Metrics]:
        pass