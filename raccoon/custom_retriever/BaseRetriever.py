from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from raccoon.types import Corpus, Metrics, Queries, Results

if TYPE_CHECKING:
    from raccoon.custom_retriever.reranker.BaseReranker import BaseReranker


class BaseRetriever(ABC):
    retriever_type = "base"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        corpus: Corpus | None = None,
        queries: Queries | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self.config = config or {}
        self.corpus = corpus
        self.queries = queries
        self.processed_documents: list = []
        self.results: Results = {}
        self.is_ready = False
        self.reranker = reranker
        self.metrics: Metrics = {}
        self.retrieval_metrics: Metrics = {
            self.retriever_type: {},
            "rerank": {}
        }
        self.rerank_results: Results = {}
        self.rerank_metrics: Metrics = {}

    def create_index(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support create_index().")

    def index_corpus(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support index_corpus().")

    def encode(self, *args, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not support encode().")

    @abstractmethod
    def search(self, top_k: int | None = None, *args, **kwargs) -> Results:
        pass

    def add_retrieval_result(self, result: tuple[dict[str, Any]]) -> None:
        for d in result:
            metric = list(d.keys())[0].split("@")[0].lower()
            self.retrieval_metrics[self.retriever_type][metric] = d
    
    def add_rerank_retrieval_result(self, result: tuple[dict[str, Any]]) -> None:
        for d in result:
            metric = list(d.keys())[0].split("@")[0].lower()
            self.retrieval_metrics["rerank"][metric] = d
            
    def store_rerank_results(self) -> dict:
        if self.reranker is None:
            return {}

        self.rerank_results, self.rerank_metrics = self.reranker.rerank(
            self.corpus,
            self.queries,
            self.results,
        )
        return self.rerank_results

    
