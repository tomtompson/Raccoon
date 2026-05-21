from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from time import perf_counter, time
from typing import Any

from raccoon.custom_retriever.util.Reranker import Reranker




class BaseRetriever(ABC):
    retriever_type = "base"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        corpus: dict | None = None,
        queries: dict | None = None,
        reranker: Reranker | None = None
    ) -> None:
        self.config = config or {}
        self.corpus = corpus
        self.queries = queries
        self.processed_documents: list = []
        self.results: list[dict[str, Any]] = []
        self.is_ready = False
        self.reranker = reranker
        self.metrics = {}
        self.retrieval_metrics = {}
        self.rerank_results = {}
        self.rerank_metrics = {}
        self.rerank_retrieval_metrics = {}

    @abstractmethod
    def create_index(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def index_corpus(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, top_k: int, *args, **kwargs) -> dict:
        pass

    def add_retrieval_result(self, result: tuple[dict[str, Any]]) -> None:
        for d in result:
            metric = list(d.keys())[0].split("@")[0].lower()
            self.retrieval_metrics[metric] = d
            
    def store_rerank_results(self) -> dict:
        if self.reranker is None:
            return {}

        self.rerank_results = self.reranker.rerank_with_transformers(
            self.corpus,
            self.queries,
            self.results,
        )
        self.rerank_metrics = getattr(self.reranker, "rerank_metrics", {}) or {}
        return self.rerank_results

    
