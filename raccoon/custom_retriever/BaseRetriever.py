from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from time import perf_counter, time
from typing import Any

from util.Reranker import Reranker




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

    @abstractmethod
    def create_index(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def index_corpous(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def encode(self, *args, **kwargs):
        pass

    @abstractmethod
    def search(self, top_k: int, *args, **kwargs) -> dict:
        pass

    