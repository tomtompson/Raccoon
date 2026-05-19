from raccoon.custom_retriever.BaseRetriever import BaseRetriever
from raccoon.custom_retriever.util.Reranker import Reranker
from raccoon.custom_retriever.util.utils import (rrf_fuse)
from typing import Tuple, List, Any
from numpy import maximum

class HybridRetriever(BaseRetriever):
    retriever_type = "hybrid"

    def __init__(self, config=None, corpus=None, queries=None, reranker=None, retrievers=None, k=60):
        super().__init__(config, corpus, queries, reranker)
        self.retrievers = retrievers or []
        self.k = k

    def create_index(self, *args, **kwargs) -> None:
        self.index_corpus(*args, **kwargs)

    def encode(self, *args, **kwargs):
        raise NotImplementedError("HybridRetriever does not encode directly.")

    def _call_index_method(self, retriever, *args, **kwargs):
        method = getattr(retriever, "index_corpus", None)
        if not callable(method):
            return

        try:
            method(*args, **kwargs)
        except NotImplementedError:
            return

    def index_corpus(self, *args, **kwargs) -> None:
        for retriever, _weight in self.retrievers:
            self._call_index_method(retriever, *args, **kwargs)

    def search(self, top_k: int | None = None) -> dict:
        top_k = top_k or self.config.get("top_k", 20)
        per_query_rankings = {}

        for retriever, weight in self.retrievers:
            results = retriever.search(top_k=top_k)

            for query_id, doc_scores in results.items():
                ranking = list(doc_scores.keys())
                per_query_rankings.setdefault(str(query_id), []).append((ranking, weight))

        self.results = {
            query_id: dict(list(rrf_fuse(rankings, k=self.k).items())[:top_k])
            for query_id, rankings in per_query_rankings.items()
        }

        if self.reranker is not None:
            self.reranker.rerank_with_transformers(self.corpus, self.queries, self.results)

        return self.results
