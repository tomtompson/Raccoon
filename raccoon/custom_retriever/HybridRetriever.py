from raccoon.custom_retriever.BaseRetriever import BaseRetriever
from raccoon.custom_retriever.util.utils import (rrf_fuse)
from time import perf_counter

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
        index_start = perf_counter()
        for retriever, _weight in self.retrievers:
            self._call_index_method(retriever, *args, **kwargs)
        index_latency = perf_counter() - index_start
        document_count = len(self.corpus or {})
        self.metrics["index_time"] = {
            "indexing": {
                "time_in_seconds": index_latency,
                "documents": document_count,
                "docs_per_second": document_count / index_latency if index_latency else 0.0,
                "retrievers": len(self.retrievers),
            },
        }

    def search(self, top_k: int | None = None) -> dict:
        top_k = top_k or self.config.get("top_k", 20)
        per_query_rankings = {}
        search_start = perf_counter()

        for retriever, weight in self.retrievers:
            results = retriever.search(top_k=top_k)

            for query_id, doc_scores in results.items():
                ranking = list(doc_scores.keys())
                per_query_rankings.setdefault(str(query_id), []).append((ranking, weight))

        self.results = {
            query_id: dict(list(rrf_fuse(rankings, k=self.k).items())[:top_k])
            for query_id, rankings in per_query_rankings.items()
        }
        search_latency = perf_counter() - search_start
        query_count = len(self.results)
        result_counts = [len(row) for row in self.results.values()]
        total_results = sum(result_counts)
        scores = [score for row in self.results.values() for score in row.values()]
        self.metrics["query_time"] = {
            "search": {
                "time_in_seconds": search_latency,
                "time_per_query_in_seconds": search_latency / query_count if query_count else 0.0,
                "queries_per_second": query_count / search_latency if search_latency else 0.0,
                "queries": query_count,
                "top_k": top_k,
                "retrievers": len(self.retrievers),
                "rrf_k": self.k,
                "total_results": total_results,
                "avg_results_per_query": total_results / query_count if query_count else 0.0,
                "min_results_per_query": min(result_counts) if result_counts else 0,
                "max_results_per_query": max(result_counts) if result_counts else 0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
            },
        }

        if self.reranker is not None:
            self.store_rerank_results()

        return self.results
