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
    # TODO: THIS AINT IT CHIEF, THIS IS JUST A QUICK FIX TO GET THE HYBRID RETRIEVER WORKING. BUT IS NOT REPRESENT THE INDEX OF EACH METHOD
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
        retriever_timings = []

        fusion_pipeline_start = perf_counter()

        for retriever, weight in self.retrievers:
            retriever_name = retriever.__class__.__name__

            if isinstance(retriever, dict):
                results = retriever
                measured_retriever_time = 0.0
                retriever_name = "precomputed"
            elif hasattr(retriever, "retrieval_metrics") and retriever.retrieval_metrics:
                results = retriever.results
                measured_retriever_time = (
                    retriever.metrics
                    .get("query_time", {})
                    .get("search", {})
                    .get("time_in_seconds", 0.0)
                )
            else:
                results = retriever.search(top_k=top_k)
                measured_retriever_time = (
                    retriever.metrics
                    .get("query_time", {})
                    .get("search", {})
                    .get("time_in_seconds", 0.0)
                )

            retriever_timings.append({
                "retriever": retriever_name,
                "weight": weight,
                "time_in_seconds": measured_retriever_time,
                "queries": len(results),
                "avg_time_per_query": measured_retriever_time / len(results) if results else 0.0,
            })

            for query_id, doc_scores in results.items():
                ranking = list(doc_scores.keys())
                per_query_rankings.setdefault(str(query_id), []).append((ranking, weight))

        rrf_start = perf_counter()

        self.results = {
            query_id: dict(list(rrf_fuse(rankings, k=self.k).items())[:top_k])
            for query_id, rankings in per_query_rankings.items()
        }

        rrf_latency = perf_counter() - rrf_start

        retriever_latency = sum(
            timing["time_in_seconds"] for timing in retriever_timings
        )

        total_search_latency = retriever_latency + rrf_latency
        wrapper_latency = perf_counter() - fusion_pipeline_start

        query_count = len(self.results)
        result_counts = [len(row) for row in self.results.values()]
        total_results = sum(result_counts)
        scores = [score for row in self.results.values() for score in row.values()]

        self.metrics["query_time"] = {
            "search": {
                "time_in_seconds": total_search_latency,
                "retriever_time_in_seconds": retriever_latency,
                "rrf_time_in_seconds": rrf_latency,
                "wrapper_time_in_seconds": wrapper_latency,
                "execution_mode": "sequential",
                "time_per_query_in_seconds": total_search_latency / query_count if query_count else 0.0,
                "queries_per_second": query_count / total_search_latency if total_search_latency else 0.0,
                "queries": query_count,
                "top_k": top_k,
                "retrievers": len(self.retrievers),
                "rrf_k": self.k,
                "retriever_timings": retriever_timings,
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