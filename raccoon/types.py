from __future__ import annotations

from typing import Any, Protocol, TypeAlias

CorpusDocument: TypeAlias = dict[str, Any]
Corpus: TypeAlias = dict[str, CorpusDocument]
Queries: TypeAlias = dict[str, str]
Qrels: TypeAlias = dict[str, dict[str, int]]
Results: TypeAlias = dict[str, dict[str, float]]
Metrics: TypeAlias = dict[str, Any]


class RetrieverProtocol(Protocol):
    retriever_type: str
    corpus: Corpus | None
    queries: Queries | None
    results: Results
    metrics: Metrics
    retrieval_metrics: Metrics
    rerank_results: Results
    rerank_metrics: Metrics

    def search(self, top_k: int | None = None, *args: Any, **kwargs: Any) -> Results:
        ...

