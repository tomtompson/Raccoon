from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from time import perf_counter, time
from typing import Any


BenchmarkData = dict[str, dict[str, Any]]
DocumentRecord = dict[str, Any]


class BaseRetriever(ABC):
    retriever_type = "base"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.benchmark_data: BenchmarkData = self._empty_benchmark_data()
        self.processed_documents: list[DocumentRecord] = []
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.query_history: list[dict[str, Any]] = []
        self.is_ready = False
        self._index_elapsed_seconds = 0.0
        self._query_counter = 0

    def preprocess(
        self,
        rows: list[dict[str, Any]],
    ) -> BenchmarkData:
        """Build the shared benchmark payload from critiquer rows."""
        benchmark = self._empty_benchmark_data()

        for index, raw_row in enumerate(rows):
            row = self._normalize_row(raw_row, index)
            document_id = str(row["metadata"]["document_id"])
            query_text = row.get("question", "")

            benchmark["documents"].setdefault(
                document_id,
                {
                    "document_id": document_id,
                    "text": row["passage"],
                    "metadata": dict(row["metadata"]),
                },
            )

            if not query_text:
                continue

            query_id = str(row["query_id"])
            benchmark["queries"].setdefault(
                query_id,
                {
                    "query_id": query_id,
                    "text": query_text,
                },
            )

            qrel_id = f"{query_id}:{document_id}"
            benchmark["qrels"][qrel_id] = {
                "qrel_id": qrel_id,
                "query_id": query_id,
                "document_id": document_id,
                "answer": row.get("answer", ""),
                "groundedness_score": row.get("groundedness_score"),
                "relevance_score": row.get("relevance_score"),
                "standalone_score": row.get("standalone_score"),
                "total_score": row.get("total_score"),
            }

        self.benchmark_data = benchmark
        self.processed_documents = self._build_processed_documents(benchmark)
        return benchmark

    def process_documents(
        self,
        rows: list[dict[str, Any]],
    ) -> list[DocumentRecord]:
        """Preprocess rows, build retriever-specific state, and mark the retriever ready."""
        started_at = perf_counter()
        self.preprocess(rows)
        self._build_index(self.processed_documents)
        self._index_elapsed_seconds = perf_counter() - started_at
        self.is_ready = True
        self._update_metrics()
        return self.processed_documents

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Run one query and store the result in the shared benchmark payload."""
        self._ensure_ready()
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        normalized_query = self._normalize_text(query)
        capped_top_k = min(top_k, len(self.processed_documents))

        started_at = perf_counter()
        hits = self._search(query=normalized_query, top_k=capped_top_k)
        elapsed_seconds = round(perf_counter() - started_at, 4)
        query_id = self._lookup_query_id(normalized_query)

        result = {
            "query_id": query_id,
            "query": normalized_query,
            "query_text": normalized_query,
            "top_k": top_k,
            "returned_count": len(hits),
            "hits": hits,
            "search_elapsed_seconds": elapsed_seconds,
            "retriever_type": self.retriever_type,
        }

        self._store_result(query_id=query_id, query_text=normalized_query, result=result)
        self.results = [result]
        self._record_query(result)
        self._update_metrics()
        return result

    def bulk_search(self, top_k: int = 5) -> dict[str, dict[str, Any]]:
        """Run retrieval for every stored benchmark query."""
        self._ensure_ready()

        results: dict[str, dict[str, Any]] = {}
        batch_results: list[dict[str, Any]] = []
        for query_id, payload in self.benchmark_data.get("queries", {}).items():
            result = self.search(str(payload.get("text", "")), top_k=top_k)
            results[query_id] = self.benchmark_data["results"][query_id]
            batch_results.append(result)

        self.results = batch_results
        return results

    # def batch_search(
    #     self,
    #     queries: list[str],
    #     top_k: int = 5,
    # ) -> list[dict[str, Any]]:
    #     """Keep the older batch API as a thin wrapper around search()."""
    #     batch_results = [self.search(query=query, top_k=top_k) for query in queries]
    #     self.results = batch_results
    #     return batch_results

    def save(self, path: str | Path) -> None:
        """Persist shared retriever state; subclasses can append their own artifacts."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "retriever_type": self.retriever_type,
            "config": self.config,
            "benchmark_data": self.benchmark_data,
            "metrics": self.metrics,
            "query_history": self.query_history,
            "processed_documents": self.processed_documents,
            "is_ready": self.is_ready,
            "query_counter": self._query_counter,
            "index_elapsed_seconds": round(self._index_elapsed_seconds, 4),
        }

        (output_path / "state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._save_artifacts(output_path)

    def load(self, path: str | Path) -> None:
        """Load shared retriever state and then restore subclass-specific artifacts."""
        input_path = Path(path)
        state = json.loads((input_path / "state.json").read_text(encoding="utf-8"))

        self.config = state.get("config", {})
        self.benchmark_data = state.get("benchmark_data", self._empty_benchmark_data())
        self.metrics = state.get("metrics", {})
        self.query_history = state.get("query_history", [])
        self.processed_documents = state.get("processed_documents", [])
        self.is_ready = bool(state.get("is_ready", False))
        self._query_counter = int(state.get("query_counter", 0))
        self._index_elapsed_seconds = float(state.get("index_elapsed_seconds", 0.0))
        self.results = list(self.benchmark_data.get("results", {}).values())
        self._load_artifacts(input_path)
        self._update_metrics()

    def reset_history(self) -> None:
        """Clear runtime search history without touching indexed documents or qrels."""
        self.results = []
        self.query_history = []
        self._query_counter = 0
        self.benchmark_data["results"] = {}
        self._update_metrics()

    @abstractmethod
    def _prepare_retrieval_state(self, documents: list[DocumentRecord]) -> None:
        """Build the child-specific search state, such as an index or embeddings."""
        pass

    @abstractmethod
    def _search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        """Execute retrieval against the child-specific search state."""
        pass

    def _build_index(self, documents: list[DocumentRecord]) -> None:
        """Keep a single entry point for subclasses that build retrieval state."""
        self._prepare_retrieval_state(documents)

    def _save_artifacts(self, path: Path) -> None:
        """Allow subclasses to persist implementation-specific state."""
        return None

    def _load_artifacts(self, path: Path) -> None:
        """Allow subclasses to restore implementation-specific state."""
        return None

    def _empty_benchmark_data(self) -> BenchmarkData:
        """Create a fresh benchmark payload with the shared retriever keys."""
        return {
            "documents": {},
            "queries": {},
            "qrels": {},
            "results": {},
        }

    def _build_processed_documents(
        self,
        benchmark: BenchmarkData,
    ) -> list[DocumentRecord]:
        """Convert stored benchmark documents into the plain records child retrievers use."""
        processed_documents: list[DocumentRecord] = []
        for payload in benchmark.get("documents", {}).values():
            processed_documents.append(
                {
                    "document_id": payload["document_id"],
                    "text": payload["text"],
                    "metadata": dict(payload.get("metadata") or {}),
                }
            )
        return processed_documents

    def _resolve_document_id(
        self,
        row: dict[str, Any],
        index: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Use an explicit document id when present, otherwise derive a stable one."""
        source_metadata = metadata if metadata is not None else dict(row.get("metadata") or {})
        for key in ("document_id", "doc_id", "id"):
            value = row.get(key)
            if value:
                return str(value)
        for key in ("document_id", "doc_id", "id"):
            value = source_metadata.get(key)
            if value:
                return str(value)
        return self._stable_document_id(row, index)

    def _resolve_query_id(self, row: dict[str, Any], question: str) -> str:
        """Use an explicit query id when present, otherwise derive a stable one."""
        for key in ("query_id", "question_id", "id"):
            value = row.get(key)
            if value:
                return str(value)
        return self._stable_query_id(question)

    def _normalize_text(self, text: str) -> str:
        """Collapse whitespace so ids and matching stay stable."""
        return " ".join(str(text).split())

    def _resolve_passage(self, row: dict[str, Any]) -> str:
        """Read and normalize the passage text from one critiquer row."""
        return self._normalize_text(str(row.get("passage", "")))

    def _normalize_row(self, row: dict[str, Any], index: int) -> dict[str, Any]:
        """Normalize one critiquer row before it is turned into benchmark data."""
        normalized_row = dict(row)
        metadata = dict(normalized_row.get("metadata") or {})
        question = self._normalize_text(str(normalized_row.get("question", "")))
        passage = self._resolve_passage(normalized_row)
        metadata["document_id"] = self._resolve_document_id(normalized_row, index, metadata)

        normalized_row["metadata"] = metadata
        normalized_row["passage"] = passage
        if question:
            normalized_row["question"] = question
            normalized_row["query_id"] = self._resolve_query_id(normalized_row, question)
        return normalized_row

    def _stable_document_id(self, row: dict[str, Any], index: int) -> str:
        """Build a deterministic id from passage content and source metadata."""
        metadata = dict(row.get("metadata") or {})
        payload = {
            "source": metadata.get("source", ""),
            "page": metadata.get("page", ""),
            "page_label": metadata.get("page_label", ""),
            "passage": self._resolve_passage(row),
            "index": index,
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _stable_query_id(self, question: str) -> str:
        """Build a deterministic id from the normalized query text."""
        payload = {"question": self._normalize_text(question)}
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _lookup_query_id(self, query: str) -> str:
        """Reuse the benchmark query id when the search matches a stored query text."""
        normalized_query = self._normalize_text(query)
        for query_id, payload in self.benchmark_data.get("queries", {}).items():
            if payload.get("text") == normalized_query:
                return query_id
        return self._stable_query_id(normalized_query)

    def _store_result(
        self,
        query_id: str,
        query_text: str,
        result: dict[str, Any],
    ) -> None:
        """Write one search result into the shared benchmark results section."""
        self.benchmark_data["results"][query_id] = {
            "query_id": query_id,
            "query_text": query_text,
            "top_k": result["top_k"],
            "returned_count": result["returned_count"],
            "hits": [dict(hit) for hit in result["hits"]],
        }

    def _build_hit(
        self,
        document_id: str,
        score: float,
        rank: int,
    ) -> dict[str, Any]:
        """Build the normalized hit payload shared by all retrievers."""
        return {
            "rank": rank,
            "score": round(float(score), 6),
            "document_id": document_id,
        }

    def _record_query(self, result: dict[str, Any]) -> None:
        """Append one search execution to the runtime query history."""
        self._query_counter += 1
        history_record = {
            "query_index": self._query_counter,
            "query_id": result["query_id"],
            "timestamp": int(time()),
            "query": result["query"],
            "top_k": result["top_k"],
            "returned_count": result["returned_count"],
            "retriever_type": result["retriever_type"],
            "search_elapsed_seconds": result["search_elapsed_seconds"],
            "top_k_document_ids": [hit["document_id"] for hit in result["hits"]],
            "scores": [hit["score"] for hit in result["hits"]],
            "hits": result["hits"],
        }
        self.query_history.append(history_record)

    def _ensure_ready(self) -> None:
        """Guard search methods until the retriever has built its search state."""
        if not self.is_ready:
            raise RuntimeError(
                "Retriever is not ready. Call process_documents(...) before search."
            )

    def _update_metrics(self) -> None:
        """Refresh summary metrics after indexing or searching."""
        search_elapsed_values = [
            record["search_elapsed_seconds"] for record in self.query_history
        ]
        top_k_values = [record["top_k"] for record in self.query_history]
        self.metrics.update(
            {
                "retriever_type": self.retriever_type,
                "processed_documents_total": len(self.processed_documents),
                "benchmark_documents_total": len(self.benchmark_data.get("documents", {})),
                "benchmark_queries_total": len(self.benchmark_data.get("queries", {})),
                "benchmark_qrels_total": len(self.benchmark_data.get("qrels", {})),
                "benchmark_results_total": len(self.benchmark_data.get("results", {})),
                "queries_total": len(self.query_history),
                "search_calls": len(self.query_history),
                "average_top_k": round(mean(top_k_values), 4) if top_k_values else 0.0,
                "elapsed_seconds_indexing": round(self._index_elapsed_seconds, 4),
                "elapsed_seconds_search_total": round(sum(search_elapsed_values), 4),
                "average_search_elapsed_seconds": round(mean(search_elapsed_values), 4)
                if search_elapsed_values
                else 0.0,
                "last_returned_count": self.query_history[-1]["returned_count"]
                if self.query_history
                else 0,
            }
        )
