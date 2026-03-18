from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from statistics import mean
from time import perf_counter, time
from typing import Any, Iterable

from langchain_core.documents import Document


class BaseRetriever(ABC):
    retriever_type = "base"

    def __init__(
        self,
        documents: list[Document] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.documents = documents or []
        self.processed_documents: list[Document] = []
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.query_history: list[dict[str, Any]] = []
        self.is_ready = False
        self._index_elapsed_seconds = 0.0
        self._query_counter = 0

    def preprocess_documents(
        self,
        documents: Iterable[Document] | None = None,
    ) -> list[Document]:
        source_documents = list(documents if documents is not None else self.documents)
        processed_documents: list[Document] = []

        for index, document in enumerate(source_documents):
            metadata = dict(document.metadata or {})
            metadata["document_id"] = self._resolve_document_id(metadata, index)

            processed_documents.append(
                Document(
                    page_content=self._normalize_text(document.page_content),
                    metadata=metadata,
                )
            )

        self.processed_documents = processed_documents
        return processed_documents

    def process_documents(
        self,
        documents: list[Document] | None = None,
    ) -> list[Document]:
        if documents is not None:
            self.documents = documents

        started_at = perf_counter()
        processed_documents = self.preprocess_documents(self.documents)
        self._build_index(processed_documents)
        self._index_elapsed_seconds = perf_counter() - started_at
        self.is_ready = True
        self._update_metrics()
        return processed_documents

    def search(self, query: str, top_k: int = 5) -> dict[str, Any]:
        self._ensure_ready()
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        started_at = perf_counter()
        hits = self._search(query=query, top_k=min(top_k, len(self.processed_documents)))
        elapsed_seconds = round(perf_counter() - started_at, 4)
        result = {
            "query": query,
            "top_k": top_k,
            "returned_count": len(hits),
            "hits": hits,
            "search_elapsed_seconds": elapsed_seconds,
            "retriever_type": self.retriever_type,
        }

        self.results = [result]
        self._record_query(result)
        self._update_metrics()
        return result

    def batch_search(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        batch_results = [self.search(query=query, top_k=top_k) for query in queries]
        self.results = batch_results
        return batch_results

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "retriever_type": self.retriever_type,
            "config": self.config,
            "metrics": self.metrics,
            "query_history": self.query_history,
            "documents": [self._serialize_document(doc) for doc in self.documents],
            "processed_documents": [
                self._serialize_document(doc) for doc in self.processed_documents
            ],
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
        input_path = Path(path)
        state = json.loads((input_path / "state.json").read_text(encoding="utf-8"))

        self.config = state.get("config", {})
        self.metrics = state.get("metrics", {})
        self.query_history = state.get("query_history", [])
        self.documents = [
            self._deserialize_document(record) for record in state.get("documents", [])
        ]
        self.processed_documents = [
            self._deserialize_document(record)
            for record in state.get("processed_documents", [])
        ]
        self.is_ready = bool(state.get("is_ready", False))
        self._query_counter = int(state.get("query_counter", 0))
        self._index_elapsed_seconds = float(state.get("index_elapsed_seconds", 0.0))
        self._load_artifacts(input_path)
        self._update_metrics()

    def reset_history(self) -> None:
        self.results = []
        self.query_history = []
        self._query_counter = 0
        self._update_metrics()

    @abstractmethod
    def _build_index(self, documents: list[Document]) -> None:
        pass

    @abstractmethod
    def _search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        pass

    def _save_artifacts(self, path: Path) -> None:
        return None

    def _load_artifacts(self, path: Path) -> None:
        return None

    def _resolve_document_id(self, metadata: dict[str, Any], index: int) -> str:
        for key in ("document_id", "doc_id", "id"):
            value = metadata.get(key)
            if value:
                return str(value)
        return f"doc-{index}-{uuid.uuid4().hex[:8]}"

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).split())

    def _serialize_document(self, document: Document) -> dict[str, Any]:
        return {
            "page_content": document.page_content,
            "metadata": document.metadata,
        }

    def _deserialize_document(self, payload: dict[str, Any]) -> Document:
        return Document(
            page_content=payload["page_content"],
            metadata=payload.get("metadata", {}),
        )

    def _build_hit(
        self,
        document: Document,
        score: float,
        rank: int,
    ) -> dict[str, Any]:
        return {
            "rank": rank,
            "score": round(float(score), 6),
            "document_id": str(document.metadata["document_id"]),
            "content": document.page_content,
            "metadata": dict(document.metadata),
        }

    def _record_query(self, result: dict[str, Any]) -> None:
        self._query_counter += 1
        history_record = {
            "query_index": self._query_counter,
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
        if not self.is_ready:
            raise RuntimeError(
                "Retriever is not ready. Call process_documents(...) before search."
            )

    def _update_metrics(self) -> None:
        search_elapsed_values = [
            record["search_elapsed_seconds"] for record in self.query_history
        ]
        top_k_values = [record["top_k"] for record in self.query_history]
        self.metrics.update(
            {
                "retriever_type": self.retriever_type,
                "documents_total": len(self.documents),
                "processed_documents_total": len(self.processed_documents),
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
