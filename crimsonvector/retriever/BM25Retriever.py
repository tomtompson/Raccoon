from __future__ import annotations

import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

from elasticsearch import Elasticsearch
from elastic_transport import ConnectionError as ElasticConnectionError
from elasticsearch import ApiError
from langchain_core.documents import Document

from .BaseRetriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"

    def __init__(
        self,
        elasticsearch_url: str,
        index_name: str,
        language: str = "english",
        documents: list[Document] | None = None,
        config: dict[str, Any] | None = None,
        content_field: str = "content",
        metadata_field: str = "metadata",
        refresh_on_write: bool = True,
        timeout: int = 120,
    ) -> None:
        super().__init__(documents=documents, config=config)
        self.elasticsearch_url = elasticsearch_url.rstrip("/")
        self.index_name = index_name
        self.language = language
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.refresh_on_write = refresh_on_write
        self.timeout = timeout
        self.client = Elasticsearch(
            hosts=[self.elasticsearch_url],
            request_timeout=self.timeout,
        )
        self.benchmark_data: dict[str, dict[str, Any]] = {
            "documents": {},
            "queries": {},
            "qrels": {},
            "results": {},
        }

    def preprocess_documents(
        self,
        documents: Iterable[Document] | Iterable[dict[str, Any]] | None = None,
    ) -> list[Document]:
        source_documents = list(documents if documents is not None else self.documents)
        if not source_documents:
            self.benchmark_data = {
                "documents": {},
                "queries": {},
                "qrels": {},
                "results": {},
            }
            self.processed_documents = []
            return []

        first_item = source_documents[0]
        if isinstance(first_item, Document):
            processed_documents = super().preprocess_documents(source_documents)
            self.documents = processed_documents
            self.benchmark_data = {
                "documents": {
                    str(document.metadata["document_id"]): {
                        "document_id": str(document.metadata["document_id"]),
                        "text": document.page_content,
                        "metadata": dict(document.metadata),
                    }
                    for document in processed_documents
                },
                "queries": {},
                "qrels": {},
                "results": {},
            }
            return processed_documents

        benchmark = self._build_benchmark_from_rows(source_documents)
        processed_documents = [
            Document(
                page_content=payload["text"],
                metadata=payload["metadata"],
            )
            for payload in benchmark["documents"].values()
        ]
        self.documents = processed_documents
        self.processed_documents = processed_documents
        self.benchmark_data = benchmark
        return processed_documents

    def process_documents(
        self,
        documents: list[Document] | list[dict[str, Any]] | None = None,
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
        result = super().search(query=query, top_k=top_k)
        query_id = self._resolve_query_id(query)
        self.benchmark_data.setdefault("results", {})[query_id] = {
            "query_id": query_id,
            "query_text": query,
            "top_k": result["top_k"],
            "returned_count": result["returned_count"],
            "hits": [
                {
                    "rank": hit["rank"],
                    "document_id": hit["document_id"],
                    "score": hit["score"],
                }
                for hit in result["hits"]
            ],
        }
        return result

    def bulk_search(self, top_k: int = 5) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for query_id, payload in self.benchmark_data.get("queries", {}).items():
            self.search(payload["text"], top_k=top_k)
            results[query_id] = self.benchmark_data["results"][query_id]
        return results

    def _build_index(self, documents: list[Document]) -> None:
        self._create_index_if_missing()
        for document in documents:
            payload = {
                self.content_field: document.page_content,
                self.metadata_field: dict(document.metadata),
            }
            self.client.index(
                index=self.index_name,
                id=str(document.metadata["document_id"]),
                document=payload,
            )

        if self.refresh_on_write:
            self.client.indices.refresh(index=self.index_name)

        self.metrics.update(
            {
                "elasticsearch_url": self.elasticsearch_url,
                "elasticsearch_index_name": self.index_name,
                "elasticsearch_refresh_on_write": self.refresh_on_write,
            }
        )

    def _search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        payload = {
            "size": top_k,
            "query": {
                "match": {
                    self.content_field: {
                        "query": query,
                    }
                }
            },
        }
        response = self.client.search(index=self.index_name, **payload)
        hits = response.get("hits", {}).get("hits", [])
        ranked_hits: list[dict[str, Any]] = []

        for rank, hit in enumerate(hits, start=1):
            source = hit.get("_source", {})
            metadata = dict(source.get(self.metadata_field, {}))
            metadata.setdefault("document_id", str(hit.get("_id", "")))
            document = Document(
                page_content=str(source.get(self.content_field, "")),
                metadata=metadata,
            )
            ranked_hits.append(
                self._build_hit(
                    document=document,
                    score=float(hit.get("_score", 0.0)),
                    rank=rank,
                )
            )

        return ranked_hits

    def _create_index_if_missing(self) -> None:
        try:
            self.client.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        self.content_field: {"type": "text", "analyzer": self.language},
                        self.metadata_field: {"type": "object", "enabled": True},
                    }
                },
            )
        except ApiError as exc:
            error_type = getattr(exc, "error", None)
            if error_type != "resource_already_exists_exception":
                raise
        except ElasticConnectionError as exc:
            raise RuntimeError(
                f"Unable to reach Elasticsearch at {self.elasticsearch_url}: {exc}"
            ) from exc

    def _save_artifacts(self, path: Path) -> None:
        (path / "bm25_artifacts.json").write_text(
            json.dumps(
                {
                    "elasticsearch_url": self.elasticsearch_url,
                    "index_name": self.index_name,
                    "language": self.language,
                    "content_field": self.content_field,
                    "metadata_field": self.metadata_field,
                    "refresh_on_write": self.refresh_on_write,
                    "timeout": self.timeout,
                    "benchmark_data": self.benchmark_data,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_artifacts(self, path: Path) -> None:
        artifact = json.loads((path / "bm25_artifacts.json").read_text(encoding="utf-8"))
        self.elasticsearch_url = artifact["elasticsearch_url"]
        self.index_name = artifact["index_name"]
        self.language = artifact.get("language", "english")
        self.content_field = artifact["content_field"]
        self.metadata_field = artifact["metadata_field"]
        self.refresh_on_write = bool(artifact["refresh_on_write"])
        self.timeout = int(artifact["timeout"])
        self.benchmark_data = artifact.get(
            "benchmark_data",
            {
                "documents": {},
                "queries": {},
                "qrels": {},
                "results": {},
            },
        )
        self.client = Elasticsearch(
            hosts=[self.elasticsearch_url],
            request_timeout=self.timeout,
        )

    def _build_benchmark_from_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        documents: dict[str, dict[str, Any]] = {}
        queries: dict[str, dict[str, Any]] = {}
        qrels: dict[str, dict[str, Any]] = {}

        for row in rows:
            question = self._normalize_text(str(row.get("question", "")))
            if not question:
                continue

            document_id = self._stable_document_id(row)
            query_id = self._stable_query_id(question)
            metadata = dict(row.get("metadata") or {})
            metadata["document_id"] = document_id

            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "text": self._normalize_text(str(row.get("passage", ""))),
                    "metadata": metadata,
                }

            if query_id not in queries:
                queries[query_id] = {
                    "query_id": query_id,
                    "text": question,
                }

            qrel_id = f"{query_id}:{document_id}"
            qrels[qrel_id] = {
                "query_id": query_id,
                "document_id": document_id,
                "answer": row.get("answer", ""),
                "groundedness_score": row.get("groundedness_score"),
                "relevance_score": row.get("relevance_score"),
                "standalone_score": row.get("standalone_score"),
                "total_score": row.get("total_score"),
            }

        return {
            "documents": documents,
            "queries": queries,
            "qrels": qrels,
            "results": {},
        }

    def _resolve_query_id(self, query: str) -> str:
        normalized_query = self._normalize_text(query)
        for query_id, payload in self.benchmark_data.get("queries", {}).items():
            if payload.get("text") == normalized_query:
                return query_id
        return self._stable_query_id(normalized_query)

    def _stable_document_id(self, row: dict[str, Any]) -> str:
        metadata = dict(row.get("metadata") or {})
        payload = {
            "source": metadata.get("source", ""),
            "page": metadata.get("page", ""),
            "page_label": metadata.get("page_label", ""),
            "passage": self._normalize_text(str(row.get("passage", ""))),
        }
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _stable_query_id(self, question: str) -> str:
        payload = {"question": self._normalize_text(question)}
        return hashlib.sha1(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
