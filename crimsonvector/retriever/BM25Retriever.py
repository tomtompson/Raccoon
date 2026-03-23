from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elastic_transport import ConnectionError as ElasticConnectionError
from elasticsearch import ApiError, Elasticsearch

from .BaseRetriever import BaseRetriever, DocumentRecord


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"

    def __init__(
        self,
        elasticsearch_url: str,
        index_name: str,
        language: str = "english",
        config: dict[str, Any] | None = None,
        content_field: str = "content",
        metadata_field: str = "metadata",
        topk:int = 20,
        refresh_on_write: bool = True,
        timeout: int = 120,
    ) -> None:
        super().__init__(config=config)
        self.topk = topk
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

    def _prepare_retrieval_state(self, documents: list[DocumentRecord]) -> None:
        """Create the BM25 index and write every processed document into it."""
        self._create_index_if_missing()
        for document in documents:
            payload = {
                self.content_field: document["text"],
                self.metadata_field: dict(document["metadata"]),
            }
            self.client.index(
                index=self.index_name,
                id=str(document["document_id"]),
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
            "source": False,
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
            document_id = str(metadata.get("document_id") or hit.get("_id", ""))
            metadata["document_id"] = document_id
            ranked_hits.append(
                self._build_hit(
                    document_id=document_id,
                    score=float(hit.get("_score", 0.0)),
                    rank=rank,
                )
            )

        return ranked_hits

    def _create_index_if_missing(self) -> None:
        """Create the BM25 index once and ignore the already-exists case."""
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
        """Persist BM25-specific connection and index settings."""
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
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_artifacts(self, path: Path) -> None:
        """Restore BM25-specific settings and recreate the Elasticsearch client."""
        artifact = json.loads((path / "bm25_artifacts.json").read_text(encoding="utf-8"))
        self.elasticsearch_url = artifact["elasticsearch_url"]
        self.index_name = artifact["index_name"]
        self.language = artifact.get("language", "english")
        self.content_field = artifact["content_field"]
        self.metadata_field = artifact["metadata_field"]
        self.refresh_on_write = bool(artifact["refresh_on_write"])
        self.timeout = int(artifact["timeout"])
        self.client = Elasticsearch(
            hosts=[self.elasticsearch_url],
            request_timeout=self.timeout,
        )
