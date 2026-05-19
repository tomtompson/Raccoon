from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from elastic_transport import ConnectionError as ElasticConnectionError
from elasticsearch import ApiError, Elasticsearch

from .BaseRetriever import BaseRetriever
from raccoon.custom_retriever.util.Reranker import Reranker
from raccoon.custom_retriever.util.utils import pretty_print_dict


class BM25Retriever(BaseRetriever):
    retriever_type = "bm25"

    def __init__(
        self,
        elasticsearch_url: str | None = None,
        index_name: str | None = None,
        language: str = "english",
        config: dict[str, Any] | None = None,
        corpus: dict | None = None,
        queries: dict | None = None,
        reranker: Reranker | None = None,
        content_field: str = "content",
        metadata_field: str = "metadata",
        topk: int = 20,
        refresh_on_write: bool = True,
        timeout: int = 120,
    ) -> None:
        super().__init__(
            config=config,
            corpus=corpus,
            queries=queries,
            reranker=reranker,
        )

        self.elasticsearch_url = (
            elasticsearch_url.rstrip("/") if elasticsearch_url else None
        )
        self.index_name = index_name
        self.language = language
        self.content_field = content_field
        self.metadata_field = metadata_field
        if isinstance(topk, int):
            self.topk = topk
        elif isinstance(topk, list) and topk:
            self.topk = max(topk)
        else:
            raise ValueError("topk must be an int or non-empty list of ints")
        self.refresh_on_write = refresh_on_write
        self.timeout = timeout
        self.reranker = reranker

        self.client: Elasticsearch | None = None
        if self.elasticsearch_url:
            self.client = Elasticsearch(
                hosts=[self.elasticsearch_url],
                request_timeout=self.timeout,
            )

    def create_index(self, *args, **kwargs) -> None:
        if not self.client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not self.index_name:
            raise ValueError("index_name must be set before creating the index.")

        try:
            self.client.indices.create(
                index=self.index_name,
                mappings={
                    "properties": {
                        self.content_field: {
                            "type": "text",
                            "analyzer": self.language,
                        },
                        self.metadata_field: {
                            "type": "object",
                            "enabled": True,
                        },
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

    def index_corpus(self, *args, **kwargs) -> None:
        if not self.client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not self.index_name:
            raise ValueError("index_name must be set before indexing.")
        if not self.corpus:
            raise ValueError("No corpus available to index.")

        self.create_index()
        for doc_id, doc in self.corpus.items():
            text = doc.get("text", "")
            metadata = {
                k: v for k, v in doc.items()
                if k != "text"
            }
            metadata["document_id"] = str(doc_id)

            payload = {
                self.content_field: text,
                self.metadata_field: metadata,
            }

            self.client.index(
                index=self.index_name,
                id=str(doc_id),
                document=payload,
            )

        if self.refresh_on_write:
            self.client.indices.refresh(index=self.index_name)

        self.is_ready = True

    def encode(self, *args, **kwargs):
        raise NotImplementedError("Lexical BM25 retriever does not support encode().")

    def search(self, top_k: int | None = None, *args, **kwargs) -> dict:
        if not self.client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not self.index_name:
            raise ValueError("index_name must be set before searching.")
        if not self.queries:
            raise ValueError("No queries available for searching.")

        top_k = top_k or self.topk
        results: dict[str, dict[str, float]] = {}

        for query_id, query_text in self.queries.items():
            payload = {
                "size": top_k,
                "_source": False,
                "query": {
                    "match": {
                        self.content_field: {
                            "query": query_text,
                        }
                    }
                },
            }

            response = self.client.search(index=self.index_name, **payload)
            hits = response.get("hits", {}).get("hits", [])

            query_results: dict[str, float] = {}
            for hit in hits:
                doc_id = str(hit.get("_id"))
                score = float(hit.get("_score", 0.0))
                query_results[doc_id] = score

            results[str(query_id)] = query_results

        self.results = results

        stats = self.client.indices.stats(index=self.index_name)

        self.metrics = {
            "index_time": {
                "docs": stats["indices"][self.index_name]["primaries"]["docs"],
                "index": stats["indices"][self.index_name]["primaries"]["indexing"],
            },
            "query_time": {
                "search": stats["indices"][self.index_name]["primaries"]["search"],
            }
        }


        if self.reranker is not None:
            self.reranker.rerank_with_transformers(self.corpus, self.queries, self.results)

        return results

    def save_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

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
                    "topk": self.topk,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        artifact = json.loads((path / "bm25_artifacts.json").read_text(encoding="utf-8"))

        self.elasticsearch_url = artifact["elasticsearch_url"]
        self.index_name = artifact["index_name"]
        self.language = artifact.get("language", "english")
        self.content_field = artifact["content_field"]
        self.metadata_field = artifact["metadata_field"]
        self.refresh_on_write = bool(artifact["refresh_on_write"])
        self.timeout = int(artifact["timeout"])
        self.topk = int(artifact.get("topk", 20))

        self.client = Elasticsearch(
            hosts=[self.elasticsearch_url],
            request_timeout=self.timeout,
        )