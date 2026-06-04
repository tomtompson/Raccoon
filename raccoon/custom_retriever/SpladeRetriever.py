from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch
from elastic_transport import ConnectionError as ElasticConnectionError
from elasticsearch import ApiError, Elasticsearch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from raccoon.logging_utils import get_logger
from .BaseRetriever import BaseRetriever

if TYPE_CHECKING:
    from raccoon.custom_retriever.util.Reranker import Reranker

log = get_logger(__name__)


class SpladeRetriever(BaseRetriever):
    retriever_type = "splade"

    def __init__(
        self,
        elasticsearch_url: str | None = None,
        language: str = "english",
        index_name: str | None = None,
        model_name: str = "naver/splade-cocondenser-ensembledistil",
        config: dict[str, Any] | None = None,
        corpus: dict | None = None,
        queries: dict | None = None,
        reranker: Reranker | None = None,
        content_field: str = "content",
        metadata_field: str = "metadata",
        vector_field: str = "splade",
        topk: int = 20,
        batch_size: int = 8,
        max_length: int = 256,
        max_features: int = 256,
        refresh_on_write: bool = True,
        timeout: int = 120,
        device: str | None = None,
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
        self.language = language
        self.index_name = index_name
        self.model_name = model_name
        self.content_field = content_field
        self.metadata_field = metadata_field
        self.vector_field = vector_field

        if isinstance(topk, int):
            self.topk = topk
        elif isinstance(topk, list) and topk:
            self.topk = max(topk)
        else:
            raise ValueError("topk must be an int or non-empty list of ints")

        self.batch_size = batch_size
        self.max_length = max_length
        self.max_features = max_features
        self.refresh_on_write = refresh_on_write
        self.timeout = timeout

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.client: Elasticsearch | None = None
        if self.elasticsearch_url:
            self.client = Elasticsearch(
                hosts=[self.elasticsearch_url],
                request_timeout=self.timeout,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

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
                        self.vector_field: {
                            "type": "rank_features",
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

    def encode(self, texts: list[str]) -> list[dict[str, float]]:
        encoded_vectors: list[dict[str, float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                output = self.model(**inputs)
                logits = output.logits

                sparse = torch.log1p(torch.relu(logits))
                sparse = sparse * inputs["attention_mask"].unsqueeze(-1)
                sparse = torch.max(sparse, dim=1).values

            for vector in sparse:
                nonzero = torch.nonzero(vector > 0, as_tuple=False).squeeze(-1)

                if nonzero.numel() == 0:
                    encoded_vectors.append({})
                    continue

                values = vector[nonzero]

                if values.numel() > self.max_features:
                    top_values, top_indices = torch.topk(values, self.max_features)
                    nonzero = nonzero[top_indices]
                    values = top_values

                sparse_dict = {
                    f"t_{int(token_id)}": float(weight)
                    for token_id, weight in zip(nonzero, values)
                    if float(weight) > 0.0
                }

                encoded_vectors.append(sparse_dict)

        return encoded_vectors

    def index_corpus(self, *args, **kwargs) -> None:
        if not self.client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not self.index_name:
            raise ValueError("index_name must be set before indexing.")
        if not self.corpus:
            raise ValueError("No corpus available to index.")

        index_start = perf_counter()
        log.info(
            "Indexing SPLADE corpus: index=%s documents=%d model=%s",
            self.index_name,
            len(self.corpus),
            self.model_name,
        )

        self.create_index()

        doc_items = list(self.corpus.items())
        doc_texts = [doc.get("text", "") for _, doc in doc_items]
        doc_vectors = self.encode(doc_texts)

        for (doc_id, doc), sparse_vector in zip(doc_items, doc_vectors):
            text = doc.get("text", "")
            metadata = {
                k: v for k, v in doc.items()
                if k != "text"
            }
            metadata["document_id"] = str(doc_id)

            payload = {
                self.content_field: text,
                self.metadata_field: metadata,
                self.vector_field: sparse_vector,
            }

            self.client.index(
                index=self.index_name,
                id=str(doc_id),
                document=payload,
            )

        if self.refresh_on_write:
            self.client.indices.refresh(index=self.index_name)

        self.is_ready = True
        index_latency = perf_counter() - index_start
        document_count = len(self.corpus)

        storage_size_bytes = 0
        storage_size_mb = 0.0

        try:
            stats = self.client.indices.stats(index=self.index_name)
            storage_size_bytes = (
                stats["indices"][self.index_name]["total"]["store"]["size_in_bytes"]
            )
            storage_size_mb = storage_size_bytes / (1024 * 1024)
        except Exception as exc:
            log.warning(
                "Unable to retrieve index stats for %s at %s: %s",
                self.index_name,
                self.elasticsearch_url,
                exc,
            )

        self.metrics["index_time"] = {
            "indexing": {
                "time_in_seconds": index_latency,
                "documents": document_count,
                "docs_per_second": document_count / index_latency if index_latency else 0.0,
                "backend": "elasticsearch",
                "model": self.model_name,
                "device": self.device,
                "storage_size_bytes": storage_size_bytes,
                "storage_size_mb": storage_size_mb,
                "max_features": self.max_features,
            },
        }

        log.info("Finished SPLADE indexing in %.2fs", index_latency)

    def _build_splade_query(self, query_vector: dict[str, float], top_k: int) -> dict:
        should_clauses = []

        for feature_name, weight in query_vector.items():
            should_clauses.append(
                {
                    "rank_feature": {
                        "field": f"{self.vector_field}.{feature_name}",
                        "boost": weight,
                    }
                }
            )

        if not should_clauses:
            return {
                "size": top_k,
                "_source": False,
                "query": {
                    "match_none": {}
                },
            }

        return {
            "size": top_k,
            "_source": False,
            "query": {
                "bool": {
                    "should": should_clauses,
                    "minimum_should_match": 1,
                }
            },
        }

    def search(self, top_k: int | None = None, *args, **kwargs) -> dict:
        if not self.client:
            raise ValueError("Elasticsearch client is not initialized.")
        if not self.index_name:
            raise ValueError("index_name must be set before searching.")
        if not self.queries:
            raise ValueError("No queries available for searching.")

        top_k = top_k or self.topk
        results: dict[str, dict[str, float]] = {}

        search_start = perf_counter()
        log.info(
            "Starting SPLADE search: index=%s queries=%d top_k=%s model=%s",
            self.index_name,
            len(self.queries),
            top_k,
            self.model_name,
        )

        query_ids = list(self.queries.keys())
        query_texts = [self.queries[qid] for qid in query_ids]
        query_vectors = self.encode(query_texts)

        for query_id, query_vector in zip(query_ids, query_vectors):
            payload = self._build_splade_query(query_vector, top_k)
            response = self.client.search(index=self.index_name, **payload)
            hits = response.get("hits", {}).get("hits", [])

            query_results: dict[str, float] = {}
            for hit in hits:
                doc_id = str(hit.get("_id"))
                score = float(hit.get("_score", 0.0))
                query_results[doc_id] = score

            results[str(query_id)] = query_results

        self.results = results
        search_latency = perf_counter() - search_start
        query_count = len(self.queries)

        result_counts = [len(row) for row in results.values()]
        total_results = sum(result_counts)
        scores = [score for row in results.values() for score in row.values()]

        self.metrics["query_time"] = {
            "search": {
                "time_in_seconds": search_latency,
                "time_per_query_in_seconds": search_latency / query_count if query_count else 0.0,
                "queries_per_second": query_count / search_latency if search_latency else 0.0,
                "queries": query_count,
                "top_k": top_k,
                "backend": "elasticsearch",
                "model": self.model_name,
                "device": self.device,
                "total_results": total_results,
                "avg_results_per_query": total_results / query_count if query_count else 0.0,
                "min_results_per_query": min(result_counts) if result_counts else 0,
                "max_results_per_query": max(result_counts) if result_counts else 0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "max_features": self.max_features,
            },
        }

        log.info("Finished SPLADE search in %.2fs", search_latency)

        if self.reranker is not None:
            self.store_rerank_results()

        return results

    def save_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        (path / "splade_artifacts.json").write_text(
            json.dumps(
                {
                    "elasticsearch_url": self.elasticsearch_url,
                    "language": self.language,
                    "index_name": self.index_name,
                    "model_name": self.model_name,
                    "content_field": self.content_field,
                    "metadata_field": self.metadata_field,
                    "vector_field": self.vector_field,
                    "topk": self.topk,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length,
                    "max_features": self.max_features,
                    "refresh_on_write": self.refresh_on_write,
                    "timeout": self.timeout,
                    "device": self.device,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_artifacts(self, path: str | Path) -> None:
        path = Path(path)
        artifact = json.loads(
            (path / "splade_artifacts.json").read_text(encoding="utf-8")
        )

        self.elasticsearch_url = artifact["elasticsearch_url"]
        self.language = artifact.get("language", "english")
        self.index_name = artifact["index_name"]
        self.model_name = artifact["model_name"]
        self.content_field = artifact["content_field"]
        self.metadata_field = artifact["metadata_field"]
        self.vector_field = artifact["vector_field"]
        self.topk = int(artifact.get("topk", 20))
        self.batch_size = int(artifact.get("batch_size", 8))
        self.max_length = int(artifact.get("max_length", 256))
        self.max_features = int(artifact.get("max_features", 256))
        self.refresh_on_write = bool(artifact["refresh_on_write"])
        self.timeout = int(artifact["timeout"])
        self.device = artifact.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.client = Elasticsearch(
            hosts=[self.elasticsearch_url],
            request_timeout=self.timeout,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()