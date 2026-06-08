from __future__ import annotations

import json
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Sequence, TypeVar

from raccoon.logging_utils import get_logger

from .BaseRetriever import BaseRetriever

if TYPE_CHECKING:
    from raccoon.custom_retriever.reranker.BaseReranker import BaseReranker

log = get_logger(__name__)

T = TypeVar("T")


class DenseRetrieverChroma(BaseRetriever):
    retriever_type = "dense_chroma"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        corpus: dict[str, dict[str, Any]] | None = None,
        queries: dict[str, str] | None = None,
        reranker: BaseReranker | None = None,
        model_id: str | None = None,
        collection_name: str | None = None,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_ssl: bool = False,
        chroma_headers: dict[str, str] | None = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
        max_length: int | None = None,
        device: str | None = None,
        query_prompt_name: str | None = None,
        passage_prompt_name: str | None = None,
        normalize_embeddings: bool = False,
        topk: int | list[int] = 20,
        batch_size: int = 128,
        upsert_batch_size: int = 5_000,
        query_batch_size: int = 1,
        show_progress_bar: bool = True,
        reset_collection: bool = False,
        chroma_client: Any | None = None,
        sentence_model: Any | None = None,
    ) -> None:
        super().__init__(config=config, corpus=corpus, queries=queries, reranker=reranker)

        if not model_id:
            raise ValueError("model_id is required for DenseRetrieverChroma")
        if not collection_name:
            raise ValueError("collection_name is required for DenseRetrieverChroma")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if upsert_batch_size <= 0:
            raise ValueError("upsert_batch_size must be greater than zero")
        if query_batch_size <= 0:
            raise ValueError("query_batch_size must be greater than zero")

        if isinstance(topk, int) and topk > 0:
            self.topk = topk
        elif isinstance(topk, list) and topk and all(isinstance(k, int) and k > 0 for k in topk):
            self.topk = max(topk)
        else:
            raise ValueError("topk must be a positive int or non-empty list of positive ints")

        self.model_id = model_id
        self.collection_name = collection_name
        self.chroma_host = chroma_host
        self.chroma_port = int(chroma_port)
        self.chroma_ssl = chroma_ssl
        self.chroma_headers = chroma_headers
        self.tenant = tenant
        self.database = database
        self.max_length = max_length
        self.device = device
        self.query_prompt_name = query_prompt_name
        self.passage_prompt_name = passage_prompt_name
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.upsert_batch_size = upsert_batch_size
        self.query_batch_size = query_batch_size
        self.show_progress_bar = show_progress_bar
        self.reset_collection = reset_collection
        self.collection: Any | None = None
        self._reset_applied = False
        self._last_collection_setup_latency = 0.0

        model_load_start = perf_counter()
        self.sentence_model = sentence_model
        self._owns_sentence_model = sentence_model is None
        self._load_model()
        self._model_load_latency = perf_counter() - model_load_start

        client_start = perf_counter()
        if chroma_client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("chromadb is required; install the 'chroma' optional dependency") from exc

            chroma_client = chromadb.HttpClient(
                host=self.chroma_host,
                port=self.chroma_port,
                ssl=self.chroma_ssl,
                headers=self.chroma_headers,
                tenant=self.tenant,
                database=self.database,
            )
        self.client = chroma_client
        self._client_initialization_latency = perf_counter() - client_start

        self.metrics["setup"] = {
            "model_loading": {
                "time_in_seconds": self._model_load_latency,
                "model": self.model_id,
                "device": self.device,
            },
            "client_initialization": {
                "time_in_seconds": self._client_initialization_latency,
                "backend": "chroma",
                "host": self.chroma_host,
                "port": self.chroma_port,
                "ssl": self.chroma_ssl,
            },
        }

    def _load_model(self) -> None:
        if self.sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required; install the 'chroma' optional dependency"
                ) from exc

            log.info("Loading Chroma dense retriever model=%s device=%s", self.model_id, self.device or "auto")
            self.sentence_model = SentenceTransformer(
                model_name_or_path=self.model_id,
                trust_remote_code=True,
                device=self.device,
            )
        if self.max_length is not None:
            self.sentence_model.max_seq_length = self.max_length
        self.max_length = getattr(self.sentence_model, "max_seq_length", self.max_length)
        self.device = str(self.device or getattr(self.sentence_model, "device", ""))

    def _unload_model(self) -> None:
        if not self._owns_sentence_model:
            return
        self.sentence_model = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _batches(values: Iterable[T], size: int) -> Iterator[list[T]]:
        batch: list[T] = []
        for value in values:
            batch.append(value)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _document_text(document: dict[str, Any]) -> str:
        return f"{document.get('title', '')}\n{document.get('text', '')}".strip()

    @staticmethod
    def _metadata(doc_id: str, document: dict[str, Any]) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {"document_id": doc_id}
        for key, value in document.items():
            if key == "text" or value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                metadata[str(key)] = value
            else:
                metadata[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        return metadata

    @staticmethod
    def _as_list(embeddings: Any) -> list[list[float]]:
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        return [[float(value) for value in embedding] for embedding in embeddings]

    def encode(self, texts: Sequence[str], prompt_name: str | None = None) -> list[list[float]]:
        self._load_model()
        encode_kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": self.show_progress_bar,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize_embeddings,
        }
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name

        embeddings = self.sentence_model.encode(list(texts), **encode_kwargs)
        encoded = self._as_list(embeddings)
        if len(encoded) != len(texts):
            raise RuntimeError("The embedding model returned a different number of embeddings than input texts")
        return encoded

    def _validate_collection(self) -> None:
        if self.collection is None:
            return

        metadata = getattr(self.collection, "metadata", None) or {}
        stored_model = metadata.get("raccoon:model_id")
        if stored_model and stored_model != self.model_id:
            raise ValueError(
                f"Collection {self.collection_name!r} was created for model {stored_model!r}, "
                f"not {self.model_id!r}"
            )

        configuration = getattr(self.collection, "configuration", None) or {}
        hnsw = configuration.get("hnsw") or {}
        space = hnsw.get("space") or metadata.get("hnsw:space")
        if space and space != "cosine":
            raise ValueError(
                f"Collection {self.collection_name!r} uses distance {space!r}; cosine is required"
            )

    def create_index(self, *args: Any, **kwargs: Any) -> None:
        if self.collection is not None:
            return

        setup_start = perf_counter()
        try:
            self.client.heartbeat()

            if self.reset_collection and not self._reset_applied:
                try:
                    self.client.delete_collection(name=self.collection_name)
                except Exception as exc:
                    message = str(exc).lower()
                    if "not found" not in message and "does not exist" not in message:
                        raise
                self._reset_applied = True

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"raccoon:model_id": self.model_id},
                configuration={"hnsw": {"space": "cosine"}},
                embedding_function=None,
            )
            self._validate_collection()
        except Exception as exc:
            self.collection = None
            raise RuntimeError(
                f"Unable to initialize Chroma collection {self.collection_name!r} at "
                f"{self.chroma_host}:{self.chroma_port}: {exc}"
            ) from exc
        finally:
            self._last_collection_setup_latency = perf_counter() - setup_start

    def _effective_upsert_batch_size(self) -> int:
        try:
            return max(1, min(self.upsert_batch_size, int(self.client.get_max_batch_size())))
        except (AttributeError, TypeError, ValueError):
            return self.upsert_batch_size

    def index_corpus(
        self,
        corpus: dict[str, dict[str, Any]] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if corpus is not None:
            self.corpus = corpus
        if not self.corpus:
            raise ValueError("No corpus available to index.")

        index_start = perf_counter()
        self.create_index()
        if self.collection is None:
            raise RuntimeError("Chroma collection was not initialized")

        document_count = len(self.corpus)
        encode_latency = 0.0
        upsert_latency = 0.0
        upsert_batches = 0
        effective_batch_size = self._effective_upsert_batch_size()

        log.info(
            "Indexing Chroma corpus: collection=%s documents=%d model=%s batch_size=%d",
            self.collection_name,
            document_count,
            self.model_id,
            effective_batch_size,
        )

        for batch in self._batches(self.corpus.items(), effective_batch_size):
            ids = [str(doc_id) for doc_id, _ in batch]
            documents = [self._document_text(document) for _, document in batch]
            metadatas = [
                self._metadata(str(doc_id), document)
                for doc_id, document in batch
            ]

            encode_start = perf_counter()
            embeddings = self.encode(documents, prompt_name=self.passage_prompt_name)
            encode_latency += perf_counter() - encode_start

            upsert_start = perf_counter()
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            upsert_latency += perf_counter() - upsert_start
            upsert_batches += 1

        collection_count = int(self.collection.count())
        index_latency = perf_counter() - index_start
        self.is_ready = True
        setup_metrics = self.metrics.pop("setup", None)
        self.metrics["index_time"] = {
            "indexing": {
                "time_in_seconds": index_latency,
                "collection_setup_time_in_seconds": self._last_collection_setup_latency,
                "encoding_time_in_seconds": encode_latency,
                "upsert_time_in_seconds": upsert_latency,
                "documents": document_count,
                "collection_documents": collection_count,
                "docs_per_second": document_count / index_latency if index_latency else 0.0,
                "backend": "chroma",
                "model": self.model_id,
                "device": self.device,
                "collection": self.collection_name,
                "upsert_batches": upsert_batches,
                "upsert_batch_size": effective_batch_size,
            },
        }
        if setup_metrics is not None:
            self.metrics["setup"] = setup_metrics
        log.info("Finished Chroma indexing in %.2fs", index_latency)

    def _prepare_for_search(self) -> int:
        if not self.is_ready:
            if self.corpus:
                self.index_corpus()
            else:
                self.create_index()
                if self.collection is None:
                    raise RuntimeError("Chroma collection was not initialized")
                if self.collection.count() == 0:
                    raise ValueError("No corpus is available and the Chroma collection is empty")
                self.is_ready = True

        if self.collection is None:
            self.create_index()
        if self.collection is None:
            raise RuntimeError("Chroma collection was not initialized")
        return int(self.collection.count())

    def search(self, *args: Any, **kwargs: Any) -> dict[str, dict[str, float]]:
        try:
            return self._search(*args, **kwargs)
        finally:
            self._unload_model()

    def _search(
        self,
        top_k: int | None = None,
        queries: dict[str, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, dict[str, float]]:
        if queries is not None:
            self.queries = queries
        if not self.queries:
            raise ValueError("No queries available for searching.")

        top_k = top_k or self.topk
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        collection_count = self._prepare_for_search()
        if self.collection is None:
            raise RuntimeError("Chroma collection was not initialized")

        search_start = perf_counter()
        query_encode_latency = 0.0
        database_latency = 0.0
        request_count = 0
        results: dict[str, dict[str, float]] = {}
        requested_results = min(top_k + 1, collection_count)

        log.info(
            "Starting Chroma search: collection=%s queries=%d top_k=%d query_batch_size=%d",
            self.collection_name,
            len(self.queries),
            top_k,
            self.query_batch_size,
        )

        for batch in self._batches(self.queries.items(), self.query_batch_size):
            query_ids = [str(query_id) for query_id, _ in batch]
            query_texts = [str(query_text) for _, query_text in batch]

            encode_start = perf_counter()
            query_embeddings = self.encode(query_texts, prompt_name=self.query_prompt_name)
            query_encode_latency += perf_counter() - encode_start

            database_start = perf_counter()
            response = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=requested_results,
                include=["distances"],
            )
            database_latency += perf_counter() - database_start
            request_count += 1

            response_ids = response.get("ids") or []
            response_distances = response.get("distances") or []
            for row_index, query_id in enumerate(query_ids):
                ids = response_ids[row_index] if row_index < len(response_ids) else []
                distances = response_distances[row_index] if row_index < len(response_distances) else []
                query_results: dict[str, float] = {}
                for doc_id, distance in zip(ids, distances):
                    doc_id = str(doc_id)
                    if doc_id == query_id or distance is None:
                        continue
                    query_results[doc_id] = 1.0 - float(distance)
                    if len(query_results) == top_k:
                        break
                results[query_id] = query_results

        search_latency = perf_counter() - search_start
        query_count = len(self.queries)
        result_counts = [len(row) for row in results.values()]
        scores = [score for row in results.values() for score in row.values()]
        total_results = sum(result_counts)

        self.results = results
        setup_metrics = self.metrics.pop("setup", None)
        self.metrics["query_time"] = {
            "search": {
                "time_in_seconds": search_latency,
                "encoding_time_in_seconds": query_encode_latency,
                "database_time_in_seconds": database_latency,
                "time_per_query_in_seconds": search_latency / query_count if query_count else 0.0,
                "database_time_per_query_in_seconds": database_latency / query_count if query_count else 0.0,
                "queries_per_second": query_count / search_latency if search_latency else 0.0,
                "queries": query_count,
                "top_k": top_k,
                "backend": "chroma",
                "model": self.model_id,
                "device": self.device,
                "collection": self.collection_name,
                "collection_documents": collection_count,
                "database_requests": request_count,
                "query_batch_size": self.query_batch_size,
                "total_results": total_results,
                "avg_results_per_query": total_results / query_count if query_count else 0.0,
                "min_results_per_query": min(result_counts) if result_counts else 0,
                "max_results_per_query": max(result_counts) if result_counts else 0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
                "score_function": "cosine",
            },
        }
        if setup_metrics is not None:
            self.metrics["setup"] = setup_metrics
        log.info(
            "Finished Chroma search in %.2fs (database round trips %.2fs)",
            search_latency,
            database_latency,
        )

        if self.reranker is not None:
            self.store_rerank_results()

        return results
