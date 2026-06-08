from __future__ import annotations

from typing import Any

import pytest

from raccoon.custom_retriever.DenseRetrieverChroma import DenseRetrieverChroma
from raccoon.report.StaticRetrieverReport import StaticRetrieverReport


class FakeSentenceModel:
    max_seq_length = 512
    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def encode(self, texts: list[str], **kwargs: Any) -> list[list[float]]:
        self.calls.append((texts, kwargs))
        return [[float(len(text)), 1.0] for text in texts]


class FakeCollection:
    metadata = {"raccoon:model_id": "fake-model"}
    configuration = {"hnsw": {"space": "cosine"}}

    def __init__(self) -> None:
        self.records: dict[str, tuple[list[float], str, dict[str, Any]]] = {}
        self.upsert_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls.append(kwargs)
        for doc_id, embedding, document, metadata in zip(
            kwargs["ids"],
            kwargs["embeddings"],
            kwargs["documents"],
            kwargs["metadatas"],
        ):
            self.records[doc_id] = (embedding, document, metadata)

    def count(self) -> int:
        return len(self.records)

    def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
        self.query_calls.append(kwargs)
        response_ids: list[list[str]] = []
        response_distances: list[list[float]] = []
        for embedding in kwargs["query_embeddings"]:
            if embedding[0] == 2.0:
                response_ids.append(["q1", "d1", "d2"])
                response_distances.append([0.0, 0.2, 0.4])
            else:
                response_ids.append(["d2", "d1", "q1"])
                response_distances.append([0.1, 0.3, 0.8])
        return {"ids": response_ids, "distances": response_distances}


class FakeChromaClient:
    def __init__(self, collection: FakeCollection | None = None) -> None:
        self.collection = collection or FakeCollection()
        self.heartbeats = 0
        self.get_or_create_calls: list[dict[str, Any]] = []

    def heartbeat(self) -> int:
        self.heartbeats += 1
        return 1

    def get_or_create_collection(self, **kwargs: Any) -> FakeCollection:
        self.get_or_create_calls.append(kwargs)
        return self.collection

    def get_max_batch_size(self) -> int:
        return 2


def build_retriever(
    client: FakeChromaClient | None = None,
    model: FakeSentenceModel | None = None,
) -> DenseRetrieverChroma:
    return DenseRetrieverChroma(
        corpus={
            "d1": {"title": "One", "text": "First", "tags": ["a", "b"]},
            "d2": {"title": "Two", "text": "Second"},
            "q1": {"title": "Query document", "text": "Self match"},
        },
        queries={"q1": "aa", "q2": "bbbb"},
        model_id="fake-model",
        collection_name="test-collection",
        topk=2,
        upsert_batch_size=10,
        query_batch_size=1,
        show_progress_bar=False,
        chroma_client=client or FakeChromaClient(),
        sentence_model=model or FakeSentenceModel(),
    )


def test_index_corpus_creates_collection_and_upserts_in_bounded_batches() -> None:
    client = FakeChromaClient()
    model = FakeSentenceModel()
    retriever = build_retriever(client=client, model=model)

    retriever.index_corpus()

    assert client.heartbeats == 1
    assert client.get_or_create_calls[0]["configuration"] == {"hnsw": {"space": "cosine"}}
    assert [len(call["ids"]) for call in client.collection.upsert_calls] == [2, 1]
    assert client.collection.records["d1"][2]["tags"] == '["a", "b"]'
    assert retriever.is_ready is True
    assert retriever.metrics["index_time"]["indexing"]["documents"] == 3
    assert retriever.metrics["index_time"]["indexing"]["upsert_batches"] == 2
    assert all(len(texts) <= 2 for texts, _ in model.calls)


def test_index_corpus_is_idempotent_by_document_id() -> None:
    client = FakeChromaClient()
    retriever = build_retriever(client=client)

    retriever.index_corpus()
    retriever.index_corpus()

    assert client.collection.count() == 3
    assert len(client.collection.upsert_calls) == 4


def test_search_auto_indexes_and_measures_remote_requests() -> None:
    client = FakeChromaClient()
    retriever = build_retriever(client=client)

    results = retriever.search()

    assert results == {
        "q1": {"d1": pytest.approx(0.8), "d2": pytest.approx(0.6)},
        "q2": {"d2": pytest.approx(0.9), "d1": pytest.approx(0.7)},
    }
    assert len(client.collection.upsert_calls) == 2
    assert len(client.collection.query_calls) == 2
    search_metrics = retriever.metrics["query_time"]["search"]
    assert search_metrics["backend"] == "chroma"
    assert search_metrics["database_requests"] == 2
    assert search_metrics["database_time_in_seconds"] >= 0.0
    assert search_metrics["encoding_time_in_seconds"] >= 0.0


def test_create_index_wraps_connection_failures() -> None:
    class BrokenClient(FakeChromaClient):
        def heartbeat(self) -> int:
            raise OSError("connection refused")

    retriever = build_retriever(client=BrokenClient())

    with pytest.raises(RuntimeError, match="Unable to initialize Chroma collection"):
        retriever.create_index()


def test_static_report_accepts_chroma_metrics(tmp_path) -> None:
    retriever = build_retriever()
    retriever.search()
    retriever.add_retrieval_result(({"NDCG@10": 0.7}, {"Recall@10": 0.8}))

    report = StaticRetrieverReport()
    practical = report._practical_limit_rows([retriever], {})[0]
    config = dict(report._config_rows(retriever, {"config_rows": 50}))
    visible_metrics = dict(report._flatten(retriever.metrics)[:20])

    assert practical["index_time"] is not None
    assert practical["query_time"] is not None
    assert practical["qps"] is not None
    assert practical["storage_mb"] is None
    assert config["collection_name"] == "test-collection"
    assert config["chroma_host"] == "localhost"
    assert "query_time.search.queries_per_second" in visible_metrics

    path = report.generate_report(
        "Chroma Retriever Report",
        [retriever],
        output_path=tmp_path / "chroma-report.pdf",
        config={"comparison_metric": ["NDCG@10", "Recall@10"]},
    )

    assert path.exists()
    assert path.stat().st_size > 0
