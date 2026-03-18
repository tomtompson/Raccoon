import tempfile
import unittest
from pathlib import Path

import torch
from langchain_core.documents import Document

from crimsonvector.retriever import (
    BM25Retriever,
    LocalEmbeddingRetriever,
    OllamaEmbeddingRetriever,
)


class StubLocalEmbeddingRetriever(LocalEmbeddingRetriever):
    def _load_encoder(self):
        return object()

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        mapping = {
            "alpha install guide": [1.0, 0.0],
            "beta troubleshooting notes": [0.0, 1.0],
            "alpha query": [1.0, 0.0],
            "beta query": [0.0, 1.0],
        }
        vectors = [mapping[text] for text in texts]
        return torch.tensor(vectors, dtype=torch.float32)


class StubOllamaEmbeddingRetriever(OllamaEmbeddingRetriever):
    def _embed_text(self, text: str) -> list[float]:
        mapping = {
            "deployment runbook": [1.0, 0.0],
            "incident playbook": [0.0, 1.0],
            "deploy query": [1.0, 0.0],
        }
        return mapping[text]


class StubBM25Retriever(BM25Retriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indexed_payloads = {}
        self.client = StubElasticsearchClient(self)


class StubIndicesClient:
    def __init__(self, retriever):
        self.retriever = retriever

    def create(self, index, mappings):
        return {"acknowledged": True, "index": index, "mappings": mappings}

    def refresh(self, index):
        return {"_shards": {"successful": 1}, "index": index}


class StubElasticsearchClient:
    def __init__(self, retriever):
        self.retriever = retriever
        self.indices = StubIndicesClient(retriever)

    def index(self, index, id, document):
        self.retriever.indexed_payloads[id] = document
        return {"result": "created", "_index": index, "_id": id}

    def search(self, index, **payload):
        query = payload["query"]["match"][self.retriever.content_field]["query"]
        hits = []
        for document_id, source in self.retriever.indexed_payloads.items():
            content = source[self.retriever.content_field]
            score = float(sum(token in content for token in query.split()))
            if score <= 0:
                continue
            hits.append(
                {
                    "_id": document_id,
                    "_score": score,
                    "_source": source,
                }
            )
        hits.sort(key=lambda item: item["_score"], reverse=True)
        return {"hits": {"hits": hits[: payload["size"]]}}


class RetrieverTest(unittest.TestCase):
    def test_bm25_search_returns_ranked_hits_and_tracks_history(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
            documents=[
                Document(
                    page_content="alpha release installation procedure",
                    metadata={"source": "alpha.txt"},
                ),
                Document(
                    page_content="beta troubleshooting and rollback guide",
                    metadata={"source": "beta.txt"},
                ),
            ]
        )

        retriever.process_documents()
        result = retriever.search("installation alpha", top_k=2)

        self.assertEqual(result["hits"][0]["metadata"]["source"], "alpha.txt")
        self.assertEqual(result["returned_count"], 1)
        self.assertGreater(result["hits"][0]["score"], 0.0)
        self.assertEqual(retriever.metrics["queries_total"], 1)
        self.assertEqual(len(retriever.query_history), 1)
        self.assertEqual(retriever.metrics["elasticsearch_index_name"], "unit-test-index")
        self.assertEqual(
            retriever.query_history[0]["top_k_document_ids"][0],
            result["hits"][0]["document_id"],
        )

    def test_preprocess_assigns_document_id(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
            documents=[Document(page_content="alpha", metadata={"source": "unit"})]
        )

        processed = retriever.preprocess_documents()

        self.assertEqual(len(processed), 1)
        self.assertIn("document_id", processed[0].metadata)
        self.assertEqual(processed[0].metadata["source"], "unit")

    def test_local_embedding_search_returns_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            retriever = StubLocalEmbeddingRetriever(
                documents=[
                    Document(page_content="alpha install guide", metadata={"source": "a"}),
                    Document(page_content="beta troubleshooting notes", metadata={"source": "b"}),
                ],
                model_name="stub-model",
                query_cache_dir=tmp_dir,
            )

            retriever.process_documents()
            result = retriever.search("alpha query", top_k=2)

            self.assertEqual(result["retriever_type"], "local_embedding")
            self.assertEqual(result["hits"][0]["metadata"]["source"], "a")
            self.assertAlmostEqual(result["hits"][0]["score"], 1.0, places=5)
            self.assertEqual(retriever.metrics["embedding_dimension"], 2)
            self.assertEqual(retriever.metrics["faiss_index_type"], "IndexFlatIP")
            self.assertEqual(len(list(Path(tmp_dir).glob("*.json"))), 1)

    def test_ollama_embedding_search_uses_local_similarity(self) -> None:
        retriever = StubOllamaEmbeddingRetriever(
            ollama_url="http://localhost:11434",
            model_name="stub-ollama",
            documents=[
                Document(page_content="deployment runbook", metadata={"source": "deploy"}),
                Document(page_content="incident playbook", metadata={"source": "incident"}),
            ],
        )

        retriever.process_documents()
        result = retriever.search("deploy query", top_k=1)

        self.assertEqual(result["hits"][0]["metadata"]["source"], "deploy")
        self.assertEqual(retriever.metrics["retriever_type"], "ollama_embedding")

    def test_batch_search_records_each_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            retriever = StubLocalEmbeddingRetriever(
                documents=[
                    Document(page_content="alpha install guide", metadata={"source": "a"}),
                    Document(page_content="beta troubleshooting notes", metadata={"source": "b"}),
                ],
                model_name="stub-model",
                query_cache_dir=tmp_dir,
            )
            retriever.process_documents()

            results = retriever.batch_search(["alpha query", "beta query"], top_k=1)

            self.assertEqual(len(results), 2)
            self.assertEqual(len(retriever.query_history), 2)
            self.assertEqual(retriever.query_history[1]["query"], "beta query")
            self.assertEqual(len(list(Path(tmp_dir).glob("*.json"))), 2)

    def test_save_and_load_round_trip_preserves_searchability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "query-cache"
            retriever = StubLocalEmbeddingRetriever(
                documents=[
                    Document(page_content="alpha install guide", metadata={"source": "a"}),
                    Document(page_content="beta troubleshooting notes", metadata={"source": "b"}),
                ],
                model_name="stub-model",
                query_cache_dir=cache_dir,
            )
            retriever.process_documents()
            retriever.search("alpha query", top_k=1)

            retriever.save(tmp_dir)
            loaded = StubLocalEmbeddingRetriever(model_name="ignored")
            loaded.load(tmp_dir)
            result = loaded.search("alpha query", top_k=1)

        self.assertTrue(loaded.is_ready)
        self.assertEqual(result["hits"][0]["metadata"]["source"], "a")
        self.assertEqual(loaded.query_history[0]["query"], "alpha query")
        self.assertEqual(loaded.query_history[1]["query"], "alpha query")
        self.assertIsNotNone(loaded.query_cache_dir)

    def test_search_before_processing_raises(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
            documents=[Document(page_content="alpha", metadata={})]
        )

        with self.assertRaises(RuntimeError):
            retriever.search("alpha", top_k=1)

    def test_top_k_larger_than_corpus_is_safe(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
            documents=[Document(page_content="alpha beta", metadata={})]
        )
        retriever.process_documents()

        result = retriever.search("alpha", top_k=5)

        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(len(result["hits"]), 1)


if __name__ == "__main__":
    unittest.main()
