import tempfile
import unittest
from pathlib import Path

from raccoon.retriever import BM25Retriever


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
        )

        retriever.process_documents(
            [
                {
                    "passage": "alpha release installation procedure",
                    "metadata": {"source": "alpha.txt"},
                },
                {
                    "passage": "beta troubleshooting and rollback guide",
                    "metadata": {"source": "beta.txt"},
                },
            ]
        )
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

    def test_process_documents_assigns_document_id(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )

        processed = retriever.process_documents(
            [{"passage": "alpha", "metadata": {"source": "unit"}}]
        )

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["metadata"]["source"], "unit")
        self.assertIn("document_id", processed[0]["metadata"])
        self.assertEqual(processed[0]["document_id"], processed[0]["metadata"]["document_id"])

    def test_preprocess_builds_shared_benchmark_payload(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )

        benchmark = retriever.preprocess(
            [
                {
                    "question": "What is alpha?",
                    "answer": "alpha",
                    "passage": "alpha is the first item",
                    "metadata": {"source": "unit"},
                    "groundedness_score": 5,
                    "relevance_score": 4,
                    "standalone_score": 5,
                    "total_score": 14,
                }
            ]
        )

        self.assertEqual(len(benchmark["documents"]), 1)
        self.assertEqual(len(benchmark["queries"]), 1)
        self.assertEqual(len(benchmark["qrels"]), 1)
        self.assertEqual(benchmark["results"], {})

    def test_process_documents_preserves_explicit_ids(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )

        retriever.process_documents(
            [
                {
                    "passage": "alpha",
                    "question": "what is alpha",
                    "query_id": "query-1",
                    "metadata": {"source": "unit", "document_id": "doc-1"},
                }
            ]
        )

        query_payload = next(iter(retriever.benchmark_data["queries"].values()))
        document_payload = next(iter(retriever.benchmark_data["documents"].values()))
        self.assertEqual(query_payload["query_id"], "query-1")
        self.assertEqual(document_payload["document_id"], "doc-1")

    def test_save_and_load_round_trip_preserves_searchability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            retriever = StubBM25Retriever(
                elasticsearch_url="http://localhost:9200",
                index_name="unit-test-index",
            )
            retriever.process_documents(
                [
                    {
                        "passage": "alpha install guide",
                        "metadata": {"source": "a"},
                    },
                    {
                        "passage": "beta troubleshooting notes",
                        "metadata": {"source": "b"},
                    },
                ]
            )
            retriever.search("alpha", top_k=1)

            retriever.save(tmp_dir)
            loaded = StubBM25Retriever(
                elasticsearch_url="http://localhost:9200",
                index_name="ignored",
            )
            loaded.load(tmp_dir)
            loaded.client = StubElasticsearchClient(loaded)
            loaded._build_index(loaded.processed_documents)
            result = loaded.search("alpha", top_k=1)

        self.assertTrue(loaded.is_ready)
        self.assertEqual(result["hits"][0]["metadata"]["source"], "a")
        self.assertEqual(loaded.query_history[0]["query"], "alpha")
        self.assertEqual(loaded.query_history[1]["query"], "alpha")

    def test_search_before_processing_raises(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )

        with self.assertRaises(RuntimeError):
            retriever.search("alpha", top_k=1)

    def test_top_k_larger_than_corpus_is_safe(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )
        retriever.process_documents([{"passage": "alpha beta", "metadata": {}}])

        result = retriever.search("alpha", top_k=5)

        self.assertEqual(result["returned_count"], 1)
        self.assertEqual(len(result["hits"]), 1)

    def test_bm25_processes_critiquer_rows_without_breaking_generator_flow(self) -> None:
        retriever = StubBM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )
        rows = [
            {
                "question": "What chunk overlap is configured?",
                "answer": "The chunk overlap is 200.",
                "passage": "The loader uses a chunk overlap of 200.",
                "metadata": {"source": "unit-test"},
                "groundedness_score": 5,
                "relevance_score": 4,
                "standalone_score": 5,
                "total_score": 14,
            }
        ]

        processed = retriever.process_documents(rows)
        result = retriever.search("chunk overlap", top_k=1)
        benchmark = retriever.benchmark_data

        self.assertEqual(len(processed), 1)
        self.assertEqual(processed[0]["text"], "The loader uses a chunk overlap of 200.")
        self.assertEqual(len(benchmark["queries"]), 1)
        query_payload = next(iter(benchmark["queries"].values()))
        self.assertEqual(query_payload["text"], "What chunk overlap is configured?")
        self.assertEqual(len(benchmark["qrels"]), 1)
        self.assertEqual(result["hits"][0]["content"], "The loader uses a chunk overlap of 200.")
        stored_hit = next(iter(benchmark["results"].values()))["hits"][0]
        self.assertEqual(stored_hit["content"], "The loader uses a chunk overlap of 200.")
        self.assertEqual(stored_hit["metadata"]["source"], "unit-test")


if __name__ == "__main__":
    unittest.main()
