import unittest
from urllib import error
from urllib import request

from raccoon.generator import BaseGenerator, OllamaGenerator
from raccoon.retriever import BM25Retriever


class StubRetriever:
    def __init__(self, benchmark_data):
        self.benchmark_data = benchmark_data
        self.bulk_search_calls: list[int] = []

    def bulk_search(self, top_k: int = 5):
        self.bulk_search_calls.append(top_k)
        queries = self.benchmark_data.get("queries", {})
        results = self.benchmark_data.setdefault("results", {})
        for query_id, payload in queries.items():
            results[query_id] = {
                "query_id": query_id,
                "query_text": payload["text"],
                "top_k": top_k,
                "returned_count": 1,
                "hits": [
                    {
                        "rank": 1,
                        "document_id": f"doc-{query_id}",
                        "score": 0.9,
                        "content": f"context for {payload['text']}",
                    }
                ],
            }
        return results


class StubGenerator(BaseGenerator):
    generator_type = "stub"

    def _load_llm(self):
        return "stub://llm"

    def call_llm(self, query: str, retrieved: list[dict[str, object]]) -> str:
        if retrieved:
            return f"{query} -> {retrieved[0]['content']}"
        return f"{query} -> no context"


class StubOllamaGenerator(OllamaGenerator):
    def __init__(self, *args, response_body='{"response":"final answer"}', **kwargs):
        self._response_body = response_body
        super().__init__(*args, **kwargs)

    def call_llm(self, query: str, retrieved: list[dict[str, object]]) -> str:
        return super().call_llm(query=query, retrieved=retrieved)

    def _urlopen(self, req):
        raise NotImplementedError


class BaseGeneratorTest(unittest.TestCase):
    def test_generate_uses_existing_retrieval_results(self) -> None:
        retriever = StubRetriever(
            {
                "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
                "results": {
                    "q1": {
                        "query_id": "q1",
                        "hits": [
                            {
                                "rank": 1,
                                "document_id": "doc-1",
                                "score": 0.95,
                                "content": "alpha context",
                            }
                        ],
                    }
                },
            }
        )

        generator = StubGenerator(retriever=retriever)
        results = generator.generate()

        self.assertEqual(retriever.bulk_search_calls, [])
        self.assertEqual(results[0]["query_id"], "q1")
        self.assertEqual(results[0]["answer"], "alpha -> alpha context")
        self.assertEqual(generator.metrics["generator_type"], "stub")
        self.assertEqual(generator.metrics["generated"], 1)

    def test_generate_calls_bulk_search_when_results_are_empty(self) -> None:
        retriever = StubRetriever(
            {
                "queries": {
                    "q1": {"query_id": "q1", "text": "alpha"},
                    "q2": {"query_id": "q2", "text": "beta"},
                },
                "results": {},
            }
        )

        generator = StubGenerator(retriever=retriever, top_k=3)
        results = generator.generate()

        self.assertEqual(retriever.bulk_search_calls, [3])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[1]["query_id"], "q2")
        self.assertEqual(generator.metrics["queries_total"], 2)
        self.assertEqual(generator.metrics["llm_call_count"], 2)

    def test_generate_allows_queries_with_no_hits(self) -> None:
        retriever = StubRetriever(
            {
                "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
                "results": {"q1": {"query_id": "q1", "hits": []}},
            }
        )

        generator = StubGenerator(retriever=retriever)
        results = generator.generate()

        self.assertEqual(results[0]["answer"], "alpha -> no context")

    def test_generate_uses_explicit_search_results_without_searching_again(self) -> None:
        retriever = StubRetriever(
            {
                "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
                "results": {},
            }
        )
        precomputed_results = {
            "q1": {
                "query_id": "q1",
                "hits": [
                    {
                        "rank": 1,
                        "document_id": "doc-1",
                        "score": 0.95,
                        "content": "alpha from precomputed results",
                    }
                ],
            }
        }

        generator = StubGenerator(retriever=retriever)
        results = generator.generate(search_results=precomputed_results)

        self.assertEqual(retriever.bulk_search_calls, [])
        self.assertEqual(results[0]["answer"], "alpha -> alpha from precomputed results")


class OllamaGeneratorTest(unittest.TestCase):
    def test_call_llm_returns_trimmed_response_text(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b'{"response":" final answer  "}'

        original_urlopen = request.urlopen
        try:
            request.urlopen = lambda req, timeout=0: Response()
            generator = OllamaGenerator(
                retriever=StubRetriever({"queries": {}, "results": {}}),
                ollama_url="http://localhost:11434",
                model_id="llama3",
            )
            answer = generator.call_llm(
                query="what is alpha",
                retrieved=[{"rank": 1, "score": 1.0, "document_id": "doc-1", "content": "alpha"}],
            )
        finally:
            request.urlopen = original_urlopen

        self.assertEqual(answer, "final answer")

    def test_call_llm_rejects_non_json(self) -> None:
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return b"not-json"

        original_urlopen = request.urlopen
        try:
            request.urlopen = lambda req, timeout=0: Response()
            generator = OllamaGenerator(
                retriever=StubRetriever({"queries": {}, "results": {}}),
                ollama_url="http://localhost:11434",
                model_id="llama3",
            )
            with self.assertRaises(RuntimeError):
                generator.call_llm(query="alpha", retrieved=[])
        finally:
            request.urlopen = original_urlopen

    def test_call_llm_wraps_url_errors(self) -> None:
        original_urlopen = request.urlopen
        try:
            request.urlopen = lambda req, timeout=0: (_ for _ in ()).throw(
                error.URLError("connection refused")
            )
            generator = OllamaGenerator(
                retriever=StubRetriever({"queries": {}, "results": {}}),
                ollama_url="http://localhost:11434",
                model_id="llama3",
            )
            with self.assertRaises(RuntimeError):
                generator.call_llm(query="alpha", retrieved=[])
        finally:
            request.urlopen = original_urlopen


class GeneratorIntegrationRegressionTest(unittest.TestCase):
    def test_generator_uses_hit_content_saved_in_benchmark_results(self) -> None:
        retriever = StubRetriever(
            {
                "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
                "results": {
                    "q1": {
                        "query_id": "q1",
                        "hits": [
                            {
                                "rank": 1,
                                "document_id": "doc-1",
                                "score": 0.95,
                                "content": "alpha context",
                                "metadata": {"source": "unit"},
                            }
                        ],
                    }
                },
            }
        )

        generator = StubGenerator(retriever=retriever)
        results = generator.generate()

        self.assertEqual(results[0]["answer"], "alpha -> alpha context")

    def test_bm25_search_persists_full_hits_in_benchmark_results(self) -> None:
        retriever = BM25Retriever(
            elasticsearch_url="http://localhost:9200",
            index_name="unit-test-index",
        )
        retriever.is_ready = True
        retriever.processed_documents = [object()]
        retriever.benchmark_data = {
            "documents": {},
            "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
            "qrels": {},
            "results": {},
        }
        retriever._search = lambda query, top_k: [
            {
                "rank": 1,
                "score": 0.9,
                "document_id": "doc-1",
                "content": "alpha context",
                "metadata": {"source": "unit"},
            }
        ]

        retriever.search("alpha", top_k=1)

        stored_hit = retriever.benchmark_data["results"]["q1"]["hits"][0]
        self.assertEqual(stored_hit["content"], "alpha context")
        self.assertEqual(stored_hit["metadata"], {"source": "unit"})


if __name__ == "__main__":
    unittest.main()
