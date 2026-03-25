import unittest

from raccoon.eval.retriever.RetrievelEval import RetrievelEval


class StubRetriever:
    def __init__(self, benchmark_data):
        self.benchmark_data = benchmark_data
        self.bulk_search_calls: list[int] = []
        self._bulk_search_result = benchmark_data

    def bulk_search(self, top_k: int = 5):
        self.bulk_search_calls.append(top_k)
        self.benchmark_data = self._bulk_search_result
        return self.benchmark_data


class RetrievalEvalTest(unittest.TestCase):
    def test_process_qrels_builds_binary_labels(self) -> None:
        evaluator = RetrievelEval()

        qrels = evaluator.process_qrels(
            {
                "q1:d1": {"query_id": "q1", "document_id": "d1"},
                "q1:d2": {"query_id": "q1", "document_id": "d2"},
                "q2:d3": {"query_id": "q2", "document_id": "d3"},
            }
        )

        self.assertEqual(qrels, {"q1": {"d1": 1, "d2": 1}, "q2": {"d3": 1}})

    def test_build_run_converts_hits_and_keeps_best_duplicate_score(self) -> None:
        evaluator = RetrievelEval()

        run = evaluator.build_run(
            {
                "q1": {
                    "hits": [
                        {"document_id": "d1", "score": 0.2},
                        {"document_id": "d1", "score": 0.9},
                        {"document_id": "d2", "score": 0.3},
                    ]
                }
            }
        )

        self.assertEqual(run, {"q1": {"d1": 0.9, "d2": 0.3}})

    def test_evaluate_reports_summary_and_diagnostics(self) -> None:
        benchmark_data = {
            "queries": {
                "q1": {"query_id": "q1", "text": "alpha"},
                "q2": {"query_id": "q2", "text": "beta"},
                "q3": {"query_id": "q3", "text": "gamma"},
            },
            "qrels": {
                "q1:d1": {"query_id": "q1", "document_id": "d1"},
                "q2:d2": {"query_id": "q2", "document_id": "d2"},
            },
            "results": {
                "q1": {
                    "query_id": "q1",
                    "hits": [
                        {"rank": 1, "document_id": "d1", "score": 2.0},
                        {"rank": 2, "document_id": "d9", "score": 1.0},
                    ],
                },
                "q2": {
                    "query_id": "q2",
                    "hits": [
                        {"rank": 1, "document_id": "d9", "score": 3.0},
                        {"rank": 2, "document_id": "d2", "score": 2.5},
                    ],
                },
                "q3": {
                    "query_id": "q3",
                    "hits": [{"rank": 1, "document_id": "d7", "score": 1.0}],
                },
            },
        }

        evaluator = RetrievelEval(top_k=[1, 2])
        report = evaluator.evaluate(benchmark_data)

        self.assertEqual(report["diagnostics"]["queries_total"], 3)
        self.assertEqual(report["diagnostics"]["queries_with_qrels"], 2)
        self.assertEqual(report["diagnostics"]["queries_with_results"], 3)
        self.assertEqual(report["diagnostics"]["queries_evaluated"], 2)
        self.assertEqual(report["diagnostics"]["queries_missing_qrels"], ["q3"])
        self.assertEqual(report["diagnostics"]["queries_missing_results"], [])
        self.assertEqual(report["diagnostics"]["judgment_mode"], "binary")
        self.assertEqual(report["summary"]["Accuracy@1"], 0.5)
        self.assertEqual(report["summary"]["Accuracy@2"], 1.0)
        self.assertEqual(report["summary"]["HitRate@1"], 0.5)
        self.assertEqual(report["summary"]["HitRate@2"], 1.0)
        self.assertEqual(report["summary"]["MRR@1"], 0.5)
        self.assertEqual(report["summary"]["MRR@2"], 0.75)
        self.assertEqual(report["summary"]["Recall@1"], 0.5)
        self.assertEqual(report["summary"]["Recall@2"], 1.0)
        self.assertIn("q1", report["per_query"])
        self.assertIn("q2", report["per_query"])
        self.assertNotIn("q3", report["per_query"])

    def test_evaluate_uses_retriever_and_triggers_bulk_search_when_results_missing(self) -> None:
        benchmark_data = {
            "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
            "qrels": {"q1:d1": {"query_id": "q1", "document_id": "d1"}},
            "results": {
                "q1": {
                    "query_id": "q1",
                    "hits": [{"rank": 1, "document_id": "d1", "score": 1.0}],
                }
            },
        }
        retriever = StubRetriever(benchmark_data)
        retriever.benchmark_data = {
            "queries": benchmark_data["queries"],
            "qrels": benchmark_data["qrels"],
            "results": {},
        }
        retriever._bulk_search_result = benchmark_data
        evaluator = RetrievelEval(retriever=retriever, top_k=[1, 5])

        report = evaluator.evaluate()

        self.assertEqual(retriever.bulk_search_calls, [5])
        self.assertEqual(report["summary"]["Accuracy@1"], 1.0)
        self.assertEqual(report["summary"]["HitRate@1"], 1.0)

    def test_evaluate_supports_graded_qrels(self) -> None:
        evaluator = RetrievelEval(top_k=[1, 2])
        benchmark_data = {
            "queries": {"q1": {"query_id": "q1", "text": "alpha"}},
            "qrels": {
                "q1:d1": {
                    "query_id": "q1",
                    "document_id": "d1",
                    "relevance_score": 2,
                },
                "q1:d2": {
                    "query_id": "q1",
                    "document_id": "d2",
                    "relevance_score": 1,
                },
            },
            "results": {
                "q1": {
                    "query_id": "q1",
                    "hits": [
                        {"rank": 1, "document_id": "d2", "score": 2.0},
                        {"rank": 2, "document_id": "d1", "score": 1.0},
                    ],
                }
            },
        }

        report = evaluator.evaluate(
            benchmark_data,
            relevance_mode="graded",
            graded_field="relevance_score",
        )

        self.assertEqual(report["diagnostics"]["judgment_mode"], "graded")
        self.assertEqual(report["diagnostics"]["graded_field"], "relevance_score")
        self.assertEqual(report["summary"]["Accuracy@1"], 1.0)
        self.assertEqual(report["summary"]["Recall@2"], 1.0)

    def test_backward_compatible_alias_still_exists(self) -> None:
        self.assertTrue(callable(RetrievelEval))


if __name__ == "__main__":
    unittest.main()
