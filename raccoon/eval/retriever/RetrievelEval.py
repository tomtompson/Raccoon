from __future__ import annotations

from typing import Any

import pytrec_eval

from raccoon.retriever.BaseRetriever import BaseRetriever
import collections

from pathlib import Path
import json

class RetrievelEval:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        retriever: BaseRetriever | None = None,
        top_k: list[int] | None = None,
    ) -> None:
        self.config = config or {}
        self.top_k = sorted(set(top_k or [1, 3, 5, 10, 20]))
        self.retriever = retriever

    def get_results_retriever(self) -> dict[str, Any]:
        if self.retriever is None:
            raise ValueError("A retriever instance is required to load benchmark results.")

        benchmark_data = getattr(self.retriever, "benchmark_data", {}) or {}
        stored_results = benchmark_data.get("results", {})
        if not stored_results:
            self.retriever.bulk_search(top_k=max(self.top_k))
            benchmark_data = getattr(self.retriever, "benchmark_data", {}) or {}
        return benchmark_data

    def process_qrels(
        self,
        raw_qrels: dict[str, dict[str, Any]],
        relevance_mode: str = "binary",
        graded_field: str | None = None,
    ) -> dict[str, dict[str, int]]:
        if relevance_mode not in {"binary", "graded"}:
            raise ValueError("relevance_mode must be 'binary' or 'graded'.")

        qrels: dict[str, dict[str, int]] = {}
        field_name = graded_field or "relevance_score"

        for qrel_body in raw_qrels.values():
            query_id = str(qrel_body.get("query_id", "")).strip()
            document_id = str(qrel_body.get("document_id", "")).strip()
            if not query_id or not document_id:
                continue

            if relevance_mode == "binary":
                score = 1
            else:
                raw_score = qrel_body.get(field_name)
                if raw_score is None:
                    continue
                try:
                    score = int(raw_score)
                except (TypeError, ValueError):
                    continue

            qrels.setdefault(query_id, {})[document_id] = score

        return qrels

    def build_run(
        self,
        raw_results: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        run: dict[str, dict[str, float]] = {}

        for query_id, result_body in raw_results.items():
            query_run: dict[str, float] = {}
            for hit in result_body.get("hits", []):
                document_id = str(hit.get("document_id", "")).strip()
                if not document_id:
                    continue
                try:
                    score = float(hit.get("score", 0.0))
                except (TypeError, ValueError):
                    continue
                if document_id not in query_run or score > query_run[document_id]:
                    query_run[document_id] = score
            if query_run:
                run[str(query_id)] = query_run

        return run

    def evaluate(
        self,
        raw_results: dict[str, Any] | None = None,
        relevance_mode: str = "binary",
        graded_field: str | None = None,
    ) -> dict[str, Any]:
        benchmark_data = raw_results or self.get_results_retriever()

        raw_qrels = benchmark_data.get("qrels", {})
        raw_run = benchmark_data.get("results", {})
        qrels = self.process_qrels(
            raw_qrels=raw_qrels,
            relevance_mode=relevance_mode,
            graded_field=graded_field,
        )
        run = self.build_run(raw_run)

        queries_with_qrels = set(qrels)
        queries_with_results = set(run)
        evaluated_queries = sorted(queries_with_qrels & queries_with_results)

        diagnostics = {
            "judgment_mode": relevance_mode,
            "graded_field": graded_field if relevance_mode == "graded" else None,
            "queries_total": len(benchmark_data.get("queries", {})),
            "queries_with_qrels": len(queries_with_qrels),
            "queries_with_results": len(queries_with_results),
            "queries_evaluated": len(evaluated_queries),
            "queries_missing_results": sorted(queries_with_qrels - queries_with_results),
            "queries_missing_qrels": sorted(queries_with_results - queries_with_qrels),
            "avg_returned_hits": round(
                sum(len(result.get("hits", [])) for result in raw_run.values()) / len(raw_run),
                5,
            )
            if raw_run
            else 0.0,
            "relevant_docs_per_query_avg": round(
                sum(len(query_qrels) for query_qrels in qrels.values()) / len(qrels),
                5,
            )
            if qrels
            else 0.0,
        }

        summary = self._empty_summary()
        if not evaluated_queries:
            return {
                "summary": summary,
                "diagnostics": diagnostics,
                "per_query": {},
            }

        filtered_qrels = {query_id: qrels[query_id] for query_id in evaluated_queries}
        filtered_run = {query_id: run[query_id] for query_id in evaluated_queries}

        measure_strings = self._build_measure_strings()
        evaluator = pytrec_eval.RelevanceEvaluator(filtered_qrels, measure_strings)
        per_query = evaluator.evaluate(filtered_run)
        summary = self._summarize_scores(per_query)
        summary.update(self._compute_accuracy(filtered_qrels, raw_run, evaluated_queries))
        summary.update(self._compute_hit_rate(filtered_qrels, raw_run, evaluated_queries))
        summary.update(self._compute_mrr(filtered_qrels, raw_run, evaluated_queries))
        summary = collections.OrderedDict(
                                        sorted(
                                            summary.items(),
                                            key=lambda item: (item[0].split("@")[0], int(item[0].split("@")[1])),
                                        )
                                    )
        
        self.summary_results = summary
        self.diagnostics_results = diagnostics
        self.per_query_results = per_query
        self.raw_results =  {
            "summary": summary,
            "diagnostics": diagnostics,
            "per_query": per_query,

        }

        return {
            "summary": summary,
            "diagnostics": diagnostics,
            "per_query": per_query,

        }
    
    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "summary": self.summary_results,
            "diagnostics": self.diagnostics_results,
            "per_query": self.per_query_results,
        }

        (output_path / "retriever_eval_state.json").write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        input_path = Path(path)
        state = json.loads((input_path / "retriever_eval_state.json").read_text(encoding="utf-8"))

        self.summary_results = state.get("summary", {})
        self.diagnostics_results = state.get("diagnostics", {})
        self.per_query_results = state.get("per_query", {})
        self.raw_results = {
            "summary": self.summary_results,
            "diagnostics": self.diagnostics_results,
            "per_query": self.per_query_results,
        }

    def _build_measure_strings(self) -> set[str]:
        joined_k = ",".join(str(k) for k in self.top_k)
        return {
            f"recall.{joined_k}",
            f"P.{joined_k}",
        }

    def _empty_summary(self) -> dict[str, float]:
        summary: dict[str, float] = {}
        for k in self.top_k:
            summary[f"Accuracy@{k}"] = 0.0
            summary[f"Recall@{k}"] = 0.0
            summary[f"P@{k}"] = 0.0
            summary[f"HitRate@{k}"] = 0.0
            summary[f"MRR@{k}"] = 0.0
        return summary

    def _summarize_scores(
        self,
        per_query: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        summary = self._empty_summary()
        query_count = len(per_query)
        if query_count == 0:
            return summary

        for query_scores in per_query.values():
            for k in self.top_k:
                summary[f"Recall@{k}"] += query_scores[f"recall_{k}"]
                summary[f"P@{k}"] += query_scores[f"P_{k}"]

        for key in list(summary):
            if key.startswith(("Recall@", "P@")):
                summary[key] = round(summary[key] / query_count, 5)

        return summary

    def _compute_hit_rate(
        self,
        qrels: dict[str, dict[str, int]],
        raw_run: dict[str, dict[str, Any]],
        query_ids: list[str],
    ) -> dict[str, float]:
        hit_rate = {f"HitRate@{k}": 0.0 for k in self.top_k}
        query_count = len(query_ids)
        if query_count == 0:
            return hit_rate

        for query_id in query_ids:
            relevant_doc_ids = {
                document_id for document_id, score in qrels[query_id].items() if score > 0
            }
            hits = list(raw_run.get(query_id, {}).get("hits", []))
            for k in self.top_k:
                top_hits = hits[:k]
                if any(hit.get("document_id") in relevant_doc_ids for hit in top_hits):
                    hit_rate[f"HitRate@{k}"] += 1.0

        for key in hit_rate:
            hit_rate[key] = round(hit_rate[key] / query_count, 5)

        return hit_rate

    def _compute_mrr(
        self,
        qrels: dict[str, dict[str, int]],
        raw_run: dict[str, dict[str, Any]],
        query_ids: list[str],
    ) -> dict[str, float]:
        mrr = {f"MRR@{k}": 0.0 for k in self.top_k}
        query_count = len(query_ids)
        if query_count == 0:
            return mrr

        for query_id in query_ids:
            relevant_doc_ids = {
                document_id for document_id, score in qrels[query_id].items() if score > 0
            }
            hits = list(raw_run.get(query_id, {}).get("hits", []))
            for k in self.top_k:
                reciprocal_rank = 0.0
                for rank, hit in enumerate(hits[:k], start=1):
                    if hit.get("document_id") in relevant_doc_ids:
                        reciprocal_rank = 1.0 / rank
                        break
                mrr[f"MRR@{k}"] += reciprocal_rank

        for key in mrr:
            mrr[key] = round(mrr[key] / query_count, 5)

        return mrr
    
    def _compute_accuracy(
        self,
        qrels: dict[str, dict[str, int]],
        raw_run: dict[str, dict[str, Any]],
        query_ids: list[str],
    ) -> dict[str, float]:
        accuracy = {f"Accuracy@{k}": 0.0 for k in self.top_k}
        query_count = len(query_ids)
        if query_count == 0:
            return accuracy

        for query_id in query_ids:
            relevant_doc_ids = {
                document_id for document_id, score in qrels[query_id].items() if score > 0
            }
            hits = list(raw_run.get(query_id, {}).get("hits", []))
            for k in self.top_k:
                top_hits = hits[:k]
                if any(hit.get("document_id") in relevant_doc_ids for hit in top_hits):
                    accuracy[f"Accuracy@{k}"] += 1.0

        for key in accuracy:
            accuracy[key] = round(accuracy[key] / query_count, 5)

        return accuracy
