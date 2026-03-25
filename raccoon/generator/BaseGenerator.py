from __future__ import annotations

from abc import ABC, abstractmethod
from statistics import mean
from time import perf_counter
from raccoon.retriever.BaseRetriever import BaseRetriever
from typing import Any


DEFAULT_PROMPT = """You are answering a user query using retrieved context. 
Use only the retrieved context when writing the answer.
If the retrieved context does not contain the right information for an answer, return "no answer".


Query: {query}

Retrieved context:
{context}

Answer:"""


class BaseGenerator(ABC):
    generator_type = "base"

    def __init__(
        self,
        retriever: BaseRetriever,
        config: dict[str, Any] | None = None,
        model_id: str | None = None,
        prompt: str | None = None,
        top_k: int = 5,
        search_results: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.retriever = retriever
        self.config = config or {}
        self.model_id = model_id
        self.prompt = prompt or DEFAULT_PROMPT
        self.top_k = top_k
        self.search_results = search_results or {}
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}
        self.endpoint = self._load_llm()

    @abstractmethod
    def _load_llm(self) -> Any:
        pass

    @abstractmethod
    def call_llm(self, query: str, retrieved: list[dict[str, Any]]) -> str:
        pass

    def generate(
        self,
        search_results: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        benchmark_data = getattr(self.retriever, "benchmark_data", {}) or {}
        queries = benchmark_data.get("queries", {})
        results = search_results or self.search_results or benchmark_data.get("results", {})

        if not queries:
            self.results = []
            self.metrics = self._build_metrics([], 0.0)
            return []

        if not results or any(query_id not in results for query_id in queries):
            results = self.retriever.bulk_search(top_k=self.top_k)
            benchmark_data = getattr(self.retriever, "benchmark_data", {}) or {}
            results = results or benchmark_data.get("results", {})

        self.search_results = results

        generated_rows: list[dict[str, Any]] = []
        llm_call_durations: list[float] = []
        run_started_at = perf_counter()

        for query_id, query_payload in queries.items():
            query_text = str(query_payload.get("text", "")).strip()
            retrieved_hits = list(results.get(query_id, {}).get("hits", []))

            call_started_at = perf_counter()
            answer = self.call_llm(query=query_text, retrieved=retrieved_hits).strip()
            call_elapsed = perf_counter() - call_started_at
            llm_call_durations.append(call_elapsed)

            generated_rows.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "answer": answer,
                    "retrieved": retrieved_hits,
                    "generation_elapsed_seconds": round(call_elapsed, 4),
                }
            )

        total_elapsed = perf_counter() - run_started_at
        self.results = generated_rows
        self.metrics = self._build_metrics(llm_call_durations, total_elapsed)
        return generated_rows

    def render_prompt(self, query: str, retrieved: list[dict[str, Any]]) -> str:
        return self.prompt.format(
            query=query,
            context=self._format_retrieved_context(retrieved),
        )

    def _format_retrieved_context(self, retrieved: list[dict[str, Any]]) -> str:
        if not retrieved:
            return "[no retrieved context]"

        blocks: list[str] = []
        for hit in retrieved:
            blocks.append(
                "\n".join(
                    [
                        f"Rank: {hit.get('rank', '')}",
                        f"Score: {hit.get('score', '')}",
                        f"Document ID: {hit.get('document_id', '')}",
                        f"Content: {hit.get('content', '')}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _build_metrics(
        self,
        llm_call_durations: list[float],
        total_elapsed: float,
    ) -> dict[str, Any]:
        query_count = len(self.results)
        return {
            "generator_type": self.generator_type,
            "queries_total": query_count,
            "generated": query_count,
            "llm_call_count": len(llm_call_durations),
            "llm_total_elapsed_seconds": round(sum(llm_call_durations), 4),
            "llm_average_elapsed_seconds": round(mean(llm_call_durations), 4)
            if llm_call_durations
            else 0.0,
            "elapsed_seconds": round(total_elapsed, 4),
            "average_elapsed_seconds_per_query": round(total_elapsed / query_count, 4)
            if query_count
            else 0.0,
            "model_id": self.model_id,
        }
