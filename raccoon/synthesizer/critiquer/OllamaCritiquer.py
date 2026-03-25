from __future__ import annotations

import copy
import json
import re
from statistics import mean
from time import perf_counter
from typing import Any
from urllib import error, request

from raccoon.analysis.critique_report import (
    DEFAULT_THRESHOLDS,
    normalize_critique_row,
    summarize_rows,
)

from .BaseCritiquer import BaseCritiquer


DEFAULT_PROMPTS = {
    "groundedness": """
You will be given a context and a question.
Assume the context is taken from technical documentation, a product knowledge base, developer guides, or other structured reference material.
Your task is to provide a 'total rating' scoring how well the question can be answered accurately and unambiguously using only the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable from the context, and 5 means that the question is directly, clearly, and unambiguously answerable from the context.
Prefer higher ratings when the context explicitly defines the concept, behavior, procedure, requirement, limitation, or configuration asked about.
Prefer lower ratings when the context is missing key details, only partially addresses the question, or would require outside assumptions.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}
Context: {context}
Answer:::
""".strip(),
"relevance": """
You will be given a question.
Assume the question is intended for a tightly related technical corpus, such as a single product's documentation set, a coherent project knowledge base, or a small family of related documents used in a RAG/IR system.
Your task is to provide a 'total rating' representing how useful this question is as a retrieval target within that corpus.
Give your answer on a scale of 1 to 5, where 1 means that the question is vague, trivial, presentation-dependent, or not useful for retrieval in the corpus, and 5 means that the question is clear, meaningful, and useful for retrieving important information from the corpus.
Prefer higher ratings for questions that capture a meaningful concept, mechanism, component, threshold, formula, behavior, definition, or relationship that someone working with this corpus would plausibly search for.
Do NOT lower the rating merely because the question uses corpus-specific terminology, product names, acronyms, or specialized domain concepts.
Prefer lower ratings for questions that are mostly biographical trivia, historical side facts, local figure-label lookups, wording tied to a specific presentation artifact, repetitive, speculative, or otherwise unlikely to help retrieve useful corpus knowledge.

Use this rubric:
- 5 = strong retrieval question for this corpus; clear, meaningful, and likely useful.
- 4 = good retrieval question; somewhat narrow but still useful.
- 3 = answerable and somewhat useful, but niche or weakly important.
- 2 = low-value retrieval target; mostly trivia or too tied to local presentation.
- 1 = vague, malformed, context-dependent, or not useful for corpus retrieval.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer:::
""".strip(),
"standalone": """
You will be given a question.
Assume the question is meant to be stored, searched, or answered within technical documentation or a product knowledge base.
Your task is to provide a 'total rating' representing how understandable and self-contained this question is without extra surrounding context.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on missing context to be understood, and 5 means that the question is clear and meaningful on its own.
Focus only on context dependence. Do NOT lower the rating merely because the question contains specialized technical terms, product names, formulas, acronyms, or corpus-specific language.
If the question refers to a particular setting or local presentation artifact, such as 'in the context', 'in the document', 'according to the passage', 'in the diagram', 'in the figure', or similar wording, the rating must be 1.
The question may contain technical nouns, domain-specific terms, acronyms, class names, API names, or product features and still deserve a 5, as long as a reader of technical documentation can understand what is being asked without needing prior conversational context.

For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}
Answer:::
""".strip(),
}


class OllamaCritiquer(BaseCritiquer):
    def __init__(
        self,
        ollama_url: str,
        model_id: str,
        outputs: list[dict[str, Any]] | None = None,
        config: dict | None = None,
        synthesizer=None,
        prompts: dict[str, str] | None = None,
        timeout: int = 120,
    ):
        super().__init__(
            config=config,
            synthesizer=synthesizer,
            model_id=model_id,
        )
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.outputs = outputs or []
        self.prompts = self.load_prompt(prompts)
        self.endpoint = self._load_llm()

    def _load_llm(self) -> str:
        return f"{self.ollama_url}/api/generate"

    def load_prompt(self, prompts: dict[str, str] | None = None) -> dict[str, str]:
        merged_prompts = DEFAULT_PROMPTS.copy()
        if prompts:
            merged_prompts.update(prompts)
        return merged_prompts

    def call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": 4096},
        }

        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with status {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Unable to reach Ollama at {self.endpoint}: {exc.reason}") from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

        return str(parsed.get("response", "")).strip()

    def critique(self, outputs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        critique_inputs = self.resolve_critique_inputs(outputs)
        results: list[dict[str, Any]] = []
        criterion_durations: dict[str, list[float]] = {
            "groundedness": [],
            "relevance": [],
            "standalone": [],
        }
        run_started_at = perf_counter()

        for output in critique_inputs:
            result = copy.deepcopy(output)
            evaluations: dict[str, str] = {}

            groundedness_started_at = perf_counter()
            evaluations["groundedness"] = self.call_llm(
                self.prompts["groundedness"].format(
                    context=self.resolve_context(result),
                    question=result["question"],
                )
            )
            criterion_durations["groundedness"].append(perf_counter() - groundedness_started_at)

            relevance_started_at = perf_counter()
            evaluations["relevance"] = self.call_llm(
                self.prompts["relevance"].format(question=result["question"])
            )
            criterion_durations["relevance"].append(perf_counter() - relevance_started_at)

            standalone_started_at = perf_counter()
            evaluations["standalone"] = self.call_llm(
                self.prompts["standalone"].format(question=result["question"])
            )
            criterion_durations["standalone"].append(perf_counter() - standalone_started_at)

            for criterion, evaluation in evaluations.items():
                score, rationale = self.parse_evaluation(evaluation)
                result[f"{criterion}_score"] = score
                result[f"{criterion}_eval"] = rationale

            normalized_result = normalize_critique_row(result, DEFAULT_THRESHOLDS)
            normalized_result["critique_elapsed_seconds"] = round(
                sum(criterion_durations[criterion][-1] for criterion in criterion_durations),
                4,
            )
            results.append(normalized_result)

        self.results = results
        total_elapsed = perf_counter() - run_started_at
        summary = summarize_rows(results)
        llm_call_durations = [
            duration
            for durations in criterion_durations.values()
            for duration in durations
        ]
        self.metrics = {
            "generated": len(results),
            "total_rows": summary.total_rows,
            "review_required": summary.review_required,
            "average_total_score": summary.average_total_score,
            "median_total_score": summary.median_total_score,
            "score_distributions": summary.score_distributions,
            "low_groundedness": sum(1 for row in results if row["low_groundedness"]),
            "low_relevance": sum(1 for row in results if row["low_relevance"]),
            "low_standalone": sum(1 for row in results if row["low_standalone"]),
            "context_dependent": sum(1 for row in results if row["context_dependent"]),
            "likely_trivia": sum(1 for row in results if row["likely_trivia"]),
            "llm_call_count": len(llm_call_durations),
            "criterion_call_count": {criterion: len(durations) for criterion, durations in criterion_durations.items()},
            "criterion_elapsed_seconds": {
                criterion: round(sum(durations), 4) for criterion, durations in criterion_durations.items()
            },
            "criterion_average_elapsed_seconds": {
                criterion: round(mean(durations), 4) if durations else 0.0
                for criterion, durations in criterion_durations.items()
            },
            "llm_total_elapsed_seconds": round(sum(llm_call_durations), 4),
            "llm_average_elapsed_seconds": round(mean(llm_call_durations), 4) if llm_call_durations else 0.0,
            "elapsed_seconds": round(total_elapsed, 4),
            "average_elapsed_seconds_per_row": round(total_elapsed / len(results), 4) if results else 0.0,
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "model_id": self.model_id,
        }
        return results
    
    def filter(
        self,
        min_groundedness: int = 4,
        min_relevance: int = 3,
        min_standalone: int = 4,
        results: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        critique_outputs = results or self.results
        filtered = [
            output
            for output in critique_outputs
            if output.get("groundedness_score", 0) >= min_groundedness
            and output.get("relevance_score", 0) >= min_relevance
            and output.get("standalone_score", 0) >= min_standalone
        ]

        self.metrics.update(
            {
                "filtered": len(filtered),
                "filtered_out": len(critique_outputs) - len(filtered),
                "passed_filter_rate": round(len(filtered) / len(critique_outputs), 4) if critique_outputs else 0.0,
                "min_groundedness": min_groundedness,
                "min_relevance": min_relevance,
                "min_standalone": min_standalone,
            }
        )

        self.results = filtered
        return filtered

    def resolve_critique_inputs(
        self,
        outputs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if outputs is not None:
            return outputs

        if self.outputs:
            return self.outputs

        synthesizer = self.synthesizer
        if synthesizer is not None:
            if getattr(synthesizer, "results", None):
                return list(synthesizer.results)

            processed_documents = list(getattr(synthesizer, "processed_documents", []) or [])
            if processed_documents and all(
                row.get("passage") and row.get("question") and row.get("answer")
                for row in processed_documents
            ):
                self.outputs = processed_documents
                return processed_documents

            if getattr(synthesizer, "documents", None):
                synthesized_outputs = synthesizer.synthesize()
                self.outputs = synthesized_outputs
                return synthesized_outputs

        return []

    def resolve_context(self, output: dict[str, Any]) -> str:
        passage = output.get("passage")
        if passage:
            return str(passage)
        raise KeyError("Critique input must include a non-empty 'passage'.")

    def parse_evaluation(self, evaluation: str) -> tuple[int, str]:
        rating_match = re.search(r"Total rating:\s*([1-5])\b", evaluation)
        rationale_match = re.search(
            r"Evaluation:\s*(.*?)\s*Total rating:",
            evaluation,
            flags=re.DOTALL,
        )

        if rating_match is None or rationale_match is None:
            raise RuntimeError("Critique response is missing Evaluation or Total rating.")

        return int(rating_match.group(1)), rationale_match.group(1).strip()
