from __future__ import annotations

import copy
import json
import re
from typing import Any
from urllib import error, request

from .BaseCritiquer import BaseCritiquer


DEFAULT_PROMPTS = {
    "groundedness": """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

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
Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

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
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

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
        sythesizer=None,
        prompts: dict[str, str] | None = None,
        timeout: int = 120,
    ):
        super().__init__(config=config, sythesizer=sythesizer, model_id=model_id)
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
        critique_inputs = outputs or self.outputs or getattr(self.sythesizer, "results", [])
        results: list[dict[str, Any]] = []

        for output in critique_inputs:
            result = copy.deepcopy(output)
            evaluations = {
                "groundedness": self.call_llm(
                    self.prompts["groundedness"].format(
                        context=self._resolve_context(result),
                        question=result["question"],
                    )
                ),
                "relevance": self.call_llm(
                    self.prompts["relevance"].format(question=result["question"])
                ),
                "standalone": self.call_llm(
                    self.prompts["standalone"].format(question=result["question"])
                ),
            }

            for criterion, evaluation in evaluations.items():
                score, rationale = self._parse_evaluation(evaluation)
                result[f"{criterion}_score"] = score
                result[f"{criterion}_eval"] = rationale

            results.append(result)

        self.results = results
        self.metrics = {"generated": len(results)}
        return results

    def filter(
        self,
        min_groundedness: int = 4,
        min_relevance: int = 4,
        min_standalone: int = 4,
        outputs: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        critique_outputs = outputs or self.results
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
                "min_groundedness": min_groundedness,
                "min_relevance": min_relevance,
                "min_standalone": min_standalone,
            }
        )
        return filtered

    def _resolve_context(self, output: dict[str, Any]) -> str:
        for key in ("context", "passage", "content"):
            value = output.get(key)
            if value:
                return str(value)
        raise KeyError("Critique input must include one of: context, passage, content.")

    def _parse_evaluation(self, evaluation: str) -> tuple[int, str]:
        rating_match = re.search(r"Total rating:\s*([1-5])\b", evaluation)
        rationale_match = re.search(
            r"Evaluation:\s*(.*?)\s*Total rating:",
            evaluation,
            flags=re.DOTALL,
        )

        if rating_match is None or rationale_match is None:
            raise RuntimeError("Critique response is missing Evaluation or Total rating.")

        return int(rating_match.group(1)), rationale_match.group(1).strip()
