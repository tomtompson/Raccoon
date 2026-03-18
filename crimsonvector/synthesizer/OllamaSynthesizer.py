from __future__ import annotations

import json
from statistics import mean
from time import perf_counter
from typing import Any
from urllib import error, request

from langchain_core.documents import Document

from .BaseSynthesizer import BaseSynthesizer


DEFAULT_PROMPT = """Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".


Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


class OllamaSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        ollama_url: str,
        model_id: str,
        documents: list[Document] | None = None,
        config: dict | None = None,
        prompt: str | None = None,
        timeout: int = 120,
    ):
        super().__init__(
            config=config,
            model_id=model_id,
            documents=documents,
            prompt=prompt or DEFAULT_PROMPT,
        )
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.endpoint = self._load_llm()

    def _load_llm(self) -> str:
        return f"{self.ollama_url}/api/generate"

    def call_llm(self, passage: str) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "prompt": self.prompt.format(context=passage),
            "stream": False,
            "options": {
                "num_ctx": 4096
            }
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
            return json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

    def synthesize(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        document_count = len(self.documents or [])
        llm_call_durations: list[float] = []
        run_started_at = perf_counter()

        for document in self.documents or []:
            call_started_at = perf_counter()
            generated = self.call_llm(document.page_content)
            call_elapsed = perf_counter() - call_started_at
            llm_call_durations.append(call_elapsed)
            generated_text = generated.get("response", "")
            qa_pair = self._parse_generated_response(generated_text)

            results.append(
                {
                    "passage": document.page_content,
                    "generated": generated_text,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "metadata": document.metadata,
                    "synthesis_elapsed_seconds": round(call_elapsed, 4),
                }
            )

        self.results = results
        total_elapsed = perf_counter() - run_started_at
        self.metrics = {
            "documents_total": document_count,
            "generated": len(results),
            "questions_generated": len(results),
            "answers_generated": len(results),
            "llm_call_count": len(llm_call_durations),
            "llm_total_elapsed_seconds": round(sum(llm_call_durations), 4),
            "llm_average_elapsed_seconds": round(mean(llm_call_durations), 4) if llm_call_durations else 0.0,
            "elapsed_seconds": round(total_elapsed, 4),
            "average_elapsed_seconds_per_document": round(total_elapsed / document_count, 4) if document_count else 0.0,
            "model_id": self.model_id,
        }
        return results

    def _parse_generated_response(self, generated_text: str) -> dict[str, str]:
        if "Factoid question: " not in generated_text or "Answer: " not in generated_text:
            raise RuntimeError("Generated Ollama response was not valid to extract")

        try:
            question = generated_text.split("Factoid question: ", 1)[1].split("Answer: ", 1)[0].strip()
            answer = generated_text.split("Answer: ", 1)[1].strip()
        except Exception as exc:
            raise RuntimeError("Generated Ollama response was not valid to extract") from exc

        if not question or not answer:
            raise RuntimeError("Generated Ollama response is missing question or answer.")

        return {"question": question, "answer": answer}
