from __future__ import annotations

import json
import re
from statistics import mean
from time import perf_counter
from typing import Any
from urllib import error, request

from langchain_core.documents import Document

from .BaseSynthesizer import BaseSynthesizer
from raccoon.dataloader import BaseLoader


DEFAULT_PROMPT = """Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Return valid JSON only with exactly these keys:
{{
  "question": "...",
  "answer": "..."
}}

Now here is the context.

Context: {context} \n
JSON:"""


class OllamaSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        ollama_url: str,
        model_id: str,
        documents: list[Document | dict[str, Any]] | None = None,
        dataloader: BaseLoader | None = None,
        config: dict | None = None,
        prompt: str | None = None,
        timeout: int = 120,
    ):
        source_documents = documents if documents is not None else getattr(dataloader, "data", None)
        super().__init__(
            config=config,
            model_id=model_id,
            documents=source_documents,
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
            },
            "format": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                },
                "required": [
                    "question",
                    "answer",
                ],
            },
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
        processed_documents = self.process_documents()
        document_count = len(processed_documents)
        llm_call_durations: list[float] = []
        run_started_at = perf_counter()

        for processed_document in processed_documents:
            call_started_at = perf_counter()
            generated = self.call_llm(processed_document["passage"])
            call_elapsed = perf_counter() - call_started_at
            llm_call_durations.append(call_elapsed)
            generated_text = generated.get("response", "")
            qa_pair = self.parse_generated_response(generated_text)

            results.append(
                {
                    **processed_document,
                    "generated": generated_text,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
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

    def parse_generated_response(self, generated_text: str) -> dict[str, str]:
        generated_text = str(generated_text).strip()
        if not generated_text:
            raise RuntimeError("Generated Ollama response was empty.")

        json_pair = self._parse_json_response(generated_text)
        if json_pair is not None:
            return json_pair

        legacy_pair = self._parse_legacy_response(generated_text)
        if legacy_pair is not None:
            return legacy_pair

        raise RuntimeError("Generated Ollama response was not valid to extract")

    def _parse_json_response(self, generated_text: str) -> dict[str, str] | None:
        try:
            payload = json.loads(generated_text)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            raise RuntimeError("Generated Ollama response is missing question or answer.")

        return {"question": question, "answer": answer}

    def _parse_legacy_response(self, generated_text: str) -> dict[str, str] | None:
        match = re.search(
            r"Factoid question:\s*(.*?)\s*Answer:\s*(.*)",
            generated_text,
            flags=re.DOTALL,
        )
        if match is None:
            return None

        try:
            question = match.group(1).strip()
            answer = match.group(2).strip()
        except Exception as exc:
            raise RuntimeError("Generated Ollama response was not valid to extract") from exc

        if not question or not answer:
            raise RuntimeError("Generated Ollama response is missing question or answer.")

        return {"question": question, "answer": answer}
