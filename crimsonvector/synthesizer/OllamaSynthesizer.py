from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from langchain_core.documents import Document

from .BaseSynthesizer import BaseSynthesizer


DEFAULT_PROMPT = """You are generating synthetic question-answer data from a passage.
Use only the passage content and do not invent facts.
Return valid JSON with exactly these keys: "question", "answer".
The question should be answerable from the passage.
The answer should be concise and grounded in the passage.

Passage:
{passage}
"""


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
            "prompt": self.prompt.format(passage=passage),
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

        for document in self.documents:
            generated = self.call_llm(document.page_content)
            generated_text = generated.get("response", "")
            qa_pair = self._parse_generated_response(generated_text)

            results.append(
                {
                    "passage": document.page_content,
                    "generated": generated_text,
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "metadata": document.metadata,
                }
            )

        self.results = results
        return results

    def sythesize(self) -> list[dict[str, Any]]:
        return self.synthesize()

    def _parse_generated_response(self, generated_text: str) -> dict[str, str]:
        try:
            parsed = json.loads(generated_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Generated Ollama response was not valid JSON with question/answer fields."
            ) from exc

        question = str(parsed.get("question", "")).strip()
        answer = str(parsed.get("answer", "")).strip()

        if not question or not answer:
            raise RuntimeError("Generated Ollama response is missing question or answer.")

        return {"question": question, "answer": answer}
