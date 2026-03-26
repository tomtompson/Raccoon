from langchain_core.documents import Document
from ..BaseSynthesizer import BaseSynthesizer

from typing import Any
from urllib import error, request
import json

from time import perf_counter
from statistics import mean


MULTI_QUERY_PROMPT = """Your task is to generate multiple search-style questions for a RAG system given a context passage.

For EACH passage, generate EXACTLY three different types of questions with a corresponding answer:

1. Entity-based question
- Focus on a specific named entity (person, place, organization, date, etc.)
- Should be answerable with a concise factual answer from the text

2. Open-ended question
- A natural, user-style query that asks for explanation or broader information
- Still grounded in the text, but less factoid and more descriptive

3. Abstract / claim-based question
- A higher-level or debatable statement framed as a question
- The text should contain information that can support or contradict this claim

IMPORTANT RULES:
- Do NOT mention "context", "passage", or similar meta references
- Questions must resemble realistic search engine queries
- Keep questions concise and natural
- Ensure all questions are grounded in the provided text
- Each question must have a corresponding answer supported by the text

Return ONLY valid JSON in exactly this structure:

{
  "entity_question": "string",
  "entity_answer": "string",
  "open_question": "string",
  "open_answer": "string",
  "abstract_question": "string",
  "abstract_answer": "string"
}

Here is the text:

{text}
"""


class MultiQueryOllamaSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        ollama_url: str,
        model_id: str,
        documents: list[Document | dict[str, Any]] | None = None,
        config: dict | None = None,
        prompt: str | None = None,
        timeout: int = 120,
    ):
        super().__init__(
            config=config,
            model_id=model_id,
            documents=documents,
            prompt=prompt or MULTI_QUERY_PROMPT,
        )
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        self.endpoint = self._load_llm()

    def _load_llm(self) -> str:
        return f"{self.ollama_url}/api/generate"

    def call_llm(self, passage: str) -> dict[str, Any]:
        payload = {
            "model": self.model_id,
            "prompt": self.prompt.format(text=passage),
            "stream": False,
            "options": {
                "num_ctx": 4096
            },
            "format": {
                "type": "object",
                "properties": {
                    "entity_question": {"type": "string"},
                    "entity_answer": {"type": "string"},
                    "open_question": {"type": "string"},
                    "open_answer": {"type": "string"},
                    "abstract_question": {"type": "string"},
                    "abstract_answer": {"type": "string"},
                },
                "required": [
                    "entity_question",
                    "entity_answer",
                    "open_question",
                    "open_answer",
                    "abstract_question",
                    "abstract_answer",
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
            raise RuntimeError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.endpoint}: {exc.reason}"
            ) from exc

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
            qa_data = self.parse_generated_response(generated_text)

            results.append(
                {
                    **processed_document,
                    "generated": generated_text,
                    "entity_question": qa_data["entity_question"],
                    "entity_answer": qa_data["entity_answer"],
                    "open_question": qa_data["open_question"],
                    "open_answer": qa_data["open_answer"],
                    "abstract_question": qa_data["abstract_question"],
                    "abstract_answer": qa_data["abstract_answer"],
                    "questions": [
                        {
                            "type": "entity",
                            "question": qa_data["entity_question"],
                            "answer": qa_data["entity_answer"],
                        },
                        {
                            "type": "open",
                            "question": qa_data["open_question"],
                            "answer": qa_data["open_answer"],
                        },
                        {
                            "type": "abstract",
                            "question": qa_data["abstract_question"],
                            "answer": qa_data["abstract_answer"],
                        },
                    ],
                    "synthesis_elapsed_seconds": round(call_elapsed, 4),
                }
            )

        self.results = results
        total_elapsed = perf_counter() - run_started_at

        total_questions = len(results) * 3
        total_answers = len(results) * 3

        self.metrics = {
            "documents_total": document_count,
            "generated": len(results),
            "questions_generated": total_questions,
            "answers_generated": total_answers,
            "llm_call_count": len(llm_call_durations),
            "llm_total_elapsed_seconds": round(sum(llm_call_durations), 4),
            "llm_average_elapsed_seconds": round(mean(llm_call_durations), 4)
            if llm_call_durations
            else 0.0,
            "elapsed_seconds": round(total_elapsed, 4),
            "average_elapsed_seconds_per_document": round(total_elapsed / document_count, 4)
            if document_count
            else 0.0,
            "model_id": self.model_id,
        }
        return results

    def parse_generated_response(self, generated_text: str) -> dict[str, str]:
        try:
            parsed = json.loads(generated_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Generated Ollama response was not valid JSON: {generated_text}"
            ) from exc

        required_fields = [
            "entity_question",
            "entity_answer",
            "open_question",
            "open_answer",
            "abstract_question",
            "abstract_answer",
        ]

        for field in required_fields:
            value = parsed.get(field)
            if not isinstance(value, str) or not value.strip():
                raise RuntimeError(
                    f"Generated Ollama response is missing or has an empty field: {field}"
                )

        return {
            "entity_question": parsed["entity_question"].strip(),
            "entity_answer": parsed["entity_answer"].strip(),
            "open_question": parsed["open_question"].strip(),
            "open_answer": parsed["open_answer"].strip(),
            "abstract_question": parsed["abstract_question"].strip(),
            "abstract_answer": parsed["abstract_answer"].strip(),
        }