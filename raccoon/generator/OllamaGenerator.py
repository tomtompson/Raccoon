from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from .BaseGenerator import BaseGenerator


class OllamaGenerator(BaseGenerator):
    generator_type = "ollama"

    def __init__(
        self,
        retriever: Any,
        ollama_url: str,
        model_id: str,
        config: dict[str, Any] | None = None,
        prompt: str | None = None,
        top_k: int = 5,
        timeout: int = 120,
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.timeout = timeout
        super().__init__(
            retriever=retriever,
            config=config,
            model_id=model_id,
            prompt=prompt,
            top_k=top_k,
        )

    def _load_llm(self) -> str:
        return f"{self.ollama_url}/api/generate"

    def call_llm(self, query: str, retrieved: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model_id,
            "prompt": self.render_prompt(query=query, retrieved=retrieved),
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
            raise RuntimeError(
                f"Ollama request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Unable to reach Ollama at {self.endpoint}: {exc.reason}"
            ) from exc

        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Ollama returned a non-JSON response.") from exc

        return str(parsed.get("response", "")).strip()
