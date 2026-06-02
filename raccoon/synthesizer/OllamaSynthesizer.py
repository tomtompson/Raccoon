from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import ollama
from langchain_core.documents import Document

from .BaseSynthesizer import BaseSynthesizer
from raccoon.dataloader import BaseLoader
from .helper.helper import (
    _extract_json_array,
    _extract_json_object,
    _safe_int,
)
import random

class OllamaSynthesizer(BaseSynthesizer):
    def __init__(
        self,
        prompt_paths: dict[str, str | Path] | None = None,
        prompts: dict[str, str] | None = None,
        host: str | None = None,
        num_ctx: int = 8192,
    ):
        self.prompts = {}
        self.host = host
        self.num_ctx = num_ctx

        if prompts:
            self.prompts.update(prompts)

        if prompt_paths:
            self.load_prompts(prompt_paths)

        if self.host:
            self.client = ollama.Client(host=self.host)
        else:
            self.client = ollama.Client()

    def _ollama_chat_json(
        self,
        model: str,
        prompt: str,
        schema: dict,
        temperature: float = 0.0,
        think: bool = False,
    ) -> str:
        try:
            response = self.client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_ctx": self.num_ctx,
                },
                think=think,
                format=schema,
            )
        except Exception as exc:
            raise RuntimeError(f"Ollama chat call failed for model '{model}': {exc}") from exc

        try:
            return response["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected Ollama response shape: {response}") from exc

    def generate_description_of_ds(
        self,
        model: str,
        language: str,
        corpus: Dict,
        queries: Dict,
        description_length: int = 100
    ) -> str:

        if language.lower() in ("english", "en"):
            language = "english"
            prompt_name = "description_generation_english"
        elif language.lower() in ("dutch", "nl", "nederlands"):
            language = "dutch"
            prompt_name = "description_generation_dutch"
        else:
            raise ValueError("language must be one of: english, en, dutch, nl, nederlands")

        if prompt_name not in self.prompts:
            prompt = self.load_prompt(f"prompts/description/{prompt_name}.txt")
            self.prompts.update({prompt_name: prompt})

        key_c, value_c = random.choice(list(corpus.items()))
        key_q, value_q = random.choice(list(queries.items()))

        prompt_query_generation = self.render_prompt(
            "query_generation",
            n_candidates=8,
            parent_text=value_c["text"][:2200],
        )

        promt_candidate_judging = self.render_prompt(
            "candidate_judging",
            query="Voorbeeldvraag",
            candidates_json=json.dumps([{
                "child_id": "voorbeeld_id",
                "title": "Voorbeeld titel",
                "text": "Voorbeeld tekst",
            }], ensure_ascii=False),
        )

        prompt = self.render_prompt(
            prompt_name,
            language=language,
            example_corpus_text=value_c["text"][:2000],
            example_query_text=value_q[:2000],
            description_length=description_length,
            query_generation_prompt=prompt_query_generation,
            candidate_judging_prompt=promt_candidate_judging,
        )

        schema = {
            "type": "string",
        }

        content = self._ollama_chat_json(
            model=model,
            prompt=prompt,
            schema=schema,
            temperature=0.7,
            think=False,
        )

        return content.strip()

    def _generate_realistic_query_candidates_from_parent(
        self,
        parent_text: str,
        model: str,
        n_candidates: int = 8,
        parent_text_limit: int = 2200,
    ) -> list[str]:
        prompt = self.render_prompt(
            "query_generation",
            n_candidates=n_candidates,
            parent_text=parent_text[:parent_text_limit],
        )

        schema = {
            "type": "array",
            "items": {"type": "string"},
        }

        content = self._ollama_chat_json(
            model=model,
            prompt=prompt,
            schema=schema,
            temperature=0.9,
            think=False,
        )

        queries = _extract_json_array(content)
        queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
        return queries[:n_candidates]

    def _validate_query_grounding(
        self,
        query: str,
        parent_text: str,
        model: str,
        parent_text_limit: int = 2200,
    ) -> dict:
        prompt = self.render_prompt(
            "query_validation",
            query=query,
            parent_text=parent_text[:parent_text_limit],
        )

        schema = {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "grounded_in_text": {"type": "boolean"},
                "realistic_user_query": {"type": "boolean"},
                "too_broad": {"type": "boolean"},
                "too_case_specific": {"type": "boolean"},
                "copied_from_text": {"type": "boolean"},
                "artificial_or_benchmarky": {"type": "boolean"},
                "unsupported_by_text": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": [
                "keep",
                "grounded_in_text",
                "realistic_user_query",
                "too_broad",
                "too_case_specific",
                "copied_from_text",
                "artificial_or_benchmarky",
                "unsupported_by_text",
                "explanation",
            ],
        }

        content = self._ollama_chat_json(
            model=model,
            prompt=prompt,
            schema=schema,
            temperature=0.0,
            think=False,
        )

        return _extract_json_object(content)

    def _judge_candidates_with_ollama_realistic(
        self,
        query: str,
        candidates: list,
        model: str,
        candidate_text_limit: int = 700,
    ) -> list[dict]:
        formatted_candidates = []
        for c in candidates:
            formatted_candidates.append({
                "child_id": c["child_id"],
                "title": (c.get("title", "") or "")[:180],
                "text": (c.get("text", "") or "")[:candidate_text_limit],
            })

        prompt = self.render_prompt(
            "candidate_judging",
            query=query,
            candidates_json=json.dumps(formatted_candidates, ensure_ascii=False),
        )

        schema = {
        "type": "array",
        "minItems": len(formatted_candidates),
        "maxItems": len(formatted_candidates),
        "items": {
            "type": "object",
            "properties": {
                "child_id": {"type": "string"},
                "score": {"type": "integer", "minimum": 0, "maximum": 3},
                "reason": {"type": "string"},
            },
            "required": ["child_id", "score", "reason"],
        },
    }

        content = self._ollama_chat_json(
            model=model,
            prompt=prompt,
            schema=schema,
            temperature=0.0,
            think=False,
        )

        judgments = _extract_json_array(content)

        cleaned = []
        for j in judgments:
            child_id = str(j.get("child_id", "")).strip()
            score = max(0, min(3, _safe_int(j.get("score", 0), 0)))
            reason = str(j.get("reason", "")).strip()

            if child_id:
                cleaned.append({
                    "child_id": child_id,
                    "score": score,
                    "reason": reason,
                })

        return cleaned

    def _judge_query_distribution_quality(
        self,
        query: str,
        judgments: list[dict],
        model: str,
    ) -> dict:
        prompt = self.render_prompt(
            "distribution_validation",
            query=query,
            judgments_json=json.dumps(judgments, ensure_ascii=False),
        )

        schema = {
            "type": "object",
            "properties": {
                "keep": {"type": "boolean"},
                "answerable": {"type": "boolean"},
                "plausible_distribution": {"type": "boolean"},
                "too_generic": {"type": "boolean"},
                "too_artificial": {"type": "boolean"},
                "explanation": {"type": "string"},
            },
            "required": [
                "keep",
                "answerable",
                "plausible_distribution",
                "too_generic",
                "too_artificial",
                "explanation",
            ],
        }

        content = self._ollama_chat_json(
            model=model,
            prompt=prompt,
            schema=schema,
            temperature=0.0,
            think=False,
        )

        return _extract_json_object(content)
