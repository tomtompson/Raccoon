from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(slots=True)
class SynthesisConfig:
    output_dir: str
    embedding_id: str
    query_model: str
    query_validation_model: str
    judge_model: str
    qrel_validation_model: str
    embedding_kwargs: dict[str, Any] | None = None
    queries_per_parent_to_generate: int = 2
    max_queries_to_keep_per_parent: int = 1
    dense_k: int = 15
    bm25_k: int = 15
    rrf_top_k: int = 15
    same_topic_negative_k: int = 3
    random_negative_k: int = 3
    min_score_to_keep_in_qrels: int = 2
    max_qrels_per_query: int = 3
    include_source_parent_children: bool = True
    parent_text_limit_prompt: int = 3000
    candidate_text_limit_prompt: int = 800
    max_estimated_tokens: int = 5000
    overwrite_corpus: bool = False
    max_parents: int | None = None
    random_seed: int = 42
    target_queries_per_source: int | None = None
    shuffle_parents: bool = True
    reranker_id: str | None = None
    rerank_pool_size: int = 70
    rerank_keep_top_k: int = 40
    allow_copy_like_queries: bool = False
    min_query_tokens: int = 3
    max_query_tokens: int = 20

    @classmethod
    def from_inputs(
        cls,
        config: "SynthesisConfig | dict[str, Any] | None" = None,
        **overrides: Any,
    ) -> "SynthesisConfig":
        if config is None:
            data: dict[str, Any] = {}
        elif isinstance(config, cls):
            data = asdict(config)
        elif isinstance(config, dict):
            data = dict(config)
        else:
            raise TypeError("config must be a SynthesisConfig, dict, or None")

        data.update({key: value for key, value in overrides.items() if value is not None})
        valid_fields = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - valid_fields)
        if unknown:
            raise ValueError(f"Unknown synthesis config values: {', '.join(unknown)}")

        if data.get("embedding_kwargs") is None:
            data["embedding_kwargs"] = {}

        missing = [
            key
            for key in (
                "output_dir",
                "embedding_id",
                "query_model",
                "query_validation_model",
                "judge_model",
                "qrel_validation_model",
            )
            if not data.get(key)
        ]
        if missing:
            raise ValueError(f"Missing required synthesis config values: {', '.join(missing)}")

        return cls(**data)
