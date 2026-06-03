from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from raccoon.logging_utils import get_logger
from .BaseReranker import BaseReranker

log = get_logger(__name__)


class ColBERTReranker(BaseReranker):
    def __init__(
        self,
        model_id: str = "colbert-ir/colbertv2.0",
        top_k: int = 100,
        batch_size: int = 16,
        max_query_length: int = 32,
        max_doc_length: int = 180,
        device: str = "cuda",
        cache_doc_embeddings: bool = True,
        config: dict | None = None,
    ) -> None:
        self.config = config or {}
        self.model_id = model_id
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_query_length = max_query_length
        self.max_doc_length = max_doc_length
        self.device = device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        self.cache_doc_embeddings = cache_doc_embeddings

        self.doc_embedding_cache: dict[str, torch.Tensor] = {}

        log.info(
            "Initializing ColBERT reranker model=%s device=%s top_k=%s batch_size=%s",
            self.model_id,
            self.device,
            self.top_k,
            self.batch_size,
        )

        self._load_model()

    def _load_model(self) -> None:
        load_start = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            model_kwargs["dtype"] = torch.float16

        self.model = AutoModel.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device).eval()

        log.info("Loaded ColBERT reranker in %.2fs", time.perf_counter() - load_start)

    @staticmethod
    def _doc_to_text(doc: dict[str, Any]) -> str:
        title = str(doc.get("title", "") or "")
        text = str(doc.get("text", "") or "")
        return f"{title}\n{text}".strip() if title and text else title or text

    def _encode_texts(
        self,
        texts: list[str],
        max_length: int,
    ) -> list[torch.Tensor]:
        all_embeddings: list[torch.Tensor] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]

            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model(**inputs)

            token_embeddings = outputs.last_hidden_state
            token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=-1)

            attention_mask = inputs["attention_mask"].bool()

            for emb, mask in zip(token_embeddings, attention_mask):
                # Remove padding tokens. This still keeps CLS/SEP, which is acceptable
                # for a simple ColBERT-style reranker.
                all_embeddings.append(emb[mask].detach().cpu())

            del inputs, outputs, token_embeddings

        return all_embeddings

    @staticmethod
    def _maxsim_score(query_embedding: torch.Tensor, doc_embedding: torch.Tensor, device: str) -> float:
        q = query_embedding.to(device)
        d = doc_embedding.to(device)

        similarity = q @ d.T
        max_per_query_token = similarity.max(dim=1).values
        score = max_per_query_token.sum()

        return float(score.float().cpu().item())

    def rerank(self, corpus, queries, results):
        rerank_results: dict[str, dict[str, float]] = {}

        query_ids = list(results.keys())

        log.info(
            "Starting ColBERT rerank: queries=%d top_k=%s batch_size=%s device=%s",
            len(query_ids),
            self.top_k,
            self.batch_size,
            self.device,
        )

        doc_text_cache: dict[str, str] = {
            str(doc_id): self._doc_to_text(doc)
            for doc_id, doc in corpus.items()
        }

        total_pairs = 0
        total_batches = 0
        total_input_tokens = 0
        per_query_times: list[float] = []

        use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")

        if use_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

        wall_start = time.perf_counter()

        for qi, query_id in enumerate(query_ids, start=1):
            if use_cuda:
                torch.cuda.synchronize()

            query_start = time.perf_counter()

            query_text = queries.get(query_id, queries.get(str(query_id), ""))
            if not query_text:
                try:
                    query_text = queries.get(int(query_id), "")
                except (TypeError, ValueError):
                    query_text = ""

            candidate_scores = results[query_id]
            top_docs = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
            doc_ids = [str(doc_id) for doc_id, _ in top_docs]

            query_embedding = self._encode_texts(
                [str(query_text)],
                max_length=self.max_query_length,
            )[0]

            missing_doc_ids: list[str] = []
            missing_doc_texts: list[str] = []

            for doc_id in doc_ids:
                if self.cache_doc_embeddings and doc_id in self.doc_embedding_cache:
                    continue

                missing_doc_ids.append(doc_id)
                missing_doc_texts.append(doc_text_cache.get(doc_id, ""))

            if missing_doc_texts:
                doc_embeddings = self._encode_texts(
                    missing_doc_texts,
                    max_length=self.max_doc_length,
                )

                for doc_id, emb in zip(missing_doc_ids, doc_embeddings):
                    if self.cache_doc_embeddings:
                        self.doc_embedding_cache[doc_id] = emb

                total_batches += (len(missing_doc_texts) + self.batch_size - 1) // self.batch_size

            scored_docs: list[tuple[str, float]] = []

            for doc_id in doc_ids:
                if self.cache_doc_embeddings:
                    doc_embedding = self.doc_embedding_cache[doc_id]
                else:
                    doc_text = doc_text_cache.get(doc_id, "")
                    doc_embedding = self._encode_texts(
                        [doc_text],
                        max_length=self.max_doc_length,
                    )[0]

                score = self._maxsim_score(
                    query_embedding=query_embedding,
                    doc_embedding=doc_embedding,
                    device=self.device,
                )

                scored_docs.append((doc_id, score))
                total_pairs += 1
                total_input_tokens += int(query_embedding.shape[0] + doc_embedding.shape[0])

            reranked = {
                doc_id: score
                for doc_id, score in sorted(scored_docs, key=lambda x: x[1], reverse=True)
            }

            rerank_results[str(query_id)] = reranked

            if use_cuda:
                torch.cuda.synchronize()

            query_time = time.perf_counter() - query_start
            per_query_times.append(query_time)

            if qi % 50 == 0 or qi == len(query_ids):
                log.info("ColBERT reranked %d/%d queries", qi, len(query_ids))

        if use_cuda:
            torch.cuda.synchronize()

        total_time = time.perf_counter() - wall_start

        avg_time_query = sum(per_query_times) / len(per_query_times) if per_query_times else 0.0
        pairs_per_sec = total_pairs / total_time if total_time > 0 else 0.0
        queries_per_sec = len(query_ids) / total_time if total_time > 0 else 0.0
        tokens_per_sec = total_input_tokens / total_time if total_time > 0 else 0.0

        peak_mem = None
        if use_cuda:
            peak_mem = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        rerank_metrics = {
            "total_queries": len(query_ids),
            "total_pairs": total_pairs,
            "total_batches": total_batches,
            "total_input_tokens": total_input_tokens,
            "total_wall_time_sec": round(total_time, 4),
            "avg_time_query_sec": round(avg_time_query, 4),
            "pairs_per_sec": round(pairs_per_sec, 2),
            "queries_per_sec": round(queries_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_gpu_memory_gb": round(peak_mem, 2) if peak_mem is not None else None,
            "cached_doc_embeddings": len(self.doc_embedding_cache),
            "backend": "torch_colbert_maxsim_reranker",
        }

        return rerank_results, rerank_metrics