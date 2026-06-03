from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from raccoon.logging_utils import get_logger
from .BaseReranker import BaseReranker

log = get_logger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(
        self,
        model_id: str,
        top_k: int,
        batch_size: int,
        max_length: int,
        device: str = "cuda",
        config: dict | None = None,
    ) -> None:
        self.config = config or {}
        self.model_id = model_id
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        log.info(
            "Initializing reranker model=%s device=%s top_k=%s batch_size=%s max_length=%s",
            self.model_id,
            self.device,
            self.top_k,
            self.batch_size,
            self.max_length,
        )
        self._load_model()
        self.rerank_retrieval_metrics = {}

    def _load_model(self) -> None:
        load_start = time.perf_counter()
        log.info("Loading reranker tokenizer: %s", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
            log.info("CUDA available; loading reranker with float16 weights")
        if "jina" in self.model_id.lower():
            model_kwargs["use_flash_attn"] = True
            log.info("Enabled Jina flash attention for reranker")

        log.info("Loading reranker model: %s", self.model_id)
        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device).eval()

        self.tokenizer = tokenizer
        self.reranker = reranker_model
        log.info("Loaded reranker model in %.2fs", time.perf_counter() - load_start)


    def rerank(self, corpus, queries, results):
        rerank_results = {}
        query_ids = list(results.keys())
        log.info(
            "Starting rerank: queries=%d top_k=%s batch_size=%s max_length=%s device=%s",
            len(query_ids),
            self.top_k,
            self.batch_size,
            self.max_length,
            self.device,
        )

        total_pairs = 0
        total_batches = 0
        total_input_tokens = 0

        per_query_times = []
        per_batch_times = []

        doc_text_cache = {}
        for doc_id, doc in corpus.items():
            title = doc.get("title", "")
            text = doc.get("text", "")
            if title and text:
                doc_text_cache[str(doc_id)] = f"{title}\n{text}"
            else:
                doc_text_cache[str(doc_id)] = title if title else text

        use_cuda = torch.cuda.is_available() and str(self.device).startswith("cuda")

        if use_cuda:
            log.info("CUDA reranking enabled on device=%s", self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()
        else:
            log.info("CUDA reranking disabled; using device=%s", self.device)

        wall_start = time.perf_counter()

        with torch.inference_mode():
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
                doc_scores = results[query_id]

                top_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
                doc_ids = [doc_id for doc_id, _ in top_docs]

                pairs = [[query_text, doc_text_cache.get(str(doc_id), "")] for doc_id in doc_ids]
                scores = []

                for i in range(0, len(pairs), self.batch_size):
                    batch_start = time.perf_counter()

                    batch_pairs = pairs[i:i + self.batch_size]

                    inputs = self.tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    )

                    if "attention_mask" in inputs:
                        total_input_tokens += int(inputs["attention_mask"].sum().item())
                    else:
                        total_input_tokens += int(inputs["input_ids"].numel())

                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    outputs = self.reranker(**inputs)
                    logits = outputs.logits.squeeze(-1)

                    if logits.dim() == 0:
                        batch_scores = [float(logits.float().cpu().item())]
                    else:
                        batch_scores = logits.float().cpu().tolist()

                    scores.extend(batch_scores)

                    total_pairs += len(batch_pairs)
                    total_batches += 1

                    if use_cuda:
                        torch.cuda.synchronize()

                    batch_time = time.perf_counter() - batch_start
                    per_batch_times.append(batch_time)

                    del inputs, outputs, logits

                reranked = {
                    doc_id: score
                    for doc_id, score in sorted(
                        zip(doc_ids, scores),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                }
                rerank_results[query_id] = reranked

                if use_cuda:
                    torch.cuda.synchronize()

                query_time = time.perf_counter() - query_start
                per_query_times.append(query_time)

                if qi % 50 == 0 or qi == len(query_ids):
                    log.info("Reranked %d/%d queries", qi, len(query_ids))

        if use_cuda:
            torch.cuda.synchronize()

        total_time = time.perf_counter() - wall_start

        avg_time_query = (
            sum(per_query_times) / len(per_query_times)
            if per_query_times else 0.0
        )

        avg_time_batch = (
            sum(per_batch_times) / len(per_batch_times)
            if per_batch_times else 0.0
        )

        pairs_per_sec = (
            total_pairs / total_time
            if total_time > 0 else 0.0
        )

        tokens_per_sec = (
            total_input_tokens / total_time
            if total_time > 0 and total_input_tokens > 0 else 0.0
        )

        queries_per_sec = (
            len(query_ids) / total_time
            if total_time > 0 else 0.0
        )

        peak_mem = None
        if use_cuda:
            peak_mem = (
                torch.cuda.max_memory_allocated(self.device)
                / (1024 ** 3)
            )

        log.info("Reranking stats")
        log.info("Total queries: %d", len(query_ids))
        log.info("Total pairs: %d", total_pairs)
        log.info("Total batches: %d", total_batches)
        log.info("Total input tokens: %d", total_input_tokens)
        log.info("Total wall time: %.2fs", total_time)
        log.info("Avg time/query: %.4fs", avg_time_query)
        log.info("Avg time/batch: %.4fs", avg_time_batch)
        log.info("Pairs/sec: %.2f", pairs_per_sec)
        log.info("Queries/sec: %.2f", queries_per_sec)
        log.info("Tokens/sec: %.2f", tokens_per_sec)

        if use_cuda:
            log.info("Peak GPU memory: %.2f GB", peak_mem)


        rerank_metrics = {
            "total_queries": len(query_ids),
            "total_pairs": total_pairs,
            "total_batches": total_batches,
            "total_input_tokens": total_input_tokens,
            "total_wall_time_sec": round(total_time, 4),
            "avg_time_query_sec": round(avg_time_query, 4),
            "avg_time_batch_sec": round(avg_time_batch, 4),
            "pairs_per_sec": round(pairs_per_sec, 2),
            "queries_per_sec": round(queries_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_gpu_memory_gb": round(peak_mem, 2) if peak_mem is not None else None,
        }

        return rerank_results , rerank_metrics
    
