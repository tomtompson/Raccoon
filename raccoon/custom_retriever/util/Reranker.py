from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Any
import torch
import time

class Reranker():
    def __init__ (
            self,
            model_id: str,
            top_k: int,
            batch_size: int,
            max_lenght: int,
            device = "cuda",
            config: dict | None = None
    ):
        self.config = config
        self.model_id = model_id
        self.top_k = top_k
        self.batch_size = batch_size
        self.max_length = max_lenght
        self.device = device
        self._load_reranker()
        self.rerank_retrieval_metrics = {}        

    def _load_reranker(self,) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        model_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        if torch.cuda.is_available():
            model_kwargs["dtype"] = torch.float16
        if "jina" in self.model_id.lower():
            model_kwargs["use_flash_attn"] = True

        reranker_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device).eval()

        self.tokenizer = tokenizer
        self.reranker = reranker_model


    def rerank_with_transformers(
            self,
            corpus,
            queries,
            results,):
        
        rerank_results = {}
        query_ids = list(results.keys())

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
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize()

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
                    print(f"Reranked {qi}/{len(query_ids)} queries")

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

        peak_mem = None
        if use_cuda:
            peak_mem = (
                torch.cuda.max_memory_allocated(self.device)
                / (1024 ** 3)
            )

        print("\nReranking stats")
        print(f"Total queries: {len(query_ids)}")
        print(f"Total pairs: {total_pairs}")
        print(f"Total batches: {total_batches}")
        print(f"Total input tokens: {total_input_tokens}")
        print(f"Total wall time: {total_time:.2f}s")
        print(f"Avg time/query: {avg_time_query:.4f}s")
        print(f"Avg time/batch: {avg_time_batch:.4f}s")
        print(f"Pairs/sec: {pairs_per_sec:.2f}")
        print(f"Tokens/sec: {tokens_per_sec:.2f}")

        if use_cuda:
            print(f"Peak GPU memory: {peak_mem:.2f} GB")


        rerank_metrics = {
            "total_queries": len(query_ids),
            "total_pairs": total_pairs,
            "total_batches": total_batches,
            "total_input_tokens": total_input_tokens,
            "total_wall_time_sec": round(total_time, 4),
            "avg_time_query_sec": round(avg_time_query, 4),
            "avg_time_batch_sec": round(avg_time_batch, 4),
            "pairs_per_sec": round(pairs_per_sec, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "peak_gpu_memory_gb": round(peak_mem, 2) if peak_mem is not None else None,
        }

        return rerank_results , rerank_metrics
    