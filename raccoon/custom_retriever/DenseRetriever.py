from __future__ import annotations

import heapq
import os
from glob import glob
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from .BaseRetriever import BaseRetriever
from raccoon.custom_retriever.util.Reranker import Reranker
from .util.utils import pickle_load, save_embeddings


class DenseRetrieverSentenceBert(BaseRetriever):
    retriever_type = "dense"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        corpus: dict[str, dict[str, Any]] | None = None,
        queries: dict[str, str] | None = None,
        reranker: Reranker | None = None,
        model_id: str | None = None,
        max_length: int | None = None,
        device: str | None = None,
        query_prompt_name: str | None = None,
        passage_prompt_name: str | None = None,
        normalize_embeddings: bool = False,
        topk: int | list[int] = 20,
        batch_size: int = 128,
        corpus_chunk_size: int = 50_000,
        query_chunk_size: int = 1024,
        show_progress_bar: bool = True,
    ) -> None:
        super().__init__(config=config, corpus=corpus, queries=queries, reranker=reranker)

        if isinstance(topk, int):
            self.topk = topk
        elif isinstance(topk, list) and topk:
            self.topk = max(topk)
        else:
            raise ValueError("topk must be an int or non-empty list of ints")

        if not model_id:
            raise ValueError("model_id is required for DenseRetrieverSentenceBert")

        self.batch_size = batch_size
        self.corpus_chunk_size = corpus_chunk_size
        self.query_chunk_size = query_chunk_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings

        prompts = None
        if query_prompt_name and passage_prompt_name:
            prompts = {"query": query_prompt_name, "passage": passage_prompt_name}

        self.sentence_model = SentenceTransformer(
            model_name_or_path=model_id,
            trust_remote_code=True,
            device=device,
            prompts=prompts,
        )
        if max_length is not None:
            self.sentence_model.max_seq_length = max_length

    @staticmethod
    def _sorted_corpus(corpus: dict[str, dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
        ids = sorted(
            corpus,
            key=lambda cid: len(str(corpus[cid].get("title", "")) + str(corpus[cid].get("text", ""))),
            reverse=True,
        )
        return ids, [corpus[cid] for cid in ids]

    @staticmethod
    def _resolve_corpus_files(encode_output_path: str, corpus_filename: str) -> list[str]:
        pattern = os.path.join(encode_output_path, corpus_filename if "*" in corpus_filename else "corpus.*.pkl")
        files = glob(pattern)

        def shard_key(path: str) -> tuple[int, str]:
            tail = Path(path).stem.split(".")[-1]
            return (int(tail), path) if tail.isdigit() else (10**9, path)

        return sorted(files, key=shard_key)

    @staticmethod
    def _score(q: torch.Tensor, c: torch.Tensor, score_function: str) -> torch.Tensor:
        if score_function == "dot":
            return q @ c.T
        if score_function == "cos_sim":
            qn = torch.nn.functional.normalize(q, p=2, dim=1)
            cn = torch.nn.functional.normalize(c, p=2, dim=1)
            return qn @ cn.T
        raise ValueError("score_function must be one of: dot, cos_sim")

    def encode(
        self,
        corpus: dict[str, dict[str, Any]] | None = None,
        queries: dict[str, str] | None = None,
        encode_output_path: str = "./embeddings",
        overwrite: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        **kwargs,
    ) -> dict[str, Any]:
        corpus = corpus or self.corpus
        queries = queries or self.queries
        if not queries:
            raise ValueError("queries are required")
        if not corpus:
            raise ValueError("corpus is required")

        os.makedirs(encode_output_path, exist_ok=True)
        query_file = os.path.join(encode_output_path, query_filename)

        if overwrite or not os.path.exists(query_file):
            qids = list(queries.keys())
            qtxt = [queries[qid] for qid in qids]
            qemb = self.sentence_model.encode(
                qtxt,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                prompt_name="query",
            )
            save_embeddings(qemb, qids, query_file)

        cids, cdocs = self._sorted_corpus(corpus)
        for shard_idx, start in enumerate(range(0, len(cdocs), self.corpus_chunk_size)):
            shard_name = corpus_filename.replace("*", str(shard_idx)) if "*" in corpus_filename else f"corpus.{shard_idx}.pkl"
            shard_file = os.path.join(encode_output_path, shard_name)
            if not overwrite and os.path.exists(shard_file):
                continue

            end = min(start + self.corpus_chunk_size, len(cdocs))
            docs = cdocs[start:end]
            ids = cids[start:end]
            passages = [f"{d.get('title', '')}\n{d.get('text', '')}".strip() for d in docs]
            cemb = self.sentence_model.encode(
                passages,
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=self.normalize_embeddings,
                prompt_name="passage",
            )
            save_embeddings(cemb, ids, shard_file)

        return {
            "query_embeddings_file": query_file,
            "corpus_embeddings_files": self._resolve_corpus_files(encode_output_path, corpus_filename),
        }

    def search(
        self,
        top_k: int | None = None,
        score_function: str = "dot",
        corpus: dict[str, dict[str, Any]] | None = None,
        queries: dict[str, str] | None = None,
        encode_output_path: str = "./embeddings",
        overwrite_embeddings: bool = False,
        query_filename: str = "queries.pkl",
        corpus_filename: str = "corpus.*.pkl",
        use_faiss: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        top_k = top_k or self.topk

        encode_start = perf_counter()
        artifacts = self.encode(
            corpus=corpus,
            queries=queries,
            encode_output_path=encode_output_path,
            overwrite=overwrite_embeddings,
            query_filename=query_filename,
            corpus_filename=corpus_filename,
        )
        encode_latency = perf_counter() - encode_start

        search_start = perf_counter()
        q_np, q_ids = pickle_load(artifacts["query_embeddings_file"])
        q_emb = torch.as_tensor(q_np, dtype=torch.float32)
        shard_files = artifacts["corpus_embeddings_files"]
        if not shard_files:
            raise ValueError("No corpus embedding shards found")

        embedding_size_bytes = os.path.getsize(artifacts["query_embeddings_file"])
        embedding_size_bytes += sum(os.path.getsize(f) for f in shard_files)
        query_count = len(q_ids)
        corpus_count = 0
        embedding_dim = int(q_emb.shape[1]) if q_emb.ndim > 1 else 0

        if use_faiss:
            try:
                import faiss  # type: ignore
            except Exception as exc:
                raise RuntimeError("use_faiss=True but faiss is not installed") from exc

            index = faiss.IndexFlatIP(q_emb.shape[1])
            all_cids: list[str] = []
            for f in shard_files:
                c_np, c_ids = pickle_load(f)
                corpus_count += len(c_ids)
                c_np = c_np.astype("float32")
                if score_function == "cos_sim":
                    faiss.normalize_L2(c_np)
                index.add(c_np)
                all_cids.extend(str(x) for x in c_ids)

            q_np = q_np.astype("float32")
            if score_function == "cos_sim":
                faiss.normalize_L2(q_np)
            scores, idxs = index.search(q_np, top_k + 1)

            out: dict[str, dict[str, float]] = {}
            for qi, qid in enumerate(q_ids):
                row: dict[str, float] = {}
                for rank, cid_idx in enumerate(idxs[qi].tolist()):
                    if cid_idx < 0:
                        continue
                    cid = all_cids[cid_idx]
                    if cid == str(qid):
                        continue
                    row[cid] = float(scores[qi][rank])
                    if len(row) >= top_k:
                        break
                out[str(qid)] = row
            self.results = out
            self.metrics = self._build_metrics(
                encode_latency=encode_latency,
                search_latency=perf_counter() - search_start,
                query_count=query_count,
                corpus_count=corpus_count,
                shard_count=len(shard_files),
                embedding_dim=embedding_dim,
                embedding_size_bytes=embedding_size_bytes,
                top_k=top_k,
                score_function=score_function,
                use_faiss=use_faiss,
                results=out,
            )
            if self.reranker is not None:
                self.store_rerank_results()
            return out

        heaps: dict[str, list[tuple[float, str]]] = {str(qid): [] for qid in q_ids}

        for shard_file in shard_files:
            c_np, c_ids = pickle_load(shard_file)
            corpus_count += len(c_ids)
            c_emb = torch.as_tensor(c_np, dtype=torch.float32)

            for qs in range(0, len(q_ids), self.query_chunk_size):
                qe = min(qs + self.query_chunk_size, len(q_ids))
                s = self._score(q_emb[qs:qe], c_emb, score_function)
                s = torch.nan_to_num(s, nan=-1.0)

                k = min(top_k, s.shape[1])
                vals, idxs = torch.topk(s, k=k, dim=1, largest=True, sorted=False)

                for local_i, qid in enumerate(q_ids[qs:qe]):
                    qid = str(qid)
                    for ci, val in zip(idxs[local_i].tolist(), vals[local_i].tolist()):
                        cid = str(c_ids[ci])
                        if cid == qid:
                            continue
                        heap = heaps[qid]
                        item = (float(val), cid)
                        if len(heap) < top_k:
                            heapq.heappush(heap, item)
                        else:
                            heapq.heappushpop(heap, item)

        out: dict[str, dict[str, float]] = {}
        for qid, heap in heaps.items():
            best = sorted(heap, key=lambda x: x[0], reverse=True)
            out[qid] = {cid: score for score, cid in best}

        self.results = out
        self.metrics = self._build_metrics(
            encode_latency=encode_latency,
            search_latency=perf_counter() - search_start,
            query_count=query_count,
            corpus_count=corpus_count,
            shard_count=len(shard_files),
            embedding_dim=embedding_dim,
            embedding_size_bytes=embedding_size_bytes,
            top_k=top_k,
            score_function=score_function,
            use_faiss=use_faiss,
            results=out,
        )

        if self.reranker is not None:
            self.store_rerank_results()

        return out

    @staticmethod
    def _build_metrics(
        *,
        encode_latency: float,
        search_latency: float,
        query_count: int,
        corpus_count: int,
        shard_count: int,
        embedding_dim: int,
        embedding_size_bytes: int,
        top_k: int,
        score_function: str,
        use_faiss: bool,
        results: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        per_query_latency = search_latency / query_count if query_count else 0.0
        result_counts = [len(row) for row in results.values()]
        scores = [score for row in results.values() for score in row.values()]
        total_results = sum(result_counts)
        avg_results = total_results / query_count if query_count else 0.0
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "index_time": {
                "encoding": {
                    "time_in_seconds": encode_latency,
                    "documents": corpus_count,
                    "queries": query_count,
                    "shards": shard_count,
                    "embedding_dim": embedding_dim,
                    "size_in_bytes": embedding_size_bytes,
                    "size_in_mb": embedding_size_bytes / (1024 * 1024),
                    "docs_per_second": corpus_count / encode_latency if encode_latency else 0.0,
                },
            },
            "query_time": {
                "search": {
                    "time_in_seconds": search_latency,
                    "time_per_query_in_seconds": per_query_latency,
                    "queries_per_second": query_count / search_latency if search_latency else 0.0,
                    "queries": query_count,
                    "top_k": top_k,
                    "score_function": score_function,
                    "use_faiss": use_faiss,
                    "backend": "faiss" if use_faiss else "torch",
                    "total_results": total_results,
                    "avg_results_per_query": avg_results,
                    "min_results_per_query": min(result_counts) if result_counts else 0,
                    "max_results_per_query": max(result_counts) if result_counts else 0,
                    "min_score": min(scores) if scores else 0.0,
                    "max_score": max(scores) if scores else 0.0,
                    "avg_score": avg_score,
                },
            },
        }

    def create_index(self, *args, **kwargs) -> None:
        raise NotImplementedError("Dense retriever does not support create_index().")

    def index_corpus(self, *args, **kwargs) -> None:
        raise NotImplementedError("Dense retriever does not support index_corpous().")
