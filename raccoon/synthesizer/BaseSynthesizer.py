from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, Dict, List, Optional

from langchain_core.documents import Document
from raccoon.dataloader import BaseLoader

import json
from pathlib import Path

import random

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from collections import Counter, defaultdict

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from .helper.helper import (
    _build_child_records_from_vectorstore,
    _load_existing_query_texts,
    _load_processed_parent_ids,
    _load_saved_query_counts_by_source,
    _normalize_text,
    _mark_parent_processed,
    _looks_artificial,
    _looks_too_long,
    _looks_too_short,
    _query_is_too_copy_like,
    _merge_dictionaries,
    _append_jsonl,
    _query_has_enough_signal,
    _select_qrels_realistic,
    _count_scores,
    _sample_random_negatives,
    _sample_same_topic_negatives_via_overlap,
    _rrf_fuse,
    _estimate_judge_tokens,
    _build_faiss_index)

from .helper.SimpleBM25 import SimpleBM25

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme
from rich.progress import Progress

# =========================
# Helpers LOgging
# =========================
console = Console(theme=Theme({"logging.level.query": "green",
                               "logging.level.judge": "orange1",
                               "logging.level.start": "cyan3",
                               "logging.level.end": "bright_magenta",
                               "logging.level.rerank": "sky_blue2"}))
                               
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, console=console)]
)

QUERY_LEVEL = 25
logging.addLevelName(QUERY_LEVEL, "QUERY")
def query(self, message, *args, **kwargs):
    if self.isEnabledFor(QUERY_LEVEL):
        self._log(QUERY_LEVEL, message, args, **kwargs)
        
JUDGE_LEVEL = 26
logging.addLevelName(JUDGE_LEVEL, "JUDGE")
def judge(self, message, *args, **kwargs):
    if self.isEnabledFor(JUDGE_LEVEL):
        self._log(JUDGE_LEVEL, message, args, **kwargs)

START_LEVEL = 27
logging.addLevelName(START_LEVEL, "START")
def start(self, message, *args, **kwargs):
    if self.isEnabledFor(START_LEVEL):
        self._log(START_LEVEL, message, args, **kwargs)

END_LEVEL = 28
logging.addLevelName(END_LEVEL, "END")
def end(self, message, *args, **kwargs):
    if self.isEnabledFor(END_LEVEL):
        self._log(END_LEVEL, message, args, **kwargs)

RERANK_LEVEL = 29
logging.addLevelName(RERANK_LEVEL, "RERANK")
def rerank(self, message, *args, **kwargs):
    if self.isEnabledFor(RERANK_LEVEL):
        self._log(RERANK_LEVEL, message, args, **kwargs)


logging.Logger.query = query
logging.Logger.judge = judge
logging.Logger.start = start
logging.Logger.end = end
logging.Logger.rerank = rerank
log = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("huggingface").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class BaseSynthesizer(ABC):
    @abstractmethod
    def _generate_realistic_query_candidates_from_parent(self, parent_text, model, n_candidates, parent_text_limit):
        pass
    @abstractmethod
    def _validate_query_grounding(self, query, parent_text, model,  parent_text_limit):
        pass
    @abstractmethod
    def _judge_query_distribution_quality(self, query, judgments, model):
        pass
    @abstractmethod
    def _judge_candidates_with_ollama_realistic(self, query, candidates, model, candidate_text_limit):
        pass

    @abstractmethod
    def generate_description_of_ds(self, text: str, model: str, language: str, corpus: Dict, queries: Dict, description_length: int = 100) -> str:
        pass    

    def _pool_candidates_realistic(
        self,
        query: str,
        parent_id: str,
        vectorstore,
        child_records: List[dict],
        child_records_by_id: Dict[str, dict],
        parent_to_children: Dict[str, List[str]],
        bm25_index: SimpleBM25,
        dense_k: int,
        bm25_k: int,
        rrf_top_k: int,
        same_topic_negative_k: int = 8,
        random_negative_k: int = 4,
        include_source_parent_children: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List[dict]:
        if rng is None:
            rng = random.Random(42)

        pooled_retrieval: Dict[str, dict] = {}
        pooled_negatives: Dict[str, dict] = {}

        # Dense candidates
        try:
            dense_hits_raw = vectorstore.similarity_search_with_scores(query, k=dense_k,)
        except Exception:
            dense_hits_raw = vectorstore.similarity_search(query, k=dense_k)
            dense_hits_raw = [(doc, None) for doc in dense_hits_raw]

        dense_hits_for_rrf = []
        for item in dense_hits_raw:
            if isinstance(item, tuple) and len(item) == 2:
                doc, retrieval_score = item
            else:
                doc, retrieval_score = item, None

            child_id = str(doc.metadata.get("child_id") or "")
            if not child_id or child_id not in child_records_by_id:
                continue

            rec = child_records_by_id[child_id]
            dense_hits_for_rrf.append((rec, float(retrieval_score) if retrieval_score is not None else 0.0))

            pooled_retrieval[child_id] = {
                "child_id": child_id,
                "text": rec["text"],
                "title": rec["title"],
                "parent_id": rec["parent_id"],
                "source_doc_id": rec["source_doc_id"],
                "chunk_index": rec["chunk_index"],
                "dense_score": float(retrieval_score) if retrieval_score is not None else None,
                "bm25_score": None,
                "rrf_score": None,
                "rerank_score": None,
                "sources": ["dense"],
            }

        # BM25 candidates
        bm25_hits_raw = bm25_index.search(query, top_k=bm25_k)
        bm25_hits_for_rrf = []
        for rec, bm25_score in bm25_hits_raw:
            child_id = rec["child_id"]
            bm25_hits_for_rrf.append((rec, bm25_score))

            if child_id in pooled_retrieval:
                pooled_retrieval[child_id]["bm25_score"] = float(bm25_score)
                if "bm25" not in pooled_retrieval[child_id]["sources"]:
                    pooled_retrieval[child_id]["sources"].append("bm25")
            else:
                pooled_retrieval[child_id] = {
                    "child_id": child_id,
                    "text": rec["text"],
                    "title": rec["title"],
                    "parent_id": rec["parent_id"],
                    "source_doc_id": rec["source_doc_id"],
                    "chunk_index": rec["chunk_index"],
                    "dense_score": None,
                    "bm25_score": float(bm25_score),
                    "rrf_score": None,
                    "rerank_score": None,
                    
                    "sources": ["bm25"],
                }

        # RRF candidates
        rrf_hits = _rrf_fuse(dense_hits_for_rrf, bm25_hits_for_rrf, rrf_k=60)[:rrf_top_k]
        for rec, rrf_score in rrf_hits:
            child_id = rec["child_id"]
            if child_id in pooled_retrieval:
                pooled_retrieval[child_id]["rrf_score"] = float(rrf_score)
                if "rrf" not in pooled_retrieval[child_id]["sources"]:
                    pooled_retrieval[child_id]["sources"].append("rrf")
            else:
                pooled_retrieval[child_id] = {
                    "child_id": child_id,
                    "text": rec["text"],
                    "title": rec["title"],
                    "parent_id": rec["parent_id"],
                    "source_doc_id": rec["source_doc_id"],
                    "chunk_index": rec["chunk_index"],
                    "dense_score": None,
                    "bm25_score": None,
                    "rrf_score": float(rrf_score),
                    "rerank_score": None,
                    "sources": ["rrf"],
                }

        # Source parent children
        if include_source_parent_children and parent_id:
            for child_id in parent_to_children.get(parent_id, []):
                rec = child_records_by_id.get(child_id)
                if not rec:
                    continue

                if child_id in pooled_retrieval:
                    if "source_parent" not in pooled_retrieval[child_id]["sources"]:
                        pooled_retrieval[child_id]["sources"].append("source_parent")
                else:
                    pooled_retrieval[child_id] = {
                        "child_id": child_id,
                        "text": rec["text"],
                        "title": rec["title"],
                        "parent_id": rec["parent_id"],
                        "source_doc_id": rec["source_doc_id"],
                        "chunk_index": rec["chunk_index"],
                        "dense_score": None,
                        "bm25_score": None,
                        "rrf_score": None,
                        "rerank_score": None,
                        "sources": ["source_parent"],
                    }

        exclude_ids = set(pooled_retrieval.keys())

        # Same-topic negatives from other parents
        for rec in _sample_same_topic_negatives_via_overlap(
            query=query,
            parent_id=parent_id,
            child_records=child_records,
            exclude_ids=exclude_ids,
            limit=same_topic_negative_k,
        ):
            child_id = rec["child_id"]
            pooled_negatives[child_id] = {
                "child_id": child_id,
                "text": rec["text"],
                "title": rec["title"],
                "parent_id": rec["parent_id"],
                "source_doc_id": rec["source_doc_id"],
                "chunk_index": rec["chunk_index"],
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": None,
                "rerank_score": None,
                "sources": ["hard_negative_topic"],
            }
            exclude_ids.add(child_id)

        # Random negatives
        for rec in _sample_random_negatives(
            child_records=child_records,
            exclude_ids=exclude_ids,
            limit=random_negative_k,
            rng=rng,
        ):
            child_id = rec["child_id"]
            pooled_negatives[child_id] = {
                "child_id": child_id,
                "text": rec["text"],
                "title": rec["title"],
                "parent_id": rec["parent_id"],
                "source_doc_id": rec["source_doc_id"],
                "chunk_index": rec["chunk_index"],
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": None,
                "rerank_score": None,
                "sources": ["random_negative"],
            }
            exclude_ids.add(child_id)

        def sort_key(x):
            sources = set(x["sources"])
            source_bonus = 1 if "source_parent" in sources else 0
            overlap_bonus = 1 if ("dense" in sources and "bm25" in sources) else 0
            rrf_score = x["rrf_score"] if x["rrf_score"] is not None else -1e9
            dense_score = x["dense_score"] if x["dense_score"] is not None else -1e9
            bm25_score = x["bm25_score"] if x["bm25_score"] is not None else -1e9
            return (source_bonus, overlap_bonus, rrf_score, dense_score, bm25_score)

        return sorted(pooled_retrieval.values(), key=sort_key, reverse=True), sorted(pooled_negatives.values(), key = sort_key, reverse = True)

    def _rerank_candidate_pool(
        self,
        query: str,
        candidates: List[dict],
        reranker_model,
        tokenizer,
        rerank_pool_size: int = 50,
        rerank_keep_top_k: int = 20,
        batch_size: int = 8,
        max_length: int = 512,
    ) -> List[dict]:
        if not candidates:
            return []

        initial = candidates[:rerank_pool_size]
        device = next(reranker_model.parameters()).device

        pairs = []
        for c in initial:
            title = c.get("title", "") or ""
            text = c.get("text", "") or ""
            doc_text = f"{title}\n{text}" if title and text else (title or text)
            pairs.append([query, doc_text])

        scores = []

        import torch
        with Progress() as progress:
            task = progress.add_task(f"Reranking query {query} .....", total= len(pairs))
            with torch.no_grad():
                for i in range(0, len(pairs), batch_size):
                    batch_pairs = pairs[i:i + batch_size]

                    inputs = tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    ).to(device)

                    logits = reranker_model(**inputs).logits

                    if logits.ndim > 1:
                        batch_scores = logits.squeeze(-1).detach().cpu().tolist()
                    else:
                        batch_scores = logits.detach().cpu().tolist()

                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]

                    scores.extend(batch_scores)
                    progress.update(task, advance=batch_size)

        reranked = []
        for c, s in zip(initial, scores):
            row = dict(c)
            row["rerank_score"] = float(s)
            reranked.append(row)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)

        # keep reranked head + untouched tail removed
        return reranked[:rerank_keep_top_k]

    def _judge_candidates_batched(
    self,
    query: str,
    candidates: list,
    model: str,
    candidate_text_limit: int = 700,
    max_estimated_tokens: int = 4500,
) -> list:
        if not candidates:
            return []

        est = _estimate_judge_tokens(query, candidates, candidate_text_limit)

        if est <= max_estimated_tokens or len(candidates) == 1:
            log.judge(f"[green]Estimate token count =  {est}, max count = {max_estimated_tokens}[/green]")
            result = self._judge_candidates_with_ollama_realistic(
                query=query,
                candidates=candidates,
                model=model,
                candidate_text_limit=candidate_text_limit,
            )

            log.judge(
                f"Leaf judge result "
                f"| query='{query}' "
                f"| input_candidates={len(candidates)} "
                f"| returned_judgments={len(result)}"
            )

            if len(result) == 0:
                log.judge(
                    f"[red]Judge returned empty for non-empty candidate batch[/red] "
                    f"| candidate_ids={[c.get('child_id') for c in candidates[:5]]}"
                )

            return result

        mid = len(candidates) // 2
        
        log.judge(f"[purple4] Splitting documents candidates have, estimated lenght = {est}, new lenght candicates {len(candidates)} -> {mid}")

        left = self._judge_candidates_batched(
            query=query,
            candidates=candidates[:mid],
            model=model,
            candidate_text_limit=candidate_text_limit,
            max_estimated_tokens=max_estimated_tokens,
        )
        right = self._judge_candidates_batched(
            query=query,
            candidates=candidates[mid:],
            model=model,
            candidate_text_limit=candidate_text_limit,
            max_estimated_tokens=max_estimated_tokens,
        )
        return left + right


    def synthesize_beir(self, parent_chunks : list, child_chunks : list, output_dir : str, embedding_id : str, embedding_kwargs : dict, 
                        query_model : str, query_validation_model : str, judge_model : str,
                        qrel_validation_model: str, queries_per_parent_to_generate : int, max_queries_to_keep_per_parent: int,
                        dense_k : int, bm25_k : int, rrf_top_k : int, same_topic_negative_k : int, 
                        random_negative_k : int, min_score_to_keep_in_qrels : int, max_qrels_per_query : int,
                        include_source_parent_children : bool, parent_text_limit_prompt : int, candidate_text_limit_prompt : int, max_estimated_tokens : int,
                        overwrite_corpus : int, max_parents : int, random_seed : int, target_queries_per_source: int,
                        shuffle_parents : bool, reranker_id : str, rerank_pool_size : int, rerank_keep_top_k : int,
                        allow_copy_like_queries: bool = False, min_query_tokens: int = 3, max_query_tokens: int = 20, ):
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        corpus_path = output_path / "corpus.jsonl"
        queries_path = output_path / "queries.jsonl"
        qrels_dir = output_path / "qrels"
        qrels_dir.mkdir(parents=True, exist_ok=True)
        qrels_path = qrels_dir / "test.tsv"
        debug_path = output_path / "qrels_debug.jsonl"
        progress_path = output_path / "processed_parents.txt"

        rng = random.Random(random_seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        log.info(f"Using device for reranker: {device}")

        if reranker_id is not None:
            reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_id)
            reranker_model = AutoModelForSequenceClassification.from_pretrained(
                reranker_id,
                torch_dtype=dtype,
            ).to(device)
            reranker_model.eval()

        vectorstore = _build_faiss_index(child_chunks, embedding_id, embedding_kwargs,)
        child_records = _build_child_records_from_vectorstore(vectorstore)
        child_records_by_id = {r["child_id"]: r for r in child_records}

        parent_to_children = defaultdict(list)
        for rec in child_records:
            if rec["parent_id"]:
                parent_to_children[rec["parent_id"]].append(rec["child_id"])

        bm25_index = SimpleBM25(child_records)

        if overwrite_corpus or not corpus_path.exists():
            with open(corpus_path, "w+", encoding="utf-8") as f:
                for rec in child_records:
                    row = {
                        "_id": rec["child_id"],
                        "title": rec["title"],
                        "text": rec["text"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not qrels_path.exists():
            with open(qrels_path, "w+", encoding="utf-8") as f:
                f.write("query-id\tcorpus-id\tscore\n")

        seen_query_texts, query_counter = _load_existing_query_texts(queries_path)
        processed_parent_ids = _load_processed_parent_ids(progress_path)
        saved_queries_by_source = _load_saved_query_counts_by_source(debug_path)

        # Group parents by source document
        parents_by_source = defaultdict(list)
        for idx, parent in enumerate(parent_chunks):
            meta = parent.metadata or {}
            source = str(meta.get("source") or meta.get("source_file") or "unknown_source")
            parents_by_source[source].append((idx, parent))

        # Shuffle parents within each source
        if shuffle_parents:
            for source in parents_by_source:
                rng.shuffle(parents_by_source[source])

        processed_count = 0
        skipped_count = 0

        for source, source_parents in parents_by_source.items():
            if (
                target_queries_per_source is not None
                and saved_queries_by_source[source] >= target_queries_per_source
            ):
                log.info(f"Skipping source={source}, already reached cap={target_queries_per_source}")
                continue

            log.start(
                f"Starting source={source}"
                f" | parents={len(source_parents)}"
                f" | saved_queries_so_far={saved_queries_by_source[source]}"
                f"{'' if target_queries_per_source is None else f'/{target_queries_per_source}'}"
            )

            for idx, parent in source_parents:
                if max_parents is not None and processed_count >= max_parents:
                    break

                if (
                    target_queries_per_source is not None
                    and saved_queries_by_source[source] >= target_queries_per_source
                ):
                    log.end(f"Reached cap for source={source}: {target_queries_per_source}")
                    break

                parent_meta = parent.metadata or {}
                parent_id = str(
                    parent_meta.get("id")
                    or parent_meta.get("parent_id")
                    or f"parent_{idx}"
                )

                if parent_id in processed_parent_ids:
                    skipped_count += 1
                    continue

                parent_text = _normalize_text(parent.page_content)
                if not parent_text:
                    _mark_parent_processed(progress_path, parent_id)
                    continue

                try:
                    raw_queries = self._generate_realistic_query_candidates_from_parent(
                        parent_text=parent_text,
                        model=query_model,
                        n_candidates=queries_per_parent_to_generate,
                        parent_text_limit=parent_text_limit_prompt,
                    )
                except Exception as e:
                    log.info(f"Skipping parent {parent_id} due to query generation error: {e}")
                    _mark_parent_processed(progress_path, parent_id)
                    continue

                kept_queries_for_parent = 0

                for query in raw_queries:
                    if (
                        target_queries_per_source is not None
                        and saved_queries_by_source[source] >= target_queries_per_source
                    ):
                        break

                    if kept_queries_for_parent >= max_queries_to_keep_per_parent:
                        break
                    log.query(f"[blue]Validating query[/blue]: {query}")
                    normalized_query = _normalize_text(query).lower()
                    if not normalized_query:
                        log.query(f"[red]Rejected empty query [/red]| parent_id={parent_id}")
                        continue
                    if normalized_query in seen_query_texts:
                        log.query(f"[red]Rejected duplicate query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _looks_too_short(query, min_tokens=min_query_tokens):
                        log.query(f"[red]Rejected too short query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _looks_too_long(query, max_tokens=max_query_tokens):
                        log.query(f"[red]Rejected too long query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _looks_artificial(query):
                        log.query(f"[red]Rejected artificial query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if not allow_copy_like_queries and _query_is_too_copy_like(query, parent_text):
                        log.query(f"[red]Rejected copy-like query [/red]| parent_id={parent_id} | query={query}")
                        continue

                    try:
                        qv = self._validate_query_grounding(
                            query=query,
                            parent_text=parent_text,
                            model=query_validation_model,
                            parent_text_limit=parent_text_limit_prompt,
                        )
                    except Exception as e:
                        log.error(f"Query validation failed for parent={parent_id}, query='{query}': {e}")
                        continue

                    reasons = []

                    copied_from_text = qv.get("copied_from_text", False)
                    if not qv.get("keep", False) and not (allow_copy_like_queries and copied_from_text):
                        reasons.append("keep")
                    if not qv.get("grounded_in_text", False):
                        reasons.append("grounded_in_text")
                    if not qv.get("realistic_user_query", False):
                        reasons.append("realistic_user_query")
                    if copied_from_text and not allow_copy_like_queries:
                        reasons.append("copied_from_text")
                    if qv.get("too_case_specific", False):
                        reasons.append("too_case_specific")
                    if qv.get("artificial_or_benchmarky", False):
                        reasons.append("artificial_or_benchmarky")
                    if qv.get("unsupported_by_text", False):
                        reasons.append("unsupported_by_text")

                    if reasons:
                        log.query(f"[red]Rejected[/red] | reasons={reasons} | explanation={qv.get("explanation","None")}")
                        continue
                    else:
                        log.query(f"[green]Query passed validaiton explenation : [/green]{qv.get("explanation","None")}")

                    log.judge(f"Creating candidate pooling for query : {query}")
                    pooled_candidates, pooled_negatives  = self._pool_candidates_realistic(
                        query=query,
                        parent_id=parent_id,
                        vectorstore=vectorstore,
                        child_records=child_records,
                        child_records_by_id=child_records_by_id,
                        parent_to_children=parent_to_children,
                        bm25_index=bm25_index,
                        dense_k=dense_k,
                        bm25_k=bm25_k,
                        rrf_top_k=rrf_top_k,
                        same_topic_negative_k=same_topic_negative_k,
                        random_negative_k=random_negative_k,
                        include_source_parent_children=include_source_parent_children,
                        rng=rng,
                    )
                    
                    len_pooled_candidates_before = len(pooled_candidates)
                    len_pooled_negatives = len(pooled_negatives)

                    if not pooled_candidates:
                        continue
                    else:
                        candidates = _merge_dictionaries(pooled_candidates, pooled_negatives)
                    
                    if reranker_id is not None:
                        log.rerank(f"Reranking retrieved candidates, count before {len_pooled_candidates_before}")
                        pooled_candidates = self._rerank_candidate_pool(
                            query=query,
                            candidates=pooled_candidates,
                            reranker_model=reranker_model,
                            tokenizer=reranker_tokenizer,
                            rerank_pool_size=rerank_pool_size,
                            rerank_keep_top_k=rerank_keep_top_k,
                            batch_size=8,
                            max_length=512,
                        )
                        len_pooled_candidates_rerank = len(pooled_candidates)
                        candidates = _merge_dictionaries(pooled_candidates, pooled_negatives)
                        
                        log.rerank(f"Reranking finished new candidate count = {len_pooled_candidates_rerank}")

                    log.judge(
                        f"Calling judge model | query='{query}' | "
                        f"candidates={len(candidates)} | "
                        f"model={judge_model}"
                    )

                    log.judge(
                        f"Judge candidate ids | "
                        f"ids={[c.get('child_id') for c in candidates[:5]]}"
                    )


                    try:
                        len_total_candidates = len(candidates)
                        log.judge(f"Lenght total candidates = {len_total_candidates}, negatives = {len_pooled_negatives}, positieves = {len_total_candidates - len_pooled_negatives} ")
                        judgments = self._judge_candidates_batched(
                            query=query,
                            candidates=candidates,
                            model=judge_model,
                            candidate_text_limit=candidate_text_limit_prompt,
                            max_estimated_tokens=max_estimated_tokens,
                        )
                    except Exception as e:
                        log.error(f"Judge failed for parent={parent_id}, query='{query}': {e}")
                        continue

                    score_dist = Counter(j["score"] for j in judgments)

                    log.judge(
                        f"Judging complete for query='{query}' | "
                        f"total_judgments={len(judgments)} | "
                        f"score_distribution={dict(score_dist)}"
                    )  

                    judged_by_id = {j["child_id"]: j["score"] for j in judgments}
                    judgment_lookup = {j["child_id"]: j for j in judgments}

                    if not _query_has_enough_signal(
                        judged_by_id=judged_by_id,
                        min_ge2=1,
                        min_any_positive=1,
                    ):
                        continue

                    try:
                        qdist = self._judge_query_distribution_quality(
                            query=query,
                            judgments=judgments,
                            model=qrel_validation_model,
                        )
                    except Exception as e:
                        log.error(f"Qrel distribution validation failed for parent={parent_id}, query='{query}': {e}")
                        continue

                    if not qdist.get("keep", False):
                        log.query(f"[red]Rejected by distribution validation[/red] | reason=keep | qdist={qdist}")
                        continue
                    if not qdist.get("answerable", False):
                        log.query(f"[red]Rejected by distribution validation[/red] | reason=answerable | qdist={qdist}")
                        continue
                    if not qdist.get("plausible_distribution", False):
                        log.query(f"[red]Rejected by distribution validation[/red] | reason=plausible_distribution | qdist={qdist}")
                        continue

                    selected_qrels = _select_qrels_realistic(
                        pooled_candidates=candidates,
                        judged_by_id=judged_by_id,
                        min_score_to_keep=min_score_to_keep_in_qrels,
                        max_qrels_per_query=max_qrels_per_query,
                    )

                    score_histogram = _count_scores(judged_by_id)

                    log.query(
                        f"QREL selection complete "
                        f"| query='{query}' "
                        f"| score_histogram={score_histogram} "
                        f"| selected={len(selected_qrels)} "
                        f"| min_score={min_score_to_keep_in_qrels}"
                    )


                    if not selected_qrels:
                        log.query(
                            f"[red]Rejected after qrel selection: no qrels kept[/red] "
                            f"| min_score_to_keep={min_score_to_keep_in_qrels} "
                            f"| histogram={_count_scores(judged_by_id)}"
                        )
                        continue

                    query_id = f"q{query_counter}"
                    query_counter += 1
                    seen_query_texts.add(normalized_query)
                    kept_queries_for_parent += 1
                    saved_queries_by_source[source] += 1

                    _append_jsonl(queries_path, {
                        "_id": query_id,
                        "text": query,
                    })

                    with open(qrels_path, "a", encoding="utf-8") as f:
                        for row in selected_qrels:
                            f.write(f'{query_id}\t{row["corpus_id"]}\t{row["score"]}\n')

                    for row in selected_qrels:
                        judgment = judgment_lookup.get(row["corpus_id"], {})
                        _append_jsonl(debug_path, {
                            "query_id": query_id,
                            "query": query,
                            "query_source": source,
                            "query_parent_id": parent_id,
                            "corpus_id": row["corpus_id"],
                            "score": row["score"],
                            "judge_score_raw": judgment.get("score"),
                            "judge_reason": judgment.get("reason"),
                            "dense_score": row["dense_score"],
                            "bm25_score": row["bm25_score"],
                            "rerank_score": row["rerank_score"],
                            "rrf_score": row["rrf_score"],
                            "candidate_sources": row["candidate_sources"],
                            "candidate_parent_id": row["parent_id"],
                            "source_doc_id": row["source_doc_id"],
                            "chunk_index": row["chunk_index"],
                            "query_validation": qv,
                            "distribution_validation": qdist,
                            "score_histogram": _count_scores(judged_by_id),
                        })

                    log.end(
                        f"Saved query for source"
                        f" | source={source}"
                        f" | saved_for_source={saved_queries_by_source[source]}"
                        f"{'' if target_queries_per_source is None else f'/{target_queries_per_source}'}"
                        f" | parent_id={parent_id}"
                        f" | query_id={query_id}"
                    )

                _mark_parent_processed(progress_path, parent_id)
                processed_count += 1

                log.end(
                    f"Processed parent {processed_count}"
                    f" | source={source}"
                    f" | parent_id={parent_id}"
                    f" | queries_saved_for_parent={kept_queries_for_parent}"
                    f" | saved_for_source={saved_queries_by_source[source]}"
                    f"{'' if target_queries_per_source is None else f'/{target_queries_per_source}'}"
                    f"\n"
                )
                

            if max_parents is not None and processed_count >= max_parents:
                break
        log.end(f"Saved corpus to:  {corpus_path}")
        log.end(f"Saved queries to: {queries_path}")
        log.end(f"Saved qrels to:   {qrels_path}")
        log.end(f"Saved debug to:   {debug_path}")
        log.end(f"Progress file:    {progress_path}")
        log.end(f"Skipped already processed parents: {skipped_count}")
        log.end("Saved queries per source:")
        for source, count in sorted(saved_queries_by_source.items()):
            print(f"  {source}: {count}")

    # =========================
    # Load Chunks
    # =========================

    def load_chunks(self, path: Path):
        with open(path, "r+", encoding="utf-8") as f:
            data = json.load(f)

        with Progress() as progress:
            task = progress.add_task(f"Loading chunks from {path}...", total=len(data))
            docs = []
            for d in data:
                page_content = d.get("content", d.get("page_content"))
                docs.append(Document(page_content=page_content, metadata=d.get("metadata", {})))
                progress.update(task, advance=1)

        return docs

    # =========================
    # Prompt functions
    # =========================

    def load_prompt(self, path: str | Path) -> str:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def load_prompts(self, prompt_paths: dict[str, str | Path]) -> dict[str, str]:
        loaded: dict[str, str] = {}
        for name, path in prompt_paths.items():
            loaded[name] = self.load_prompt(path)
        self.prompts.update(loaded)
        return loaded

    def render_prompt(self, prompt_name: str, **kwargs) -> str:
        template = self.prompts.get(prompt_name)
        if template is None:
            raise KeyError(
                f"Prompt '{prompt_name}' not loaded. "
                f"Loaded prompts: {sorted(self.prompts.keys())}"
            )

        try:
            return template.format(**kwargs)
        except KeyError as exc:
            raise KeyError(
                f"Missing placeholder {exc} when rendering prompt '{prompt_name}'."
            ) from exc