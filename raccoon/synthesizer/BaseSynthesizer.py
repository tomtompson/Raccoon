from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

from langchain_core.documents import Document
from raccoon.dataloader import BaseLoader

import json
from pathlib import Path

import random

from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from collections import Counter, defaultdict

from raccoon.synthesizer.helper import (
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
    _count_scores


)
from raccoon.synthesizer.SimpleBM25 import SimpleBM25

import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.theme import Theme

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
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

class BaseSynthesizer(ABC):
    def __init__(
        self,
        config: dict | None = None,
        model_id: str | None = None,
        documents: list[Document | dict[str, Any]] | None = None,
    ):
        self.config = config or {}
        self.model_id = model_id
        self.documents = documents or []
        self.processed_documents: list[dict[str, Any]] = []
        self.results: list[dict[str, Any]] = []
        self.metrics: dict[str, Any] = {}

    @abstractmethod
    def _generate_realistic_query_candidates_from_parent(self):
        pass
    @abstractmethod
    def _validate_query_grounding(self):
        pass
    @abstractmethod
    def _judge_query_distribution_quality(self):
        pass

    def _pool_candidates_realistic(self):
        pass

    def _rerank_candidate_pool(self):
        pass

    def _judge_candidates_batched(self):
        pass


    def _build_faiss_index(self, child_chunks, embedding_id:str, model_kwargs: dict, encode_kwargs: dict, n_cluster:int, show_progress = False) -> FAISS:
    
        embedder = HuggingFaceEmbeddings(model_name = embedding_id, model_kwargs = model_kwargs, show_progress = show_progress)

        text = [chunk.page_content for chunk in child_chunks]

        embeddings = embedder.embed_documents(text)

        metadatas = [chunk.metadata for chunk in child_chunks]

        ids = [metadata["child_id"] for metadata in metadatas]

        text_embedding_pairs = list(zip(text, embeddings))

        X = np.array(embeddings, dtype="float32")

        clusterer = MiniBatchKMeans(
            n_clusters=n_cluster,
            random_state=42,
            batch_size=1024,
            n_init="auto",
        )

        labels = clusterer.fit_predict(X)

        for metadata, label in zip(metadatas, labels):
            metadata["cluster"] = int(label)



        vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=embedder,
            metadatas=metadatas,
            ids=ids,
        )

        return vectorstore


    def synthesize_beir(self, parent_chunks : list, vectorstore : FAISS, output_dir : str,
                        query_model : str, query_validation_model : str, judge_model : str,
                        qrel_validation_model: str, queries_per_parent_to_generate : int, max_queries_to_keep_per_parent: int,
                        dense_k : int, bm25_k : int, rrf_top_k : int, same_topic_negative_k : int, 
                        random_negative_k : int, min_score_to_keep_in_qrels : int, max_qrels_per_query : int,
                        include_source_parent_children : bool, parent_text_limit_prompt : int, candidate_text_limit_prompt : int, max_estimated_tokens : int,
                        overwrite_corpus : int, max_parents : int, random_seed : int, target_queries_per_source: int,
                        shuffle_parents : bool, reranker_model : Any, reranker_tokenizer : Any, rerank_pool_size : int, rerank_keep_top_k : int, ):
        
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

        child_records = _build_child_records_from_vectorstore(vectorstore)
        child_records_by_id = {r["child_id"]: r for r in child_records}

        parent_to_children = defaultdict(list)
        for rec in child_records:
            if rec["parent_id"]:
                parent_to_children[rec["parent_id"]].append(rec["child_id"])

        bm25_index = SimpleBM25(child_records)

        if overwrite_corpus or not corpus_path.exists():
            with open(corpus_path, "w", encoding="utf-8") as f:
                for rec in child_records:
                    row = {
                        "_id": rec["child_id"],
                        "title": rec["title"],
                        "text": rec["text"],
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if not qrels_path.exists():
            with open(qrels_path, "w", encoding="utf-8") as f:
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
                    if _looks_too_short(query, min_tokens=3):
                        log.query(f"[red]Rejected too short query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _looks_too_long(query, max_tokens=20):
                        log.query(f"[red]Rejected too long query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _looks_artificial(query):
                        log.query(f"[red]Rejected artificial query [/red]| parent_id={parent_id} | query={query}")
                        continue
                    if _query_is_too_copy_like(query, parent_text):
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

                    if not qv.get("keep", False):
                        reasons.append("keep")
                    if not qv.get("grounded_in_text", False):
                        reasons.append("grounded_in_text")
                    if not qv.get("realistic_user_query", False):
                        reasons.append("realistic_user_query")
                    if qv.get("copied_from_text", False):
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

                    if reranker_model is not None and reranker_tokenizer is not None:
                        log.rerank(f"Reranking retrieved candidates, count before {len_pooled_candidates_before}")
                        pooled_candidates = _rerank_candidate_pool(
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
                    len_total_candidates = len(candidates)
                    log.rerank(f"Reranking finished new candidate count = {len_pooled_candidates_rerank}, "
                            f"negatives = {len_pooled_negatives}, "
                            f"total candidate count = {len_total_candidates}")
                    try:
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
                        continue
                    if not qdist.get("answerable", False):
                        continue
                    if not qdist.get("plausible_distribution", False):
                        continue

                    selected_qrels = _select_qrels_realistic(
                        pooled_candidates=candidates,
                        judged_by_id=judged_by_id,
                        min_score_to_keep=min_score_to_keep_in_qrels,
                        max_qrels_per_query=max_qrels_per_query,
                    )

                    if not selected_qrels:
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