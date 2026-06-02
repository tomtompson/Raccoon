from .BaseRetriever import BaseRetriever
from raccoon.custom_retriever.util.Reranker import Reranker
from typing import Any, Dict, Set, List, Optional, Tuple
from raccoon.custom_retriever.util.SimpleBM25 import SimpleBM25
from raccoon.custom_retriever.util.InMemoryEmbeddingStore import EmbeddingStore
from raccoon.custom_retriever.util.ConceptExtractor import ConceptExtractor
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
from pathlib import Path

import os
import faiss
import pickle
import time
import torch
import json 
import hashlib
import numpy as np
import igraph as ig
import re
from raccoon.logging_utils import get_logger
from .util.utils import (join_title_text, min_max_normalize, rrf_fuse)

log = get_logger(__name__)


class LinearRagRetriever(BaseRetriever):
    retriever_type = "linear"
    CACHE_SCHEMA_VERSION = 1
    def __init__(
        self,
        config: Dict[str, Any] | None = None,
        corpus: Dict | None = None,
        queries: Dict | None = None,
        reranker: Reranker | None = None
    ) -> None:
        self.config = config or {}
        self.corpus = corpus
        self.queries = queries
        self.processed_documents: list = []
        self.results: list[Dict[str, Any]] = []
        self.is_ready = False
        self.reranker = reranker
        self.embedding_model = None
        self.extractor = None
        self.passage_store = None
        self.concept_store = None
        self.sentence_store = None
        self.bm25 = None
        self.use_gpu_for_spacy = torch.cuda.is_available() and self.config.get("use_gpu_for_spacy", True)
        self.query_graph_path = None
        self.query_graph_query_id = None
        self.query_graph_query = None
        super().__init__(config, corpus, queries, reranker)
    def create_index(self, *args, **kwargs) -> None:
        raise NotImplementedError("Linear retriever does not support create_index().")
    def encode(self, *args, **kwargs):
         raise NotImplementedError("Linear retriever does not support encode().")
    
    def index_corpus(self, *args, **kwargs) -> None:
        index_start = time.perf_counter()
        model_load_start = time.perf_counter()
        self._load_embedding_model()
        model_load_latency = time.perf_counter() - model_load_start

        cache_load_start = time.perf_counter()
        if self._load_embedding_stores():
            cache_load_latency = time.perf_counter() - cache_load_start
            self.metrics["index_time"] = {
                "indexing": self._index_metrics_payload(
                    time_in_seconds=time.perf_counter() - index_start,
                    cache_loaded=True,
                    phase_times={
                        "model_load_time_in_seconds": model_load_latency,
                        "cache_load_time_in_seconds": cache_load_latency,
                    },
                )
            }
            return
        cache_load_latency = time.perf_counter() - cache_load_start
        
        log.info("Preparing passages")

        self.doc_ids = list(self.corpus.keys())
        self.doc_texts = {doc_id: join_title_text(self.corpus[doc_id]) for doc_id in self.doc_ids}

        log.info("Building local BM25 index")
        bm25_start = time.perf_counter()
        self.bm25 = SimpleBM25()
        self.bm25.build(self.doc_texts)
        bm25_latency = time.perf_counter() - bm25_start

        log.info("Running spaCy concept extraction")
        concept_start = time.perf_counter()
        self.extractor = ConceptExtractor(
            self.config.get("spacy_model_name", "nl_core_news_sm"),
            use_gpu=self.use_gpu_for_spacy and torch.cuda.is_available(),
            min_concept_len=self.config.get("min_concept_len", 6),
            max_concept_words=self.config.get("max_concept_words", 4),
            stop_concept=self.config.get("stop_concept",[])
        )

        passage_id_to_concepts: Dict[str, Set[str]] = {}
        sentence_to_concepts: Dict[str, Set[str]] = defaultdict(set)
        concept_to_sentences: Dict[str, Set[str]] = defaultdict(set)
        sentence_to_passage_ids: Dict[str, Set[str]] = defaultdict(set)
        all_sentences: Set[str] = set()

        items = [(doc_id, self.doc_texts[doc_id]) for doc_id in self.doc_ids]
        total = len(items)
        total_start_time = time.perf_counter()

        ner_batch_size = self.config.get("ner_batch_size", 64)
        next_print = 1000
        for start in range(0, total, ner_batch_size):
            batch_start_time = time.perf_counter()
            batch = items[start:start + ner_batch_size]
            batch_doc_ids = [doc_id for doc_id, _ in batch]
            batch_texts = [text for _, text in batch]

            docs = list(self.extractor.nlp.pipe(batch_texts, batch_size=ner_batch_size))

            for doc_id, doc in zip(batch_doc_ids, docs):
                passage_concepts, sent_to_concepts, sentence_list = self.extractor.extract_from_doc(doc)
                passage_id_to_concepts[doc_id] = set(passage_concepts)

                for sent in sentence_list:
                    all_sentences.add(sent)
                    sentence_to_passage_ids[sent].add(doc_id)

                for sent, concepts in sent_to_concepts.items():
                    sentence_to_concepts[sent].update(concepts)
                    for concept in concepts:
                        concept_to_sentences[concept].add(sent)

            done = min(start + ner_batch_size, total)

            if done >= next_print or done == total:
                log.info(
                    f"Concept extraction: {done}/{total}, "
                    f"batch time={time.perf_counter() - batch_start_time:.3f}s, "
                    f"total time={time.perf_counter() - total_start_time:.3f}s"
                )
                next_print += 1000

            del batch, batch_doc_ids, batch_texts, docs

        del items

        # Filter high-frequency and low-frequency concepts.
        concept_df = Counter()
        for _, concepts in passage_id_to_concepts.items():
            for c in concepts:
                concept_df[c] += 1

        total_docs = max(1, len(self.doc_ids))
        valid_concepts = {
            c for c, df in concept_df.items()

            if df >= self.config.get("min_concept_df", 1) and (df / total_docs) <= self.config.get("max_concept_df_ratio", 0.30)
        }

        log.info("Concepts before filtering: %d", len(concept_df))
        log.info("Concepts after filtering: %d", len(valid_concepts))

        passage_id_to_concepts = {
            doc_id: {c for c in concepts if c in valid_concepts}
            for doc_id, concepts in passage_id_to_concepts.items()
        }
        sentence_to_concepts = {
            sent: {c for c in concepts if c in valid_concepts}
            for sent, concepts in sentence_to_concepts.items()
        }
        concept_to_sentences = defaultdict(set)
        for sent, concepts in sentence_to_concepts.items():
            for c in concepts:
                concept_to_sentences[c].add(sent)
        concept_latency = time.perf_counter() - concept_start

        # Unload spaCy from GPU before embedding phase.
        log.info("Unloading spaCy before embedding phase")
        del self.extractor
        self.extractor = None

        concept_texts = sorted(valid_concepts)
        sentence_texts = sorted(all_sentences)

        self.passage_store = EmbeddingStore(
            self.embedding_model,
            self.config.get("embed_batch_size", 192),
            backend="faiss",
        )

        self.concept_store = EmbeddingStore(
            self.embedding_model,
            self.config.get("embed_batch_size", 192),
            backend="memory",
        )

        self.sentence_store = EmbeddingStore(
            self.embedding_model,
            self.config.get("embed_batch_size", 192),
            backend="memory",
        )

        log.info("Num docs: %d", len(self.doc_ids))
        log.info("Num concepts: %d", len(concept_texts))
        log.info("Num sentences: %d", len(sentence_texts))

        log.info("Encoding passages")
        embedding_start = time.perf_counter()
        self.passage_store.build(
            [(doc_id, self.doc_texts[doc_id]) for doc_id in self.doc_ids],
            prefix_text="passage",
        )

        log.info("Encoding concepts")
        self.concept_store.build(
            [(f"concept::{i}", concept) for i, concept in enumerate(concept_texts)],
            prefix_text="entity",
        )

        log.info("Encoding sentences")
        self.sentence_store.build(
            [(f"sent::{i}", sent) for i, sent in enumerate(sentence_texts)],
            prefix_text="sentence",
        )
        embedding_latency = time.perf_counter() - embedding_start

        log.info("Reloading spaCy on CPU for query concept extraction")
        self.extractor = ConceptExtractor(self.config.get("spacy_model_name", "nl_core_news_sm"), use_gpu=self.use_gpu_for_spacy and torch.cuda.is_available(),
                                            min_concept_len=self.config.get("min_concept_len", 6),
                                            max_concept_words=self.config.get("max_concept_words", 4),
                                            stop_concept=self.config.get("stop_concept",[]))

        self.concept_text_to_id = {
            text: item_id for item_id, text in self.concept_store.id_to_text.items()
        }
        self.sentence_text_to_id = {
            text: item_id for item_id, text in self.sentence_store.id_to_text.items()
        }

        self.passage_id_to_concept_ids: Dict[str, List[str]] = {
            doc_id: [self.concept_text_to_id[c] for c in concepts if c in self.concept_text_to_id]
            for doc_id, concepts in passage_id_to_concepts.items()
        }
        self.concept_id_to_sentence_ids: Dict[str, List[str]] = {
            self.concept_text_to_id[c]: [
                self.sentence_text_to_id[s]
                for s in sents
                if s in self.sentence_text_to_id
            ]
            for c, sents in concept_to_sentences.items()
            if c in self.concept_text_to_id
        }
        self.sentence_id_to_concept_ids: Dict[str, List[str]] = {
            self.sentence_text_to_id[s]: [
                self.concept_text_to_id[c]
                for c in concepts
                if c in self.concept_text_to_id
            ]
            for s, concepts in sentence_to_concepts.items()
            if s in self.sentence_text_to_id
        }
        self.sentence_id_to_passage_ids: Dict[str, List[str]] = {
            self.sentence_text_to_id[s]: list(pids)
            for s, pids in sentence_to_passage_ids.items()
            if s in self.sentence_text_to_id
        }

        self.concept_id_to_passage_ids: Dict[str, Set[str]] = defaultdict(set)
        for doc_id, concept_ids in self.passage_id_to_concept_ids.items():
            for cid in concept_ids:
                self.concept_id_to_passage_ids[cid].add(doc_id)

        del concept_texts, sentence_texts

        save_start = time.perf_counter()
        self._save_embedding_stores()
        cache_save_latency = time.perf_counter() - save_start
        self.metrics["index_time"] = {
            "indexing": self._index_metrics_payload(
                time_in_seconds=time.perf_counter() - index_start,
                cache_loaded=False,
                phase_times={
                    "model_load_time_in_seconds": model_load_latency,
                    "cache_lookup_time_in_seconds": cache_load_latency,
                    "bm25_build_time_in_seconds": bm25_latency,
                    "concept_extraction_time_in_seconds": concept_latency,
                    "embedding_build_time_in_seconds": embedding_latency,
                    "cache_save_time_in_seconds": cache_save_latency,
                },
            )
        }


    def search(self, top_k: int, *args, **kwargs) -> dict:
        search_start = time.perf_counter()
        results = {}
        items = list(self.queries.items())
        total_queries = len(items)
        resolved_top_k = top_k or self.config.get("retrieval_top_k", 100)

        for i, (qid, query) in enumerate(items, start=1):
            ranked_doc_ids, ranked_scores, g, graph_scores = self._search_once(
                query,
                top_k=resolved_top_k
            )
            results[qid] = {
                doc_id: float(score)
                for doc_id, score in zip(ranked_doc_ids, ranked_scores)
            }

            if (
                self.config.get("export_query_graph", False)
                and self.query_graph_path is None
            ):
                self.query_graph_path = self.export_query_graph_png(
                    query=query,
                    g=g,
                    graph_scores=graph_scores,
                    max_nodes=self.config.get("query_graph_max_nodes", 35),
                    output_dir=self.config.get("query_graph_output_dir", "report_graphs"),
                )
                self.query_graph_query_id = qid
                self.query_graph_query = query

            if i == 1 or i % 10 == 0 or i == total_queries:
                log.info("Retrieving queries: %d/%d", i, total_queries)
        self.results = results
        search_latency = time.perf_counter() - search_start
        result_counts = [len(row) for row in results.values()]
        scores = [score for row in results.values() for score in row.values()]
        total_results = sum(result_counts)
        self.metrics["query_time"] = {
            "search": {
                "time_in_seconds": search_latency,
                "time_per_query_in_seconds": search_latency / total_queries if total_queries else 0.0,
                "queries_per_second": total_queries / search_latency if search_latency else 0.0,
                "queries": total_queries,
                "top_k": resolved_top_k,
                "dense_candidate_k": self.config.get("dense_candidate_k", 500),
                "bm25_candidate_k": self.config.get("bm25_candidate_k", 500),
                "graph_candidate_k": self.config.get("graph_candidate_k", 500),
                "dense_rrf_weight": self.config.get("dense_rrf_weight", 1),
                "bm25_rrf_weight": self.config.get("bm25_rrf_weight", 0.35),
                "graph_rrf_weight": self.config.get("graph_rrf_weight", 0.75),
                "rrf_k": self.config.get("rrf_k", 60),
                "total_results": total_results,
                "avg_results_per_query": total_results / total_queries if total_queries else 0.0,
                "min_results_per_query": min(result_counts) if result_counts else 0,
                "max_results_per_query": max(result_counts) if result_counts else 0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "avg_score": sum(scores) / len(scores) if scores else 0.0,
            },
        }
        if self.reranker is not None:
            self.store_rerank_results()
        return results

    def _search_once(self, question: str, top_k: Optional[int] = None) -> Tuple[List[str], List[float], Any, Tuple[str,float]]:
        if top_k is None:
            top_k = self.config.get("retrieval_top_k", 100)

        query_embedding = self._encode_query(question)

        dense = self.dense_ranking(query_embedding, top_k=self.config.get("dense_candidate_k", 500))
        bm25 = self.bm25_ranking(question, top_k=self.config.get("bm25_candidate_k", 500))
        graph, g = self.graph_ranking(
            query_embedding=query_embedding,
            dense_ranking=dense,
            bm25_ranking=bm25,
            question=question,
            top_k=self.config.get("graph_candidate_k", 500),
        )

        dense_ids = [doc_id for doc_id, _ in dense]
        bm25_ids = [doc_id for doc_id, _ in bm25]
        graph_ids = [doc_id for doc_id, _ in graph]

        fused = rrf_fuse(
            [
                (dense_ids, self.config.get("dense_rrf_weight", 1)),
                (bm25_ids, self.config.get("bm25_rrf_weight", 0.35)),
                (graph_ids, self.config.get("graph_rrf_weight", 0.75)),
            ],
            k=self.config.get("rrf_k", 60),
        )

        ranked = list(fused.items())[:top_k]
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        ranked_scores = [float(score) for _, score in ranked]

        return ranked_doc_ids, ranked_scores, g, graph
    
    def _encode_query(self, question: str) -> np.ndarray:
        return self.embedding_model.encode(
            [f"query: {question}"],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0].astype(np.float32)
    

    def dense_ranking(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        indices, scores = self.passage_store.search(query_embedding, top_k=top_k)
        ranking = []
        for idx, score in zip(indices, scores):
            if idx < 0:
                continue
            doc_id = self.passage_store.ids[int(idx)]
            ranking.append((doc_id, float(score)))
        return ranking

    def bm25_ranking(self, question: str, top_k: int) -> List[Tuple[str, float]]:
        return self.bm25.search(question, top_k=top_k)

    def get_query_concept_seed_ids(self, question: str, query_embedding: np.ndarray) -> Dict[str, float]:
        seed_scores: Dict[str, float] = {}

        query_concepts = self.extractor.extract_query_concepts(question) if self.extractor else set()

        # Exact concept matches from extracted query concepts.
        for concept in query_concepts:
            cid = self.concept_text_to_id.get(concept)
            if cid:
                seed_scores[cid] = max(seed_scores.get(cid, 0.0), 1.0)

        # Exact substring matches. This catches phrases that spaCy may not extract from the query.
        q_lower = question.lower()
        for concept_text, cid in self.concept_text_to_id.items():
            if concept_text in q_lower:
                seed_scores[cid] = max(seed_scores.get(cid, 0.0), 1.0)

        # Embedding concept matches.
        if self.concept_store.embeddings is not None and len(self.concept_store.ids) > 0:
            sims = np.dot(self.concept_store.embeddings, query_embedding.reshape(-1, 1)).flatten()
            top_idx = np.argsort(sims)[::-1][:self.config.get("local_graph_concept_seed_k", 80)]
            for idx in top_idx:
                score = float(sims[int(idx)])
                if score < 0.35:
                    continue
                cid = self.concept_store.ids[int(idx)]
                seed_scores[cid] = max(seed_scores.get(cid, 0.0), score)

        return seed_scores

    def get_query_sentence_seed_ids(self, query_embedding: np.ndarray) -> Dict[str, float]:
        seed_scores: Dict[str, float] = {}
        if self.sentence_store.embeddings is None or len(self.sentence_store.ids) == 0:
            return seed_scores

        sims = np.dot(self.sentence_store.embeddings, query_embedding.reshape(-1, 1)).flatten()
        top_idx = np.argsort(sims)[::-1][:self.config.get("local_graph_sentence_seed_k", 80)]

        for idx in top_idx:
            score = float(sims[int(idx)])
            if score < 0.35:
                continue
            sid = self.sentence_store.ids[int(idx)]
            seed_scores[sid] = score

        return seed_scores

    def build_local_graph(
        self,
        dense_docs: List[str],
        bm25_docs: List[str],
        seed_concepts: Dict[str, float],
        seed_sentences: Dict[str, float],
    ) -> Tuple[ig.Graph, Dict[str, int], Set[str]]:
        """
        Builds a local graph with only query-relevant passages, concepts, and sentences.
        This is much less noisy than global personalized PageRank.
        """
        candidate_passages: Set[str] = set(dense_docs) | set(bm25_docs)
        candidate_concepts: Set[str] = set(seed_concepts.keys())
        candidate_sentences: Set[str] = set(seed_sentences.keys())

        # Expand concepts from candidate passages.
        for doc_id in list(candidate_passages):
            for cid in self.passage_id_to_concept_ids.get(doc_id, []):
                candidate_concepts.add(cid)

        # Expand sentences from seed concepts.
        for cid in list(seed_concepts.keys()):
            for sid in self.concept_id_to_sentence_ids.get(cid, [])[:200]:
                candidate_sentences.add(sid)
                for doc_id in self.sentence_id_to_passage_ids.get(sid, []):
                    candidate_passages.add(doc_id)

        # Expand passages from seed sentences.
        for sid in list(seed_sentences.keys()):
            for doc_id in self.sentence_id_to_passage_ids.get(sid, []):
                candidate_passages.add(doc_id)
            for cid in self.sentence_id_to_concept_ids.get(sid, []):
                candidate_concepts.add(cid)

        # Expand passages from seed concepts directly.
        for cid in list(seed_concepts.keys()):
            for doc_id in self.concept_id_to_passage_ids.get(cid, set()):
                candidate_passages.add(doc_id)

        g = ig.Graph(directed=False)
        node_names: List[str] = []

        for doc_id in candidate_passages:
            node_names.append(doc_id)
        for cid in candidate_concepts:
            node_names.append(cid)
        for sid in candidate_sentences:
            node_names.append(sid)

        node_names = list(dict.fromkeys(node_names))
        g.add_vertices(node_names)
        name_to_idx = {name: i for i, name in enumerate(node_names)}

        node_type = []
        for name in node_names:
            if name.startswith("concept::"):
                node_type.append("concept")
            elif name.startswith("sent::"):
                node_type.append("sentence")
            else:
                node_type.append("passage")
        g.vs["node_type"] = node_type

        edges = []
        weights = []

        # Passage-concept edges.
        for doc_id in candidate_passages:
            if doc_id not in name_to_idx:
                continue
            concept_ids = self.passage_id_to_concept_ids.get(doc_id, [])
            if not concept_ids:
                continue

            doc_text = self.doc_texts.get(doc_id, "").lower()
            counts = Counter()
            for cid in concept_ids:
                if cid not in candidate_concepts or cid not in name_to_idx:
                    continue
                concept_text = self.concept_store.id_to_text[cid].lower()
                c = doc_text.count(concept_text)
                if c > 0:
                    counts[cid] = c

            if not counts:
                continue

            max_tf = max(counts.values())
            for cid, tf in counts.items():
                tf_norm = tf / max_tf
                weight = 0.75 + tf_norm
                edges.append((name_to_idx[doc_id], name_to_idx[cid]))
                weights.append(float(weight))

        # Sentence-passage edges.
        for sid in candidate_sentences:
            if sid not in name_to_idx:
                continue
            for doc_id in self.sentence_id_to_passage_ids.get(sid, []):
                if doc_id in name_to_idx:
                    edges.append((name_to_idx[sid], name_to_idx[doc_id]))
                    weights.append(1.0)

        # Sentence-concept edges.
        for sid in candidate_sentences:
            if sid not in name_to_idx:
                continue
            for cid in self.sentence_id_to_concept_ids.get(sid, []):
                if cid in name_to_idx:
                    edges.append((name_to_idx[sid], name_to_idx[cid]))
                    weights.append(1.0)

        if edges:
            g.add_edges(edges)
            g.es["weight"] = weights
        else:
            g.es["weight"] = []

        return g, name_to_idx, candidate_passages

    def graph_ranking(
        self,
        query_embedding: np.ndarray,
        dense_ranking: List[Tuple[str, float]],
        bm25_ranking: List[Tuple[str, float]],
        question: str,
        top_k: int,
    ) ->    Tuple[List[Tuple[str, float]], ig.Graph]:
        dense_docs = [doc_id for doc_id, _ in dense_ranking[:self.config.get("local_graph_dense_seed_k", 200)]]
        bm25_docs = [doc_id for doc_id, _ in bm25_ranking[:self.config.get("local_graph_bm25_seed_k", 200)]]

        seed_concepts = self.get_query_concept_seed_ids(question, query_embedding)
        seed_sentences = self.get_query_sentence_seed_ids(query_embedding)

        if not dense_docs and not bm25_docs and not seed_concepts and not seed_sentences:
            return [], ig.Graph()

        g, name_to_idx, candidate_passages = self.build_local_graph(
            dense_docs=dense_docs,
            bm25_docs=bm25_docs,
            seed_concepts=seed_concepts,
            seed_sentences=seed_sentences,
        )

        if g.vcount() == 0:
            return [], g

        reset = np.zeros(g.vcount(), dtype=np.float64)

        # Dense passage seeds.
        dense_scores = np.array([score for _, score in dense_ranking[:self.config.get("local_graph_dense_seed_k", 200)]], dtype=np.float32)
        dense_scores_norm = min_max_normalize(dense_scores) if dense_scores.size else dense_scores
        for (doc_id, _), score in zip(dense_ranking[:self.config.get("local_graph_dense_seed_k", 200)], dense_scores_norm):
            if doc_id in name_to_idx:
                reset[name_to_idx[doc_id]] += 1.00 * float(score)

        # BM25 passage seeds.
        bm25_scores = np.array([score for _, score in bm25_ranking[:self.config.get("local_graph_bm25_seed_k", 200)]], dtype=np.float32)
        bm25_scores_norm = min_max_normalize(bm25_scores) if bm25_scores.size else bm25_scores
        for (doc_id, _), score in zip(bm25_ranking[:self.config.get("local_graph_bm25_seed_k", 200)], bm25_scores_norm):
            if doc_id in name_to_idx:
                reset[name_to_idx[doc_id]] += 0.80 * float(score)

        # Concept seeds.
        for cid, score in seed_concepts.items():
            if cid in name_to_idx:
                reset[name_to_idx[cid]] += 0.70 * float(score)

        # Sentence seeds.
        for sid, score in seed_sentences.items():
            if sid in name_to_idx:
                reset[name_to_idx[sid]] += 0.35 * float(score)

        if reset.sum() <= 0:
            return [], g

        try:
            scores = g.personalized_pagerank(
                vertices=range(g.vcount()),
                damping=self.config.get("ppr_damping", 0.6),
                directed=False,
                weights="weight" if g.ecount() > 0 else None,
                reset=reset,
                implementation="prpack",
            )
        except Exception:
            scores = g.personalized_pagerank(
                vertices=range(g.vcount()),
                damping=self.config.get("ppr_damping", 0.6),
                directed=False,
                weights="weight" if g.ecount() > 0 else None,
                reset=reset,
            )
        
        g.vs["pagerank_score"] = [float(score) for score in scores]
        
        doc_scores = []
        for doc_id in candidate_passages:
            idx = name_to_idx.get(doc_id)
            if idx is not None:
                doc_scores.append((doc_id, float(scores[idx])))

        return sorted(doc_scores, key=lambda x: x[1], reverse=True)[:top_k], g

    def _load_embedding_model(self):
        if self.embedding_model is not None:
            return

        device = self.config.get(
            "device",
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_name = self.config.get(
            "embedding_model_name",
            "snowflake/snowflake-arctic-embed-l-v2.0"
        )

        log.info("Loading embedding model on %s", device)

        self.embedding_model = SentenceTransformer(
            model_name,
            device=str(device),
            trust_remote_code=True,
        )

        max_seq_length = self.config.get("max_seq_length")
        if max_seq_length:
            self.embedding_model.max_seq_length = max_seq_length

        if str(device).startswith("cuda"):
            self.embedding_model = self.embedding_model.half()
            torch.backends.cuda.matmul.allow_tf32 = True

    def _unload_embedding_model(self):
        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None

    def _get_cache_path(self):
        dataset_name = self.config.get("dataset_name", "default_dataset")
        model_name = self.config.get(
            "embedding_model_name",
            "snowflake/snowflake-arctic-embed-l-v2.0"
        )

        cache_config = {
            "dataset_name": dataset_name,
            "dense_model": model_name,
            "max_seq_length": self.config.get("max_seq_length"),
            "min_concept_len": self.config.get("min_concept_len"),
            "max_concept_words": self.config.get("max_concept_words"),
            "min_concept_df": self.config.get("min_concept_df"),
            "max_concept_df_ratio": self.config.get("max_concept_df_ratio"),
        }

        cache_suffix = hashlib.md5(
            json.dumps(cache_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:10]

        safe_dataset = str(dataset_name).replace("/", "-").replace(" ", "-")
        safe_model = str(model_name).split("/")[-1]

        cache_root = self.config.get("cache_path", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache") )
        path = os.path.join(cache_root, f"{safe_model}-{safe_dataset}-{cache_suffix}")

        os.makedirs(path, exist_ok=True)
        return path

    def _cache_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.CACHE_SCHEMA_VERSION,
            "embedding_model": self.config.get("embedding_model_name", "snowflake/snowflake-arctic-embed-l-v2.0"),
            "spacy_model": self.config.get("spacy_model_name", "nl_core_news_sm"),
            "dataset_name": self.config.get("dataset_name", "default_dataset"),
            "max_seq_length": self.config.get("max_seq_length"),
            "min_concept_len": self.config.get("min_concept_len"),
            "max_concept_words": self.config.get("max_concept_words"),
            "min_concept_df": self.config.get("min_concept_df"),
            "max_concept_df_ratio": self.config.get("max_concept_df_ratio"),
        }

    def _cache_size_bytes(self) -> int:
        path = self._get_cache_path()
        total_size = 0
        for root, _, files in os.walk(path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
        return total_size

    def _index_metrics_payload(
        self,
        *,
        time_in_seconds: float,
        cache_loaded: bool,
        phase_times: Dict[str, float] | None = None,
    ) -> Dict[str, Any]:
        passage_count = len(getattr(self, "doc_ids", []) or [])
        concept_count = len(getattr(getattr(self, "concept_store", None), "ids", []) or [])
        sentence_count = len(getattr(getattr(self, "sentence_store", None), "ids", []) or [])
        passage_store = getattr(self, "passage_store", None)
        passage_index = getattr(passage_store, "index", None)
        embedding_dim = int(getattr(passage_store, "dim", 0) or getattr(passage_index, "d", 0) or 0)
        cache_size_bytes = self._cache_size_bytes()
        passage_concept_links = sum(len(v) for v in getattr(self, "passage_id_to_concept_ids", {}).values())
        concept_sentence_links = sum(len(v) for v in getattr(self, "concept_id_to_sentence_ids", {}).values())
        sentence_passage_links = sum(len(v) for v in getattr(self, "sentence_id_to_passage_ids", {}).values())

        return {
            "time_in_seconds": time_in_seconds,
            "documents": passage_count,
            "concepts": concept_count,
            "sentences": sentence_count,
            "concepts_per_document": concept_count / passage_count if passage_count else 0.0,
            "sentences_per_document": sentence_count / passage_count if passage_count else 0.0,
            "passage_concept_links": passage_concept_links,
            "concept_sentence_links": concept_sentence_links,
            "sentence_passage_links": sentence_passage_links,
            "embedding_dim": embedding_dim,
            "size_in_bytes": cache_size_bytes,
            "size_in_mb": cache_size_bytes / (1024 * 1024),
            "docs_per_second": passage_count / time_in_seconds if time_in_seconds else 0.0,
            "cache_loaded": cache_loaded,
            "cache_path": self._get_cache_path(),
            "embedding_model": self.config.get("embedding_model_name", "snowflake/snowflake-arctic-embed-l-v2.0"),
            "spacy_model": self.config.get("spacy_model_name", "nl_core_news_sm"),
            "backend": "faiss",
            "phase_times": phase_times or {},
        }

    def _save_embedding_stores(self):
        path = self._get_cache_path()

        # FAISS passage index
        if self.passage_store is not None:
            index = self.passage_store.index

            # If index is on GPU, move to CPU before saving
            try:
                index = faiss.index_gpu_to_cpu(index)
            except Exception:
                pass

            faiss.write_index(index, os.path.join(path, "passage.index"))

            with open(os.path.join(path, "passage_store_meta.pkl"), "wb") as f:
                pickle.dump({
                    "ids": self.passage_store.ids,
                    "id_to_text": self.passage_store.id_to_text,
                    "metadata": getattr(self.passage_store, "metadata", None),
                }, f)

        # In-memory stores
        if self.concept_store is not None:
            with open(os.path.join(path, "concept_store.pkl"), "wb") as f:
                pickle.dump(self.concept_store, f)

        if self.sentence_store is not None:
            with open(os.path.join(path, "sentence_store.pkl"), "wb") as f:
                pickle.dump(self.sentence_store, f)

        # BM25
        if self.bm25 is not None:
            with open(os.path.join(path, "bm25.pkl"), "wb") as f:
                pickle.dump(self.bm25, f)

        # Corpus / document text
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump({
                "corpus": self.corpus,
                "doc_ids": self.doc_ids,
                "doc_texts": self.doc_texts,
            }, f)

        # Graph mappings
        with open(os.path.join(path, "graph_mappings.pkl"), "wb") as f:
            pickle.dump({
                "concept_text_to_id": self.concept_text_to_id,
                "sentence_text_to_id": self.sentence_text_to_id,
                "passage_id_to_concept_ids": self.passage_id_to_concept_ids,
                "concept_id_to_sentence_ids": self.concept_id_to_sentence_ids,
                "sentence_id_to_concept_ids": self.sentence_id_to_concept_ids,
                "sentence_id_to_passage_ids": self.sentence_id_to_passage_ids,
                "concept_id_to_passage_ids": dict(self.concept_id_to_passage_ids),
            }, f)

        with open(os.path.join(path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(self._cache_manifest(), f, ensure_ascii=False, indent=2)

        log.info("[CACHE] Saved embedding cache to %s", path)

    def _load_embedding_stores(self):
        path = self._get_cache_path()

        required_files = [
            "passage.index",
            "passage_store_meta.pkl",
            "concept_store.pkl",
            "sentence_store.pkl",
            "bm25.pkl",
            "documents.pkl",
            "graph_mappings.pkl",
            "manifest.json",
        ]

        missing = [
            file for file in required_files
            if not os.path.exists(os.path.join(path, file))
        ]

        if missing:
            log.info("[CACHE] Missing cache files: %s", missing)
            return False

        try:
            with open(os.path.join(path, "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
            expected_manifest = self._cache_manifest()
            if manifest != expected_manifest:
                log.info("[CACHE] Manifest mismatch for %s", path)
                return False

            # Documents
            with open(os.path.join(path, "documents.pkl"), "rb") as f:
                docs = pickle.load(f)

            self.corpus = docs["corpus"]
            self.doc_ids = docs["doc_ids"]
            self.doc_texts = docs["doc_texts"]

            # FAISS passage store
            passage_index = faiss.read_index(os.path.join(path, "passage.index"))

            with open(os.path.join(path, "passage_store_meta.pkl"), "rb") as f:
                passage_meta = pickle.load(f)

            self.passage_store = EmbeddingStore(
                self.embedding_model,
                self.config.get("embed_batch_size", 128),
                backend="faiss",
            )

            self.passage_store.index = passage_index
            self.passage_store.ids = passage_meta["ids"]
            self.passage_store.id_to_text = passage_meta["id_to_text"]

            if passage_meta.get("metadata") is not None:
                self.passage_store.metadata = passage_meta["metadata"]

            # Concept store
            with open(os.path.join(path, "concept_store.pkl"), "rb") as f:
                self.concept_store = pickle.load(f)

            # Sentence store
            with open(os.path.join(path, "sentence_store.pkl"), "rb") as f:
                self.sentence_store = pickle.load(f)

            # BM25
            with open(os.path.join(path, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)

            # Graph mappings
            with open(os.path.join(path, "graph_mappings.pkl"), "rb") as f:
                mappings = pickle.load(f)

            self.concept_text_to_id = mappings["concept_text_to_id"]
            self.sentence_text_to_id = mappings["sentence_text_to_id"]
            self.passage_id_to_concept_ids = mappings["passage_id_to_concept_ids"]
            self.concept_id_to_sentence_ids = mappings["concept_id_to_sentence_ids"]
            self.sentence_id_to_concept_ids = mappings["sentence_id_to_concept_ids"]
            self.sentence_id_to_passage_ids = mappings["sentence_id_to_passage_ids"]
            self.concept_id_to_passage_ids = defaultdict(
                set,
                {
                    k: set(v)
                    for k, v in mappings["concept_id_to_passage_ids"].items()
                }
            )

            # Needed for query concept extraction
            self.extractor = ConceptExtractor(
                self.config.get("spacy_model_name","nl_core_news_sm"),
                use_gpu=self.use_gpu_for_spacy and torch.cuda.is_available(),
                min_concept_len=self.config.get("min_concept_len", 6),
                max_concept_words=self.config.get("max_concept_words", 4),
                stop_concept=self.config.get("stop_concept",[])
            )

            log.info("[CACHE] Loaded embedding cache from %s", path)
            return True

        except Exception as e:
            log.warning("[CACHE] Failed loading cache from %s: %s", path, e)
            return False
    
    def export_query_graph_png(
    self,
    query: str,
    g,
    graph_scores=None,
    max_nodes: int = 35,
    output_dir: str = "report_graphs",
    ) -> str | None:
        if g is None or g.vcount() == 0:
            return None

        safe_query = re.sub(r"[^a-zA-Z0-9_-]+", "_", query.lower()).strip("_")[:80]
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if "pagerank_score" not in g.vs.attributes():
            return None

        max_passages = self.config.get("query_graph_max_passages", 8)
        max_concepts = self.config.get("query_graph_max_concepts", 14)

        selected_indices = []

        for node_type, limit in [
            ("passage", max_passages),
            ("concept", max_concepts),
        ]:
            type_indices = [
                v.index for v in g.vs
                if v["node_type"] == node_type
            ]

            top_type_indices = sorted(
                type_indices,
                key=lambda i: float(g.vs[i]["pagerank_score"]),
                reverse=True,
            )[:limit]

            selected_indices.extend(top_type_indices)

        selected_indices = list(dict.fromkeys(selected_indices))[:max_nodes]

        if not selected_indices:
            return None

        subg = g.subgraph(selected_indices)

        labels = []
        colors = []
        sizes = []

        max_score = max(subg.vs["pagerank_score"]) if subg.vcount() else 1.0

        for v in subg.vs:
            node_type = v["node_type"]
            name = v["name"]
            score = float(v["pagerank_score"])

            sizes.append(18 + 45 * (score / max_score if max_score > 0 else 0))

            if node_type == "passage":
                doc = self.corpus.get(name, {})
                text = doc.get("text", str(name)) if isinstance(doc, dict) else str(doc)
                labels.append("DOC: " + text[:45])
                colors.append("lightblue")

            elif node_type == "concept":
                concept = self.concept_store.id_to_text.get(name, name)
                labels.append("ENT: " + concept[:35])
                colors.append("orange")

            else:
                labels.append(str(name)[:35])
                colors.append("gray")

        subg.vs["label"] = labels

        if "weight" in subg.es.attributes():
            weights = [float(w) for w in subg.es["weight"]]
        else:
            weights = [1.0 for _ in subg.es]

        max_weight = max(weights) if weights else 1.0
        edge_widths = [
            0.5 + 2.0 * (w / max_weight if max_weight > 0 else 0)
            for w in weights
        ]

        png_path = output_path / f"{safe_query}_query_graph.png"

        layout = subg.layout("fr")

        ig.plot(
            subg,
            target=str(png_path),
            layout=layout,
            bbox=(1600, 950),
            margin=90,
            vertex_label=subg.vs["label"],
            vertex_label_size=15,
            vertex_color=colors,
            vertex_size=sizes,
            edge_width=edge_widths,
            edge_color="gray",
        )

        return str(png_path)
