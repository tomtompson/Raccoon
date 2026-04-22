from rich.progress import Progress
from typing import List

import json 
from pathlib import Path
from typing import Tuple
import re
from collections import Counter, defaultdict
import random
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import torch
# =========================
# Helpers
# =========================

def _merge_dictionaries(dict_a:dict, dict_b:dict) -> List[dict]:
    merged = {c["child_id"]: dict(c) for c in dict_b}
    for c in dict_a:
        merged[c["child_id"]] = dict(c)
    return list(merged.values())

def _extract_json_array(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON array from Ollama response.")


def _extract_json_object(text: str):
    text = text.strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group(0))

    raise ValueError("Could not parse JSON object from Ollama response.")


def _append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_existing_query_texts(queries_path: Path) -> Tuple[set, int]:
    seen = set()
    max_q = -1

    if not queries_path.exists():
        return seen, 0

    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            text = str(row.get("text", "")).strip().lower()
            qid = str(row.get("_id", ""))

            if text:
                seen.add(text)

            if qid.startswith("q"):
                try:
                    max_q = max(max_q, int(qid[1:]))
                except Exception:
                    pass

    return seen, max_q + 1


def _load_processed_parent_ids(progress_path: Path) -> set:
    done = set()

    if not progress_path.exists():
        return done

    with open(progress_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(line)

    return done


def _mark_parent_processed(progress_path: Path, parent_id: str):
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(parent_id + "\n")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"[a-zà-ÿ0-9]+", text)


def _jaccard_tokens(a: str, b: str) -> float:
    ta = set(_tokenize(a))
    tb = set(_tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _char_ngram_overlap(a: str, b: str, n: int = 5) -> float:
    def ngrams(s: str):
        s = re.sub(r"\s+", " ", s.lower()).strip()
        if len(s) < n:
            return {s} if s else set()
        return {s[i:i+n] for i in range(len(s) - n + 1)}

    na = ngrams(a)
    nb = ngrams(b)
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def _looks_too_short(query: str, min_tokens: int = 3) -> bool:
    return len(_tokenize(query)) < min_tokens


def _looks_too_long(query: str, max_tokens: int = 20) -> bool:
    return len(_tokenize(query)) > max_tokens


def _looks_artificial(query: str) -> bool:
    q = query.lower().strip()

    bad_patterns = [
        r"^volgens de tekst\b",
        r"^in deze tekst\b",
        r"^wat staat er in\b",
        r"^welke definities gelden\b",
        r"^wat zijn de regels\b$",
        r"^wanneer geldt de wet\b$",
        r"^wat moet de werkgever doen\b$",
        r"^wat moet de opdrachtgever doen\b$",
        r"^wat zegt artikel\b",
        r"\bde gegeven tekst\b",
        r"\bhet document\b",
        r"\bde passage\b",
        r"\bdeze wetstekst\b",
    ]
    return any(re.search(p, q) for p in bad_patterns)


def _query_is_too_copy_like(query: str, source_text: str) -> bool:
    excerpt = source_text[:2500]
    jac = _jaccard_tokens(query, excerpt)
    ch = _char_ngram_overlap(query, excerpt, n=6)

    query_tokens = _tokenize(query)
    if len(query_tokens) <= 5:
        return False

    return jac > 0.72 or ch > 0.62


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default
    
# =========================
# Load Saved Quwery Count
# =========================
def _load_saved_query_counts_by_source(debug_path: Path) -> dict:
    counts = defaultdict(int)

    if not debug_path.exists():
        return counts

    seen_query_ids = set()

    with open(debug_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            query_id = row.get("query_id")
            source = row.get("query_source")

            if query_id and source and query_id not in seen_query_ids:
                counts[source] += 1
                seen_query_ids.add(query_id)

    return counts
# =========================
# FAISS Builder
# =========================
def _build_faiss_index(chunks, embedding_id:str, model_kwargs: dict, show_progress = False) -> FAISS:

    embed_device = "cuda" if torch.cuda.is_available() else "cpu"

    # override / inject device into kwargs
    embedding_kwargs = model_kwargs or {}
    embedding_kwargs["device"] = embed_device
        
    embedder = HuggingFaceEmbeddings(model_name = embedding_id, model_kwargs = embedding_kwargs, show_progress = show_progress)

    text = [chunk.page_content for chunk in chunks]

    embeddings = embedder.embed_documents(text)

    metadatas = [chunk.metadata for chunk in chunks]

    ids = [metadata["child_id"] for metadata in metadatas]

    text_embedding_pairs = list(zip(text, embeddings))

    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embedder,
        metadatas=metadatas,
        ids=ids,
    )

    return vectorstore

def _build_child_records_from_vectorstore(vectorstore) -> List[dict]:
    docstore = vectorstore.docstore._dict
    child_records = []
    with Progress() as progress:
        task = progress.add_task("Building child records from vector store...", total=len(docstore.items()))
        for doc_id, doc in docstore.items():
            metadata = doc.metadata or {}
            child_id = str(metadata.get("child_id") or doc_id)
            parent_id = str(metadata.get("parent_id") or "")
            title = metadata.get("title", "") or ""
            text = doc.page_content or ""
            source_doc_id = str(
                metadata.get("source_doc_id")
                or metadata.get("source")
                or metadata.get("source_file")
                or metadata.get("document_id")
                or ""
            )
            chunk_index = metadata.get("chunk_index")
            if chunk_index is None:
                chunk_index = metadata.get("child_index")
            try:
                chunk_index = int(chunk_index) if chunk_index is not None else None
            except Exception:
                chunk_index = None

            child_records.append({
                "child_id": child_id,
                "parent_id": parent_id,
                "title": title,
                "text": text,
                "source_doc_id": source_doc_id,
                "chunk_index": chunk_index,
                "doc": doc,
            })
            progress.update(task, advance=1)

    return child_records
# =========================
# Query/qrel selection
# =========================

def _count_scores(judged_by_id: dict) -> dict:
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in judged_by_id.values():
        v = int(v)
        counts[v] = counts.get(v, 0) + 1
    return counts


def _query_has_enough_signal(
    judged_by_id: dict,
    min_ge2: int = 1,
    min_any_positive: int = 1,
) -> bool:
    counts = _count_scores(judged_by_id)
    total_positive = counts.get(1, 0) + counts.get(2, 0) + counts.get(3, 0)
    ge2 = counts.get(2, 0) + counts.get(3, 0)
    return total_positive >= min_any_positive and ge2 >= min_ge2


def _select_qrels_realistic(
    pooled_candidates: list,
    judged_by_id: dict,
    min_score_to_keep: int = 1,
    max_qrels_per_query: int = 8,
) -> list:
    selected = []

    for candidate in pooled_candidates:
        child_id = candidate["child_id"]
        rel = int(judged_by_id.get(child_id, 0))
        if rel < min_score_to_keep:
            continue

        selected.append({
            "corpus_id": child_id,
            "score": rel,
            "dense_score": candidate.get("dense_score"),
            "bm25_score": candidate.get("bm25_score"),
            "rrf_score": candidate.get("rrf_score"),
            "rerank_score": candidate.get("rerank_score"),
            "candidate_sources": candidate.get("sources", []),
            "parent_id": candidate.get("parent_id"),
            "source_doc_id": candidate.get("source_doc_id"),
            "chunk_index": candidate.get("chunk_index"),
        })

    # Best qrels first: by relevance label, then retrieval support
    def key_fn(x):
        sources = set(x.get("candidate_sources", []))
        overlap_bonus = 1 if ("dense" in sources and "bm25" in sources) else 0
        return (
            x["score"],
            x["rerank_score"] if x.get("rerank_score") is not None else -1e9,
            overlap_bonus,
            x["rrf_score"] if x["rrf_score"] is not None else -1e9,
            x["dense_score"] if x["dense_score"] is not None else -1e9,
            x["bm25_score"] if x["bm25_score"] is not None else -1e9,
        )

    selected.sort(key=key_fn, reverse=True)
    return selected[:max_qrels_per_query]

# =========================
# Retrieval fusion
# =========================

def _rrf_fuse(
    dense_hits: List[Tuple[dict, float]],
    bm25_hits: List[Tuple[dict, float]],
    rrf_k: int = 60,
):
    ranked = defaultdict(float)
    doc_lookup = {}

    for rank, item in enumerate(dense_hits, start=1):
        doc = item[0]
        child_id = str(doc.get("child_id"))
        doc_lookup[child_id] = doc
        ranked[child_id] += 1.0 / (rrf_k + rank)

    for rank, item in enumerate(bm25_hits, start=1):
        doc = item[0]
        child_id = str(doc.get("child_id"))
        doc_lookup[child_id] = doc
        ranked[child_id] += 1.0 / (rrf_k + rank)

    fused = sorted(ranked.items(), key=lambda x: x[1], reverse=True)
    return [(doc_lookup[child_id], score) for child_id, score in fused]

# =========================
# Candidate Poolingsss
# =========================

def _sample_same_topic_negatives_via_overlap(
    query: str,
    parent_id: str,
    child_records: List[dict],
    exclude_ids: set,
    limit: int,
) -> List[dict]:
    scored = []
    for rec in child_records:
        if rec["child_id"] in exclude_ids:
            continue
        if rec["parent_id"] == parent_id:
            continue

        overlap = _jaccard_tokens(query, f'{rec.get("title", "")}\n{rec.get("text", "")}')
        if overlap > 0:
            scored.append((rec, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [rec for rec, _ in scored[:limit]]


def _sample_random_negatives(
    child_records: List[dict],
    exclude_ids: set,
    limit: int,
    rng: random.Random,
) -> List[dict]:
    pool = [rec for rec in child_records if rec["child_id"] not in exclude_ids]
    if not pool or limit <= 0:
        return []
    if len(pool) <= limit:
        return pool
    return rng.sample(pool, limit)

# =========================
# Chunk helper
# =========================
def _estimate_judge_tokens(query: str, candidates: list, candidate_text_limit: int) -> int:
    base_prompt_chars = 4500  # estimated 
    total_chars = base_prompt_chars + len(query)

    for c in candidates:
        title = (c.get("title", "") or "")[:180]
        text = (c.get("text", "") or "")[:candidate_text_limit]
        child_id = c.get("child_id", "") or ""
        total_chars += len(title) + len(text) + len(child_id) + 80

    est_token = total_chars // 4
    return est_token  # rough estimate