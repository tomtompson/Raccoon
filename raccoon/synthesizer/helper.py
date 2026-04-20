from rich.progress import Progress
from typing import List

import json 
from pathlib import Path
from typing import Tuple
import re
from collections import Counter, defaultdict
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
