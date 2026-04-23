from typing import Tuple, Dict
from pathlib import Path
import json
from datasets import load_dataset


def load_local_beir_dataset(dataset_dir: str, split: str = "test") -> Tuple[dict, dict, dict, str]:
    dataset_path = Path(dataset_dir)
    corpus_file = dataset_path / "corpus.jsonl"
    queries_file = dataset_path / "queries.jsonl"
    qrels_file = dataset_path / "qrels" / f"{split}.tsv"

    if not corpus_file.exists():
        raise FileNotFoundError(f"Missing corpus file: {corpus_file}")
    if not queries_file.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_file}")
    if not qrels_file.exists():
        raise FileNotFoundError(f"Missing qrels file: {qrels_file}")

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    with corpus_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            doc_id = row["_id"]
            corpus[doc_id] = {
                "title": row.get("title", "") or "",
                "text": row.get("text", "") or "",
            }

    with queries_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = row["_id"]
            queries[query_id] = row["text"]

    with qrels_file.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if i == 0 and "query-id" in line.lower():
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid qrels row in {qrels_file}: '{line}'. Expected 3 tab-separated columns."
                )
            qid, did, score = parts
            qrels.setdefault(qid, {})[did] = int(score)

    dataset_name = dataset_path.name
    return corpus, queries, qrels, dataset_name


def load_hf_beir_dataset(dataset_name: str, split: str = "test") -> Tuple[dict, dict, dict, str]:
    hf_corpus = load_dataset(dataset_name, "corpus", split="corpus")
    hf_qrels = load_dataset(dataset_name, "default", split=split)
    hf_queries = load_dataset(dataset_name, "queries", split="queries")

    corpus: Dict[str, Dict[str, str]] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, int]] = {}

    for row in hf_corpus:
        doc_id = row["_id"]
        corpus[doc_id] = {
            "title": row.get("title", "") or "",
            "text": row["text"],
        }

    for row in hf_queries:
        query_id = row["_id"]
        queries[query_id] = row["text"]

    for row in hf_qrels:
        qid = row["query-id"]
        did = row["corpus-id"]
        score = int(row["score"])
        qrels.setdefault(qid, {})[did] = score

    return corpus, queries, qrels, dataset_name


def validate_dataset(corpus: dict, queries: dict, qrels: dict) -> None:
    print(f"queries: {len(queries)}")
    print(f"qrels: {len(qrels)}")
    print(f"corpus: {len(corpus)}")

    qrel_qids = set(qrels.keys())
    query_qids = set(queries.keys())
    overlap_q = qrel_qids & query_qids

    corpus_ids = set(corpus.keys())
    qrel_docids = set()
    for docs in qrels.values():
        qrel_docids.update(docs.keys())
    overlap_d = corpus_ids & qrel_docids

    print(f"overlap qrels∩queries: {len(overlap_q)} / {len(qrel_qids)}")
    print(f"overlap qrels_docs∩corpus: {len(overlap_d)} / {len(qrel_docids)}")

    if len(overlap_q) != len(qrel_qids):
        missing_q = list(qrel_qids - query_qids)[:10]
        raise ValueError(f"Some qrels query ids are missing in queries.jsonl: {missing_q}")

    if len(overlap_d) != len(qrel_docids):
        missing_d = list(qrel_docids - corpus_ids)[:10]
        raise ValueError(f"Some qrels corpus ids are missing in corpus.jsonl: {missing_d}")