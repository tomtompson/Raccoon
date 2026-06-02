from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


log = logging.getLogger("query_scenario_benchmark")

# =============================================================================
# Parameters: edit these, then run `python test/query_scenario_benchmark.py`
# =============================================================================

PROMPT = REPO_ROOT / "prompts/example_rechtspraak/query_scenarios/query_generation_semantic.txt"
DATASET_DIR = REPO_ROOT / "datasets/synthetic_prompt_benchmark"
REPORT_DIR = REPO_ROOT / "reports/query_scenario_benchmark"

RUN_METHODS = ["bm25", "dense", "hybrid", "linearrag"]
REUSE_EXISTING_DATASET = False
SKIP_GENERATION = False
WRITE_PDF = True

OLLAMA_HOST = "http://localhost:11434"
NUM_CTX = 12288


ELASTICSEARCH_URL = None
ELASTICSEARCH_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
VERBOSE = False

# =============================================================================
# Fixed benchmark defaults. Change these too if you need deeper tuning.
# =============================================================================

PARENT_CHUNKS = REPO_ROOT / "data/processed/chunks_recht/parent_chunks.json"
CHILD_CHUNKS = REPO_ROOT / "data/processed/chunks_recht/child_chunks.json"

VALIDATION_PROMPTS = {
    "query_validation": REPO_ROOT / "prompts/example_rechtspraak/query_validation.txt",
    "candidate_judging": REPO_ROOT / "prompts/example_rechtspraak/candidate_judging.txt",
    "distribution_validation": REPO_ROOT / "prompts/example_rechtspraak/distribution_validation.txt",
}

K_VALUES = [1,5,10, 50]
TOP_K = 50
AVAILABLE_METHODS = ["bm25", "dense", "hybrid", "linearrag"]

SYNTHESIS = {
    "embedding_id": "Snowflake/snowflake-arctic-embed-l-v2.0",
    "embedding_kwargs": {},
    "query_model": "qwen3:8b",
    "query_validation_model": "qwen3:8b",
    "judge_model": "qwen3:8b",
    "qrel_validation_model": "qwen3:8b",
    "queries_per_parent_to_generate": 2,
    "max_queries_to_keep_per_parent": 1,
    "dense_k": 25,
    "bm25_k": 15,
    "rrf_top_k": 10,
    "same_topic_negative_k": 3,
    "random_negative_k": 3,
    "min_score_to_keep_in_qrels": 2,
    "max_qrels_per_query": 3,
    "include_source_parent_children": True,
    "parent_text_limit_prompt": 3000,
    "candidate_text_limit_prompt": 800,
    "max_estimated_tokens": 8000,
    "overwrite_corpus": False,
    "max_parents": None,
    "random_seed": 42,
    "target_queries_per_source": 1,
    "shuffle_parents": True,
    "reranker_id": "BAAI/bge-reranker-v2-m3",
    "rerank_pool_size": 70,
    "rerank_keep_top_k": 40,
    "allow_copy_like_queries": False,
    "min_query_tokens": 3,
    "max_query_tokens": 24,
}

DENSE = {
    "model_id": "snowflake/snowflake-arctic-embed-l-v2.0",
    "max_length": 256,
    "query_prompt_name": "query",
    "passage_prompt_name": "document",
    "normalize_embeddings": True,
    "batch_size": 128,
    "corpus_chunk_size": 50_000,
    "query_chunk_size": 1024,
    "show_progress_bar": True,
    "score_function": "dot",
    "use_faiss": False,
}

LINEARRAG = {
    "embedding_model_name": "snowflake/snowflake-arctic-embed-l-v2.0",
    "spacy_model_name": "nl_core_news_sm",
    "max_seq_length": 256,
    "embed_batch_size": 192,
    "ner_batch_size": 64,
    "dense_candidate_k": 500,
    "bm25_candidate_k": 500,
    "graph_candidate_k": 500,
    "local_graph_dense_seed_k": 200,
    "local_graph_bm25_seed_k": 200,
    "local_graph_sentence_seed_k": 80,
    "local_graph_concept_seed_k": 80,
    "dense_rrf_weight": 0.0,
    "bm25_rrf_weight": 0.0,
    "graph_rrf_weight": 1.0,
    "rrf_k": 60,
    "min_concept_len": 4,
    "max_concept_words": 6,
    "min_concept_df": 1,
    "max_concept_df_ratio": 0.30,
    "stop_concept": [
        "artikel", "rechtbank", "zaak", "zaken", "eiser", "gedaagde",
        "verzoeker", "verweerster", "werknemer", "werkgever", "partij",
        "partijen", "overeenkomst", "arbeidsovereenkomst", "datum",
        "januari", "februari", "maart", "april", "mei", "juni",
        "juli", "augustus", "september", "oktober", "november",
        "december", "lid", "grond", "beroep", "besluit", "uitspraak",
        "rechter", "kantonrechter", "proces", "procedure", "verzoek",
        "vordering",
    ],
    "ppr_damping": 0.6,
    "ppr_max_result_docs": 500,
}


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def beir_exists(path: Path) -> bool:
    return all((path / relative).exists() for relative in ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv"))


def generate_dataset() -> Path:
    dataset_dir = repo_path(DATASET_DIR)
    if (REUSE_EXISTING_DATASET or SKIP_GENERATION) and beir_exists(dataset_dir):
        log.info("Reusing dataset at %s", dataset_dir)
        return dataset_dir
    if SKIP_GENERATION:
        raise FileNotFoundError(f"Missing BEIR dataset at {dataset_dir}")

    from raccoon.synthesizer import OllamaSynthesizer

    prompt = repo_path(PROMPT)
    synth = OllamaSynthesizer(
        host=OLLAMA_HOST,
        num_ctx=NUM_CTX,
        prompt_paths={
            "query_generation": prompt,
            **VALIDATION_PROMPTS,
        },
    )

    parent_chunks = synth.load_chunks(PARENT_CHUNKS)
    child_chunks = synth.load_chunks(CHILD_CHUNKS)

    synthesis = dict(SYNTHESIS)

    log.info("Generating dataset at %s with prompt %s", dataset_dir, prompt)
    synth.synthesize_beir(
        parent_chunks=parent_chunks,
        child_chunks=child_chunks,
        output_dir=str(dataset_dir),
        **synthesis,
    )
    return dataset_dir


def load_beir(dataset_dir: Path) -> tuple[dict[str, dict[str, str]], dict[str, str], dict[str, dict[str, int]], str]:
    from raccoon.dataloader.utils import load_local_beir_dataset, validate_dataset

    corpus, queries, qrels, name = load_local_beir_dataset(str(dataset_dir), "test")
    validate_dataset(corpus, queries, qrels)
    return corpus, queries, qrels, name


@contextlib.contextmanager
def elasticsearch(needed: bool) -> Iterator[str | None]:
    if not needed:
        yield None
        return
    if ELASTICSEARCH_URL:
        yield ELASTICSEARCH_URL.rstrip("/")
        return

    try:
        from testcontainers.elasticsearch import ElasticSearchContainer
    except Exception as exc:
        log.warning("Elasticsearch testcontainer is unavailable: %s", exc)
        yield None
        return

    try:
        with ElasticSearchContainer(ELASTICSEARCH_IMAGE) as container:
            url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
            log.info("Started Elasticsearch at %s", url)
            yield url
    except Exception as exc:
        log.warning("Could not start Elasticsearch testcontainer: %s", exc)
        yield None


def index_name(dataset_name: str, method: str) -> str:
    digest = hashlib.sha1(f"{dataset_name}:{method}".encode("utf-8")).hexdigest()[:10]
    return f"query-scenario-{method}-{digest}"


def build_bm25(corpus: dict[str, dict[str, str]], queries: dict[str, str], dataset_name: str, url: str, method: str) -> Any:
    from raccoon.custom_retriever.BM25Retriever import BM25Retriever

    return BM25Retriever(
        elasticsearch_url=url,
        index_name=index_name(dataset_name, method),
        language="dutch",
        corpus=corpus,
        queries=queries,
        topk=TOP_K,
    )


def build_dense(corpus: dict[str, dict[str, str]], queries: dict[str, str]) -> Any:
    from raccoon.custom_retriever.DenseRetriever import DenseRetrieverSentenceBert

    return DenseRetrieverSentenceBert(
        corpus=corpus,
        queries=queries,
        device=device(),
        topk=TOP_K,
        model_id=DENSE["model_id"],
        max_length=DENSE["max_length"],
        query_prompt_name=DENSE["query_prompt_name"],
        passage_prompt_name=DENSE["passage_prompt_name"],
        normalize_embeddings=DENSE["normalize_embeddings"],
        batch_size=DENSE["batch_size"],
        corpus_chunk_size=DENSE["corpus_chunk_size"],
        query_chunk_size=DENSE["query_chunk_size"],
        show_progress_bar=DENSE["show_progress_bar"],
    )


def run_bm25(corpus: dict[str, dict[str, str]], queries: dict[str, str], dataset_name: str, url: str | None) -> tuple[Any, dict[str, dict[str, float]]]:
    if not url:
        raise RuntimeError("Elasticsearch is required for BM25")
    retriever = build_bm25(corpus, queries, dataset_name, url, "bm25")
    retriever.index_corpus()
    return retriever, retriever.search(top_k=TOP_K)


def run_dense(corpus: dict[str, dict[str, str]], queries: dict[str, str], report_dir: Path) -> tuple[Any, dict[str, dict[str, float]]]:
    retriever = build_dense(corpus, queries)
    results = retriever.search(
        top_k=TOP_K,
        encode_output_path=str(report_dir / "embeddings" / "dense"),
        score_function=DENSE["score_function"],
        use_faiss=DENSE["use_faiss"],
    )
    return retriever, results


def run_hybrid(corpus: dict[str, dict[str, str]], queries: dict[str, str], dataset_name: str, url: str | None, report_dir: Path) -> tuple[Any, dict[str, dict[str, float]]]:
    if not url:
        raise RuntimeError("Elasticsearch is required for hybrid retrieval")

    from raccoon.custom_retriever.HybridRetriever import HybridRetriever

    bm25 = build_bm25(corpus, queries, dataset_name, url, "hybrid")
    bm25.index_corpus()
    bm25_results = bm25.search(top_k=TOP_K)

    dense = build_dense(corpus, queries)
    dense_results = dense.search(
        top_k=TOP_K,
        encode_output_path=str(report_dir / "embeddings" / "hybrid_dense"),
        score_function=DENSE["score_function"],
        use_faiss=DENSE["use_faiss"],
    )

    retriever = HybridRetriever(
        config={"top_k": TOP_K},
        corpus=corpus,
        queries=queries,
        retrievers=[(bm25_results, 1.0), (dense_results, 1.0)],
        k=60,
    )
    return retriever, retriever.search(top_k=TOP_K)


def run_linearrag(corpus: dict[str, dict[str, str]], queries: dict[str, str], dataset_name: str) -> tuple[Any, dict[str, dict[str, float]]]:
    from raccoon.custom_retriever.LinearRagRetriever import LinearRagRetriever

    config = dict(LINEARRAG)
    config["dataset_name"] = dataset_name
    config["retrieval_top_k"] = TOP_K
    config["device"] = device()

    retriever = LinearRagRetriever(config=config, corpus=corpus, queries=queries)
    retriever.index_corpus()
    return retriever, retriever.search(top_k=TOP_K)


def evaluate(qrels: dict[str, dict[str, int]], retriever: Any, results: dict[str, dict[str, float]]) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    from beir.retrieval.evaluation import EvaluateRetrieval

    metrics = EvaluateRetrieval().evaluate(qrels=qrels, results=results, k_values=K_VALUES)
    retriever.retrieval_metrics = metrics
    return metrics


def metric(metrics: tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, float]] | None, name: str, k: int) -> float | None:
    if not metrics:
        return None
    ndcg, score_map, recall, precision = metrics
    lookup = {
        "ndcg": ndcg.get(f"NDCG@{k}"),
        "map": score_map.get(f"MAP@{k}"),
        "recall": recall.get(f"Recall@{k}"),
        "precision": precision.get(f"P@{k}"),
    }
    value = lookup[name]
    return float(value) if value is not None else None


def latency_ms(retriever: Any) -> float | None:
    search = (getattr(retriever, "metrics", {}) or {}).get("query_time", {}).get("search", {})
    if search.get("time_per_query_in_seconds") is not None:
        return float(search["time_per_query_in_seconds"]) * 1000
    if search.get("time_in_seconds") is not None and search.get("queries"):
        return float(search["time_in_seconds"]) / float(search["queries"]) * 1000
    return None


def run_method(
    method: str,
    corpus: dict[str, dict[str, str]],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    dataset_name: str,
    report_dir: Path,
    elasticsearch_url: str | None,
) -> dict[str, Any]:
    try:
        if method == "bm25":
            retriever, results = run_bm25(corpus, queries, dataset_name, elasticsearch_url)
        elif method == "dense":
            retriever, results = run_dense(corpus, queries, report_dir)
        elif method == "hybrid":
            retriever, results = run_hybrid(corpus, queries, dataset_name, elasticsearch_url, report_dir)
        elif method == "linearrag":
            retriever, results = run_linearrag(corpus, queries, dataset_name)
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = evaluate(qrels, retriever, results)
        return {
            "method": method,
            "status": "ok",
            "note": "",
            "retriever": retriever,
            "results": results,
            "metrics": metrics,
            "latency_ms": latency_ms(retriever),
        }
    except Exception as exc:
        log.exception("%s failed", method)
        return {
            "method": method,
            "status": "failed",
            "note": str(exc),
            "retriever": None,
            "results": None,
            "metrics": None,
            "latency_ms": None,
        }


def summary_row(run: dict[str, Any]) -> dict[str, Any]:
    row = {
        "method": run["method"],
        "status": run["status"],
        "latency_ms": run["latency_ms"],
        "note": run["note"],
    }
    for name in ("ndcg", "map", "recall", "precision"):
        for k in K_VALUES:
            row[f"{name}@{k}"] = metric(run["metrics"], name, k)
    return row


def write_csv_report(report_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = report_dir / "summary.csv"
    fields = ["method", "status"]
    for name in ("ndcg", "map", "recall", "precision"):
        fields.extend(f"{name}@{k}" for k in K_VALUES)
    fields.extend(["latency_ms", "note"])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_report(dataset_dir: Path, report_dir: Path, rows: list[dict[str, Any]]) -> Path:
    path = report_dir / "query_scenario_report.md"
    ok_rows = [row for row in rows if row["status"] == "ok" and row.get("ndcg@10") is not None]
    best = max(ok_rows, key=lambda row: float(row["ndcg@10"])) if ok_rows else None

    lines = [
        "# Query Scenario Benchmark Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Prompt: `{repo_path(PROMPT)}`",
        f"Dataset: `{dataset_dir}`",
        "",
        "| Method | Status | nDCG@10 | nDCG@50 | MAP@10 | Recall@10 | Precision@10 | Latency ms/query | Note |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['status']} | {fmt(row.get('ndcg@10'))} | "
            f"{fmt(row.get('ndcg@50'))} | {fmt(row.get('map@10'))} | "
            f"{fmt(row.get('recall@10'))} | {fmt(row.get('precision@10'))} | "
            f"{fmt(row.get('latency_ms'))} | {row.get('note') or ''} |"
        )

    lines += ["", "## Recommendation", ""]
    if best:
        lines.append(
            f"Best method by nDCG@10: **{best['method']}** "
            f"({fmt(best.get('ndcg@10'))}). Check latency before using it as the default RAG retriever."
        )
    else:
        lines.append("No successful retriever run produced nDCG@10.")

    lines += [
        "",
        "For production RAG, compare the winning method against hybrid retrieval when exact legal terms and paraphrased user questions both matter.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_json_report(report_dir: Path, runs: list[dict[str, Any]]) -> Path:
    path = report_dir / "raw_results.json"
    payload = [
        {
            "method": run["method"],
            "status": run["status"],
            "note": run["note"],
            "latency_ms": run["latency_ms"],
            "results": run["results"],
        }
        for run in runs
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_pdf_report(report_dir: Path, runs: list[dict[str, Any]]) -> Path | None:
    retrievers = [run["retriever"] for run in runs if run["retriever"] is not None]
    if not retrievers:
        return None

    from raccoon.report.StaticRetrieverReport import StaticRetrieverReport

    return StaticRetrieverReport().generate_report(
        title="Query Scenario Benchmark",
        retrievers=retrievers,
        output_path=report_dir / "retriever_report.pdf",
        config={
            "comparison_metric": "NDCG@10",
            "retrieval_metric_rows": 24,
            "metric_rows": 16,
            "sample_queries": 3,
            "sample_results_per_query": 3,
            "sample_text_chars": 220,
        },
    )


def selected_methods() -> list[str]:
    methods = [method.strip() for method in RUN_METHODS] if isinstance(RUN_METHODS, (list, tuple, set)) else [
        method.strip() for method in str(RUN_METHODS).split(",")
    ]
    methods = [method for method in methods if method]
    unknown = sorted(set(methods) - set(AVAILABLE_METHODS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Choose from {AVAILABLE_METHODS}")
    return methods


def main() -> int:
    logging.basicConfig(
        level=logging.DEBUG if VERBOSE else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    methods = selected_methods()
    report_dir = repo_path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = generate_dataset()
    corpus, queries, qrels, dataset_name = load_beir(dataset_dir)

    needs_elasticsearch = any(method in {"bm25", "hybrid"} for method in methods)
    with elasticsearch(needs_elasticsearch) as url:
        runs = [
            run_method(method, corpus, queries, qrels, dataset_name, report_dir, url)
            for method in methods
        ]

    rows = [summary_row(run) for run in runs]
    csv_path = write_csv_report(report_dir, rows)
    markdown_path = write_markdown_report(dataset_dir, report_dir, rows)
    json_path = write_json_report(report_dir, runs)
    pdf_path = write_pdf_report(report_dir, runs) if WRITE_PDF else None

    log.info("Wrote CSV report to %s", csv_path)
    log.info("Wrote Markdown report to %s", markdown_path)
    log.info("Wrote raw JSON to %s", json_path)
    if pdf_path:
        log.info("Wrote PDF report to %s", pdf_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
