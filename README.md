# Raccoon

Raccoon is a Python toolkit for building and evaluating retrieval-augmented generation
(RAG) retrieval benchmarks. It helps you load documents, generate BEIR-style benchmark
datasets, run several retrievers, evaluate retrieval metrics, rerank results, and produce
static PDF reports.

The repo currently focuses on practical benchmark workflows rather than a polished CLI
product. Runnable workflows live in `examples/`; pytest tests live in `test/`.

## What Is Included

- Document loaders for local files and SQL tables.
- Synthetic BEIR dataset generation with Ollama prompts.
- BM25 retrieval through Elasticsearch.
- Dense retrieval with SentenceTransformers.
- Hybrid retrieval with reciprocal rank fusion.
- LinearRAG retrieval with dense, BM25, graph, and concept signals.
- Optional transformer reranking.
- Static PDF reports for comparing retrievers and reranked results.
- A dataset inspection CLI in `cli/cli.py`.

## Installation

Raccoon supports Python `>=3.10,<3.13`.

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

GPU/CUDA support is optional:

```bash
pip install -e ".[gpu]"
```

If you run in a restricted environment and install the GPU extra, set a writable CuPy
cache directory:

```bash
export CUPY_CACHE_DIR=/tmp/cupy-cache
```

## Runtime Requirements

Different workflows need different services and models:

- BM25 examples need Elasticsearch. The example scripts use Docker through
  `testcontainers`, or you can pass an external Elasticsearch URL where supported.
- Dataset synthesis needs Ollama running locally, usually at `http://localhost:11434`,
  with the configured models pulled, for example `ollama pull qwen3:8b`.
- Dense retrieval and reranking download Hugging Face model.
- LinearRAG uses spaCy. 
- SQL loading needs a database connection string, for example through
  `RACCOON_SQL_URL`.

## Basic Usage

### Create A Synthetic BEIR Dataset

This turns saved parent/child chunks into a BEIR-style dataset with `corpus.jsonl`,
`queries.jsonl`, `qrels/test.tsv`, and `qrels_debug.jsonl`.

```python
from raccoon.synthesizer import OllamaSynthesizer

synth = OllamaSynthesizer(
    host="http://localhost:11434",
    prompt_paths={
        "query_generation": "prompts/example_rechtspraak/query_scenarios/query_generation_long_form.txt",
        "query_validation": "prompts/example_rechtspraak/query_validation.txt",
        "candidate_judging": "prompts/example_rechtspraak/candidate_judging.txt",
        "distribution_validation": "prompts/example_rechtspraak/distribution_validation.txt",
    },
)

parent_chunks = synth.load_chunks("data/processed/chunks_recht/parent_chunks.json")
child_chunks = synth.load_chunks("data/processed/chunks_recht/child_chunks.json")

synth.synthesize_beir(
    parent_chunks=parent_chunks,
    child_chunks=child_chunks,
    output_dir="data/processed/rechtspraken/beir_realistic_TEST",
    embedding_id="Snowflake/snowflake-arctic-embed-l-v2.0",
    embedding_kwargs={},
    query_model="qwen3:8b",
    query_validation_model="qwen3:8b",
    judge_model="qwen3:8b",
    qrel_validation_model="qwen3:8b",
    queries_per_parent_to_generate=2,
    max_queries_to_keep_per_parent=1,
    dense_k=15,
    bm25_k=15,
    rrf_top_k=15,
    same_topic_negative_k=3,
    random_negative_k=3,
    min_score_to_keep_in_qrels=2,
    max_qrels_per_query=3,
    include_source_parent_children=True,
    parent_text_limit_prompt=3000,
    candidate_text_limit_prompt=800,
    max_estimated_tokens=5000,
    overwrite_corpus=False,
    max_parents=None,
    random_seed=42,
    target_queries_per_source=1,
    shuffle_parents=True,
    reranker_id=None,
    rerank_pool_size=70,
    rerank_keep_top_k=40,
)
```

### Load And Chunk Local Documents

```python
from pathlib import Path

from raccoon.dataloader import FixedDocumentLoader

loader = FixedDocumentLoader(
    path=Path("data/raw/rechtspraak"),
    chunk_size=4000,
    chunk_overlap=200,
)

parent_docs, child_docs = loader.get_data()
loader.save_chunked_documents(
    "data/processed/chunks_recht/parent_chunks.json",
    "data/processed/chunks_recht/child_chunks.json",
)
```

Supported local file extensions are `.txt`, `.md`, `.pdf`, and `.docx`.



### Load A BEIR-Style Dataset

```python
from raccoon.dataloader.utils import load_local_beir_dataset, validate_dataset

corpus, queries, qrels, dataset_name = load_local_beir_dataset(
    "data/processed/rechtspraken/beir_realistic_TEST",
    split="test",
)
validate_dataset(corpus, queries, qrels)
```

The expected local BEIR layout is:

```text
dataset/
  corpus.jsonl
  queries.jsonl
  qrels/
    test.tsv
```

### Run Dense Retrieval

```python
from raccoon.custom_retriever import DenseRetrieverSentenceBert

retriever = DenseRetrieverSentenceBert(
    corpus=corpus,
    queries=queries,
    model_id="snowflake/snowflake-arctic-embed-l-v2.0",
    max_length=256,
    device="cpu",
    query_prompt_name="query",
    passage_prompt_name="document",
)

results = retriever.search(
    top_k=20,
    encode_output_path="data/embeddings/dense",
)
```

### Run BM25 Retrieval

```python
from raccoon.custom_retriever import BM25Retriever

retriever = BM25Retriever(
    elasticsearch_url="http://localhost:9200",
    index_name="raccoon-bm25",
    language="dutch",
    corpus=corpus,
    queries=queries,
    topk=20,
)

retriever.index_corpus()
results = retriever.search()
```

### Evaluate With BEIR

```python
from beir.retrieval.evaluation import EvaluateRetrieval

evaluator = EvaluateRetrieval()
metrics = evaluator.evaluate(
    qrels=qrels,
    results=retriever.results,
    k_values=[1, 3, 5, 10, 50],
)
retriever.add_retrieval_result(metrics)
```

### Generate A PDF Report

```python
from raccoon.report import StaticRetrieverReport

report = StaticRetrieverReport()
report.generate_report(
    title="Retriever Benchmark",
    retrievers=[retriever],
    qrels=qrels,
    output_path="reports/retriever_benchmark.pdf",
    config={
        "comparison_metric": ["NDCG@10", "Recall@10"],
        "sample_queries": 3,
        "sample_results_per_query": 3,
    },
)
```

### Generate A Dataset Description For The Report

If the report should explain what kind of dataset was evaluated, generate a short
description with `OllamaSynthesizer.generate_description_of_ds()` and pass it to
`StaticRetrieverReport.generate_report()` as `ds_description`.

```python
from raccoon.synthesizer import OllamaSynthesizer
from raccoon.report import StaticRetrieverReport

synth = OllamaSynthesizer(
    host="http://localhost:11434",
    prompt_paths={
        "query_generation": "prompts/example_rechtspraak/query_scenarios/query_generation_long_form.txt",
        "query_validation": "prompts/example_rechtspraak/query_validation.txt",
        "candidate_judging": "prompts/example_rechtspraak/candidate_judging.txt",
        "distribution_validation": "prompts/example_rechtspraak/distribution_validation.txt",
    },
)

description = synth.generate_description_of_ds(
    model="qwen3:8b",
    language="dutch",
    corpus=corpus,
    queries=queries,
    description_length=250,
)

StaticRetrieverReport().generate_report(
    title="Retriever Benchmark",
    ds_description=description,
    retrievers=[retriever],
    qrels=qrels,
    output_path="reports/retriever_benchmark.pdf",
)
```

## Class And Config Reference

Most workflows use the same data shapes:

- `corpus`: `dict[doc_id, {"title": str, "text": str}]`
- `queries`: `dict[query_id, query_text]`
- `qrels`: `dict[query_id, dict[doc_id, relevance_score]]`
- `results`: `dict[query_id, dict[doc_id, score]]`

### `FixedDocumentLoader`

Loads `.txt`, `.md`, `.pdf`, and `.docx` files from one file or a directory, then creates
parent and child chunks.

| Argument | Why pass it |
| --- | --- |
| `path` | File or directory to load. |
| `config` | Optional metadata/config holder for downstream code. |
| `chunk_size` | Parent chunk size in characters. Child chunks are roughly one quarter of this. |
| `chunk_overlap` | Overlap between parent chunks; child overlap is roughly one quarter of this. |
| `separators` | Custom split separators if the default text splitting is not right. |
| `recursive` | Whether directory loading should include nested files. |
| `silent_errors` | Whether directory loader errors should be skipped. |
| `text_encoding` | Encoding for `.txt` and `.md` files. |

Important methods:

- `get_data()` returns `(parent_docs, child_docs)`.
- `save_chunked_documents(parent_output_path, child_output_path)` writes JSON chunk files
  for the synthesizer.

### `SQLDocumentLoader`

Loads rows from a SQL table, combines selected content columns into document text, stores
selected metadata columns, then creates parent and child chunks.

| Argument | Why pass it |
| --- | --- |
| `connection_string` | SQLAlchemy connection URL, for example PostgreSQL or SQLite. |
| `table` | Table to query. |
| `content_columns` | Columns joined into the document text. |
| `metadata_columns` | Columns copied into `Document.metadata`. |
| `where` | Optional SQL `WHERE` clause for filtering rows. |
| `chunk_size` | Parent chunk size. |
| `chunk_overlap` | Parent chunk overlap. |
| `separators` | Optional splitter separators. |

### `OllamaSynthesizer`

Generates synthetic queries and qrels from parent/child chunks using Ollama-hosted LLMs.
It writes a BEIR-style dataset with `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`,
`qrels_debug.jsonl`, and `processed_parents.txt`.

Constructor arguments:

| Argument | Why pass it |
| --- | --- |
| `prompt_paths` | Load prompt templates from files. Expected keys include `query_generation`, `query_validation`, `candidate_judging`, and `distribution_validation`. |
| `prompts` | Provide prompt template strings directly instead of files. |
| `host` | Ollama host, for example `http://localhost:11434`. |
| `num_ctx` | Ollama context window size used for chat calls. |

Useful methods:

- `load_chunks(path)` reads saved parent/child chunk JSON files.
- `load_prompt(path)` and `load_prompts(prompt_paths)` load prompt templates.
- `render_prompt(prompt_name, **kwargs)` renders a loaded prompt.
- `generate_description_of_ds(model, language, corpus, queries, description_length)` creates
  a short dataset description for reports.
- `synthesize_beir(...)` creates the synthetic benchmark dataset.

`synthesize_beir()` parameters:

| Argument | Why pass it |
| --- | --- |
| `parent_chunks` | Parent `Document` chunks used as sources for query generation. |
| `child_chunks` | Child `Document` chunks used as candidate retrievable corpus entries. |
| `output_dir` | Destination for BEIR files and debug/progress files. |
| `embedding_id` | Embedding model used to build the FAISS candidate index during synthesis. |
| `embedding_kwargs` | Extra embedding model options. |
| `query_model` | Ollama model used to generate candidate user queries. |
| `query_validation_model` | Ollama model used to reject ungrounded or unrealistic queries. |
| `judge_model` | Ollama model used to score candidate documents for each query. |
| `qrel_validation_model` | Ollama model used to validate the score distribution for a query. |
| `queries_per_parent_to_generate` | Number of raw queries requested per parent chunk. |
| `max_queries_to_keep_per_parent` | Cap on accepted queries saved from one parent. |
| `dense_k` | Number of dense candidates to pool before judging. |
| `bm25_k` | Number of BM25 candidates to pool before judging. |
| `rrf_top_k` | Number of reciprocal-rank-fused candidates to include. |
| `same_topic_negative_k` | Hard negatives sampled from overlapping topics. |
| `random_negative_k` | Random negatives sampled from unrelated children. |
| `min_score_to_keep_in_qrels` | Minimum judge score written to qrels. |
| `max_qrels_per_query` | Maximum relevant documents kept per query. |
| `include_source_parent_children` | Ensures children from the source parent are included as candidates. |
| `parent_text_limit_prompt` | Max source text characters sent to query/validation prompts. |
| `candidate_text_limit_prompt` | Max candidate text characters sent to judging prompts. |
| `max_estimated_tokens` | Splits candidate judging batches when prompts would be too large. |
| `overwrite_corpus` | Rewrites `corpus.jsonl` even if it already exists. |
| `max_parents` | Optional cap for small or debug synthesis runs. |
| `random_seed` | Reproducible shuffle and negative sampling. |
| `target_queries_per_source` | Cap accepted queries per source document. |
| `shuffle_parents` | Randomizes parent processing within each source. |
| `reranker_id` | Optional transformer reranker for candidate pooling before LLM judging. |
| `rerank_pool_size` | Number of pooled candidates reranked. |
| `rerank_keep_top_k` | Number of reranked candidates kept for judging. |
| `allow_copy_like_queries` | Allows query text that closely copies the source text. |
| `min_query_tokens` | Rejects queries that are too short. |
| `max_query_tokens` | Rejects queries that are too long. |

### `BM25Retriever`

Indexes corpus text into Elasticsearch and runs lexical retrieval.

| Argument | Why pass it |
| --- | --- |
| `elasticsearch_url` | Elasticsearch endpoint. Required for indexing/searching. |
| `index_name` | Elasticsearch index name. |
| `language` | Elasticsearch analyzer language, for example `english` or `dutch`. |
| `config` | Optional config metadata shown in reports. |
| `corpus` | Documents to index. |
| `queries` | Queries to search. |
| `reranker` | Optional `Reranker` applied after retrieval. |
| `content_field` | Elasticsearch text field name. |
| `metadata_field` | Elasticsearch metadata field name. |
| `topk` | Default number of results returned by `search()`. |
| `refresh_on_write` | Refresh index after writes so new docs are searchable immediately. |
| `timeout` | Elasticsearch request timeout. |

Important methods: `index_corpus()`, `search(top_k=None)`, `save_artifacts(path)`,
and `load_artifacts(path)`.

### `DenseRetrieverSentenceBert`

Embeds queries and documents with SentenceTransformers and searches with PyTorch or FAISS.

Constructor arguments:

| Argument | Why pass it |
| --- | --- |
| `model_id` | Required SentenceTransformers/Hugging Face model id. |
| `corpus`, `queries` | Data to encode and search. |
| `reranker` | Optional reranker for retrieved results. |
| `max_length` | Sets model max sequence length. |
| `device` | `cpu`, `cuda`, or other torch device. |
| `query_prompt_name` | Prompt name used for query encoding when the model supports prompts. |
| `passage_prompt_name` | Prompt name used for document encoding. |
| `normalize_embeddings` | Normalize embeddings at encode time. |
| `topk` | Default result count. |
| `batch_size` | Encoding batch size. |
| `corpus_chunk_size` | Number of corpus documents per embedding shard. |
| `query_chunk_size` | Number of query embeddings scored at once. |
| `show_progress_bar` | Show model encoding progress. |

`search()` arguments:

| Argument | Why pass it |
| --- | --- |
| `top_k` | Result count for this search. |
| `score_function` | `dot` or `cos_sim`. |
| `encode_output_path` | Directory for cached query/corpus embeddings. |
| `overwrite_embeddings` | Recompute embeddings even if cache files exist. |
| `query_filename` | Query embedding cache filename. |
| `corpus_filename` | Corpus shard filename pattern. |
| `use_faiss` | Use FAISS instead of chunked PyTorch scoring. |

### `HybridRetriever`

Combines rankings from child retrievers or precomputed result dictionaries with reciprocal
rank fusion.

| Argument | Why pass it |
| --- | --- |
| `retrievers` | List of `(retriever_or_results, weight)` pairs to fuse. |
| `k` | RRF smoothing constant. Larger values reduce the impact of top rank positions. |
| `corpus`, `queries` | Data used for reranking/reporting. |
| `reranker` | Optional reranker applied after fusion. |
| `config` | Optional config; `config["top_k"]` is used when `search(top_k=None)`. |

### `LinearRagRetriever`

Builds a richer retrieval index with dense passage embeddings, local BM25, extracted
concepts, sentence/concept embeddings, and graph-based propagation.

Constructor arguments are `config`, `corpus`, `queries`, and optional `reranker`.

Common `config` keys:

| Key | Why pass it |
| --- | --- |
| `embedding_model_name` | SentenceTransformer model for passage, concept, sentence, and query embeddings. |
| `device` | Torch device for embedding model. |
| `max_seq_length` | Optional max sequence length for the embedding model. |
| `dataset_name` | Used in cache naming. |
| `cache_path` | Root directory for saved LinearRAG caches. |
| `spacy_model_name` | spaCy model for concept extraction. |
| `use_gpu_for_spacy` | Allows spaCy GPU usage when CUDA is available. |
| `embed_batch_size` | Batch size for embedding stores. |
| `ner_batch_size` | Batch size for spaCy concept extraction. |
| `retrieval_top_k` | Default final result count. |
| `dense_candidate_k` | Candidate count from dense retrieval. |
| `bm25_candidate_k` | Candidate count from local BM25. |
| `graph_candidate_k` | Candidate count from graph retrieval. |
| `dense_rrf_weight` | Weight for dense ranking in final RRF fusion. |
| `bm25_rrf_weight` | Weight for BM25 ranking in final RRF fusion. |
| `graph_rrf_weight` | Weight for graph ranking in final RRF fusion. |
| `rrf_k` | RRF smoothing constant. |
| `local_graph_dense_seed_k` | Dense seed documents used to build/query the local graph. |
| `local_graph_bm25_seed_k` | BM25 seed documents used in the local graph. |
| `local_graph_sentence_seed_k` | Sentence seed count for graph expansion. |
| `local_graph_concept_seed_k` | Concept seed count for graph expansion. |
| `min_concept_len` | Filters very short extracted concepts. |
| `max_concept_words` | Filters overly long concepts. |
| `min_concept_df` | Drops concepts that appear in too few documents. |
| `max_concept_df_ratio` | Drops concepts that appear in too many documents. |
| `stop_concept` | Domain stoplist for concepts. |
| `ppr_damping` | Damping factor for personalized PageRank-style graph scoring. |
| `ppr_max_result_docs` | Caps graph result document count. |
| `export_query_graph` | Writes a query graph image for report inclusion. |
| `query_graph_max_nodes` | Max nodes in exported graph image. |
| `query_graph_output_dir` | Directory for graph images. |
| `query_graph_max_passages` | Max passage nodes in graph export. |
| `query_graph_max_concepts` | Max concept nodes in graph export. |

Call `index_corpus()` before `search(top_k=...)`.

### `Reranker`

Reranks a retriever's top results with a transformer sequence-classification model.
Pass the same `Reranker` instance into any retriever via `reranker=...`.

| Argument | Why pass it |
| --- | --- |
| `model_id` | Hugging Face reranker model id. |
| `top_k` | Number of retrieved docs reranked per query. |
| `batch_size` | Query-document pair batch size. |
| `max_length` | Tokenizer truncation length. |
| `device` | Torch device. |
| `config` | Optional metadata/config holder. |

### `StaticRetrieverReport`

Generates a PDF report from one or more retriever objects.

`generate_report()` arguments:

| Argument | Why pass it |
| --- | --- |
| `title` | Report title. |
| `ds_description` | Optional generated dataset description. |
| `retrievers` | One retriever or a list of retrievers with `results`, `metrics`, and retrieval metrics. |
| `qrels` | Optional relevance judgments for rank distribution analysis. |
| `output_path` | PDF file path or directory. |
| `config` | Report display and comparison options. |
| `n_samples` | Number of sample queries shown; defaults from `config`. |
| `language` | `english` or `dutch` report text. |

Report `config` keys:

| Key | Why pass it |
| --- | --- |
| `sample_queries` | Number of sample queries to include. |
| `sample_results_per_query` | Number of top documents shown per sample query. |
| `sample_text_chars` | Text characters shown per sampled document. |
| `metric_rows` | Max runtime metric rows per retriever. |
| `retrieval_metric_rows` | Max retrieval metric rows. |
| `config_rows` | Max configuration rows per retriever. |
| `comparison_metric` | Metrics used in main comparison charts, for example `["NDCG@10", "Recall@10"]`. |
| `rerank_comparison_metric` | Metrics used for rerank comparison helpers. |
| `query_graph_doc_rows` | Number of retrieved docs shown near an exported query graph. |
| `query_graph_doc_chars` | Text characters shown per graph-adjacent doc row. |
| `scalability_target_docs` | Document count used for rough scaling estimates. |
| `scalability_target_queries` | Query count used for rough scaling estimates. |

After BEIR evaluation, call `retriever.add_retrieval_result(eval_results)` for base
results and `retriever.add_rerank_retrieval_result(eval_results)` for reranked results so
the report can read the metrics.

## Repository Layout

- `raccoon/`: library code.
- `examples/`: runnable workflows and integration examples.
- `test/`: pytest tests.
- `prompts/`: prompt templates for synthesis and validation.
- `data/`: local raw and processed data used by the examples.
- `reports/`: generated report outputs.

## Examples

The scripts in `examples/` are useful workflow examples, but most are integration scripts
with local assumptions. Make sure you adjust them accordingly.

- `examples/document.py`: chunks documents from `data/raw/rechtspraak`.
- `examples/sql.py`: loads documents from a SQL table and chunks them.
- `examples/synthesizer.py`: generates a synthetic BEIR dataset with Ollama.
- `examples/retrieverbm25.py`: evaluates BM25 through an Elasticsearch testcontainer.
- `examples/retrieverdense.py`: evaluates dense retrieval and writes embeddings.
- `examples/retrieverlinear.py`: evaluates LinearRAG.
- `examples/report.py`: runs BM25 and generates a report.
- `examples/test-report.py`: generates a report from fake retriever objects.
- `examples/query_scenario_benchmark.py`: orchestrates synthesis, retriever runs, metrics,
  and report generation for prompt scenario benchmarking.
- `examples/end-to-end.py`: runs the complete benchmark workflow from raw documents to
  synthetic data, retriever evaluation, stored metrics, and a PDF report.

Run a script from the repo root, for example:

```bash
python examples/document.py
```

For scripts that use external systems, confirm the required data, Docker daemon, Ollama
models, Hugging Face access, or database credentials first.

## Full Benchmark Workflow

`examples/end-to-end.py` shows the complete workflow used for a retrieval benchmark:

1. Load raw documents from `data/raw/rechtspraak`.
2. Chunk the documents into parent and child chunks under `data/processed/chunks_600`.
3. Use `OllamaSynthesizer` and the Rechtspraak prompt templates to generate a synthetic
   BEIR-style dataset under `data/processed/rechtspraken/beir_600_semantic`.
4. Load the generated `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`.
5. Generate a short Dutch dataset description with Ollama for the final report.
6. Initialize a transformer reranker with `BAAI/bge-reranker-v2-m3`.
7. Run BM25 with an Elasticsearch testcontainer.
8. Run dense retrieval with `snowflake/snowflake-arctic-embed-l-v2.0`; embeddings are
   cached under `data/processed/rechtspraken/beir_600_semantic/encode/`.
9. Run hybrid retrieval by fusing BM25 and dense results with reciprocal rank fusion.
10. Run LinearRAG with dense, BM25, graph, and concept-based signals; its cache is stored
    under `data/processed/rechtspraken/beir_600_semantic/linear_rag_cache`.
11. Evaluate each retriever with BEIR `EvaluateRetrieval` for the configured `k` values.
12. Evaluate the reranked results separately for each retriever.
13. Append every retriever's retrieval metrics, runtime metrics, and rerank timings to
    `data/processed/rechtspraken/beir_600_semantic/eval_results.json`.
14. Add each retriever object to the `RETRIEVERS` list.
15. Generate a Dutch PDF report at `data/processed/report_graph.pdf` with the dataset
    description, qrels, comparison metrics, runtime trade-offs, rerank sections, samples,
    and exported query graph pages when available.

Run the full workflow with:

```bash
python examples/end-to-end.py
```
Again this workflow has local assumptions. It requires local raw data, Ollama with the
configured models, Docker for Elasticsearch, Hugging Face model downloads, and spaCy model.

## Dataset Inspector CLI

`cli/cli.py` inspects generated synthetic datasets that include `qrels_debug.jsonl`:

```bash
python cli/cli.py data/processed/rechtspraken/beir_realistic_TEST --summary
python cli/cli.py data/processed/rechtspraken/beir_realistic_TEST --query q1
python cli/cli.py data/processed/rechtspraken
```

The interactive mode lets you list queries, inspect qrels/debug judgments, and copy a
query bundle when a clipboard backend is available.


## Library Readiness Notes

The package is usable for local experiments, but keep these points in mind when using it
as a library:

- Prefer imports from subpackages, for example `raccoon.dataloader` and
  `raccoon.custom_retriever`. The top-level `raccoon` package intentionally stays light.
- Keep `pyproject.toml` as the source of truth for dependencies. `requirements.txt`
  delegates to an editable install of the project.
- Keep runnable workflows in `examples/` and true unit tests in `test/test_*.py`.
- BM25, synthesis, dense retrieval, reranking, and LinearRAG each have heavyweight
  runtime requirements. Document those requirements in any downstream project that uses
  this package.
