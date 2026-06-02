import json
import os
from pathlib import Path

from raccoon.synthesizer import OllamaSynthesizer
from raccoon.dataloader import FixedDocumentLoader


INPUT_PATH = Path("data/processed/chunks_recht")
RERANKER_ID = "BAAI/bge-reranker-v2-m3"



def main() -> None:
    synth = OllamaSynthesizer(
    host="http://localhost:11434",
    prompt_paths={
        "query_generation": "prompts/example_rechtspraak/query_generation.txt",
        "query_validation": "prompts/example_rechtspraak/query_validation.txt",
        "candidate_judging": "prompts/example_rechtspraak/candidate_judging.txt",
        "distribution_validation": "prompts/example_rechtspraak/distribution_validation.txt",
    },)
    parent = synth.load_chunks(INPUT_PATH / "parent_chunks.json")
    child = synth.load_chunks(INPUT_PATH / "child_chunks.json")

    synth.synthesize_beir(
        parent_chunks=parent,
        child_chunks=child,
        output_dir="data/processed/rechtspraken/beir_realistic_TEST",
        embedding_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        embedding_kwargs= {},
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
        max_estimated_tokens= 5000,
        overwrite_corpus=False,
        max_parents=None,
        random_seed=42,
        target_queries_per_source=1,
        shuffle_parents=True,
        reranker_id= #RERANKER_ID,
        None,
        rerank_pool_size= 70,
        rerank_keep_top_k=40,
    )


if __name__ == "__main__":
    main()
