from pathlib import Path

from raccoon.custom_retriever.DenseRetriever import DenseRetrieverSentenceBert
from raccoon.custom_retriever.HybridRetriever import HybridRetriever
from raccoon.custom_retriever.LinearRagRetriever import LinearRagRetriever
from raccoon.report.StaticRetrieverReport import StaticRetrieverReport
from raccoon.dataloader.FixedDocumentLoader import FixedDocumentLoader
from raccoon.synthesizer import OllamaSynthesizer
from pathlib import Path
from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.BM25Retriever import BM25Retriever
from raccoon.custom_retriever.util.utils import append_results
from raccoon.custom_retriever.util.Reranker import Reranker 

from testcontainers.elasticsearch import ElasticSearchContainer
from beir.retrieval.evaluation import EvaluateRetrieval
import torch


SOURCE_PATH_RAW = Path("data/raw/rechtspraak")
OUTPUT_PATH_CHUNKS = Path("data/processed/chunks_recht")


QUERY_PROMPT_PATH = "prompts/query_scenarios/query_generation_ambiguous.txt"
INPUT_PATH = Path("data/processed/rechtspraken/beir_600")

def main() -> None:

    #======================================================
    # Generate description of dataset
    #======================================================

    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")


    synth = OllamaSynthesizer(
    host="http://localhost:11434",
    prompt_paths={
        "query_generation": QUERY_PROMPT_PATH,
        "query_validation": "prompts/example_rechtspraak/query_validation.txt",
        "candidate_judging": "prompts/example_rechtspraak/candidate_judging.txt",
        "distribution_validation": "prompts/example_rechtspraak/distribution_validation.txt",
    },)

    description = synth.generate_description_of_ds(
        model="qwen3:8b",
        language="dutch",
        corpus=corpus,
        queries=queries,
        description_length=250,
    )
    print(description)
    with open("data/processed/rechtspraken/beir_600/description.txt", "w+") as f:
        f.write(description)

if __name__ == "__main__":
    main()
