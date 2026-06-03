from pathlib import Path
import torch
from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.DenseRetriever import DenseRetrieverSentenceBert
from raccoon.custom_retriever.reranker.CrossEncoderReranker import CrossEncoderReranker

from beir.retrieval.evaluation import EvaluateRetrieval

INPUT_PATH = Path("data/processed/rechtspraken/rechtspraken_beir_semantic")


TOP_K = 20
MODEL_ID = "snowflake/snowflake-arctic-embed-l-v2.0"
MAX_LENGTH = 252
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QUERY_PROMPT_NAME = "query"
PASSAGE_PROMPT_NAME = "document"
ENCODE_PATH = Path("data/processed/rechtspraken/rechtspraken_beir_semantic/encode/")

def main() -> None:
    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")

    retriever = DenseRetrieverSentenceBert(corpus=corpus, 
                                           queries=queries,
                                           model_id=MODEL_ID,
                                           max_length=MAX_LENGTH,
                                           device=DEVICE,
                                           query_prompt_name=QUERY_PROMPT_NAME,
                                           passage_prompt_name=PASSAGE_PROMPT_NAME,
                                           reranker=CrossEncoderReranker(
                                                        model_id="BAAI/bge-reranker-v2-m3",
                                                        top_k=TOP_K,
                                                        batch_size=16,
                                                        max_length=MAX_LENGTH,
                                                        device=DEVICE,
                                                    ))
    retriever.search(top_k=TOP_K, 
                     encode_output_path=ENCODE_PATH,
                     )
    eval = EvaluateRetrieval()
    eval_results = eval.evaluate(qrels=qrels, results=retriever.results, k_values=[1, 3, 5, 10, 20],)
    print(F"BASE: {eval_results}")

    eval_results_rerank = eval.evaluate(qrels=qrels, results=retriever.rerank_results, k_values=[1, 3, 5, 10, 20],)
    print(F"RERANK: {eval_results_rerank}")



if __name__ == "__main__":
    main()
