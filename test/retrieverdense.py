from pathlib import Path
from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.DenseRetriever import DenseRetrieverSentenceBert

from beir.retrieval.evaluation import EvaluateRetrieval

INPUT_PATH = Path("data/processed/rechtspraken/beir_realistic_TEST")

INPUT_PATH = Path("data/processed/critique_filter.jsonl")
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
TOP_K = 20
MODEL_ID = "snowflake/snowflake-arctic-embed-l-v2.0"
MAX_LENGHT = 206
DEVICE = "cpu"
QUERY_PROMPT_NAME = "query"
PASSAGE_PROMPT_NAME = "document"


def main() -> None:
    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")

    retriever = DenseRetrieverSentenceBert(corpus=corpus, 
                                           queries=queries,
                                           model_id=MODEL_ID,
                                           max_length=MAX_LENGHT,
                                           device=DEVICE,
                                           query_prompt_name=QUERY_PROMPT_NAME,
                                           passage_prompt_name=PASSAGE_PROMPT_NAME,)
    retriever.search(top_k=TOP_K, 
                     encode_output_path="data/",
                     )
    eval = EvaluateRetrieval()
    eval_results = eval.evaluate(qrels=qrels, results=retriever.results, k_values=[1, 3, 5, 10, 50],)
    print(eval_results)



if __name__ == "__main__":
    main()
