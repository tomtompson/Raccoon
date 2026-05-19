from pathlib import Path
from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.BM25Retriever import BM25Retriever

from testcontainers.elasticsearch import ElasticSearchContainer
from beir.retrieval.evaluation import EvaluateRetrieval

INPUT_PATH = Path("data/processed/rechtspraken/beir_realistic_TEST")

INPUT_PATH = Path("data/processed/critique_filter.jsonl")
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
TOP_K = 20

def main() -> None:
    corpus, queries, qrels, name = load_local_beir_dataset("data/processed/rechtspraken/beir_realistic_TEST", "test")


    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        )
        retriever = BM25Retriever(
            elasticsearch_url=elasticsearch_url,
            index_name="raccoon-bm25-test",
            corpus = corpus,
            queries = queries,
            topk=TOP_K,
        )
        retriever.index_corpus()
        retriever.search()
        eval = EvaluateRetrieval()
        eval_results = eval.evaluate(qrels=qrels, results=retriever.results, k_values=[1, 3, 5, 10, 50],)
        print(eval_results)
        print(retriever.metrics)



if __name__ == "__main__":
    main()
