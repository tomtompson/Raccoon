from pathlib import Path

from raccoon.dataloader.utils import load_local_beir_dataset
from raccoon.custom_retriever.SpladeRetriever import SpladeRetriever
from raccoon.custom_retriever.util.utils import append_results

from testcontainers.elasticsearch import ElasticSearchContainer
from beir.retrieval.evaluation import EvaluateRetrieval


INPUT_PATH = Path("data/processed/rechtspraken/rechtspraken_beir_ambiguous")
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"

TOP_K = 20
MODEL_NAME = "sparse-encoder/splade-robbert-dutch-base-v1"


def main() -> None:
    corpus, queries, qrels, name = load_local_beir_dataset(INPUT_PATH, "test")

    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:"
            f"{container.get_exposed_port(container.port)}"
        )

        retriever = SpladeRetriever(
            elasticsearch_url=elasticsearch_url,
            language="dutch",
            index_name="raccoon-splade-test",
            model_name=MODEL_NAME,
            corpus=corpus,
            queries=queries,
            topk=TOP_K,
            batch_size=8,
            max_length=256,
            max_features=1024,
        )

        retriever.index_corpus()
        retriever.search()

        eval = EvaluateRetrieval()
        eval_results = eval.evaluate(
            qrels=qrels,
            results=retriever.results,
            k_values=[1, 3, 5, 10, 50],
        )

        retriever.add_retrieval_result(eval_results)

        print(retriever.retrieval_metrics)



if __name__ == "__main__":
    main()