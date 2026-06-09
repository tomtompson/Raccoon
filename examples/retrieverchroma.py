from __future__ import annotations

from pathlib import Path
from time import monotonic, sleep
from urllib.error import URLError
from urllib.request import urlopen

from beir.retrieval.evaluation import EvaluateRetrieval
from testcontainers.core.container import DockerContainer

from raccoon.custom_retriever.DenseRetrieverChroma import DenseRetrieverChroma
from raccoon.dataloader.utils import load_local_beir_dataset


INPUT_PATH = Path("data/processed/rechtspraken/rechtspraken_beir_semantic")
CHROMA_IMAGE = "chromadb/chroma:1.5.9"
CHROMA_PORT = 8000
COLLECTION_NAME = "raccoon-dense-test"
MODEL_ID = "snowflake/snowflake-arctic-embed-l-v2.0"
TOP_K = 20


def wait_for_chroma(host: str, port: int, timeout: float = 60.0) -> None:
    heartbeat_url = f"http://{host}:{port}/api/v2/heartbeat"
    deadline = monotonic() + timeout
    last_error: Exception | None = None

    while monotonic() < deadline:
        try:
            with urlopen(heartbeat_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except (OSError, URLError) as exc:
            last_error = exc
        sleep(0.5)

    raise RuntimeError(f"Chroma did not become ready at {heartbeat_url}: {last_error}")


def main() -> None:
    corpus, queries, qrels, _ = load_local_beir_dataset(INPUT_PATH, "test")

    with DockerContainer(CHROMA_IMAGE).with_exposed_ports(CHROMA_PORT) as container:
        chroma_host = container.get_container_host_ip()
        chroma_port = int(container.get_exposed_port(CHROMA_PORT))
        wait_for_chroma(chroma_host, chroma_port)

        retriever = DenseRetrieverChroma(
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=COLLECTION_NAME,
            corpus=corpus,
            queries=queries,
            model_id=MODEL_ID,
            normalize_embeddings=True,
            topk=TOP_K,
        )

        retriever.index_corpus()
        retriever.search()

        evaluation = EvaluateRetrieval()
        evaluation_results = evaluation.evaluate(
            qrels=qrels,
            results=retriever.results,
            k_values=[1, 3, 5, 10, 50],
        )
        retriever.add_retrieval_result(evaluation_results)

        print(retriever.retrieval_metrics)
        print(retriever.metrics)


if __name__ == "__main__":
    main()
