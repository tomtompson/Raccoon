import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testcontainers.elasticsearch import ElasticSearchContainer

from raccoon.analysis.critique_report import load_critique_rows
from raccoon.retriever import BM25Retriever


INPUT_PATH = Path("data/processed/critique_filter.jsonl")
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
TOP_K = 20


def main() -> None:
    rows = load_critique_rows(INPUT_PATH)

    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        )
        retriever = BM25Retriever(
            elasticsearch_url=elasticsearch_url,
            index_name="raccoon-bm25-test",
        )
        retriever.process_documents(rows)
        retriever.bulk_search(top_k=TOP_K)
        retriever.save("data/processed")


if __name__ == "__main__":
    main()
