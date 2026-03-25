import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from testcontainers.elasticsearch import ElasticSearchContainer

from raccoon.analysis.critique_report import load_critique_rows
from raccoon.generator import OllamaGenerator
from raccoon.retriever import BM25Retriever


INPUT_PATH = Path("data/processed/critique_filter.jsonl")
OUTPUT_PATH = Path("data/processed/generated_answers.json")
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "gemma3:27b")
TOP_K = 1


def main() -> None:
    rows = load_critique_rows(INPUT_PATH)

    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        )
        retriever = BM25Retriever(
            elasticsearch_url=elasticsearch_url,
            index_name="raccoon-generator-test",
        )
        retriever.process_documents(rows)

        generator = OllamaGenerator(
            retriever=retriever,
            ollama_url=OLLAMA_URL,
            model_id=MODEL_ID,
            top_k=TOP_K,
        )
        results = generator.generate()

    payload = {
        "results": results,
        "metrics": generator.metrics,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(results)} generated answers to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
