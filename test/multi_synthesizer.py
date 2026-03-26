import json
import os
from pathlib import Path

from raccoon.dataloader.utils import load_langchain_documents
from raccoon.synthesizer.multi_query import MultiQueryOllamaSynthesizer


INPUT_PATH = Path("data/processed/documents.jsonl")
OUTPUT_PATH = Path("data/processed/generated_qa_multi.jsonl")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "gemma3:27b")


def main() -> None:
    documents = load_langchain_documents(INPUT_PATH)
    synthesizer = MultiQueryOllamaSynthesizer(
        ollama_url=OLLAMA_URL,
        model_id=MODEL_ID,
        documents=documents,
    )
    results = synthesizer.synthesize()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, indent=2))
            handle.write("\n")

    print(f"Saved {len(results)} generated QA pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
