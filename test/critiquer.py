import os
from pathlib import Path

from raccoon.dataloader.utils import (
    load_langchain_documents,
    load_synthesized_results,
    save_synthesized_results,
)
from raccoon.synthesizer import OllamaSynthesizer
from raccoon.synthesizer.critiquer.OllamaCritiquer import OllamaCritiquer


INPUT_PATH = Path("data/processed/documents.jsonl")
SYNTHESIZED_PATH = Path("data/processed/generated_qa.jsonl")
OUTPUT_PATH = Path("data/processed/critique.jsonl")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "gemma3:27b")


def main() -> None:
    if SYNTHESIZED_PATH.exists():
        synthesized_rows = load_synthesized_results(SYNTHESIZED_PATH)
        synthesizer = OllamaSynthesizer(
            ollama_url=OLLAMA_URL,
            model_id=MODEL_ID,
            documents=synthesized_rows,
        )
    else:
        documents = load_langchain_documents(INPUT_PATH)
        synthesizer = OllamaSynthesizer(
            ollama_url=OLLAMA_URL,
            model_id=MODEL_ID,
            documents=documents,
        )
        save_synthesized_results(synthesizer.synthesize(), SYNTHESIZED_PATH)

    critiquer = OllamaCritiquer(
        OLLAMA_URL,
        MODEL_ID,
        synthesizer=synthesizer,
    )
    results = critiquer.critique()

    save_synthesized_results(results, OUTPUT_PATH)

    print(f"Saved {len(results)} critiqued to {OUTPUT_PATH}")
if __name__ == "__main__":
    main()
