import json
import os
from pathlib import Path

from crimsonvector.synthesizer.critiquer.OllamaCritiquer import OllamaCritiquer


OUTPUT_PATH = Path("data/processed/critique_filter.jsonl")
INPUT_PATH = Path("data/processed/critique.jsonl")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_ID = os.getenv("OLLAMA_MODEL", "gemma3:27b")


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8") as handle:
        results = [json.loads(line) for line in handle if line.strip()]

    critiquer = OllamaCritiquer(OLLAMA_URL, MODEL_ID)
    print(f"Previous count {len(results)}")
    results = critiquer.filter(min_relevance=2,results=results)
    print(f"New count {len(results)}")
    

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False))
            handle.write("\n")

    print(f"Saved {len(results)} critiqued to {OUTPUT_PATH}")
if __name__ == "__main__":
    main()
