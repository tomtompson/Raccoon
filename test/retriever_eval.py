import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from crimsonvector.eval.retriever.RetrievelEval import RetrievelEval


INPUT_PATH = Path("data/processed/bm25_results.json")
OUTPUT_PATH = Path("data/processed/bm25_eval.json")
TOP_K = 20


def main() -> None:
    with open(INPUT_PATH) as json_file:
        rows = json.load(json_file)
    
    eva = RetrievelEval()

    results = eva.evaluate(raw_results=rows)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
