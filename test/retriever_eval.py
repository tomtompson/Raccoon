import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from raccoon.eval.retriever.RetrievelEval import RetrievelEval
from raccoon.retriever.BM25Retriever import BM25Retriever



def main() -> None:

    retriever = BM25Retriever()
    retriever.load("data/processed")

    eva = RetrievelEval(retriever=retriever)
    

    eva.evaluate()

    eva.save("data/processed")

if __name__ == "__main__":
    main()
