import json
import sys
from pathlib import Path

from raccoon.report.StaticSingleRetrieverReport import StaticSingleRetrieverReport
from raccoon.retriever.BM25Retriever import BM25Retriever
from raccoon.eval.retriever.RetrievelEval import RetrievelEval


def main() -> None:
    retriever = BM25Retriever()
    retriever.load("data/processed/")

    eva = RetrievelEval()
    eva.load("data/processed/")
    
    
    report = StaticSingleRetrieverReport(title="BM25 Eval Report", retriever=retriever, retriever_eval=eva)
    report.generate_report()

if __name__ == "__main__":
    main()
