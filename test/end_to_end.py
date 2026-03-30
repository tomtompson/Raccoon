from raccoon.dataloader import FixedDocumentLoader
from raccoon.synthesizer import OllamaSynthesizer
from raccoon.synthesizer.critiquer import OllamaCritiquer
from raccoon.retriever import BM25Retriever
from raccoon.eval.retriever import RetrievelEval
from raccoon.report import StaticSingleRetrieverReport

from testcontainers.elasticsearch import ElasticSearchContainer
from pathlib import Path


OLLAMA_URL = "http://localhost:11434"
MODEL_ID = "gemma3:27b"
ELASTIC_IMAGE = "docker.elastic.co/elasticsearch/elasticsearch:8.13.4"
TOP_K = 20
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
DOCUMENTS_PATH = PROCESSED_DIR / "documents.jsonl"
SYNTHESIS_PATH = PROCESSED_DIR / "synthesized.jsonl"
CRITIQUE_PATH = PROCESSED_DIR / "critique.json"
RETRIEVER_DIR = PROCESSED_DIR / "retriever"
EVAL_DIR = PROCESSED_DIR / "eval"
REPORT_PATH = PROCESSED_DIR / "retriever_report.pdf"

def main() -> None:
    doc_loader = FixedDocumentLoader(RAW_DATA_DIR)
    doc_loader.load_data()
    doc_loader.save_documents(DOCUMENTS_PATH)

    synthesizer = OllamaSynthesizer(OLLAMA_URL, MODEL_ID, None, doc_loader)
    synthesizer.synthesize()
    synthesizer.save_results(SYNTHESIS_PATH)

    critique = OllamaCritiquer(OLLAMA_URL, MODEL_ID, synthesizer=synthesizer)
    critique.critique()
    critique.filter()
    critique.save(CRITIQUE_PATH)

    with ElasticSearchContainer(ELASTIC_IMAGE) as container:
        elasticsearch_url = (
            f"http://{container.get_container_host_ip()}:{container.get_exposed_port(container.port)}"
        )
        retriever = BM25Retriever(
            elasticsearch_url=elasticsearch_url,
            index_name="raccoon-bm25-test",
        )
        retriever.process_documents(critique.results)
        retriever.bulk_search(top_k=TOP_K)
        retriever.save(RETRIEVER_DIR)
    
    evaluator = RetrievelEval(retriever=retriever)
    evaluator.evaluate()
    evaluator.save(EVAL_DIR)

    report = StaticSingleRetrieverReport(
        "BM25",
        retriever=retriever,
        retriever_eval=evaluator,
    )
    
    report.generate_report(REPORT_PATH)


if __name__ == "__main__":
    main()





