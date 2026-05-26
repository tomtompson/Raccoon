from pathlib import Path

from raccoon.report.StaticRetrieverReport import StaticRetrieverReport


class FakeReranker:
    def __init__(self):
        self.model_id = "BAAI/bge-reranker-v2-m3"
        self.top_k = 10
        self.batch_size = 16
        self.device = "cuda"


class FakeRetriever:
    def __init__(
        self,
        name,
        retriever_type,
        ndcg,
        recall,
        rerank_ndcg,
        rerank_recall,
        index_time,
        query_time,
        rerank_time,
    ):
        self.__class__.__name__ = name

        self.retriever_type = retriever_type

        self.config = {
            "model": "snowflake/snowflake-arctic-embed-l-v2.0",
            "top_k": 10,
        }

        self.metrics = {
            "index_time": {
                "time_in_seconds": index_time
            },
            "query_time": {
                "time_in_seconds": query_time
            }
        }

        self.retrieval_metrics = {
            retriever_type: {
                "NDCG@1": ndcg - 0.08,
                "NDCG@3": ndcg - 0.04,
                "NDCG@5": ndcg - 0.02,
                "NDCG@10": ndcg,
                "NDCG@20": ndcg + 0.02,

                "Recall@1": recall - 0.25,
                "Recall@3": recall - 0.15,
                "Recall@5": recall - 0.08,
                "Recall@10": recall,
                "Recall@20": recall + 0.08,

                "MAP@1": ndcg - 0.12,
                "MAP@3": ndcg - 0.10,
                "MAP@5": ndcg - 0.09,
                "MAP@10": ndcg - 0.08,
                "MAP@20": ndcg - 0.07,

                "Precision@1": recall - 0.15,
                "Precision@3": recall - 0.20,
                "Precision@5": recall - 0.23,
                "Precision@10": recall - 0.25,
                "Precision@20": recall - 0.30,
            },
            "rerank": {
                "NDCG@1": rerank_ndcg - 0.08,
                "NDCG@3": rerank_ndcg - 0.04,
                "NDCG@5": rerank_ndcg - 0.02,
                "NDCG@10": rerank_ndcg,
                "NDCG@20": rerank_ndcg + 0.02,

                "Recall@1": rerank_recall - 0.25,
                "Recall@3": rerank_recall - 0.15,
                "Recall@5": rerank_recall - 0.08,
                "Recall@10": rerank_recall,
                "Recall@20": rerank_recall + 0.08,

                "MAP@1": rerank_ndcg - 0.10,
                "MAP@3": rerank_ndcg - 0.08,
                "MAP@5": rerank_ndcg - 0.07,
                "MAP@10": rerank_ndcg - 0.06,
                "MAP@20": rerank_ndcg - 0.05,

                "Precision@1": rerank_recall - 0.12,
                "Precision@3": rerank_recall - 0.17,
                "Precision@5": rerank_recall - 0.20,
                "Precision@10": rerank_recall - 0.22,
                "Precision@20": rerank_recall - 0.26,
            }
        }

        self.rerank_metrics = {
            "total_wall_time_sec": rerank_time
        }

        self.reranker = FakeReranker()

        self.corpus = {
            "d1": {
                "title": "AI Regulation",
                "text": "This document discusses European AI regulation and compliance."
            },
            "d2": {
                "title": "RAG Systems",
                "text": "Dense retrieval and reranking improve retrieval quality."
            },
        }

        self.queries = {
            "q1": "How does reranking improve RAG?",
            "q2": "What is dense retrieval?"
        }

        self.results = {
            "q1": [
                ("d1", 0.82),
                ("d2", 0.71),
            ]
        }

        self.rerank_results = {
            "q1": [
                ("d2", 0.95),
                ("d1", 0.80),
            ]
        }


retrievers = [
    FakeRetriever(
        name="DenseRetrieverSentenceBert",
        retriever_type="dense",
        ndcg=0.51,
        recall=0.63,
        rerank_ndcg=0.59,
        rerank_recall=0.72,
        index_time=12.4,
        query_time=1.8,
        rerank_time=3.5,
    ),

    FakeRetriever(
        name="BM25Retriever",
        retriever_type="bm25",
        ndcg=0.34,
        recall=0.49,
        rerank_ndcg=0.41,
        rerank_recall=0.58,
        index_time=4.1,
        query_time=0.7,
        rerank_time=2.2,
    ),

    FakeRetriever(
        name="LinearRAGRetriever",
        retriever_type="linear_rag",
        ndcg=0.57,
        recall=0.70,
        rerank_ndcg=0.64,
        rerank_recall=0.79,
        index_time=25.0,
        query_time=3.9,
        rerank_time=4.8,
    ),
]

report = StaticRetrieverReport()

report.generate_report(
    title="Retriever Benchmark Test",
    retrievers=retrievers,
    output_path=Path("data/test_report.pdf"),
    language="dutch",
    config={
        "comparison_metric": [
            "NDCG@10",
            "Recall@10",
            "MAP@10",
            "Precision@10",
        ],
        "sample_queries": 1,
        "sample_results_per_query": 2,
    }
)

print("Done.")